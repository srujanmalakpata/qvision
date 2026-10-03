import gzip
import hashlib
import io

import numpy as np
import pytest
import torch

import qvision.data
from qvision.data import (
    MEAN,
    STD,
    compute_mean_std,
    download_fashion_mnist,
    load_split,
    load_test_set,
    make_splits,
    normalize,
    read_idx,
    split_indices,
)
from tests.conftest import synthetic_images, write_idx


def test_read_idx_round_trips_shape_and_values(tmp_path):
    array = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
    for name in ("a.idx", "a.idx.gz"):
        write_idx(tmp_path / name, array)
        np.testing.assert_array_equal(read_idx(tmp_path / name), array)


def test_read_idx_rejects_bad_magic(tmp_path):
    path = tmp_path / "bad.gz"
    with gzip.open(path, "wb") as fh:
        fh.write(b"\x01\x02\x08\x01" + b"\x00" * 8)
    with pytest.raises(ValueError, match="magic"):
        read_idx(path)


def test_read_idx_rejects_truncated_payload(tmp_path):
    path = tmp_path / "short.idx"
    path.write_bytes(bytes([0, 0, 8, 1]) + (10).to_bytes(4, "big") + b"\x00" * 3)
    with pytest.raises(ValueError, match="expected 10"):
        read_idx(path)


def test_load_split_shapes_and_dtypes(fashion_dir):
    images, labels = load_split(fashion_dir, train=True)
    assert images.shape == (120, 28, 28) and images.dtype == np.uint8
    assert labels.shape == (120,) and labels.dtype == np.int64
    assert labels.min() >= 0 and labels.max() <= 9


def test_normalize_shape_range_and_statistics():
    images, _ = synthetic_images(64)
    x = normalize(images)
    assert x.shape == (64, 1, 28, 28) and x.dtype == torch.float32
    # Pixel p maps to (p/255 - MEAN) / STD.
    np.testing.assert_allclose(
        x[0, 0].numpy(), (images[0] / 255.0 - MEAN) / STD, rtol=1e-5, atol=1e-5
    )
    # A pixel at the dataset mean intensity maps to ~0; one STD above it maps to ~1.
    at_mean = np.full((1, 28, 28), round(MEAN * 255), dtype=np.uint8)
    assert abs(float(normalize(at_mean).mean())) < 0.5 / 255 / STD + 1e-6


def test_compute_mean_std_on_a_known_array():
    # Half the pixels black, half white: mean 0.5 and (population) std 0.5 after scaling.
    images = np.zeros((2, 28, 28), dtype=np.uint8)
    images[1] = 255
    assert compute_mean_std(images) == pytest.approx((0.5, 0.5))


def test_normalisation_constants_match_the_recorded_training_statistics():
    # compute_mean_std on the real 60k training images gives (0.28604, 0.35302); the constants
    # are those values rounded to 4 decimals (re-checked in VERIFICATION.md, which needs the
    # downloaded dataset). This guards against an accidental edit of the constants.
    assert pytest.approx(0.28604, abs=5e-5) == MEAN
    assert pytest.approx(0.35302, abs=5e-5) == STD


def test_normalize_rejects_float_input():
    with pytest.raises(TypeError):
        normalize(np.zeros((1, 28, 28), dtype=np.float32))


def test_split_indices_are_deterministic_disjoint_and_complete():
    train_a, val_a = split_indices(100, 20, seed=7)
    train_b, val_b = split_indices(100, 20, seed=7)
    np.testing.assert_array_equal(train_a, train_b)
    np.testing.assert_array_equal(val_a, val_b)
    assert len(val_a) == 20 and len(train_a) == 80
    assert set(train_a).isdisjoint(val_a)
    assert sorted(np.concatenate([train_a, val_a]).tolist()) == list(range(100))
    assert not np.array_equal(val_a, split_indices(100, 20, seed=8)[1])


def test_make_splits_respects_subset(fashion_dir):
    splits = make_splits(
        *load_split(fashion_dir, True), *load_split(fashion_dir, False), 20, 0, train_subset=50
    )
    assert len(splits.train) == 50 and len(splits.val) == 20 and len(splits.test) == 40
    x, y = splits.train[0]
    assert x.shape == (1, 28, 28) and y.dtype == torch.int64


class FakeServer:
    """Stands in for ``urllib.request.urlopen``: serves fixed payloads and counts requests."""

    def __init__(self, payloads: dict[str, bytes]) -> None:
        self.payloads = payloads
        self.requests: list[str] = []

    def __call__(self, url, timeout=None):
        name = url.rsplit("/", 1)[-1]
        self.requests.append(name)
        return io.BytesIO(self.payloads[name])


@pytest.fixture
def fake_mirror(monkeypatch):
    payloads = {name: f"fake payload {name}".encode() for name in qvision.data.FILES}
    checksums = {name: hashlib.md5(raw).hexdigest() for name, raw in payloads.items()}
    monkeypatch.setattr(qvision.data, "FILES", checksums)
    server = FakeServer(payloads)
    monkeypatch.setattr("urllib.request.urlopen", server)
    return server


def test_download_verifies_checksums_and_is_idempotent(tmp_path, fake_mirror):
    root = download_fashion_mnist(tmp_path, base_url="http://mirror.invalid/")
    assert sorted(fake_mirror.requests) == sorted(qvision.data.FILES)  # 4 downloads
    for name, payload in fake_mirror.payloads.items():
        assert (root / name).read_bytes() == payload

    fake_mirror.requests.clear()
    download_fashion_mnist(tmp_path, base_url="http://mirror.invalid/")
    assert fake_mirror.requests == []  # valid cache: nothing re-downloaded

    corrupt = next(iter(qvision.data.FILES))
    (tmp_path / corrupt).write_bytes(b"truncated")
    download_fashion_mnist(tmp_path, base_url="http://mirror.invalid/")
    assert fake_mirror.requests == [corrupt]  # only the corrupt file is fetched again
    assert (tmp_path / corrupt).read_bytes() == fake_mirror.payloads[corrupt]


def test_download_rejects_a_tampered_payload_and_leaves_no_partial_file(tmp_path, fake_mirror):
    tampered = next(iter(qvision.data.FILES))
    fake_mirror.payloads[tampered] = b"something else entirely"
    with pytest.raises(OSError, match="checksum mismatch"):
        download_fashion_mnist(tmp_path, base_url="http://mirror.invalid/")
    assert not (tmp_path / tampered).exists()
    assert not list(tmp_path.glob("*.part"))


def test_load_test_set_returns_only_the_normalised_test_split(fashion_dir, monkeypatch):
    monkeypatch.setattr(qvision.data, "download_fashion_mnist", lambda root, *a, **k: root)
    test = load_test_set(fashion_dir)
    images, labels = load_split(fashion_dir, train=False)
    assert torch.equal(test.tensors[0], normalize(images))
    assert torch.equal(test.tensors[1], torch.from_numpy(labels))
