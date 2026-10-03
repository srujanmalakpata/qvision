"""Fashion-MNIST download, IDX parsing, normalisation and deterministic splits.

The dataset is fetched from the official Zalando Research bucket and verified against the MD5
checksums published in the Fashion-MNIST README, then cached under ``data/`` (git-ignored).
No torchvision dependency: the IDX format is simple enough to parse with NumPy.
"""

from __future__ import annotations

import gzip
import hashlib
import shutil
import struct
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

BASE_URL = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"

# filename -> md5 (from https://github.com/zalandoresearch/fashion-mnist#get-the-data)
FILES: dict[str, str] = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}

CLASS_NAMES: tuple[str, ...] = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)

# Per-pixel mean/std of the 60k training images after scaling to [0, 1].
# `compute_mean_std` reproduces them; they are fixed so serving does not need the dataset.
MEAN = 0.2860
STD = 0.3530

_IDX_DTYPE_UBYTE = 0x08


def md5sum(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_fashion_mnist(root: str | Path, base_url: str = BASE_URL) -> Path:
    """Download any missing/corrupt files into ``root`` and return it. Idempotent."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        target = root / name
        if target.exists() and md5sum(target) == expected:
            continue
        tmp = target.with_suffix(".part")
        with urllib.request.urlopen(base_url + name, timeout=60) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        actual = md5sum(tmp)
        if actual != expected:
            tmp.unlink()
            raise OSError(f"checksum mismatch for {name}: expected {expected}, got {actual}")
        tmp.replace(target)
    return root


def read_idx(path: str | Path) -> np.ndarray:
    """Parse an (optionally gzipped) IDX file of unsigned bytes into a NumPy array.

    Layout: 2 zero bytes, a dtype code (0x08 = ubyte), the number of dimensions, then one
    big-endian uint32 per dimension, then the raw data.
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:
        raw = fh.read()
    if len(raw) < 4 or raw[0] != 0 or raw[1] != 0:
        raise ValueError(f"{path.name}: not an IDX file (bad magic)")
    dtype_code, ndim = raw[2], raw[3]
    if dtype_code != _IDX_DTYPE_UBYTE:
        raise ValueError(f"{path.name}: unsupported IDX dtype 0x{dtype_code:02x}")
    header_end = 4 + 4 * ndim
    shape = struct.unpack(f">{ndim}I", raw[4:header_end])
    data = np.frombuffer(raw, dtype=np.uint8, offset=header_end)
    if data.size != int(np.prod(shape)):
        raise ValueError(f"{path.name}: expected {np.prod(shape)} values, found {data.size}")
    return data.reshape(shape).copy()  # frombuffer is read-only; own the memory


def load_split(root: str | Path, train: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(images uint8 [N, 28, 28], labels int64 [N])`` for the train or test split."""
    prefix = "train" if train else "t10k"
    root = Path(root)
    images = read_idx(root / f"{prefix}-images-idx3-ubyte.gz")
    labels = read_idx(root / f"{prefix}-labels-idx1-ubyte.gz").astype(np.int64)
    if images.shape[0] != labels.shape[0]:
        raise ValueError("image/label count mismatch")
    return images, labels


def normalize(images: np.ndarray) -> torch.Tensor:
    """uint8 ``[N, 28, 28]`` -> float32 ``[N, 1, 28, 28]`` standardised with MEAN/STD."""
    if images.dtype != np.uint8:
        raise TypeError(f"expected uint8 images, got {images.dtype}")
    x = torch.from_numpy(np.ascontiguousarray(images)).float().div_(255.0)
    return x.sub_(MEAN).div_(STD).unsqueeze(1)


def compute_mean_std(images: np.ndarray) -> tuple[float, float]:
    x = images.astype(np.float64) / 255.0
    return float(x.mean()), float(x.std())


@dataclass(frozen=True)
class Splits:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset


def split_indices(n: int, val_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically shuffle ``range(n)`` and carve off a validation set."""
    if not 0 < val_size < n:
        raise ValueError(f"val_size must be in (0, {n}), got {val_size}")
    perm = np.random.default_rng(seed).permutation(n)
    return perm[val_size:], perm[:val_size]


def make_splits(
    train_images: np.ndarray,
    train_labels: np.ndarray,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    val_size: int,
    seed: int,
    train_subset: int | None = None,
) -> Splits:
    train_idx, val_idx = split_indices(len(train_images), val_size, seed)
    if train_subset is not None:
        train_idx = train_idx[:train_subset]

    def to_ds(images: np.ndarray, labels: np.ndarray) -> TensorDataset:
        return TensorDataset(normalize(images), torch.from_numpy(labels))

    return Splits(
        train=to_ds(train_images[train_idx], train_labels[train_idx]),
        val=to_ds(train_images[val_idx], train_labels[val_idx]),
        test=to_ds(test_images, test_labels),
    )


def load_fashion_mnist(
    root: str | Path, val_size: int, seed: int, train_subset: int | None = None
) -> Splits:
    """Download (if needed), parse and split Fashion-MNIST."""
    download_fashion_mnist(root)
    return make_splits(
        *load_split(root, train=True),
        *load_split(root, train=False),
        val_size=val_size,
        seed=seed,
        train_subset=train_subset,
    )


def load_test_set(root: str | Path) -> TensorDataset:
    """Download/verify (if needed) and return only the normalised 10k test split.

    Cheaper than ``load_fashion_mnist(...).test`` for evaluation commands: the 60k training
    images are never parsed or normalised.
    """
    download_fashion_mnist(root)
    images, labels = load_split(root, train=False)
    return TensorDataset(normalize(images), torch.from_numpy(labels))
