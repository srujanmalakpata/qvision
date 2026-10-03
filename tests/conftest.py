"""Shared fixtures. Nothing here touches the network or the real dataset."""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from qvision.config import ModelConfig
from qvision.data import normalize
from qvision.model import SmallCNN

TINY = ModelConfig(channels=(4, 8), hidden=16, dropout=0.0)


@pytest.fixture(autouse=True)
def _offline_and_few_threads(monkeypatch):
    """Tests must never hit the network; fail loudly if anything tries to download."""

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted during tests")

    monkeypatch.setattr("urllib.request.urlopen", no_network)
    torch.set_num_threads(2)


def write_idx(path: Path, array: np.ndarray) -> None:
    """Write a uint8 array in IDX format (gzipped if the name ends in .gz)."""
    header = bytes([0, 0, 0x08, array.ndim]) + struct.pack(f">{array.ndim}I", *array.shape)
    payload = header + array.astype(np.uint8).tobytes()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wb") as fh:
        fh.write(payload)


def synthetic_images(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Learnable toy data: class k is a bright horizontal band at rows 2k..2k+5 plus noise."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 10, size=n)
    images = rng.integers(0, 60, size=(n, 28, 28)).astype(np.uint8)
    for i, k in enumerate(labels):
        images[i, 2 * k : 2 * k + 6, 4:24] = 220
    return images, labels.astype(np.int64)


@pytest.fixture
def synthetic_dataset() -> TensorDataset:
    images, labels = synthetic_images(512)
    return TensorDataset(normalize(images), torch.from_numpy(labels))


@pytest.fixture
def fashion_dir(tmp_path: Path) -> Path:
    """A fake Fashion-MNIST directory with tiny train/test splits in real IDX format."""
    train_x, train_y = synthetic_images(120, seed=1)
    test_x, test_y = synthetic_images(40, seed=2)
    write_idx(tmp_path / "train-images-idx3-ubyte.gz", train_x)
    write_idx(tmp_path / "train-labels-idx1-ubyte.gz", train_y.astype(np.uint8))
    write_idx(tmp_path / "t10k-images-idx3-ubyte.gz", test_x)
    write_idx(tmp_path / "t10k-labels-idx1-ubyte.gz", test_y.astype(np.uint8))
    return tmp_path


@pytest.fixture
def tiny_model() -> SmallCNN:
    torch.manual_seed(0)
    return SmallCNN(TINY).eval()
