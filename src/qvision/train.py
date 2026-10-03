"""Deterministic training loop with early stopping on validation loss."""

from __future__ import annotations

import contextlib
import copy
import math
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from qvision.config import Config
from qvision.model import SmallCNN, save_checkpoint
from qvision.tracking import RunTracker


def set_seed(seed: int) -> None:
    """Seed every RNG we use (Python, legacy NumPy global RNG, torch)."""
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - seeds any third-party code using the legacy global RNG
    torch.manual_seed(seed)


@contextlib.contextmanager
def deterministic(seed: int, threads: int | None = None) -> Iterator[None]:
    """Seed RNGs, force deterministic kernels and fix the thread count for the ``with`` block.

    ``use_deterministic_algorithms`` and ``set_num_threads`` are process-wide switches, so the
    previous values are restored on exit; otherwise every later caller in the same process
    (evaluation, a server, other tests) would silently inherit them.
    """
    prev_deterministic = torch.are_deterministic_algorithms_enabled()
    prev_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    prev_threads = torch.get_num_threads()
    set_seed(seed)
    torch.use_deterministic_algorithms(True)
    if threads is not None:
        torch.set_num_threads(threads)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev_deterministic, warn_only=prev_warn_only)
        torch.set_num_threads(prev_threads)


class EarlyStopping:
    """Stop when the monitored loss has not improved by ``min_delta`` for ``patience`` epochs."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        if patience < 1:
            raise ValueError("patience must be >= 1")
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        """Record an epoch's value; return True if it is a new best."""
        if value < self.best - self.min_delta:
            self.best = value
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, float]:
    """One pass over ``loader``; trains if an optimizer is given. Returns (mean loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    loss_sum, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
    if seen == 0:
        raise ValueError("run_epoch got an empty DataLoader")
    return loss_sum / seen, correct / seen


@dataclass
class FitResult:
    model: SmallCNN
    best_epoch: int
    best_val_loss: float
    best_val_acc: float
    epochs_run: int
    stopped_early: bool
    train_seconds: float
    history: list[dict[str, float]] = field(default_factory=list)


def fit(
    cfg: Config,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    tracker: RunTracker | None = None,
    log: Callable[[str], None] = print,
) -> FitResult:
    """Train ``SmallCNN`` per ``cfg``; restore and return the best (lowest val loss) weights."""
    with deterministic(cfg.seed, cfg.threads):
        return _fit(cfg, train_ds, val_ds, tracker, log)


def _fit(
    cfg: Config,
    train_ds: TensorDataset,
    val_ds: TensorDataset,
    tracker: RunTracker | None,
    log: Callable[[str], None],
) -> FitResult:
    tc = cfg.train
    model = SmallCNN(cfg.model)
    generator = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=tc.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=512)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=tc.epochs)
    stopper = EarlyStopping(tc.patience, tc.min_delta)

    if tracker is not None:
        tracker.log_config(cfg.to_dict())

    best_state = copy.deepcopy(model.state_dict())
    best_epoch, best_val_acc = 0, 0.0
    history: list[dict[str, float]] = []
    start = time.perf_counter()
    epoch = 0
    stopped_early = False
    for epoch in range(1, tc.epochs + 1):
        t0 = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = run_epoch(model, train_loader, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader)
        scheduler.step()
        if not (math.isfinite(train_loss) and math.isfinite(val_loss)):
            # A diverged run would otherwise "succeed" by restoring the random initial weights.
            raise FloatingPointError(
                f"non-finite loss at epoch {epoch} (train {train_loss}, val {val_loss}); "
                "lower the learning rate"
            )
        row = {
            "epoch": epoch,
            "lr": round(lr, 8),
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "seconds": round(time.perf_counter() - t0, 2),
        }
        history.append(row)
        if tracker is not None:
            tracker.log_metrics(row)
        if stopper.step(val_loss):
            best_state = copy.deepcopy(model.state_dict())
            best_epoch, best_val_acc = epoch, val_acc
        log(
            f"epoch {epoch:2d}  train_loss {train_loss:.4f}  train_acc {train_acc:.4f}  "
            f"val_loss {val_loss:.4f}  val_acc {val_acc:.4f}  ({row['seconds']}s)"
        )
        if stopper.should_stop and epoch < tc.epochs:
            stopped_early = True
            log(f"early stopping: no val_loss improvement for {tc.patience} epochs")
            break

    model.load_state_dict(best_state)
    model.eval()
    result = FitResult(
        model=model,
        best_epoch=best_epoch,
        best_val_loss=stopper.best,
        best_val_acc=best_val_acc,
        epochs_run=epoch,
        stopped_early=stopped_early,
        train_seconds=round(time.perf_counter() - start, 1),
        history=history,
    )
    if tracker is not None:
        save_checkpoint(model, tracker.run_dir / "model.pt", extra={"best_epoch": best_epoch})
    return result
