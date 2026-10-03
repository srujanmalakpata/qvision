import csv
import json

import pytest
import torch

from qvision.config import Config, DataConfig, TrainConfig, config_from_dict, load_config
from qvision.model import load_checkpoint
from qvision.tracking import RunTracker
from qvision.train import EarlyStopping, fit
from tests.conftest import TINY


def smoke_config(tmp_path, epochs=4, patience=3) -> Config:
    return Config(
        seed=123,
        run_name="smoke",
        runs_dir=str(tmp_path),
        threads=2,
        data=DataConfig(val_size=128),
        model=TINY,
        train=TrainConfig(epochs=epochs, batch_size=64, lr=5e-3, patience=patience),
    )


def split(ds, n_val=128):
    x, y = ds.tensors
    return (
        torch.utils.data.TensorDataset(x[n_val:], y[n_val:]),
        torch.utils.data.TensorDataset(x[:n_val], y[:n_val]),
    )


def test_early_stopping_counts_bad_epochs():
    stopper = EarlyStopping(patience=2, min_delta=0.01)
    assert stopper.step(1.0) is True
    assert stopper.step(0.995) is False  # not better by min_delta
    assert not stopper.should_stop
    assert stopper.step(0.5) is True  # improvement resets the counter
    assert stopper.step(0.6) is False and stopper.step(0.7) is False
    assert stopper.should_stop
    with pytest.raises(ValueError):
        EarlyStopping(patience=0)


def test_training_smoke_learns_and_logs_run_directory(tmp_path, synthetic_dataset):
    """512 synthetic samples, tiny CNN, a few epochs on CPU: must learn and log artifacts."""
    cfg = smoke_config(tmp_path)
    train_ds, val_ds = split(synthetic_dataset)
    result = fit(cfg, train_ds, val_ds, RunTracker(cfg.run_dir), log=lambda _: None)

    first, last = result.history[0], result.history[-1]
    assert last["train_loss"] < first["train_loss"]
    assert result.best_val_acc > 0.5  # chance is 0.1

    run_dir = cfg.run_dir
    with open(run_dir / "metrics.csv") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == result.epochs_run
    assert set(rows[0]) >= {"epoch", "train_loss", "val_loss", "val_acc", "lr"}
    assert load_config(run_dir / "config.yaml") == cfg
    assert json.loads((run_dir / "env.json").read_text())["torch"] == torch.__version__

    restored = load_checkpoint(run_dir / "model.pt")
    x = val_ds.tensors[0][:4]
    with torch.inference_mode():
        torch.testing.assert_close(restored(x), result.model(x))


def test_training_is_deterministic(tmp_path, synthetic_dataset):
    train_ds, val_ds = split(synthetic_dataset)
    a = fit(smoke_config(tmp_path, epochs=2), train_ds, val_ds, log=lambda _: None)
    b = fit(smoke_config(tmp_path, epochs=2), train_ds, val_ds, log=lambda _: None)
    state_b = b.model.state_dict()
    for name, pa in a.model.state_dict().items():
        pb = state_b[name]
        assert torch.equal(pa, pb), name

    def without_timing(history):
        return [{k: v for k, v in row.items() if k != "seconds"} for row in history]

    assert without_timing(a.history) == without_timing(b.history)


@pytest.mark.filterwarnings(r"ignore:Detected call of `lr_scheduler.step\(\)`:UserWarning")
def test_fit_stops_early_and_restores_best_weights(tmp_path, synthetic_dataset, monkeypatch):
    """Script the validation losses so the control flow is tested exactly."""
    import qvision.train as train_mod

    val_losses = iter([1.0, 0.5, 0.7, 0.8, 0.9, 0.95])
    epoch = {"n": 0}

    def scripted_epoch(model, loader, optimizer=None):
        if optimizer is not None:  # training pass: stamp the epoch number into the weights
            epoch["n"] += 1
            with torch.no_grad():
                model.classifier[-1].bias.fill_(epoch["n"])
            return 0.0, 0.0
        return next(val_losses), 0.1 * epoch["n"]

    monkeypatch.setattr(train_mod, "run_epoch", scripted_epoch)
    cfg = smoke_config(tmp_path, epochs=6, patience=2)
    result = fit(cfg, *split(synthetic_dataset), log=lambda _: None)

    assert result.stopped_early and result.epochs_run == 4  # epochs 3 and 4 did not improve
    assert result.best_epoch == 2 and result.best_val_loss == 0.5
    assert result.best_val_acc == pytest.approx(0.2)
    assert torch.all(result.model.classifier[-1].bias == 2)  # weights from epoch 2 restored


def test_fit_restores_process_wide_torch_settings(tmp_path, synthetic_dataset):
    """fit() turns on deterministic kernels and pins threads only for its own duration."""
    torch.use_deterministic_algorithms(False)
    torch.set_num_threads(3)
    try:
        fit(smoke_config(tmp_path, epochs=1), *split(synthetic_dataset), log=lambda _: None)
        assert not torch.are_deterministic_algorithms_enabled()
        assert torch.get_num_threads() == 3
    finally:
        torch.set_num_threads(2)


@pytest.mark.parametrize(
    "raw, message",
    [
        ({"model": {"channels": [8, 16, 32]}}, "model.channels"),
        ({"model": {"channels": [8, 0]}}, "model.channels"),
        ({"model": {"dropout": 1.0}}, "model.dropout"),
        ({"train": {"epochs": 0}}, "train.epochs"),
        ({"train": {"lr": -1e-3}}, "train.lr"),
        ({"train": {"batch_size": 0}}, "train.batch_size"),
        ({"data": {"val_size": -5}}, "data.val_size"),
        ({"data": {"train_subset": 0}}, "data.train_subset"),
        ({"data": {"train_subset": 64}, "train": {"batch_size": 128}}, "train_subset"),
        ({"threads": 0}, "threads"),
    ],
)
def test_config_rejects_invalid_values(raw, message):
    with pytest.raises(ValueError, match=message):
        config_from_dict(raw)


def test_config_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown TrainConfig"):
        config_from_dict({"train": {"epochz": 3}})
    cfg = config_from_dict({"model": {"channels": [8, 16]}})
    assert cfg.model.channels == (8, 16)


def scripted_run_epoch(val_losses):
    """A fake run_epoch: training passes do nothing, validation passes return scripted losses."""
    losses = iter(val_losses)

    def run_epoch(model, loader, optimizer=None):
        if optimizer is not None:
            return 0.0, 0.0
        return next(losses), 0.5

    return run_epoch


@pytest.mark.filterwarnings(r"ignore:Detected call of `lr_scheduler.step\(\)`:UserWarning")
def test_patience_running_out_on_the_final_epoch_is_not_an_early_stop(
    tmp_path, synthetic_dataset, monkeypatch
):
    import qvision.train as train_mod

    monkeypatch.setattr(train_mod, "run_epoch", scripted_run_epoch([1.0, 0.5, 0.6, 0.7]))
    logs: list[str] = []
    cfg = smoke_config(tmp_path, epochs=4, patience=2)
    result = fit(cfg, *split(synthetic_dataset), log=logs.append)

    assert result.epochs_run == 4 and not result.stopped_early
    assert result.best_epoch == 2
    assert not any("early stopping" in line for line in logs)


@pytest.mark.filterwarnings(r"ignore:Detected call of `lr_scheduler.step\(\)`:UserWarning")
def test_non_finite_validation_loss_raises_instead_of_saving_random_weights(
    tmp_path, synthetic_dataset, monkeypatch
):
    import qvision.train as train_mod

    monkeypatch.setattr(train_mod, "run_epoch", scripted_run_epoch([float("nan")] * 3))
    cfg = smoke_config(tmp_path, epochs=3, patience=2)
    tracker = RunTracker(cfg.run_dir)
    with pytest.raises(FloatingPointError, match="non-finite loss at epoch 1"):
        fit(cfg, *split(synthetic_dataset), tracker, log=lambda _: None)
    assert not (cfg.run_dir / "model.pt").exists()
