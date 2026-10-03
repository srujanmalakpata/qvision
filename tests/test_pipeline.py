"""End-to-end: CLI train -> evaluate -> quantize -> export-onnx on a fake IDX dataset."""

import json
import os

import numpy as np
import pytest
import torch
import yaml

import qvision.data
import qvision.onnx_export
from qvision.cli import main
from qvision.config import Config, DataConfig
from qvision.model import SmallCNN, save_checkpoint
from tests.conftest import TINY


@pytest.fixture
def offline_dataset(fashion_dir, monkeypatch):
    # The fake files are not the real Fashion-MNIST, so skip checksum-verified download.
    monkeypatch.setattr(qvision.data, "download_fashion_mnist", lambda root, *a, **k: root)
    return fashion_dir


def test_cli_pipeline_end_to_end(tmp_path, offline_dataset, monkeypatch, capsys):
    config = {
        "seed": 1,
        "run_name": "e2e",
        "runs_dir": str(tmp_path / "runs"),
        "threads": 2,
        "data": {"root": str(offline_dataset), "val_size": 20},
        "model": {"channels": [4, 8], "hidden": 16, "dropout": 0.0},
        "train": {"epochs": 2, "batch_size": 32, "patience": 2},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(config))
    run_dir = tmp_path / "runs" / "e2e"
    out_dir = tmp_path / "results"

    main(["train", "--config", str(cfg_path)])
    summary = json.loads((run_dir / "summary.json").read_text())
    assert summary["epochs_run"] == 2 and 0 <= summary["test_accuracy"] <= 1

    main(["evaluate", "--run", str(run_dir), "--out-dir", str(out_dir)])
    evaluation = json.loads((out_dir / "evaluation.json").read_text())
    assert evaluation["n"] == 40

    main(["quantize", "--run", str(run_dir), "--out-dir", str(out_dir), "--repeats", "2"])
    study = json.loads((out_dir / "quantization.json").read_text())
    assert study["settings"]["quantized_engine"] == torch.backends.quantized.engine
    assert study["settings"]["quantized_engine"] in {"x86", "fbgemm", "qnnpack"}
    assert set(study["variants"]) == {
        "fp32",
        "int8_weight_only",
        "int4_weight_only",
        "int8_torch_dynamic",
    }
    fp32 = study["variants"]["fp32"]
    assert fp32["accuracy_delta_pp"] == 0 and fp32["top1_agreement_with_fp32"] == 1.0
    sizes = {k: v["size_bytes"] for k, v in study["variants"].items()}
    assert sizes["int4_weight_only"] < sizes["int8_weight_only"] < sizes["fp32"]

    main(["export-onnx", "--run", str(run_dir), "--out-dir", str(out_dir), "--repeats", "2"])
    onnx_report = json.loads((out_dir / "onnx.json").read_text())
    assert onnx_report["parity_on_test_set"]["allclose"]
    assert (run_dir / "model.onnx").exists()
    capsys.readouterr()


def test_export_onnx_exits_non_zero_when_parity_fails(tmp_path, offline_dataset, monkeypatch):
    """A broken runtime must fail the command, not just write `allclose: false` to a file."""
    run_dir = tmp_path / "runs" / "broken"
    cfg = Config(run_name="broken", data=DataConfig(root=str(offline_dataset), val_size=20))
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
    torch.manual_seed(0)
    save_checkpoint(SmallCNN(TINY), run_dir / "model.pt")

    class WrongSession:
        def run(self, output_names, feed):
            (x,) = feed.values()
            return [np.zeros((len(x), 10), dtype=np.float32)]

    monkeypatch.setattr(qvision.onnx_export, "ort_session", lambda *a, **k: WrongSession())
    out_dir = tmp_path / "results"
    with pytest.raises(SystemExit, match="differ"):
        main(["export-onnx", "--run", str(run_dir), "--out-dir", str(out_dir), "--repeats", "1"])
    report = json.loads((out_dir / "onnx.json").read_text())
    assert report["parity_on_test_set"]["allclose"] is False


@pytest.mark.parametrize(
    "argv, env, expected",
    [
        (["serve", "--run", "runs/a"], "runs/b/model.pt", "runs/a/model.pt"),
        (["serve"], "runs/b/model.pt", "runs/b/model.pt"),
        (["serve"], None, "runs/fmnist-cnn/model.pt"),
    ],
)
def test_serve_run_flag_overrides_environment(monkeypatch, argv, env, expected):
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    if env is None:
        monkeypatch.delenv("QV_CHECKPOINT", raising=False)
    else:
        monkeypatch.setenv("QV_CHECKPOINT", env)
    main(argv)
    assert os.environ["QV_CHECKPOINT"] == expected
