"""Command-line entry point: ``qvision <command>`` (or ``python -m qvision``)."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from qvision.config import Config, load_config
from qvision.data import load_fashion_mnist, load_split, load_test_set
from qvision.tracking import RunTracker, environment_info, write_json

DEFAULT_RUN = "runs/fmnist-cnn"


def _run_config(run_dir: Path) -> Config:
    return load_config(run_dir / "config.yaml")


def _test_set(cfg: Config):
    return load_test_set(cfg.data.root)


def cmd_download(args: argparse.Namespace) -> None:
    from qvision.data import download_fashion_mnist

    root = download_fashion_mnist(args.root)
    images, labels = load_split(root, train=True)
    print(f"Fashion-MNIST ready in {root}: train images {images.shape}, labels {labels.shape}")


def cmd_train(args: argparse.Namespace) -> None:
    from qvision.evaluate import evaluate_model
    from qvision.model import count_parameters
    from qvision.train import fit

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg = replace(cfg, train=replace(cfg.train, epochs=args.epochs))
    if args.train_subset is not None:
        cfg = replace(cfg, data=replace(cfg.data, train_subset=args.train_subset))
    if args.run_name is not None:
        cfg = replace(cfg, run_name=args.run_name)

    splits = load_fashion_mnist(cfg.data.root, cfg.data.val_size, cfg.seed, cfg.data.train_subset)
    print(f"train {len(splits.train)}  val {len(splits.val)}  test {len(splits.test)}")
    tracker = RunTracker(cfg.run_dir)
    result = fit(cfg, splits.train, splits.val, tracker)
    torch.set_num_threads(cfg.threads)
    # Same bootstrap settings as `qvision evaluate`, so both commands report the same CI.
    test = evaluate_model(result.model, splits.test, seed=cfg.seed)
    summary = {
        "run_dir": str(cfg.run_dir),
        "parameters": count_parameters(result.model),
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "stopped_early": result.stopped_early,
        "best_val_loss": result.best_val_loss,
        "best_val_acc": result.best_val_acc,
        "train_seconds": result.train_seconds,
        "test_accuracy": test["accuracy"],
        "test_accuracy_ci95": test["accuracy_ci95"],
        "test_macro_f1": test["macro_f1"],
        "test_ece": test["ece_15_bins"],
        "environment": environment_info(),
    }
    tracker.log_summary(summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "environment"}, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    from qvision.evaluate import evaluate_model, report_to_markdown
    from qvision.model import load_checkpoint

    run_dir = Path(args.run)
    cfg = _run_config(run_dir)
    torch.set_num_threads(cfg.threads)
    report = evaluate_model(load_checkpoint(run_dir / "model.pt"), _test_set(cfg), seed=cfg.seed)
    write_json(Path(args.out_dir) / "evaluation.json", report)
    md = report_to_markdown(report)
    (Path(args.out_dir) / "evaluation.md").write_text(md + "\n", encoding="utf-8")
    print(md)


def cmd_quantize(args: argparse.Namespace) -> None:
    from qvision.model import load_checkpoint
    from qvision.study import run_study, study_to_markdown

    run_dir = Path(args.run)
    cfg = _run_config(run_dir)
    torch.set_num_threads(cfg.threads)
    study = run_study(
        load_checkpoint(run_dir / "model.pt"), _test_set(cfg), repeats=args.repeats, seed=cfg.seed
    )
    study["settings"]["environment"] = environment_info()
    write_json(Path(args.out_dir) / "quantization.json", study)
    md = study_to_markdown(study)
    (Path(args.out_dir) / "quantization.md").write_text(md + "\n", encoding="utf-8")
    print(md)


def cmd_export_onnx(args: argparse.Namespace) -> None:
    from qvision.model import load_checkpoint
    from qvision.onnx_export import check_parity, compare_latency, export_onnx

    run_dir = Path(args.run)
    cfg = _run_config(run_dir)
    torch.set_num_threads(cfg.threads)
    model = load_checkpoint(run_dir / "model.pt")
    path = export_onnx(model, run_dir / "model.onnx")
    test = _test_set(cfg)
    images = test.tensors[0]
    parity = check_parity(model, path, images)
    report = {
        "onnx_path": str(path),
        "onnx_size_bytes": path.stat().st_size,
        "parity_on_test_set": parity,
        "latency_batch_256": compare_latency(
            model, path, images[:256], cfg.threads, repeats=args.repeats
        ),
        "latency_method": "round-robin interleaved PyTorch vs ONNX Runtime, same thread budget",
        "environment": environment_info(),
    }
    write_json(Path(args.out_dir) / "onnx.json", report)
    print(json.dumps(report, indent=2))
    if not parity["allclose"]:
        raise SystemExit("ONNX Runtime outputs differ from PyTorch beyond tolerance")


def cmd_sample_png(args: argparse.Namespace) -> None:
    from PIL import Image

    from qvision.data import CLASS_NAMES

    images, labels = load_split(args.root, train=False)
    Image.fromarray(images[args.index], mode="L").save(args.out)
    print(f"wrote {args.out} (test image {args.index}, label: {CLASS_NAMES[labels[args.index]]})")


def cmd_serve(args: argparse.Namespace) -> None:
    import os

    import uvicorn

    # An explicit --run wins over an exported QV_CHECKPOINT; without --run the environment
    # variable (or the default run) is used.
    if args.run is not None:
        os.environ["QV_CHECKPOINT"] = str(Path(args.run) / "model.pt")
    else:
        os.environ.setdefault("QV_CHECKPOINT", str(Path(DEFAULT_RUN) / "model.pt"))
    uvicorn.run("qvision.serve:app_factory", factory=True, host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qvision", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download and verify Fashion-MNIST")
    p.add_argument("--root", default="data/fashion-mnist")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("train", help="train the CNN and log a run directory")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--epochs", type=int)
    p.add_argument("--train-subset", type=int, help="train on only N examples (smoke runs)")
    p.add_argument("--run-name")
    p.set_defaults(func=cmd_train)

    for name, func, help_text in [
        ("evaluate", cmd_evaluate, "full test-set evaluation report"),
        ("quantize", cmd_quantize, "fp32 vs INT8/INT4 quantization study"),
        ("export-onnx", cmd_export_onnx, "export to ONNX and check ONNX Runtime parity"),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--run", default=DEFAULT_RUN)
        p.add_argument("--out-dir", default="results")
        if name != "evaluate":
            p.add_argument("--repeats", type=int, default=50, help="latency timing repeats")
        p.set_defaults(func=func)

    p = sub.add_parser("sample-png", help="write a test-set image as a PNG for /predict")
    p.add_argument("--root", default="data/fashion-mnist")
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--out", default="sample.png")
    p.set_defaults(func=cmd_sample_png)

    p = sub.add_parser("serve", help="run the FastAPI service")
    p.add_argument("--run", help=f"run directory (overrides $QV_CHECKPOINT; default {DEFAULT_RUN})")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
