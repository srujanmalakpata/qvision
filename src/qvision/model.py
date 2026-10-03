"""The convolutional network and checkpoint (de)serialisation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn

from qvision.config import ModelConfig


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """3x3 conv -> batch norm -> ReLU. The conv has no bias because BN adds one."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class SmallCNN(nn.Module):
    """A VGG-style CNN for 1x28x28 inputs.

    Two stages of (conv, conv, 2x2 max-pool) take the image from 28x28 to 7x7, then a small
    MLP head produces logits for the 10 classes.
    """

    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c1, c2 = self.cfg.channels
        self.features = nn.Sequential(
            _conv_block(1, c1),
            _conv_block(c1, c1),
            nn.MaxPool2d(2),  # 28 -> 14
            _conv_block(c1, c2),
            _conv_block(c2, c2),
            nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(c2 * 7 * 7, self.cfg.hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden, self.cfg.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def save_checkpoint(model: SmallCNN, path: Path, extra: dict | None = None) -> None:
    """Save weights plus everything needed to rebuild the model without the training config."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_config": asdict(model.cfg), "state_dict": model.state_dict()}
    payload.update(extra or {})
    torch.save(payload, path)


def load_checkpoint(path: Path) -> SmallCNN:
    """Rebuild a SmallCNN from a checkpoint written by :func:`save_checkpoint` (eval mode)."""
    payload = torch.load(path, map_location="cpu", weights_only=True)
    cfg_dict = dict(payload["model_config"])
    cfg_dict["channels"] = tuple(cfg_dict["channels"])
    model = SmallCNN(ModelConfig(**cfg_dict))
    model.load_state_dict(payload["state_dict"])
    return model.eval()
