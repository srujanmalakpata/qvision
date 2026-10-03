"""Hand-written post-training weight quantization (symmetric, per output channel).

For a weight tensor ``W`` with output channels along axis 0, each channel ``c`` gets its own
scale ``s_c = max|W_c| / q_max`` and is stored as integers ``q = clamp(round(W / s_c),
-q_max, q_max)``. Dequantization is ``W_hat = q * s_c``. With round-to-nearest the error per
element is bounded by ``s_c / 2``, and the largest-magnitude weight of each channel maps to
``+-q_max``, so it is reproduced up to float rounding (``(amax / q_max) * q_max`` is not always
bit-identical to ``amax`` in float32; the difference is about one ulp).

* INT8: ``q_max = 127`` (the "restricted" symmetric range; -128 is never used so that the
  grid is symmetric: ``w`` and ``-w`` always map to ``q`` and ``-q``).
* INT4: ``q_max = 7``; two 4-bit values are packed into each byte for storage.

This is *weight-only* quantization: activations stay in float32 and the integer weights are
dequantized inside ``forward``. It shrinks the model on disk / in memory; it does not use
integer matmul kernels, so it is not expected to be faster on CPU (see DESIGN.md).
"""

from __future__ import annotations

import copy
import io
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

SUPPORTED_BITS = (4, 8)


def qmax_for(bits: int) -> int:
    if bits not in SUPPORTED_BITS:
        raise ValueError(f"bits must be one of {SUPPORTED_BITS}, got {bits}")
    return 2 ** (bits - 1) - 1


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack a 1-D int8 tensor with values in [-8, 7] into uint8, two values per byte.

    Element ``2k`` goes in the low nibble and ``2k + 1`` in the high nibble. Odd lengths
    are padded with a zero nibble.
    """
    if q.dim() != 1:
        raise ValueError("pack_int4 expects a 1-D tensor")
    if q.numel() and (q.min() < -8 or q.max() > 7):
        raise ValueError("values out of int4 range [-8, 7]")
    if q.numel() % 2:
        q = torch.cat([q, q.new_zeros(1)])
    nibbles = (q.to(torch.int16) & 0x0F).to(torch.uint8)  # two's complement low 4 bits
    return nibbles[0::2] | (nibbles[1::2] << 4)


def unpack_int4(packed: torch.Tensor, numel: int) -> torch.Tensor:
    """Inverse of :func:`pack_int4`; returns ``numel`` int8 values."""
    lo = (packed & 0x0F).to(torch.int8)
    hi = (packed >> 4).to(torch.int8)
    out = torch.stack([lo, hi], dim=1).flatten()[:numel]
    return torch.where(out > 7, out - 16, out)  # sign-extend 4-bit two's complement


@dataclass
class QuantizedTensor:
    """Integer codes + per-channel float scales for one weight tensor."""

    data: torch.Tensor  # int8 codes (bits=8) or packed uint8 nibbles (bits=4)
    scale: torch.Tensor  # float32, shape [channels]
    bits: int
    shape: torch.Size

    def codes(self) -> torch.Tensor:
        """Integer codes as int8, shaped ``[channels, -1]``."""
        q = unpack_int4(self.data, self.shape.numel()) if self.bits == 4 else self.data
        return q.reshape(self.shape[0], -1)

    def dequantize(self) -> torch.Tensor:
        return (self.codes().float() * self.scale[:, None]).reshape(self.shape)


def quantize_per_channel(weight: torch.Tensor, bits: int = 8) -> QuantizedTensor:
    """Symmetric per-output-channel quantization of ``weight`` (channels on axis 0)."""
    qmax = qmax_for(bits)
    w = weight.detach().float().reshape(weight.shape[0], -1)
    amax = w.abs().amax(dim=1)
    # An all-zero channel would give scale 0 (division by zero); any positive scale encodes it
    # exactly as zeros, so use 1.0.
    scale = torch.where(amax > 0, amax / qmax, torch.ones_like(amax))
    q = torch.clamp(torch.round(w / scale[:, None]), -qmax, qmax).to(torch.int8)
    data = pack_int4(q.flatten()) if bits == 4 else q
    return QuantizedTensor(data=data, scale=scale, bits=bits, shape=weight.shape)


class _WeightOnlyQuantBase(nn.Module):
    """Stores quantized weights as buffers so they appear in ``state_dict``."""

    def _init_quant(self, weight: torch.Tensor, bias: torch.Tensor | None, bits: int) -> None:
        qt = quantize_per_channel(weight, bits)
        self.bits = bits
        self.weight_shape = tuple(weight.shape)
        self.register_buffer("qweight", qt.data)
        self.register_buffer("scale", qt.scale)
        self.register_buffer("bias", None if bias is None else bias.detach().clone())

    def dequantized_weight(self) -> torch.Tensor:
        qt = QuantizedTensor(self.qweight, self.scale, self.bits, torch.Size(self.weight_shape))
        return qt.dequantize()


class QuantLinear(_WeightOnlyQuantBase):
    def __init__(self, linear: nn.Linear, bits: int) -> None:
        super().__init__()
        self._init_quant(linear.weight, linear.bias, bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequantized_weight(), self.bias)

    def extra_repr(self) -> str:
        return f"weight={self.weight_shape}, bits={self.bits}, per_channel_symmetric"


class QuantConv2d(_WeightOnlyQuantBase):
    def __init__(self, conv: nn.Conv2d, bits: int) -> None:
        super().__init__()
        if conv.padding_mode != "zeros":
            raise NotImplementedError("only zero padding is supported")
        self._init_quant(conv.weight, conv.bias, bits)
        self.stride, self.padding = conv.stride, conv.padding
        self.dilation, self.groups = conv.dilation, conv.groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantized_weight()
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self) -> str:
        return f"weight={self.weight_shape}, bits={self.bits}, per_channel_symmetric"


def quantize_model_weights(model: nn.Module, bits: int = 8) -> nn.Module:
    """Return a copy of ``model`` with every Conv2d/Linear replaced by a weight-only quant module.

    BatchNorm and everything else stay in float32.
    """
    qmodel = copy.deepcopy(model).eval()

    def replace(module: nn.Module) -> None:
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                setattr(module, name, QuantLinear(child, bits))
            elif isinstance(child, nn.Conv2d):
                setattr(module, name, QuantConv2d(child, bits))
            else:
                replace(child)

    replace(qmodel)
    return qmodel


def weight_error_stats(model: nn.Module, bits: int) -> dict[str, float]:
    """Worst-case absolute and mean relative (L2) weight error over all Conv2d/Linear layers."""
    max_abs, rel_errors = 0.0, []
    for module in model.modules():
        if isinstance(module, nn.Conv2d | nn.Linear):
            w = module.weight.detach()
            w_hat = quantize_per_channel(w, bits).dequantize()
            max_abs = max(max_abs, (w - w_hat).abs().max().item())
            rel_errors.append(((w - w_hat).norm() / w.norm()).item())
    return {"max_abs_error": max_abs, "mean_relative_l2_error": sum(rel_errors) / len(rel_errors)}


def serialized_size_bytes(model: nn.Module) -> int:
    """Serialized state-dict size in bytes, as written by ``torch.save``."""
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes
