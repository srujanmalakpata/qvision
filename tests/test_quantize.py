import pytest
import torch
from torch import nn

from qvision.quantize import (
    QuantConv2d,
    QuantLinear,
    pack_int4,
    qmax_for,
    quantize_model_weights,
    quantize_per_channel,
    serialized_size_bytes,
    unpack_int4,
    weight_error_stats,
)

SHAPES = [(8, 3, 3, 3), (16, 32), (1, 5), (7, 1, 3, 3), (128, 1568)]


@pytest.mark.parametrize("bits", [8, 4])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("seed", range(3))
def test_error_bounded_by_half_scale(bits, shape, seed):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(shape, generator=gen) * (seed + 1)
    qt = quantize_per_channel(w, bits)
    err = (w - qt.dequantize()).abs().reshape(shape[0], -1)
    bound = qt.scale[:, None] / 2
    assert torch.all(err <= bound * (1 + 1e-5) + 1e-7)


@pytest.mark.parametrize("bits", [8, 4])
def test_codes_are_in_symmetric_range_and_channel_max_is_reproduced(bits):
    w = torch.randn(32, 64)
    qt = quantize_per_channel(w, bits)
    codes = qt.codes()
    qmax = qmax_for(bits)
    assert codes.dtype == torch.int8
    assert codes.min() >= -qmax and codes.max() <= qmax
    # The largest-magnitude weight in each channel maps to +-qmax and is reproduced up to
    # float rounding: (amax / qmax) * qmax can differ from amax by about one ulp.
    assert torch.all(codes.abs().amax(dim=1) == qmax)
    w_hat = qt.dequantize()
    idx = w.abs().argmax(dim=1)
    rows = torch.arange(32)
    torch.testing.assert_close(w_hat[rows, idx], w[rows, idx], rtol=1e-6, atol=0)


@pytest.mark.parametrize("bits", [8, 4])
def test_requantizing_dequantized_weights_is_idempotent(bits):
    qt = quantize_per_channel(torch.randn(16, 3, 3, 3), bits)
    again = quantize_per_channel(qt.dequantize(), bits)
    torch.testing.assert_close(again.scale, qt.scale)
    assert torch.equal(again.codes(), qt.codes())


def test_zero_channel_and_zero_weights_stay_zero():
    w = torch.randn(4, 10)
    w[2] = 0.0
    w[0, 3] = 0.0
    w_hat = quantize_per_channel(w, 8).dequantize()
    assert torch.all(w_hat[2] == 0) and w_hat[0, 3] == 0
    assert torch.isfinite(w_hat).all()


def test_scale_is_per_channel():
    w = torch.stack([torch.full((5,), 0.01), torch.full((5,), 100.0)])
    qt = quantize_per_channel(w, 8)
    torch.testing.assert_close(qt.scale, torch.tensor([0.01 / 127, 100.0 / 127]))
    torch.testing.assert_close(qt.dequantize(), w)


@pytest.mark.parametrize("n", [0, 1, 2, 7, 16, 1001])
def test_int4_pack_unpack_round_trip(n):
    q = torch.randint(-8, 8, (n,), dtype=torch.int8)
    packed = pack_int4(q)
    assert packed.dtype == torch.uint8 and packed.numel() == (n + 1) // 2
    assert torch.equal(unpack_int4(packed, n), q)


def test_int4_pack_layout_low_nibble_first():
    packed = pack_int4(torch.tensor([1, -1], dtype=torch.int8))
    assert packed.tolist() == [0xF1]


def test_pack_rejects_out_of_range():
    with pytest.raises(ValueError):
        pack_int4(torch.tensor([8], dtype=torch.int8))


def test_unsupported_bits():
    with pytest.raises(ValueError):
        quantize_per_channel(torch.randn(2, 2), bits=3)


def test_quant_modules_match_float_modules_with_dequantized_weights():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=True)
    lin = nn.Linear(20, 6)
    x_img, x_vec = torch.randn(2, 3, 9, 9), torch.randn(5, 20)
    for bits in (8, 4):
        qconv, qlin = QuantConv2d(conv, bits), QuantLinear(lin, bits)
        ref_conv = nn.functional.conv2d(
            x_img, quantize_per_channel(conv.weight, bits).dequantize(), conv.bias, 2, 1
        )
        torch.testing.assert_close(qconv(x_img), ref_conv)
        ref_lin = nn.functional.linear(
            x_vec, quantize_per_channel(lin.weight, bits).dequantize(), lin.bias
        )
        torch.testing.assert_close(qlin(x_vec), ref_lin)


def test_quantized_model_is_smaller_and_close_to_fp32(tiny_model):
    x = torch.randn(8, 1, 28, 28)
    with torch.inference_mode():
        ref = tiny_model(x)
        q8 = quantize_model_weights(tiny_model, 8)(x)
        q4 = quantize_model_weights(tiny_model, 4)(x)
    err8 = (q8 - ref).abs().max().item()
    err4 = (q4 - ref).abs().max().item()
    assert err8 < 0.05 * ref.abs().max().item()
    assert err8 < err4  # more bits, less error
    fp32_size = serialized_size_bytes(tiny_model)
    assert serialized_size_bytes(quantize_model_weights(tiny_model, 8)) < fp32_size
    assert serialized_size_bytes(quantize_model_weights(tiny_model, 4)) < serialized_size_bytes(
        quantize_model_weights(tiny_model, 8)
    )


def test_quantize_model_weights_does_not_mutate_original(tiny_model):
    before = {k: v.clone() for k, v in tiny_model.state_dict().items()}
    qmodel = quantize_model_weights(tiny_model, 8)
    assert any(isinstance(m, QuantConv2d) for m in qmodel.modules())
    assert not any(isinstance(m, QuantConv2d) for m in tiny_model.modules())
    for k, v in tiny_model.state_dict().items():
        assert torch.equal(v, before[k])


def test_quantized_state_dict_round_trips(tiny_model, tmp_path):
    qmodel = quantize_model_weights(tiny_model, 4)
    path = tmp_path / "q.pt"
    torch.save(qmodel.state_dict(), path)
    fresh = quantize_model_weights(tiny_model, 4)
    fresh.load_state_dict(torch.load(path, weights_only=True))
    x = torch.randn(2, 1, 28, 28)
    with torch.inference_mode():
        torch.testing.assert_close(fresh(x), qmodel(x))


def test_weight_error_stats_on_a_known_tensor():
    # One Linear layer, one channel, weights [1.0, 0.5, 0.3]: INT4 scale = 1/7.
    lin = nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        lin.weight.copy_(torch.tensor([[1.0, 0.5, 0.3]]))
    stats = weight_error_stats(nn.Sequential(lin), bits=4)
    # 0.5 * 7 = 3.5 rounds to 4 (half-to-even) -> 4/7; 0.3 * 7 = 2.1 -> 2/7.
    w_hat = torch.tensor([1.0, 4 / 7, 2 / 7])
    err = torch.tensor([1.0, 0.5, 0.3]) - w_hat
    assert stats["max_abs_error"] == pytest.approx(err.abs().max().item(), rel=1e-6)
    expected_rel = (err.norm() / torch.tensor([1.0, 0.5, 0.3]).norm()).item()
    assert stats["mean_relative_l2_error"] == pytest.approx(expected_rel, rel=1e-5)
