import numpy as np
import torch

from qvision.onnx_export import INPUT_NAME, OUTPUT_NAME, check_parity, export_onnx, ort_session


def test_onnx_export_matches_pytorch(tiny_model, tmp_path):
    path = export_onnx(tiny_model, tmp_path / "model.onnx")
    inputs = torch.randn(64, 1, 28, 28, generator=torch.Generator().manual_seed(0))
    parity = check_parity(tiny_model, path, inputs, atol=1e-4, rtol=1e-4, batch_size=25)
    assert parity["allclose"], parity
    assert parity["top1_agreement"] == 1.0
    assert parity["max_abs_logit_diff"] < 1e-4


def test_onnx_model_has_dynamic_batch(tiny_model, tmp_path):
    path = export_onnx(tiny_model, tmp_path / "model.onnx")
    session = ort_session(path)
    for batch in (1, 3, 17):
        x = np.random.default_rng(batch).standard_normal((batch, 1, 28, 28)).astype(np.float32)
        (logits,) = session.run([OUTPUT_NAME], {INPUT_NAME: x})
        assert logits.shape == (batch, 10)
