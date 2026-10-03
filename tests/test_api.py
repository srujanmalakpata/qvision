import io
import struct
import threading
import zlib

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from qvision.data import CLASS_NAMES, normalize
from qvision.serve import MAX_UPLOAD_BYTES, app_factory, create_app, preprocess_png


def png_bytes(size=(28, 28), mode="L", fmt="PNG") -> bytes:
    rng = np.random.default_rng(0)
    if mode == "L":
        array = rng.integers(0, 256, (size[1], size[0]), dtype=np.uint8)
    else:
        array = rng.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def client(tiny_model):
    return TestClient(create_app(tiny_model, model_name="tiny-random"))


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "tiny-random"}


def test_predict_returns_class_and_probabilities(client, tiny_model):
    raw = png_bytes()
    resp = client.post("/predict", files={"file": ("x.png", raw, "image/png")})
    assert resp.status_code == 200
    body = resp.json()
    assert list(body["probabilities"]) == list(CLASS_NAMES)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-5)
    assert body["label"] == CLASS_NAMES[body["class_index"]]
    assert body["confidence"] == pytest.approx(max(body["probabilities"].values()))
    # The API must agree with calling the model directly on the same preprocessed input.
    expected = int(tiny_model(preprocess_png(raw)).argmax())
    assert body["class_index"] == expected


def encode_png(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_grayscale_png_round_trips_to_the_training_tensor():
    # Independent of preprocess_png: the served tensor must equal what training feeds the model.
    pixels = np.random.default_rng(1).integers(0, 256, (28, 28), dtype=np.uint8)
    x = preprocess_png(encode_png(Image.fromarray(pixels, mode="L")))
    assert x.shape == (1, 1, 28, 28)
    assert torch.equal(x, normalize(pixels[None]))


@pytest.mark.parametrize("mode", ["RGB", "P"])
def test_colour_and_palette_pngs_match_pillow_grayscale(mode):
    rgb = np.random.default_rng(2).integers(0, 256, (28, 28, 3), dtype=np.uint8)
    image = Image.fromarray(rgb, mode="RGB")
    if mode == "P":
        image = image.convert("P")
    raw = encode_png(image)
    expected = normalize(np.array(Image.open(io.BytesIO(raw)).convert("L"))[None])
    assert torch.equal(preprocess_png(raw), expected)


def test_app_factory_caps_torch_threads(tmp_path, monkeypatch, tiny_model):
    from qvision.model import save_checkpoint

    save_checkpoint(tiny_model, tmp_path / "model.pt")
    monkeypatch.setenv("QV_CHECKPOINT", str(tmp_path / "model.pt"))
    monkeypatch.delenv("QV_QUANT_BITS", raising=False)
    monkeypatch.delenv("QV_TORCH_THREADS", raising=False)
    app_factory()
    assert torch.get_num_threads() == 1
    monkeypatch.setenv("QV_TORCH_THREADS", "zero")
    with pytest.raises(ValueError, match="QV_TORCH_THREADS"):
        app_factory()


def test_predict_accepts_rgb_png(client):
    resp = client.post("/predict", files={"file": ("x.png", png_bytes(mode="RGB"), "image/png")})
    assert resp.status_code == 200


def test_predict_rejects_wrong_size(client):
    resp = client.post("/predict", files={"file": ("x.png", png_bytes((32, 32)), "image/png")})
    assert resp.status_code == 422
    assert "28x28" in resp.json()["detail"]


def test_predict_rejects_non_image_and_non_png(client):
    resp = client.post("/predict", files={"file": ("x.png", b"not an image", "image/png")})
    assert resp.status_code == 422
    jpeg = png_bytes(mode="RGB", fmt="JPEG")
    resp = client.post("/predict", files={"file": ("x.jpg", jpeg, "image/jpeg")})
    assert resp.status_code == 422


def test_predict_requires_file(client):
    assert client.post("/predict").status_code == 422


def test_app_loads_quantized_checkpoint_from_env(tmp_path, monkeypatch, tiny_model):
    from qvision.model import save_checkpoint

    save_checkpoint(tiny_model, tmp_path / "model.pt")
    monkeypatch.setenv("QV_CHECKPOINT", str(tmp_path / "model.pt"))
    monkeypatch.setenv("QV_QUANT_BITS", "8")
    client = TestClient(create_app())
    assert client.get("/health").json()["model"] == "model.pt (int8 weight-only)"
    resp = client.post("/predict", files={"file": ("x.png", png_bytes(), "image/png")})
    assert resp.status_code == 200


def png_header_only(width: int, height: int) -> bytes:
    """A valid PNG signature + IHDR declaring ``width x height`` 8-bit grayscale, with a tiny
    IDAT. Decoding it would need ``width * height`` bytes of memory; the file is ~70 bytes."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(b"\x00" * 64))
        + chunk(b"IEND", b"")
    )


def test_huge_declared_dimensions_are_rejected_before_decoding(client, monkeypatch):
    from PIL import PngImagePlugin

    def no_decode(self, *args, **kwargs):
        raise AssertionError("pixel data was decoded")

    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", no_decode)
    raw = png_header_only(9000, 9000)  # 81 Mpx declared, below Pillow's own bomb limit
    assert len(raw) < 200
    resp = client.post("/predict", files={"file": ("x.png", raw, "image/png")})
    assert resp.status_code == 422
    assert "9000x9000" in resp.json()["detail"]


def test_decompression_bomb_sized_header_is_a_client_error(client):
    raw = png_header_only(30000, 30000)  # Pillow raises DecompressionBombError at open()
    resp = client.post("/predict", files={"file": ("x.png", raw, "image/png")})
    assert resp.status_code == 422


def test_sixteen_bit_png_is_rejected_not_clipped(client):
    gradient = np.linspace(0, 62720, 28 * 28, dtype=np.uint16).reshape(28, 28)
    buf = io.BytesIO()
    Image.fromarray(gradient).save(buf, format="PNG")
    resp = client.post("/predict", files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert resp.status_code == 422
    assert "mode" in resp.json()["detail"]


def test_oversized_request_is_rejected_from_content_length(client):
    big = b"\x00" * (2 << 20)
    resp = client.post("/predict", files={"file": ("x.png", big, "image/png")})
    assert resp.status_code == 413


def test_file_just_over_the_upload_cap_is_rejected(client):
    # Passes the request-level limit (which leaves room for multipart headers) but not the
    # per-file 1 MiB cap.
    raw = b"\x00" * (MAX_UPLOAD_BYTES + 1)
    resp = client.post("/predict", files={"file": ("x.png", raw, "image/png")})
    assert resp.status_code == 413
    assert "1 MiB" in resp.json()["detail"]


def test_chunked_upload_without_length_is_refused(client):
    def body():
        yield b"--x\r\n"

    resp = client.post(
        "/predict",
        content=body(),
        headers={"content-type": "multipart/form-data; boundary=x"},
    )
    assert resp.status_code == 411


def test_invalid_quant_bits_fails_at_startup(tmp_path, monkeypatch, tiny_model):
    from qvision.model import save_checkpoint

    save_checkpoint(tiny_model, tmp_path / "model.pt")
    monkeypatch.setenv("QV_CHECKPOINT", str(tmp_path / "model.pt"))
    monkeypatch.setenv("QV_QUANT_BITS", "3")
    with pytest.raises(ValueError, match="QV_QUANT_BITS"):
        create_app()


class GatedModel(torch.nn.Module):
    """Blocks inside forward() until the test opens the gate."""

    def __init__(self, inner: torch.nn.Module) -> None:
        super().__init__()
        self.inner = inner
        self.entered = threading.Event()
        self.gate = threading.Event()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.entered.set()
        self.gate.wait(timeout=10)
        return self.inner(x)


def test_health_answers_while_a_prediction_is_running(tiny_model):
    model = GatedModel(tiny_model)
    # One TestClient context = one event loop shared by both requests, like a real server.
    with TestClient(create_app(model, model_name="gated")) as client:
        result = {}

        def call_predict():
            files = {"file": ("x.png", png_bytes(), "image/png")}
            result["status"] = client.post("/predict", files=files).status_code

        worker = threading.Thread(target=call_predict)
        worker.start()
        try:
            assert model.entered.wait(timeout=10), "predict never reached the model"
            health = client.get("/health")
            # If /predict blocked the event loop, /health would only return after the gate
            # times out, by which point the prediction would have finished.
            assert health.status_code == 200
            assert worker.is_alive() and "status" not in result
        finally:
            model.gate.set()
            worker.join(timeout=10)
    assert result["status"] == 200
