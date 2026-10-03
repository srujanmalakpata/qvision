"""FastAPI inference service: POST a 28x28 PNG to /predict, get the class and probabilities.

The model is loaded once at startup from ``QV_CHECKPOINT`` (default ``runs/fmnist-cnn/model.pt``).
Set ``QV_QUANT_BITS=8`` or ``4`` to serve this repo's weight-only quantized variant instead.
Tests inject a model directly through :func:`create_app`.

Concurrency model: ``/predict`` is a plain ``def`` endpoint, so FastAPI runs it in its worker
threadpool. PNG decoding and the torch forward pass are CPU-bound and synchronous; running them
on the event loop (``async def``) would block every other request, including ``/health``, for
the duration of each prediction. Because up to 40 requests can run in that pool at once,
:func:`app_factory` caps torch's intra-op threads (``QV_TORCH_THREADS``, default 1) so concurrent
requests do not each spawn one thread per core and oversubscribe the CPU.

Untrusted-input limits, cheapest check first:

1. :class:`BodySizeLimit` rejects a request from its ``Content-Length`` header before any of the
   body is read (413), and rejects chunked uploads that do not declare a length (411).
2. The uploaded file itself is capped at 1 MiB (413).
3. :func:`preprocess_png` checks format, dimensions and colour mode from the PNG *header*
   before decoding any pixels, so a tiny file that declares a huge image is rejected without
   allocating memory for it (422).
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from torch import nn

from qvision.data import CLASS_NAMES, normalize
from qvision.model import load_checkpoint
from qvision.quantize import quantize_model_weights

DEFAULT_CHECKPOINT = "runs/fmnist-cnn/model.pt"
IMAGE_SIZE = (28, 28)
MAX_UPLOAD_BYTES = 1 << 20
# Room for the multipart boundary and part headers around a maximum-size file.
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + (16 << 10)
# Modes that convert to 8-bit grayscale without losing the 0..255 intensity scale. 16-bit and
# 32-bit modes (I;16, I, F) are rejected: convert("L") clips them instead of rescaling.
ALLOWED_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA"})
QUANT_BITS_CHOICES = ("4", "8")


class Prediction(BaseModel):
    label: str
    class_index: int
    confidence: float
    probabilities: dict[str, float]


class Health(BaseModel):
    status: str
    model: str


class InvalidImageError(ValueError):
    """Raised when an upload is not a decodable 28x28 image."""


class BodySizeLimit:
    """ASGI middleware: refuse oversized POST bodies before reading them.

    Without this, Starlette parses (and spools to a temp file) the whole multipart body before
    the endpoint gets a chance to look at its size.
    """

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "http" and scope["method"] in {"POST", "PUT", "PATCH"}:
            headers = dict(scope["headers"])
            length = headers.get(b"content-length")
            if length is None and b"chunked" in headers.get(b"transfer-encoding", b""):
                response = JSONResponse({"detail": "Content-Length required"}, status_code=411)
                await response(scope, receive, send)
                return
            if length is not None and (not length.isdigit() or int(length) > self.max_bytes):
                response = JSONResponse({"detail": "request body too large"}, status_code=413)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


def preprocess_png(raw: bytes) -> torch.Tensor:
    """PNG bytes -> normalised tensor ``[1, 1, 28, 28]``.

    The image is converted to 8-bit grayscale. Like Fashion-MNIST, a light garment on a dark
    background is expected; nothing is resized, because silently rescaling would change the
    input distribution the model was trained on.

    ``Image.open`` parses only the header, so format, size and mode are checked before any pixel
    data is decompressed.
    """
    try:
        with Image.open(io.BytesIO(raw)) as img:
            if img.format != "PNG":
                raise InvalidImageError(f"expected a PNG, got {img.format}")
            if img.size != IMAGE_SIZE:
                raise InvalidImageError(f"expected a 28x28 image, got {img.size[0]}x{img.size[1]}")
            if img.mode not in ALLOWED_MODES:
                raise InvalidImageError(
                    f"unsupported PNG mode {img.mode!r}; use 8-bit grayscale, RGB or palette"
                )
            gray = img.convert("L")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise InvalidImageError("could not decode image") from exc
    return normalize(np.array(gray, dtype=np.uint8)[None])


def load_serving_model() -> tuple[nn.Module, str]:
    path = Path(os.environ.get("QV_CHECKPOINT", DEFAULT_CHECKPOINT))
    if not path.exists():
        raise RuntimeError(f"checkpoint not found: {path} (run `make train` first)")
    bits = os.environ.get("QV_QUANT_BITS", "").strip()
    if bits and bits not in QUANT_BITS_CHOICES:
        raise ValueError(
            f"QV_QUANT_BITS must be one of {QUANT_BITS_CHOICES} or unset, got {bits!r}"
        )
    model = load_checkpoint(path)
    if bits:
        return quantize_model_weights(model, int(bits)), f"{path.name} (int{bits} weight-only)"
    return model, f"{path.name} (fp32)"


def create_app(model: nn.Module | None = None, model_name: str = "injected") -> FastAPI:
    app = FastAPI(title="qvision", version="0.1.0")
    app.add_middleware(BodySizeLimit, max_bytes=MAX_REQUEST_BYTES)
    if model is None:
        model, model_name = load_serving_model()
    model.eval()
    app.state.model = model
    app.state.model_name = model_name

    @app.get("/health", response_model=Health)
    def health() -> Health:
        return Health(status="ok", model=app.state.model_name)

    # Plain `def`: FastAPI runs it in a threadpool, so CPU-bound decoding and inference do not
    # block the event loop (and /health keeps answering while a prediction is running).
    @app.post("/predict", response_model=Prediction)
    def predict(file: UploadFile = File(...)) -> Prediction:  # noqa: B008
        raw = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="image larger than 1 MiB")
        try:
            x = preprocess_png(raw)
        except InvalidImageError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        with torch.inference_mode():
            probs = torch.softmax(app.state.model(x), dim=1)[0]
        idx = int(probs.argmax())
        return Prediction(
            label=CLASS_NAMES[idx],
            class_index=idx,
            confidence=float(probs[idx]),
            probabilities={name: float(p) for name, p in zip(CLASS_NAMES, probs, strict=True)},
        )

    return app


def torch_threads_from_env() -> int:
    raw = os.environ.get("QV_TORCH_THREADS", "1").strip()
    if not raw.isdigit() or int(raw) < 1:
        raise ValueError(f"QV_TORCH_THREADS must be a positive integer, got {raw!r}")
    return int(raw)


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn --factory qvision.serve:app_factory``.

    Sets the process-wide torch thread count here rather than in :func:`create_app`, so tests
    that build apps in-process do not change it as a side effect.
    """
    torch.set_num_threads(torch_threads_from_env())
    return create_app()
