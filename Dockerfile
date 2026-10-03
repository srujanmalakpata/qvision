# Inference image: CPU-only PyTorch + FastAPI. Build after `make train`, which writes
# runs/fmnist-cnn/model.pt (weights are not committed to git).

# --- build stage: resolve the locked runtime dependencies into /opt/venv -------------------
FROM python:3.11-slim AS build

RUN pip install --no-cache-dir "uv==0.8.17"

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

WORKDIR /src

# Dependencies first (cached layer). --no-default-groups leaves out the dev and export
# (onnx/onnxruntime/onnxscript) groups: serving needs neither.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-default-groups --no-install-project

# The project itself, installed as a regular (non-editable) package so /src is not needed later.
COPY src ./src
RUN uv sync --frozen --no-default-groups --no-editable

# --- runtime stage: only the virtualenv and the checkpoint; no uv, no sources --------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    QV_CHECKPOINT=/app/model/model.pt

COPY --from=build /opt/venv /opt/venv
COPY runs/fmnist-cnn/model.pt /app/model/model.pt

RUN useradd --create-home --uid 10001 app
USER app
WORKDIR /app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"

# 0.0.0.0 inside the container; publish it on the host as 127.0.0.1:8000 (see `make docker-run`).
CMD ["uvicorn", "qvision.serve:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
