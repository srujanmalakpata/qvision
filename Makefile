# Reproduce everything: `make all` (downloads data, trains, evaluates, quantizes, exports ONNX).
UV ?= uv
CONFIG ?= configs/default.yaml
# RUN must match <runs_dir>/<run_name> in CONFIG.
RUN ?= runs/fmnist-cnn
IMAGE ?= qvision:local

.PHONY: setup data train smoke evaluate quantize onnx all test lint format serve sample docker-build docker-run clean

setup:            ## create .venv with CPU-only PyTorch
	$(UV) sync

data: setup       ## download + checksum Fashion-MNIST into data/
	$(UV) run qvision download

train: data       ## full training run -> runs/fmnist-cnn/ (always retrains)
	$(UV) run qvision train --config $(CONFIG)

# File target so evaluate/quantize/onnx wait for a checkpoint even under `make -j`; it trains
# only when the checkpoint is missing (use `make train` to force a retrain).
$(RUN)/model.pt:
	$(MAKE) train

smoke: data       ## short (under ~1 min) training smoke run on 2k images -> runs/smoke/
	$(UV) run qvision train --config configs/smoke.yaml

evaluate: $(RUN)/model.pt ## accuracy, CI, per-class P/R/F1, confusion matrix, ECE -> results/
	$(UV) run qvision evaluate --run $(RUN)

quantize: $(RUN)/model.pt ## fp32 vs INT8/INT4 study -> results/quantization.{json,md}
	$(UV) run qvision quantize --run $(RUN)

onnx: $(RUN)/model.pt     ## export to ONNX + ONNX Runtime parity -> results/onnx.json
	$(UV) run qvision export-onnx --run $(RUN)

all: train
	$(MAKE) evaluate quantize onnx

test:
	$(UV) run pytest -q

lint:
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format:
	$(UV) run ruff format .
	$(UV) run ruff check --fix .

sample:           ## write a test image to sample.png for curl
	$(UV) run qvision sample-png --index 0 --out sample.png

serve:
	$(UV) run qvision serve --run $(RUN)

docker-build:     ## needs runs/fmnist-cnn/model.pt (run `make train` first)
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 127.0.0.1:8000:8000 $(IMAGE)

clean:
	rm -rf runs smoke-results .pytest_cache .ruff_cache sample.png
