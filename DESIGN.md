# qvision design notes

This document describes the architecture, design tradeoffs, and possible extensions.

## Goals and non-goals

Goals: reproducible CPU training, evaluation with uncertainty estimates, a quantization
comparison using a custom weight quantizer, ONNX export with runtime parity checks, and an HTTP
inference service.

Non-goals: state-of-the-art accuracy, GPU training, a hyper-parameter search, and production
deployment. The service is validated, never deployed.

## Data pipeline

* **No torchvision.** Fashion-MNIST ships as four gzipped IDX files. The IDX format is a 4-byte
  magic number (`00 00 <dtype> <ndim>`), the dimension sizes as big-endian uint32, then the raw
  bytes, so a 20-line NumPy parser is enough (`data.read_idx`). Dropping torchvision removes a
  large dependency and its version pinning against torch.
* **Checksums.** Every downloaded file is MD5-checked against the values published with the
  dataset, written to a `.part` file and renamed only when it verifies. Interrupted downloads
  therefore never leave a corrupt cache. MD5 is used because the dataset publishes MD5 and the
  check is only for integrity. It would not stop a deliberate attacker.
* **Fixed normalisation constants** (mean 0.2860, std 0.3530). They were computed from the 60k
  training images (`compute_mean_std` reproduces them) and are hard-coded so that the serving
  container does not need the dataset.
* **Validation split.** 5,000 of the 60,000 training images, chosen by a seeded
  `numpy.random.default_rng` permutation. The 10k test set is used only for final reporting.
  Early stopping and model selection look only at validation loss.
* **In-memory tensors.** The whole dataset as float32 is about 200 MB, which is small enough to
  normalise once and keep in a `TensorDataset`. That avoids DataLoader worker processes, which
  are a common source of non-determinism and slowness on small CPUs.

## Model

`SmallCNN` uses two VGG-style stages (`conv3x3-BN-ReLU` twice, then max-pool) with 16 and 32
channels and a 128-unit MLP head, for about 219k parameters. The width was chosen to keep a full
10-epoch run on a 2-thread CPU budget in the 6-33 minute range (6 min on a quiet machine). Doubling both widths to (32, 64)
roughly quadruples the multiply-adds of every conv layer after the first (they scale with
in_channels x out_channels), so it would take several times longer; that is an estimate from the
arithmetic, not a measured run. Wider CNNs typically reach about 93-94%+ on Fashion-MNIST; the
width is one line of config to change.

## Training

* **Determinism**: `random`, NumPy and torch are all seeded, `torch.use_deterministic_algorithms(True)`
  is set, and a seeded `torch.Generator` drives DataLoader shuffling. A test trains twice and
  asserts bit-identical weights. The deterministic flag and the thread count are process-wide
  switches, so `train.deterministic()` is a context manager that restores the previous values
  when `fit` returns; otherwise evaluation, a server or later tests in the same process would
  silently inherit them. Determinism holds for a fixed library version, thread count
  and CPU type. Different BLAS builds can still differ in the last bits.
* **AdamW with cosine LR decay** gives good results with almost no tuning. **Early stopping**
  tracks validation loss (patience 3, `min_delta` 1e-4) and **restores the best weights**.
  Without the restore, the returned model would be from the last epoch, not the best one. A
  unit test scripts the validation losses and checks the restore.
* **Tracking**: each run directory gets `config.yaml`, `env.json`, a per-epoch `metrics.csv` and
  `summary.json`. This is an MLflow-lite alternative: no server and no database, readable with
  `cat` or pandas. If several people needed to compare many runs, MLflow would be worth adding.

## Evaluation

All metrics are implemented in NumPy (`metrics.py`). The tests check the confusion matrix and
P/R/F1 against scikit-learn, McNemar against SciPy's `binomtest` (both test-only dependencies),
and ECE and the bootstrap against hand-computed cases.

* **Bootstrap 95% CI for accuracy**: 2,000 resamples of the 10k test predictions. The CI shows
  sampling uncertainty from the test set. It does not include training-seed variance, which
  would need several training runs.
* **ECE** (expected calibration error) uses 15 equal-width confidence bins of top-label
  confidence. The per-bin count, mean confidence and accuracy (a reliability table) are written
  to `evaluation.json`/`.md` next to the single ECE number. Calibration is reported but not
  corrected. Temperature scaling on the validation set would be the obvious next step.
* **Paired model comparison.** fp32 and a quantized variant classify the same 10k images,
  so their errors are strongly correlated. An unpaired ±0.5 pp accuracy CI ignores this
  correlation and is too conservative for the accuracy delta. `metrics.paired_comparison` counts the
  discordant images (b = only fp32 right, c = only the variant right), runs an exact McNemar
  test (under H0, `min(b, c)` ~ Binomial(b + c, 0.5)), and computes a paired-bootstrap CI of the
  accuracy delta by resampling image indices once per replicate and applying them to both
  correctness vectors. The study reports all three for every variant.

## Quantization

The core is `quantize.quantize_per_channel`:

```
scale_c = max |W_c| / q_max            (q_max = 127 for INT8, 7 for INT4)
q       = clamp(round(W / scale_c), -q_max, q_max)
W_hat   = q * scale_c
```

Decisions:

* **Symmetric, not affine.** Trained weights are roughly zero-centred, so an affine zero-point
  would sit near 0 anyway and buys almost no extra resolution. Fixing the zero-point at 0 also
  keeps integer kernels simple: with a weight zero-point `z_w`, an integer GEMM needs an extra
  correction term `z_w * sum_k x_k` for every output, and symmetric weights remove it. (Both
  schemes represent 0 exactly, because an affine zero-point is an integer by construction.)
  Affine or unsigned ranges matter more for *activations*: after ReLU they are all >= 0, and a
  symmetric signed range would waste half its codes. This quantizer is weight-only, so that
  case does not arise here.
* **Per output channel, not per tensor.** Channel magnitudes differ a lot, especially after BN.
  With a single scale, small channels would collapse to a few levels. Per-channel scales cost
  only 4 bytes per channel.
* **Restricted range (±127, ±7).** Not using -128 / -8 keeps the grid symmetric. The INT4
  packer still accepts -8 so that it works as a general-purpose function.
* **INT4 packing**: two two's-complement nibbles per byte, low nibble first. Unpacking
  sign-extends with `v - 16 if v > 7`.
* **Weight-only.** Activations stay fp32, and weights are dequantized inside `forward`. This
  measures accuracy and storage size honestly. It is **not** a speed optimisation: dequantizing
  on every call adds work, and the results show it. Real speedups need integer kernels (as in
  the `torch.ao` dynamic variant for Linear layers) or fused low-bit GEMMs.
* **No BN folding.** BN follows each conv, and per-channel weight scaling commutes with BN's
  per-channel affine transform, so folding is not needed for weight-only quantization. It
  *would* be needed for full integer inference.

The comparison baseline is **`torch.ao.quantization.quantize_dynamic`**. It quantizes Linear
weights to INT8 and quantizes activations per batch at runtime, using FBGEMM integer kernels.
It does not touch convolutions. `torch.ao.quantization` is deprecated in favour of the
separate `torchao` package (PyTorch 2.14 prints a deprecation warning), so `pyproject.toml`
pins `torch<2.15` and the code imports it lazily. Moving the baseline to `torchao` is listed
under next steps.

## ONNX

The model is exported with the `torch.export`-based exporter (`dynamo=True`, the default
since PyTorch 2.9) with a symbolic batch dimension and opset 18. The legacy TorchScript exporter
still works but is deprecated. Parity is **measured, not assumed**: the CLI runs the whole test
set through both runtimes and fails if `allclose(atol=1e-4, rtol=1e-4)` does not hold. It also
records the maximum absolute logit difference and top-1 agreement.

## Serving

* FastAPI with an app factory (`create_app(model=None)`). Tests inject a tiny random model, so
  API tests need neither a checkpoint nor network access.
* The service accepts **PNG only**, at **exactly 28x28**, in an 8-bit mode (`L`, `LA`, `P`,
  `RGB`, `RGBA` or 1-bit), converted to 8-bit grayscale. It returns `422` instead of resizing.
  Silent resizing or colour inversion would feed the model a distribution it was never trained
  on, and the API would return confident nonsense. 16-bit PNGs are rejected because
  `convert("L")` clips them to 255 instead of rescaling.
* **Untrusted input is checked cheapest-first.** A pure ASGI middleware (`BodySizeLimit`)
  rejects a POST whose `Content-Length` exceeds 1 MiB plus 16 KiB of multipart overhead (413)
  before any of the body is read, and refuses chunked uploads with no declared length (411).
  The endpoint then caps the file itself at 1 MiB (413). Finally `preprocess_png` checks the
  format, the declared size and the mode from the PNG *header* (`Image.open` does not decode
  pixels) before calling `convert`. Without that order, a ~315 KB PNG that declares 9000x9000
  pixels was fully decoded (1.9 s and +378 MB peak RSS in a decode-before-validation probe)
  before the 28x28 check rejected it; header validation rejects it in
  tens of milliseconds with a few MB of extra memory. Uvicorn's HTTP parser enforces the declared `Content-Length`, so a client
  cannot send more than it announced. A reverse proxy limit would still be the first line of
  defence in a real deployment.
* **Threading model.** `/predict` is a plain `def`, so FastAPI runs it in its worker threadpool
  (AnyIO, 40 threads by default) and the event loop stays free for `/health` and other
  requests. Image decoding and the forward pass are synchronous, CPU-bound work; declaring the
  endpoint `async def` would run them on the event loop and serialize everything, including
  the Docker HEALTHCHECK, behind each prediction. Concurrent torch inference from several
  threads is safe in eval mode (no shared mutable state in the forward pass). A test holds a
  prediction open inside the model and checks that `/health` still answers. Because up to 40
  requests can be in torch at once, `app_factory` sets `torch.set_num_threads` from
  `QV_TORCH_THREADS` (default 1); otherwise each concurrent request would use one intra-op
  thread per core and oversubscribe the CPU. Additional
  throughput would require uvicorn workers (one process per core) or request batching.
* `QV_QUANT_BITS=8|4` serves the hand-quantized variant; any other value fails at startup.
* The Docker image is a two-stage build: uv resolves the locked runtime dependencies into
  `/opt/venv` in the first stage, and the runtime stage copies only that venv and the
  checkpoint (no uv, no sources, no dev or ONNX-export packages). It uses the CPU-only torch
  wheel, runs as a non-root user, and has a HEALTHCHECK. The service binds to `127.0.0.1` by default. The container binds `0.0.0.0`
  internally and `make docker-run` publishes it on host loopback only.

## Testing strategy

* Unit tests: IDX parsing, the checksum-verified download (against a fake mirror), normalisation,
  splits, metrics (against scikit-learn/SciPy and hand-computed cases), and quantizer
  properties (error ≤ scale/2 over random shapes and seeds, idempotent re-quantization, exact
  channel max, zero handling, INT4 pack/unpack round trip, state-dict round trip).
* Training: a smoke run on 512 synthetic samples must learn (val acc > 0.5, chance is 0.1), a
  determinism test, and scripted early-stopping tests (stop and restore, patience running out on
  the final epoch is not an early stop, a NaN loss raises).
* Integration: ONNX parity (and a non-zero exit when parity fails), the API with `TestClient`
  (including the 411/413/422 limits, a header-only check for huge declared dimensions, and
  `/health` answering while a prediction is in flight), and an end-to-end CLI pipeline (train,
  evaluate, quantize, export) on fake IDX files written to a temp directory.
* An autouse fixture replaces `urllib.request.urlopen` with a function that raises, so any test
  that tried to download would fail.

## Alternatives considered

| Choice | Alternative | Why not (here) |
|---|---|---|
| Hand-rolled IDX loader | torchvision `FashionMNIST` | Large extra dependency for 20 lines of code |
| CSV/JSON run dirs | MLflow | Server/DB overhead for file-based run tracking |
| Weight-only fake-quant modules | Real INT8 GEMM via `torch._int_mm` | Private API with shape constraints; scope |
| `torch.ao` dynamic quant baseline | `torchao` | Extra package; `torch.ao` is still built in to 2.14 |
| Strict 28x28 input | Auto-resize + invert | Hides distribution shift from the caller |

## Possible extensions

1. Static INT8 quantization with calibration (activation observers), BN folding, and integer
   convs, to get an actual CPU speedup rather than only smaller weights.
2. GPTQ/AWQ-style error-compensating INT4, and per-group (e.g. 32-element) scales to recover
   the INT4 accuracy loss. Each row would be zero-padded to a multiple of the group size (conv1
   has 9 weights per output channel and conv2/conv3 have 144, not divisible by 32; conv4 (288)
   and the Linear rows (1568, 128) already are), with the
   original length stored so `dequantize` can trim the padding.
3. Temperature scaling for calibration, and multi-seed training to report seed variance.
4. Quantize the ONNX graph (ONNX Runtime `quantize_dynamic`/QDQ) and compare it with the PyTorch
   variants.
5. Move the baseline from `torch.ao` to `torchao` before PyTorch removes the former.
