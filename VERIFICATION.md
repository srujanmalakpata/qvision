# qvision test record

This record contains measured results and test status for the environment below. It does not
establish production readiness. The service is validated, never deployed.

## 2026-10-03 — macOS arm64 regression validation

Environment: macOS 27.0.1 arm64, Python 3.11.15, uv 0.11.21, PyTorch 2.14.1,
NumPy 2.4.6 and Ruff 0.16.10. A fresh process reports default quantization engine `none`
and supported engines `[qnnpack]`. The implementation now selects `qnnpack` before dynamic
INT8 weight packing. Tests also cover preference for `x86`/`fbgemm` and clear failure when no
supported engine is available.

The default uv cache/tool directories are outside writable paths. Commands used a writable
cache under `/private/tmp`; locked packages and the real Fashion-MNIST files were recovered
from a previous local cache. Fresh network access remains blocked. Training used fresh run
directories with the documented configs, changing only data/output paths. Two torch threads
were used. Timings were measured under concurrent load and are not performance guarantees.
The committed `results/` files retain the earlier Linux measurements; this run did not replace
them or infer an engine for historical JSON that lacks `settings.quantized_engine`.

`<check-root>` below is `/private/tmp/qvision-verification-2026-10-03`; command logs and new
JSON outputs are stored there locally.

| Check / command | Result | Evidence |
|---|---|---|
| Fresh `uv run pytest -q` dependency fetch | BLOCKED | PyPI DNS lookup failed; default uv cache access was also sandbox-blocked |
| `uv sync --frozen --offline --python <cached-python-3.11>` | PASS | Fresh project-local `.venv`, 50 locked packages installed from cache |
| `uv lock --check --offline` | PASS | Resolved 56 packages; lockfile unchanged |
| `uvx ruff@0.16.10 check . && uvx ruff@0.16.10 format --check .` | BLOCKED | Standalone tool resolution needs unavailable PyPI access; format was attempted separately and also blocked |
| `uv run ruff check .` / `uv run ruff format --check .` | PASS | Same locked Ruff 0.16.10; all checks passed, 32 files already formatted |
| `uv run pytest -q` with writable cached environment | PASS | **133 passed in 13.04 s**, including the fresh-process dynamic INT8 test and end-to-end CLI JSON engine assertion |
| `uv run qvision download --root <check-root>/fresh-data` | BLOCKED | Official dataset hostname DNS lookup failed |
| `uv run qvision download --root <check-root>/data/fashion-mnist` | PASS | All four copied real dataset files checksum-verified; train shape `(60000, 28, 28)` |
| `uv run qvision train --config <check-root>/smoke.yaml` | PASS | Unchanged smoke settings: 2 epochs, 2,048 training images, test accuracy 0.7427 |
| `uv run qvision evaluate --run <check-root>/runs/smoke --out-dir <check-root>/smoke-results` | PASS | Full 10,000-image test-set report |
| `uv run qvision quantize --run <check-root>/runs/smoke --out-dir <check-root>/smoke-results --repeats 5` | PASS | Matches CI; all four variants, recorded engine `qnnpack` |
| `uv run qvision export-onnx --run <check-root>/runs/smoke --out-dir <check-root>/smoke-results --repeats 5` | PASS | Matches CI; allclose and 100% top-1 agreement on 10,000 images |
| `uv run qvision sample-png --root <check-root>/data/fashion-mnist --out <check-root>/sample.png` | PASS | Generated a real test-set PNG |
| Smoke-checkpoint API via `TestClient(app_factory())`, fp32 / INT8 / INT4 | PASS | `/health` and `/predict` returned 200 for all three; each response had ten probabilities summing to one |
| `uv run qvision train --config <check-root>/default.yaml` | PASS | Full 10 epochs, 218,586 parameters, best epoch 10; test accuracy 0.9306, CI [0.9256, 0.9354], ECE 0.0151614; training 280.7 s under concurrent load |
| `uv run qvision evaluate --run <check-root>/runs/fmnist-cnn --out-dir <check-root>/full-results` | PASS | n=10,000; accuracy 0.9306, CI [0.9256, 0.9354] |
| `uv run qvision quantize --run <check-root>/runs/fmnist-cnn --out-dir <check-root>/full-results --repeats 50` | PASS | All four variants; engine `qnnpack`; dynamic INT8 accuracy 0.9308, serialized size 279,811 bytes |
| Same quantization command, `--repeats 60 --out-dir <check-root>/repeat-1`, then `repeat-2` | PASS | Both complete; engine `qnnpack`, identical accuracies and serialized sizes across all three studies |
| `uv run qvision export-onnx --run <check-root>/runs/fmnist-cnn --out-dir <check-root>/full-results --repeats 50` | PASS | n=10,000; max logit difference 1.5258789e-5, 100% top-1 agreement, allclose true |
| `docker info` / `docker buildx build --check .` | BLOCKED | Permission denied accessing the Colima Docker socket |
| Docker CI image build and container smoke test | BLOCKED | Requires the same inaccessible Docker daemon; no image/container created |
| Hosted GitHub Actions | NOT_RUN | No hosted workflow queried or triggered; local checks cannot establish hosted status |
| Markdown local-link validation / workflow YAML parsing | PASS | All local doc links resolve; workflow jobs are lint, test, train-smoke and docker |
| `make -n all sample serve`, tracked-artifact audit and ignore checks | PASS | Quickstart targets expand correctly; no tracked model/build/cache files; build, data, run and HTTP session outputs ignored |
| `git diff --check` | PASS | No whitespace errors |

The four earlier macOS failures (pytest, full pipeline, 50-repeat study and two 60-repeat
studies) shared the `NoQEngine` cause. All four checks now pass locally on macOS arm64.
Runtime selection is centralized in
`study.select_quantized_engine`; each new study writes the actual backend to
`settings.quantized_engine`. A subprocess regression test exercises the build's real default
without depending on engine changes made by earlier tests. No existing assertion was weakened
and no test was skipped.

Remaining limits: fresh network installation/downloads and Docker are BLOCKED; hosted CI and
multi-seed training are NOT_RUN. API checks used the in-process ASGI client, not a newly
deployed service. The earlier macOS Docker run built the unchanged Dockerfile and passed
container checks; that historical success is not a current Docker PASS.

## Earlier Linux measurements (2026-10-03)

The following sections preserve the earlier Linux results and their measurement conditions.
They do not establish current hosted CI status.

### Environment

- **Measurement date**: 2026-10-03. Full-pipeline results use fresh data, run directories and caches.
- **Machine**: shared 4-vCPU Linux container (Intel Xeon Processor @ 2.80 GHz, 15 GB RAM, x86_64,
  Linux 6.18, glibc 2.39), shared with other builds. The 1-minute load average is 0.9 to 6.0
  for the primary measurements, recorded in `results/train_summary.json`,
  `results/quantization.json` and `results/onnx.json` → `environment.load_average_1m`.
  Additional measurements at load 13-28 show substantially higher latency.
- **Software**: Python 3.11.15, uv 0.8.17, PyTorch 2.14.1+cpu, ONNX 1.23.1, ONNX Runtime 1.30.0,
  onnxscript 0.7.2, NumPy 2.4.6, SciPy 1.17.1, scikit-learn 1.9.1, FastAPI 0.142.2, Starlette
  1.7.0, uvicorn 0.54.0, Pillow 12.3.0, ruff 0.16.10, Docker 29.6.2 with buildx.
- **Threads**: `torch.set_num_threads(2)` and ONNX Runtime `intra_op_num_threads=2`.

## Commands and results

| # | Command | Result | Key output |
|---|---|---|---|
| 1 | `uv sync` (fresh `.venv`) | PASS | CPU torch `2.14.1+cpu` installed into project-local `.venv` |
| 2 | `uv lock` / `uv lock --check` | PASS | `Resolved 56 packages` |
| 3 | `uv run ruff check .` | PASS | `All checks passed!` |
| 4 | `uv run ruff format --check .` | PASS | `32 files already formatted` |
| 5 | `uv run pytest -q` | PASS | `126 passed in 13.05s` (73 test functions, some parametrized; no network, CPU, load 2.7) |
| 6 | `make clean && rm -rf data && make all` | PASS | `real 7m16.090s` end to end (download, train, evaluate, quantize, export-onnx), load 3.8 at start, 5.7 at end |
| 7 | `uv run qvision download` (inside `make all`) | PASS | 4 files downloaded from the official bucket and MD5-verified, train `(60000, 28, 28)` |
| 8 | `compute_mean_std` on the 60k training images | PASS | `(0.2860405969887955, 0.35302424451492254)`, matching `MEAN = 0.2860`, `STD = 0.3530` |
| 9 | `uv run qvision train --config configs/default.yaml` | PASS | 10 epochs, `stopped_early: false`, best epoch 10, best val loss 0.17701382174491884, val acc 0.9366, **test acc 0.9301**, CI [0.9250, 0.9349], `train_seconds 366.5`, epochs 32-49 s. Repeated training produces identical per-epoch losses (bit-reproducible) |
| 10 | `uv run qvision evaluate --run runs/fmnist-cnn` | PASS | acc 0.9301, **95% CI 0.9250-0.9349** (2,000 resamples), macro-F1 0.9299, ECE 0.0167 |
| 11 | `uv run qvision quantize --run runs/fmnist-cnn` (run 1, stored in `results/`, 50 repeats) | PASS | see table below; load 6.0 |
| 12 | same with `--repeats 60 --out-dir <output-dir>` (runs 2 and 3, latency only) | PASS | b=1 fp32 / INT8 / INT4 / dynamic: 0.60 / 0.87 / 2.41 / 0.69 ms (load 3.3) and 0.68 / 0.88 / 2.92 / 0.73 ms (load 3.2); b=256: 57.4 / 57.9 / 60.4 / 58.1 ms and 62.7 / 61.6 / 65.7 / 63.0 ms |
| 13 | `uv run qvision export-onnx --run runs/fmnist-cnn` | PASS | n=10000, max \|Δlogit\| 9.54e-6, top-1 agreement 1.0, `allclose: true`, ONNX 899,672 bytes; b=256 medians ONNX Runtime 15.9 ms vs PyTorch 56.4 ms (3.5×, load 5.7). No torchvision-registration log noise and no warnings printed |
| 14 | same, `--out-dir <output-dir>` (second latency run) | PASS | ONNX Runtime 15.9 ms vs PyTorch 55.2 ms (3.5×, load 0.9); parity identical |
| 15 | `uv run qvision train --config configs/smoke.yaml --run-name smoke{1,2}` | PASS | 2 epochs on 2,048 images, test acc 0.7422 both times; `diff` of `metrics.csv` without the `seconds` column is empty; 6.2 s and 6.3 s wall |
| 16 | `make -n evaluate` / `make -n RUN=runs/missing evaluate` | PASS | with a checkpoint: only `qvision evaluate`; without one: `make train` first (evaluate/quantize/onnx depend on `$(RUN)/model.pt`) |
| 17 | `qvision serve --run runs/fmnist-cnn --port 8041` + curl | PASS | `/health` → `{"status":"ok","model":"model.pt (fp32)"}`; test image 0 → `Ankle boot`, confidence 0.99999785; 3 MB upload → `413`; non-image → `422 could not decode image` |
| 18 | `QV_QUANT_BITS=4 qvision serve --run runs/fmnist-cnn --port 8042` + curl | PASS | `{"model":"model.pt (int4 weight-only)"}`, test image 0 → `Ankle boot` (0.99999893) |
| 19 | `yaml.safe_load(open('.github/workflows/ci.yml'))` | PASS | jobs `lint, test, train-smoke, docker` |
| 20 | `docker buildx build --check .` | PASS | `Check complete, no warnings found.` |
| 21 | `docker build --network host <build-context>` (two-stage image, modified context) | PASS | built in 85 s, image 1.54 GB; build of the unmodified repository context is NOT_RUN |
| 22 | `docker run -p 127.0.0.1:8043:8000` + curl + `docker inspect` | PASS | `/health` ok, test image 0 → `Ankle boot` (0.99999785), runs as UID 10001, `onnx`/`onnxruntime`/`pytest`/`sklearn` absent, HEALTHCHECK status `healthy` after 30 s. |
| 23 | GitHub Actions workflow on GitHub | NOT_RUN | Hosted workflow status was not verified during this run |

## Quantization results (`results/quantization.md`, run 1)

| Variant | Accuracy | Δ vs fp32 (pp) | Paired 95% CI of Δ (pp) | Discordant b/c | McNemar p | Agreement | Size (KB) | Compression | Latency b=1 (ms, median) | Latency b=256 (ms, median) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 0.9301 | +0.00 | +0.00 to +0.00 | 0/0 | 1 | 1.0000 | 863.5 | 1.00x | 0.53 | 63.99 |
| int8_weight_only | 0.9303 | +0.02 | -0.03 to +0.08 | 3/5 | 0.727 | 0.9991 | 226.8 | 3.81x | 0.77 | 60.72 |
| int4_weight_only | 0.9272 | -0.29 | -0.56 to -0.01 | 109/80 | 0.0414 | 0.9795 | 120.2 | 7.18x | 2.26 | 61.71 |
| int8_torch_dynamic | 0.9303 | +0.02 | -0.05 to +0.09 | 5/7 | 0.774 | 0.9986 | 273.4 | 3.16x | 0.61 | 59.70 |

- Weight error: INT8 max |ΔW| 0.00280, mean relative L2 0.0068. INT4 max |ΔW| 0.0514, mean
  relative L2 0.122.
- Repeated runs produce identical accuracy, sizes and weight errors; they are deterministic.
- Latency ratios vs fp32 over three runs (load 3.2-6.0): batch 1 own INT8
  1.29-1.46×, own INT4 4.03-4.29×, torch dynamic 1.07-1.15×; batch 256 all variants 0.93-1.05×
  with the sign changing between runs, i.e. no consistent speed-up or slow-down.
- **Latency depends on load, not only on the code.** Measurements at load 13-29 give fp32
  b=256 at 229-472 ms and INT4 b=1 at 10-30× fp32. At load 3.2-6.0, fp32 b=256 is 57-64 ms
  and INT4 b=1 is about 4× fp32. Measurements at load 8-16 give INT4 b=1 at 4.1-4.4× and
  fp32 b=256 at 60-92 ms. The weight-only variants are never meaningfully faster than fp32
  in these measurements.

## Bugs found by testing and fixed

- A mistyped MD5 for `t10k-labels-idx1-ubyte.gz` blocked downloads. The corrected checksum
  accepts the dataset: 10,000 labels, exactly 1,000 per class. Download tests check checksums,
  idempotent calls, corrupt-file replacement, and tampered-payload cleanup.
- Sequential latency timing exposed variants to different load conditions. Timing uses
  interleaved (round-robin) calls and records `load_average_1m`. `test_bench.py` checks that
  every candidate is timed.
- A brittle early-stopping test depended on noisy training losses. Scripted `run_epoch` results
  make the stop-and-restore test deterministic.
- `/predict` decoded images before checking size and performed synchronous work on the event
  loop. Header checks, `Content-Length` limits, 16-bit PNG rejection and threadpool execution
  prevent these problems. API tests cover rejection before decoding and `/health` responding
  during inference.
- Unpaired accuracy intervals omitted correlated errors between models. Exact McNemar and
  paired-bootstrap comparisons account for shared test images; tests check discordant counts
  and compare McNemar with SciPy.
- `fit()` left process-wide torch settings changed. It restores the deterministic flag and
  thread count; `test_fit_restores_process_wide_torch_settings` checks restoration.
- Invalid configuration and serving options were not consistently rejected or applied.
  Configuration is validated, `qvision serve --run` overrides `QV_CHECKPOINT`, and invalid
  `QV_QUANT_BITS` fails. Tests cover each behavior.
- `FitResult.stopped_early` was true when patience ran out on the final scheduled epoch.
  It is set only when the loop breaks early; a regression test covers the final-epoch case.
- NaN/inf loss caused `fit()` to save random initial weights and write `Infinity` to
  `summary.json`. It raises `FloatingPointError` naming the epoch; a regression test covers
  non-finite validation loss.
- API tests computed expected tensors through `preprocess_png` itself. Tests compare the
  served tensor with `normalize(pixels[None])` and Pillow grayscale conversion for RGB/palette
  inputs, using independent expected values.
- `export_onnx` suppressed all Python warnings and permanently changed a logger level. Its
  context manager changes only the torchvision-registration logger threshold and restores it;
  no Python warnings are filtered. The recorded export emits no warnings for this model.
- `evaluate`/`quantize`/`export-onnx` loaded the 60k training split unnecessarily.
  `load_test_set` loads only the test split; a regression test checks that behavior.
- The 40-thread request pool could oversubscribe the CPU. `app_factory` caps torch intra-op
  threads via `QV_TORCH_THREADS` (default 1); an API test checks the cap.
- `make -j` could evaluate before the checkpoint existed. Targets depend on `$(RUN)/model.pt`;
  the recorded `make -n` checks cover existing and missing checkpoints.

## Test and documentation coverage

- Download tests use an in-memory fixture for four downloads, an idempotent second call,
  corrupt-file replacement, and tampered payloads with no target or `.part` file left behind.
- `scipy` is a direct dev dependency for metric cross-checks; no `slow` pytest marker is defined.
- Convolution weight row lengths are 9, 144, 144 and 288; Linear row lengths are 1568 and 128.
  Input modes and early-stopping behavior are described in `DESIGN.md`; the `RunTracker`
  interface is defined in `src/qvision/tracking.py`.

## Limits and tests not run

- **GitHub Actions**: NOT_RUN. Hosted workflow status has not been verified here.
- **Multi-seed training**: NOT_RUN. The reported numbers come from one seed.
- **macOS dynamic INT8 comparison**: the earlier `NoQEngine` failure is fixed by runtime
  backend selection in `study.select_quantized_engine`. The 2026-10-03 macOS checks above
  validate the fix in a fresh process and record the selected engine in study JSON.
- **Docker**: the Linux results above used a modified build context. The earlier macOS run
  built the unchanged Dockerfile and passed health, prediction, UID and dependency-exclusion
  checks. Docker access in the current sandbox is BLOCKED by socket permissions.
- Early stopping does not trigger in the full run: validation loss is still improving at epoch
  10. Unit tests exercise stopping and best-weight restoration.
- The service is validated, never deployed. Production deployment is out of scope.
