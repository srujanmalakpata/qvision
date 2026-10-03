# qvision model card: SmallCNN for Fashion-MNIST (`runs/fmnist-cnn/model.pt`)

## Model details

- **Architecture**: a VGG-style CNN with four 3×3 conv layers (conv-BatchNorm-ReLU), 16 then 32
  channels, two 2×2 max-pools, dropout 0.25, and a 1568→128→10 MLP head. 218,586 parameters.
- **Framework**: PyTorch 2.14.1 (CPU). The model is also exported to ONNX (opset 18, dynamic
  batch).
- **Training**: AdamW (lr 2e-3, weight decay 1e-4), cosine LR decay, batch size 128, at most 10
  epochs, early stopping on validation loss (patience 3) that restores the best weights (it did
  not trigger in the reported run: the best epoch was the last), seed 42.
  The full config is in `configs/default.yaml` and was copied to `results/train_config.yaml` for
  the reported run.
- **Variants**: fp32 (reference), INT8 and INT4 weight-only from this repo's per-channel
  symmetric quantizer, and INT8 dynamic (`torch.ao`, Linear layers only).
- **Weights are not distributed.** Run `make train` to reproduce them.

## Intended use

- Reproducible training and evaluation with uncertainty estimates, post-training quantization,
  ONNX export and local model serving.
- Classifying **Fashion-MNIST-style** images: 28×28 grayscale, a single centred clothing item,
  light on a dark background.

**Out of scope**: real product photos, colour images, multi-object images, any decision about
people, and any production or commercial use. The model is validated, never deployed.

## Data

- **Fashion-MNIST** (Xiao, Rasul & Vollgraf, 2017; Zalando Research, MIT licence): 70,000
  grayscale 28×28 images across 10 balanced classes (T-shirt/top, Trouser, Pullover, Dress,
  Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot).
- Split: the 60k official training images were split into 55,000 train and 5,000 validation
  (seeded permutation). The 10k official test images were used only for final reporting.
- Preprocessing: scale to [0, 1], then standardise with mean 0.2860 and std 0.3530 (computed
  on the 60k training images). No augmentation.
- Downloaded from the official bucket and MD5-verified. Not redistributed.

## Metrics (10,000-image test set, single training run)

| Metric | fp32 |
|---|---:|
| Accuracy | 0.9301 (95% bootstrap CI 0.9250-0.9349) |
| Macro-F1 | 0.9299 |
| Expected calibration error (15 bins) | 0.0167 |

Per-class F1: Trouser 0.991, Bag 0.987, Sandal 0.985, Ankle boot 0.974, Sneaker 0.972,
Dress 0.930, Pullover 0.896, Coat 0.893, T-shirt/top 0.880, **Shirt 0.789**.

| Quantized variant | Accuracy | Δ vs fp32 | Paired 95% CI of Δ | McNemar p | Size vs fp32 |
|---|---:|---:|---:|---:|---:|
| INT8 weight-only (own) | 0.9303 | +0.02 pp | −0.03 to +0.08 pp | 0.73 | 3.81× smaller |
| INT4 weight-only (own) | 0.9272 | −0.29 pp | −0.56 to −0.01 pp | 0.041 | 7.18× smaller |
| INT8 dynamic (`torch.ao`) | 0.9303 | +0.02 pp | −0.05 to +0.09 pp | 0.77 | 3.16× smaller |

The deltas are compared with a paired test (exact McNemar on the images where fp32 and the
variant disagree, plus a paired bootstrap), because both models are scored on the same 10,000
images. INT8 shows no detectable change; INT4's drop is small but statistically detectable.

Measured on a shared 4-vCPU Linux container on 2026-10-03. Latency results and caveats are
in README.md and `results/`.

## Limitations and risks

- **Shirt vs T-shirt/top/Pullover/Coat confusion** is the main error mode: 100 of the 1,000
  shirts were predicted as T-shirt/top. These classes look alike at this resolution.
- **Distribution shift**: Fashion-MNIST images are centred, size-normalised and inverted
  (bright object on a black background). Ordinary photos do not match, and the model will still
  produce confident outputs for them. The API rejects inputs that are not 28×28 PNGs, but it
  cannot detect out-of-distribution content.
- **Calibration** was measured, not corrected. ECE is low on in-distribution test data (81% of
  predictions fall in the 0.933-1.0 confidence bin, with accuracy 0.986 there against a mean
  confidence of 0.995; most lower bins are slightly over-confident, see the reliability table
  in `results/evaluation.md`), but that says nothing about out-of-distribution inputs.
- **Single seed**: the CI covers test-set sampling, not variation between training runs.
- **INT4** changes about 2% of top-1 predictions relative to fp32 (top-1 agreement 0.9795): it
  gets 109 test images wrong that fp32 got right and 80 right that fp32 got wrong, so aggregate
  accuracy moves only 0.29 pp, but that drop is statistically detectable (McNemar p 0.041).
- The weight-only quantized variants are smaller but **not faster** on CPU.

## Ethical considerations

The dataset contains product images only, with no people and no personal data. The main risk is
misuse outside the intended scope, for example presenting the model's confidence on real photos
as meaningful.

## How to reproduce

```bash
make setup && make all     # writes runs/fmnist-cnn/ and results/
```
