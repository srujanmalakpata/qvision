| Variant | Accuracy | Δ vs fp32 (pp) | Paired 95% CI of Δ (pp) | Discordant b/c | McNemar p | Agreement | Size (KB) | Compression | Latency batch 1 (ms, median) | Latency batch 256 (ms, median) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fp32 | 0.9301 | +0.00 | +0.00 to +0.00 | 0/0 | 1 | 1.0000 | 863.5 | 1.00x | 0.53 | 63.99 |
| int8_weight_only | 0.9303 | +0.02 | -0.03 to +0.08 | 3/5 | 0.727 | 0.9991 | 226.8 | 3.81x | 0.77 | 60.72 |
| int4_weight_only | 0.9272 | -0.29 | -0.56 to -0.01 | 109/80 | 0.0414 | 0.9795 | 120.2 | 7.18x | 2.26 | 61.71 |
| int8_torch_dynamic | 0.9303 | +0.02 | -0.05 to +0.09 | 5/7 | 0.774 | 0.9986 | 273.4 | 3.16x | 0.61 | 59.70 |

b = test images only fp32 classifies correctly, c = only the variant does. McNemar's exact test and the paired bootstrap use the same 10000 test images for both models, accounting for correlated errors an unpaired accuracy CI omits.
