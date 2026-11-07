# CLEVR-4 axis recovery summary

All scores use the exact 10531-image order from the recent semantic run (ordered-ID SHA-256 `f8ff5b4ea9ab483befa17fbb8ea4e28b0051adad5d870b7a97b31af0d5749197`). K-means uses K=10 and scikit-learn defaults; NMI uses arithmetic normalization. Highest-configuration comparisons use seed 0 so the control seeds are not treated as hyperparameter trials.

| taxonomy | highest configuration overall (seed 0) | NMI | best unmodified foundation model | NMI | gap | conservative seed spread | gap exceeds spread? |
|---|---|---:|---|---:|---:|---:|:---:|
| texture | dinov2-base (p=10) | 0.427720 | clip-base-patch16 | 0.045034 | +0.382685 | 0.015506 | yes |
| shape | dinov2-base | 0.927913 | dinov2-base | 0.927913 | +0.000000 | 0.020534 | no |
| color | mae-base (p=5) | 0.495658 | mae-base | 0.073311 | +0.422347 | 0.000540 | yes |
| count | mae-base | 0.146718 | mae-base | 0.146718 | +0.000000 | 0.001484 | no |

## Part A: representations without pretrained foundation models

| taxonomy | best no-pretraining configuration (seed 0) | NMI | best unmodified foundation model | NMI | gap | conservative seed spread | absolute gap exceeds spread? |
|---|---|---:|---|---:|---:|---:|:---:|
| texture | UAE | 0.028232 | clip-base-patch16 | 0.045034 | -0.016803 | 0.015506 | yes |
| shape | wavelet_scattering-64-3 | 0.093644 | dinov2-base | 0.927913 | -0.834268 | 0.020534 | yes |
| color | UAE | 0.033054 | mae-base | 0.073311 | -0.040257 | 0.000540 | yes |
| count | UAE | 0.145429 | mae-base | 0.146718 | -0.001289 | 0.001484 | no |

Part A is negative on texture: the best no-pretraining representation was `UAE` at NMI 0.028232, and the best scattering variant (`wavelet_scattering-64-3`) reached only 0.012518, versus 0.045034 for `clip-base-patch16`. No no-pretraining arm exceeded the best unmodified foundation model on any taxonomy.

The best scattering control is `wavelet_scattering-64-3`, selected before its extra seeds by highest seed-0 texture NMI. “Conservative seed spread” is the larger population standard deviation of NMI across seeds 0–2 for unmodified DINOv2-base and this scattering variant.

## Seed-control NMI standard deviations

| taxonomy | DINOv2-base | best scattering |
|---|---:|---:|
| texture | 0.015506 | 0.000740 |
| shape | 0.020534 | 0.001164 |
| color | 0.000131 | 0.000540 |
| count | 0.000100 | 0.001484 |

## Best spectral-surgery settings

- texture: `dinov2-base` p=10, NMI 0.427720.
- shape: `dinov2-base` p=0, NMI 0.927913.
- color: `mae-base` p=5, NMI 0.495658.
- count: `mae-base` p=0, NMI 0.146718.

Spectral surgery did recover a previously weak texture axis: DINOv2 rose from NMI 0.034604 at p=0 to 0.427720 at p=10. It did not reproduce the reported roughly sevenfold MAE shape gain at p=3: NMI changed from 0.408491 to 0.414936.

MAE was evaluated as an intact-image encoder with inference `mask_ratio=0.0`; otherwise the Hugging Face model applies a stochastic 75% pre-training mask in evaluation mode. Sparse random projection uses the repository default 64×64 input and 1,000 output dimensions.
