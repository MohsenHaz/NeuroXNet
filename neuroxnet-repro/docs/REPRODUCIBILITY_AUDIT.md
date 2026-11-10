# Reproducibility status

## Contents of the repository
Model, preprocessing, training, cross-validation, inference, Grad-CAM, latency, evaluation and consistency code; baseline,
ablation and cross-validation configurations; unit tests; pinned environment files; reporting checklists; CSV schemas.

## Reproduction levels
| Level | Requirement |
|---|---|
| Code execution and unit tests | This repository and the pinned environment |
| Table regeneration for NeuroXNet | The deposited per-study predictions (internal test and external) placed in `predictions/` and `data/external/` |
| Training from scratch | CQ500 images, the derived severity labels and the frozen split files |
| Verification of the trained model | The deposited checkpoint and its SHA-256 checksum |
| Grad-CAM quantitative reproduction | The ROI subset identifiers and masks, where release is permitted |
| External validation | The deposited de-identified external predictions; the external images are not redistributable |

## Consistency gate
`src/neuroxnet/eval/consistency.py` rejects header-only templates and checks probability/prediction consistency,
confusion-matrix totals, exact-ratio attainability of accuracy, sensitivity and specificity (each against its own
denominator), macro-F1 reconstruction and the Dice/IoU identity. `scripts/11_audit_release.py` lists which of the required
artifacts are present.
