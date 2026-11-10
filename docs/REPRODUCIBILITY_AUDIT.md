# NeuroXNet reproducibility audit

**Audit basis:** `NeuroXNet_revised_28-10-2026.docx` and the accompanying `neuroxnet-repro.zip`, inspected together.  
**Scope:** source/code and artifact audit; no clinical data, labels, checkpoints or predictions were fabricated.

## Executive finding

The archive contains a useful implementation scaffold: model modules, data/preprocessing code, training and evaluation scripts, configurations, seven unit tests, environment descriptions and reporting checklists. The included unit tests pass in the inspection environment (7/7), and Python source compilation completes.

It is **not yet a complete numerical reproduction package**. The uploaded ZIP contains no trained weights, prediction CSVs, patient-level split files, severity-label file, external prediction file, ROI masks, preprocessed arrays, result tables, figures, or training logs. Several of these are described by the README and manuscript as deposited or available, but they are absent from this supplied archive.

The manuscript's key numerical findings can be documented, but they cannot be independently regenerated from this ZIP alone.

## What is present

- NeuroXNet architecture: EfficientNetB4 backbone, MSAM, CSAG, FPN, detection and severity heads.
- Baseline model implementations and YAML configurations.
- DICOM loading, brain-windowing, slice selection, resizing and normalization code.
- Training, cross-validation, inference, Grad-CAM, latency and evaluation scripts.
- Seven unit tests; all passed in the inspection environment.
- Environment files and CLAIM/TRIPOD+AI checklist drafts.
- `docs/DEVIATIONS.md`, which already records several intended changes from the submitted manuscript.

## What is absent from the uploaded ZIP

The following paths now exist in the updated repository, but several contain templates/placeholders rather than actual study artifacts:

| Required artifact | Intended path | Status in supplied ZIP |
|---|---|---|
| CQ500 study-level split | `data/splits/cq500_split_seed42.csv` | Header-only template; generator added, historical IDs not supplied |
| Five-fold assignments | `data/splits/cq500_cv5_folds.csv` | Header-only template; generator added, historical IDs not supplied |
| Derived reader and consensus labels | `data/labels/cq500_severity_labels.csv` | Header-only template; source labels not supplied |
| Fixed annotation protocol | `data/labels/annotation_protocol.pdf` | Placeholder PDF; replace with approved protocol |
| Grad-CAM ROI study IDs and masks | `data/labels/gradcam_roi_subset_ids.csv`, `data/roi_masks/` | Header-only ID template; masks absent |
| External model predictions | `data/external/external_predictions_deidentified.csv` | Header-only schema; predictions absent |
| Trained final checkpoint and checksums | `checkpoints/neuroxnet_final.pt`, `checkpoints/SHA256SUMS.txt` | Checksum instructions present; weights absent |
| Baseline/ablation/CV checkpoints | `checkpoints/baselines/`, `checkpoints/ablation/`, `checkpoints/cv5/` | Empty |
| Per-study prediction scores | `predictions/` | Header-only templates; no predictions |
| Preprocessed 128×256×256 studies | `data/preprocessed/` | Empty |
| Generated tables, figures and logs | `results/`, `logs/` | Output directories and README/status files; no experimental outputs |

The CQ500 source data itself is external and must be downloaded under its terms. The derived severity labels and external cohort are not part of the public CQ500 release.

## Reproduction levels

1. **Code-level execution:** partly supported. The code and tests are included, but execution of the full pipeline requires data and labels that are absent.
2. **Training from scratch:** not possible from this ZIP alone. The author must supply the derived labels and split assignments, and the CQ500 images must be obtained separately.
3. **Regeneration of manuscript tables:** not possible from this ZIP alone. The scripts expect per-study prediction files that are absent.
4. **Verification of the reported trained model:** not possible from this ZIP alone. The final checkpoint and NeuroXNet per-study scores are absent.
5. **External validation reproduction:** not possible from this ZIP alone. External images are restricted and the promised de-identified external prediction CSV is absent.
6. **Grad-CAM quantitative reproduction:** not possible from this ZIP alone. ROI identifiers and masks are absent.
7. **Exact environment reproduction:** partly specified, not locked. `requirements.txt` uses lower bounds; `environment.yml` pins Python/PyTorch/CUDA but leaves most packages floating.

## Manuscript-to-code discrepancies requiring resolution

### 1. Through-plane resampling is not implemented

The manuscript states that CT volumes are resampled using trilinear interpolation and describes slice-thickness standardization. The supplied `preprocess.py` performs in-plane resizing at the existing slice count. `dicom_io.py` reads slice thickness and pixel spacing, but does not resample the volume in the through-plane direction or use pixel spacing to standardize physical dimensions.

**Required author decision:** specify the target voxel spacing and orientation policy used for the reported experiments, then implement it or revise the manuscript to describe the actual preprocessing. Do not invent a target spacing.

### 2. Normalization-statistic computation differs from the stated input pipeline

`01_preprocess.py` calculates mean and standard deviation over each complete brain-windowed volume. It does so before the fixed 128-slice selection and before in-plane resizing. The manuscript describes population statistics computed from training data, while the model consumes selected and resized slices.

**Required author decision:** confirm the exact tensor population used to obtain 0.4723 and 0.2281. Recompute and document the statistics using that exact pipeline, or revise the code/configuration to match the originally used calculation.

### 3. Ablation protocol differs from the manuscript

The manuscript states that the ablation experiment used a separate internal split and warns that its values are not interchangeable with the 73-study held-out test results. `docs/DEVIATIONS.md` says the package changes the ablation to the same 73-study test split and copies the NeuroXNet held-out predictions into the full-model ablation row.

**Consequence:** the regenerated Table 12 is a new experiment and cannot be presented as the originally reported ablation table. Update the manuscript and identify the new split/protocol, or restore the original split and its retained evidence if available. Copying the final model row is acceptable only as a clearly labelled reference row, not as an independently rerun ablation.

### 4. Model selection is not fully specified in the manuscript

The training code selects the checkpoint using the sum of validation AUROC and validation macro-F1 (`val_auroc + val_macro_f1`) and early-stops on a combined validation loss. The manuscript says model selection was based on the best validation AUROC and best macro-F1, but does not specify the scalar combination or tie-breaking rule.

**Required author decision:** state whether the two metrics were jointly combined, whether one task had priority, or whether separate checkpoints were selected.

### 5. Parameter-count definition needs alignment

The latency script counts the NeuroXNet backbone plus MSAM/CSAG/FPN while excluding the two task heads; baseline parameter counts are backbone-only. The manuscript describes 19.4M as the total NeuroXNet parameter count. The manuscript and script must use the same definition and state whether heads are included.

### 6. Reproducibility of augmentation needs a worker-seeding policy

The dataset stores a manually seeded `torch.Generator`, while training uses multiple DataLoader workers. A worker-specific seeding policy for this generator is not defined. Add deterministic worker initialization or per-study/per-epoch augmentation seeds, and record the policy. This matters for repeatability of training augmentations.

### 7. External reader protocol is not equivalent to internal annotation

The CQ500 labels were independently assigned by two neuroradiologists with adjudication; the external cohort was labelled by one experienced neuroradiologist without a second reader or adjudication. The manuscript acknowledges this as a component of domain shift. Keep this distinction explicit in every external validation description.

### 8. Numerical comparisons and statistical claims

The manuscript states that baseline comparisons are descriptive and no paired test was performed because baseline per-study scores were not retained. Do not claim statistically significant superiority. The package's current evaluation script attempts paired McNemar and paired bootstrap AUC comparisons when predictions exist; those are new analyses and require aligned per-study predictions and a manuscript update.

## Reported results to preserve as manuscript-reported values

| Evaluation | Manuscript-reported result |
|---|---|
| Internal binary detection | Accuracy 97.26% (71/73); sensitivity 96.55% (28/29); specificity 97.73% (43/44); AUROC 0.9891 |
| Internal four-class severity | Accuracy 94.52% (69/73); macro-F1 0.9201 |
| External binary detection | Accuracy 95.60% (174/182); sensitivity 96.51%; specificity 94.79%; AUROC 0.9734 |
| External severity | Accuracy 91.21% (166/182); macro-F1 0.8876 |
| Grad-CAM ventricular ROI overlap | Mean IoU 0.779; mean Dice 0.875 |
| Full 128-slice inference | 21.6 ms per study on NVIDIA A100 40 GB (as reported) |
| Reader agreement on CQ500 labels | Linearly weighted Cohen's κ 0.85 (95% CI 0.80–0.90) |
| Internal data split | 344 train / 74 validation / 73 test; seed 42 |
| Dataset class totals | 298 normal / 78 mild / 69 moderate / 46 severe |
| External cohort | 182 studies: 96 normal / 34 mild / 31 moderate / 21 severe |

These are transcribed as manuscript-reported results, not independently re-computed from this ZIP.

## Important interpretation limits

- This is a retrospective development and external validation study, not a prospective clinical workflow or reader study.
- The four-level severity taxonomy is a study-specific operational grading protocol. The 0.35 and 0.40 boundaries are not claimed to be established guideline thresholds.
- The mild class overlaps with age-related ventricular enlargement in some older adults.
- The model does not distinguish communicating from obstructive hydrocephalus.
- The model is 2D slice-wise; it does not use 3D convolutions or inter-slice feature learning. Study-level predictions average slice-level logits.
- External raw DICOM and institutional identifiers are restricted.
- No clinical deployment claim should be made from these retrospective results.

## Release recommendation

Do not describe the current uploaded ZIP as a complete reproducibility deposit. It is a code package that requires restricted/author-supplied inputs and missing result artifacts. Before public release, either attach the permitted artifacts or clearly mark each as restricted/not retained and revise the manuscript's Data Availability Statement accordingly.

## Changes in the updated repository

- Added `scripts/make_frozen_splits.py`, which enforces 344/74/73 split sizes, test class counts 44/12/10/7, and five-fold sizes 99/98/98/99/97 with positive counts 39/38/39/39/38. It was tested against synthetic metadata only; this verifies allocation arithmetic, not historical patient-ID identity.
- Reworked `scripts/01_preprocess.py` to process one study at a time, create output directories, and compute training-only normalization statistics incrementally after slice selection and in-plane resizing.
- Reduced default training memory load to one study per batch and eight slices per chunk, enabled non-reentrant activation checkpointing during training, and added deterministic DataLoader worker seeding. Actual A100 40-GB peak memory and runtime remain to be measured.
- Pinned package versions and expanded the consistency gate. Empty templates are not treated as completed experiment outputs.
- Compilation passed and all seven included tests passed in the build environment.
