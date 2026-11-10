# NeuroXNet experimental protocol

This document converts the manuscript's stated protocol into an operational runbook. It does not resolve undocumented implementation choices.

## 1. Research task and labels

Each study has:
- binary label: normal (0) versus hydrocephalus-positive (1);
- severity label: normal (0), mild (1), moderate (2), severe (3).

The manuscript's grading rules:
- Normal: Evans' index <0.30 and no temporal horn dilation.
- Mild: 0.30 ≤ Evans' index <0.35, slight temporal horn dilation and slight periventricular lucency.
- Moderate: 0.35 ≤ Evans' index <0.40, slight-to-moderate temporal horn dilation and slight periventricular lucency.
- Severe: Evans' index ≥0.40, severe temporal horn dilation and marked periventricular hypodensity/lucency.

The manuscript states that qualitative findings take precedence when the index and qualitative assessment conflict. The Evans' index is measured on native-resolution axial source images. The label is study-level.

Two neuroradiologists independently read all 491 CQ500 studies; discordances were adjudicated by a senior neuroradiologist. Pre-adjudication weighted Cohen's κ is reported as 0.85 (95% CI 0.80–0.90). External labels were assigned by one experienced neuroradiologist without a second reader/adjudication.

## 2. Data and splits

CQ500: 491 adult non-contrast head CT studies, 193,317 slices, one study per patient.
- Train: 344 studies.
- Validation: 74 studies.
- Internal held-out test: 73 studies.
- Seed: 42.
- Stratification: four-level severity label.
- No patient, series or slice overlap across splits.

Class counts:
| Split | Normal | Mild | Moderate | Severe | Total |
|---|---:|---:|---:|---:|---:|
| Train | 209 | 55 | 48 | 32 | 344 |
| Validation | 45 | 11 | 11 | 7 | 74 |
| Test | 44 | 12 | 10 | 7 | 73 |
| Total | 298 | 78 | 69 | 46 | 491 |

External cohort: 182 studies; 96 normal, 34 mild, 31 moderate, 21 severe. It is test-only and must not be used for training, tuning or model selection.

## 3. Preprocessing stated in manuscript

- Primary axial non-contrast series only.
- Brain window: WL 40 HU, WW 80 HU.
- 128 consecutive axial slices centred on the mid-volume index floor(S/2).
- If fewer than 128 slices: symmetric zero-padding; padded slices are included in the study-level logit average.
- In-plane size: 256×256.
- Normalize using training-population mean 0.4723 and standard deviation 0.2281.
- Manuscript also states trilinear resampling and slice-thickness handling; supplied code does not currently implement through-plane resampling. Resolve this before claiming exact implementation equivalence.

## 4. Architecture

Slice-wise 2D forward pass:
1. ImageNet-pretrained EfficientNetB4 feature extractor.
2. MSAM on four feature levels, with kernel scales 1, 3, 5, 7 and learnable scale weights.
3. CSAG channel attention followed by spatial attention.
4. FPN top-down fusion.
5. Parallel binary detection and four-class severity heads.

No 3D convolution, inter-slice pooling or volumetric context is used in the network forward pass.

Study-level inference:
- compute independent logits for each of the 128 selected slices;
- arithmetic mean of slice-level detection logits and severity logits;
- sigmoid for binary probability and softmax for four-class probabilities;
- binary threshold 0.5; severity argmax.

## 5. Training configuration

Manuscript/config values:
- PyTorch 2.0.1; CUDA 11.8; NVIDIA A100 40 GB.
- ImageNet-pretrained EfficientNetB4 (timm).
- AdamW; initial LR 1e-4; cosine annealing.
- Maximum 100 epochs.
- Batch size 8 studies (8×128 slice inputs per optimization step).
- Early stopping patience 15 on validation loss.
- Seed 42.
- Label smoothing ε=0.1 for severity.
- Class-weighted binary loss.
- Spatial augmentation: rotation ±10°, horizontal flip p=0.5, translation ±0.1.
- Intensity augmentation: scale 0.9–1.1, gamma 0.8–1.2, Gaussian noise σ≤0.02.
- Reported regularization coefficients: 0.005 L2 and 0.005 feature orthogonality; total λr=0.01.
- Config currently sets λd=1.0 and λs=1.5; confirm these values in the manuscript because the table's loss-weight cell is not rendered with numerical values.

The supplied code selects a checkpoint by `validation AUROC + validation macro-F1` and early-stops on combined validation loss. The manuscript must state this exact rule or the code must be changed to the historical rule.

## 6. Baselines

ResNet50, VGG16, DenseNet121, EfficientNetB0, EfficientNetB4, TransUNet and Attention U-Net. Manuscript states same patient-level training split, preprocessing and optimization schedule. The supplied code adapts the classifier heads for the two tasks. Confirm architecture-specific training details and initialization before claiming strict baseline equivalence.

## 7. Ablations

Manuscript table includes:
- backbone only;
- +MSAM;
- +CSAG;
- +FPN;
- +label smoothing;
- detection only;
- focal loss;
- class-balanced loss;
- full joint model.

The ablation configurations are provided for completeness. If they are run, they must use the same split and the 73-study held-out test set as the main evaluation, and their per-study predictions must be saved. The manuscript reports no ablation results.

## 8. Evaluation and uncertainty

Binary detection:
- AUROC, AUPRC, accuracy, sensitivity, specificity, precision, F1;
- fixed threshold 0.5;
- Wilson score 95% CI for proportions when integer counts are available.

Severity:
- accuracy, balanced accuracy, macro-F1, weighted-F1, confusion matrix, Cohen's κ and linearly weighted κ;
- Brier score for calibration where probabilities are available.

Grad-CAM:
- threshold normalized activation map at 0.5;
- compare with expert ventricular ROI;
- report IoU and Dice (Dice = 2 IoU/(1+IoU)).

Latency:
- A100 40 GB;
- 128 slices per complete study;
- include preprocessing, slice selection, resizing and forward pass;
- record warm-up count, timed repetitions, median/dispersion, CUDA/PyTorch versions and exact model batch/chunk policy.

Cross-validation:
- five folds, patient-level and stratified;
- keep fold-wise results separate from the single 73-study held-out test evaluation.

## 9. Required outputs

For each model/run, save:
- checkpoint and SHA-256;
- config snapshot;
- git commit/hash;
- seed and environment versions;
- epoch history;
- one-row-per-study validation/test predictions with labels and probabilities;
- confusion matrices and metric tables;
- exact split/fold IDs.

Do not overwrite the original held-out test predictions when rerunning a new protocol.
