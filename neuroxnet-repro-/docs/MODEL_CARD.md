# Model card: NeuroXNet

**Task.** Study-level binary hydrocephalus detection and four-level severity grading (normal, mild, moderate, severe) from non-contrast head CT.
**Architecture.** EfficientNet-B4 backbone (ImageNet, timm), Multi-Scale Attention Module on F2-F5, Channel-Spatial Attention Gate, Feature Pyramid Network, detection head and severity head; 2D slice model, study-level mean of slice logits over 128 mid-volume slices.
**Input.** DICOM axial non-contrast series; brain window WL 40 / WW 80 HU; 128 slices centred on floor(S/2), zero-padded if S < 128; 256 x 256; normalised with training-set mean and std.
**Training data.** CQ500 (491 studies, India, multi-centre), four-level labels derived by two neuroradiologists with adjudication (weighted kappa 0.85); 344/74/73 patient-level split, seed 42.
**Evaluation.** Held-out CQ500 test set (73), external single-institution adult cohort (182, single reader), five-fold CV. All metrics from `results/tables`.
**Intended use.** Research only; decision support that must be confirmed by a radiologist. Not validated prospectively, not for paediatric imaging, not for obstructive/communicating subtyping, not validated in patients with marked cerebral atrophy.
**Known limitations.** Small moderate and severe classes; the 0.35 and 0.40 Evans' index boundaries are operational and not guideline-derived; the mild interval overlaps with age-related ventricular enlargement; external labels from one reader; no reader study.
**Weights.** `checkpoints/neuroxnet_final.pt`, SHA-256 in `checkpoints/SHA256SUMS.txt`.
