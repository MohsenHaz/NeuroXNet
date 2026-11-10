# TRIPOD+AI checklist (Collins et al., BMJ 2024) - location of each item

| Item | Where |
|---|---|
| Title, abstract (1-2) | Manuscript |
| Introduction: background, objectives (3-4) | Section 1 |
| Methods: data sources, participants, outcome, predictors (5-8) | Sections 3.2, 3.9; outcome = four-level label from the fixed protocol |
| Sample size (9) | 491 development, 182 external; precision discussed with Wilson CIs |
| Missing data (10) | No imputation; excluded studies listed in `logs/preprocess_exclusions.csv` after `01_preprocess.py` |
| Analytical methods: model, training, selection, evaluation, calibration (11-13) | `configs/`, `src/neuroxnet/training`, `src/neuroxnet/eval` (ROC, reliability, Brier, decision curve) |
| Class imbalance (14) | Inverse-frequency weights Eq. (14), stratified splits |
| Fairness / subgroups (15) | Table 11 (vendor, thickness, age) |
| Model output and interpretation (16) | Study-level probabilities, threshold 0.5, tier only, no management recommendation |
| Development vs evaluation data (17) | Table 1; external cohort never used in development |
| Results: participants, model performance, updating (18-22) | Tables 5-15, Figures 3-9 |
| Discussion: limitations, usability, interpretation (23-26) | Sections 4.13-4.14, 5 |
| Other: registration, protocol, open science, funding, conflicts (27-29) | Not registered; code/weights/predictions on Zenodo; funding and conflicts statements |
