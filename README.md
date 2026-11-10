# NeuroXNet reproducibility package

This package retrains and re-evaluates NeuroXNet (EfficientNet-B4 backbone, Multi-Scale
Attention Module, Channel-Spatial Attention Gate, Feature Pyramid Network, dual-task heads)
for hydrocephalus detection and four-level severity grading on non-contrast head CT, and it
regenerates every number, table and figure of the manuscript from stored per-study
prediction files. Nothing in `results/` is typed by hand: every table is produced by
`scripts/07_evaluate_all.py` from `predictions/**/*.csv`, and `scripts/10_make_tables_figures.py`
turns those tables into the figures. `src/neuroxnet/eval/consistency.py` is a gate that
refuses to build the tables if any reported proportion is not an exact ratio of its counts.

## Quick start (Colab or a workstation with one GPU)

```bash
git clone <this repository> neuroxnet-repro && cd neuroxnet-repro
pip install -r environment/requirements.txt
bash scripts/00_download_cq500.sh                    # CQ500 DICOM + public labels
python scripts/01_preprocess.py --config configs/neuroxnet.yaml
bash scripts/02_train_main.sh                        # NeuroXNet, seed 42, writes predictions/internal_test/neuroxnet.csv
bash scripts/03_train_baselines.sh                   # 7 baselines, same split, writes predictions/internal_test/<baseline>.csv
bash scripts/04_train_ablation.sh                    # 8 ablation configurations on the SAME 73-study test split
bash scripts/05_cross_validate.sh                    # 5-fold CV with inner validation split, per-fold prediction CSVs
bash scripts/06_predict_all.sh                       # re-runs inference from checkpoints (optional, for verification)
python scripts/07_evaluate_all.py                    # tables (CSV) + consistency gate
python scripts/08_gradcam_eval.py                    # Grad-CAM IoU/Dice/precision/recall on the annotated subset
python scripts/09_benchmark_latency.py               # Table 14
python scripts/10_make_tables_figures.py             # Figures 3, 4, 5, 6, 7 and the CV figure
```

`notebooks/NeuroXNet_Colab.ipynb` runs the same steps in order inside Colab.

## What is public and what is restricted

| Item | Status |
|---|---|
| Code, configs, tests, docs | public (this repository) |
| Trained NeuroXNet weights, SHA-256 in `checkpoints/SHA256SUMS.txt` | public (Zenodo deposit) |
| Patient-level split (`data/splits/cq500_split_seed42.csv`) and CV folds | public: CQ500 study IDs are public |
| Per-study predictions, internal test and external (de-identified) | public (Zenodo deposit) |
| Derived four-level severity labels (`data/labels/cq500_severity_labels.csv`) | available from the corresponding author on request, subject to the review of the annotation sessions |
| External cohort raw DICOM, institution name | restricted |

## Prediction CSV schema

Every model, every split, one row per study, threshold fixed at 0.5:

```
study_id,split,y_bin,y_sev,p0,p1,p2,p3,p_hydro,pred_bin,pred_sev,seed,git_hash
```

`p0..p3` are the post-softmax severity probabilities (normal, mild, moderate, severe),
`p_hydro` the sigmoid detection probability, `pred_bin = int(p_hydro >= 0.5)`,
`pred_sev = argmax(p0..p3)`. Severity labels: 0 normal, 1 mild, 2 moderate, 3 severe.

## Severity label CSV schema

```
study_id,reader1,reader2,final_label,evans_index,temporal_horn_dilation,periventricular_lucency,adjudicated_flag
```

## Which script produces which manuscript item

| Manuscript item | Produced by |
|---|---|
| Table 5 (binary detection, all models, counts + Wilson CI + AUC) | `07_evaluate_all.py` -> `results/tables/table5_binary.csv` |
| McNemar + bootstrap AUC difference, NeuroXNet vs each baseline | `07_evaluate_all.py` -> `results/tables/table5b_paired_tests.csv` |
| Table 6 (four-class, all models, from confusion matrices) | `results/tables/table6_severity.csv` |
| Table 7 (per-class metrics + one-vs-rest AUC) | `results/tables/table7_perclass.csv` |
| Figure 3 (ROC, internal + external), Figure 5 (reliability), Figure 6 (decision curve) | `10_make_tables_figures.py` |
| Figure 4 (confusion matrix) | `10_make_tables_figures.py` |
| Tables 8, 9, 10, 11 (external) | `results/tables/table8_external.csv`, `table9_external_cm.csv`, `table10_external_perclass.csv`, `table11_subgroups.csv` |
| Table 12 + Figure 7 (ablation) | `results/tables/table12_ablation.csv` |
| Table 13 (Grad-CAM localisation) | `08_gradcam_eval.py` -> `results/tables/table13_gradcam.csv` |
| Table 14 (latency) | `09_benchmark_latency.py` -> `results/tables/table14_latency.csv` |
| Table 15 (five-fold CV, per-fold counts + AUC) | `results/tables/table15_cv.csv` |
| Consistency report | `results/checks/consistency_report.txt` |

## Tests

```bash
pytest -q tests/
```

The tests pin Wilson intervals to known values, check the study-level logit aggregation,
check slice selection for S < 128, odd S and even S, and check that every metric derived
from a confusion matrix reproduces from its counts.
