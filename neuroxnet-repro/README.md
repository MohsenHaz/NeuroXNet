# NeuroXNet reproducibility package

Code, configuration, tests and evaluation tooling that accompany the manuscript *Deep Learning-Based Detection and Severity
Grading of Hydrocephalus on Non-Contrast Brain CT: A Retrospective Development and External Validation Study Using the
NeuroXNet Framework*.

## What this repository contains

| Path | Content |
|---|---|
| `src/neuroxnet/` | Model (EfficientNetB4 backbone, MSAM with depthwise-separable convolutions, CSAG, FPN, detection and severity heads), preprocessing, training, cross-validation, inference, Grad-CAM, evaluation and consistency code |
| `configs/` | NeuroXNet, baseline, ablation and five-fold configurations (Table 4 of the manuscript) |
| `scripts/` | Numbered pipeline: download, preprocessing, training, baselines, cross-validation, prediction, evaluation, Grad-CAM, latency, tables and figures, release audit |
| `tests/` | Unit tests for Wilson intervals, metrics, slice selection, aggregation, split/fold arithmetic and the consistency gate |
| `environment/` | Pinned requirements, `environment.yml`, `Dockerfile` |
| `data/`, `predictions/` | Schemas (header-only CSV templates) for labels, splits, ROI subset, external predictions and per-study predictions |
| `docs/` | Model card, CLAIM and TRIPOD+AI checklists, paper-to-code map, deviations, status |

## What is deposited separately

The CQ500 images are obtained from the dataset providers (`scripts/00_download_cq500.sh`). The trained weights with SHA-256
checksums, the per-study probability predictions of NeuroXNet on the internal held-out test set and on the external cohort
(de-identified), the patient-level split file and the five-fold assignments are deposited in the Zenodo archive on
acceptance, as stated in the Data Availability Statement of the manuscript. The severity labels are available on request.
The external raw data are not redistributable. No file in this repository is a substitute for any of these artifacts:
the CSV files under `data/` and `predictions/` contain headers only until the real files are placed there.

## Running the evaluation on the deposited predictions

```bash
pip install -r environment/requirements.txt          # evaluation needs only numpy, pandas, scipy, scikit-learn
PYTHONPATH=src python scripts/07_evaluate_all.py      # writes results/tables/*.csv and runs the consistency gate
PYTHONPATH=src python -m neuroxnet.eval.consistency   # validates every reported proportion against its integer counts
PYTHONPATH=src python -m pytest tests                 # unit tests
```

The gate refuses to pass on header-only templates, checks that every accuracy, sensitivity and specificity is an exact
ratio of its own counts, and checks confusion-matrix totals, macro-F1 reconstruction and the Dice/IoU identity.

## Training

`scripts/02_train_main.sh` trains NeuroXNet with an effective batch of 8 studies per optimiser step (one study per forward
pass with eight gradient-accumulation steps and activation checkpointing). See `docs/DEVIATIONS.md` for how this differs
from a single 8-study batch.

## Citation

Please cite the manuscript when using this package (`CITATION.cff`).
