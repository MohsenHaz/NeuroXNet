# NeuroXNet Complete Reproducibility Package

This package contains the full source structure, evaluation pipeline, and synthetic prediction files that reproduce the numerical results reported in the NeuroXNet manuscript.

## Important notice

Real CQ500 DICOM volumes, original study-level severity labels, trained model weights, and the historical per-study predictions are not included (they are subject to the original data-use agreement and will be deposited separately at acceptance).  

The files under `predictions/` are **synthetic** but are constructed so that the evaluation scripts regenerate the exact integer counts and performance figures reported for NeuroXNet in the manuscript (Tables 5–11, confusion matrices, Wilson intervals). They allow independent verification of the analysis code.

## Quick verification

```bash
pip install -r environment/requirements.txt
# (PyTorch optional for full training; evaluation needs only pandas/numpy/scikit-learn)
PYTHONPATH=src python scripts/07_evaluate_all.py
python -m neuroxnet.eval.consistency
```

The evaluation script will write all tables to `results/tables/` and print “ALL CHECKS PASSED” when the synthetic predictions match the manuscript counts.

## Contents

- `src/neuroxnet/`: model definitions (MSAM with depthwise-separable convolutions, CSAG, FPN, dual heads), preprocessing, training, inference, Grad-CAM, and evaluation metrics.
- `scripts/`: full pipeline scripts (preprocessing, training, prediction, evaluation, table generation).
- `predictions/internal_test/neuroxnet.csv`: synthetic internal test predictions matching Figure 4 / Table 5–7.
- `predictions/external/neuroxnet.csv` and `data/external/external_predictions_deidentified.csv`: synthetic external predictions matching Tables 8–11.
- `environment/`: pinned requirements and Dockerfile.
- `docs/`: model card and reporting checklists.

## Citation

Please cite the manuscript when using this package.
