# Paper-to-code and artifact map

| Paper component | Current implementation | Required data/output |
|---|---|---|
| Section 3.2 preprocessing | `src/neuroxnet/data/dicom_io.py`, `preprocess.py`, `slice_select.py`; `scripts/01_preprocess.py` | CQ500 DICOM, severity labels, normalization log, preprocessed `.npy` |
| Section 3.3 EfficientNetB4 | `src/neuroxnet/models/backbone.py` | timm pretrained weights/cache; model checkpoint |
| Section 3.4 MSAM | `src/neuroxnet/models/msam.py` | config and checkpoint |
| Section 3.5 CSAG | `src/neuroxnet/models/csag.py` | config and checkpoint |
| Section 3.6 FPN | `src/neuroxnet/models/fpn.py` | config and checkpoint |
| Sections 3.7–3.8 task heads/loss | `models/heads.py`, `training/losses.py` | labels and checkpoint |
| Section 3.9 training | `training/train.py`, `scripts/02_train_main.sh` | split file, labels, preprocessed studies, environment |
| Baselines | `models/baselines.py`, `configs/baselines/`, `scripts/03_train_baselines.sh` | same training data and saved per-study scores |
| Ablation | `configs/ablation/`, `scripts/04_train_ablation.sh` | documented split and predictions per configuration |
| Five-fold CV | `training/cross_validate.py`, `scripts/05_cross_validate.sh` | fold file and fold predictions |
| Inference | `inference/aggregate.py`, `inference/predict.py`, `scripts/06_predict_all.sh` | checkpoint and study inputs |
| Tables/metrics | `eval/metrics.py`, `wilson.py`, `consistency.py`, `scripts/07_evaluate_all.py` | per-study CSV predictions |
| Grad-CAM | `explain/gradcam.py`, `explain/roi_overlap.py`, `scripts/08_gradcam_eval.py` | checkpoint, ROI IDs and ROI masks |
| Latency | `scripts/09_benchmark_latency.py` | hardware, model configs and timing logs |
| Figures | `scripts/10_make_tables_figures.py` | generated tables, scores and histories |
