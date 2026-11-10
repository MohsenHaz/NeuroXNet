#!/usr/bin/env bash
# Re-runs inference from the saved checkpoints and compares with the stored prediction CSVs (verification step).
set -euo pipefail
python - <<'PY'
import sys, pathlib, pandas as pd, numpy as np; sys.path.insert(0,"src")
from neuroxnet.config import load_config; from neuroxnet.data.dataset import read_split
from neuroxnet.inference.predict import predict_from_checkpoint
ids = read_split(load_config("configs/neuroxnet.yaml"))["test"]
pairs = [("checkpoints/neuroxnet_final.pt","predictions/internal_test/neuroxnet.csv")] + \
        [(f"checkpoints/baselines/{b}.pt", f"predictions/internal_test/{b}.csv") for b in "resnet50 vgg16 densenet121 effb0 effb4 transunet attunet".split()]
for ck, csv in pairs:
    if not pathlib.Path(ck).exists(): print("missing", ck); continue
    new = predict_from_checkpoint(ck, ids, "test", f"results/checks/rerun_{pathlib.Path(csv).name}")
    old = pd.read_csv(csv).set_index("study_id").loc[new.study_id]
    print(csv, "max |dp_hydro| =", float(np.abs(old.p_hydro.values - new.p_hydro.values).max()), "decisions identical:", bool((old.pred_bin.values == new.pred_bin.values).all()))
PY
