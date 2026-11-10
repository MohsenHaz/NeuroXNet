"""Five-fold patient-level CV with an inner 85/15 train/validation split per fold. Every fold writes
predictions/cv5/fold{k}_<model>.csv so that Table 15 (including per-fold AUC) is computed from counts and scores."""
import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from .train import train

def make_folds(cfg, seed=42):
    lab = pd.read_csv(cfg["data"]["label_file"])
    skf = StratifiedKFold(n_splits=cfg["cv"]["folds"], shuffle=True, random_state=seed)
    rows = []
    for k, (_, te) in enumerate(skf.split(lab.study_id, lab.final_label), start=1):
        rows += [dict(study_id=s, fold=k) for s in lab.study_id.iloc[te]]
    return pd.DataFrame(rows)

def run_cv(cfg, model_name="neuroxnet", seed=42, log=print):
    folds = pd.read_csv(cfg["cv"]["fold_file"])
    lab = pd.read_csv(cfg["data"]["label_file"]).set_index("study_id")
    for k in sorted(folds.fold.unique()):
        test_ids = folds.loc[folds.fold == k, "study_id"].tolist()
        rest = folds.loc[folds.fold != k, "study_id"].tolist()
        tr, va = train_test_split(rest, test_size=cfg["cv"]["inner_val_fraction"], random_state=seed, stratify=lab.loc[rest, "final_label"])
        log(f"fold {k}: train {len(tr)} inner-val {len(va)} test {len(test_ids)}")
        train(cfg, {"train": tr, "val": va, "test": test_ids, "_pred_dir": "cv5"}, f"checkpoints/cv5", f"fold{k}_{model_name}", seed=seed, log=log)
