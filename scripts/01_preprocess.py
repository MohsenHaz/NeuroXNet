"""Converts every CQ500 study to data/preprocessed/<study_id>.npy (128 x 256 x 256), logs the training-set
normalisation constants, and (with --make-splits) creates the patient-level 70/15/15 split and the 5-fold assignment."""
import argparse, json, pathlib, sys, numpy as np, pandas as pd
sys.path.insert(0, "src")
from neuroxnet.config import load_config
from neuroxnet.data.dicom_io import read_series, select_primary_series
from neuroxnet.data.preprocess import brain_window, preprocess_volume, training_set_statistics
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/neuroxnet.yaml"); ap.add_argument("--raw", default="data/raw/cq500")
ap.add_argument("--make-splits", action="store_true"); ap.add_argument("--seed", type=int, default=42); a = ap.parse_args()
cfg = load_config(a.config); d = cfg["data"]
lab = pd.read_csv(d["label_file"])
out = pathlib.Path(d["root"]); out.mkdir(parents=True, exist_ok=True)

if a.make_splits:
    tr, rest = train_test_split(lab, test_size=0.30, random_state=a.seed, stratify=lab.final_label)
    va, te = train_test_split(rest, test_size=73 / len(rest), random_state=a.seed, stratify=rest.final_label)
    pd.concat([tr.assign(split="train"), va.assign(split="val"), te.assign(split="test")])[["study_id", "split"]].to_csv(d["split_file"], index=False)
    from neuroxnet.training.cross_validate import make_folds
    make_folds(load_config("configs/cv5.yaml"), a.seed).to_csv("data/splits/cq500_cv5_folds.csv", index=False)
    print("splits written; commit them and never regenerate after training")

split = pd.read_csv(d["split_file"]).set_index("study_id")["split"]
# pass 1: training-set statistics on [0,1] windowed voxels (resized, selected window) so that the config constants are reproducible
hu_cache = {}
for sid in lab.study_id:
    sdir = select_primary_series(pathlib.Path(a.raw) / sid)
    vol, thick, _ = read_series(sdir)
    if not (d["thickness_range_mm"][0] <= thick <= d["thickness_range_mm"][1]) and thick > 0:
        print(f"warning: {sid} slice thickness {thick} mm outside configured range")
    hu_cache[sid] = vol
train_vols = (brain_window(hu_cache[s], d["window_level"], d["window_width"]) for s in lab.study_id if split.get(s) == "train")
mean, std = training_set_statistics(train_vols)
json.dump({"mean": mean, "std": std}, open("logs/normalisation_constants.json", "w"), indent=2)
print(f"training-set mean={mean:.4f} std={std:.4f}  (config holds {d['normalisation']['mean']}, {d['normalisation']['std']})")
cfg["data"]["normalisation"] = {"mean": mean, "std": std}
for sid, vol in hu_cache.items():
    v, mask = preprocess_volume(vol, cfg)
    np.save(out / f"{sid}.npy", v.astype(np.float16))
print(f"preprocessed {len(hu_cache)} studies -> {out}")
