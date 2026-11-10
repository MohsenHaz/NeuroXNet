"""Run at the external institution: preprocess de-identified DICOM studies, predict with the frozen checkpoint and write the
de-identified prediction CSV (no images, no dates, no institution). Expects a CSV with study_id,series_dir,y_sev,vendor,slice_thickness_mm,age."""
import argparse, sys, pathlib, numpy as np, pandas as pd, torch
sys.path.insert(0, "src")
from neuroxnet.data.dicom_io import read_series; from neuroxnet.data.preprocess import preprocess_volume
from neuroxnet.inference.predict import load_model; from neuroxnet.training.train import forward_study
ap = argparse.ArgumentParser(); ap.add_argument("--index", required=True); ap.add_argument("--ckpt", default="checkpoints/neuroxnet_final.pt")
ap.add_argument("--out", default="data/external/external_predictions_deidentified.csv"); a = ap.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model, cfg, seed = load_model(a.ckpt, device)
idx = pd.read_csv(a.index); rows = []
for _, r in idx.iterrows():
    vol, thick, _ = read_series(r.series_dir); x, _ = preprocess_volume(vol, cfg); x = torch.from_numpy(x).to(device)
    with torch.no_grad():
        dl, sl, _ = forward_study(model, x, cfg["train"]["slice_chunk"])
    ph = float(torch.sigmoid(dl.float())); ps = torch.softmax(sl.float(), 0).cpu().numpy()
    rows.append(dict(study_id=f"EXT{len(rows)+1:04d}", split="external", y_bin=int(r.y_sev > 0), y_sev=int(r.y_sev), p0=ps[0], p1=ps[1], p2=ps[2], p3=ps[3], p_hydro=ph,
                     pred_bin=int(ph >= 0.5), pred_sev=int(ps.argmax()), seed=seed, git_hash="external", vendor=r.vendor,
                     slice_thickness_group="<= 2 mm" if r.slice_thickness_mm <= 2 else "> 2 mm", age_group="< 50 years" if r.age < 50 else ">= 50 years"))
pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv(a.out, index=False); print("wrote", a.out, len(rows), "studies")
