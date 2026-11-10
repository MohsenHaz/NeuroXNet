"""Grad-CAM localisation on the 100-study annotated subset (Table 13) and real Grad-CAM panels for Figures 7-8 and 10.
Needs data/labels/gradcam_roi_subset_ids.csv and data/roi_masks/<study_id>.npz (consensus ventricular masks)."""
import sys, pathlib, numpy as np, pandas as pd, torch
sys.path.insert(0, "src")
from neuroxnet.inference.predict import load_model
from neuroxnet.explain.gradcam import GradCAM
from neuroxnet.explain.roi_overlap import overlap_counts, metrics_from_counts, summarise
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, cfg, _ = load_model("checkpoints/neuroxnet_final.pt", device); cam = GradCAM(model)
subset = pd.read_csv("data/labels/gradcam_roi_subset_ids.csv")
root = pathlib.Path(cfg["data"]["root"]); rows = []; examples = {}
for _, r in subset.iterrows():
    x = torch.from_numpy(np.load(root / f"{r.study_id}.npy").astype(np.float32)).to(device)
    mask = np.load(f"data/roi_masks/{r.study_id}.npz")["mask"]
    heat = cam(x[:, None]).cpu().numpy()                                   # S,H,W
    # evaluate on slices that carry annotation (mask non-empty) so that padded or extracranial slices do not dilute the overlap
    sl = [i for i in range(mask.shape[0]) if mask[i].any()] or [mask.shape[0] // 2]
    inter = union = a = g = 0
    for i in sl:
        c = overlap_counts(heat[i], mask[i], 0.5); inter += c[0]; union += c[1]; a += c[2]; g += c[3]
    rows.append(dict(study_id=r.study_id, cohort=r.cohort, cls=int(r["class"]), **metrics_from_counts(inter, union, a, g)))
    if int(r["class"]) not in examples:
        i = max(sl, key=lambda j: mask[j].sum()); examples[int(r["class"])] = (x[i].cpu().numpy(), heat[i], mask[i], r.study_id)
pathlib.Path("results/tables").mkdir(parents=True, exist_ok=True)
pd.DataFrame(rows).to_csv("results/tables/table13_gradcam_per_study.csv", index=False)
tab = summarise(rows); tab["cls"] = tab["cls"].map({0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe", "overall": "Overall"})
tab.round(3).to_csv("results/tables/table13_gradcam.csv", index=False); print(tab.round(3))
# real Grad-CAM panels (Figure 7 attention maps per class; Figure 8 CT + overlay + ROI contour)
m, s = cfg["data"]["normalisation"]["mean"], cfg["data"]["normalisation"]["std"]
fig, ax = plt.subplots(3, 4, figsize=(7.16, 5.4), dpi=800)
for col, c in enumerate(sorted(examples)):
    ct, h, mk, sid = examples[c]; ct = np.clip(ct * s + m, 0, 1)
    ax[0, col].imshow(ct, cmap="gray"); ax[1, col].imshow(ct, cmap="gray"); ax[1, col].imshow(h, cmap="jet", alpha=0.45)
    ax[2, col].imshow(ct, cmap="gray"); ax[2, col].imshow(h >= 0.5, cmap="Reds", alpha=0.4); ax[2, col].contour(mk, levels=[0.5], colors="lime", linewidths=0.6)
    for rr in range(3): ax[rr, col].axis("off")
    ax[0, col].set_title(["(a) normal", "(b) mild", "(c) moderate", "(d) severe"][c], fontsize=8, y=-0.18)
pathlib.Path("results/figures").mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig("results/figures/fig8_gradcam_real_cases.png", dpi=800); print("wrote results/figures/fig8_gradcam_real_cases.png")
