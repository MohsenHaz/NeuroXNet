"""Parameter counts for Table 12 (feature extractor, ImageNet head excluded) from the built models, so the numbers cannot drift."""
import sys, pathlib, pandas as pd; sys.path.insert(0, "src")
from neuroxnet.config import load_config; from neuroxnet.models import build; from neuroxnet.models.neuroxnet import count_parameters
ABL = {"backbone_only": "Backbone Only", "plus_msam": "+ MSAM", "plus_csag": "+ CSAG", "plus_fpn": "+ FPN", "plus_label_smoothing": "+ Label Smoothing", "det_only": "Detection Only (No Severity Head)", "focal": "Focal Loss", "cb": "Class-Balanced Loss"}
rows = []
for k, name in ABL.items():
    cfg = load_config(f"configs/ablation/{k}.yaml"); cfg["model"]["pretrained"] = False; m = build(cfg)
    rows.append(dict(Configuration=name, params_M=round(count_parameters(m, True) / 1e6, 2), params_total_M=round(count_parameters(m) / 1e6, 2)))
cfg = load_config("configs/neuroxnet.yaml"); cfg["model"]["pretrained"] = False; m = build(cfg)
rows.append(dict(Configuration="Full Model (Joint, Ours)", params_M=round(count_parameters(m, True) / 1e6, 2), params_total_M=round(count_parameters(m) / 1e6, 2)))
pathlib.Path("results/tables").mkdir(parents=True, exist_ok=True); pd.DataFrame(rows).to_csv("results/tables/param_counts.csv", index=False); print(pd.DataFrame(rows))
