"""Builds every manuscript table from predictions/**/*.csv, then runs the consistency gate.
Tables are written as CSV to results/tables (one file per table) and as a single results/tables/ALL_TABLES.md."""
import sys, pathlib, json, numpy as np, pandas as pd
sys.path.insert(0, "src")
from neuroxnet.eval.metrics import binary_metrics, multiclass_metrics, brier_binary
from neuroxnet.eval.wilson import fmt_pct_ci
from neuroxnet.eval.roc import bootstrap_auc_ci
from neuroxnet.eval.mcnemar import mcnemar_exact, paired_bootstrap_auc_diff
from neuroxnet.eval import consistency

P = pathlib.Path("predictions"); T = pathlib.Path("results/tables"); T.mkdir(parents=True, exist_ok=True)
NAMES = {"resnet50": "ResNet50", "vgg16": "VGG16", "densenet121": "DenseNet121", "effb0": "EfficientNetB0", "effb4": "EfficientNetB4",
         "transunet": "TransUNet", "attunet": "Attention U-Net", "neuroxnet": "NeuroXNet (Ours)"}
ABL = {"backbone_only": "Backbone Only", "plus_msam": "+ MSAM", "plus_csag": "+ CSAG", "plus_fpn": "+ FPN", "plus_label_smoothing": "+ Label Smoothing",
       "det_only": "Detection Only (No Severity Head)", "focal": "Focal Loss", "cb": "Class-Balanced Loss", "full_model": "Full Model (Joint, Ours)"}
md = []

def load(p):
    path = pathlib.Path(p)
    if not path.exists(): return None
    df = pd.read_csv(path)
    return df if not df.empty else None

def P4(df): return df[["p0", "p1", "p2", "p3"]].values

def binary_row(name, df, with_auc_ci=True):
    m = binary_metrics(df.y_bin, df.pred_bin, df.p_hydro)
    lo, hi = bootstrap_auc_ci(df.y_bin, df.p_hydro) if with_auc_ci else (np.nan, np.nan)
    return dict(Method=name, TP=m["TP"], FP=m["FP"], FN=m["FN"], TN=m["TN"], acc_pct=round(100 * m["acc"], 2), acc_ci=fmt_pct_ci(m["TP"] + m["TN"], m["N"]),
                sens_pct=round(100 * m["sens"], 2), sens_ci=fmt_pct_ci(m["TP"], m["Pos"]), spec_pct=round(100 * m["spec"], 2), spec_ci=fmt_pct_ci(m["TN"], m["Neg"]),
                f1=round(m["f1"], 4), auc=round(m["auc"], 4), auc_ci=f"[{lo:.4f}–{hi:.4f}]", brier=round(brier_binary(df.y_bin, df.p_hydro), 3))

def sev_row(name, df):
    m = multiclass_metrics(df.y_sev, df.pred_sev, P4(df))
    return dict(Method=name, acc_ci=fmt_pct_ci(int(np.trace(m["cm"])), m["N"]), macro_f1=round(m["macro_f1"], 4), weighted_f1=round(m["weighted_f1"], 4),
                kappa=round(m["kappa"], 4), weighted_kappa=round(m["weighted_kappa"], 4), brier=round(m["brier"], 3), N=m["N"])

def perclass_table(m, with_auc=True):
    rows = []
    for r in m["per_class"]:
        rows.append(dict(cls=["Normal", "Mild", "Moderate", "Severe"][r["cls"]], N=r["N"], prec=round(r["prec"], 4), prec_ci=f"[{100*r['prec_ci'][0]:.1f}–{100*r['prec_ci'][1]:.1f}]",
                         rec=round(r["rec"], 4), rec_ci=f"[{100*r['rec_ci'][0]:.1f}–{100*r['rec_ci'][1]:.1f}]", f1=round(r["f1"], 4), spec=round(r["spec"], 4),
                         auc=round(r.get("auc", np.nan), 4) if with_auc else np.nan))
    rows.append(dict(cls="Macro Avg", N="", prec=round(m["macro_prec"], 4), prec_ci="", rec=round(m["macro_rec"], 4), rec_ci="", f1=round(m["macro_f1"], 4),
                     spec=round(m["macro_spec"], 4), auc=round(m.get("macro_auc", np.nan), 4) if with_auc else np.nan))
    return pd.DataFrame(rows)

def cm_table(m):
    cls = ["Normal", "Mild", "Moderate", "Severe"]
    df = pd.DataFrame(m["cm"], columns=[f"Pred {c}" for c in cls]); df.insert(0, "Actual", [f"Actual {c}" for c in cls]); df["Total"] = m["cm"].sum(1)
    tot = dict(Actual="Total", **{f"Pred {c}": int(v) for c, v in zip(cls, m["cm"].sum(0))}, Total=int(m["cm"].sum()))
    return pd.concat([df, pd.DataFrame([tot])], ignore_index=True)

def save(df, name, title):
    df.to_csv(T / name, index=False); md.append(f"## {title}\n\n" + df.to_markdown(index=False) + "\n")

# ---- internal test: Tables 5, 5b, 6, 7, confusion matrix ----
internal = {k: load(P / "internal_test" / f"{k}.csv") for k in NAMES}
present = {k: v for k, v in internal.items() if v is not None}
if present:
    save(pd.DataFrame([binary_row(NAMES[k], v) for k, v in present.items()]), "table5_binary.csv", "Table 5: binary detection, held-out CQ500 test set")
    save(pd.DataFrame([sev_row(NAMES[k], v) for k, v in present.items() if v.pred_sev.min() >= 0]), "table6_severity.csv", "Table 6: four-class severity, held-out CQ500 test set")
    nx = present.get("neuroxnet")
    if nx is not None:
        m = multiclass_metrics(nx.y_sev, nx.pred_sev, P4(nx))
        save(cm_table(m), "table4_internal_cm.csv", "Figure 4 data: internal four-class confusion matrix")
        save(perclass_table(m), "table7_perclass.csv", "Table 7: per-class metrics, internal test set")
        rows = []
        for k, v in present.items():
            if k == "neuroxnet": continue
            v = v.set_index("study_id").loc[nx.study_id].reset_index()
            mc = mcnemar_exact(nx.y_bin, nx.pred_bin, v.pred_bin); bd = paired_bootstrap_auc_diff(nx.y_bin, nx.p_hydro, v.p_hydro)
            rows.append(dict(Comparison=f"NeuroXNet vs {NAMES[k]}", **mc, **bd))
        save(pd.DataFrame(rows), "table5b_paired_tests.csv", "Paired comparisons: exact McNemar on binary decisions and paired bootstrap AUC difference (2000 resamples)")

# ---- external: Tables 8-11 ----
ext = load("data/external/external_predictions_deidentified.csv")
if ext is None: ext = load(P / "external" / "neuroxnet.csv")
if ext is not None:
    b = binary_row("NeuroXNet", ext); m = multiclass_metrics(ext.y_sev, ext.pred_sev, P4(ext))
    save(pd.DataFrame([dict(Task="Binary Detection", acc_ci=b["acc_ci"], sens_ci=b["sens_ci"], spec_ci=b["spec_ci"], auc=b["auc"], auc_ci=b["auc_ci"], macro_f1=""),
                       dict(Task="Severity Classification", acc_ci=fmt_pct_ci(int(np.trace(m["cm"])), m["N"]), sens_ci="", spec_ci="", auc="", auc_ci="", macro_f1=round(m["macro_f1"], 4))]),
         "table8_external.csv", "Table 8: external cohort aggregate results")
    save(cm_table(m), "table9_external_cm.csv", "Table 9: external four-class confusion matrix")
    save(perclass_table(m, with_auc=False), "table10_external_perclass.csv", "Table 10: external per-class metrics")
    rows = []
    for col, label in (("vendor", "{} scanner"), ("slice_thickness_group", "Slice {}"), ("age_group", "Age {}")):
        if col in ext.columns:
            for g, sub in ext.groupby(col, sort=True):
                r = binary_row(label.format(g), sub, with_auc_ci=False); r["N"] = len(sub); rows.append(r)
    if rows:
        save(pd.DataFrame(rows), "table11_subgroups.csv", "Table 11: external subgroup binary detection")

# ---- ablation: Table 12 (same split as Tables 5-7) ----
rows = []
for k, name in ABL.items():
    df = load(P / "ablation" / f"{k}.csv")
    if df is None: continue
    bm = binary_metrics(df.y_bin, df.pred_bin, df.p_hydro); r = dict(Configuration=name, det_acc=f"{100*bm['acc']:.2f} ({bm['TP']+bm['TN']}/{bm['N']})", det_auc=round(bm["auc"], 4))
    if df.pred_sev.min() >= 0:
        m = multiclass_metrics(df.y_sev, df.pred_sev, P4(df)); f = m["per_class"]
        r.update(sev_acc=f"{100*m['acc']:.2f} ({int(np.trace(m['cm']))}/{m['N']})", macro_f1=round(m["macro_f1"], 4), f1_mod_sev=round((f[2]["f1"] + f[3]["f1"]) / 2, 4), brier=round(m["brier"], 3))
    else:
        r.update(sev_acc="–", macro_f1="–", f1_mod_sev="–", brier=round(brier_binary(df.y_bin, df.p_hydro), 3))
    rows.append(r)
if rows:
    pc = load("results/tables/param_counts.csv")
    df = pd.DataFrame(rows)
    if pc is not None: df = df.merge(pc, on="Configuration", how="left")
    save(df, "table12_ablation.csv", "Table 12: ablation on the held-out test set (identical split to Tables 5-7)")

# ---- CV: Table 15 ----
rows = []
for k in range(1, 6):
    df = load(P / "cv5" / f"fold{k}_neuroxnet.csv")
    if df is None: continue
    b = binary_row(f"Fold {k}", df, with_auc_ci=False); m = multiclass_metrics(df.y_sev, df.pred_sev, P4(df))
    rows.append(dict(Fold=f"Fold {k}", N=b["TP"] + b["FP"] + b["FN"] + b["TN"], Pos=b["TP"] + b["FN"], Neg=b["TN"] + b["FP"], TP=b["TP"], FP=b["FP"], FN=b["FN"], TN=b["TN"],
                     acc_pct=b["acc_pct"], sev_acc_pct=round(100 * m["acc"], 2), sev_correct=int(np.trace(m["cm"])), sens_pct=b["sens_pct"], spec_pct=b["spec_pct"], f1=b["f1"], auc=b["auc"], macro_f1=round(m["macro_f1"], 4)))
if rows:
    df = pd.DataFrame(rows); num = ["acc_pct", "sev_acc_pct", "sens_pct", "spec_pct", "f1", "auc", "macro_f1"]
    mean = {c: round(df[c].mean(), 4) for c in num}; std = {c: round(df[c].std(ddof=1), 4) for c in num}
    df = pd.concat([df, pd.DataFrame([dict(Fold="Mean", **mean), dict(Fold="Std (sample)", **std)])], ignore_index=True)
    save(df, "table15_cv.csv", "Table 15: five-fold cross-validation (per-fold counts, accuracies as integer ratios, per-fold AUC)")

open(T / "ALL_TABLES.md", "w").write("\n".join(md))
ok = consistency.run()
print("tables written to", T, "| consistency:", "PASS" if ok else "FAIL (see results/checks/consistency_report.txt)")
sys.exit(0 if ok else 1)
