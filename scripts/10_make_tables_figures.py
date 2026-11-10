"""Figures regenerated from predictions/**/*.csv and results/tables/*.csv. IEEE two-column sizing, Times New Roman, 800 dpi,
legends outside the axes below the x-axis label, written (a)/(b) panel labels under each panel."""
import sys, pathlib, numpy as np, pandas as pd
sys.path.insert(0, "src")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from neuroxnet.eval.roc import empirical_roc
from neuroxnet.eval.calibration import reliability_bins, reliability_multiclass
from neuroxnet.eval.decision_curve import net_benefit
from neuroxnet.eval.metrics import multiclass_metrics, brier_binary, brier_multiclass
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7})
P = pathlib.Path("predictions"); F = pathlib.Path("results/figures"); F.mkdir(parents=True, exist_ok=True)
NAMES = {"neuroxnet": "NeuroXNet", "effb4": "EfficientNetB4", "densenet121": "DenseNet121", "resnet50": "ResNet50", "vgg16": "VGG16", "effb0": "EfficientNetB0", "transunet": "TransUNet", "attunet": "Attention U-Net"}
def load(p): return pd.read_csv(p) if pathlib.Path(p).exists() else None
def panel(ax, lab): ax.text(0.5, -0.30, lab, transform=ax.transAxes, ha="center", va="top", fontsize=8)
def legend_below(fig, ax, ncol):
    h, l = ax.get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, -0.02))
internal = {k: load(P / "internal_test" / f"{k}.csv") for k in NAMES}; internal = {k: v for k, v in internal.items() if v is not None}
ext = load("data/external/external_predictions_deidentified.csv")
if ext is None: ext = load(P / "external" / "neuroxnet.csv")

# Figure 3: ROC, (a) internal all models, (b) external NeuroXNet
if internal:
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 3.2))
    for k, df in internal.items():
        fpr, tpr, _, auc = empirical_roc(df.y_bin, df.p_hydro)
        ax[0].step(fpr, tpr, where="post", lw=1.4 if k == "neuroxnet" else 0.9, label=f"{NAMES[k]} (AUC = {auc:.4f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.6)
    if ext is not None:
        fpr, tpr, _, auc = empirical_roc(ext.y_bin, ext.p_hydro); ax[1].step(fpr, tpr, where="post", lw=1.4, color="C0", label=f"NeuroXNet external (AUC = {auc:.4f})"); ax[1].plot([0, 1], [0, 1], "k--", lw=0.6)
    for a, lab in zip(ax, ["(a) Internal CQ500 test set (N = 73)", "(b) External cohort (N = 182)"]):
        a.set_xlabel("False-positive rate"); a.set_ylabel("Sensitivity"); a.set_xlim(-0.01, 1.01); a.set_ylim(-0.01, 1.01); panel(a, lab)
    h0, l0 = ax[0].get_legend_handles_labels(); h1, l1 = ax[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(F / "fig3_roc.png", dpi=800, bbox_inches="tight"); plt.close(fig)

nx = internal.get("neuroxnet")
if nx is not None:
    P4 = nx[["p0", "p1", "p2", "p3"]].values
    # Figure 4: confusion matrix
    m = multiclass_metrics(nx.y_sev, nx.pred_sev, P4); cm = m["cm"]; cls = ["Normal", "Mild", "Moderate", "Severe"]
    fig, ax = plt.subplots(figsize=(3.5, 3.1)); ax.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4): ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(4)); ax.set_yticks(range(4)); ax.set_xticklabels(cls, rotation=30, ha="right"); ax.set_yticklabels(cls); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(F / "fig4_confusion_matrix.png", dpi=800); plt.close(fig)
    # Figure 5: reliability
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 3.2))
    for a, rows, lab, bs in ((ax[0], reliability_bins(nx.y_bin, nx.p_hydro), "(a) Binary detection", brier_binary(nx.y_bin, nx.p_hydro)),
                             (ax[1], reliability_multiclass(nx.y_sev, P4), "(b) Four-class severity (one-vs-rest, pooled)", brier_multiclass(nx.y_sev, P4))):
        r = pd.DataFrame(rows).dropna(); a.plot([0, 1], [0, 1], "k--", lw=0.6, label="Perfect calibration")
        a.plot(r.mean_pred, r.obs_freq, "o-", ms=3, lw=1, label=f"NeuroXNet (Brier = {bs:.3f})")
        for _, b in r.iterrows(): a.annotate(f"n={int(b.n)}", (b.mean_pred, b.obs_freq), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=6)
        a.set_xlabel("Mean predicted probability"); a.set_ylabel("Observed frequency"); a.set_xlim(-0.02, 1.02); a.set_ylim(-0.02, 1.10); panel(a, lab)
    pd.concat([pd.DataFrame(reliability_bins(nx.y_bin, nx.p_hydro)).assign(task="binary"), pd.DataFrame(reliability_multiclass(nx.y_sev, P4)).assign(task="severity")]).to_csv("results/tables/tableS1_reliability_bins.csv", index=False)
    legend_below(fig, ax[0], 2); fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(F / "fig5_reliability.png", dpi=800, bbox_inches="tight"); plt.close(fig)
    # Figure 6: decision curve
    w = np.linspace(0.01, 0.99, 197); fig, ax = plt.subplots(figsize=(3.5, 3.0))
    for k, df in internal.items():
        nb, nb_all, nb_none = net_benefit(df.y_bin, df.p_hydro, w); ax.plot(w, nb, lw=1.4 if k == "neuroxnet" else 0.8, label=NAMES[k])
    ax.plot(w, nb_all, "k-", lw=0.8, label="Treat all"); ax.plot(w, nb_none, "k:", lw=0.8, label="Treat none")
    ax.set_ylim(-0.05, max(0.45, nx.y_bin.mean() + 0.05)); ax.set_xlabel("Threshold probability w"); ax.set_ylabel("Net benefit")
    legend_below(fig, ax, 3); fig.tight_layout(rect=(0, 0.22, 1, 1)); fig.savefig(F / "fig6_decision_curve.png", dpi=800, bbox_inches="tight"); plt.close(fig)

# Figure 7: ablation from Table 12 CSV
t12 = load("results/tables/table12_ablation.csv")
if t12 is not None:
    t = t12[t12.Configuration != "Full Model (Joint, Ours)"].copy()
    t["det"] = t.det_acc.str.split(" ").str[0].astype(float); t["sev"] = pd.to_numeric(t.sev_acc.str.split(" ").str[0], errors="coerce")
    fig, ax = plt.subplots(1, 2, figsize=(7.16, 3.3)); x = np.arange(len(t)); wd = 0.38
    ax[0].bar(x - wd / 2, t.det, wd, label="Detection accuracy"); ax[0].bar(x + wd / 2, t.sev.fillna(0), wd, label="Severity accuracy")
    ax[0].set_ylim(min(t.det.min(), t.sev.min()) - 3, 100); ax[0].set_ylabel("Accuracy (%)")
    ax[1].plot(x, pd.to_numeric(t.macro_f1, errors="coerce"), "o-", ms=3, label="Macro-F1"); ax[1].plot(x, pd.to_numeric(t.brier, errors="coerce"), "s--", ms=3, label="Brier score (severity)")
    ax[1].set_ylabel("Macro-F1 / Brier")
    for a, lab in zip(ax, ["(a) Detection and severity accuracy", "(b) Macro-F1 and Brier score"]):
        a.set_xticks(x); a.set_xticklabels(t.Configuration, rotation=35, ha="right", fontsize=6); panel(a, lab)
    h0, l0 = ax[0].get_legend_handles_labels(); h1, l1 = ax[1].get_legend_handles_labels()
    fig.legend(h0 + h1, l0 + l1, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.10, 1, 1)); fig.savefig(F / "fig7_ablation.png", dpi=800, bbox_inches="tight"); plt.close(fig)

# CV training dynamics figure from checkpoints/cv5/fold*_neuroxnet_history.csv
hists = [load(f"checkpoints/cv5/fold{k}_neuroxnet_history.csv") for k in range(1, 6)]
if any(h is not None for h in hists):
    fig, ax = plt.subplots(1, 3, figsize=(7.16, 2.6))
    for k, h in enumerate(hists, 1):
        if h is None: continue
        ax[0].plot(h.epoch, h.train_loss, lw=0.8, label=f"Fold {k} train"); ax[0].plot(h.epoch, h.val_loss, lw=0.8, ls="--", label=f"Fold {k} val")
        ax[1].plot(h.epoch, 100 * h.val_auroc, lw=0.8, label=f"Fold {k}"); ax[2].plot(h.epoch, 100 * h.val_macro_f1, lw=0.8, label=f"Fold {k}")
    for a, yl, lab in zip(ax, ["Loss", "Validation AUROC (%)", "Validation macro-F1 (%)"], ["(a) Loss", "(b) Detection", "(c) Severity"]):
        a.set_xlabel("Epoch"); a.set_ylabel(yl); panel(a, lab)
    legend_below(fig, ax[1], 5); fig.tight_layout(rect=(0, 0.10, 1, 1)); fig.savefig(F / "fig9_training_dynamics.png", dpi=800, bbox_inches="tight"); plt.close(fig)
print("figures written to", F)
