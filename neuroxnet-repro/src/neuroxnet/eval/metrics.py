"""Every metric that the manuscript reports, computed from integer counts (confusion matrices) or from stored scores.
Nothing here is ever typed by hand into a table."""
import numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score, confusion_matrix
from .wilson import wilson

def binary_counts(y, pred):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum()); fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum()); tn = int(((y == 0) & (pred == 0)).sum())
    return dict(TP=tp, FP=fp, FN=fn, TN=tn)

def binary_metrics_from_counts(c):
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    n, pos, neg = tp + fp + fn + tn, tp + fn, tn + fp
    out = dict(N=n, Pos=pos, Neg=neg, **c)
    out.update(acc=(tp + tn) / n, sens=tp / pos if pos else float("nan"), spec=tn / neg if neg else float("nan"),
               prec=tp / (tp + fp) if tp + fp else float("nan"), f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"))
    out["acc_ci"], out["sens_ci"], out["spec_ci"] = wilson(tp + tn, n), wilson(tp, pos), wilson(tn, neg)
    return out

def binary_metrics(y, pred, score=None):
    out = binary_metrics_from_counts(binary_counts(y, pred))
    out["auc"] = roc_auc_score(y, score) if score is not None and len(set(np.asarray(y).tolist())) > 1 else float("nan")
    return out

def brier_binary(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))

def brier_multiclass(y, P):
    """Eq. (31): mean over studies of the sum over classes of squared differences."""
    Y = np.eye(P.shape[1])[np.asarray(y).astype(int)]
    return float(np.mean(((np.asarray(P) - Y) ** 2).sum(1)))

def multiclass_metrics(y, pred, P=None, n_classes=4):
    """Accuracy, per-class precision/recall/F1/specificity (+Wilson CIs), macro/weighted F1, kappa, weighted kappa,
    one-vs-rest AUC per class (if P given), all from the confusion matrix cm[true, pred]."""
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    cm = confusion_matrix(y, pred, labels=list(range(n_classes)))
    n = int(cm.sum()); per = []
    for c in range(n_classes):
        tp = int(cm[c, c]); fn = int(cm[c].sum() - tp); fp = int(cm[:, c].sum() - tp); tn = n - tp - fn - fp
        prec = tp / (tp + fp) if tp + fp else 0.0; rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        row = dict(cls=c, N=int(cm[c].sum()), TP=tp, FP=fp, FN=fn, TN=tn, prec=prec, rec=rec, f1=f1,
                   spec=tn / (tn + fp) if tn + fp else 0.0, prec_ci=wilson(tp, tp + fp), rec_ci=wilson(tp, tp + fn))
        if P is not None:
            yc = (y == c).astype(int)
            row["auc"] = roc_auc_score(yc, np.asarray(P)[:, c]) if yc.min() != yc.max() else float("nan")
        per.append(row)
    support = cm.sum(1)
    out = dict(N=n, cm=cm, acc=float(np.trace(cm) / n), acc_ci=wilson(int(np.trace(cm)), n), per_class=per,
               macro_f1=float(np.mean([r["f1"] for r in per])), weighted_f1=float(np.sum([r["f1"] * s for r, s in zip(per, support)]) / n),
               kappa=float(cohen_kappa_score(y, pred, labels=list(range(n_classes)))),
               weighted_kappa=float(cohen_kappa_score(y, pred, labels=list(range(n_classes)), weights="linear")),
               macro_prec=float(np.mean([r["prec"] for r in per])), macro_rec=float(np.mean([r["rec"] for r in per])), macro_spec=float(np.mean([r["spec"] for r in per])))
    if P is not None:
        out["macro_auc"] = float(np.nanmean([r["auc"] for r in per])); out["brier"] = brier_multiclass(y, P)
    return out
