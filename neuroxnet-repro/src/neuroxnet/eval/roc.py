import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score

def empirical_roc(y, score):
    """Exact step-function ROC (no smoothing) and trapezoidal AUC. Steps are multiples of 1/Pos and 1/Neg by construction."""
    fpr, tpr, thr = roc_curve(y, score, drop_intermediate=False)
    return fpr, tpr, thr, roc_auc_score(y, score)

def bootstrap_auc_ci(y, score, iters=2000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed); y, score = np.asarray(y), np.asarray(score); vals = []
    for _ in range(iters):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        vals.append(roc_auc_score(y[idx], score[idx]))
    return float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))
