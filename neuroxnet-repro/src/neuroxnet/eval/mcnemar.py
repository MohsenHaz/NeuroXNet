"""Paired comparison of NeuroXNet against each baseline on the same studies: exact McNemar test on the discordant
pairs of binary decisions, and a paired bootstrap confidence interval for the AUC difference."""
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score

def mcnemar_exact(y, pred_a, pred_b):
    y, a, b = (np.asarray(v).astype(int) for v in (y, pred_a, pred_b))
    ca, cb = a == y, b == y
    n01 = int((ca & ~cb).sum())        # A right, B wrong
    n10 = int((~ca & cb).sum())        # A wrong, B right
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    return dict(n_A_right_B_wrong=n01, n_A_wrong_B_right=n10, discordant=n01 + n10, p_value=float(p))

def paired_bootstrap_auc_diff(y, score_a, score_b, iters=2000, seed=42, alpha=0.05):
    rng = np.random.default_rng(seed); y, sa, sb = (np.asarray(v) for v in (y, score_a, score_b)); d = []
    for _ in range(iters):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        d.append(roc_auc_score(y[idx], sa[idx]) - roc_auc_score(y[idx], sb[idx]))
    point = roc_auc_score(y, sa) - roc_auc_score(y, sb)
    return dict(auc_diff=float(point), ci_lo=float(np.percentile(d, 100 * alpha / 2)), ci_hi=float(np.percentile(d, 100 * (1 - alpha / 2))))
