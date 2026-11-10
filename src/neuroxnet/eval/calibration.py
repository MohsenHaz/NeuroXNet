import numpy as np

def reliability_bins(y, p, n_bins=10):
    """Binned reliability data for Figure 5 and supplementary Table S1: mean predicted probability, observed frequency and count per bin."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1); rows = []
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        rows.append(dict(bin_lo=edges[i], bin_hi=edges[i + 1], n=int(m.sum()), mean_pred=float(p[m].mean()) if m.any() else np.nan, obs_freq=float(y[m].mean()) if m.any() else np.nan))
    return rows

def reliability_multiclass(y, P, n_bins=10):
    """One-vs-rest reliability pooled over the four classes (each study contributes four (p_c, 1[y=c]) pairs)."""
    y = np.asarray(y).astype(int); P = np.asarray(P)
    pp = P.ravel(); yy = (np.eye(P.shape[1])[y]).ravel()
    return reliability_bins(yy, pp, n_bins)
