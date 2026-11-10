import numpy as np

def net_benefit(y, p, thresholds):
    """Net benefit NB(w) = TP/N - FP/N * w/(1-w) with predictions positive when p >= w. Treat-all: prevalence - (1-prevalence) w/(1-w)
    (zero at w = prevalence); treat-none: 0. Everything follows from the stored scores, so the curve and its caption cannot disagree."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float); n = len(y); prev = y.mean()
    model, all_ = [], []
    for w in thresholds:
        pred = p >= w; tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        odds = w / (1 - w)
        model.append(tp / n - fp / n * odds); all_.append(prev - (1 - prev) * odds)
    return np.array(model), np.array(all_), np.zeros(len(thresholds))
