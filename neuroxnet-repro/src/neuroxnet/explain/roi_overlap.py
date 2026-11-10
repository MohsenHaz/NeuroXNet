"""Thresholded Grad-CAM (0.5) against consensus ventricular ROI masks: IoU, Dice, precision, recall per study, and
per-class means with N written to the output (Table 13). Dice is derived from the same intersection/union counts, so
Dice = 2 IoU / (1 + IoU) holds by construction."""
import numpy as np

def overlap_counts(cam, mask, threshold=0.5):
    a = cam >= threshold; g = mask.astype(bool)
    inter = int((a & g).sum()); union = int((a | g).sum())
    return inter, union, int(a.sum()), int(g.sum())

def metrics_from_counts(inter, union, a_sum, g_sum):
    iou = inter / union if union else 0.0
    return dict(iou=iou, dice=2 * iou / (1 + iou), precision=inter / a_sum if a_sum else 0.0, recall=inter / g_sum if g_sum else 0.0)

def summarise(rows):
    """rows: list of dict(study_id, cohort, cls, iou, dice, precision, recall). Returns per-class and overall (macro over classes) table."""
    import pandas as pd
    df = pd.DataFrame(rows)
    per = df.groupby("cls").agg(n_internal=("cohort", lambda c: int((c == "internal").sum())), n_external=("cohort", lambda c: int((c == "external").sum())),
                                n_total=("study_id", "count"), iou=("iou", "mean"), precision=("precision", "mean"), recall=("recall", "mean"), dice=("dice", "mean")).reset_index()
    overall = dict(cls="overall", n_internal=int(per.n_internal.sum()), n_external=int(per.n_external.sum()), n_total=int(per.n_total.sum()),
                   iou=per.iou.mean(), precision=per.precision.mean(), recall=per.recall.mean(), dice=per.dice.mean())
    return pd.concat([per, pd.DataFrame([overall])], ignore_index=True)
