import torch

def aggregate_study(det_logits, sev_logits=None):
    """Study-level aggregation (Section 3.3 / Inference Protocol): arithmetic mean of the slice-level logits over the
    128 selected slices (padded slices included), then sigmoid / softmax. det_logits [S], sev_logits [S,4] or None."""
    p_h = torch.sigmoid(det_logits.mean(0))
    p_s = torch.softmax(sev_logits.mean(0), dim=0) if sev_logits is not None else None
    return p_h, p_s

def aggregate_logits(det_logits, sev_logits=None):
    """Same as aggregate_study but returns the mean logits (used by the loss, which is computed on study-level logits)."""
    return det_logits.mean(0), (sev_logits.mean(0) if sev_logits is not None else None)
