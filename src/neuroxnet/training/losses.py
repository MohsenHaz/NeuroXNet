"""Eq. (1), (14), (17), (18) plus the focal and class-balanced alternatives of the ablation."""
import torch, torch.nn.functional as F

def class_weights_inverse_frequency(n_pos, n_neg):
    n = n_pos + n_neg
    return n / (2.0 * n_pos), n / (2.0 * n_neg)              # w_p, w_n

def detection_loss(logit, y, w_p, w_n):
    """Weighted BCE, Eq. (14). logit, y: [B]."""
    w = torch.where(y > 0.5, torch.full_like(y, w_p), torch.full_like(y, w_n))
    return (w * F.binary_cross_entropy_with_logits(logit, y, reduction="none")).mean()

def severity_loss(logits, y, eps=0.1):
    """Label-smoothing cross-entropy, Eq. (17), target (1-eps)*onehot + eps/4."""
    return F.cross_entropy(logits, y, label_smoothing=eps)

def focal_detection_loss(logit, y, gamma=2.0, w_p=1.0, w_n=1.0):
    p = torch.sigmoid(logit)
    pt = torch.where(y > 0.5, p, 1 - p)
    w = torch.where(y > 0.5, torch.full_like(y, w_p), torch.full_like(y, w_n))
    return (w * (1 - pt) ** gamma * F.binary_cross_entropy_with_logits(logit, y, reduction="none")).mean()

def class_balanced_weights(counts, beta=0.999):
    eff = (1 - beta ** torch.as_tensor(counts, dtype=torch.float32)) / (1 - beta)
    w = 1.0 / eff
    return w / w.sum() * len(counts)

def regularisation(model, fagg, lambda_l2, lambda_orth):
    """Eq. (18): L2 on all trainable parameters plus feature orthogonality on the GAP-pooled F_agg of the batch
    (B x C matrix normalised per channel so that F^T F compares with the identity)."""
    l2 = sum((p ** 2).sum() for p in model.parameters() if p.requires_grad)
    z = fagg.mean(dim=(2, 3))                                       # B,C
    z = z / (z.norm(dim=0, keepdim=True) + 1e-8)
    gram = z.t() @ z
    orth = ((gram - torch.eye(gram.shape[0], device=gram.device)) ** 2).sum()
    return lambda_l2 * l2 + lambda_orth * orth

def total_loss(cfg, model, det_logit, sev_logits, fagg, y_bin, y_sev, w_p, w_n, sev_counts):
    L = cfg["loss"]
    if L["type"] == "focal":
        ld = focal_detection_loss(det_logit, y_bin, L.get("focal_gamma", 2.0), w_p, w_n)
    else:
        ld = detection_loss(det_logit, y_bin, w_p, w_n)
    ls = torch.tensor(0.0, device=det_logit.device)
    if sev_logits is not None:
        if L["type"] == "class_balanced":
            ls = F.cross_entropy(sev_logits, y_sev, weight=class_balanced_weights(sev_counts, L.get("cb_beta", 0.999)).to(det_logit.device), label_smoothing=L["label_smoothing"])
        else:
            ls = severity_loss(sev_logits, y_sev, L["label_smoothing"])
    lr = regularisation(model, fagg, L["lambda_l2"], L["lambda_orth"])
    return L["lambda_d"] * ld + L["lambda_s"] * ls + lr, {"det": ld.item(), "sev": float(ls), "reg": lr.item()}
