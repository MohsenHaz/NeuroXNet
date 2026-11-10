"""Grad-CAM, Eq. (24)-(25), on the last convolutional map of the network (F_agg for NeuroXNet)."""
import torch, torch.nn.functional as F

class GradCAM:
    def __init__(self, model):
        self.model = model
    def __call__(self, x, target="severity", class_idx=None):
        """x: [S,1,H,W] slices of one study. Returns heat maps [S,H,W] normalised to [0,1] per study.
        target 'severity' uses the predicted (or given) severity class score; 'detection' uses the detection logit."""
        self.model.eval()
        x = x.requires_grad_(True)
        det, sev, fagg = self.model(x)
        fagg.retain_grad()
        if target == "severity" and sev is not None:
            c = int(sev.mean(0).argmax()) if class_idx is None else class_idx
            score = sev[:, c].sum()
        else:
            score = det.sum()
        self.model.zero_grad(set_to_none=True); score.backward()
        alpha = fagg.grad.mean(dim=(2, 3), keepdim=True)                     # Eq. (25)
        cam = F.relu((alpha * fagg).sum(1))                                   # Eq. (24)
        cam = F.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = cam - cam.amin(); cam = cam / (cam.amax() + 1e-8)
        return cam.detach()
