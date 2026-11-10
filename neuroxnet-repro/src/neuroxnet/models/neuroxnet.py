import torch, torch.nn as nn
from .backbone import Backbone
from .msam import MSAM
from .csag import CSAG
from .fpn import FPN
from .heads import DetectionHead, SeverityHead

class NeuroXNet(nn.Module):
    """Slice-level network. forward(x[B,1,H,W]) -> (det_logit[B], sev_logits[B,4] or None, F_agg).
    Flags reproduce the ablation rows of Table 12. With all flags False the model is the backbone with the two heads
    (the 'Backbone Only' row); the 'EfficientNetB4' baseline of Tables 5-7 is the same configuration built by models/baselines.py."""
    def __init__(self, backbone="efficientnet_b4", pretrained=True, use_msam=True, use_csag=True, use_fpn=True,
                 severity_head=True, msam_kernels=(1, 3, 5, 7), csag_reduction=16, fpn_channels=128, det_channels=128):
        super().__init__()
        self.backbone = Backbone(backbone, pretrained)
        ch = self.backbone.channels                  # [C1..C5]
        self.use_msam, self.use_csag, self.use_fpn = use_msam, use_csag, use_fpn
        self.msam = nn.ModuleList([MSAM(c, msam_kernels) for c in ch[1:]]) if use_msam else None   # four levels F2..F5
        self.csag = nn.ModuleList([CSAG(c, csag_reduction) for c in ch[1:]]) if use_csag else None
        self.fpn = FPN(ch[1:], fpn_channels) if use_fpn else None
        agg_ch = fpn_channels if use_fpn else ch[-1]
        self.det_head = DetectionHead(agg_ch, det_channels)
        self.sev_head = SeverityHead(agg_ch, 4) if severity_head else None
    def forward(self, x):
        feats = self.backbone(x)[1:]                 # F2..F5
        if self.use_msam:
            feats = [m(f) for m, f in zip(self.msam, feats)]
        if self.use_csag:
            feats = [c(f) for c, f in zip(self.csag, feats)]
        fagg = self.fpn(feats) if self.use_fpn else feats[-1]
        return self.det_head(fagg), (self.sev_head(fagg) if self.sev_head is not None else None), fagg

def count_parameters(model, feature_extractor_only=False):
    """Parameter count used for Table 12 / Table 14 and Figure 1. Counts trainable parameters."""
    mods = [model.backbone] + [m for m in (model.msam, model.csag, model.fpn) if m is not None] if feature_extractor_only else [model]
    return sum(p.numel() for m in mods for p in m.parameters() if p.requires_grad)

def build_model(cfg):
    m = dict(cfg["model"]); m.pop("name", None)
    return NeuroXNet(**m)
