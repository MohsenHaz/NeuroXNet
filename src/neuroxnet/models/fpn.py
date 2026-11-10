import torch, torch.nn as nn, torch.nn.functional as F

class FPN(nn.Module):
    """Feature Pyramid Network, Eq. (11)-(12): top-down pathway over F2..F5, outputs F_agg at the resolution of F2."""
    def __init__(self, in_channels, out_channels=128):
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(c, out_channels, 1) for c in in_channels])     # F2..F5
        self.smooth = nn.ModuleList([nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels])
        self.agg = nn.Conv2d(out_channels * len(in_channels), out_channels, 3, padding=1)
    def forward(self, feats):                       # feats = [F2,F3,F4,F5]
        p = self.lateral[-1](feats[-1])
        outs = [p]
        for i in range(len(feats) - 2, -1, -1):
            p = self.lateral[i](feats[i]) + F.interpolate(p, size=feats[i].shape[-2:], mode="nearest")
            outs.insert(0, self.smooth[i](p))       # Eq. (11)
        h, w = feats[0].shape[-2:]
        outs = [F.interpolate(o, size=(h, w), mode="bilinear", align_corners=False) for o in outs]
        return self.agg(torch.cat(outs, 1))         # Eq. (12)
