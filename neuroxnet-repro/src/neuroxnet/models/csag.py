import torch, torch.nn as nn

class CSAG(nn.Module):
    """Channel-Spatial Attention Gate, Eq. (8)-(10)."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(2 * channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.spatial = nn.Conv2d(2, 1, 7, padding=3, bias=False)
    def forward(self, f):
        gap, gmp = f.mean(dim=(2, 3)), f.amax(dim=(2, 3))
        ac = torch.sigmoid(self.fc2(torch.relu(self.fc1(torch.cat([gap, gmp], 1)))))   # Eq. (8)
        f1 = ac[:, :, None, None] * f
        s = torch.cat([f1.mean(1, keepdim=True), f1.amax(1, keepdim=True)], 1)
        as_ = torch.sigmoid(self.spatial(s))                                           # Eq. (9)
        return as_ * f1                                                                # Eq. (10)
