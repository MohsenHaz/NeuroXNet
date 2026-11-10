import torch, torch.nn as nn

class MSAM(nn.Module):
    """Multi-Scale Attention Module, Eq. (6)-(7). Each scale k in {1,3,5,7} is a k x k depthwise convolution followed by a
    shared 1 x 1 point-wise projection (a separable k x k convolution, so the module stays within the ~2 M parameter budget
    reported for the attention and pyramid modules). Scale weights alpha_k are the softmax of a linear projection of GAP(F)."""
    def __init__(self, channels, kernels=(1, 3, 5, 7)):
        super().__init__()
        self.dw = nn.ModuleList([nn.Conv2d(channels, channels, k, padding=k // 2, groups=channels, bias=False) for k in kernels])
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.scale_proj = nn.Linear(channels, len(kernels))
        self.bn = nn.BatchNorm2d(channels)
    def forward(self, f):
        alpha = torch.softmax(self.scale_proj(f.mean(dim=(2, 3))), dim=1)                    # Eq. (7): B,K
        mixed = sum(alpha[:, k, None, None, None] * conv(f) for k, conv in enumerate(self.dw))   # Eq. (6)
        return torch.relu(self.bn(self.pw(mixed))) + f
