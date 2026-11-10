import torch.nn as nn

class DetectionHead(nn.Module):
    """Eq. (13): 1x1 conv, GAP, linear -> one logit (sigmoid applied after study-level averaging)."""
    def __init__(self, in_channels, det_channels=128):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, det_channels, 1)
        self.fc = nn.Linear(det_channels, 1)
    def forward(self, f):
        return self.fc(self.reduce(f).mean(dim=(2, 3))).squeeze(1)

class SeverityHead(nn.Module):
    """Eq. (15)-(16): two 3x3 conv-BN-ReLU blocks, GAP, linear -> four logits."""
    def __init__(self, in_channels, n_classes=4):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True))
        self.fc = nn.Linear(in_channels, n_classes)
    def forward(self, f):
        return self.fc(self.block(f).mean(dim=(2, 3)))
