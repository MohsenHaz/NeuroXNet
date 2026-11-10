"""Baselines of Tables 5-7 and 14: ImageNet backbones from timm with the NeuroXNet task heads, plus
TransUNet-style and Attention U-Net-style encoders adapted to classification."""
import torch, torch.nn as nn, torch.nn.functional as F
from .backbone import Backbone
from .heads import DetectionHead, SeverityHead

class CNNBaseline(nn.Module):
    """ResNet50 / VGG16 / DenseNet121 / EfficientNet-B0 / EfficientNet-B4: last stage feature map -> heads."""
    def __init__(self, backbone, pretrained=True, severity_head=True, det_channels=128):
        super().__init__()
        self.backbone = Backbone(backbone, pretrained)
        c = self.backbone.channels[-1]
        self.det_head, self.sev_head = DetectionHead(c, det_channels), (SeverityHead(c) if severity_head else None)
    def forward(self, x):
        f = self.backbone(x)[-1]
        return self.det_head(f), (self.sev_head(f) if self.sev_head is not None else None), f

class AttentionGate(nn.Module):
    """Additive attention gate of Oktay et al. applied to encoder skip features."""
    def __init__(self, g_ch, x_ch, inter):
        super().__init__()
        self.wg, self.wx = nn.Conv2d(g_ch, inter, 1), nn.Conv2d(x_ch, inter, 1)
        self.psi = nn.Conv2d(inter, 1, 1)
    def forward(self, g, x):
        g = F.interpolate(self.wg(g), size=x.shape[-2:], mode="bilinear", align_corners=False)
        return x * torch.sigmoid(self.psi(F.relu(g + self.wx(x))))

class AttentionUNetClassifier(nn.Module):
    """Attention U-Net encoder (ResNet34 stages) with attention gates from the deepest stage onto F3 and F4, decoder
    reduced to one up-sampling fusion, followed by the two heads."""
    def __init__(self, pretrained=True, severity_head=True):
        super().__init__()
        self.backbone = Backbone("resnet34", pretrained)
        c = self.backbone.channels
        self.ag3, self.ag4 = AttentionGate(c[4], c[2], c[2] // 2), AttentionGate(c[4], c[3], c[3] // 2)
        self.fuse = nn.Sequential(nn.Conv2d(c[2] + c[3] + c[4], 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.det_head, self.sev_head = DetectionHead(256), (SeverityHead(256) if severity_head else None)
    def forward(self, x):
        f = self.backbone(x)
        g = f[4]
        a3, a4 = self.ag3(g, f[2]), self.ag4(g, f[3])
        size = f[3].shape[-2:]
        z = torch.cat([F.interpolate(a3, size=size, mode="bilinear", align_corners=False), a4,
                       F.interpolate(g, size=size, mode="bilinear", align_corners=False)], 1)
        z = self.fuse(z)
        return self.det_head(z), (self.sev_head(z) if self.sev_head is not None else None), z

class TransUNetClassifier(nn.Module):
    """TransUNet-style hybrid: ResNet50 CNN encoder, patch tokens from the deepest stage through a ViT bottleneck
    (12 layers, 768 dim, 12 heads), tokens reshaped back to a map, followed by the two heads."""
    def __init__(self, pretrained=True, severity_head=True, dim=768, depth=12, heads=12):
        super().__init__()
        self.backbone = Backbone("resnet50", pretrained)
        c = self.backbone.channels[-1]
        self.embed = nn.Conv2d(c, dim, 1)
        self.pos = nn.Parameter(torch.zeros(1, 64, dim))
        enc = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(dim)
        self.det_head, self.sev_head = DetectionHead(dim), (SeverityHead(dim) if severity_head else None)
    def forward(self, x):
        f = self.embed(self.backbone(x)[-1])                          # B,dim,8,8 for 256x256 input
        B, D, H, W = f.shape
        t = f.flatten(2).transpose(1, 2)
        pos = self.pos if self.pos.shape[1] == H * W else F.interpolate(self.pos.transpose(1, 2).reshape(1, D, 8, 8), size=(H, W), mode="bilinear").flatten(2).transpose(1, 2)
        t = self.norm(self.transformer(t + pos))
        z = t.transpose(1, 2).reshape(B, D, H, W)
        return self.det_head(z), (self.sev_head(z) if self.sev_head is not None else None), z

def build_baseline(cfg):
    m = cfg["model"]; bb = m["backbone"]
    if bb == "attunet":
        return AttentionUNetClassifier(m.get("pretrained", True), m.get("severity_head", True))
    if bb == "transunet":
        return TransUNetClassifier(m.get("pretrained", True), m.get("severity_head", True))
    return CNNBaseline(bb, m.get("pretrained", True), m.get("severity_head", True))
