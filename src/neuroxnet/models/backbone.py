import timm, torch.nn as nn

class Backbone(nn.Module):
    """timm feature extractor returning the five stage outputs {F1..F5} (Eq. 5). Classifier head excluded."""
    def __init__(self, name="efficientnet_b4", pretrained=True, in_chans=1):
        super().__init__()
        self.net = timm.create_model(name, pretrained=pretrained, features_only=True, in_chans=in_chans)
        self.channels = list(self.net.feature_info.channels())[-5:]
    def forward(self, x):
        return self.net(x)[-5:]
