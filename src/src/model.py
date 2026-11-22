
import torch
import torch.nn as nn
from torchvision import models


class MobileNetV3VideoClassifier(nn.Module):
    """
    MobileNetV3-Large video classifier
    with feature-level temporal average pooling.
    """

    def __init__(self, num_classes=101, pretrained=True):

        super().__init__()

        weights = (
            models.MobileNet_V3_Large_Weights.DEFAULT
            if pretrained else None
        )

        backbone = models.mobilenet_v3_large(
            weights=weights
        )

        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.classifier = backbone.classifier

        self.classifier[-1] = nn.Linear(
            self.classifier[-1].in_features,
            num_classes
        )

    def forward(self, x):

        batch, frames, channels, height, width = x.shape

        x = x.reshape(
            batch * frames,
            channels,
            height,
            width
        )

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = x.reshape(batch, frames, -1)
        x = x.mean(dim=1)

        return self.classifier(x)
