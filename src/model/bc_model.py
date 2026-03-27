"""
BehavioralCloningModel: ResNet18 backbone + FC Head for driving control.

Architecture:
  Input: (batch, 3, 224, 224) normalized RGB
  ResNet18 backbone (ImageNet pretrained) → 512-d feature
  FC Head: 512→256 (ReLU+Dropout0.5) → 128 (ReLU+Dropout0.3) → 2
  Output: steering (tanh, [-1,1]), throttle (sigmoid, [0,1])
"""

from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class BehavioralCloningModel(nn.Module):
    """
    End-to-end behavioral cloning model.

    Uses ResNet18 as feature extractor with a fully connected head
    that outputs steering and throttle control values.

    Args:
        pretrained: Whether to use ImageNet pretrained weights.
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()

        # ResNet18 backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)

        # Remove the final FC layer — keep conv layers + avgpool
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
            resnet.avgpool,
        )
        self._feature_dim = 512

        # FC Head
        self.fc_head = nn.Sequential(
            nn.Linear(self._feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            image: (batch, 3, 224, 224) normalized tensor.

        Returns:
            steering: (batch, 1) in [-1, 1] via tanh.
            throttle: (batch, 1) in [0, 1] via sigmoid.
        """
        features = self.backbone(image)
        features = features.view(features.size(0), -1)  # (batch, 512)

        raw = self.fc_head(features)  # (batch, 2)

        steering = torch.tanh(raw[:, 0:1])
        throttle = torch.sigmoid(raw[:, 1:2])

        return steering, throttle

    def freeze_backbone(self):
        """Freeze ResNet18 backbone parameters (Phase 1 training)."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze ResNet18 backbone for full fine-tuning (Phase 2)."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_features(self, image: torch.Tensor) -> torch.Tensor:
        """Extract 512-d feature vector (used by RL warm-start)."""
        features = self.backbone(image)
        return features.view(features.size(0), -1)
