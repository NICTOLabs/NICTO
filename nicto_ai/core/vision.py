"""
VGG16 Vision Encoder Integration
Proven CNN architecture for visual feature extraction
Integrated into NICTO's Perception Cortex
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class VGG16Block(nn.Module):
    """VGG16 convolutional block (2-3 conv layers + pooling)"""

    def __init__(self, in_channels: int, out_channels: int, n_convs: int = 2):
        super().__init__()
        layers = []
        for i in range(n_convs):
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class VGG16Encoder(nn.Module):
    """
    VGG16 Feature Extractor

    Architecture:
    - Block 1: 2x Conv(64) + MaxPool
    - Block 2: 2x Conv(128) + MaxPool
    - Block 3: 3x Conv(256) + MaxPool
    - Block 4: 3x Conv(512) + MaxPool
    - Block 5: 3x Conv(512) + MaxPool

    Total: 13 conv layers + 3 FC layers (original VGG16)
    """

    def __init__(self, pretrained: bool = False):
        super().__init__()

        # VGG16 feature layers
        self.block1 = VGG16Block(3, 64, n_convs=2)
        self.block2 = VGG16Block(64, 128, n_convs=2)
        self.block3 = VGG16Block(128, 256, n_convs=3)
        self.block4 = VGG16Block(256, 512, n_convs=3)
        self.block5 = VGG16Block(512, 512, n_convs=3)

        self.feature_dim = 512 * 7 * 7  # 25088 for 224x224 input

        if pretrained:
            self._load_pretrained_weights()

    def _load_pretrained_weights(self):
        """Load ImageNet pretrained VGG16 weights"""
        try:
            import torchvision.models as models
            vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

            # Copy conv layers
            self.block1[0].weight.data.copy_(vgg16.features[0].weight.data)
            self.block1[0].bias.data.copy_(vgg16.features[0].bias.data)
            self.block1[3].weight.data.copy_(vgg16.features[2].weight.data)
            self.block1[3].bias.data.copy_(vgg16.features[2].bias.data)

            self.block2[0].weight.data.copy_(vgg16.features[5].weight.data)
            self.block2[0].bias.data.copy_(vgg16.features[5].bias.data)
            self.block2[3].weight.data.copy_(vgg16.features[7].weight.data)
            self.block2[3].bias.data.copy_(vgg16.features[7].bias.data)

            for i, idx in enumerate([10, 12, 14]):
                self.block3[i * 3].weight.data.copy_(vgg16.features[idx].weight.data)
                self.block3[i * 3].bias.data.copy_(vgg16.features[idx].bias.data)

            for i, idx in enumerate([17, 19, 21]):
                self.block4[i * 3].weight.data.copy_(vgg16.features[idx].weight.data)
                self.block4[i * 3].bias.data.copy_(vgg16.features[idx].bias.data)

            for i, idx in enumerate([24, 26, 28]):
                self.block5[i * 3].weight.data.copy_(vgg16.features[idx].weight.data)
                self.block5[i * 3].bias.data.copy_(vgg16.features[idx].bias.data)

            print("Loaded VGG16 pretrained weights")
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract visual features

        Args:
            x: Image tensor [batch, 3, 224, 224]

        Returns:
            features: [batch, 25088]
        """
        x = self.block1(x)   # [B, 64, 112, 112]
        x = self.block2(x)   # [B, 128, 56, 56]
        x = self.block3(x)   # [B, 256, 28, 28]
        x = self.block4(x)   # [B, 512, 14, 14]
        x = self.block5(x)   # [B, 512, 7, 7]
        x = x.view(x.size(0), -1)  # [B, 25088]
        return x

    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract spatial feature maps (before flattening)

        Returns: [batch, 512, 7, 7]
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x


class VGG16ForNICTO(nn.Module):
    """
    VGG16 adapted for NICTO AI Perception Cortex

    Projects VGG16 features into NICTO's embedding space
    """

    def __init__(self, nicto_dim: int = 8192, pretrained: bool = True):
        super().__init__()
        self.nicto_dim = nicto_dim

        # VGG16 backbone
        self.vgg16 = VGG16Encoder(pretrained=pretrained)

        # Projection to NICTO dimension
        self.projection = nn.Sequential(
            nn.Linear(self.vgg16.feature_dim, 4096),
            nn.LayerNorm(4096),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(4096, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(2048, nicto_dim),
            nn.LayerNorm(nicto_dim),
        )

        # Visual tokenizer (patch embeddings for sequence output)
        self.patch_projector = nn.Sequential(
            nn.Linear(512, nicto_dim),
            nn.LayerNorm(nicto_dim),
        )

    def forward(self, images: torch.Tensor) -> dict:
        """
        Process images through VGG16

        Args:
            images: [batch, 3, 224, 224]

        Returns:
            Dictionary with:
            - global_features: [batch, nicto_dim]
            - patch_features: [batch, 49, nicto_dim] (7x7 patches)
            - raw_features: [batch, 25088]
        """
        # Global features
        raw_features = self.vgg16(images)  # [B, 25088]
        global_features = self.projection(raw_features)  # [B, nicto_dim]

        # Spatial features for patch-level processing
        spatial = self.vgg16.forward_spatial(images)  # [B, 512, 7, 7]
        batch_size, channels, h, w = spatial.shape

        # Reshape to patches: [B, 512, 49] -> [B, 49, 512]
        patches = spatial.view(batch_size, channels, h * w).permute(0, 2, 1)

        # Project patches to NICTO dimension
        patch_features = self.patch_projector(patches)  # [B, 49, nicto_dim]

        return {
            "global_features": global_features,
            "patch_features": patch_features,
            "raw_features": raw_features,
        }


class VGG16Backbone(nn.Module):
    """
    Standalone VGG16 for pretraining or feature extraction
    """

    def __init__(self, n_classes: int = 1000, pretrained: bool = False):
        super().__init__()
        self.features = nn.Sequential(
            VGG16Block(3, 64, 2),
            VGG16Block(64, 128, 2),
            VGG16Block(128, 256, 3),
            VGG16Block(256, 512, 3),
            VGG16Block(512, 512, 3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, n_classes),
        )

        if pretrained:
            self._load_pretrained()

    def _load_pretrained(self):
        try:
            import torchvision.models as models
            vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            self.features.load_state_dict(vgg16.features.state_dict())
            self.classifier[0].weight.data.copy_(vgg16.classifier[0].weight.data)
            self.classifier[0].bias.data.copy_(vgg16.classifier[0].bias.data)
            self.classifier[3].weight.data.copy_(vgg16.classifier[3].weight.data)
            self.classifier[3].bias.data.copy_(vgg16.classifier[3].bias.data)
            self.classifier[6].weight.data.copy_(vgg16.classifier[6].weight.data)
            self.classifier[6].bias.data.copy_(vgg16.classifier[6].bias.data)
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x.view(x.size(0), -1)
