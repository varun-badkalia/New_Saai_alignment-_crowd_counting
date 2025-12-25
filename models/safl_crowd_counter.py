import torch
import torch.nn as nn
from .backbones import InputStemRGB, InputStemThermal, make_dilated_vgg_backbone
from .saai_aligner import SemanticAdversarialAligner

class SAFLCrowdCounter(nn.Module):
    """
    Main model class for SAAI-based Crowd Counting.
    Integrates separate backbones, SAAI alignment, and fusion.
    """
    def __init__(self, pretrained=True, feature_dim=512, num_prototypes=64):
        super(SAFLCrowdCounter, self).__init__()

        # 1. Input Stems (Process raw images first)
        self.rgb_stem = InputStemRGB()
        self.thermal_stem = InputStemThermal()

        # 2. Feature Extractors (Dilated VGG-16)
        # Keeps spatial resolution at 1/8 instead of 1/32
        self.rgb_backbone = make_dilated_vgg_backbone(pretrained)
        self.thermal_backbone = make_dilated_vgg_backbone(pretrained)

        # 3. The Semantic Aligner (SAAI)
        # Matches Semantic features (Prototypes) and Spatial features (Cross-Attn)
        self.saai_aligner = SemanticAdversarialAligner(
            feature_dim=feature_dim,
            num_prototypes=num_prototypes,
            num_heads=8
        )

        # 4. Fusion Layer
        # Concatenates RGB(512) + Thermal(512) -> 1024 channels
        # Reduces to 256 for the head
        self.fusion = nn.Sequential(
            nn.Conv2d(feature_dim * 2, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # 5. Density Regression Head
        # Maps fused features to a 1-channel density map
        self.density_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True) # Enforce non-negative density (ReLU)
        )

    def forward(self, rgb, thermal):
        # 1. Extract Features
        # Stem -> Backbone
        rgb_feat = self.rgb_backbone(self.rgb_stem(rgb))
        thermal_feat = self.thermal_backbone(self.thermal_stem(thermal))

        # 2. Align Features (SAAI)
        # Returns aligned features + domain predictions for loss
        rgb_aligned, thermal_aligned, dom_r, dom_t = self.saai_aligner(rgb_feat, thermal_feat)

        # 3. Fuse Features
        concat_feats = torch.cat([rgb_aligned, thermal_aligned], dim=1) # [B, 1024, H, W]
        fused = self.fusion(concat_feats)

        # 4. Predict Density
        density = self.density_head(fused)

        return density, dom_r, dom_t