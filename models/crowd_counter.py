# models/crowd_counter.py
"""
Simple Shared Backbone Crowd Counter
- Shared VGG16 backbone (weight sharing between RGB/Thermal)
- Simple concatenation fusion
- Direct density prediction
- NO complex SAAI alignment (proven to hurt performance)
"""

import torch
import torch.nn as nn
import torchvision.models as models


class SAAICrowdCounter(nn.Module):
    """
    Simple RGB-Thermal crowd counter with shared backbone.
    Named 'SAAI' for backward compatibility, but doesn't use SAAI.
    """
    
    def __init__(self, backbone_name='vgg16', pretrained=True, feature_dim=512):
        super(SAAICrowdCounter, self).__init__()
        
        # Separate input stems (RGB and Thermal have different characteristics)
        self.rgb_stem = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.thermal_stem = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        
        # SHARED VGG16 backbone (weight sharing between modalities)
        if backbone_name == 'vgg16':
            vgg = models.vgg16_bn(pretrained=pretrained)
            # Skip first conv (we have separate stems), keep rest
            self.shared_backbone = nn.Sequential(*list(vgg.features.children())[1:])
            backbone_out_channels = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Feature adapters (Identity in this case)
        self.rgb_adapter = nn.Identity()
        self.thermal_adapter = nn.Identity()
        
        # Simple fusion module (concatenate RGB + Thermal)
        self.fusion_module = nn.Sequential(
            nn.Conv2d(1024, 512, 3, padding=1),  # 512*2 = 1024 input
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        # Density prediction head
        self.regression_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True)  # Non-negative densities
        )
        
        # Fake SAAI aligner for compatibility
        self.saai_aligner = FakeSAAIAligner()
    
    def forward(self, rgb, thermal):
        """
        Args:
            rgb: [B, 3, H, W]
            thermal: [B, 3, H, W]
        Returns:
            density_map: [B, 1, H', W']
            domain_pred_rgb: [B, 2] (fake)
            domain_pred_thermal: [B, 2] (fake)
        """
        # Process through separate stems
        rgb_feat = self.rgb_stem(rgb)
        thermal_feat = self.thermal_stem(thermal)
        
        # Pass through SHARED backbone
        rgb_feat = self.shared_backbone(rgb_feat)
        thermal_feat = self.shared_backbone(thermal_feat)
        
        # Adapters (Identity)
        rgb_feat = self.rgb_adapter(rgb_feat)
        thermal_feat = self.thermal_adapter(thermal_feat)
        
        # Simple concatenation fusion
        fused = torch.cat([rgb_feat, thermal_feat], dim=1)
        fused = self.fusion_module(fused)
        
        # Predict density
        density_map = self.regression_head(fused)
        
        # Fake domain predictions for compatibility with training loop
        B = rgb.size(0)
        fake_domain = torch.zeros(B, 2, device=rgb.device)
        
        return density_map, fake_domain, fake_domain


class FakeSAAIAligner(nn.Module):
    """Fake SAAI module for visualization compatibility"""
    def forward(self, rgb_features, thermal_features):
        B = rgb_features.size(0)
        fake_domain = torch.zeros(B, 2, device=rgb_features.device)
        return rgb_features, thermal_features, fake_domain, fake_domain