# models/safl_crowd_counter.py
"""
SAAI Crowd Counter with SEPARATE backbones + 8× downsampling
Best of both worlds!
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import make_dilated_vgg_backbone, InputStemRGB, InputStemThermal, FeatureAdapter
from .saai_aligner import SemanticAdversarialAligner


class SAFLCrowdCounter(nn.Module):
    """
    RGB-Thermal Crowd Counter with:
    - SEPARATE dilated VGG16 backbones (8× downsampling)
    - Full SAAI alignment (prototypes + cross-attention + adversarial)
    - Dense prediction head
    """
    
    def __init__(self, 
                 pretrained=True, 
                 feature_dim=512,
                 num_prototypes=64):
        super(SAFLCrowdCounter, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Separate input stems
        self.rgb_stem = InputStemRGB()
        self.thermal_stem = InputStemThermal()
        
        # SEPARATE dilated backbones (KEY: Not shared!)
        self.rgb_backbone = make_dilated_vgg_backbone(pretrained=pretrained)
        self.thermal_backbone = make_dilated_vgg_backbone(pretrained=pretrained)
        
        # Feature adapters (if needed)
        self.rgb_adapter = nn.Identity()
        self.thermal_adapter = nn.Identity()
        
        # FULL SAAI alignment module
        self.saai_aligner = SemanticAdversarialAligner(
            feature_dim=feature_dim,
            num_prototypes=num_prototypes,
            num_heads=8
        )
        
        # Fusion module
        self.fusion_module = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
        
        # Density prediction head
        self.density_head = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.ReLU(inplace=True)  # Non-negative density
        )
        
    def forward(self, rgb, thermal):
        """
        Forward pass
        
        Args:
            rgb: [B, 3, H, W] RGB images
            thermal: [B, 3, H, W] Thermal images
        
        Returns:
            density_map: [B, 1, H/8, W/8] Predicted density map
            domain_pred_rgb: [B, 2] Domain predictions for RGB
            domain_pred_thermal: [B, 2] Domain predictions for Thermal
        """
        # Input stems
        rgb_feat = self.rgb_stem(rgb)          # [B, 64, H, W]
        thermal_feat = self.thermal_stem(thermal)  # [B, 64, H, W]
        
        # SEPARATE backbones (each modality gets optimal processing)
        rgb_feat = self.rgb_backbone(rgb_feat)      # [B, 512, H/8, W/8]
        thermal_feat = self.thermal_backbone(thermal_feat)  # [B, 512, H/8, W/8]
        
        # Adapters
        rgb_feat = self.rgb_adapter(rgb_feat)
        thermal_feat = self.thermal_adapter(thermal_feat)
        
        # FULL SAAI alignment (prototypes + cross-attention + adversarial)
        rgb_aligned, thermal_aligned, domain_pred_rgb, domain_pred_thermal = \
            self.saai_aligner(rgb_feat, thermal_feat)
        
        # Fusion
        fused = torch.cat([rgb_aligned, thermal_aligned], dim=1)
        fused = self.fusion_module(fused)  # [B, 256, H/8, W/8]
        
        # Density prediction
        density_map = self.density_head(fused)  # [B, 1, H/8, W/8]
        
        return density_map, domain_pred_rgb, domain_pred_thermal