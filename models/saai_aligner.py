import torch
import torch.nn as nn
import torch.nn.functional as F

class SemanticAdversarialAligner(nn.Module):
    """
    SAAI module for RGB-Thermal Crowd Counting.
    Features:
    1. Prototype Alignment (Semantic)
    2. Cross-Attention (Soft Spatial)
    3. Domain Adversarial Discriminator
    """
    
    def __init__(self, feature_dim=512, num_prototypes=64, num_heads=8):
        super(SemanticAdversarialAligner, self).__init__()
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        
        # 1. Semantic Prototypes (Learnable "Cluster Centers")
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, feature_dim))
        nn.init.kaiming_normal_(self.prototypes, mode='fan_out', nonlinearity='relu')
        
        # 2. Cross-Modal Attention (Soft Alignment)
        self.cross_attention_rgb = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        self.cross_attention_thermal = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        
        # Feature Transformations
        self.rgb_transform = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.BatchNorm2d(feature_dim)
        )
        
        self.thermal_transform = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, 1),
            nn.BatchNorm2d(feature_dim)
        )
        
        # 3. Domain Discriminator (Adversarial)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)  # 0: RGB, 1: Thermal
        )

    def prototype_alignment(self, features):
        """Align features with learned semantic prototypes"""
        B, C, H, W = features.shape
        features_flat = features.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        
        # Compute similarity to prototypes
        prototypes = F.normalize(self.prototypes, dim=1)
        features_norm = F.normalize(features_flat, dim=2)
        similarities = torch.matmul(features_norm, prototypes.T)  # [B, HW, num_prototypes]
        
        # Soft assignment
        assignments = F.softmax(similarities, dim=2)
        aligned_features = torch.matmul(assignments, prototypes)  # [B, HW, C]
        
        return aligned_features.permute(0, 2, 1).view(B, C, H, W)

    def forward(self, rgb_features, thermal_features):
        """
        Args:
            rgb_features: [B, C, H, W]
            thermal_features: [B, C, H, W]
        Returns:
            rgb_final, thermal_final, domain_pred_rgb, domain_pred_thermal
        """
        B, C, H, W = rgb_features.shape
        
        # Transform
        rgb_transformed = self.rgb_transform(rgb_features)
        thermal_transformed = self.thermal_transform(thermal_features)
        
        # Semantic Alignment (Prototypes)
        rgb_proto = self.prototype_alignment(rgb_transformed)
        thermal_proto = self.prototype_alignment(thermal_transformed)
        
        # Spatial Alignment (Cross-Attention)
        rgb_flat = rgb_proto.view(B, C, -1).permute(0, 2, 1)      # [B, HW, C]
        thermal_flat = thermal_proto.view(B, C, -1).permute(0, 2, 1)
        
        # RGB queries Thermal
        rgb_attended, _ = self.cross_attention_rgb(rgb_flat, thermal_flat, thermal_flat)
        # Thermal queries RGB
        thermal_attended, _ = self.cross_attention_thermal(thermal_flat, rgb_flat, rgb_flat)
        
        # Reshape & Residual
        rgb_aligned = rgb_attended.permute(0, 2, 1).view(B, C, H, W)
        thermal_aligned = thermal_attended.permute(0, 2, 1).view(B, C, H, W)
        
        rgb_final = rgb_aligned + rgb_features
        thermal_final = thermal_aligned + thermal_features
        
        # Domain Discrimination
        rgb_pooled = F.adaptive_avg_pool2d(rgb_final, 1).view(B, -1)
        thermal_pooled = F.adaptive_avg_pool2d(thermal_final, 1).view(B, -1)
        
        domain_pred_rgb = self.domain_classifier(rgb_pooled)
        domain_pred_thermal = self.domain_classifier(thermal_pooled)
        
        return rgb_final, thermal_final, domain_pred_rgb, domain_pred_thermal