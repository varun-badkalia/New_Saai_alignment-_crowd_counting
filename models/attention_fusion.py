import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn

class ModalityAttentionFusion(nn.Module):
    """Ultra-simple element-wise fusion with learnable weights"""
    
    def __init__(self, in_channels=512):
        super(ModalityAttentionFusion, self).__init__()
        
        # Just learn per-channel weights
        self.rgb_weight = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        self.thermal_weight = nn.Parameter(torch.ones(1, in_channels, 1, 1))
        
        # Single refinement conv
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, rgb_features, thermal_features):
        # Weighted sum
        fused = rgb_features * self.rgb_weight + thermal_features * self.thermal_weight
        
        # Refine
        output = self.refine(fused)
        
        return output
    
  