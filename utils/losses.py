# utils/losses.py
"""
Simple 2-component loss: Density MSE + Count constraint
NO domain adversarial loss (proven to hurt performance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAAIBrokerLoss(nn.Module):
    """Simple crowd counting loss"""
    
    def __init__(self, alpha=0.0, beta=0.01, gamma=10.0):
        super(SAAIBrokerLoss, self).__init__()
        self.alpha = alpha  # Ignored (no domain loss)
        self.beta = beta    # Count loss weight
        self.gamma = gamma  # Density MSE weight
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, density_map, keypoints_list, targets_list,
                domain_pred_rgb, domain_pred_thermal, gt_density_maps=None):
        """
        Compatible with existing training code signature.
        Ignores domain predictions (they're fake).
        """
        device = density_map.device
        B = density_map.size(0)
        
        # Main loss: Density MSE (if GT density available)
        density_loss = torch.tensor(0.0, device=device)
        if gt_density_maps is not None:
            if gt_density_maps.shape != density_map.shape:
                gt_density_maps = F.interpolate(
                    gt_density_maps,
                    size=density_map.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            density_loss = self.mse_loss(density_map, gt_density_maps)
        
        # Count loss (global constraint)
        count_loss = torch.tensor(0.0, device=device)
        if gt_density_maps is not None:
            pred_counts = density_map.sum(dim=(1, 2, 3))
            gt_counts = gt_density_maps.sum(dim=(1, 2, 3))
            count_loss = self.l1_loss(pred_counts, gt_counts)
        
        # Total loss
        main_loss = self.gamma * density_loss
        total_loss = main_loss + self.beta * count_loss
        
        return total_loss, {
            'density_loss': main_loss.item(),
            'count_loss': count_loss.item(),
            'domain_loss': 0.0,  # Fake for logging
            'ssim_loss': 0.0,
            'regional_loss': 0.0,
            'total_loss': total_loss.item()
        }