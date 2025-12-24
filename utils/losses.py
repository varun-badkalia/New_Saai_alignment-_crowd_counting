# utils/losses.py (COMPLETE FIX)
import torch
import torch.nn as nn
import torch.nn.functional as F


class SAFLCrowdLoss(nn.Module):
    """
    Complete loss with regional (GAME) optimization
    """
    
    def __init__(self, 
                 w_density=100.0,    # Much higher
                 w_count=0.001,      # Much lower
                 w_domain=0.5,       # Higher
                 w_regional=1.0):    # NEW: Regional loss
        super(SAFLCrowdLoss, self).__init__()
        
        self.w_density = w_density
        self.w_count = w_count
        self.w_domain = w_domain
        self.w_regional = w_regional
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def regional_loss(self, pred_density, gt_density, levels=[1, 2, 3]):
        """Regional count loss (directly optimizes GAME metrics)"""
        B, _, H, W = pred_density.shape
        total_loss = 0.0
        
        for L in levels:
            cells = 2 ** L
            h_step, w_step = H // cells, W // cells
            
            for i in range(cells):
                for j in range(cells):
                    h_start = i * h_step
                    h_end = (i + 1) * h_step if i < cells - 1 else H
                    w_start = j * w_step
                    w_end = (j + 1) * w_step if j < cells - 1 else W
                    
                    pred_region = pred_density[:, :, h_start:h_end, w_start:w_end].sum(dim=(2, 3))
                    gt_region = gt_density[:, :, h_start:h_end, w_start:w_end].sum(dim=(2, 3))
                    
                    total_loss += F.l1_loss(pred_region, gt_region)
        
        return total_loss / (len(levels) * cells * cells)
        
    def forward(self, pred_density, gt_density, 
                domain_pred_rgb, domain_pred_thermal):
        """Compute total loss"""
        device = pred_density.device
        B = pred_density.size(0)
        
        # 1. Density MSE loss (PRIMARY)
        if gt_density is not None:
            if gt_density.shape != pred_density.shape:
                gt_density = F.interpolate(
                    gt_density, 
                    size=pred_density.shape[2:],
                    mode='bilinear',
                    align_corners=False
                )
            density_loss = self.mse_loss(pred_density, gt_density)
        else:
            density_loss = torch.tensor(0.0, device=device)
        
        # 2. Count loss (WEAK constraint)
        pred_count = pred_density.sum(dim=(2, 3)).squeeze()
        gt_count = gt_density.sum(dim=(2, 3)).squeeze() if gt_density is not None else pred_count
        count_loss = self.l1_loss(pred_count, gt_count)
        
        # 3. Regional loss (NEW - optimizes GAME directly!)
        if gt_density is not None:
            regional = self.regional_loss(pred_density, gt_density)
        else:
            regional = torch.tensor(0.0, device=device)
        
        # 4. Domain adversarial loss
        rgb_labels = torch.zeros(B, dtype=torch.long, device=device)
        thermal_labels = torch.ones(B, dtype=torch.long, device=device)
        
        domain_loss = (
            self.ce_loss(domain_pred_rgb, rgb_labels) +
            self.ce_loss(domain_pred_thermal, thermal_labels)
        ) / 2.0
        
        # Total loss
        total_loss = (
            self.w_density * density_loss +
            self.w_count * count_loss +
            self.w_regional * regional +
            self.w_domain * domain_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'density': density_loss.item(),
            'count': count_loss.item(),
            'regional': regional.item(),
            'domain': domain_loss.item(),
            # Contributions
            'density_contrib': (self.w_density * density_loss).item(),
            'count_contrib': (self.w_count * count_loss).item(),
            'regional_contrib': (self.w_regional * regional).item(),
            'domain_contrib': (self.w_domain * domain_loss).item()
        }
        
        return total_loss, loss_dict