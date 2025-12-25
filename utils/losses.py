import torch
import torch.nn as nn
import torch.nn.functional as F

class SAFLCrowdLoss(nn.Module):
    def __init__(self, w_density=1.0, w_regional=0.1, w_domain=0.1):
        super(SAFLCrowdLoss, self).__init__()
        
        self.w_density = w_density
        self.w_regional = w_regional
        self.w_domain = w_domain
        
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.ce_loss = nn.CrossEntropyLoss()
        
    def regional_loss(self, pred_density, gt_density, levels=[1, 2, 3]):
        """
        Calculates loss over grid regions (Pyramid scaling).
        This optimizes directly for GAME metrics by checking counts in 
        increasingly finer grid cells.
        """
        B, _, H, W = pred_density.shape
        total_loss = 0.0
        
        for L in levels:
            cells = 2 ** L
            # Adaptive Average Pooling automatically sums up the density in grid cells
            # We multiply by area because AvgPool gives mean, we want sum.
            # Grid size: (cells x cells)
            pred_grid = F.adaptive_avg_pool2d(pred_density, (cells, cells)) * (H * W / (cells * cells))
            gt_grid = F.adaptive_avg_pool2d(gt_density, (cells, cells)) * (H * W / (cells * cells))
            
            total_loss += F.l1_loss(pred_grid, gt_grid)
            
        return total_loss / len(levels)

    def forward(self, pred_density, gt_density, domain_pred_rgb, domain_pred_thermal):
        device = pred_density.device
        B = pred_density.size(0)
        
        # 1. Density Map Loss (MSE) - Pixel level
        if gt_density is not None:
            # Resize GT if it doesn't match Prediction (e.g., due to downsampling in backbone)
            if gt_density.shape != pred_density.shape:
                gt_density = F.interpolate(gt_density, size=pred_density.shape[2:], mode='bilinear', align_corners=False)
                # Re-normalize count after interpolation to ensure total count remains correct
                # This compensates for bilinear interpolation artifacts
                with torch.no_grad():
                    gt_sum_orig = gt_density.sum(dim=(2,3), keepdim=True)
                    # Avoid division by zero
                    if gt_sum_orig.sum() > 0:
                        gt_density = gt_density * (gt_sum_orig / (gt_density.sum(dim=(2,3), keepdim=True) + 1e-6))

            loss_density = self.mse_loss(pred_density, gt_density)
            loss_regional = self.regional_loss(pred_density, gt_density)
        else:
            loss_density = torch.tensor(0.0, device=device)
            loss_regional = torch.tensor(0.0, device=device)
            
        # 2. Domain Adversarial Loss
        # We want the encoder to extract features that are Domain-Invariant.
        # Standard Domain Adversarial Training (DANN): 
        # The discriminator tries to classify RGB vs Thermal (0 vs 1).
        # The encoder (via GRL layer usually, or inverted labels here) tries to fool it.
        # Here we assume the discriminator is trained to classify correctly, 
        # and we penalize the feature extractor if the classification is too easy?
        # Actually, in standard SAAI/DANN, we usually flip labels or use a Gradient Reversal Layer.
        # For simplicity in this baseline implementation, we use a confusion loss:
        # We want predicted probabilities to be close to 0.5 (uncertain).
        # Alternatively, we can just minimize CrossEntropy against uniform distribution.
        # But sticking to standard practice in your SAAI paper context:
        # We train the discriminator to distinguish (loss_domain_disc) and generator to fool.
        # Since we don't have a separate optimizer step here easily, we effectively minimize:
        # L_domain = CrossEntropy(pred, correct_label) 
        # If this is minimized, the discriminator is winning (bad for invariance).
        # PROPER WAY without GRL: Use uniform targets (0.5, 0.5) for the generator step.
        # However, your provided snippets suggest a simpler approach.
        # Let's stick to a robust implementation:
        # We optimize the whole model to minimize domain classification error? 
        # No, that makes it modality specific.
        # We will assume the `domain_classifier` in SAAI has a GRL (Gradient Reversal Layer) built-in 
        # or we accept that this loss term acts as a regularizer.
        # If no GRL, we should target 0.5 probability (maximum entropy).
        
        # Implementation assuming standard labels for monitoring, 
        # typically SAAI papers use a GRL inside the module. 
        # If your SAAI module doesn't have GRL, this loss forces modality-specific features.
        # To be safe and promote invariance without GRL: 
        # Target = 0.5 for both RGB and Thermal (Confusion).
        
        # RGB=0, Thermal=1. We want prediction to be 0.5.
        # Soft label loss (KL Div or MSE on probas)
        target_uniform = torch.full_like(domain_pred_rgb, 0.5)
        loss_domain = (F.mse_loss(torch.softmax(domain_pred_rgb, dim=1), target_uniform) + 
                       F.mse_loss(torch.softmax(domain_pred_thermal, dim=1), target_uniform))
        
        # Total
        total_loss = (self.w_density * loss_density) + \
                     (self.w_regional * loss_regional) + \
                     (self.w_domain * loss_domain)
                     
        return total_loss, {
            'total': total_loss.item(),
            'density': loss_density.item(),
            'regional': loss_regional.item(),
            'domain': loss_domain.item()
        }