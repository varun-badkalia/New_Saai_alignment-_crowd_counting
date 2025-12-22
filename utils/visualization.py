# utils/visualization.py
import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Denormalize image tensor for visualization"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def visualize_feature_maps(features, title="Features"):
    """Visualize high-dimensional features by taking mean across channels"""
    if features.dim() == 4:
        feat_mean = features[0].mean(dim=0).detach().cpu().numpy()
    else:
        feat_mean = features.detach().cpu().numpy()
    
    feat_mean = (feat_mean - feat_mean.min()) / (feat_mean.max() - feat_mean.min() + 1e-8)
    return feat_mean


def create_overlay(rgb, thermal):
    """Create overlay of RGB and Thermal to check alignment"""
    if isinstance(rgb, torch.Tensor):
        rgb = rgb.detach().cpu().numpy()
    if isinstance(thermal, torch.Tensor):
        thermal = thermal.detach().cpu().numpy()
    
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    thermal = (thermal - thermal.min()) / (thermal.max() - thermal.min() + 1e-8)
    
    overlay = np.zeros((*rgb.shape, 3))
    overlay[:, :, 0] = rgb * 255
    overlay[:, :, 1] = thermal * 255
    overlay[:, :, 2] = 0
    
    return overlay.astype(np.uint8)


def check_alignment_score(rgb_feat, thermal_feat):
    """Calculate alignment score between RGB and Thermal features"""
    rgb_norm = F.normalize(rgb_feat.flatten(1), p=2, dim=1)
    thermal_norm = F.normalize(thermal_feat.flatten(1), p=2, dim=1)
    
    similarity = (rgb_norm * thermal_norm).sum(dim=1).mean().item()
    
    rgb_flat = rgb_feat[0].mean(0).flatten()
    thermal_flat = thermal_feat[0].mean(0).flatten()
    spatial_corr = np.corrcoef(
        rgb_flat.detach().cpu().numpy(),
        thermal_flat.detach().cpu().numpy()
    )[0, 1]
    
    return {
        'cosine_similarity': similarity,
        'spatial_correlation': spatial_corr,
        'alignment_quality': (similarity + spatial_corr) / 2
    }


def extract_features_correctly(model, rgb, thermal):
    """
    Extract features from YOUR actual model architecture:
    - rgb_backbone (separate)
    - thermal_backbone (separate)
    - saai_aligner
    """
    features = {}
    
    try:
        # YOUR model has separate backbones, not shared!
        rgb_backbone_out = model.rgb_backbone(rgb)
        thermal_backbone_out = model.thermal_backbone(thermal)
        features['rgb_backbone'] = rgb_backbone_out
        features['thermal_backbone'] = thermal_backbone_out
        
        # Adapters (Identity in your case, but call them anyway)
        rgb_features = model.rgb_adapter(rgb_backbone_out)
        thermal_features = model.thermal_adapter(thermal_backbone_out)
        features['rgb_features'] = rgb_features
        features['thermal_features'] = thermal_features
        
        # SAAI alignment
        rgb_aligned, thermal_aligned, domain_rgb, domain_thermal = model.saai_aligner(
            rgb_features, thermal_features
        )
        features['rgb_aligned'] = rgb_aligned
        features['thermal_aligned'] = thermal_aligned
        features['domain_rgb'] = domain_rgb
        features['domain_thermal'] = domain_thermal
        
        return features, True
        
    except Exception as e:
        print(f"⚠️ Error during feature extraction: {e}")
        import traceback
        traceback.print_exc()
        return {}, False


def visualize_saai_alignment(model, rgb, thermal, save_path, density_gt=None):
    """
    Visualization for YOUR model architecture (separate backbones)
    """
    model.eval()
    
    with torch.no_grad():
        # Get final prediction
        pred_density, _, _ = model(rgb, thermal)
        
        # Extract intermediate features
        features, success = extract_features_correctly(model, rgb, thermal)
    
    if success and len(features) > 0:
        # Calculate alignment scores
        before_align = check_alignment_score(
            features['rgb_features'], 
            features['thermal_features']
        )
        after_align = check_alignment_score(
            features['rgb_aligned'], 
            features['thermal_aligned']
        )
        improvement = after_align['alignment_quality'] - before_align['alignment_quality']
        improvement_pct = (improvement / before_align['alignment_quality'] * 100) if before_align['alignment_quality'] > 1e-6 else 0
        
        # Create figure
        fig = plt.figure(figsize=(20, 12))
        
        # Row 1: Inputs and outputs
        ax1 = plt.subplot(3, 5, 1)
        rgb_vis = denormalize(rgb[0]).permute(1, 2, 0).cpu().numpy()
        ax1.imshow(rgb_vis)
        ax1.set_title('RGB Input', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = plt.subplot(3, 5, 2)
        thermal_vis = denormalize(thermal[0]).permute(1, 2, 0).cpu().numpy()
        ax2.imshow(thermal_vis)
        ax2.set_title('Thermal Input', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3 = plt.subplot(3, 5, 3)
        overlay_input = create_overlay(rgb_vis.mean(axis=2), thermal_vis.mean(axis=2))
        ax3.imshow(overlay_input)
        ax3.set_title('Input Overlay\n(Yellow=Aligned)', fontsize=11, fontweight='bold')
        ax3.axis('off')
        
        ax4 = plt.subplot(3, 5, 4)
        pred_vis = pred_density[0, 0].cpu().numpy()
        im4 = ax4.imshow(pred_vis, cmap='jet')
        ax4.set_title(f'Predicted Density\nCount: {pred_vis.sum():.1f}', fontsize=11)
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        if density_gt is not None:
            ax5 = plt.subplot(3, 5, 5)
            gt_vis = density_gt[0, 0].cpu().numpy()
            im5 = ax5.imshow(gt_vis, cmap='jet')
            error = abs(pred_vis.sum() - gt_vis.sum())
            ax5.set_title(f'GT Density\nCount: {gt_vis.sum():.1f}\nError: {error:.1f}', fontsize=11)
            ax5.axis('off')
            plt.colorbar(im5, ax=ax5, fraction=0.046)
        
        # Row 2: Features BEFORE SAAI alignment
        ax6 = plt.subplot(3, 5, 6)
        rgb_feat_vis = visualize_feature_maps(features['rgb_features'])
        im6 = ax6.imshow(rgb_feat_vis, cmap='hot')
        ax6.set_title(f'RGB Features\n(Before SAAI)\nSim: {before_align["cosine_similarity"]:.3f}', fontsize=10)
        ax6.axis('off')
        plt.colorbar(im6, ax=ax6, fraction=0.046)
        
        ax7 = plt.subplot(3, 5, 7)
        thermal_feat_vis = visualize_feature_maps(features['thermal_features'])
        im7 = ax7.imshow(thermal_feat_vis, cmap='hot')
        ax7.set_title(f'Thermal Features\n(Before SAAI)\nCorr: {before_align["spatial_correlation"]:.3f}', fontsize=10)
        ax7.axis('off')
        plt.colorbar(im7, ax=ax7, fraction=0.046)
        
        ax8 = plt.subplot(3, 5, 8)
        overlay_before = create_overlay(rgb_feat_vis, thermal_feat_vis)
        ax8.imshow(overlay_before)
        ax8.set_title(f'Before Alignment\nQuality: {before_align["alignment_quality"]:.3f}', fontsize=10)
        ax8.axis('off')
        
        # Row 3: Features AFTER SAAI alignment
        ax9 = plt.subplot(3, 5, 11)
        rgb_aligned_vis = visualize_feature_maps(features['rgb_aligned'])
        im9 = ax9.imshow(rgb_aligned_vis, cmap='hot')
        ax9.set_title(f'RGB Aligned\n(After SAAI)\nSim: {after_align["cosine_similarity"]:.3f}', fontsize=10)
        ax9.axis('off')
        plt.colorbar(im9, ax=ax9, fraction=0.046)
        
        ax10 = plt.subplot(3, 5, 12)
        thermal_aligned_vis = visualize_feature_maps(features['thermal_aligned'])
        im10 = ax10.imshow(thermal_aligned_vis, cmap='hot')
        ax10.set_title(f'Thermal Aligned\n(After SAAI)\nCorr: {after_align["spatial_correlation"]:.3f}', fontsize=10)
        ax10.axis('off')
        plt.colorbar(im10, ax=ax10, fraction=0.046)
        
        ax11 = plt.subplot(3, 5, 13)
        overlay_after = create_overlay(rgb_aligned_vis, thermal_aligned_vis)
        ax11.imshow(overlay_after)
        color = 'green' if improvement > 0.05 else 'orange' if improvement > 0 else 'red'
        ax11.set_title(f'After Alignment\nQuality: {after_align["alignment_quality"]:.3f}', 
                      fontsize=10, color=color, fontweight='bold')
        ax11.axis('off')
        
        # Metrics panel
        ax12 = plt.subplot(3, 5, 14)
        ax12.axis('off')
        
        # Determine status
        if improvement > 0.05:
            status = "✓ WORKING"
            status_color = 'green'
        elif improvement > 0:
            status = "⚠ WEAK"
            status_color = 'orange'
        else:
            status = "✗ NOT WORKING"
            status_color = 'red'
        
        metrics_text = f"""
SAAI ALIGNMENT ANALYSIS:

Architecture:
  • Separate Backbones
  • SAAI Aligner Active

Before SAAI:
  Cosine:  {before_align['cosine_similarity']:.4f}
  Spatial: {before_align['spatial_correlation']:.4f}
  Quality: {before_align['alignment_quality']:.4f}

After SAAI:
  Cosine:  {after_align['cosine_similarity']:.4f}
  Spatial: {after_align['spatial_correlation']:.4f}
  Quality: {after_align['alignment_quality']:.4f}

Improvement:
  Absolute: {improvement:+.4f}
  Relative: {improvement_pct:+.1f}%

Status: {status}
        """
        
        ax12.text(0.05, 0.5, metrics_text, fontsize=8, family='monospace',
                 verticalalignment='center',
                 bbox=dict(boxstyle='round', facecolor=status_color, alpha=0.3))
        
        title_color = 'green' if improvement > 0.05 else 'orange' if improvement > 0 else 'red'
        plt.suptitle(f'SAAI Alignment Visualization - {status}', 
                    fontsize=14, fontweight='bold', color=title_color)
        
    else:
        # Fallback simple visualization
        fig = plt.figure(figsize=(15, 5))
        
        ax1 = plt.subplot(1, 5, 1)
        rgb_vis = denormalize(rgb[0]).permute(1, 2, 0).cpu().numpy()
        ax1.imshow(rgb_vis)
        ax1.set_title('RGB Input', fontsize=12, fontweight='bold')
        ax1.axis('off')
        
        ax2 = plt.subplot(1, 5, 2)
        thermal_vis = denormalize(thermal[0]).permute(1, 2, 0).cpu().numpy()
        ax2.imshow(thermal_vis)
        ax2.set_title('Thermal Input', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        ax3 = plt.subplot(1, 5, 3)
        overlay = create_overlay(rgb_vis.mean(axis=2), thermal_vis.mean(axis=2))
        ax3.imshow(overlay)
        ax3.set_title('Overlay', fontsize=12)
        ax3.axis('off')
        
        ax4 = plt.subplot(1, 5, 4)
        pred_vis = pred_density[0, 0].cpu().numpy()
        im4 = ax4.imshow(pred_vis, cmap='jet')
        ax4.set_title(f'Predicted\n{pred_vis.sum():.1f}', fontsize=12)
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046)
        
        ax5 = plt.subplot(1, 5, 5)
        ax5.axis('off')
        ax5.text(0.5, 0.5, '⚠️ Feature\nextraction\nfailed', 
                ha='center', va='center', fontsize=12,
                bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
        
        plt.suptitle('SAAI Visualization - Error', fontsize=14, color='red')
        
        before_align = {'alignment_quality': 0.0}
        after_align = {'alignment_quality': 0.0}
        improvement = 0.0
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    if success:
        status_emoji = "✅" if improvement > 0.05 else "⚠️" if improvement > 0 else "❌"
        print(f"{status_emoji} Visualization saved to {save_path}")
        print(f"📊 SAAI Alignment: {improvement:+.4f} ({improvement_pct:+.1f}%)")
        if improvement > 0.05:
            print(f"   Status: WORKING - SAAI is improving alignment!")
        elif improvement > 0:
            print(f"   Status: WEAK - SAAI has minimal effect")
        else:
            print(f"   Status: NOT WORKING - SAAI may be hurting performance")
    else:
        print(f"❌ Visualization saved (fallback mode)")
    
    return {
        'before': before_align,
        'after': after_align,
        'improvement': improvement
    }