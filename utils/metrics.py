# utils/metrics.py (CORRECTED)
import torch
import numpy as np


def calculate_game(pred_density, gt_count):

    if not isinstance(pred_density, np.ndarray):
        pred_density = pred_density.cpu().numpy() if torch.is_tensor(pred_density) else np.array(pred_density)
    
    H, W = pred_density.shape
    results = {}
    
    # GAME(0): Global count error
    pred_total = pred_density.sum()
    results[0] = abs(pred_total - gt_count)
    
    # GAME(1, 2, 3): Grid-based errors
    for level in [1, 2, 3]:
        num_cells = 2 ** level  # 2, 4, 8
        h_step = H / num_cells
        w_step = W / num_cells
        
        total_error = 0.0
        
        for i in range(num_cells):
            for j in range(num_cells):
                # Cell boundaries
                h_start = int(i * h_step)
                h_end = int((i + 1) * h_step) if i < num_cells - 1 else H
                w_start = int(j * w_step)
                w_end = int((j + 1) * w_step) if j < num_cells - 1 else W
                
                # Predicted count in this cell
                pred_region = pred_density[h_start:h_end, w_start:w_end].sum()
                
                # Ground truth count for this cell (proportional to area)
                cell_area = (h_end - h_start) * (w_end - w_start)
                total_area = H * W
                gt_region = gt_count * (cell_area / total_area)
                
                # Accumulate absolute error
                total_error += abs(pred_region - gt_region)
        
        # CORRECTED: Don't divide! GAME is total error, not average
        results[level] = total_error  # ← FIXED!
    
    return results


def calculate_mae_rmse(pred_density, gt_count):
    """Calculate MAE and RMSE"""
    if torch.is_tensor(pred_density):
        pred_count = pred_density.sum(dim=(2, 3)).squeeze()
        if torch.is_tensor(gt_count):
            gt_count_tensor = gt_count
        else:
            gt_count_tensor = torch.tensor(gt_count, dtype=torch.float32, device=pred_density.device)
        
        mae = torch.mean(torch.abs(pred_count - gt_count_tensor)).item()
        mse = torch.mean((pred_count - gt_count_tensor) ** 2).item()
        rmse = np.sqrt(mse)
    else:
        pred_count = pred_density.sum()
        mae = abs(pred_count - gt_count)
        rmse = np.sqrt((pred_count - gt_count) ** 2)
    
    return mae, rmse