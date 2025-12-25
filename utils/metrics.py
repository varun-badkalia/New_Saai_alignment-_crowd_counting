import torch
import numpy as np

def calculate_game(pred_density, gt_count):
    """
    Calculate GAME (Grid Average Mean Absolute Error) at levels 0, 1, 2, 3.
    
    Args:
        pred_density: Tensor or Array [H, W] (Single image density map)
        gt_count: Float or Int (Total ground truth count)
        
    Returns:
        dict: {0: error, 1: error, 2: error, 3: error}
    """
    if torch.is_tensor(pred_density):
        pred_density = pred_density.detach().cpu().numpy()
        
    H, W = pred_density.shape
    out = {}

    # GAME(0) is simply the global count error
    pred_total = pred_density.sum()
    out[0] = abs(pred_total - gt_count)

    # GAME(L) divides image into 4^L grids
    for L in [1, 2, 3]:
        num_grid_cells = 2 ** L
        err = 0.0
        
        # Grid steps
        h_step = H / num_grid_cells
        w_step = W / num_grid_cells
        
        for i in range(num_grid_cells):
            for j in range(num_grid_cells):
                # Define cell boundaries
                h0, h1 = int(i * h_step), int((i + 1) * h_step)
                w0, w1 = int(j * w_step), int((j + 1) * w_step)
                
                # Handle edge case at last cell
                if i == num_grid_cells - 1: h1 = H
                if j == num_grid_cells - 1: w1 = W
                
                # Sum prediction in this cell
                pred_c = pred_density[h0:h1, w0:w1].sum()
                
                # Estimate GT in this cell (Assuming uniform distribution for GAME calculation 
                # if point coordinates aren't available during simple validation. 
                # Ideally, use actual point counts per grid if available).
                # Note: For strict GAME evaluation, you should pass grid-wise GT counts. 
                # Here we use area-proportional estimation for simplicity in standard validation loops.
                cell_area = (h1 - h0) * (w1 - w0)
                total_area = H * W
                gt_c = gt_count * (cell_area / total_area)
                
                err += abs(pred_c - gt_c)
                
        out[L] = err

    return out

def calculate_mae_rmse(pred_density, gt_count):
    """
    Calculate global MAE and RMSE.
    """
    if torch.is_tensor(pred_density):
        pred_count = pred_density.sum().item()
    else:
        pred_count = pred_density.sum()
        
    if torch.is_tensor(gt_count):
        gt_count = gt_count.item()
        
    diff = pred_count - gt_count
    mae = abs(diff)
    mse = diff ** 2
    return mae, mse