import numpy as np

def calculate_game(pred_map, gt_count, levels=(0, 1, 2, 3)):
    """Calculate GAME metrics"""
    h, w = pred_map.shape
    results = {}
    
    for level in levels:
        if level == 0:
            results[0] = abs(pred_map.sum() - float(gt_count))
        else:
            S = 2 ** level
            total = 0.0
            for i in range(S):
                for j in range(S):
                    h0, h1 = i * h // S, (i + 1) * h // S
                    w0, w1 = j * w // S, (j + 1) * w // S
                    region_pred = pred_map[h0:h1, w0:w1].sum()
                    region_area = (h1 - h0) * (w1 - w0)
                    region_gt = float(gt_count) * (region_area / (h * w))
                    total += abs(region_pred - region_gt)
            results[level] = total
    
    return results