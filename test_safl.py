# test_safl.py
"""
Testing script for SAFL crowd counter
"""
import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets.broker_crowd_dataset import BrokerCrowdDataset, crowd_collate
from models import SAFLCrowdCounter
from utils import calculate_game, calculate_mae_rmse


def test():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', required=True, help='Path to test data')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint')
    parser.add_argument('--save-dir', default='./test_results', help='Save directory')
    parser.add_argument('--num-prototypes', type=int, default=8)
    parser.add_argument('--fusion-method', type=str, default='concat')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Dataset
    test_root = os.path.join(args.data_path, 'test')
    test_dataset = BrokerCrowdDataset(
        test_root,
        crop_size=384,
        downsample_ratio=8,
        method='test',
        enable_gt_density=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=crowd_collate
    )
    
    # Model
    model = SAFLCrowdCounter(
        pretrained=False,
        num_prototypes=args.num_prototypes,
        fusion_method=args.fusion_method
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Testing on {len(test_dataset)} images...")
    
    # Test
    game_sums = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    mae_sum = 0.0
    mse_sum = 0.0
    results = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            rgb, thermal, _, gt_count, name = batch
            rgb = rgb.to(device)
            thermal = thermal.to(device)
            
            pred_density, _, _ = model(rgb, thermal)
            
            pred_map = pred_density[0, 0].cpu().numpy()
            gt_c = int(gt_count.item()) if isinstance(gt_count, torch.Tensor) else int(gt_count)
            pred_c = pred_map.sum()
            
            # GAME metrics
            games = calculate_game(pred_map, gt_c)
            for k in games:
                game_sums[k] += games[k]
            
            # MAE and MSE
            mae_sum += abs(pred_c - gt_c)
            mse_sum += (pred_c - gt_c) ** 2
            
            results.append({
                'name': name,
                'gt': gt_c,
                'pred': pred_c,
                'error': abs(pred_c - gt_c)
            })
    
    # Calculate averages
    n = len(test_loader)
    for k in game_sums:
        game_sums[k] /= n
    mae = mae_sum / n
    rmse = np.sqrt(mse_sum / n)
    
    # Print results
    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)
    print(f"MAE:     {mae:.2f}")
    print(f"RMSE:    {rmse:.2f}")
    print(f"GAME(0): {game_sums[0]:.2f}")
    print(f"GAME(1): {game_sums[1]:.2f}")
    print(f"GAME(2): {game_sums[2]:.2f}")
    print(f"GAME(3): {game_sums[3]:.2f}")
    print("=" * 60)
    
    # Save results
    results_file = os.path.join(args.save_dir, 'test_results.txt')
    with open(results_file, 'w') as f:
        f.write("Test Results\n")
        f.write("=" * 60 + "\n")
        f.write(f"MAE:     {mae:.2f}\n")
        f.write(f"RMSE:    {rmse:.2f}\n")
        f.write(f"GAME(0): {game_sums[0]:.2f}\n")
        f.write(f"GAME(1): {game_sums[1]:.2f}\n")
        f.write(f"GAME(2): {game_sums[2]:.2f}\n")
        f.write(f"GAME(3): {game_sums[3]:.2f}\n")
        f.write("=" * 60 + "\n\n")
        f.write("Per-image results:\n")
        for r in results:
            f.write(f"{r['name']}: GT={r['gt']}, Pred={r['pred']:.1f}, Error={r['error']:.1f}\n")
    
    print(f"\nResults saved to {results_file}")


if __name__ == '__main__':
    test()