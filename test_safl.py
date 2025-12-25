# test_v25.py
"""
Testing script for SAAI V2.5
"""
import os
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets.broker_crowd_dataset import BrokerCrowdDataset, crowd_collate
from models import SAAICrowdCounterV25
from utils import calculate_game


def get_args():
    parser = argparse.ArgumentParser(description='SAAI V2.5 Testing')
    
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output-stride', type=int, default=2)
    parser.add_argument('--workers', type=int, default=4)
    
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("SAAI V2.5 Testing")
    print("=" * 80)
    
    # Dataset
    test_root = os.path.join(args.data_path, 'test')
    
    test_dataset = BrokerCrowdDataset(
        test_root,
        crop_size=384,
        downsample_ratio=2,
        method='test',
        enable_gt_density=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=crowd_collate
    )
    
    print(f"Test images: {len(test_dataset)}")
    
    # Model
    model = SAAICrowdCounterV25(
        pretrained=False,
        output_stride=args.output_stride
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded checkpoint from: {args.checkpoint}")
    
    # Testing
    game_sums = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    mae_sum = 0.0
    mse_sum = 0.0
    n = 0
    
    results = []
    
    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            rgb, thermal, _, gt_count, name = batch
            rgb = rgb.to(device)
            thermal = thermal.to(device)
            
            # Forward
            predictions, _, _, _, _, _ = model(rgb, thermal)
            pred_density = predictions['main']
            
            # Metrics
            pred_map = pred_density[0, 0].cpu().numpy()
            gt_c = int(gt_count.item()) if isinstance(gt_count, torch.Tensor) else int(gt_count)
            
            games = calculate_game(pred_map, gt_c)
            for k in games:
                game_sums[k] += games[k]
            
            pred_cnt = pred_map.sum()
            mae = abs(pred_cnt - gt_c)
            mae_sum += mae
            mse_sum += (pred_cnt - gt_c) ** 2
            
            results.append({
                'name': name,
                'gt_count': gt_c,
                'pred_count': pred_cnt,
                'mae': mae,
                'game0': games[0]
            })
            
            n += 1
    
    # Final metrics
    for k in game_sums:
        game_sums[k] /= n
    mae_final = mae_sum / n
    rmse_final = np.sqrt(mse_sum / n)
    
    print("\n" + "=" * 80)
    print("TEST RESULTS")
    print("=" * 80)
    print(f"MAE:     {mae_final:.2f}")
    print(f"RMSE:    {rmse_final:.2f}")
    print(f"GAME(0): {game_sums[0]:.2f}")
    print(f"GAME(1): {game_sums[1]:.2f}")
    print(f"GAME(2): {game_sums[2]:.2f}")
    print(f"GAME(3): {game_sums[3]:.2f}")
    print("=" * 80)
    
    # Save results
    import json
    results_file = args.checkpoint.replace('.pth', '_test_results.json')
    with open(results_file, 'w') as f:
        json.dump({
            'metrics': {
                'mae': mae_final,
                'rmse': rmse_final,
                'game0': game_sums[0],
                'game1': game_sums[1],
                'game2': game_sums[2],
                'game3': game_sums[3]
            },
            'per_image': results
        }, f, indent=2)
    
    print(f"Results saved to: {results_file}")


if __name__ == '__main__':
    main()
