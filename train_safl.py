# train_safl.py
"""
Training script with proper domain adversarial loss
"""
import os
import argparse
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets.broker_crowd_dataset import BrokerCrowdDataset, crowd_collate
from models import SAFLCrowdCounter
from utils import SAFLCrowdLoss, calculate_game


def get_args():
    parser = argparse.ArgumentParser(description='SAAI RGB-Thermal Crowd Counting')
    
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--w-regional', type=float, default=1.0)
    parser.add_argument('--crop-size', type=int, default=384)
    parser.add_argument('--downsample-ratio', type=int, default=8)
    parser.add_argument('--num-prototypes', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    
    # Loss weights
    parser.add_argument('--w-density', type=float, default=10.0)
    parser.add_argument('--w-count', type=float, default=0.01)
    parser.add_argument('--w-domain', type=float, default=0.1)
    
    parser.add_argument('--save-dir', type=str, default='./results_saai_fixed')
    parser.add_argument('--val-freq', type=int, default=2)
    parser.add_argument('--save-freq', type=int, default=10)
    
    return parser.parse_args()


def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_file = os.path.join(save_dir, f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def validate(model, val_loader, device, logger):
    model.eval()
    game_sums = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    mae_sum = 0.0
    mse_sum = 0.0
    n = 0
    
    with torch.no_grad():
        for batch in val_loader:
            rgb, thermal, _, gt_count, name = batch
            rgb = rgb.to(device)
            thermal = thermal.to(device)
            
            pred_density, _, _ = model(rgb, thermal)
            
            pred_map = pred_density[0, 0].cpu().numpy()
            gt_c = int(gt_count.item()) if isinstance(gt_count, torch.Tensor) else int(gt_count)
            
            games = calculate_game(pred_map, gt_c)
            for k in games:
                game_sums[k] += games[k]
            
            pred_cnt = pred_map.sum()
            mae_sum += abs(pred_cnt - gt_c)
            mse_sum += (pred_cnt - gt_c) ** 2
            n += 1
    
    for k in game_sums:
        game_sums[k] /= n
    mae = mae_sum / n
    rmse = np.sqrt(mse_sum / n)
    
    logger.info(f"Validation Results:")
    logger.info(f"  MAE: {mae:.2f}")
    logger.info(f"  RMSE: {rmse:.2f}")
    logger.info(f"  GAME(0): {game_sums[0]:.2f}")
    logger.info(f"  GAME(1): {game_sums[1]:.2f}")
    logger.info(f"  GAME(2): {game_sums[2]:.2f}")
    logger.info(f"  GAME(3): {game_sums[3]:.2f}")
    
    return game_sums, mae, rmse


def main():
    args = get_args()
    logger = setup_logging(args.save_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("=" * 80)
    logger.info("SAAI RGB-Thermal Crowd Counting (SEPARATE Backbones + 8× Downsample)")
    logger.info("=" * 80)
    logger.info(f"Device: {device}")
    logger.info(f"Prototypes: {args.num_prototypes}")
    logger.info(f"Downsampling: 8×")
    logger.info(f"Loss weights: Density={args.w_density}, Count={args.w_count}, Domain={args.w_domain}")
    logger.info("=" * 80)
    
    # Datasets
    train_root = os.path.join(args.data_path, 'train')
    val_root = os.path.join(args.data_path, 'val')
    
    train_dataset = BrokerCrowdDataset(
        train_root,
        crop_size=args.crop_size,
        downsample_ratio=args.downsample_ratio,
        method='train',
        enable_gt_density=True
    )
    
    val_dataset = BrokerCrowdDataset(
        val_root,
        crop_size=args.crop_size,
        downsample_ratio=args.downsample_ratio,
        method='val',
        enable_gt_density=False
    )
    
    logger.info(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=crowd_collate,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(1, args.workers // 2),
        collate_fn=crowd_collate
    )
    
    # Model
    model = SAFLCrowdCounter(
        pretrained=True,
        feature_dim=512,
        num_prototypes=args.num_prototypes
    ).to(device)
    
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")
    
    # Loss
    criterion = SAFLCrowdLoss(
        w_density=args.w_density,
        w_count=args.w_count,
        w_domain=args.w_domain,
        w_regional=args.w_regional
    ).to(device)
    


    # NEW (proper rates):
    optimizer = torch.optim.Adam([
        {'params': model.rgb_stem.parameters(), 'lr': args.lr * 0.5},     # 5e-5
        {'params': model.thermal_stem.parameters(), 'lr': args.lr * 0.5}, # 5e-5
        {'params': model.rgb_backbone.parameters(), 'lr': args.lr * 0.5}, # 5e-5
        {'params': model.thermal_backbone.parameters(), 'lr': args.lr * 0.5}, # 5e-5
        {'params': model.saai_aligner.parameters(), 'lr': args.lr},       # 1e-4
        {'params': model.fusion_module.parameters(), 'lr': args.lr * 2.0}, # 2e-4
        {'params': model.density_head.parameters(), 'lr': args.lr * 2.0}   # 2e-4
    ], weight_decay=args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[80, 120, 160],
        gamma=0.5
    )
    
    best_mae = float('inf')
    best_game0 = float('inf')
    
    logger.info("Starting training...")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        # epoch_losses = {'total': 0, 'density': 0, 'count': 0, 'domain': 0}
        epoch_losses = {'total': 0, 'density': 0, 'count': 0, 'regional': 0, 'domain': 0}
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            if len(batch) == 6:
                rgb, thermal, kps_list, targets_list, st_sizes, gt_density = batch
                gt_density = gt_density.to(device)
            else:
                rgb, thermal, kps_list, targets_list, st_sizes = batch
                gt_density = None
            
            rgb = rgb.to(device)
            thermal = thermal.to(device)
            
            optimizer.zero_grad()
            pred_density, domain_pred_rgb, domain_pred_thermal = model(rgb, thermal)
            
            loss, loss_dict = criterion(pred_density, gt_density, domain_pred_rgb, domain_pred_thermal)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key]
            n_batches += 1
            
            pbar.set_postfix({
                'Loss': f"{loss_dict['total']:.4f}",
                'Den': f"{loss_dict['density']:.4f}",
                'Cnt': f"{loss_dict['count']:.2f}",
                'Reg': f"{loss_dict['regional']:.4f}",
                'Dom': f"{loss_dict['domain']:.4f}"
            })
        
        scheduler.step()
        
        avg_losses = {k: v / n_batches for k, v in epoch_losses.items()}
        logger.info(f"Epoch {epoch} - Loss: {avg_losses['total']:.4f} | "
                   f"Density: {avg_losses['density']:.4f} | "
                   f"Count: {avg_losses['count']:.4f} | "
                   f"Regional: {avg_losses['regional']:.4f} | " 
                   f"Domain: {avg_losses['domain']:.4f} | "
                   f"LR: {scheduler.get_last_lr()[0]:.2e}")
        
        if epoch % args.val_freq == 0 or epoch == 1:
            game_results, mae, rmse = validate(model, val_loader, device, logger)
            
            if mae < best_mae:
                best_mae = mae
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'mae': mae,
                    'rmse': rmse,
                    'game': game_results
                }, os.path.join(args.save_dir, 'best_mae.pth'))
                logger.info(f"✓ Saved best MAE: {mae:.2f}")
            
            if game_results[0] < best_game0:
                best_game0 = game_results[0]
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'game0': game_results[0]
                }, os.path.join(args.save_dir, 'best_game0.pth'))
                logger.info(f"✓ Saved best GAME(0): {game_results[0]:.2f}")
        
        if epoch % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pth'))
    
    logger.info("=" * 80)
    logger.info(f"Training completed! Best MAE: {best_mae:.2f} | Best GAME(0): {best_game0:.2f}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()