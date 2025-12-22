# train_saai.py
"""
Training script for simple shared-backbone crowd counter
"""

import os
import argparse
import logging
import numpy as np
from datetime import datetime
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from datasets.broker_crowd_dataset import BrokerCrowdDataset, crowd_collate
from models.crowd_counter import SAAICrowdCounter
from utils.losses import SAAIBrokerLoss
from utils.visualization import visualize_saai_alignment


def setup_logging(save_dir):
    os.makedirs(save_dir, exist_ok=True)
    log_filename = os.path.join(save_dir, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_filename)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    return logger, log_filename


def calculate_broker_game_single(pred_map, gt_count):
    h, w = pred_map.shape
    results = {}
    results[0] = abs(pred_map.sum() - float(gt_count))
    
    for level in [1, 2, 3]:
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


def validate(model, val_loader, device, logger):
    model.eval()
    game_sums = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    mse_acc = 0.0
    n = 0
    
    with torch.no_grad():
        for batch in val_loader:
            rgb, thermal, _, gt_count, name = batch
            rgb, thermal = rgb.to(device), thermal.to(device)
            
            pred_density, _, _ = model(rgb, thermal)
            pred_map = pred_density[0, 0].cpu().numpy()
            
            gt_c = int(gt_count.item()) if isinstance(gt_count, torch.Tensor) else int(gt_count)
            
            games = calculate_broker_game_single(pred_map, gt_c)
            for k in game_sums:
                game_sums[k] += games[k]
            
            pred_cnt = pred_map.sum()
            mse_acc += (pred_cnt - gt_c) ** 2
            n += 1
    
    for k in game_sums:
        game_sums[k] /= float(n)
    rmse = float(np.sqrt(mse_acc / n))
    
    return game_sums, rmse


def visualize_alignment_sample(model, val_loader, device, save_dir, epoch):
    model.eval()
    
    for batch in val_loader:
        rgb, thermal, _, gt_count, name = batch
        rgb, thermal = rgb.to(device), thermal.to(device)
        
        save_path = os.path.join(save_dir, f'alignment_epoch{epoch:03d}_{name}.png')
        metrics = visualize_saai_alignment(model, rgb, thermal, save_path)
        break
    
    return metrics


def get_args():
    parser = argparse.ArgumentParser(description='Simple Crowd Counter Training')
    
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--crop-size', type=int, default=384)
    parser.add_argument('--downsample-ratio', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--enable-gt-density', action='store_true')
    parser.add_argument('--save-dir', default='./results_simple')
    parser.add_argument('--val-freq', type=int, default=2)
    parser.add_argument('--vis-freq', type=int, default=10)
    
    return parser.parse_args()


def main():
    args = get_args()
    logger, log_file = setup_logging(args.save_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info("=" * 60)
    logger.info("🚀 Simple Shared-Backbone Crowd Counter Training")
    logger.info(f"Device: {device}")
    logger.info(f"Downsample: {args.downsample_ratio}×")
    logger.info(f"GT Density: {'ENABLED ✓' if args.enable_gt_density else 'DISABLED'}")
    logger.info(f"Architecture: Shared VGG16 backbone (NO SAAI)")
    logger.info("=" * 60)
    
    train_root = os.path.join(args.data_path, 'train')
    val_root = os.path.join(args.data_path, 'val')
    
    train_set = BrokerCrowdDataset(
        train_root, crop_size=args.crop_size,
        downsample_ratio=args.downsample_ratio,
        method='train', enable_gt_density=args.enable_gt_density
    )
    
    val_set = BrokerCrowdDataset(
        val_root, crop_size=args.crop_size,
        downsample_ratio=args.downsample_ratio,
        method='val', enable_gt_density=False
    )
    
    logger.info(f"Train: {len(train_set)} | Val: {len(val_set)}")
    
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, collate_fn=crowd_collate,
        pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False,
        num_workers=max(0, args.workers // 2), collate_fn=crowd_collate
    )
    
    # Create simple model
    model = SAAICrowdCounter(
        backbone_name='vgg16',
        pretrained=True,
        feature_dim=512
    ).to(device)
    
    # SIMPLIFIED optimizer - ONE learning rate for everything
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    logger.info(f"Optimizer: Adam with LR={args.lr:.2e} (same for all layers)")
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[50, 100, 150], gamma=0.5
    )
    
    # Simple loss function
    criterion = SAAIBrokerLoss(alpha=0.0, beta=0.01, gamma=10.0)
    logger.info("Loss: Density MSE (10.0×) + Count L1 (0.01×)")
    
    best_game0 = float('inf')
    best_game2 = float('inf')
    best_rmse = float('inf')
    
    logger.info("Starting training...")
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_density = 0.0
        epoch_count = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            if len(batch) == 6:
                rgb, thermal, kps_list, targets_list, st_sizes, gt_density = batch
                gt_density = gt_density.to(device)
            else:
                rgb, thermal, kps_list, targets_list, st_sizes = batch
                gt_density = None
            
            rgb, thermal = rgb.to(device), thermal.to(device)
            
            optimizer.zero_grad()
            pred_density, _, _ = model(rgb, thermal)
            
            loss, loss_dict = criterion(
                pred_density, kps_list, targets_list,
                None, None, gt_density_maps=gt_density
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_density += loss_dict['density_loss']
            epoch_count += loss_dict['count_loss']
            n_batches += 1
            
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Den': f"{loss_dict['density_loss']:.4f}",
                'Cnt': f"{loss_dict['count_loss']:.2f}"
            })
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(
            f"Epoch {epoch} | "
            f"Loss: {epoch_loss/n_batches:.4f} | "
            f"Density: {epoch_density/n_batches:.4f} | "
            f"Count: {epoch_count/n_batches:.4f} | "
            f"LR: {current_lr:.2e}"
        )
        
        # Validation
        if epoch % args.val_freq == 0 or epoch == 1:
            logger.info("Running validation...")
            game_results, rmse = validate(model, val_loader, device, logger)
            
            logger.info("🔍 Validation Results:")
            logger.info(f"   GAME(0): {game_results[0]:.2f}")
            logger.info(f"   GAME(1): {game_results[1]:.2f}")
            logger.info(f"   GAME(2): {game_results[2]:.2f}")
            logger.info(f"   GAME(3): {game_results[3]:.2f}")
            logger.info(f"   RMSE:    {rmse:.2f}")
            
            # Visualize
            if epoch % args.vis_freq == 0 or epoch == 1:
                try:
                    logger.info("📸 Creating visualization...")
                    align_metrics = visualize_alignment_sample(
                        model, val_loader, device, args.save_dir, epoch
                    )
                except Exception as e:
                    logger.warning(f"   ⚠️ Visualization failed: {e}")
            
            # Save best checkpoints
            if game_results[0] < best_game0:
                best_game0 = game_results[0]
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'game0': best_game0,
                    'game_results': game_results
                }, os.path.join(args.save_dir, 'best_game0.pth'))
                logger.info(f"✅ Saved best GAME0: {best_game0:.2f}")
            
            if game_results[2] < best_game2:
                best_game2 = game_results[2]
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'game2': best_game2
                }, os.path.join(args.save_dir, 'best_game2.pth'))
                logger.info(f"✅ Saved best GAME2: {best_game2:.2f}")
            
            if rmse < best_rmse:
                best_rmse = rmse
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'rmse': best_rmse
                }, os.path.join(args.save_dir, 'best_rmse.pth'))
                logger.info(f"✅ Saved best RMSE: {best_rmse:.2f}")
    
    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best GAME0: {best_game0:.2f}")
    logger.info(f"Best GAME2: {best_game2:.2f}")
    logger.info(f"Best RMSE: {best_rmse:.2f}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()