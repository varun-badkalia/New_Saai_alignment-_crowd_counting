import os
import argparse
import time
import logging
from datetime import datetime
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.broker_crowd_dataset import BrokerCrowdDataset, crowd_collate
from VarunPipeline.models.crowd_counter import SAAICrowdCounter
from utils.losses import SAAIBrokerLoss
from utils.metrics import calculate_mae, calculate_rmse

import numpy as np

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
    """pred_map numpy 2D array, gt_count integer"""
    h, w = pred_map.shape
    results = {}
    results[0] = abs(pred_map.sum() - float(gt_count))
    
    for level in [1, 2, 3]:
        S = 2 ** level
        total = 0.0
        for i in range(S):
            for j in range(S):
                h0 = i * h // S
                h1 = (i + 1) * h // S
                w0 = j * w // S
                w1 = (j + 1) * w // S
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
            rgb = rgb.to(device)
            thermal = thermal.to(device)

            # Extract gt_count first
            if isinstance(gt_count, torch.Tensor):
                gt_c = int(gt_count.item())
            elif isinstance(gt_count, (list, tuple, np.ndarray)):
                gt_c = int(gt_count[0])
            else:
                gt_c = int(gt_count)

            pred_density, _, _ = model(rgb, thermal)
            
            if n == 0:
                print(f"\n🔍 Validation Debug:")
                print(f"  Input RGB shape: {rgb.shape}")
                print(f"  Pred density shape: {pred_density.shape}")
                print(f"  Pred density sum: {pred_density.sum().item():.2f}")
                print(f"  Pred density min/max: {pred_density.min().item():.4f} / {pred_density.max().item():.4f}")
                print(f"  GT count: {gt_c}")
            
            # Upsample to original image size
            original_h, original_w = rgb.shape[2], rgb.shape[3]
            pred_density_upsampled = F.interpolate(
                pred_density, 
                size=(original_h, original_w),
                mode='bilinear', 
                align_corners=False
            )
            
            # Renormalize to preserve count
            original_count = pred_density.sum().item()
            upsampled_count = pred_density_upsampled.sum().item()
            if upsampled_count > 0:
                pred_density_upsampled = pred_density_upsampled * (original_count / upsampled_count)
            
            pred_map = pred_density_upsampled[0, 0].cpu().numpy()
            
            if n == 0:
                print(f"  Upsampled shape: {pred_map.shape}")
                print(f"  Upsampled pred sum: {pred_map.sum():.2f}")

            games = calculate_broker_game_single(pred_map, gt_c)
            for k in game_sums:
                game_sums[k] += games[k]

            pred_cnt = pred_map.sum()
            mse_acc += (pred_cnt - gt_c) ** 2
            n += 1

    for k in game_sums:
        game_sums[k] /= float(max(n, 1))
    rmse = float(np.sqrt(mse_acc / max(n, 1)))
    logger.info(f"VAL  GAME0:{game_sums[0]:.2f}  GAME1:{game_sums[1]:.2f}  GAME2:{game_sums[2]:.2f}  GAME3:{game_sums[3]:.2f}  RMSE:{rmse:.2f}")
    return game_sums, rmse

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data-path', required=True)
    p.add_argument('--save-dir', default='./Results/shared_enhanced')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--crop-size', type=int, default=384)
    p.add_argument('--downsample-ratio', type=int, default=32)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--enable-enhanced-loss', action='store_true')
    return p.parse_args()

def main():
    args = get_args()
    logger, log_file = setup_logging(args.save_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"🚀 Using device: {device}")
    logger.info(f"📝 Training log: {log_file}")

    train_root = os.path.join(args.data_path, 'train')
    val_root = os.path.join(args.data_path, 'val')

    logger.info("📂 Creating datasets...")
    train_set = BrokerCrowdDataset(train_root, crop_size=args.crop_size,
                                  downsample_ratio=args.downsample_ratio,
                                  method='train', enable_gt_density=args.enable_enhanced_loss)
    val_set = BrokerCrowdDataset(val_root, crop_size=args.crop_size,
                                downsample_ratio=args.downsample_ratio,
                                method='val', enable_gt_density=False)

    logger.info(f"   Train: {len(train_set)}  Val: {len(val_set)}  GT density: {'yes' if args.enable_enhanced_loss else 'no'}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, collate_fn=crowd_collate, 
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=max(0, args.workers//2), collate_fn=crowd_collate)

    model = SAAICrowdCounter(backbone_name='vgg16', pretrained=True, feature_dim=512).to(device)

    if hasattr(model, 'thermal_stem') and hasattr(model.thermal_stem, 'conv'):
        model.thermal_stem.conv = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1).to(device)

    backbone_params = []
    if hasattr(model, 'shared_backbone'):
        backbone_params += list(model.rgb_stem.parameters()) if hasattr(model, 'rgb_stem') else []
        backbone_params += list(model.thermal_stem.parameters()) if hasattr(model, 'thermal_stem') else []
        backbone_params += list(model.shared_backbone.parameters())

    saai_params = list(model.saai_aligner.parameters()) if hasattr(model, 'saai_aligner') else []
    head_params = list(model.fusion_module.parameters()) + list(model.regression_head.parameters())

    optimizer = torch.optim.Adam([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': saai_params, 'lr': args.lr},
        {'params': head_params, 'lr': args.lr},
    ], weight_decay=1e-4)

    criterion = SAAIBrokerLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.7)

    best_game0 = float('inf')

    logger.info("Start training...")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=120)
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
            
            total_loss, loss_dict = criterion(pred_density, kps_list, targets_list,
                                              domain_pred_rgb, domain_pred_thermal,
                                              gt_density_maps=gt_density)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += total_loss.item()
            n_batches += 1

            pbar.set_postfix({'Loss': f'{running_loss / n_batches:.4f}', 
                            'Den': f"{loss_dict['density_loss']:.4f}",
                            'Dom': f"{loss_dict['domain_loss']:.4f}", 
                            'Count': f"{loss_dict['count_loss']:.4f}"})

        scheduler.step()
        epoch_time = time.time() - t0
        logger.info(f"Epoch {epoch} finished in {epoch_time:.1f}s. Train loss: {running_loss / max(1, n_batches):.6f}")

        if epoch % 5 == 0 or epoch == 1:
            game_results, rmse = validate(model, val_loader, device, logger)
            game0 = game_results[0]
            if game0 < best_game0:
                best_game0 = game0
                ckpt_path = os.path.join(args.save_dir, f'best_game0_epoch{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'game0': game0,
                }, ckpt_path)
                logger.info(f"✔ Saved best checkpoint to {ckpt_path}")

    logger.info("Training completed.")

if __name__ == "__main__":
    main()