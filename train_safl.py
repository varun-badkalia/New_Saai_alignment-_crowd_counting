import os
import argparse
import logging
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from datasets.broker_crowd_dataset import BrokerCrowdDataset, crowd_collate
from models.safl_crowd_counter import SAFLCrowdCounter
from utils.losses import SAFLCrowdLoss
from utils.metrics import calculate_game, calculate_mae_rmse

def get_args():
    parser = argparse.ArgumentParser(description='SAAI Only Training')
    parser.add_argument('--data-path', type=str, required=True, help='Path to dataset root')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--save-dir', type=str, default='./results_saai_baseline', help='Directory to save results')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--w-density', type=float, default=1.0, help='Weight for density loss')
    parser.add_argument('--w-regional', type=float, default=0.1, help='Weight for regional loss')
    parser.add_argument('--w-domain', type=float, default=0.1, help='Weight for domain loss')
    
    return parser.parse_args()

def validate(model, val_loader, device):
    model.eval()
    mae_sum, mse_sum = 0.0, 0.0
    game_sums = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
    n = 0
    with torch.no_grad():
        for batch in val_loader:
            if len(batch) == 5:
                 rgb, thermal, _, gt_count, name = batch
            else:
                 # fallback for safety
                 rgb, thermal, _, gt_count = batch[0], batch[1], batch[2], batch[3]

            rgb, thermal = rgb.to(device), thermal.to(device)
            gt_c = gt_count.item() if torch.is_tensor(gt_count) else gt_count
            
            pred_density, _, _ = model(rgb, thermal)
            pred_cnt = pred_density.sum().item()
            
            mae_sum += abs(pred_cnt - gt_c)
            mse_sum += (pred_cnt - gt_c) ** 2
            
            games = calculate_game(pred_density[0,0], gt_c)
            for k in games: game_sums[k] += games[k]
            n += 1
            
    return mae_sum/n, np.sqrt(mse_sum/n), {k: v/n for k,v in game_sums.items()}

def main():
    args = get_args()
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(filename=os.path.join(args.save_dir, 'train.log'), level=logging.INFO, 
                        format='%(asctime)s %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Data
    train_ds = BrokerCrowdDataset(os.path.join(args.data_path, 'train'), method='train')
    val_ds = BrokerCrowdDataset(os.path.join(args.data_path, 'val'), method='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=args.workers, collate_fn=crowd_collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, 
                            num_workers=args.workers, collate_fn=crowd_collate)
    
    # Model
    model = SAFLCrowdCounter().to(device)
    
    # Loss
    criterion = SAFLCrowdLoss(
        w_density=args.w_density,
        w_regional=args.w_regional,
        w_domain=args.w_domain
    ).to(device)
    
    # Optimizer (Different LRs for backbone vs head)
    optimizer = torch.optim.Adam([
        {'params': model.rgb_backbone.parameters(), 'lr': args.lr * 0.1},
        {'params': model.thermal_backbone.parameters(), 'lr': args.lr * 0.1},
        {'params': model.saai_aligner.parameters(), 'lr': args.lr},
        {'params': model.density_head.parameters(), 'lr': args.lr},
        {'params': model.fusion.parameters(), 'lr': args.lr}
    ], weight_decay=1e-4)
    
    best_mae = float('inf')
    
    logging.info("Starting SAAI Baseline Training...")
    for epoch in range(1, args.epochs+1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        epoch_loss = 0
        
        for batch in pbar:
            if len(batch) == 6:
                rgb, thermal, _, _, _, gt_density = batch
                gt_density = gt_density.to(device)
            else:
                rgb, thermal, _, _, _, = batch
                gt_density = None
                
            rgb, thermal = rgb.to(device), thermal.to(device)
            
            optimizer.zero_grad()
            pred, d_rgb, d_th = model(rgb, thermal)
            loss, loss_dict = criterion(pred, gt_density, d_rgb, d_th)
            
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
        logging.info(f"Epoch {epoch} Loss: {epoch_loss/len(train_loader):.4f}")

        # Validation
        if epoch % 1 == 0:
            mae, rmse, games = validate(model, val_loader, device)
            log_msg = f"Val: MAE={mae:.2f} | RMSE={rmse:.2f} | GAME0={games[0]:.2f} | GAME1={games[1]:.2f} | GAME2={games[2]:.2f} | GAME3={games[3]:.2f}"
            logging.info(log_msg)
            print(log_msg)
            
            if mae < best_mae:
                best_mae = mae
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
                logging.info("Saved Best Model!")

if __name__ == '__main__':
    main()