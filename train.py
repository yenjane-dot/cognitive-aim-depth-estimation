"""Training script for Cognitive-Aim Experiment B reproduction.

Implements the complete training pipeline with four-layer cognitive architecture,
curiosity-driven learning, and progressive optimization strategies.
"""

import os
import argparse
import yaml
import random
import numpy as np
from typing import Dict, Tuple
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.model import create_model
from src.dataset import create_dataloaders, collate_fn
from src.utils import setup_logging, save_checkpoint, load_checkpoint


class ScaleInvariantLoss(nn.Module):
    """Scale-invariant logarithmic loss for depth estimation."""
    
    def __init__(self, lambda_reg: float = 0.5):
        super().__init__()
        self.lambda_reg = lambda_reg
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Convert to log space
        log_pred = torch.log(pred + 1e-8)
        log_target = torch.log(target + 1e-8)
        
        # Compute differences
        diff = log_pred - log_target
        
        # Scale-invariant loss
        n = diff.numel()
        loss = torch.sum(diff ** 2) / n - self.lambda_reg * (torch.sum(diff) ** 2) / (n ** 2)
        
        return loss


class CuriosityLoss(nn.Module):
    """Curiosity-driven loss for uncertainty estimation."""
    
    def __init__(self):
        super().__init__()
        
    def forward(self, curiosity_scores: torch.Tensor, prediction_error: torch.Tensor) -> torch.Tensor:
        # Encourage curiosity in regions with high prediction error
        target_curiosity = torch.sigmoid(prediction_error.detach())
        loss = nn.functional.mse_loss(curiosity_scores.mean(dim=1), target_curiosity)
        return loss


class Trainer:
    """Main trainer class for Experiment B."""
    
    def __init__(self, config: Dict, data_dir: str, output_dir: str):
        self.config = config
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Setup logging
        setup_logging(output_dir)
        self.logger = logging.getLogger(__name__)
        
        # Setup tensorboard
        self.writer = SummaryWriter(os.path.join(output_dir, 'tensorboard'))
        
        # Initialize model
        self.model = create_model(config).to(self.device)
        
        # Setup data loaders
        self.train_loader, self.val_loader = create_dataloaders(
            data_dir, config, 
            batch_size=config['training']['batch_size']
        )
        
        # Setup optimizer and scheduler
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Setup loss functions
        self.depth_loss = ScaleInvariantLoss(lambda_reg=config['training']['loss']['lambda'])
        self.curiosity_loss = CuriosityLoss()
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.curiosity_warmup_epochs = config['model']['curiosity']['warmup_epochs']
        
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer."""
        if self.config['training']['optimizer'] == 'AdamW':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['training']['optimizer']}")
            
    def _setup_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler."""
        if self.config['training']['scheduler'] == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=1e-6
            )
        else:
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
            
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_depth_loss = 0.0
        total_curiosity_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            images = batch['images'].to(self.device)
            depths = batch['depths'].to(self.device)
            exif_data = None
            if 'exif' in batch:
                exif_data = {k: v.to(self.device) for k, v in batch['exif'].items()}
                
            # Forward pass
            self.optimizer.zero_grad()
            
            # Get predictions
            pred_depths = self.model(images, exif_data)
            
            # Compute depth loss
            depth_loss = self.depth_loss(pred_depths.squeeze(), depths)
            
            # Compute curiosity loss (after warmup)
            curiosity_loss = torch.tensor(0.0, device=self.device)
            if self.epoch >= self.curiosity_warmup_epochs and self.model.curiosity:
                # Get curiosity scores from model
                with torch.no_grad():
                    _, patch_tokens = self.model.perception(images)
                    curiosity_scores = self.model.curiosity(patch_tokens)
                    
                # Compute prediction error for curiosity target
                pred_error = torch.abs(pred_depths.squeeze() - depths)
                curiosity_loss = self.curiosity_loss(curiosity_scores, pred_error)
                
            # Total loss
            total_batch_loss = depth_loss + 0.1 * curiosity_loss
            
            # Backward pass
            total_batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += total_batch_loss.item()
            total_depth_loss += depth_loss.item()
            total_curiosity_loss += curiosity_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{total_batch_loss.item():.4f}',
                'Depth': f'{depth_loss.item():.4f}',
                'Curiosity': f'{curiosity_loss.item():.4f}'
            })
            
            # Log to tensorboard
            global_step = self.epoch * len(self.train_loader) + batch_idx
            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('Train/BatchLoss', total_batch_loss.item(), global_step)
                self.writer.add_scalar('Train/DepthLoss', depth_loss.item(), global_step)
                self.writer.add_scalar('Train/CuriosityLoss', curiosity_loss.item(), global_step)
                
        avg_loss = total_loss / len(self.train_loader)
        avg_depth_loss = total_depth_loss / len(self.train_loader)
        avg_curiosity_loss = total_curiosity_loss / len(self.train_loader)
        
        return avg_loss, avg_depth_loss
        
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                images = batch['images'].to(self.device)
                depths = batch['depths'].to(self.device)
                exif_data = None
                if 'exif' in batch:
                    exif_data = {k: v.to(self.device) for k, v in batch['exif'].items()}
                    
                # Forward pass
                pred_depths = self.model(images, exif_data)
                
                # Compute loss
                loss = self.depth_loss(pred_depths.squeeze(), depths)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.val_loader)
        return avg_loss
        
    def save_predictions(self, epoch: int):
        """Save prediction visualizations."""
        if not self.config['logging']['visualize_predictions']:
            return
            
        self.model.eval()
        save_dir = os.path.join(self.output_dir, f'predictions_epoch_{epoch}')
        os.makedirs(save_dir, exist_ok=True)
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= 5:  # Save only first 5 batches
                    break
                    
                images = batch['images'].to(self.device)
                depths = batch['depths'].to(self.device)
                exif_data = None
                if 'exif' in batch:
                    exif_data = {k: v.to(self.device) for k, v in batch['exif'].items()}
                    
                pred_depths = self.model(images, exif_data)
                
                # Only generate prediction images, do not save npy files
        # Can add prediction image generation code here
                    
    def train(self):
        """Main training loop."""
        self.logger.info(f"Starting training for {self.config['training']['epochs']} epochs")
        
        for epoch in range(self.config['training']['epochs']):
            self.epoch = epoch
            
            # Train
            train_loss, train_depth_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Log metrics
            self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
            self.writer.add_scalar('Train/DepthLoss', train_depth_loss, epoch)
            self.writer.add_scalar('Val/Loss', val_loss, epoch)
            self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], epoch)
            
            self.logger.info(
                f"Epoch {epoch}: Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}"
            )
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                
            if epoch % self.config['logging']['save_interval'] == 0 or is_best:
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'config': self.config
                }, is_best, self.output_dir)
                
            # Save predictions
            if epoch % self.config['logging']['save_interval'] == 0:
                self.save_predictions(epoch)
                
        self.logger.info(f"Training completed. Best validation loss: {self.best_val_loss:.6f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Cognitive-Aim Experiment B Training')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Set random seeds for reproducibility
    if 'reproduction' in config and config['reproduction']['deterministic']:
        seed = config['reproduction']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize trainer
    trainer = Trainer(config, args.data_dir, args.output_dir)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.epoch = checkpoint['epoch'] + 1
        trainer.best_val_loss = checkpoint['best_val_loss']
        print(f"Resumed from epoch {checkpoint['epoch']}")
        
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()