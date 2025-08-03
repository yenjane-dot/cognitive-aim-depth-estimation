"""Utility functions for Cognitive-Aim Experiment B reproduction.

Includes logging setup, checkpoint management, metrics calculation, and visualization.
"""

import os
import logging
import shutil
from typing import Dict, Any, Tuple
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch


def setup_logging(output_dir: str, log_level: int = logging.INFO):
    """Setup logging configuration."""
    log_file = os.path.join(output_dir, 'training.log')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Prevent duplicate logs
    root_logger.propagate = False


def save_checkpoint(
    state: Dict[str, Any], 
    is_best: bool, 
    output_dir: str, 
    filename: str = 'checkpoint.pth'
):
    """Save model checkpoint."""
    checkpoint_path = os.path.join(output_dir, filename)
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(output_dir, 'model_best.pth')
        shutil.copyfile(checkpoint_path, best_path)
        
    # Also save epoch-specific checkpoint
    epoch_path = os.path.join(output_dir, f'checkpoint_epoch_{state["epoch"]}.pth')
    torch.save(state, epoch_path)


def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """Load model checkpoint."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def calculate_depth_metrics(
    pred: torch.Tensor, 
    target: torch.Tensor, 
    mask: torch.Tensor = None
) -> Dict[str, float]:
    """Calculate standard depth estimation metrics."""
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    else:
        pred = pred.flatten()
        target = target.flatten()
        
    # Remove invalid values
    valid_mask = (target > 0) & (pred > 0) & torch.isfinite(pred) & torch.isfinite(target)
    pred = pred[valid_mask]
    target = target[valid_mask]
    
    if len(pred) == 0:
        return {
            'rmse': float('inf'),
            'mae': float('inf'),
            'delta_1': 0.0,
            'delta_2': 0.0,
            'delta_3': 0.0,
            'rel_error': float('inf'),
            'log_error': float('inf')
        }
    
    # Convert to numpy for calculations
    pred_np = pred.cpu().numpy()
    target_np = target.cpu().numpy()
    
    # RMSE
    rmse = np.sqrt(np.mean((pred_np - target_np) ** 2))
    
    # MAE
    mae = np.mean(np.abs(pred_np - target_np))
    
    # Relative error
    rel_error = np.mean(np.abs(pred_np - target_np) / target_np)
    
    # Log error
    log_error = np.mean(np.abs(np.log(pred_np + 1e-8) - np.log(target_np + 1e-8)))
    
    # Delta metrics
    ratio = np.maximum(pred_np / target_np, target_np / pred_np)
    delta_1 = np.mean(ratio < 1.25)
    delta_2 = np.mean(ratio < 1.25 ** 2)
    delta_3 = np.mean(ratio < 1.25 ** 3)
    
    return {
        'rmse': float(rmse),
        'mae': float(mae),
        'delta_1': float(delta_1),
        'delta_2': float(delta_2),
        'delta_3': float(delta_3),
        'rel_error': float(rel_error),
        'log_error': float(log_error)
    }


def visualize_depth_prediction(
    image: torch.Tensor,
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    save_path: str = None
) -> plt.Figure:
    """Visualize depth prediction results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Convert tensors to numpy
    if isinstance(image, torch.Tensor):
        # Denormalize image
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        image_np = image.permute(1, 2, 0).cpu().numpy()
    else:
        image_np = image
        
    pred_np = pred_depth.cpu().numpy() if isinstance(pred_depth, torch.Tensor) else pred_depth
    gt_np = gt_depth.cpu().numpy() if isinstance(gt_depth, torch.Tensor) else gt_depth
    
    # Plot image
    axes[0].imshow(image_np)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Plot predicted depth
    im1 = axes[1].imshow(pred_np, cmap='plasma')
    axes[1].set_title('Predicted Depth')
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    
    # Plot ground truth depth
    im2 = axes[2].imshow(gt_np, cmap='plasma')
    axes[2].set_title('Ground Truth Depth')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
    return fig


# Depth color functionality removed


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_model_size(model: torch.nn.Module) -> float:
    """Get model size in MB."""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def create_training_summary(config: Dict, model: torch.nn.Module, output_dir: str):
    """Create training summary report."""
    total_params, trainable_params = count_parameters(model)
    model_size = get_model_size(model)
    
    summary = f"""
# Cognitive-Aim Experiment B Training Summary

## Model Configuration
- Architecture: Four-layer Cognitive Architecture
- Backbone: {config['model']['backbone']}
- LoRA Enabled: {config['model']['perception_foundation']['enable_lora']}
- Curiosity Mechanism: {config['model']['curiosity']['enable']}
- Iterations: {config['model']['focal_cognition']['iterations']}

## Model Statistics
- Total Parameters: {total_params:,}
- Trainable Parameters: {trainable_params:,}
- Model Size: {model_size:.2f} MB

## Training Configuration
- Epochs: {config['training']['epochs']}
- Batch Size: {config['training']['batch_size']}
- Learning Rate: {config['training']['learning_rate']}
- Optimizer: {config['training']['optimizer']}
- Scheduler: {config['training']['scheduler']}

## Data Configuration
- Image Size: {config['dataset']['image_size']}
- Use EXIF: {config['dataset']['use_exif']}
- Augmentation: {config['training']['augmentation']['enable']}

## Reproduction Settings
- Seed: {config.get('reproduction', {}).get('seed', 'Not set')}
- Deterministic: {config.get('reproduction', {}).get('deterministic', False)}
"""
    
    with open(os.path.join(output_dir, 'training_summary.md'), 'w') as f:
        f.write(summary)
        
    return summary


def validate_config(config: Dict) -> bool:
    """Validate configuration file."""
    required_keys = [
        'model', 'training', 'dataset', 'logging'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
            
    # Validate model config
    model_config = config['model']
    required_model_keys = [
        'perception_foundation', 'ambient_cognition', 
        'focal_cognition', 'experience_integration', 'curiosity'
    ]
    
    for key in required_model_keys:
        if key not in model_config:
            raise ValueError(f"Missing required model config key: {key}")
            
    return True


def setup_experiment_directory(output_dir: str, config: Dict) -> str:
    """Setup experiment directory structure."""
    # Create subdirectories
    subdirs = ['checkpoints', 'logs', 'predictions', 'tensorboard']
    for subdir in subdirs:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
        
    # Save config
    import yaml
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
        
    return output_dir