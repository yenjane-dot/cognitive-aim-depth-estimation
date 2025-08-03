"""Inference script for Cognitive-Aim Experiment B reproduction.

Implements pure inference pipeline with four-layer cognitive architecture,
curiosity-driven processing, and depth estimation.
"""

import os
import argparse
import yaml
import numpy as np
from typing import Dict, Optional, Tuple
import logging
from pathlib import Path
from scipy.ndimage import zoom

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

from src.model import create_model
from src.utils import setup_logging


class CognitiveAimInference:
    """Main inference class for Cognitive-Aim model."""
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = 'auto'):
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        print(f"Using device: {self.device}")
        
        # Load configuration
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # Enable debug output for model initialization
        self.config['debug_init'] = True
        
        # Ensure cognitive modules are properly configured
        if 'cognitive_modules' not in self.config:
            print("Warning: cognitive_modules not found in config, using default settings")
            self.config['cognitive_modules'] = [
                'ambient_stream',
                'iterative_focal_stream', 
                'exif_prior_database'
            ]
        print(f"Default cognitive modules set: {self.config['cognitive_modules']}")
        
        # Setup camera info for EXIF processing
        camera_info = None
        if 'exif_prior_database' in self.config.get('cognitive_modules', []):
            num_cameras = 71  # default
            if 'exif_config' in self.config:
                num_cameras = self.config['exif_config'].get('num_cameras', 71)
            elif 'model' in self.config and 'exif_config' in self.config['model']:
                num_cameras = self.config['model']['exif_config'].get('num_cameras', 71)
            
            camera_info = {'num_cameras': num_cameras}
            print(f"EXIF camera info: {num_cameras} camera models")
        
        # Initialize model
        self.model = create_model(self.config, camera_info).to(self.device)
        
        # Print model status
        print(f"Model status: use_ambient={getattr(self.model, 'use_ambient', False)}, "
              f"use_focal={getattr(self.model, 'use_focal', False)}, "
              f"use_exif={getattr(self.model, 'use_exif', False)}")
        
        # Load checkpoint
        self._load_checkpoint(checkpoint_path)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Setup image transforms
        self._setup_transforms()
        
        # Setup camera mapping for EXIF
        self.camera_to_id = {'unknown': 0}  # Simplified mapping
        
        print("Model initialization complete, ready for inference")
        
    def _load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint with compatibility handling."""
        print(f"Loading checkpoint: {checkpoint_path}")
        
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            
            # Try different checkpoint formats
            model_state = None
            if 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
                print("Detected standard checkpoint format (model_state_dict)")
            elif 'main_model' in checkpoint:
                model_state = checkpoint['main_model']
                print("Detected main model format (main_model)")
            else:
                model_state = checkpoint
                print("Detected direct state dict format")
                
            # Filter compatible parameters
            filtered_state = {}
            model_dict = self.model.state_dict()
            
            loaded_count = 0
            skipped_count = 0
            
            for key, value in model_state.items():
                if key in model_dict:
                    # Check if dimensions match
                    if model_dict[key].shape == value.shape:
                        filtered_state[key] = value
                        loaded_count += 1
                    else:
                        print(f"Skipping parameter {key}: dimension mismatch ({value.shape} vs {model_dict[key].shape})")
                        skipped_count += 1
                else:
                    print(f"Skipping parameter {key}: not found in current model")
                    skipped_count += 1
            
            # Load compatible parameters
            self.model.load_state_dict(filtered_state, strict=False)
            print(f"Successfully loaded {loaded_count}/{len(model_state)} parameters")
            
            if skipped_count > 0:
                print(f"Skipped {skipped_count} incompatible parameters")
            
            # Initialize missing parameters
            missing_keys = set(model_dict.keys()) - set(filtered_state.keys())
            if missing_keys:
                print(f"Initialized {len(missing_keys)} missing parameters with default values")
                for key in missing_keys:
                    if 'weight' in key:
                        if len(model_dict[key].shape) > 1:
                            torch.nn.init.xavier_uniform_(model_dict[key])
                        else:
                            torch.nn.init.normal_(model_dict[key], 0, 0.01)
                    elif 'bias' in key:
                        torch.nn.init.zeros_(model_dict[key])
                        
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Continuing with randomly initialized weights...")
            
    def _setup_transforms(self):
        """Setup image preprocessing transforms."""
        image_size = self.config['dataset']['image_size']
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        elif isinstance(image_size, list) and len(image_size) == 1:
            image_size = (image_size[0], image_size[0])
        else:
            image_size = tuple(image_size)
            
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Image preprocessing setup: size={image_size}")
        
    def _extract_exif_data(self, image_path: str) -> Optional[Dict]:
        """Extract EXIF data from image with robust error handling."""
        try:
            from PIL import Image
            from PIL.ExifTags import TAGS
            import warnings
            
            # 忽略PIL的EXIF警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                image = Image.open(image_path)
                
                # 尝试获取EXIF数据，处理损坏的情况
                try:
                    exif_data = image._getexif()
                except Exception:
                    # 如果_getexif()失败，尝试使用getexif()方法
                    try:
                        exif_data = image.getexif()
                        if exif_data:
                            # 转换为字典格式
                            exif_dict = {}
                            for tag_id, value in exif_data.items():
                                exif_dict[tag_id] = value
                            exif_data = exif_dict
                    except Exception:
                        exif_data = None
                
                if exif_data is None:
                    return None
                    
                # 提取核心EXIF信息：焦距、光圈、曝光时间、ISO
                extracted = {}
                
                def safe_extract_value(value):
                    """安全提取EXIF值，处理分数格式"""
                    try:
                        if isinstance(value, tuple) and len(value) == 2:
                            return float(value[0]) / float(value[1])
                        elif isinstance(value, (int, float)):
                            return float(value)
                        else:
                            return float(str(value))
                    except:
                        return None
                
                for tag_id, value in exif_data.items():
                    try:
                        tag = TAGS.get(tag_id, tag_id)
                        
                        if tag == 'FocalLength':
                            focal_length = safe_extract_value(value)
                            if focal_length and 10 <= focal_length <= 500:  # 合理范围检查
                                extracted['FocalLength'] = focal_length
                                
                        elif tag == 'FNumber':
                            aperture = safe_extract_value(value)
                            if aperture and 1.0 <= aperture <= 32.0:  # 合理范围检查
                                extracted['FNumber'] = aperture
                                
                        elif tag == 'ExposureTime':
                            exposure = safe_extract_value(value)
                            if exposure and 0.0001 <= exposure <= 30.0:  # 合理范围检查
                                extracted['ExposureTime'] = exposure
                                
                        elif tag == 'ISOSpeedRatings':
                            iso = safe_extract_value(value)
                            if iso and 50 <= iso <= 25600:  # 合理范围检查
                                extracted['ISOSpeedRatings'] = int(iso)
                                
                        elif tag in ['Make', 'Model']:
                            if isinstance(value, str) and value.strip():
                                extracted[tag] = value.strip()
                                
                    except Exception:
                        # 忽略单个标签的提取错误，继续处理其他标签
                        continue
                        
                return extracted if extracted else None
                
        except Exception as e:
            # 只在完全无法读取时显示错误
            if "Corrupt EXIF" not in str(e):
                print(f"EXIF提取失败: {e}")
            return None
            
    def _process_exif_for_model(self, exif_data: Optional[Dict]) -> Dict:
        """Process EXIF data for model input. Always returns valid data - uses defaults when EXIF is missing."""
        try:
            # 如果有EXIF数据，优先使用真实值；如果没有，使用默认值
            if exif_data is not None:
                # 使用真实EXIF数据
                processed = {
                    'focal_length': torch.tensor([exif_data.get('FocalLength', 50.0)], dtype=torch.float32),
                    'aperture': torch.tensor([exif_data.get('FNumber', 2.8)], dtype=torch.float32),
                    'iso': torch.tensor([exif_data.get('ISOSpeedRatings', 100)], dtype=torch.float32),
                    'camera_idx': torch.tensor([self.camera_to_id.get(exif_data.get('Model', 'unknown'), 0)], dtype=torch.long)
                }
            else:
                # 没有EXIF数据时，使用合理的默认值
                print("No EXIF data detected, using default camera parameters")
                processed = {
                    'focal_length': torch.tensor([50.0], dtype=torch.float32),  # 标准镜头焦距
                    'aperture': torch.tensor([2.8], dtype=torch.float32),       # 常见光圈值
                    'iso': torch.tensor([100], dtype=torch.float32),            # 低ISO值
                    'camera_idx': torch.tensor([0], dtype=torch.long)           # 默认相机索引
                }
            
            # Move to device
            for key, value in processed.items():
                processed[key] = value.to(self.device)
                
            return processed
            
        except Exception as e:
            print(f"EXIF processing failed, using default values: {e}")
            # 异常情况下也返回默认值
            processed = {
                'focal_length': torch.tensor([50.0], dtype=torch.float32),
                'aperture': torch.tensor([2.8], dtype=torch.float32),
                'iso': torch.tensor([100], dtype=torch.float32),
                'camera_idx': torch.tensor([0], dtype=torch.long)
            }
            for key, value in processed.items():
                processed[key] = value.to(self.device)
            return processed
            
    def predict(self, image_path: str, instruction: Optional[str] = None) -> Tuple[float, float, Dict]:
        """Perform depth prediction on a single image.
        
        Args:
            image_path: Path to input image
            instruction: Optional instruction for guided attention (e.g., 'center', 'left', 'right')
            
        Returns:
            Tuple of (depth_value, confidence_score, metadata)
        """
        print(f"\nStarting inference: {image_path}")
        
        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            print(f"Original image size: {original_size}")
        except Exception as e:
            raise ValueError(f"Cannot load image {image_path}: {e}")
            
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        print(f"Preprocessed tensor size: {image_tensor.shape}")
        
        # Extract EXIF data
        exif_raw = self._extract_exif_data(image_path)
        exif_data = self._process_exif_for_model(exif_raw)  # 现在总是返回有效数据
        
        if exif_raw is not None:
            print(f"EXIF data: focal_length={exif_raw.get('FocalLength', 'N/A')}, "
                  f"aperture={exif_raw.get('FNumber', 'N/A')}, "
                  f"ISO={exif_raw.get('ISOSpeedRatings', 'N/A')}")
        else:
            print("No EXIF data detected, using default parameters: focal_length=50.0, aperture=2.8, ISO=100")
            
        # 清除之前的注意力权重
        if hasattr(self.model, '_last_attention_weights'):
            delattr(self.model, '_last_attention_weights')
        
        # Perform inference
        with torch.no_grad():
            try:
                if instruction is not None:
                    # Use guided inference if instruction is provided
                    print(f"Using guided inference: {instruction}")
                    if hasattr(self.model, 'forward_with_guidance'):
                        depth_pred, confidence = self.model.forward_with_guidance(
                            image_tensor, exif_data, instruction
                        )
                    else:
                        print("Model does not support guided inference, using standard inference")
                        depth_pred, confidence = self.model(image_tensor, exif_data)
                else:
                    # Standard inference
                    depth_pred, confidence = self.model(image_tensor, exif_data)
                    
                # Extract scalar values
                depth_value = depth_pred.squeeze().cpu().item()
                confidence_score = confidence.squeeze().cpu().item() if confidence is not None else 1.0
                
                print(f"Depth prediction: {depth_value:.4f}")
                print(f"Confidence: {confidence_score:.4f}")
                
                # 生成prediction图像（包含注意力热力图和置信度可视化）
                attention_weights = None
                if hasattr(self.model, 'get_attention_weights'):
                    try:
                        attention_weights = self.model.get_attention_weights()
                    except:
                        attention_weights = None
                self._save_prediction_image(image_path, image_tensor, depth_value, confidence_score, attention_weights, instruction)
                
                # Prepare metadata
                metadata = {
                    'image_path': image_path,
                    'original_size': original_size,
                    'processed_size': tuple(image_tensor.shape[2:]),
                    'exif_available': exif_raw is not None,  # 基于原始EXIF数据是否存在
                    'exif_source': 'real' if exif_raw is not None else 'default',  # 标记EXIF数据来源
                    'instruction': instruction,
                    'cognitive_modules': self.config.get('cognitive_modules', []),
                    'model_status': {
                        'ambient': getattr(self.model, 'use_ambient', False),
                        'focal': getattr(self.model, 'use_focal', False),
                        'exif': getattr(self.model, 'use_exif', False)
                    }
                }
                
                if exif_raw:
                    metadata['exif_data'] = exif_raw
                else:
                    # 记录使用的默认值
                    metadata['exif_data'] = {
                        'FocalLength': 50.0,
                        'FNumber': 2.8,
                        'ISOSpeedRatings': 100,
                        'Model': 'Default',
                        'Make': 'Default'
                    }
                    
                return depth_value, confidence_score, metadata
                
            except Exception as e:
                print(f"Error during inference: {e}")
                import traceback
                traceback.print_exc()
                raise
                
    def predict_batch(self, image_paths: list, instructions: Optional[list] = None) -> list:
        """Perform batch prediction on multiple images.
        
        Args:
            image_paths: List of image paths
            instructions: Optional list of instructions for each image
            
        Returns:
            List of (depth_value, confidence_score, metadata) tuples
        """
        results = []
        
        if instructions is None:
            instructions = [None] * len(image_paths)
        elif len(instructions) != len(image_paths):
            raise ValueError("Number of instructions must match number of images")
            
        for i, (image_path, instruction) in enumerate(zip(image_paths, instructions)):
            print(f"\nProcessing batch {i+1}/{len(image_paths)}")
            try:
                result = self.predict(image_path, instruction)
                results.append(result)
            except Exception as e:
                print(f"Error processing image {image_path}: {e}")
                results.append((None, None, {'error': str(e)}))
                
        return results
        
    def save_results(self, results: list, output_path: str):
        """Save prediction results to file."""
        import json
        
        def make_serializable(obj):
            """Convert non-serializable objects to serializable format."""
            if hasattr(obj, '__dict__'):
                return str(obj)
            elif hasattr(obj, 'numerator') and hasattr(obj, 'denominator'):
                # Handle IFDRational and similar fraction types
                return float(obj.numerator) / float(obj.denominator)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(item) for item in obj]
            else:
                try:
                    json.dumps(obj)
                    return obj
                except (TypeError, ValueError):
                    return str(obj)
        
        # Convert results to serializable format
        serializable_results = []
        for depth, confidence, metadata in results:
            serializable_metadata = make_serializable(metadata)
            serializable_results.append({
                'depth_value': float(depth) if depth is not None else None,
                'confidence_score': float(confidence) if confidence is not None else None,
                'metadata': serializable_metadata
            })
            
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            
        print(f"Results saved to: {output_path}")
    
    def _save_prediction_image(self, image_path: str, image_tensor: torch.Tensor, depth_value: float, confidence_score: float, attention_weights: Optional[torch.Tensor] = None, instruction: Optional[str] = None):
        """生成并保存prediction图像"""
        import matplotlib.pyplot as plt
        from pathlib import Path
        
        # 创建输出目录
        output_dir = Path('demo_results')
        output_dir.mkdir(exist_ok=True)
        
        # 生成输出文件名
        image_name = Path(image_path).stem
        if instruction:
            output_name = f"{image_name}_{instruction}_prediction.png"
        else:
            output_name = f"{image_name}_prediction.png"
        output_path = output_dir / output_name
        
        # 反归一化图像
        device = image_tensor.device
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)
        image_denorm = image_tensor.squeeze(0) * std + mean
        image_denorm = torch.clamp(image_denorm, 0, 1)
        image_np = image_denorm.permute(1, 2, 0).cpu().numpy()
        
        # 创建三列布局
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 第一列：显示原图
        axes[0].imshow(image_np)
        axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 第二列：显示注意力热力图叠加在原图上
        if attention_weights is not None:
            # 处理注意力权重，确保正确提取空间信息
            if attention_weights.dim() == 3:  # [batch, num_patches, dim]
                attn_map = attention_weights[0]  # 取第一个batch
                if attn_map.shape[-1] > 1:
                    # 如果有多个维度，取平均值而不是只取第一个维度
                    attn_map = attn_map.mean(dim=-1)  # 对最后一个维度求平均
                else:
                    attn_map = attn_map.squeeze(-1)  # 移除最后一个维度
            elif attention_weights.dim() == 2:  # [batch, num_patches] 或 [num_patches, dim]
                if attention_weights.shape[0] == 1:  # [1, num_patches]
                    attn_map = attention_weights[0]
                else:  # [num_patches, dim] - 这种情况下取平均值
                    if attention_weights.shape[1] > 1:
                        attn_map = attention_weights.mean(dim=-1)  # 对最后一个维度求平均
                    else:
                        attn_map = attention_weights.squeeze(-1)
            else:  # 1D tensor
                attn_map = attention_weights
            
            # 确保是1D tensor
            if attn_map.dim() > 1:
                attn_map = attn_map.flatten()
            
            # 增强对比度，突出聚焦区域
            attn_map = attn_map.cpu().numpy()
            
            # 使用更强的非线性变换增强对比度
            attn_map = np.power(attn_map, 3)  # 立方变换，更强地突出高值区域
            
            # 应用阈值处理，进一步增强对比度
            threshold = np.percentile(attn_map, 70)  # 取70%分位数作为阈值
            attn_map = np.where(attn_map > threshold, attn_map, attn_map * 0.3)  # 低于阈值的值减弱
            
            # 重新归一化
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
            
            # 重塑为正方形网格
            num_patches = len(attn_map)
            patch_size = int(np.sqrt(num_patches))
            if patch_size * patch_size == num_patches:
                attn_map_2d = attn_map.reshape(patch_size, patch_size)
            else:
                # 如果不是完全平方数，填充到最接近的正方形
                target_size = int(np.ceil(np.sqrt(num_patches)))
                padded_map = np.zeros(target_size * target_size)
                padded_map[:num_patches] = attn_map
                attn_map_2d = padded_map.reshape(target_size, target_size)
            
            # 先显示原图作为底图
            axes[1].imshow(image_np)
            
            # 将注意力热力图调整到与原图相同尺寸
            h, w = image_np.shape[:2]
            # 计算缩放比例
            scale_h = h / attn_map_2d.shape[0]
            scale_w = w / attn_map_2d.shape[1]
            # 使用双线性插值调整热力图尺寸
            attn_resized = zoom(attn_map_2d, (scale_h, scale_w), order=1)
            
            # 叠加热力图，使用透明度
            im = axes[1].imshow(attn_resized, cmap='plasma', alpha=0.6, interpolation='bilinear', vmin=0, vmax=1)
            axes[1].set_title('Focus Map (Overlay)', fontsize=14, fontweight='bold')
            axes[1].axis('off')
            
            # 添加颜色条
            cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            cbar.set_label('Attention Weight', rotation=270, labelpad=15)
        else:
            # 如果没有注意力权重，显示原图
            axes[1].imshow(image_np)
            axes[1].set_title('No Attention Data', fontsize=14)
            axes[1].axis('off')
        
        # 第三列：深度估计数值区
        result_ax = axes[2]
        
        # 显示深度预测和置信度数据
        result_ax.text(0.5, 0.7, f'Predicted Depth: {depth_value:.4f}m', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=result_ax.transAxes, fontsize=16, fontweight='bold')
        result_ax.text(0.5, 0.5, f'Confidence: {confidence_score:.4f}', 
                      horizontalalignment='center', verticalalignment='center',
                      transform=result_ax.transAxes, fontsize=14, color='blue')
        if instruction:
            result_ax.text(0.5, 0.3, f'Instruction: {instruction}', 
                          horizontalalignment='center', verticalalignment='center',
                          transform=result_ax.transAxes, fontsize=12)
        result_ax.set_title('Prediction Results')
        result_ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Prediction image saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Cognitive-Aim Experiment B Inference')
    parser.add_argument('--config', type=str, default='configs/experiment_B.yaml', help='Configuration file path')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/cognitive_aim_model.pth', help='Model checkpoint path')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--image_dir', type=str, help='Image directory path')
    parser.add_argument('--instruction', type=str, help='Guidance instruction (center/left/right)')
    parser.add_argument('--output', type=str, default='inference_results.json', help='Output file path')
    parser.add_argument('--device', type=str, default='auto', help='Computing device (auto/cpu/cuda)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.image and not args.image_dir:
        raise ValueError("Must specify either --image or --image_dir")
        
    if args.image and args.image_dir:
        raise ValueError("Cannot specify both --image and --image_dir")
        
    # Initialize inference engine
    print("Initializing Cognitive-Aim inference engine...")
    inference_engine = CognitiveAimInference(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device
    )
    
    # Perform inference
    if args.image:
        # Single image inference
        print(f"\nSingle image inference mode")
        # 如果没有提供指令，默认使用'center'
        instruction = args.instruction if args.instruction else 'center'
        result = inference_engine.predict(args.image, instruction)
        results = [result]
        
        # Print results
        depth, confidence, metadata = result
        print(f"\n=== Inference Results ===")
        print(f"Image: {args.image}")
        print(f"Depth value: {depth:.4f}")
        print(f"Confidence: {confidence:.4f}")
        print(f"Guidance instruction: {instruction}")
        print(f"Cognitive modules: {metadata['cognitive_modules']}")
        print(f"Model status: {metadata['model_status']}")
        
    else:
        # Batch inference
        print(f"\nBatch inference mode: {args.image_dir}")
        
        # Get all image files
        image_dir = Path(args.image_dir)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(image_dir.glob(f'*{ext}'))
            image_paths.extend(image_dir.glob(f'*{ext.upper()}'))
            
        image_paths = [str(p) for p in sorted(image_paths)]
        
        if not image_paths:
            raise ValueError(f"No image files found in directory {args.image_dir}")
            
        print(f"Found {len(image_paths)} images")
        
        # Perform batch inference
        results = inference_engine.predict_batch(image_paths)
        
        # Print summary
        successful = sum(1 for r in results if r[0] is not None)
        print(f"\n=== Batch Inference Results ===")
        print(f"Total images: {len(image_paths)}")
        print(f"Successfully processed: {successful}")
        print(f"Failed: {len(image_paths) - successful}")
        
        if successful > 0:
            depths = [r[0] for r in results if r[0] is not None]
            confidences = [r[1] for r in results if r[1] is not None]
            print(f"Depth range: {min(depths):.4f} - {max(depths):.4f}")
            print(f"Average depth: {np.mean(depths):.4f}")
            print(f"Average confidence: {np.mean(confidences):.4f}")
    
    # JSON结果生成功能已移除 - 根据用户要求只生成prediction图像
    # inference_engine.save_results(results, args.output)
    
    print(f"\nInference completed!")


if __name__ == '__main__':
    main()