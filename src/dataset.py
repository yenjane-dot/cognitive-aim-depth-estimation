"""Dataset module for Cognitive-Aim Experiment B reproduction.

Handles image loading, depth ground truth, and EXIF metadata processing.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image, ExifTags
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np


class DepthDataset(Dataset):
    """Dataset for depth estimation with EXIF metadata."""
    
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        image_size: Tuple[int, int] = (384, 384),
        use_exif: bool = True,
        augment: bool = True
    ):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.use_exif = use_exif
        
        # Load data annotations
        self.annotations = self._load_annotations()
        
        # Image transforms
        self.transform = self._get_transforms(augment and split == 'train')
        
        # EXIF feature mapping
        self.camera_to_id = self._build_camera_mapping()
        
    def _load_annotations(self) -> List[Dict]:
        """Load dataset annotations."""
        ann_file = os.path.join(self.data_dir, f'{self.split}_annotations.json')
        
        if os.path.exists(ann_file):
            with open(ann_file, 'r') as f:
                return json.load(f)
        else:
            # Fallback: scan directory for images
            return self._scan_directory()
            
    def _scan_directory(self) -> List[Dict]:
        """Scan directory for images and create annotations."""
        annotations = []
        
        images_dir = os.path.join(self.data_dir, 'images')
        depths_dir = os.path.join(self.data_dir, 'depths')
        
        if not os.path.exists(images_dir):
            return []
            
        for img_file in os.listdir(images_dir):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(images_dir, img_file)
                
                # Look for corresponding depth file
                depth_file = img_file.replace('.jpg', '_depth.npy').replace('.jpeg', '_depth.npy').replace('.png', '_depth.npy')
                depth_path = os.path.join(depths_dir, depth_file)
                
                if os.path.exists(depth_path):
                    annotations.append({
                        'image_path': img_path,
                        'depth_path': depth_path,
                        'image_id': len(annotations)
                    })
                    
        return annotations
        
    def _get_transforms(self, augment: bool = False) -> transforms.Compose:
        """Get image transforms."""
        transform_list = [
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
        
        if augment:
            transform_list.insert(-2, transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1))
            transform_list.insert(-2, transforms.RandomHorizontalFlip(p=0.5))
            
        return transforms.Compose(transform_list)
        
    def _build_camera_mapping(self) -> Dict[str, int]:
        """Build mapping from camera model to ID."""
        camera_models = set()
        
        for ann in self.annotations:
            if self.use_exif:
                exif_data = self._extract_exif(ann['image_path'])
                if exif_data and 'camera_model' in exif_data:
                    camera_models.add(exif_data['camera_model'])
                    
        # Create mapping
        camera_to_id = {model: idx for idx, model in enumerate(sorted(camera_models))}
        camera_to_id['unknown'] = len(camera_to_id)  # Default for unknown cameras
        
        return camera_to_id
        
    def _extract_exif(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Extract EXIF metadata from image."""
        try:
            with Image.open(image_path) as img:
                exif_dict = img._getexif()
                
            if exif_dict is None:
                return None
                
            exif_data = {}
            
            for tag_id, value in exif_dict.items():
                tag = ExifTags.TAGS.get(tag_id, tag_id)
                
                if tag == 'Model':
                    exif_data['camera_model'] = str(value)
                elif tag == 'FocalLength':
                    if isinstance(value, tuple):
                        exif_data['focal_length'] = float(value[0]) / float(value[1])
                    else:
                        exif_data['focal_length'] = float(value)
                elif tag == 'FNumber':
                    if isinstance(value, tuple):
                        exif_data['aperture'] = float(value[0]) / float(value[1])
                    else:
                        exif_data['aperture'] = float(value)
                elif tag == 'ISOSpeedRatings':
                    exif_data['iso'] = float(value)
                    
            return exif_data
            
        except Exception:
            return None
            
    def _normalize_exif(self, exif_data: Dict[str, Any]) -> Dict[str, float]:
        """Normalize EXIF values to reasonable ranges."""
        normalized = {
            'focal_length': min(max(exif_data.get('focal_length', 50.0), 10.0), 200.0) / 200.0,
            'aperture': min(max(exif_data.get('aperture', 2.8), 1.0), 22.0) / 22.0,
            'iso': min(max(exif_data.get('iso', 100.0), 50.0), 6400.0) / 6400.0,
        }
        
        # Camera ID
        camera_model = exif_data.get('camera_model', 'unknown')
        normalized['camera_id'] = self.camera_to_id.get(camera_model, self.camera_to_id['unknown'])
        
        return normalized
        
    def __len__(self) -> int:
        return len(self.annotations)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ann = self.annotations[idx]
        
        # Load image
        image = Image.open(ann['image_path']).convert('RGB')
        image = self.transform(image)
        
        # Load depth
        if 'depth_path' in ann and os.path.exists(ann['depth_path']):
            depth = np.load(ann['depth_path'])
            depth = torch.from_numpy(depth).float()
            
            # Resize depth to match image
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(0).unsqueeze(0),
                size=self.image_size,
                mode='bilinear',
                align_corners=False
            ).squeeze()
            
            # Normalize depth to [0, 1]
            depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
        else:
            # Dummy depth for inference
            depth = torch.zeros(self.image_size)
            
        sample = {
            'image': image,
            'depth': depth,
            'image_id': ann.get('image_id', idx)
        }
        
        # Add EXIF data if enabled
        if self.use_exif:
            exif_data = self._extract_exif(ann['image_path'])
            if exif_data:
                normalized_exif = self._normalize_exif(exif_data)
                sample['exif'] = {
                    'focal_length': torch.tensor(normalized_exif['focal_length'], dtype=torch.float32),
                    'aperture': torch.tensor(normalized_exif['aperture'], dtype=torch.float32),
                    'iso': torch.tensor(normalized_exif['iso'], dtype=torch.float32),
                    'camera_id': torch.tensor(normalized_exif['camera_id'], dtype=torch.long)
                }
            else:
                # Default EXIF values
                sample['exif'] = {
                    'focal_length': torch.tensor(0.25, dtype=torch.float32),  # 50mm normalized
                    'aperture': torch.tensor(0.127, dtype=torch.float32),     # f/2.8 normalized
                    'iso': torch.tensor(0.016, dtype=torch.float32),          # ISO 100 normalized
                    'camera_id': torch.tensor(self.camera_to_id['unknown'], dtype=torch.long)
                }
                
        return sample


def create_dataloaders(
    data_dir: str,
    config: Dict,
    batch_size: int = 8,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""
    
    # Create datasets
    train_dataset = DepthDataset(
        data_dir=data_dir,
        split='train',
        image_size=tuple(config['dataset']['image_size']),
        use_exif=config['dataset']['use_exif'],
        augment=config['training']['augmentation']['enable']
    )
    
    val_dataset = DepthDataset(
        data_dir=data_dir,
        split='val',
        image_size=tuple(config['dataset']['image_size']),
        use_exif=config['dataset']['use_exif'],
        augment=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    images = torch.stack([item['image'] for item in batch])
    depths = torch.stack([item['depth'] for item in batch])
    image_ids = torch.tensor([item['image_id'] for item in batch])
    
    result = {
        'images': images,
        'depths': depths,
        'image_ids': image_ids
    }
    
    # Handle EXIF data if present
    if 'exif' in batch[0]:
        exif_batch = {}
        for key in batch[0]['exif'].keys():
            exif_batch[key] = torch.stack([item['exif'][key] for item in batch])
        result['exif'] = exif_batch
        
    return result