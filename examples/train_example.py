"""
MS-Road Dataset Training Example
=================================

This script provides a basic example for training a detector on MS-Road.
Compatible with MMDetection and PyTorch.

Note: This is a simplified example. For production use, please refer to
MMDetection documentation for complete training pipelines.

Author: MS-Road Team
Date: 2026
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json


class MSRoadDataset(Dataset):
    """
    PyTorch Dataset for MS-Road.
    
    Note: MS-Road images have shape (8, 1200, 900) - (C, W, H)
    You may need to transpose based on your model's expected input.
    """
    
    def __init__(self, image_dir, annotation_path, transform=None):
        """
        Args:
            image_dir: Directory containing .npy files
            annotation_path: Path to COCO format annotation file
            transform: Optional transforms
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Load annotations
        with open(annotation_path, 'r') as f:
            self.coco_data = json.load(f)
        
        # Build mappings
        self.images = self.coco_data['images']
        self.image_to_anns = self._build_annotation_mapping()
        
    def _build_annotation_mapping(self):
        """Build mapping from image_id to annotations."""
        mapping = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in mapping:
                mapping[img_id] = []
            mapping[img_id].append(ann)
        return mapping
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor with shape (8, H, W) - Note: transposed from (8, W, H)
            targets: Dict containing bounding boxes and labels
        """
        img_info = self.images[idx]
        img_id = img_info['id']
        
        # Load multispectral image
        # Assuming file_name in COCO is like "xxx.jpg", we convert to "xxx.npy"
        npy_name = Path(img_info['file_name']).stem + '.npy'
        npy_path = self.image_dir / npy_name
        
        # Load data: shape (8, 1200, 900) - (C, W, H)
        ms_image = np.load(npy_path)
        
        # Transpose to (C, H, W) for PyTorch convention
        # From (8, 1200, 900) to (8, 900, 1200)
        ms_image = ms_image.transpose(0, 2, 1)
        
        # Convert to tensor
        image = torch.from_numpy(ms_image).float()
        
        # Get annotations
        anns = self.image_to_anns.get(img_id, [])
        
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann['bbox']
            boxes.append([x, y, x + w, y + h])  # Convert to xyxy format
            labels.append(ann['category_id'])
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros(0, dtype=torch.int64),
            'image_id': torch.tensor([img_id])
        }
        
        if self.transform:
            image, target = self.transform(image, target)
        
        return image, target


def collate_fn(batch):
    """
    Custom collate function for batching.
    """
    images = []
    targets = []
    
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    
    images = torch.stack(images, 0)
    return images, targets


def create_dataloaders(train_img_dir, train_ann, val_img_dir, val_ann, 
                       batch_size=8, num_workers=4):
    """
    Create training and validation dataloaders.
    
    Args:
        train_img_dir: Training image directory
        train_ann: Training annotation path
        val_img_dir: Validation image directory
        val_ann: Validation annotation path
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        train_loader, val_loader
    """
    train_dataset = MSRoadDataset(train_img_dir, train_ann)
    val_dataset = MSRoadDataset(val_img_dir, val_ann)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader


def preprocess_for_model(ms_image, model_type='mmdet'):
    """
    Preprocess MS-Road image for different model types.
    
    Args:
        ms_image: numpy array with shape (8, 1200, 900) or (8, 900, 1200)
        model_type: 'mmdet', 'yolo', or 'rgb'
        
    Returns:
        Preprocessed image
    """
    if model_type == 'mmdet':
        # MMDetection expects (C, H, W)
        if ms_image.shape == (8, 1200, 900):
            ms_image = ms_image.transpose(0, 2, 1)  # (8, 900, 1200)
        return ms_image
    
    elif model_type == 'yolo':
        # YOLO typically expects (H, W, C)
        if ms_image.shape == (8, 1200, 900):
            ms_image = ms_image.transpose(2, 1, 0)  # (900, 1200, 8)
        elif ms_image.shape == (8, 900, 1200):
            ms_image = ms_image.transpose(1, 2, 0)  # (900, 1200, 8)
        return ms_image
    
    elif model_type == 'rgb':
        # Extract pseudo-RGB bands
        # Band 5 (idx 4) -> R, Band 3 (idx 2) -> G, Band 2 (idx 1) -> B
        r = ms_image[4]
        g = ms_image[2]
        b = ms_image[1]
        
        # Stack to (3, H, W)
        rgb = np.stack([r, g, b], axis=0)
        
        # Normalize
        rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
        
        return rgb
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def get_mmdet_config():
    """
    Get MMDetection config example for MS-Road.
    
    Note: This is a template. Please refer to MMDetection documentation
    for complete configuration.
    """
    config = """
# MMDetection Config for MS-Road Dataset

# Dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/msroad/'

# IMPORTANT: MS-Road images are 8-channel multispectral
# Shape: (8, 1200, 900) - need to handle in pipeline

img_norm_cfg = dict(
    mean=[0.0] * 8,  # 8 channels
    std=[255.0] * 8,
    to_rgb=False
)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1200, 900), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1200, 900),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ]
    )
]

# Dataset configuration
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train.json',
        img_prefix=data_root + 'train/',
        pipeline=train_pipeline,
        classes=('person', 'bike', 'car', 'bus', 'truck', 
                'tricycle', 'van', 'traffic light', 'traffic sign')
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val.json',
        img_prefix=data_root + 'val/',
        pipeline=test_pipeline,
        classes=('person', 'bike', 'car', 'bus', 'truck', 
                'tricycle', 'van', 'traffic light', 'traffic sign')
    ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_test.json',
        img_prefix=data_root + 'test/',
        pipeline=test_pipeline,
        classes=('person', 'bike', 'car', 'bus', 'truck', 
                'tricycle', 'van', 'traffic light', 'traffic sign')
    )
)

# Model configuration (example: Faster R-CNN with ResNet-50)
model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50'),
        # IMPORTANT: Change input channels from 3 to 8 for multispectral
        in_channels=8
    ),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5
    ),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]
        ),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]
        ),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
        ),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)
    ),
    roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=9,  # MS-Road has 9 categories
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]
            ),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
            ),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        )
    ),
    # ... rest of model config
)

# Training settings
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11]
)
runner = dict(type='EpochBasedRunner', max_epochs=12)

# Evaluation
evaluation = dict(interval=1, metric='bbox')
"""
    return config


def main():
    """Example usage."""
    print("MS-Road Training Example")
    print("=" * 60)
    
    # Print dataset info
    print("\n1. Dataset Information:")
    print("-" * 60)
    print("  - Images: 49,253 (Train: 34,520, Test: 14,733)")
    print("  - Instances: 512,194")
    print("  - Categories: 9")
    print("  - Image Shape: (8, 1200, 900) - (C, W, H)")
    
    # Print preprocessing info
    print("\n2. Preprocessing for Different Models:")
    print("-" * 60)
    print("  - MMDetection: (8, 900, 1200) - (C, H, W)")
    print("  - YOLO: (900, 1200, 8) - (H, W, C)")
    print("  - RGB Baseline: (3, 900, 1200) - Extract bands 5,3,2")
    
    # Print MMDet config
    print("\n3. MMDetection Config Template:")
    print("-" * 60)
    print(get_mmdet_config())
    
    print("\n" + "=" * 60)
    print("For complete training scripts, please refer to:")
    print("  - MMDetection: https://github.com/open-mmlab/mmdetection")
    print("  - Ultralytics: https://github.com/ultralytics/ultralytics")


if __name__ == '__main__':
    main()
