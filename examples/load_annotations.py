"""
MS-Road Dataset Annotation Loading Examples
============================================

This script provides examples for loading and visualizing MS-Road 
annotations in COCO format.

Author: MS-Road Team
Date: 2026
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from collections import defaultdict


# Category mapping
CATEGORIES = {
    1: 'person',
    2: 'bike',
    3: 'car',
    4: 'bus',
    5: 'truck',
    6: 'tricycle',
    7: 'van',
    8: 'traffic light',
    9: 'traffic sign'
}

# Color mapping for visualization
CATEGORY_COLORS = {
    1: '#FF6B6B',  # person - red
    2: '#4ECDC4',  # bike - cyan
    3: '#45B7D1',  # car - blue
    4: '#96CEB4',  # bus - green
    5: '#FFEAA7',  # truck - yellow
    6: '#DDA0DD',  # tricycle - purple
    7: '#98D8C8',  # van - mint
    8: '#F7DC6F',  # traffic light - gold
    9: '#BB8FCE',  # traffic sign - lavender
}


def load_coco_annotations(annotation_path):
    """
    Load COCO format annotations.
    
    Args:
        annotation_path: Path to the JSON annotation file
        
    Returns:
        dict: COCO format data
    """
    with open(annotation_path, 'r') as f:
        data = json.load(f)
    return data


def parse_annotations(coco_data):
    """
    Parse COCO annotations into a more usable format.
    
    Args:
        coco_data: COCO format data
        
    Returns:
        dict: image_id -> list of annotations
    """
    # Build image id to annotations mapping
    image_to_anns = defaultdict(list)
    for ann in coco_data['annotations']:
        image_to_anns[ann['image_id']].append(ann)
    
    return image_to_anns


def get_image_info(coco_data, image_id):
    """
    Get image information by image_id.
    
    Args:
        coco_data: COCO format data
        image_id: Image ID
        
    Returns:
        dict: Image information
    """
    for img in coco_data['images']:
        if img['id'] == image_id:
            return img
    return None


def visualize_annotations(ms_image_path, annotations, save_path=None):
    """
    Visualize multispectral image with bounding box annotations.
    
    Args:
        ms_image_path: Path to the .npy file
        annotations: List of annotation dicts
        save_path: Optional path to save the figure
    """
    # Load multispectral image
    ms_img = np.load(ms_image_path)
    
    # Create pseudo-RGB
    r = ms_img[4]  # Red band
    g = ms_img[2]  # Green band
    b = ms_img[1]  # Blue band
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(14, 10))
    ax.imshow(rgb)
    
    # Draw bounding boxes
    for ann in annotations:
        x, y, w, h = ann['bbox']
        category_id = ann['category_id']
        
        color = CATEGORY_COLORS.get(category_id, '#FFFFFF')
        
        # Draw bounding box
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        category_name = CATEGORIES.get(category_id, f'ID:{category_id}')
        ax.text(x, y-5, category_name,
                color='white', fontsize=9, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', 
                         facecolor=color, 
                         alpha=0.8))
    
    ax.set_title(f'MS-Road Detection Visualization ({len(annotations)} objects)')
    ax.axis('off')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def print_statistics(coco_data):
    """
    Print dataset statistics.
    
    Args:
        coco_data: COCO format data
    """
    print("=" * 60)
    print("MS-Road Dataset Statistics")
    print("=" * 60)
    
    num_images = len(coco_data['images'])
    num_annotations = len(coco_data['annotations'])
    
    print(f"\nTotal Images: {num_images}")
    print(f"Total Annotations: {num_annotations}")
    print(f"Average objects per image: {num_annotations / num_images:.2f}")
    
    # Count per category
    category_counts = defaultdict(int)
    for ann in coco_data['annotations']:
        category_counts[ann['category_id']] += 1
    
    print("\nAnnotations per category:")
    print("-" * 40)
    for cat_id, cat_name in CATEGORIES.items():
        count = category_counts.get(cat_id, 0)
        percentage = (count / num_annotations) * 100
        print(f"  {cat_name:15s}: {count:6d} ({percentage:5.2f}%)")
    
    print("\n" + "=" * 60)


def export_to_yolo(coco_data, output_dir):
    """
    Export COCO annotations to YOLO format.
    
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values are normalized to [0, 1].
    
    Args:
        coco_data: COCO format data
        output_dir: Directory to save YOLO format labels
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Build image id to annotations mapping
    image_to_anns = parse_annotations(coco_data)
    
    # Build image id to file_name mapping
    image_info = {img['id']: img for img in coco_data['images']}
    
    for image_id, anns in image_to_anns.items():
        img_info = image_info[image_id]
        img_w, img_h = img_info['width'], img_info['height']
        
        # Create YOLO format label file
        label_file = output_dir / f"{Path(img_info['file_name']).stem}.txt"
        
        with open(label_file, 'w') as f:
            for ann in anns:
                x, y, w, h = ann['bbox']
                
                # Convert to YOLO format (normalize and center)
                x_center = (x + w / 2) / img_w
                y_center = (y + h / 2) / img_h
                w_norm = w / img_w
                h_norm = h / img_h
                
                # COCO category_id starts from 1, YOLO from 0
                class_id = ann['category_id'] - 1
                
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")
    
    print(f"Exported YOLO labels to: {output_dir}")


def main():
    """Example usage."""
    
    # Example paths - replace with your actual paths
    annotation_path = 'path/to/annotations.json'
    image_dir = 'path/to/images'
    
    print("MS-Road Annotation Loading Example")
    print("=" * 60)
    
    # Example 1: Print statistics
    print("\n1. Dataset Statistics:")
    print("-" * 60)
    # coco_data = load_coco_annotations(annotation_path)
    # print_statistics(coco_data)
    
    # Example 2: Visualize annotations
    print("\n2. Visualizing Annotations:")
    print("-" * 60)
    # image_to_anns = parse_annotations(coco_data)
    # for image_id, anns in list(image_to_anns.items())[:3]:  # Visualize first 3
    #     img_info = get_image_info(coco_data, image_id)
    #     img_path = Path(image_dir) / img_info['file_name'].replace('.jpg', '.npy')
    #     visualize_annotations(img_path, anns, save_path=f'viz_{image_id}.png')
    
    # Example 3: Export to YOLO format
    print("\n3. Export to YOLO Format:")
    print("-" * 60)
    # export_to_yolo(coco_data, 'yolo_labels')
    
    print("\nNote: Uncomment the example code and provide actual paths to run.")


if __name__ == '__main__':
    main()
