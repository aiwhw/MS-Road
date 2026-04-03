"""
MS-Road Dataset Data Loading Examples
======================================

This script provides examples for loading and visualizing MS-Road 
multispectral images.

Note: MS-Road images are stored in .npy format with shape (8, 1200, 900)
which is (Channels, Width, Height), NOT (C, H, W).

Author: MS-Road Team
Date: 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_multispectral_image(npy_path):
    """
    Load a multispectral image from .npy file.
    
    Args:
        npy_path: Path to the .npy file
        
    Returns:
        numpy array with shape (8, 1200, 900)
        
    Note:
        The returned array has shape (C, W, H), NOT (C, H, W).
        Channel order: [395, 450, 525, 590, 650, 730, 850, 950] nm
    """
    data = np.load(npy_path)
    assert data.shape == (8, 1200, 900), \
        f"Expected shape (8, 1200, 900), got {data.shape}"
    return data


def create_pseudo_rgb(ms_image, r_band=4, g_band=2, b_band=1):
    """
    Create pseudo-RGB image from multispectral data.
    
    Default mapping:
        R: Band 5 (index 4) - 650 nm (Red)
        G: Band 3 (index 2) - 525 nm (Green)
        B: Band 2 (index 1) - 450 nm (Blue)
    
    Args:
        ms_image: numpy array with shape (8, 1200, 900)
        r_band: index for red channel (default: 4)
        g_band: index for green channel (default: 2)
        b_band: index for blue channel (default: 1)
        
    Returns:
        rgb_image: numpy array with shape (900, 1200, 3), values in [0, 1]
    """
    r = ms_image[r_band]
    g = ms_image[g_band]
    b = ms_image[b_band]
    
    # Stack and normalize
    rgb = np.stack([r, g, b], axis=-1)
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    
    return rgb


def visualize_all_bands(ms_image, save_path=None):
    """
    Visualize all 8 spectral bands.
    
    Args:
        ms_image: numpy array with shape (8, 1200, 900)
        save_path: optional path to save the figure
    """
    wavelengths = [395, 450, 525, 590, 650, 730, 850, 950]
    names = ['Near-UV', 'Blue', 'Green', 'Yellow', 'Red', 'Red Edge', 'NIR-1', 'NIR-2']
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i, (ax, wl, name) in enumerate(zip(axes, wavelengths, names)):
        ax.imshow(ms_image[i], cmap='gray')
        ax.set_title(f'Band {i+1}: {wl} nm\n({name})')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def visualize_comparison(ms_image, save_path=None):
    """
    Visualize pseudo-RGB and individual bands for comparison.
    
    Args:
        ms_image: numpy array with shape (8, 1200, 900)
        save_path: optional path to save the figure
    """
    rgb = create_pseudo_rgb(ms_image)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Pseudo-RGB
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('Pseudo-RGB (Bands 5-3-2)')
    axes[0, 0].axis('off')
    
    # Individual bands
    bands = [0, 4, 7]  # UV, Red, NIR-2
    band_names = ['Near-UV (395 nm)', 'Red (650 nm)', 'NIR-2 (950 nm)']
    
    for idx, (band_idx, name) in enumerate(zip(bands, band_names)):
        row = idx // 3
        col = idx % 3
        if idx == 0:
            ax = axes[0, 1]
        elif idx == 1:
            ax = axes[0, 2]
        else:
            ax = axes[1, 0]
            
        ax.imshow(ms_image[band_idx], cmap='gray')
        ax.set_title(name)
        ax.axis('off')
    
    # Histogram
    axes[1, 1].hist(ms_image.flatten(), bins=50, alpha=0.7)
    axes[1, 1].set_title('Pixel Value Distribution')
    axes[1, 1].set_xlabel('Pixel Value')
    axes[1, 1].set_ylabel('Frequency')
    
    # Spectral profile at center
    center_x, center_y = 600, 450
    spectral_profile = ms_image[:, center_y, center_x]
    wavelengths = [395, 450, 525, 590, 650, 730, 850, 950]
    
    axes[1, 2].plot(wavelengths, spectral_profile, 'o-')
    axes[1, 2].set_title(f'Spectral Profile at ({center_x}, {center_y})')
    axes[1, 2].set_xlabel('Wavelength (nm)')
    axes[1, 2].set_ylabel('Reflectance')
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """Example usage of the data loading functions."""
    
    # Example path - replace with your actual path
    # npy_path = 'path/to/your/image.npy'
    
    # For demonstration, create a dummy array
    print("Creating example data with shape (8, 1200, 900)...")
    example_data = np.random.rand(8, 1200, 900).astype(np.float32)
    
    print(f"Data shape: {example_data.shape}")
    print(f"Data dtype: {example_data.dtype}")
    print(f"Data range: [{example_data.min():.3f}, {example_data.max():.3f}]")
    
    # Visualize all bands
    print("\nVisualizing all bands...")
    visualize_all_bands(example_data, save_path='all_bands.png')
    
    # Visualize comparison
    print("\nVisualizing comparison...")
    visualize_comparison(example_data, save_path='comparison.png')
    
    print("\nDone!")


if __name__ == '__main__':
    main()
