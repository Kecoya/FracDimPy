#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Box-counting Method for Image Data Test Example
===============================================

This example demonstrates how to use the box-counting method to calculate
the fractal dimension of 2D image data.

The box-counting method works by covering the image with boxes of different
sizes and counting how many boxes contain part of the fractal structure.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from fracDimPy import box_counting

# Try to use scienceplots style
try:
    import scienceplots
    # plt.style.use(['ieee'])  # Uncomment to use IEEE style
except ImportError:
    pass

# Set font: Times New Roman for English, Microsoft YaHei for Chinese
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

# Data file path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "box_counting_image_data.png")

def main():
    print("="*60)
    print("Box-counting Method for Image Data Test Example")
    print("="*60)
    
    # 1. Load and preprocess image
    print(f"\n1. Loading image: {data_file}")
    try:
        from PIL import Image
        img = Image.open(data_file).convert('RGB')  # Convert to RGB
        img_array = np.array(img)
        
        # Convert to grayscale for box-counting (inverted for dark fractals)
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        img_gray = 255 - (0.2989 * r + 0.5870 * g + 0.1140 * b)  # Inverted grayscale
        
        print(f"   Image shape: {img_gray.shape}")
        print(f"   Value range: {img_gray.min():.1f} ~ {img_gray.max():.1f}")
        
        # Binarize image (0 or 255) for box-counting
        threshold = np.mean(img_gray)
        binary_img = np.where(img_gray > threshold, 255, 0).astype(np.uint8)
        print(f"   Threshold: {threshold:.2f}")
        print(f"   White pixels: {np.sum(binary_img == 255)}")
        
        # Load original grayscale image for display
        img_data = np.array(Image.open(data_file).convert('L'))
        
    except Exception as e:
        print(f"   Error: {e}")
        print("   Using random test data...")
        binary_img = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
        img_data = binary_img
    
    # 2. Calculate fractal dimension using box-counting
    print("\n2. Calculating fractal dimension...")
    D, result = box_counting(binary_img, data_type='image')
    
    # 3. Display results
    print("\n3. Results:")
    print(f"    Fractal dimension D: {D:.4f}")
    print(f"    Goodness of fit R²: {result['R2']:.4f}")
    
    # 4. Visualize results
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Top left: Original grayscale image
    axes[0, 0].imshow(img_data, cmap='gray')
    axes[0, 0].set_title('Original Grayscale Image')
    axes[0, 0].axis('off')
    
    # Top right: Binary image
    axes[0, 1].imshow(binary_img, cmap='binary')
    axes[0, 1].set_title('Binary Image (for Box-counting)')
    axes[0, 1].axis('off')
    
    # Bottom left: Log-log plot
    if 'epsilon_values' in result and 'N_values' in result:
        axes[1, 0].loglog(result['epsilon_values'], result['N_values'], 'o', markersize=6, label='Data points')
        # Draw fitting line
        if 'coefficients' in result:
            # Relationship: log(N) = a*log(1/ε) + b  =>  N = exp(b) * ε^(-a)
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = np.exp(b) * np.array(result['epsilon_values'])**(-a)
            axes[1, 0].loglog(result['epsilon_values'], fit_line, 'r-', linewidth=2,
                            label=f'Fit (D={D:.4f})')
            
            # Display equation
            C = np.exp(b)
            equation_text = f'$N(\\varepsilon) = {C:.2e} \\cdot \\varepsilon^{{-{a:.4f}}}$\n$R^2 = {result["R2"]:.4f}$'
            axes[1, 0].text(0.05, 0.95, equation_text, transform=axes[1, 0].transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        axes[1, 0].set_xlabel('ε (Box size)')
        axes[1, 0].set_ylabel('N(ε) (Number of boxes)')
        axes[1, 0].set_title('Box-counting Log-Log Plot')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Bottom right: Linear fit in log space
    if 'log_inv_epsilon' in result and 'log_N' in result:
        axes[1, 1].plot(result['log_inv_epsilon'], result['log_N'], 'o', markersize=6, label='Data points')
        if 'coefficients' in result:
            # Draw fitting line
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = a * result['log_inv_epsilon'] + b
            axes[1, 1].plot(result['log_inv_epsilon'], fit_line, 'r-', linewidth=2,
                          label=f'Fit (D={a:.4f})')
            
            # Display equation
            equation_text = f'$\\log(N) = {a:.4f} \\cdot \\log(1/\\varepsilon) + {b:.4f}$\n$R^2 = {result["R2"]:.4f}$\n$D = {a:.4f}$'
            axes[1, 1].text(0.05, 0.95, equation_text, transform=axes[1, 1].transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        axes[1, 1].set_xlabel('log(1/ε)')
        axes[1, 1].set_ylabel('log(N)')
        axes[1, 1].set_title('Linear Fit in Log Space')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(current_dir, "result_box_counting_image.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n4. Visualization saved: {output_file}")
    plt.show()
    
    print("\nExample completed!")


if __name__ == '__main__':
    main()

