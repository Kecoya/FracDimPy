#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multifractal Analysis - Image Data
===================================

This example demonstrates how to use fracDimPy to perform multifractal analysis 
on 2D image data. The multifractal characteristics of images can reveal the 
complexity, non-uniformity, and multi-scale properties of textures, with wide 
applications in medical image analysis, materials science, geological exploration, etc.

Main Features:
- Load and preprocess image data
- Calculate multifractal spectrum of images
- Extract multifractal characteristic parameters
- Visualize analysis results

Theoretical Background:
- Image multifractal analysis is based on the extension of box-counting method
- Describe the diversity of image grayscale distribution through different q-order moments
- Multifractal spectrum width reflects the degree of non-uniformity of the image
- D(0): Capacity dimension, D(1): Information dimension, D(2): Correlation dimension
"""

import numpy as np
import os
from fracDimPy import multifractal_image
import matplotlib.pyplot as plt

# Set Chinese font
import scienceplots 
plt.style.use(['science', 'no-latex'])
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# Data file path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "mf_image_shale.png")

def main():
    print("="*60)
    print("Multifractal Analysis - Image Data")
    print("="*60)
    
    # 1. Load image data
    print(f"\n1. Loading image: {data_file}")
    try:
        from PIL import Image
        img = Image.open(data_file)
        img_array = np.array(img)
        
        # Convert to grayscale if color image
        if len(img_array.shape) == 3:
            img_gray = np.mean(img_array, axis=2)
        else:
            img_gray = img_array
        
        print(f"   Image size: {img_gray.shape}")
        print(f"   Pixel range: {img_gray.min():.1f} ~ {img_gray.max():.1f}")
        
    except Exception as e:
        print(f"   Loading failed: {e}")
        return
    
    # 2. Multifractal analysis
    print("\n2. Performing multifractal analysis...")
    try:
        metrics, figure_data = multifractal_image(img_gray)
        
        # 3. Display calculation results
        print("\n3. Multifractal characteristic parameters:")
        print(f"   Capacity dimension D(0): {metrics['容量维数 D(0)'][0]:.4f}")
        print(f"   Information dimension D(1): {metrics['信息维数 D(1)'][0]:.4f}")
        print(f"   Correlation dimension D(2): {metrics['关联维数 D(2)'][0]:.4f}")
        print(f"   Hurst exponent H: {metrics['Hurst指数 H'][0]:.4f}")
        print(f"   Spectrum width: {metrics['谱宽度'][0]:.4f}")
        
    except Exception as e:
        print(f"\nAnalysis failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. Visualize results
    print("\n4. Generating visualization plots...")
    try:
        fig = plt.figure(figsize=(16, 10))
        
        # Original image
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(img_gray, cmap='gray')
        ax1.set_title('(a) Original Image')
        ax1.axis('off')
        
        # Extract analysis results
        ql = figure_data['q值']
        tau_q = figure_data['质量指数tau(q)']
        alpha_q = figure_data['奇异性指数alpha(q)']
        f_alpha = figure_data['多重分形谱f(alpha)']
        D_q = figure_data['广义维数D(q)']
        
        # tau(q) curve
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(ql, tau_q, 'o-', color='darkgreen', linewidth=2, markersize=4)
        ax2.set_xlabel(r'$q$ - Statistical Moment Order', fontsize=10)
        ax2.set_ylabel(r'$\tau(q)$ - Mass Exponent', fontsize=10)
        ax2.set_title('(b) Mass Exponent Function')
        ax2.grid(True, alpha=0.3)
        
        # alpha(q) curve
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(ql, alpha_q, 's-', color='crimson', linewidth=2, markersize=4)
        ax3.set_xlabel(r'$q$ - Statistical Moment Order', fontsize=10)
        ax3.set_ylabel(r'$\alpha(q)$ - Hölder Exponent', fontsize=10)
        ax3.set_title(r'(c) Hölder Exponent Function')
        ax3.grid(True, alpha=0.3)
        
        # f(alpha) multifractal spectrum
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(alpha_q, f_alpha, '^-', color='darkorange', linewidth=2, markersize=4)
        ax4.set_xlabel(r'$\alpha$ - Singularity Index', fontsize=10)
        ax4.set_ylabel(r'$f(\alpha)$ - Multifractal Spectrum', fontsize=10)
        ax4.set_title('(d) Multifractal Spectrum')
        ax4.grid(True, alpha=0.3)
        
        # D(q) curve
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(ql, D_q, 'd-', color='mediumpurple', linewidth=2, markersize=4)
        ax5.set_xlabel(r'$q$ - Statistical Moment Order', fontsize=10)
        ax5.set_ylabel(r'$D(q)$ - Generalized Dimension', fontsize=10)
        ax5.set_title('(e) Generalized Dimension Spectrum')
        ax5.grid(True, alpha=0.3)
        
        # Key parameters comparison
        ax6 = fig.add_subplot(2, 3, 6)
        params = ['D(0)', 'D(1)', 'D(2)', 'H', 'Spectrum Width']
        values = [
            metrics['容量维数 D(0)'][0],
            metrics['信息维数 D(1)'][0],
            metrics['关联维数 D(2)'][0],
            metrics['Hurst指数 H'][0],
            metrics['谱宽度'][0]
        ]
        colors = ['green', 'blue', 'red', 'orange', 'purple']
        bars = ax6.bar(params, values, color=colors, alpha=0.7)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax6.set_ylabel('Parameter Value')
        ax6.set_title('(f) Multifractal Characteristic Parameters')
        ax6.tick_params(axis='x', labelsize=8, rotation=15)
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = os.path.join(current_dir, "result_mf_image.png")
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nVisualization results saved: result_mf_image.png")
        plt.show()
        
    except Exception as e:
        print(f"\nVisualization failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nExample completed!")


if __name__ == '__main__':
    main()
