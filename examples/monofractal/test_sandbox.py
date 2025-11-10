#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sandbox Method Test Example
============================

This example demonstrates how to use the sandbox method to calculate
the fractal dimension of point set or image data.

The sandbox method is a local scale analysis technique that counts
the number of points within boxes of increasing radius centered at
various points in the dataset.
"""

import numpy as np
import os
from fracDimPy import sandbox_method
import matplotlib.pyplot as plt

# Set font for Chinese characters
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

# Data file path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "sandbox_data.png")

def main():
    print("="*60)
    print("Sandbox Method Test Example")
    print("="*60)
    
    # 1. Load data
    print(f"\n1. Loading data: {data_file}")
    
    # 2. Calculate fractal dimension using sandbox method
    print("\n2. Calculating fractal dimension...")
    D, result = sandbox_method(data_file)
    
    # 3. Display results
    print("\n3. Results:")
    print(f"    Fractal dimension D: {D:.4f}")
    print(f"    Goodness of fit R²: {result['R2']:.4f}")
    
    # 4. Visualize results
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
        
        img = Image.open(data_file).convert('L')
        img_array = np.array(img)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left plot: Original image
        axes[0].imshow(img_array, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Right plot: Sandbox method log-log plot
        if 'r_values' in result and 'N_values' in result:
            axes[1].loglog(result['r_values'], result['N_values'], 'o', label='Data points')
            # Draw fitting line
            if 'coefficients' in result:
                fit_line = np.exp(result['coefficients'][1]) * np.array(result['r_values'])**result['coefficients'][0]
                axes[1].loglog(result['r_values'], fit_line, 'r-', label=f'Fit (D={D:.4f})')
            axes[1].set_xlabel('r (Radius)')
            axes[1].set_ylabel('N(r) (Number of points)')
            axes[1].set_title('Sandbox Method Log-Log Plot')
            axes[1].legend()
            axes[1].grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(current_dir, "result_sandbox.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n4. Visualization saved: {output_file}")
        plt.show()
        
    except ImportError:
        print("\n4. Visualization failed: matplotlib and PIL libraries required")
    
    print("\nExample completed!")


if __name__ == '__main__':
    main()
