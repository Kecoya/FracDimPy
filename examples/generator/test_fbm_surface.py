#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fractional Brownian Motion (FBM) Surface Generation Example
===========================================================

This example demonstrates how to generate Fractional Brownian Motion (FBM) surfaces
using fracDimPy. FBM surfaces are two-dimensional random fractal surfaces with
self-similarity and isotropy, widely used in terrain generation, texture synthesis,
elevation simulation, and other fields.

Main Features:
- Generate FBM surfaces with specified fractal dimension
- Visualize in three ways: 2D heatmap, 3D surface, and contour plot
- Save result images

Theoretical Background:
- Relationship between FBM surface fractal dimension D and Hurst exponent H: D = 3 - H
- D ∈ (2, 3), larger D indicates rougher surfaces
- H ∈ (0, 1), larger H indicates smoother surfaces
"""

import numpy as np
import os
import matplotlib.pyplot as plt
# Try to use scienceplots style if available
try:
    import scienceplots
    plt.style.use(['science','no-latex'])
except ImportError:
    pass
# Set font family: Times New Roman for English, Microsoft YaHei for Chinese
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

def main():
    print("="*60)
    print("Fractional Brownian Motion Surface Generation Example")
    print("="*60)
    
    try:
        from fracDimPy import generate_fbm_surface
        
        # 1. Generate FBM surface
        print("\n1. Generating FBM surface...")
        dimension = 2.3  # Target fractal dimension
        size = 256       # Surface size (number of pixels)
        
        surface = generate_fbm_surface(dimension=dimension, size=size)
        
        H = 3 - dimension  # Calculate Hurst exponent
        print(f"   Fractal dimension D: {dimension}")
        print(f"   Hurst exponent H: {H:.2f}")
        print(f"   Surface size: {size} x {size}")
        print(f"   Value range: {surface.min():.4f} ~ {surface.max():.4f}")
        
        # 2. Visualize FBM surface
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(15, 5))
            
            # 2D heatmap view
            ax1 = fig.add_subplot(131)
            im = ax1.imshow(surface, cmap='terrain')
            ax1.set_title(f'FBM Surface-2D View (D={dimension}, H={H:.2f})')
            ax1.set_xlabel('X Coordinate')
            ax1.set_ylabel('Y Coordinate')
            plt.colorbar(im, ax=ax1, label='Height Value')
            
            # 3D surface view
            ax2 = fig.add_subplot(132, projection='3d')
            X, Y = np.meshgrid(range(size), range(size))
            # Downsample to improve rendering speed
            step = max(1, size // 50)
            ax2.plot_surface(X[::step, ::step], Y[::step, ::step], 
                           surface[::step, ::step], cmap='terrain', alpha=0.9)
            ax2.set_title('FBM Surface-3D View')
            ax2.set_xlabel('X Coordinate')
            ax2.set_ylabel('Y Coordinate')
            ax2.set_zlabel('Height Value')
            
            # Contour plot view
            ax3 = fig.add_subplot(133)
            # Draw contour lines (black lines)
            contour_lines = ax3.contour(surface, levels=15, colors='black', 
                                       linewidths=0.8, alpha=0.6)
            # Add contour labels
            ax3.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
            # Draw filled contours (colored regions)
            contour_filled = ax3.contourf(surface, levels=15, cmap='terrain', alpha=0.7)
            ax3.set_title('FBM Surface-Contour Plot')
            ax3.set_xlabel('X Coordinate')
            ax3.set_ylabel('Y Coordinate')
            plt.colorbar(contour_filled, ax=ax3, label='Height Value')
            
            plt.tight_layout()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_file = os.path.join(current_dir, "result_fbm_surface.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\n2. Visualization result saved: {output_file}")
            plt.show()
            
        except ImportError:
            print("\n2. Visualization failed: matplotlib library required")
        
    except ImportError:
        print("\nError: Related libraries required to generate FBM surfaces")
    
    print("\nExample execution completed!")


if __name__ == '__main__':
    main()

