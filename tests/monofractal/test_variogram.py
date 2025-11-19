#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Variogram Method Test Example
==============================

This example demonstrates how to use fracDimPy's variogram method to estimate
the fractal dimension of data. The variogram method estimates fractal characteristics
by analyzing spatial variability at different scales, and is widely used in
geostatistics, terrain analysis, and other fields.

Main Features:
- Supports 1D sequences and 2D surface data
- Calculates variogram and fits power-law relationship
- Estimates fractal dimension from power-law exponent
- Visualizes scale relationship of variogram

Theoretical Background:
- Variogram γ(h) describes spatial variability at distance h
- For fractal data: γ(h) ∝ h^(2H)
- H is the Hurst exponent, reflecting data smoothness
- Fractal dimension: D = E + 1 - H (E is embedding dimension)
- 1D: D = 2 - H, 2D: D = 3 - H
"""

import numpy as np
import os
from fracDimPy import variogram_method
import matplotlib.pyplot as plt

# Set font for Chinese characters
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Data file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_1d = os.path.join(current_dir, "variogram_1d_data.npy")
data_file_2d = os.path.join(current_dir, "variogram_surface_data.tif")

def main():
    print("="*60)
    print("Variogram Method Test Example")
    print("="*60)
    
    # ========== Test 1: 1D Data ==========
    print("\n[Test 1: 1D Time Series]")
    print(f"Loading data: {data_file_1d}")
    try:
        data_1d = np.load(data_file_1d)
        print(f"   Data length: {len(data_1d)} points")
        print(f"   Value range: {data_1d.min():.4f} ~ {data_1d.max():.4f}")
        
        print("\nCalculating variogram...")
        D_1d, result_1d = variogram(data_1d)
        
        print("\nResults:")
        print(f"   Fractal dimension D: {D_1d:.4f}")
        print(f"   Hurst exponent H: {result_1d['hurst']:.4f}")
        print(f"   Goodness of fit R²: {result_1d['R2']:.4f}")
        
    except Exception as e:
        print(f"   Error: {e}")
        data_1d = None
        result_1d = None
    
    # ========== Test 2: 2D Surface ==========
    print("\n[Test 2: 2D Surface Data]")
    print(f"Loading data: {data_file_2d}")
    try:
        from PIL import Image
        img = Image.open(data_file_2d)
        data_2d = np.array(img)
        
        # Convert to grayscale if multi-channel image
        if len(data_2d.shape) > 2:
            data_2d = np.mean(data_2d, axis=2)
        
        print(f"   Data shape: {data_2d.shape}")
        print(f"   Value range: {data_2d.min():.1f} ~ {data_2d.max():.1f}")
        
        print("\nCalculating variogram...")
        D_2d, result_2d = variogram(data_2d)
        
        print("\nResults:")
        print(f"   Fractal dimension D: {D_2d:.4f}")
        print(f"   Hurst exponent H: {result_2d['hurst']:.4f}")
        print(f"   Goodness of fit R²: {result_2d['R2']:.4f}")
        
    except Exception as e:
        print(f"   Error: {e}")
        data_2d = None
        result_2d = None
    
    # ========== Visualize Results ==========
    print("\nGenerating visualization...")
    
    # Determine number of subplots to draw
    n_plots = sum([data_1d is not None, data_2d is not None])
    if n_plots == 0:
        print("No data loaded successfully, skipping visualization.")
        return
    
    fig = plt.figure(figsize=(15, 5*n_plots))
    plot_idx = 1
    
    # Visualize 1D results
    if data_1d is not None and result_1d is not None:
        # Original data
        ax1 = fig.add_subplot(n_plots, 3, plot_idx)
        ax1.plot(data_1d, linewidth=0.6, color='steelblue')
        ax1.set_title('1D Time Series')
        ax1.set_xlabel('Time Index')
        ax1.set_ylabel('Value')
        ax1.grid(True, alpha=0.3)
        
        # Log-log plot
        ax2 = fig.add_subplot(n_plots, 3, plot_idx+1)
        if 'log_lags' in result_1d and 'log_variogram' in result_1d:
            ax2.plot(result_1d['log_lags'], result_1d['log_variogram'], 
                    'o', label='Observed data', markersize=6, color='blue')
            
            # Draw fitting line
            if 'coefficients' in result_1d:
                a, b = result_1d['coefficients']
                fit_line = a * result_1d['log_lags'] + b
                ax2.plot(result_1d['log_lags'], fit_line, 'r-', 
                        linewidth=2, label=f'Fit (slope={a:.4f})')
            
            ax2.set_xlabel('log(lag) - Lag distance logarithm')
            ax2.set_ylabel('log(γ) - Variogram logarithm')
            ax2.set_title('Variogram Analysis (1D)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Parameter comparison
        ax3 = fig.add_subplot(n_plots, 3, plot_idx+2)
        params = ['Fractal Dim D', 'Hurst Exp H']
        values = [D_1d, result_1d['hurst']]
        colors = ['green', 'orange']
        bars = ax3.bar(params, values, color=colors, alpha=0.7)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom')
        
        ax3.set_ylabel('Parameter Value')
        ax3.set_title(f'Fractal Parameters\nR²={result_1d["R2"]:.4f}')
        ax3.grid(True, alpha=0.3, axis='y')
        
        plot_idx += 3
    
    # Visualize 2D results
    if data_2d is not None and result_2d is not None:
        # Original data
        ax1 = fig.add_subplot(n_plots, 3, plot_idx)
        im = ax1.imshow(data_2d, cmap='terrain')
        ax1.set_title('2D Surface Data')
        ax1.axis('off')
        plt.colorbar(im, ax=ax1, fraction=0.046)
        
        # Log-log plot
        ax2 = fig.add_subplot(n_plots, 3, plot_idx+1)
        if 'log_lags' in result_2d and 'log_variogram' in result_2d:
            ax2.plot(result_2d['log_lags'], result_2d['log_variogram'], 
                    'o', label='Observed data', markersize=6, color='blue')
            
            # Draw fitting line
            if 'coefficients' in result_2d:
                a, b = result_2d['coefficients']
                fit_line = a * result_2d['log_lags'] + b
                ax2.plot(result_2d['log_lags'], fit_line, 'r-', 
                        linewidth=2, label=f'Fit (slope={a:.4f})')
            
            ax2.set_xlabel('log(lag) - Lag distance logarithm')
            ax2.set_ylabel('log(γ) - Variogram logarithm')
            ax2.set_title('Variogram Analysis (2D)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Parameter comparison
        ax3 = fig.add_subplot(n_plots, 3, plot_idx+2)
        params = ['Fractal Dim D', 'Hurst Exp H']
        values = [D_2d, result_2d['hurst']]
        colors = ['green', 'orange']
        bars = ax3.bar(params, values, color=colors, alpha=0.7)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom')
        
        ax3.set_ylabel('Parameter Value')
        ax3.set_title(f'Fractal Parameters\nR²={result_2d["R2"]:.4f}')
        ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = os.path.join(current_dir, "result_variogram.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved: {output_file}")
    plt.show()
    
    print("\nExample completed!")


if __name__ == '__main__':
    main()
