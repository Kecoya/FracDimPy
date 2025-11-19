#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Box-counting Method for Curve Data Test Example
===============================================

This example demonstrates how to use the box-counting method to calculate
the fractal dimension of 2D curve data (X, Y coordinates).

The box-counting method is one of the most commonly used methods for
calculating fractal dimensions. It works by covering the curve with boxes
of different sizes and counting how many boxes are needed.
"""

import numpy as np
import pandas as pd
import os
from fracDimPy import box_counting

# Try to use scienceplots style
try:
    import scienceplots
    # plt.style.use(['ieee'])  # Uncomment to use IEEE style
except ImportError:
    pass
import matplotlib.pyplot as plt

# Set font: Times New Roman for English, Microsoft YaHei for Chinese
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

# Data file path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "box_counting_curve_data.xlsx")

def main():
    print("="*60)
    print("Box-counting Method for Curve Data Test Example")
    print("="*60)
    
    # 1. Load data
    print(f"\n1. Loading data: {data_file}")
    df = pd.read_excel(data_file)
    print(f"   Data shape: {df.shape}")
    print(f"   Column names: {df.columns.tolist()}")
    
    # Extract X and Y coordinates
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    
    print(f"   Data points: {len(x)}")
    print(f"   X range: {x.min():.4f} ~ {x.max():.4f}")
    print(f"   Y range: {y.min():.4f} ~ {y.max():.4f}")
    
    # 2. Calculate fractal dimension using box-counting
    print("\n2. Calculating fractal dimension...")
    D, result = box_counting((x, y), data_type='curve')
    
    # 3. Display results
    print("\n3. Results:")
    print(f"    Fractal dimension D: {D:.4f}")
    print(f"    Goodness of fit R^2: {result['R2']:.4f}")
    
    # 4. Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Left plot: Original curve
    axes[0].scatter(x, y, s=1, alpha=0.6)
    axes[0].set_title('Original Curve')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(True, alpha=0.3)

    # Middle plot: log-log plot
    if 'epsilon_values' in result and 'N_values' in result:
        axes[1].loglog(result['epsilon_values'], result['N_values'], 'o', markersize=6, label='Data points')
        # Draw fitting line
        if 'coefficients' in result:
            # Relationship: log(N) = a*log(1/epsilon) + b  =>  N = exp(b) * epsilon^(-a)
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = np.exp(b) * np.array(result['epsilon_values'])**(-a)
            axes[1].loglog(result['epsilon_values'], fit_line, 'r-', linewidth=2,
                         label=f'Fit (D={D:.4f})')
            
            # Display equation
            C = np.exp(b)
            equation_text = f'$N(\\varepsilon) = {C:.2e} \\cdot \\varepsilon^{{-{a:.4f}}}$\n$R^2 = {result["R2"]:.4f}$'
            axes[1].text(0.05, 0.95, equation_text, transform=axes[1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        axes[1].set_xlabel('epsilon (Box size)')
        axes[1].set_ylabel('N(epsilon) (Number of boxes)')
        axes[1].set_title('Box-counting Log-Log Plot')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    # Right plot: Linear scale (log)
    if 'log_inv_epsilon' in result and 'log_N' in result:
        axes[2].plot(result['log_inv_epsilon'], result['log_N'], 'o', markersize=6, label='Data points')
        if 'coefficients' in result:
            # Draw fitting line
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = a * result['log_inv_epsilon'] + b
            axes[2].plot(result['log_inv_epsilon'], fit_line, 'r-', linewidth=2,
                       label=f'Fit (D={a:.4f})')
            
            # Display equation
            equation_text = f'$\\log(N) = {a:.4f} \\cdot \\log(1/\\varepsilon) + {b:.4f}$\n$R^2 = {result["R2"]:.4f}$\n$D = {a:.4f}$'
            axes[2].text(0.05, 0.95, equation_text, transform=axes[2].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        axes[2].set_xlabel('log(1/epsilon)')
        axes[2].set_ylabel('log(N)')
        axes[2].set_title('Linear Fit in Log Space')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(current_dir, "result_box_counting_curve.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n4. Visualization saved: {output_file}")
    plt.show()
    
    print("\nExample completed!")


if __name__ == '__main__':
    main()
