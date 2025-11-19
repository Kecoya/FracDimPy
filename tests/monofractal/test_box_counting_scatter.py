#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Box-counting Method for Scatter Data Test Example
==================================================

This example demonstrates how to use the box-counting method to calculate
the fractal dimension of 1D scatter data.

The scatter data can be:
1. Binary data: 0/1 values [0,1,0,0,1,1,0,...]
2. Continuous values: [1.5, 3.2, 7.8, ...]

The box-counting method works by dividing the data space into boxes of
different sizes and counting how many boxes contain at least one data point.
The fractal dimension is estimated from the scaling relationship between
box size and the number of occupied boxes.
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from fracDimPy import box_counting

# Try to use scienceplots style
try:
    import scienceplots
except ImportError:
    pass

# Set font: Times New Roman for English, Microsoft YaHei for Chinese
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# Data file path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "box_counting_scatter_data.xlsx")

def main():
    print("="*60)
    print("Box-counting Method for Scatter Data Test Example")
    print("="*60)
    
    # 1. Load data
    print(f"\n1. Loading data: {data_file}")
    df = pd.read_excel(data_file, header=None)
    print(f"   Data shape: {df.shape}")
    
    # Extract scatter data from first column
    scatter_data = df.iloc[:, 0].values
    print(f"   Data points: {len(scatter_data)}")
    print(f"   Value range: {scatter_data.min():.4f} ~ {scatter_data.max():.4f}")
    
    # Check if data is binary (0/1) or continuous
    is_binary = np.all(np.isin(scatter_data, [0, 1]))
    if is_binary:
        print(f"   Data type: Binary (0/1)")
        print(f"   Number of 1s: {np.sum(scatter_data)}")
    else:
        print(f"   Data type: Continuous values")
        print(f"   Mean: {scatter_data.mean():.4f}, Std: {scatter_data.std():.4f}")
    
    # 2. Calculate fractal dimension using box-counting
    print("\n2. Calculating fractal dimension...")
    D, result = box_counting(scatter_data, data_type='scatter')
    
    # 3. Display results
    print("\n3. Results:")
    print(f"    Fractal dimension D: {D:.4f}")
    print(f"    Goodness of fit R^2: {result['R2']:.4f}")
    
    # 4. Visualize results
    print("\n4. Generating visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Left plot: Scatter plot
    if is_binary:
        # Plot positions where value is 1
        positions = np.where(scatter_data == 1)[0]
        axes[0].scatter(positions, np.ones_like(positions), s=5, alpha=0.6)
        axes[0].set_ylim([0.5, 1.5])
        axes[0].set_title('Binary Scatter Data')
        axes[0].set_xlabel('Position')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3, axis='x')
        axes[0].set_yticks([])
    else:
        # Plot continuous values
        axes[0].scatter(scatter_data, np.ones_like(scatter_data), s=5, alpha=0.6)
        axes[0].set_ylim([0.5, 1.5])
        axes[0].set_title('Continuous Scatter Data')
        axes[0].set_xlabel('Value')
        axes[0].set_ylabel('Position')
        axes[0].grid(True, alpha=0.3, axis='x')
        axes[0].set_yticks([])
    
    # Middle plot: Log-log plot
    if 'epsilon_values' in result and 'N_values' in result:
        axes[1].loglog(result['epsilon_values'], result['N_values'], 'mo', 
                      markersize=6, label='Data points')
        
        # Draw fitting line
        if 'coefficients' in result:
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = np.exp(b) * np.array(result['epsilon_values'])**(-a)
            axes[1].loglog(result['epsilon_values'], fit_line, 'r-', linewidth=2,
                          label=f'Fit (D={D:.4f})')
        
        axes[1].set_xlabel('epsilon (Box size)', fontsize=12)
        axes[1].set_ylabel('N(epsilon) (Number of boxes)', fontsize=12)
        axes[1].set_title('Box-counting Log-Log Plot', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, which='both')
    
    # Right plot: Linear fit in log space
    if 'log_inv_epsilon' in result and 'log_N' in result:
        # Plot data points
        axes[2].plot(result['log_inv_epsilon'], result['log_N'], 'r^', 
                    label='Data points', markerfacecolor='white', markersize=8, markeredgewidth=1.5)
        
        # Draw fitting line
        if 'coefficients' in result:
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = a * result['log_inv_epsilon'] + b
            axes[2].plot(result['log_inv_epsilon'], fit_line, 'c-', linewidth=2, label='Linear fit')
            
            # Display equation
            x_min, x_max = np.min(result['log_inv_epsilon']), np.max(result['log_inv_epsilon'])
            y_min, y_max = np.min(result['log_N']), np.max(result['log_N'])
            
            equation_text = (
                r'$\ln(N) = {:.4f} \ln(\frac{{1}}{{r}}) + {:.4f}$'.format(a, b) + '\n' +
                r'$R^2 = {:.6f} \qquad D = {:.4f}$'.format(result["R2"], a)
            )
            
            axes[2].text(
                x_min + 0.05 * (x_max - x_min),
                y_max - 0.15 * (y_max - y_min),
                equation_text,
                fontsize=11,
                bbox={'facecolor': 'blue', 'alpha': 0.2}
            )
        
        # Set labels
        axes[2].set_xlabel(r'$ \ln ( \frac{1}{\epsilon} ) $', fontsize=12)
        axes[2].set_ylabel(r'$ \ln ( N_{\epsilon} )$', fontsize=12)
        axes[2].set_title('Linear Fit in Log Space', fontsize=12)
        axes[2].legend(loc='lower right')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(current_dir, "result_box_counting_scatter.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   Visualization saved: {output_file}")
    plt.show()
    
    print("\nExample completed!")


if __name__ == '__main__':
    main()
