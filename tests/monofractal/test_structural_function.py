#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Structural Function Method Test Example
=======================================

This example demonstrates how to use the structural function method
to calculate the fractal dimension of 1D curve data.

The structural function method is suitable for self-affine curves
and analyzes the scaling behavior of the structure function.
"""

import numpy as np
import os
from fracDimPy import structural_function
import matplotlib.pyplot as plt
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass

# Set font for Chinese characters
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# Data file path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "structural_function_data.txt")

def main():
    print("="*60)
    print("Structural Function Method Test Example")
    print("="*60)
    
    # 1. Load data
    print(f"\n1. Loading data: {data_file}")
    
    # Load X and Y coordinates
    data = np.loadtxt(data_file)
    
    # Process data format
    if data.ndim == 1:
        # 1D data (only Y values)
        y_data = data
        x_interval = 1.0
        print(f"   Data type: 1D (Y values only)")
        print(f"   X interval: =1.0")
        print(f"   Data points: {len(y_data)}")
        print(f"   Y range: {y_data.min():.4f} ~ {y_data.max():.4f}")
    elif data.ndim == 2 and data.shape[1] >= 2:
        # 2D data (X and Y coordinates)
        x_data = data[:, 0]
        y_data = data[:, 1]
        x_interval = float(x_data[1] - x_data[0])
        print(f"   Data type: 2D (X, Y coordinates)")
        print(f"   Data points: {len(y_data)}")
        print(f"   X interval: {x_interval:.6f}")
        print(f"   Y range: {y_data.min():.4f} ~ {y_data.max():.4f}")
    else:
        raise ValueError(f"Unsupported data format, shape={data.shape}")
    
    # 2. Calculate fractal dimension using structural function method
    print("\n2. Calculating fractal dimension...")
    D, result = structural_function(y_data, x_interval=x_interval)
    
    # 3. Display results
    print("\n3. Results:")
    print(f"    Fractal dimension D: {D:.4f}")
    print(f"    Goodness of fit R^2: {result['R2']:.4f}")
    
    # 4. Visualize results
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left plot: Original curve
        axes[0].plot(y_data)
        axes[0].set_title('Original Curve')
        axes[0].set_xlabel(f'Time (interval={x_interval:.6f})')
        axes[0].set_ylabel('Value')
        axes[0].grid(True)
        
        # Right plot: Structural function log-log plot
        if 'tau_values' in result and 'S_values' in result:
            axes[1].loglog(result['tau_values'], result['S_values'], 'o', label='Data points')
            # Draw fitting line
            if 'coefficients' in result:
                fit_line = np.exp(result['coefficients'][1]) * np.array(result['tau_values'])**result['coefficients'][0]
                axes[1].loglog(result['tau_values'], fit_line, 'r-', label=f'Fit (slope={result["slope"]:.4f})')
            axes[1].set_xlabel('tau (Time lag)')
            axes[1].set_ylabel('S(tau) (Structure function)')
            axes[1].set_title(f'Structural Function (D={D:.4f})')
            axes[1].legend()
            axes[1].grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(current_dir, "result_structural_function.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n4. Visualization saved: {output_file}")
        plt.show()
        
    except ImportError:
        print("\n4. Visualization failed: matplotlib library required")
    
    print("\nExample completed!")


if __name__ == '__main__':
    main()
