#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hurst Exponent (R/S Analysis) Test Example
==========================================

This example demonstrates how to use fracDimPy's Hurst exponent method (R/S analysis)
to estimate the fractal dimension of time series data.
The Hurst exponent reflects the long-range correlation and memory effects of time series,
making it an important tool for analyzing fractal time series.

Main Features:
- Load and analyze time series data
- Calculate Hurst exponent using R/S analysis
- Calculate fractal dimension from Hurst exponent
- Visualize original series and R/S analysis results

Theoretical Background:
- Hurst exponent H ∈ (0, 1)
- H < 0.5: Anti-persistent (mean-reverting)
- H = 0.5: Random walk (no memory)
- H > 0.5: Persistent (trend-following)
- Fractal dimension D = 2 - H (for 1D signals)
- R/S analysis estimates H through rescaled range statistics
"""

import numpy as np
import os
from fracDimPy import hurst_dimension
import matplotlib.pyplot as plt

# Set font for Chinese characters
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

# Data file path
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "hurst_data.npy")

def main():
    print("="*60)
    print("Hurst Exponent (R/S Analysis) Test Example")
    print("="*60)
    
    # 1. Load test data
    print(f"\n1. Loading data: {data_file}")
    data = np.load(data_file)
    print(f"   Data length: {len(data)} points")
    print(f"   Value range: {data.min():.4f} ~ {data.max():.4f}")
    
    # 2. Calculate Hurst exponent and fractal dimension
    print("\n2. Calculating Hurst exponent...")
    D, result = hurst_dimension(data)
    
    # 3. Display calculation results
    print("\n3. Calculation results:")
    print(f"   Fractal dimension D: {D:.4f}")
    print(f"   Hurst exponent H: {result['hurst']:.4f}")
    print(f"   Goodness of fit R^2: {result['R2']:.4f}")
    
    # 4. Visualize results
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left plot: Display original time series
        axes[0].plot(data, linewidth=0.6, color='steelblue')
        axes[0].set_title('Original Time Series')
        axes[0].set_xlabel('Time Index')
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)
        
        # Right plot: R/S analysis log-log plot
        if 'log_r' in result and 'log_rs' in result:
            axes[1].plot(result['log_r'], result['log_rs'], 'o', 
                        label='Observed data points', markersize=6, color='blue')
            # Draw fitting line
            coeffs = result.get('coefficients', [result['hurst'], 0])
            fit_line = coeffs[0] * result['log_r'] + coeffs[1]
            axes[1].plot(result['log_r'], fit_line, 'r-', linewidth=2,
                        label=f'Linear fit (H={result["hurst"]:.4f})')
            axes[1].set_xlabel('log(r) - Time scale logarithm')
            axes[1].set_ylabel('log(R/S) - Rescaled range logarithm')
            axes[1].set_title('R/S Analysis Results')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(current_dir, "result_hurst.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n4. Visualization saved: {output_file}")
        plt.show()
        
    except ImportError:
        print("\n4. Visualization failed: matplotlib library required")
    
    print("\nExample completed!")


if __name__ == '__main__':
    main()
