#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fractional Brownian Motion (FBM) Curve Generation Example
=========================================================

This example demonstrates how to generate Fractional Brownian Motion (FBM) curves
using fracDimPy. FBM is an important fractal curve with self-similarity and long-range
correlation properties, widely used in financial time series, terrain modeling, and
other fields.

Main Features:
- Generate FBM curves with specified fractal dimension
- Visualize the generated curves
- Save result images

Theoretical Background:
- Relationship between FBM fractal dimension D and Hurst exponent H: D = 2 - H
- D ∈ (1, 2), larger D indicates more irregular curves
- H ∈ (0, 1), larger H indicates smoother curves
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
    print("Fractional Brownian Motion Curve Generation Example")
    print("="*60)
    
    try:
        from fracDimPy import generate_fbm_curve
        
        # 1. Generate FBM curve
        print("\n1. Generating FBM curve...")
        target_D = 1.5  # Target fractal dimension
        length = 2048   # Curve length (number of sampling points)
        
        curve, D_set = generate_fbm_curve(dimension=target_D, length=length)
        
        print(f"   Target fractal dimension: {target_D}")
        print(f"   Curve length: {length}")
        print(f"   Value range: {curve.min():.4f} ~ {curve.max():.4f}")
        
        # 2. Visualize FBM curve
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # Plot FBM curve
            ax.plot(curve, linewidth=0.8, color='steelblue')
            ax.set_title(f'Fractional Brownian Motion Curve (Fractal Dimension D={target_D})')
            ax.set_xlabel('Index Position')
            ax.set_ylabel('Curve Value')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_file = os.path.join(current_dir, "result_fbm_curve.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\n2. Visualization result saved: {output_file}")
            plt.show()
            
        except ImportError:
            print("\n2. Visualization failed: matplotlib library required")
        
    except ImportError:
        print("\nError: fbm library required to generate fractional Brownian motion")
        print("Install command: pip install fbm")
    
    print("\nExample execution completed!")


if __name__ == '__main__':
    main()

