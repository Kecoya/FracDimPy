#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Diffusion-Limited Aggregation (DLA) Generation Example
======================================================

This example demonstrates how to generate Diffusion-Limited Aggregation (DLA) structures
using fracDimPy. DLA is a process that simulates particles performing random walks via
Brownian motion and attaching to form aggregates, widely used in modeling natural phenomena
such as crystal growth, electrodeposition, and snowflake formation.

Main Features:
- Generate DLA structures with specified size and particle count
- Visualize complete DLA morphology and local zoomed regions
- Statistics on particle attachment

Theoretical Background:
- DLA structures have fractal characteristics, showing dendritic or snowflake-like morphology
- Particles perform random walks from far away, attaching once they contact existing aggregates
- Fractal dimension of DLA is approximately 1.71 (2D case)
- Has strong anisotropy and randomness
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
    print("Diffusion-Limited Aggregation (DLA) Generation Example")
    print("="*60)
    
    from fracDimPy import generate_dla
    
    # 1. Generate DLA structure
    print("\n1. Generating DLA structure...")
    size = 200           # Grid size
    particles = 100000   # Total number of particles
    
    print(f"   Grid size: {size} x {size}")
    print(f"   Total particles: {particles}")
    print("   Please wait during generation...")
    
    dla = generate_dla(size=size, num_particles=particles)
    
    occupied = np.sum(dla > 0)
    print(f"   Successfully attached particles: {occupied}")
    print(f"   Attachment success rate: {occupied/particles*100:.2f}%")
    
    # 2. Visualize DLA structure
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left plot: Display complete DLA structure
        axes[0].imshow(dla, cmap='hot', interpolation='nearest')
        axes[0].set_title(f'Complete DLA Structure (Particles={particles})')
        axes[0].axis('off')
        
        # Right plot: Display zoomed view of center region
        center = size // 2
        zoom_size = size // 4
        dla_zoom = dla[center-zoom_size:center+zoom_size, 
                       center-zoom_size:center+zoom_size]
        axes[1].imshow(dla_zoom, cmap='hot', interpolation='nearest')
        axes[1].set_title('DLA Center Region Zoomed View')
        axes[1].axis('off')
        
        plt.tight_layout()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(current_dir, "result_dla.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n2. Visualization result saved: {output_file}")
        plt.show()
        
    except ImportError:
        print("\n2. Visualization failed: matplotlib library required")
    
    print("\nExample execution completed!")


if __name__ == '__main__':
    main()

