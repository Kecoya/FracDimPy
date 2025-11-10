#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Random Walk Generation Example
===============================

This example demonstrates how to generate various types of random walk trajectories
using fracDimPy. Including Brownian motion, Lévy flights, and self-avoiding walks,
these are important stochastic process models widely used in physics, biology,
finance, and other fields.

Main Features:
- Generate Brownian motion trajectories
- Generate Lévy flight trajectories with different parameters
- Generate self-avoiding walk trajectories
- Visualize trajectory paths and density heatmaps
- Compare characteristics of different random walks

Theoretical Background:
Brownian Motion:
- Fractal dimension D = 2 (on 2D plane)
- Step length follows normal distribution, no long-range jumps

Lévy Flight:
- Step length follows power-law distribution, with long-range jumps
- Parameter α ∈ (0, 2] controls jump distance distribution
- Degenerates to Brownian motion when α = 2
- Long-range jumps occur when α < 2

Self-Avoiding Walk:
- Cannot visit already visited positions
- Fractal dimension approximately 4/3 ≈ 1.333 (on 2D plane)
- Models polymer chain behavior
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
    print("Random Walk Generation Example")
    print("="*60)
    
    from fracDimPy import generate_brownian_motion, generate_levy_flight, generate_self_avoiding_walk
    
    # 1. Generate Brownian motion
    print("\n1. Generating Brownian motion...")
    steps_bm = 10000  # Number of steps
    num_paths = 3     # Number of paths
    
    paths_bm, image_bm = generate_brownian_motion(
        steps=steps_bm, 
        size=512, 
        num_paths=num_paths
    )
    
    print(f"   Steps: {steps_bm}, Number of paths: {num_paths}")
    
    # 2. Generate Lévy flights
    print("\n2. Generating Lévy flights...")
    steps_levy = 5000
    alphas = [1.0, 1.5, 2.0]  # Different α parameters
    
    # 3. Generate self-avoiding walks
    print("\n3. Generating self-avoiding walks...")
    steps_saw = 200   # Number of steps (self-avoiding walk is computationally slow, use smaller steps)
    num_saw = 5       # Number of paths to attempt
    
    paths_saw, image_saw = generate_self_avoiding_walk(
        steps=steps_saw,
        size=512,
        num_attempts=num_saw,
        max_retries=5000  # Maximum retry count
    )
    
    print(f"   Steps: {steps_saw}, Successfully generated paths: {len(paths_saw)}")
    
    # Create comprehensive visualization with 2 rows and 5 columns
    fig = plt.figure(figsize=(20, 8))
    
    # Column 1: Brownian motion trajectories
    ax1 = fig.add_subplot(2, 5, 1)
    for i in range(num_paths):
        ax1.plot(paths_bm[i, :, 0], paths_bm[i, :, 1], 
                linewidth=0.5, alpha=0.7, label=f'Path {i+1}')
    ax1.set_title('Brownian Motion\n(Trajectory Plot)')
    ax1.set_xlabel('X Coordinate')
    ax1.set_ylabel('Y Coordinate')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal', adjustable='datalim')
    
    # Column 6: Brownian motion density heatmap
    ax2 = fig.add_subplot(2, 5, 6)
    ax2.imshow(image_bm, cmap='hot', origin='upper')
    ax2.set_title('Brownian Motion\n(Density Heatmap)')
    ax2.axis('off')
    
    # Columns 2-4: Lévy flight trajectories with different α parameters
    for idx, alpha in enumerate(alphas):
        paths_levy, image_levy = generate_levy_flight(
            steps=steps_levy,
            size=512,
            alpha=alpha,
            num_paths=1
        )
        
        print(f"   Lévy flight α={alpha}: Generated {steps_levy} steps")
        
        # Row 1: Trajectory plot
        ax_traj = fig.add_subplot(2, 5, idx+2)
        ax_traj.plot(paths_levy[0, :, 0], paths_levy[0, :, 1], 
                    linewidth=0.5, alpha=0.8, color='blue')
        ax_traj.set_title(f'Lévy Flight\n(α={alpha})')
        ax_traj.set_xlabel('X Coordinate')
        ax_traj.set_ylabel('Y Coordinate')
        ax_traj.grid(True, alpha=0.3)
        ax_traj.set_aspect('equal', adjustable='datalim')
        
        # Row 2: Density heatmap
        ax_img = fig.add_subplot(2, 5, idx+7)
        ax_img.imshow(image_levy, cmap='hot', origin='upper')
        ax_img.set_title(f'Lévy Flight\n(α={alpha})')
        ax_img.axis('off')
    
    # Column 5: Self-avoiding walk trajectories
    ax_saw_traj = fig.add_subplot(2, 5, 5)
    for i, path in enumerate(paths_saw[:3]):  # Only show first 3 paths
        ax_saw_traj.plot(path[:, 0], path[:, 1], 
                        linewidth=0.8, alpha=0.7, label=f'Path {i+1}')
    ax_saw_traj.set_title(f'Self-Avoiding Walk\n(Steps={steps_saw})')
    ax_saw_traj.set_xlabel('X Coordinate')
    ax_saw_traj.set_ylabel('Y Coordinate')
    ax_saw_traj.grid(True, alpha=0.3)
    ax_saw_traj.legend()
    ax_saw_traj.set_aspect('equal', adjustable='datalim')
    
    # Column 10: Self-avoiding walk density heatmap
    ax_saw_img = fig.add_subplot(2, 5, 10)
    ax_saw_img.imshow(image_saw, cmap='hot', origin='upper')
    ax_saw_img.set_title(f'Self-Avoiding Walk\n(Total {len(paths_saw)} paths)')
    ax_saw_img.axis('off')
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "result_random_walk.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n4. Visualization result saved: {output_file}")
    plt.show()
    
    print("\n" + "="*60)
    print("Theoretical Knowledge Supplement")
    print("="*60)
    print("   Brownian Motion:")
    print("   - Standard random walk, fractal dimension D = 2")
    print("   - Step length follows normal distribution")
    print("\n   Lévy Flight:")
    print("   - Parameter α ∈ (0, 2] controls step length distribution, characteristic of Lévy flight")
    print("   - Degenerates to Brownian motion when α = 2")
    print("   - Long-range jumps occur when α < 2")
    print("   - Widely used in animal foraging behavior and other fields")
    print("\n   Self-Avoiding Walk:")
    print("   - Fractal dimension approximately 4/3 ≈ 1.333")
    print("   - Cannot revisit already visited positions (\"no backtracking\")")
    print("   - Models polymer chains, protein folding, etc.")
    print("   - High computational complexity, slow generation speed")
    print("\nExample execution completed!")


if __name__ == '__main__':
    main()
