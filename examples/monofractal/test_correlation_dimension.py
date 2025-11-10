#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Correlation Dimension Method Test Example
==========================================

This example demonstrates how to use the correlation dimension method
to calculate the fractal dimension of point set data.

The correlation dimension is based on the correlation integral, which measures
the probability that two points are within a certain distance of each other.
It is particularly useful for analyzing chaotic attractors and time series data.

Theoretical Background:
- Uses Grassberger-Procaccia algorithm
- Correlation integral: C(r) = (1/N²) Σ Θ(r - |x_i - x_j|)
- For fractal sets: C(r) ∝ r^D
- Correlation dimension D is the slope of log(C(r)) vs log(r)
- Known theoretical values:
  - Lorenz attractor: D ≈ 2.06
  - Henon map: D ≈ 1.26
"""

import numpy as np
import os
import matplotlib.pyplot as plt
# Try to use scienceplots style
try:
    import scienceplots
    plt.style.use(['science','no-latex'])
except ImportError:
    pass

# Set font: Times New Roman for English, Microsoft YaHei for Chinese
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

def lorenz_attractor(num_steps=10000, dt=0.01):
    """Generate trajectory from Lorenz attractor
    
    The Lorenz attractor is a set of chaotic solutions to the Lorenz system
    of differential equations, which exhibits a strange attractor.
    """
    def lorenz_deriv(state, sigma=10, rho=28, beta=8/3):
        x, y, z = state
        return np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])
    
    # Initial state
    state = np.array([1.0, 1.0, 1.0])
    trajectory = [state]
    
    # 4th-order Runge-Kutta integration
    for _ in range(num_steps):
        k1 = lorenz_deriv(state)
        k2 = lorenz_deriv(state + 0.5 * dt * k1)
        k3 = lorenz_deriv(state + 0.5 * dt * k2)
        k4 = lorenz_deriv(state + dt * k3)
        state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        trajectory.append(state)
    
    return np.array(trajectory)


def henon_map(num_steps=5000, a=1.4, b=0.3):
    """Generate trajectory from Henon map
    
    The Henon map is a discrete-time dynamical system that exhibits chaotic behavior.
    """
    x, y = 0, 0
    trajectory = []
    
    for _ in range(num_steps):
        x_new = 1 - a * x**2 + y
        y_new = b * x
        x, y = x_new, y_new
        trajectory.append([x, y])
    
    return np.array(trajectory)


def main():
    print("="*60)
    print("Correlation Dimension Method Test Example")
    print("="*60)
    
    from fracDimPy import correlation_dimension, generate_fbm_curve
    
    test_cases = []
    
    # 1. Lorenz attractor
    print("\n1. Testing Lorenz attractor...")
    print("   Generating trajectory...")
    lorenz_traj = lorenz_attractor(num_steps=10000, dt=0.01)
    # Remove transient period
    lorenz_traj = lorenz_traj[1000:]
    
    D_lorenz_theory = 2.06  # Theoretical dimension ~2.06
    try:
        D_lorenz, result_lorenz = correlation_dimension(
            lorenz_traj, 
            num_points=25,
            max_samples=3000
        )
        test_cases.append({
            'name': 'Lorenz',
            'trajectory': lorenz_traj,
            'D_measured': D_lorenz,
            'D_theory': D_lorenz_theory,
            'result': result_lorenz,
            'dims': 3
        })
        print(f"   Theoretical D: ~{D_lorenz_theory:.2f}")
        print(f"   Measured D: {D_lorenz:.4f}")
        print(f"   R²: {result_lorenz['r_squared']:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. Henon map
    print("\n2. Testing Henon map...")
    henon_traj = henon_map(num_steps=5000)
    D_henon_theory = 1.26  # Theoretical dimension ~1.26
    try:
        D_henon, result_henon = correlation_dimension(
            henon_traj,
            num_points=25,
            max_samples=3000
        )
        test_cases.append({
            'name': 'Henon',
            'trajectory': henon_traj,
            'D_measured': D_henon,
            'D_theory': D_henon_theory,
            'result': result_henon,
            'dims': 2
        })
        print(f"   Theoretical D: ~{D_henon_theory:.2f}")
        print(f"   Measured D: {D_henon:.4f}")
        print(f"   R²: {result_henon['r_squared']:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Visualize results
    if test_cases:
        n_cases = len(test_cases)
        fig = plt.figure(figsize=(15, 5*n_cases))
        
        for idx, case in enumerate(test_cases):
            # Left plot: Trajectory visualization
            ax1 = fig.add_subplot(n_cases, 3, idx*3 + 1)
            if case['dims'] == 3:
                # Project 3D trajectory to 2D (X-Y plane)
                ax1.plot(case['trajectory'][:, 0], case['trajectory'][:, 1], 
                        linewidth=0.5, alpha=0.7)
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
            elif case['dims'] == 2:
                ax1.plot(case['trajectory'][:, 0], case['trajectory'][:, 1], 
                        'o', markersize=1, alpha=0.5)
                ax1.set_xlabel('X')
                ax1.set_ylabel('Y')
            else:
                ax1.plot(case['trajectory'], linewidth=0.5)
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Value')
            
            ax1.set_title(f"{case['name']}\nTheoretical D: ~{case['D_theory']:.2f}")
            ax1.grid(True, alpha=0.3)
            
            # Middle plot: Correlation dimension log-log plot
            result = case['result']
            ax2 = fig.add_subplot(n_cases, 3, idx*3 + 2)
            
            # Plot all data points (gray)
            ax2.plot(result['log_radii'], result['log_correlations'], 
                    'o', color='lightgray', label='All points', markersize=5, alpha=0.5)
            
            # Highlight fitting range (blue)
            fit_range = result['fit_range']
            ax2.plot(result['log_radii'][fit_range[0]:fit_range[1]], 
                    result['log_correlations'][fit_range[0]:fit_range[1]], 
                    'o', color='blue', label='Fitting range', markersize=6)
            
            # Draw fitting line
            fit_x = result['log_radii'][fit_range[0]:fit_range[1]]
            fit_line = np.polyval(result['coeffs'], fit_x)
            ax2.plot(fit_x, fit_line, 'r-', 
                    linewidth=2, label=f'D = {case["D_measured"]:.4f}')
            
            ax2.set_xlabel('log(r)')
            ax2.set_ylabel('log(C(r))')
            ax2.set_title(f'Correlation Dimension\nR² = {result["r_squared"]:.4f}')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Right plot: Comparison bar chart
            ax3 = fig.add_subplot(n_cases, 3, idx*3 + 3)
            categories = ['Theoretical D', 'Measured D']
            values = [case['D_theory'], case['D_measured']]
            colors = ['blue', 'orange']
            bars = ax3.bar(categories, values, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4f}',
                        ha='center', va='bottom')
            
            error = abs(case['D_measured'] - case['D_theory'])
            error_pct = error / case['D_theory'] * 100 if case['D_theory'] > 0 else 0
            ax3.set_ylabel('Dimension Value')
            ax3.set_title(f'Comparison\nError: {error:.4f} ({error_pct:.2f}%)')
            ax3.set_ylim([0, max(values) * 1.2])
            ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(current_dir, "result_correlation_dimension.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: {output_file}")
        plt.show()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("   Correlation dimension uses Grassberger-Procaccia algorithm")
    print("   It measures the scaling of correlation integral C(r)")
    print("   ")
    print("   Known theoretical values:")
    print("   - Lorenz attractor: D ≈ 2.06")
    print("   - Henon map: D ≈ 1.26")


if __name__ == '__main__':
    main()

