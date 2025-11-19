#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Information Dimension Method Test Example
==========================================

This example demonstrates how to use the information dimension method
to calculate the fractal dimension of point set data.

The information dimension is based on information entropy and measures
the amount of information needed to specify a point in the set. It is
particularly useful for analyzing chaotic systems and time series data.

Theoretical Background:
- Uses Shannon entropy: I(epsilon) = -Σ p_i log(p_i)
- p_i is the probability of finding a point in box i
- Information dimension D_I is the slope of I(epsilon) vs log(1/epsilon)
- Key properties:
  - D_I <= D_0 (capacity dimension)
  - D_I = D_0 for uniform distributions
  - D_I < D_0 for non-uniform distributions
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


def logistic_map(r, x0=0.1, num_steps=5000, transient=1000):
    """Generate trajectory from logistic map
    
    The logistic map is a polynomial mapping that exhibits chaotic behavior.
    """
    x = x0
    trajectory = []
    
    # Transient period (discard initial points)
    for _ in range(transient):
        x = r * x * (1 - x)
    
    # Collect trajectory points
    for _ in range(num_steps):
        x = r * x * (1 - x)
        trajectory.append(x)
    
    return np.array(trajectory)


def tent_map(mu, x0=0.1, num_steps=5000, transient=1000):
    """Generate trajectory from tent map
    
    The tent map is a piecewise linear map that exhibits chaotic behavior.
    """
    x = x0
    trajectory = []
    
    # Transient period
    for _ in range(transient):
        if x < 0.5:
            x = mu * x
        else:
            x = mu * (1 - x)
    
    # Collect trajectory points
    for _ in range(num_steps):
        if x < 0.5:
            x = mu * x
        else:
            x = mu * (1 - x)
        trajectory.append(x)
    
    return np.array(trajectory)


def henon_map_1d(num_steps=5000, a=1.4, b=0.3, transient=1000):
    """Generate 1D trajectory from Henon map (x component only)
    
    The Henon map is a discrete-time dynamical system that exhibits chaotic behavior.
    """
    x, y = 0, 0
    trajectory = []
    
    # Transient period
    for _ in range(transient):
        x_new = 1 - a * x**2 + y
        y_new = b * x
        x, y = x_new, y_new
    
    # Collect trajectory points
    for _ in range(num_steps):
        x_new = 1 - a * x**2 + y
        y_new = b * x
        x, y = x_new, y_new
        trajectory.append(x)
    
    return np.array(trajectory)


def generate_multifractal_series(n=5000, p=0.3):
    """Generate multifractal time series using multiplicative cascade"""
    # Calculate number of levels
    levels = int(np.log2(n))
    series = np.ones(2**levels)
    
    for level in range(levels):
        step = 2**(levels - level)
        for i in range(0, 2**levels, step):
            # Apply random multiplicative factor
            if np.random.rand() < p:
                series[i:i+step//2] *= 1.5
                series[i+step//2:i+step] *= 0.5
            else:
                series[i:i+step//2] *= 0.5
                series[i+step//2:i+step] *= 1.5
    
    return series[:n]


def main():
    print("="*60)
    print("Information Dimension Method Test Example")
    print("="*60)
    
    from fracDimPy import information_dimension
    
    # Initialize test cases list
    test_cases = []
    
    # 1. Logistic map
    print("\n1. Testing logistic map...")
    logistic_data = logistic_map(r=3.9, num_steps=5000)
    try:
        D_logistic, result_logistic = information_dimension(
            logistic_data, 
            num_points=20,
            min_boxes=5,
            max_boxes=50
        )
        test_cases.append({
            'name': 'Logistic (r=3.9)',
            'data': logistic_data,
            'D_measured': D_logistic,
            'result': result_logistic,
            'description': 'Chaotic'
        })
        print(f"   Information dimension: {D_logistic:.4f}")
        print(f"   R^2: {result_logistic['r_squared']:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. Tent map
    print("\n2. Testing tent map...")
    tent_data = tent_map(mu=1.9, num_steps=5000)
    try:
        D_tent, result_tent = information_dimension(
            tent_data,
            num_points=20,
            min_boxes=5,
            max_boxes=50
        )
        test_cases.append({
            'name': 'Tent (mu=1.9)',
            'data': tent_data,
            'D_measured': D_tent,
            'result': result_tent,
            'description': 'Chaotic'
        })
        print(f"   Information dimension: {D_tent:.4f}")
        print(f"   R^2: {result_tent['r_squared']:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. Henon map
    print("\n3. Testing Henon map...")
    henon_data = henon_map_1d(num_steps=5000)
    try:
        D_henon, result_henon = information_dimension(
            henon_data,
            num_points=20,
            min_boxes=5,
            max_boxes=50
        )
        test_cases.append({
            'name': 'Henon',
            'data': henon_data,
            'D_measured': D_henon,
            'result': result_henon,
            'description': 'Chaotic'
        })
        print(f"   Information dimension: {D_henon:.4f}")
        print(f"   R^2: {result_henon['r_squared']:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 4. Multifractal series
    print("\n4. Testing multifractal series...")
    mf_data = generate_multifractal_series(n=4096)
    try:
        D_mf, result_mf = information_dimension(
            mf_data,
            num_points=20,
            min_boxes=5,
            max_boxes=60
        )
        test_cases.append({
            'name': 'Multifractal',
            'data': mf_data,
            'D_measured': D_mf,
            'result': result_mf,
            'description': 'Multiplicative cascade'
        })
        print(f"   Information dimension: {D_mf:.4f}")
        print(f"   R^2: {result_mf['r_squared']:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Visualize results
    if test_cases:
        n_cases = len(test_cases)
        fig = plt.figure(figsize=(16, 5.5*n_cases))
        
        for idx, case in enumerate(test_cases):
            # Left plot: Time series
            ax1 = fig.add_subplot(n_cases, 3, idx*3 + 1)
            ax1.plot(case['data'][:1000], linewidth=0.8)  # Show first 1000 points
            ax1.set_title(f"{case['name']}\n{case['description']}", fontsize=10)
            ax1.set_xlabel('Time', fontsize=9)
            ax1.set_ylabel('Value', fontsize=9)
            ax1.tick_params(labelsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Middle plot: Return map
            ax_return = fig.add_subplot(n_cases, 3, idx*3 + 2)
            data = case['data']
            if len(data) > 1:
                ax_return.plot(data[:-1], data[1:], 'o', markersize=1, alpha=0.3)
                ax_return.set_xlabel('x(t)', fontsize=9)
                ax_return.set_ylabel('x(t+1)', fontsize=9)
                ax_return.set_title('Return Map', fontsize=10)
                ax_return.tick_params(labelsize=8)
                ax_return.grid(True, alpha=0.3)
            
            # Right plot: Information dimension log-log plot
            result = case['result']
            ax3 = fig.add_subplot(n_cases, 3, idx*3 + 3)
            ax3.plot(result['log_inv_epsilon'], result['information'], 
                    'o', label='Data points', markersize=6, color='blue')
            
            # Draw fitting line
            fit_line = np.polyval(result['coeffs'], result['log_inv_epsilon'])
            ax3.plot(result['log_inv_epsilon'], fit_line, 'r-', 
                    linewidth=2, label=f'D_I = {case["D_measured"]:.4f}')
            
            ax3.set_xlabel('log(1/epsilon)', fontsize=9)
            ax3.set_ylabel('I(epsilon) (Information)', fontsize=9)
            ax3.set_title(f'Information Dimension\nR^2 = {result["r_squared"]:.4f}', fontsize=10)
            ax3.tick_params(labelsize=8)
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(current_dir, "result_information_dimension.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: {output_file}")
        plt.show()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("   Information dimension uses Shannon entropy:")
    print("   I(epsilon) = -Σ p_i log(p_i)")
    print("   where p_i is the probability of finding a point in box i")
    print("   ")
    print("   Key properties:")
    print("   - D_I measures information content")
    print("   - D_I <= D_0 (capacity dimension)")
    print("   - D_I = D_0 for uniform distributions")
    print("   - D_I < D_0 for non-uniform distributions")
    print("\n   For uniform sets: D_I = D_0")
    print("   For non-uniform sets: D_I < D_0")


if __name__ == '__main__':
    main()

