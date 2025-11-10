#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

===========


"""

import numpy as np
import os
import matplotlib.pyplot as plt
#  scienceplots 
try:
    import scienceplots
    plt.style.use(['science','no-latex'])
except ImportError:
    pass
# Microsoft YaHeiTimes New Roman
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 

def lorenz_attractor(num_steps=10000, dt=0.01):
    """Lorenz"""
    def lorenz_deriv(state, sigma=10, rho=28, beta=8/3):
        x, y, z = state
        return np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
        ])
    
    # 
    state = np.array([1.0, 1.0, 1.0])
    trajectory = [state]
    
    # 4Runge-Kutta
    for _ in range(num_steps):
        k1 = lorenz_deriv(state)
        k2 = lorenz_deriv(state + 0.5 * dt * k1)
        k3 = lorenz_deriv(state + 0.5 * dt * k2)
        k4 = lorenz_deriv(state + dt * k3)
        state = state + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        trajectory.append(state)
    
    return np.array(trajectory)


def henon_map(num_steps=5000, a=1.4, b=0.3):
    """Henon"""
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
    print("")
    print("="*60)
    
    from fracDimPy import correlation_dimension, generate_fbm_curve
    
    test_cases = []
    
    # 1. Lorenz
    print("\n1. Lorenz...")
    print("   ...")
    lorenz_traj = lorenz_attractor(num_steps=10000, dt=0.01)
    # 
    lorenz_traj = lorenz_traj[1000:]
    
    D_lorenz_theory = 2.06  #  2.06
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
        print(f"   : ~{D_lorenz_theory:.2f}")
        print(f"   : {D_lorenz:.4f}")
        print(f"   R: {result_lorenz['r_squared']:.4f}")
    except Exception as e:
        print(f"   : {e}")
    
    # 2. Henon
    print("\n2. Henon...")
    henon_traj = henon_map(num_steps=5000)
    D_henon_theory = 1.26  #  1.26
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
        print(f"   : ~{D_henon_theory:.2f}")
        print(f"   : {D_henon:.4f}")
        print(f"   R: {result_henon['r_squared']:.4f}")
    except Exception as e:
        print(f"   : {e}")
    
    # 
    if test_cases:
        n_cases = len(test_cases)
        fig = plt.figure(figsize=(15, 5*n_cases))
        
        for idx, case in enumerate(test_cases):
            # 
            ax1 = fig.add_subplot(n_cases, 3, idx*3 + 1)
            if case['dims'] == 3:
                # 3D2D
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
                ax1.set_xlabel('')
                ax1.set_ylabel('')
            
            ax1.set_title(f"{case['name']}\n: ~{case['D_theory']:.2f}")
            ax1.grid(True, alpha=0.3)
            
            # log-log
            result = case['result']
            ax2 = fig.add_subplot(n_cases, 3, idx*3 + 2)
            
            # 
            ax2.plot(result['log_radii'], result['log_correlations'], 
                    'o', color='lightgray', label='', markersize=5, alpha=0.5)
            
            # 
            fit_range = result['fit_range']
            ax2.plot(result['log_radii'][fit_range[0]:fit_range[1]], 
                    result['log_correlations'][fit_range[0]:fit_range[1]], 
                    'o', color='blue', label='', markersize=6)
            
            # 
            fit_x = result['log_radii'][fit_range[0]:fit_range[1]]
            fit_line = np.polyval(result['coeffs'], fit_x)
            ax2.plot(fit_x, fit_line, 'r-', 
                    linewidth=2, label=f'D = {case["D_measured"]:.4f}')
            
            ax2.set_xlabel('log( r)')
            ax2.set_ylabel('log( C(r))')
            ax2.set_title(f'\nR = {result["r_squared"]:.4f}')
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # 
            ax3 = fig.add_subplot(n_cases, 3, idx*3 + 3)
            categories = ['', '']
            values = [case['D_theory'], case['D_measured']]
            colors = ['blue', 'orange']
            bars = ax3.bar(categories, values, color=colors, alpha=0.7)
            
            # 
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.4f}',
                        ha='center', va='bottom')
            
            error = abs(case['D_measured'] - case['D_theory'])
            error_pct = error / case['D_theory'] * 100 if case['D_theory'] > 0 else 0
            ax3.set_ylabel('')
            ax3.set_title(f'\n: {error:.4f} ({error_pct:.2f}%)')
            ax3.set_ylim([0, max(values) * 1.2])
            ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(current_dir, "result_correlation_dimension.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n: {output_file}")
        plt.show()
    
    print("\n")
    print("\n")
    print("   Grassberger-Procaccia")
    print("   ")
    print("   ")
    print("   Lorenz ~2.06")
    print("   Henon ~1.26")


if __name__ == '__main__':
    main()

