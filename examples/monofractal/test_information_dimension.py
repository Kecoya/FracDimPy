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


def logistic_map(r, x0=0.1, num_steps=5000, transient=1000):
    """Logistic"""
    x = x0
    trajectory = []
    
    # 
    for _ in range(transient):
        x = r * x * (1 - x)
    
    # 
    for _ in range(num_steps):
        x = r * x * (1 - x)
        trajectory.append(x)
    
    return np.array(trajectory)


def tent_map(mu, x0=0.1, num_steps=5000, transient=1000):
    """Tent"""
    x = x0
    trajectory = []
    
    # 
    for _ in range(transient):
        if x < 0.5:
            x = mu * x
        else:
            x = mu * (1 - x)
    
    # 
    for _ in range(num_steps):
        if x < 0.5:
            x = mu * x
        else:
            x = mu * (1 - x)
        trajectory.append(x)
    
    return np.array(trajectory)


def henon_map_1d(num_steps=5000, a=1.4, b=0.3, transient=1000):
    """Henonx"""
    x, y = 0, 0
    trajectory = []
    
    # 
    for _ in range(transient):
        x_new = 1 - a * x**2 + y
        y_new = b * x
        x, y = x_new, y_new
    
    # 
    for _ in range(num_steps):
        x_new = 1 - a * x**2 + y
        y_new = b * x
        x, y = x_new, y_new
        trajectory.append(x)
    
    return np.array(trajectory)


def generate_multifractal_series(n=5000, p=0.3):
    """"""
    # 
    levels = int(np.log2(n))
    series = np.ones(2**levels)
    
    for level in range(levels):
        step = 2**(levels - level)
        for i in range(0, 2**levels, step):
            # 
            if np.random.rand() < p:
                series[i:i+step//2] *= 1.5
                series[i+step//2:i+step] *= 0.5
            else:
                series[i:i+step//2] *= 0.5
                series[i+step//2:i+step] *= 1.5
    
    return series[:n]


def main():
    print("="*60)
    print("")
    print("="*60)
    
    from fracDimPy import information_dimension
    
    # 
    test_cases = []
    
    # 1. Logistic
    print("\n1. Logistic...")
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
            'description': ''
        })
        print(f"   : {D_logistic:.4f}")
        print(f"   R: {result_logistic['r_squared']:.4f}")
    except Exception as e:
        print(f"   : {e}")
    
    # 2. Tent
    print("\n2. Tent...")
    tent_data = tent_map(mu=1.9, num_steps=5000)
    try:
        D_tent, result_tent = information_dimension(
            tent_data,
            num_points=20,
            min_boxes=5,
            max_boxes=50
        )
        test_cases.append({
            'name': 'Tent (=1.9)',
            'data': tent_data,
            'D_measured': D_tent,
            'result': result_tent,
            'description': ''
        })
        print(f"   : {D_tent:.4f}")
        print(f"   R: {result_tent['r_squared']:.4f}")
    except Exception as e:
        print(f"   : {e}")
    
    # 3. Henon
    print("\n3. Henon...")
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
            'description': ''
        })
        print(f"   : {D_henon:.4f}")
        print(f"   R: {result_henon['r_squared']:.4f}")
    except Exception as e:
        print(f"   : {e}")
    
    # 4. 
    print("\n4. ...")
    mf_data = generate_multifractal_series(n=4096)
    try:
        D_mf, result_mf = information_dimension(
            mf_data,
            num_points=20,
            min_boxes=5,
            max_boxes=60
        )
        test_cases.append({
            'name': '',
            'data': mf_data,
            'D_measured': D_mf,
            'result': result_mf,
            'description': ''
        })
        print(f"   : {D_mf:.4f}")
        print(f"   R: {result_mf['r_squared']:.4f}")
    except Exception as e:
        print(f"   : {e}")
    
    # 
    if test_cases:
        n_cases = len(test_cases)
        fig = plt.figure(figsize=(16, 5.5*n_cases))
        
        for idx, case in enumerate(test_cases):
            # 
            ax1 = fig.add_subplot(n_cases, 3, idx*3 + 1)
            ax1.plot(case['data'][:1000], linewidth=0.8)  # 1000
            ax1.set_title(f"{case['name']}\n{case['description']}", fontsize=10)
            ax1.set_xlabel('', fontsize=9)
            ax1.set_ylabel('', fontsize=9)
            ax1.tick_params(labelsize=8)
            ax1.grid(True, alpha=0.3)
            
            # 
            ax_return = fig.add_subplot(n_cases, 3, idx*3 + 2)
            data = case['data']
            if len(data) > 1:
                ax_return.plot(data[:-1], data[1:], 'o', markersize=1, alpha=0.3)
                ax_return.set_xlabel('x(t)', fontsize=9)
                ax_return.set_ylabel('x(t+1)', fontsize=9)
                ax_return.set_title('', fontsize=10)
                ax_return.tick_params(labelsize=8)
                ax_return.grid(True, alpha=0.3)
            
            # 
            result = case['result']
            ax3 = fig.add_subplot(n_cases, 3, idx*3 + 3)
            ax3.plot(result['log_inv_epsilon'], result['information'], 
                    'o', label='', markersize=6, color='blue')
            
            # 
            fit_line = np.polyval(result['coeffs'], result['log_inv_epsilon'])
            ax3.plot(result['log_inv_epsilon'], fit_line, 'r-', 
                    linewidth=2, label=f'D_I = {case["D_measured"]:.4f}')
            
            ax3.set_xlabel('log(1/)', fontsize=9)
            ax3.set_ylabel('I() ()', fontsize=9)
            ax3.set_title(f'\nR = {result["r_squared"]:.4f}', fontsize=10)
            ax3.tick_params(labelsize=8)
            ax3.legend(fontsize=8)
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(current_dir, "result_information_dimension.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n: {output_file}")
        plt.show()
    
    print("\n")
    print("\n")
    print("   ShannonI() = - p_i  log(p_i)")
    print("   ")
    print("   ")
    print("   - ")
    print("   - ")
    print("   - ")
    print("   - ")
    print("\n    D_I D_I  D_0")
    print("   D_I  D_0")
    print("   D_I < D_0")


if __name__ == '__main__':
    main()

