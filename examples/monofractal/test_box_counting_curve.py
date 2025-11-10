#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 - 
===================


"""

import numpy as np
import pandas as pd
import os
from fracDimPy import box_counting

#  scienceplots 
try:
    import scienceplots
    # plt.style.use(['ieee'])  # 
except ImportError:
    pass
import matplotlib.pyplot as plt

# Microsoft YaHeiTimes New Roman
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 

# 
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "box_counting_curve_data.xlsx")

def main():
    print("="*60)
    print(" - ")
    print("="*60)
    
    # 1. 
    print(f"\n1. : {data_file}")
    df = pd.read_excel(data_file)
    print(f"   : {df.shape}")
    print(f"   : {df.columns.tolist()}")
    
    # XY
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    
    print(f"   : {len(x)}")
    print(f"   X: {x.min():.4f} ~ {x.max():.4f}")
    print(f"   Y: {y.min():.4f} ~ {y.max():.4f}")
    
    # 2. 
    print("\n2. ...")
    D, result = box_counting((x, y), data_type='curve')
    
    # 3. 
    print("\n3. :")
    print(f"    D: {D:.4f}")
    print(f"    R: {result['R2']:.4f}")
    
    # 4. 
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    #  - 
    axes[0].scatter(x, y, s=1, alpha=0.6)
    axes[0].set_title('')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].grid(True, alpha=0.3)

    # log-log
    if 'epsilon_values' in result and 'N_values' in result:
        axes[1].loglog(result['epsilon_values'], result['N_values'], 'o', markersize=6, label='')
        # 
        if 'coefficients' in result:
            # : log(N) = a*log(1/) + b  =>  N = exp(b) * ^(-a)
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = np.exp(b) * np.array(result['epsilon_values'])**(-a)
            axes[1].loglog(result['epsilon_values'], fit_line, 'r-', linewidth=2,
                         label=f' (D={D:.4f})')
            
            # 
            C = np.exp(b)
            equation_text = f'$N(\\varepsilon) = {C:.2e} \\cdot \\varepsilon^{{-{a:.4f}}}$\n$R^2 = {result["R2"]:.4f}$'
            axes[1].text(0.05, 0.95, equation_text, transform=axes[1].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        axes[1].set_xlabel(' ()')
        axes[1].set_ylabel('N() ()')
        axes[1].set_title(' ()')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    #  (log)
    if 'log_inv_epsilon' in result and 'log_N' in result:
        axes[2].plot(result['log_inv_epsilon'], result['log_N'], 'o', markersize=6, label='')
        if 'coefficients' in result:
            # 
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = a * result['log_inv_epsilon'] + b
            axes[2].plot(result['log_inv_epsilon'], fit_line, 'r-', linewidth=2,
                       label=f' (={a:.4f})')
            
            # 
            equation_text = f'$\\log(N) = {a:.4f} \\cdot \\log(1/\\varepsilon) + {b:.4f}$\n$R^2 = {result["R2"]:.4f}$\n$D = {a:.4f}$'
            axes[2].text(0.05, 0.95, equation_text, transform=axes[2].transAxes,
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        axes[2].set_xlabel('log(1/)')
        axes[2].set_ylabel('log(N)')
        axes[2].set_title('')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(current_dir, "result_box_counting_curve.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n4. : {output_file}")
    plt.show()
    
    print("\n")


if __name__ == '__main__':
    main()

