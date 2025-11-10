#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 - 
==================




1. 0/1 [0,1,0,0,1,1,0,...]
2.  [1.5, 3.2, 7.8, ...]
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from fracDimPy import box_counting

#  scienceplots 
try:
    import scienceplots
except ImportError:
    pass

# 
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "box_counting_scatter_data.xlsx")

def main():
    print("="*60)
    print(" - ")
    print("="*60)
    
    # 1. 
    print(f"\n1. : {data_file}")
    df = pd.read_excel(data_file, header=None)
    print(f"   : {df.shape}")
    
    # 
    scatter_data = df.iloc[:, 0].values
    print(f"   : {len(scatter_data)}")
    print(f"   : {scatter_data.min():.4f} ~ {scatter_data.max():.4f}")
    
    # 
    is_binary = np.all(np.isin(scatter_data, [0, 1]))
    if is_binary:
        print(f"   : 0/1")
        print(f"   : {np.sum(scatter_data)}")
    else:
        print(f"   : ")
        print(f"   ")
    
    # 2. 
    print("\n2. ...")
    D, result = box_counting(scatter_data, data_type='scatter')
    
    # 3. 
    print("\n3. :")
    print(f"    D: {D:.4f}")
    print(f"    R: {result['R2']:.4f}")
    
    # 4. 
    print("\n4. ...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 
    if is_binary:
        # 
        positions = np.where(scatter_data == 1)[0]
        axes[0].scatter(positions, np.ones_like(positions), s=5, alpha=0.6)
        axes[0].set_ylim([0.5, 1.5])
        axes[0].set_title('')
        axes[0].set_xlabel('')
        axes[0].set_ylabel('')
        axes[0].grid(True, alpha=0.3, axis='x')
        axes[0].set_yticks([])
    else:
        # 
        axes[0].scatter(scatter_data, np.ones_like(scatter_data), s=5, alpha=0.6)
        axes[0].set_ylim([0.5, 1.5])
        axes[0].set_title('')
        axes[0].set_xlabel('')
        axes[0].set_ylabel('')
        axes[0].grid(True, alpha=0.3, axis='x')
        axes[0].set_yticks([])
    
    # 
    if 'epsilon_values' in result and 'N_values' in result:
        axes[1].loglog(result['epsilon_values'], result['N_values'], 'mo', 
                      markersize=6, label='')
        
        # 
        if 'coefficients' in result:
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = np.exp(b) * np.array(result['epsilon_values'])**(-a)
            axes[1].loglog(result['epsilon_values'], fit_line, 'r-', linewidth=2,
                          label=f' (D={D:.4f})')
        
        axes[1].set_xlabel(' ()', fontsize=12)
        axes[1].set_ylabel('N() ()', fontsize=12)
        axes[1].set_title('', fontsize=12)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3, which='both')
    
    # 
    if 'log_inv_epsilon' in result and 'log_N' in result:
        # 
        axes[2].plot(result['log_inv_epsilon'], result['log_N'], 'r^', 
                    label='', markerfacecolor='white', markersize=8, markeredgewidth=1.5)
        
        # 
        if 'coefficients' in result:
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = a * result['log_inv_epsilon'] + b
            axes[2].plot(result['log_inv_epsilon'], fit_line, 'c-', linewidth=2, label='')
            
            # 
            x_min, x_max = np.min(result['log_inv_epsilon']), np.max(result['log_inv_epsilon'])
            y_min, y_max = np.min(result['log_N']), np.max(result['log_N'])
            
            equation_text = (
                r'$\ln(N) = {:.4f} \ln(\frac{{1}}{{r}}) + {:.4f}$'.format(a, b) + '\n' +
                r'$R^2 = {:.6f} \qquad D = {:.4f}$'.format(result["R2"], a)
            )
            
            axes[2].text(
                x_min + 0.05 * (x_max - x_min),
                y_max - 0.15 * (y_max - y_min),
                equation_text,
                fontsize=11,
                bbox={'facecolor': 'blue', 'alpha': 0.2}
            )
        
        # 
        axes[2].set_xlabel(r'$ \ln ( \frac{1}{\epsilon} ) $', fontsize=12)
        axes[2].set_ylabel(r'$ \ln ( N_{\epsilon} )$', fontsize=12)
        axes[2].set_title('', fontsize=12)
        axes[2].legend(loc='lower right')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(current_dir, "result_box_counting_scatter.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   : {output_file}")
    plt.show()
    
    print("\n")


if __name__ == '__main__':
    main()
