#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 - 
========================


"""

import numpy as np
import os
import matplotlib.pyplot as plt
from fracDimPy import box_counting

#  scienceplots 
try:
    import scienceplots
    # plt.style.use(['ieee'])  # 
except ImportError:
    pass

# Microsoft YaHeiTimes New Roman
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 

# 
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "box_counting_porous_data.npy")

def main():
    print("="*60)
    print(" - ")
    print("="*60)
    
    # 1. 
    print(f"\n1. : {data_file}")
    porous_data = np.load(data_file)
    print(f"   : {porous_data.shape}")
    print(f"   : {porous_data.dtype}")
    print(f"   : {porous_data.min()} ~ {porous_data.max()}")
    print(f"   : {porous_data.size}")
    
    # 
    if porous_data.max() > 1:
        threshold = np.mean(porous_data)
        binary_data = (porous_data > threshold).astype(np.uint8)
        print(f"   : {threshold:.2f}")
    else:
        binary_data = porous_data.astype(np.uint8)
    
    porosity = np.sum(binary_data) / binary_data.size
    print(f"   : {porosity*100:.2f}%")
    
    # 2. 
    print("\n2. ...")
    print("   ...")
    
    D, result = box_counting(binary_data, data_type='porous')
    
    # 3. 
    print("\n3. :")
    print(f"    D: {D:.4f}")
    print(f"    R: {result['R2']:.4f}")
    
    # 4. 
    print("\n4. ...")
    
    # BC-PorousMediaDraw.py
    fig = plt.figure(figsize=(16, 10))
    
    # 3DVoxels
    ax1 = fig.add_subplot(221, projection='3d')
    
    # simple
    epsilon_display = min(binary_data.shape) // 9  # 9x9x9
    if epsilon_display < 1:
        epsilon_display = 1
    
    # 
    def simplify_3d(MT, EPSILON):
        """"""
        MT_BOX_0 = np.add.reduceat(MT, np.arange(0, MT.shape[0], EPSILON), axis=0)
        MT_BOX_1 = np.add.reduceat(MT_BOX_0, np.arange(0, MT.shape[1], EPSILON), axis=1)
        MT_BOX_2 = np.add.reduceat(MT_BOX_1, np.arange(0, MT.shape[2], EPSILON), axis=2)
        return MT_BOX_2
    
    simplified_data = simplify_3d(binary_data, epsilon_display)
    voxel_data = np.where(simplified_data > 0, 1, 0)
    
    ax1.voxels(voxel_data, alpha=0.7, facecolors='#1f77b4', edgecolors='k', linewidth=0.1)
    ax1.set_title(' ()', fontsize=12)
    ax1.axis('off')
    
    # 
    ax2 = fig.add_subplot(222)
    
    if 'epsilon_values' in result and 'N_values' in result:
        # 
        ax2.plot(result['epsilon_values'], result['N_values'], 'bo-', 
                linewidth=2, markersize=6, label='')
        
        ax2.set_xlabel(' ', fontsize=12)
        ax2.set_ylabel(' N()', fontsize=12)
        ax2.set_title('', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 
        n_labels = min(5, len(result['epsilon_values']))  # 5
        step = len(result['epsilon_values']) // n_labels
        for i in range(0, len(result['epsilon_values']), step if step > 0 else 1):
            ax2.annotate(f'{result["N_values"][i]}', 
                        xy=(result['epsilon_values'][i], result['N_values'][i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
    
    # 
    ax3 = fig.add_subplot(223)
    
    if 'epsilon_values' in result and 'N_values' in result:
        # 
        ax3.loglog(result['epsilon_values'], result['N_values'], 'mo', 
                  markersize=6, label='')
        
        # 
        if 'coefficients' in result:
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = np.exp(b) * np.array(result['epsilon_values'])**(-a)
            ax3.loglog(result['epsilon_values'], fit_line, 'r-', linewidth=2,
                      label=f' (D={D:.4f})')
        
        ax3.set_xlabel(' ()', fontsize=12)
        ax3.set_ylabel('N() ()', fontsize=12)
        ax3.set_title('', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3, which='both')
    
    # 
    ax4 = fig.add_subplot(224)
    
    if 'log_inv_epsilon' in result and 'log_N' in result:
        # 
        ax4.grid(which="major", axis="both", alpha=0.3)
        
        # 
        ax4.plot(result['log_inv_epsilon'], result['log_N'], 'r^', 
                label='', markerfacecolor='white', markersize=8, markeredgewidth=1.5)
        
        # 
        if 'coefficients' in result:
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = a * result['log_inv_epsilon'] + b
            ax4.plot(result['log_inv_epsilon'], fit_line, 'c-', linewidth=2, label='')
            
            # LaTeX
            x_min, x_max = np.min(result['log_inv_epsilon']), np.max(result['log_inv_epsilon'])
            y_min, y_max = np.min(result['log_N']), np.max(result['log_N'])
            
            equation_text = (
                r'$\ln(N_r) = {:.4f} \ln(\frac{{1}}{{r}}) + {:.4f}$'.format(a, b) + '\n' +
                r'$R^2 = {:.6f} \qquad D = {:.4f}$'.format(result["R2"], a)
            )
            
            ax4.text(
                x_min + 0.05 * (x_max - x_min),
                y_max - 0.15 * (y_max - y_min),
                equation_text,
                fontsize=11,
                bbox={'facecolor': 'blue', 'alpha': 0.2}
            )
        
        # LaTeX
        ax4.set_xlabel(r'$ \ln ( \frac{1}{\epsilon} ) $', fontsize=12)
        ax4.set_ylabel(r'$ \ln ( N_{\epsilon} )$', fontsize=12)
        ax4.set_title('', fontsize=12)
        ax4.legend(loc='lower right')
    
    plt.tight_layout()
    output_file = os.path.join(current_dir, "result_box_counting_porous.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   : {output_file}")
    plt.show()
    
    print("\n")


if __name__ == '__main__':
    main()

