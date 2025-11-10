#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 - XY
========================================


"""

import numpy as np
import pandas as pd
import os
from fracDimPy import multifractal_curve
import random
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
# SciencePlots
import scienceplots 
plt.style.use(['science', 'no-latex'])
# scienceplots
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# Draw_batch_ch.py
mkl = [mk[0] for mk in Line2D.filled_markers]

# 
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "mf_curve_dual_1.xlsx")

def main():
    print("="*60)
    print(" - XY")
    print("="*60)
    
    # 1. 
    print(f"\n1. : {data_file}")
    df = pd.read_excel(data_file)
    print(f"   : {df.shape}")
    print(f"   : {df.columns.tolist()}")
    
    # XY
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    
    print(f"   X: {x.min():.4f} ~ {x.max():.4f}")
    print(f"   Y: {y.min():.4f} ~ {y.max():.4f}")
    
    # 2. 
    print("\n2. ...")
    metrics, figure_data = multifractal_curve(
        (x, y),
        use_multiprocessing=False,
        data_type='dual'
    )
    
    # 3. 
    print("\n3. :")
    print(f"    D(0): {metrics[' D(0)'][0]:.4f}")
    print(f"    D(1): {metrics[' D(1)'][0]:.4f}")
    print(f"    D(2): {metrics[' D(2)'][0]:.4f}")
    print(f"   H: {metrics['H'][0]:.4f}")
    print(f"   : {metrics[''][0]:.4f}")
    
    # 4. 
    try:
        print("\n4. ...")
        
        # 
        ql = figure_data['q']
        tau_q = figure_data['(q)']
        alpha_q = figure_data['(q)']
        f_alpha = figure_data['f()']
        D_q = figure_data['D(q)']
        
        # 2x3
        fig = plt.figure(figsize=(18, 12))
        
        # ========== 1.  ==========
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(x, y, linewidth=1, color='steelblue')
        ax1.set_xlabel('X', fontsize=11)
        ax1.set_ylabel('Y', fontsize=11)
        ax1.set_title('(a) Original Curve', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # ========== 2. X vs ln() ==========
        ax2 = plt.subplot(2, 3, 2)
        temp_q_n = max(1, int(len(ql) / 20))  # 20q
        plotted_count = 0
        for i, q_val in enumerate(ql):
            key = f'q={q_val}_X'
            key_r = f'q={q_val}_r'
            
            if key in figure_data and key_r in figure_data:
                if i % temp_q_n == 0:
                    colors = np.random.rand(3,)
                    log_r = figure_data[key_r]
                    log_X = figure_data[key]
                    
                    ax2.plot(log_r, log_X, 
                            marker=random.choice(mkl),
                            label=f'$q={q_val:.2f}$',
                            linestyle='none',
                            color=colors,
                            markersize=6)
                    
                    # 
                    coeffs = np.polyfit(log_r, log_X, 1)
                    fit_line = np.poly1d(coeffs)
                    ax2.plot(log_r, fit_line(log_r), color=colors, linewidth=1.5)
                    
                    plotted_count += 1
        
        if plotted_count > 0:
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
        ax2.set_xlabel(r'$\ln(\epsilon)$', fontsize=11)
        ax2.set_ylabel(r'$\ln(X)$', fontsize=11)
        ax2.set_title('(b) Partition Function', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # ========== 3.  (q) vs q ==========
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(ql, tau_q, 'o-', color='darkgreen', linewidth=2, markersize=4)
        ax3.set_xlabel(r'$q$', fontsize=11)
        ax3.set_ylabel(r'$\tau(q)$', fontsize=11)
        ax3.set_title('(c) Mass Exponent', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # ========== 4.  (q) vs q ==========
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(ql, alpha_q, 's-', color='crimson', linewidth=2, markersize=4)
        ax4.set_xlabel(r'$q$', fontsize=11)
        ax4.set_ylabel(r'$\alpha(q)$', fontsize=11)
        ax4.set_title(r'(d) Hlder Exponent', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # ========== 5.  f() vs  ==========
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(alpha_q, f_alpha, '^-', color='darkorange', linewidth=2, markersize=4)
        ax5.set_xlabel(r'$\alpha$', fontsize=11)
        ax5.set_ylabel(r'$f(\alpha)$', fontsize=11)
        ax5.set_title('(e) Multifractal Spectrum', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 
        idx_0 = ql.index(0) if 0 in ql else len(ql)//2
        if idx_0 < len(alpha_q):
            ax5.plot(alpha_q[idx_0], f_alpha[idx_0], 'ro', markersize=8, label='q=0')
            ax5.legend(fontsize=9)
        
        # ========== 6.  D(q) vs q ==========
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(ql, D_q, 'd-', color='mediumpurple', linewidth=2, markersize=4)
        ax6.set_xlabel(r'$q$', fontsize=11)
        ax6.set_ylabel(r'$D(q)$', fontsize=11)
        ax6.set_title('(f) Generalized Dimension', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # D(0), D(1), D(2)
        for q_val in [0, 1, 2]:
            if q_val in ql:
                idx = ql.index(q_val)
                ax6.plot(q_val, D_q[idx], 'o', markersize=8)
                ax6.text(q_val, D_q[idx], f'  D({q_val})={D_q[idx]:.3f}', 
                        fontsize=8, verticalalignment='bottom')
        
        plt.tight_layout()
        output_file = os.path.join(current_dir, "result_mf_comprehensive.png")
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n   : result_mf_comprehensive.png")
        plt.close(fig)
        
    except Exception as e:
        print(f"\n4. : {e}")
        import traceback
        traceback.print_exc()
    
    print("\n")


if __name__ == '__main__':
    main()

