#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

======================


"""

import numpy as np
import matplotlib.pyplot as plt
from fracDimPy import multifractal_curve, multifractal_image
import random
from matplotlib.lines import Line2D

# 
np.random.seed(42)
random.seed(42)

# 
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass

plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 
mkl = [mk[0] for mk in Line2D.filled_markers]


def demo_single_curve():
    """"""
    print("="*60)
    print("1: ")
    print("="*60)
    
    # 
    n = 1024
    H = 0.7  # Hurst
    data = np.cumsum(np.random.randn(n)) * (n ** H)
    
    print(f"\n: {len(data)}")
    print(f": {data.min():.4f} ~ {data.max():.4f}")
    
    # 
    print("\n...")
    metrics, figure_data = multifractal_curve(
        data,
        use_multiprocessing=False,
        data_type='single'
    )
    
    # 
    print("\n:")
    print(f"   D(0): {metrics[' D(0)'][0]:.4f}")
    print(f"   D(1): {metrics[' D(1)'][0]:.4f}")
    print(f"   D(2): {metrics[' D(2)'][0]:.4f}")
    print(f"  H: {metrics['H'][0]:.4f}")
    print(f"  : {metrics[''][0]:.4f}")
    
    # 
    print("\n...")
    visualize_results(data, figure_data, "demo_single_curve.png", data_type='single')
    print(f"  : demo_single_curve.png")


def demo_dual_curve():
    """XY"""
    print("\n" + "="*60)
    print("2: XY")
    print("="*60)
    
    # XY
    n = 1024
    x = np.linspace(0, 10, n)
    #  + 
    y = np.sin(2 * np.pi * x) + np.cumsum(np.random.randn(n)) * 0.1
    
    print(f"\n: {len(x)}")
    print(f"X: {x.min():.4f} ~ {x.max():.4f}")
    print(f"Y: {y.min():.4f} ~ {y.max():.4f}")
    
    # 
    print("\n...")
    metrics, figure_data = multifractal_curve(
        (x, y),
        use_multiprocessing=False,
        data_type='dual'
    )
    
    # 
    print("\n:")
    print(f"   D(0): {metrics[' D(0)'][0]:.4f}")
    print(f"   D(1): {metrics[' D(1)'][0]:.4f}")
    print(f"   D(2): {metrics[' D(2)'][0]:.4f}")
    print(f"  H: {metrics['H'][0]:.4f}")
    print(f"  : {metrics[''][0]:.4f}")
    
    # 
    print("\n...")
    visualize_results((x, y), figure_data, "demo_dual_curve.png", data_type='dual')
    print(f"  : demo_dual_curve.png")


def demo_image():
    """"""
    print("\n" + "="*60)
    print("3: ")
    print("="*60)
    
    # 
    size = 256
    # Diamond-Square
    img = generate_fractal_image(size)
    
    print(f"\n: {img.shape}")
    print(f": {img.min():.2f} ~ {img.max():.2f}")
    
    # 
    print("\n...")
    metrics, figure_data = multifractal_image(img)
    
    # 
    print("\n:")
    print(f"   D(0): {metrics[' D(0)'][0]:.4f}")
    print(f"   D(1): {metrics[' D(1)'][0]:.4f}")
    print(f"   D(2): {metrics[' D(2)'][0]:.4f}")
    print(f"  H: {metrics['H'][0]:.4f}")
    print(f"  : {metrics[''][0]:.4f}")
    
    # 
    print("\n...")
    visualize_results(img, figure_data, "demo_image.png", data_type='image')
    print(f"  : demo_image.png")


def generate_fractal_image(size):
    """Diamond-Square"""
    # 
    img = np.zeros((size, size))
    for scale in [64, 32, 16, 8, 4, 2]:
        noise = np.random.randn(size // scale, size // scale)
        # 
        from scipy.ndimage import zoom
        noise_scaled = zoom(noise, scale, order=1)
        if noise_scaled.shape != img.shape:
            noise_scaled = noise_scaled[:size, :size]
        img += noise_scaled * scale
    
    # 0-255
    img = (img - img.min()) / (img.max() - img.min()) * 255
    return img


def visualize_results(data, figure_data, filename, data_type='single'):
    """"""
    # 
    ql = figure_data['q']
    tau_q = figure_data['(q)']
    alpha_q = figure_data['(q)']
    f_alpha = figure_data['f()']
    D_q = figure_data['D(q)']
    
    # 
    fig = plt.figure(figsize=(18, 12))
    
    # ========== 1.  ==========
    ax1 = plt.subplot(2, 3, 1)
    if data_type == 'single':
        ax1.plot(data, linewidth=1, color='steelblue')
        ax1.set_xlabel('Index', fontsize=11)
        ax1.set_ylabel('Value', fontsize=11)
        ax1.set_title('(a) Original Data', fontsize=12, fontweight='bold')
    elif data_type == 'dual':
        x, y = data
        ax1.plot(x, y, linewidth=1, color='steelblue')
        ax1.set_xlabel('X', fontsize=11)
        ax1.set_ylabel('Y', fontsize=11)
        ax1.set_title('(a) Original Curve', fontsize=12, fontweight='bold')
    elif data_type == 'image':
        im = ax1.imshow(data, cmap='gray', interpolation='nearest')
        ax1.set_title('(a) Original Image', fontsize=12, fontweight='bold')
        ax1.axis('off')
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    ax1.grid(True, alpha=0.3)
    
    # ========== 2.  ==========
    ax2 = plt.subplot(2, 3, 2)
    temp_q_n = max(1, int(len(ql) / 20))
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
                
                coeffs = np.polyfit(log_r, log_X, 1)
                fit_line = np.poly1d(coeffs)
                ax2.plot(log_r, fit_line(log_r), color=colors, linewidth=1.5)
    
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax2.set_xlabel(r'$\ln(\epsilon)$', fontsize=11)
    ax2.set_ylabel(r'$\ln(X)$', fontsize=11)
    ax2.set_title('(b) Partition Function', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # ========== 3.  ==========
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(ql, tau_q, 'o-', color='darkgreen', linewidth=2, markersize=4)
    ax3.set_xlabel(r'$q$', fontsize=11)
    ax3.set_ylabel(r'$\tau(q)$', fontsize=11)
    ax3.set_title('(c) Mass Exponent', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # ========== 4.  ==========
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(ql, alpha_q, 's-', color='crimson', linewidth=2, markersize=4)
    ax4.set_xlabel(r'$q$', fontsize=11)
    ax4.set_ylabel(r'$\alpha(q)$', fontsize=11)
    ax4.set_title(r'(d) Hlder Exponent', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # ========== 5.  ==========
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(alpha_q, f_alpha, '^-', color='darkorange', linewidth=2, markersize=4)
    ax5.set_xlabel(r'$\alpha$', fontsize=11)
    ax5.set_ylabel(r'$f(\alpha)$', fontsize=11)
    ax5.set_title('(e) Multifractal Spectrum', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    idx_0 = ql.index(0) if 0 in ql else len(ql)//2
    if idx_0 < len(alpha_q):
        ax5.plot(alpha_q[idx_0], f_alpha[idx_0], 'ro', markersize=8, label='q=0')
        ax5.legend(fontsize=9)
    
    # ========== 6.  ==========
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(ql, D_q, 'd-', color='mediumpurple', linewidth=2, markersize=4)
    ax6.set_xlabel(r'$q$', fontsize=11)
    ax6.set_ylabel(r'$D(q)$', fontsize=11)
    ax6.set_title('(f) Generalized Dimension', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    for q_val in [0, 1, 2]:
        if q_val in ql:
            idx = ql.index(q_val)
            ax6.plot(q_val, D_q[idx], 'o', markersize=8)
            ax6.text(q_val, D_q[idx], f'  D({q_val})={D_q[idx]:.3f}', 
                    fontsize=8, verticalalignment='bottom')
    
    plt.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("")
    print("="*60)
    print("\n")
    print("1. demo_single_curve.png - ")
    print("2. demo_dual_curve.png - XY")
    print("3. demo_image.png - ")
    print("\n...\n")
    
    # 
    demo_single_curve()
    demo_dual_curve()
    demo_image()
    
    print("\n" + "="*60)
    print("")
    print("="*60)
    print("\n")
    print("6")
    print("  (a) /")
    print("  (b) ")
    print("  (c) ")
    print("  (d) ")
    print("  (e) ")
    print("  (f) ")
    print("\n: VISUALIZATION_ENHANCEMENT.md\n")

