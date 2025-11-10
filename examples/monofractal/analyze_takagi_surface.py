#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Takagi曲面Box-Counting分形维数分析
========================================
使用box-counting方法计算Takagi曲面的分形维数
生成期刊论文可用级别的矢量图像
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fracDimPy import generate_takagi_surface, box_counting

# 使用SciencePlots样式
try:
    import scienceplots 
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass

plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))


def analyze_takagi_surface(dimension=2.5, level=12, size=256, method=2):
    """
    分析单个Takagi曲面
    
    Parameters
    ----------
    dimension : float
        理论分形维数 (2, 3)
    level : int
        迭代层数
    size : int
        曲面尺寸
    method : int
        box-counting方法 (0-6)
        
    Returns
    -------
    surface : np.ndarray
        生成的Takagi曲面
    D : float
        计算得到的分形维数
    result : dict
        分析结果
    """
    print("="*70)
    print(f"分析Takagi曲面 (理论维数 D={dimension})")
    print("="*70)
    
    # 1. 生成Takagi曲面
    print(f"\n1. 生成Takagi曲面...")
    print(f"   参数: 尺寸={size}x{size}, 迭代层数={level}")
    surface = generate_takagi_surface(dimension=dimension, level=level, size=size)
    
    b = 2 ** dimension / 8
    print(f"   理论分形维数: D = {dimension}")
    print(f"   参数 b = {b:.4f}")
    print(f"   高度范围: {surface.min():.4f} ~ {surface.max():.4f}")
    print(f"   标准差: {surface.std():.4f}")
    
    # 2. 计算分形维数
    print(f"\n2. 使用Box-Counting方法计算分形维数...")
    print(f"   计算方法: method={method}")
    
    D, result = box_counting(surface, data_type='surface', method=method)
    
    # 3. 显示结果
    print(f"\n3. 计算结果:")
    print(f"   测量分形维数: D = {D:.4f}")
    print(f"   拟合优度: R² = {result['R2']:.6f}")
    print(f"   相对误差: {abs(D - dimension) / dimension * 100:.2f}%")
    
    return surface, D, result


def create_publication_figure(surface, dimension, D, result):
    """
    创建期刊论文级别的矢量图
    
    Parameters
    ----------
    surface : np.ndarray
        Takagi曲面数据
    dimension : float
        理论分形维数
    D : float
        测量分形维数
    result : dict
        box-counting分析结果
    """
    print("\n4. 生成期刊级别矢量图...")
    
    # 创建2x2布局
    fig = plt.figure(figsize=(16, 14))
    
    ny, nx = surface.shape
    X, Y = np.meshgrid(range(nx), range(ny))
    
    # ========== 1. 3D曲面视图 - 左上 ==========
    ax1 = fig.add_subplot(221, projection='3d')
    
    # 降采样以提高渲染速度
    step = max(1, min(nx, ny) // 50)
    surf = ax1.plot_surface(X[::step, ::step], Y[::step, ::step], 
                            surface[::step, ::step], 
                            cmap='terrain', alpha=0.9, 
                            linewidth=0, antialiased=True, 
                            rcount=50, ccount=50)
    
    ax1.set_title('3D Surface View', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xlabel('X', fontsize=13, labelpad=10)
    ax1.set_ylabel('Y', fontsize=13, labelpad=10)
    ax1.set_zlabel('Height', fontsize=13, labelpad=10)
    ax1.view_init(elev=25, azim=45)
    ax1.tick_params(labelsize=11)
    
    # 添加信息框
    info_text = f'Theoretical D = {dimension}\nMeasured D = {D:.4f}'
    ax1.text2D(0.02, 0.98, info_text, transform=ax1.transAxes,
              fontsize=11, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========== 2. 2D热图视图 - 右上 ==========
    ax2 = fig.add_subplot(222)
    
    im = ax2.imshow(surface, cmap='terrain', aspect='auto', origin='lower')
    ax2.set_title('2D Heatmap View', fontsize=15, fontweight='bold')
    ax2.set_xlabel('X', fontsize=13)
    ax2.set_ylabel('Y', fontsize=13)
    ax2.tick_params(labelsize=11)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Height', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # 添加统计信息
    stats_text = f'Min: {surface.min():.4f}\nMax: {surface.max():.4f}\nStd: {surface.std():.4f}'
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== 3. 对数-对数图 - 左下 ==========
    if 'epsilon_values' in result and 'N_values' in result:
        ax3 = fig.add_subplot(223)
        
        # 绘制数据点
        ax3.loglog(result['epsilon_values'], result['N_values'], 'o', 
                  color='steelblue', markersize=8, markeredgewidth=1.5,
                  markerfacecolor='white', label='Data points', zorder=3)
        
        # 绘制拟合线
        if 'coefficients' in result:
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = np.exp(b) * np.array(result['epsilon_values'])**(-a)
            ax3.loglog(result['epsilon_values'], fit_line, 'r-', 
                      linewidth=2.5, label=f'Fit line (D={D:.4f})', zorder=2)
        
        ax3.set_xlabel(r'Box size $\epsilon$', fontsize=14)
        ax3.set_ylabel(r'Number of boxes $N(\epsilon)$', fontsize=14)
        ax3.set_title('Log-Log Plot', fontsize=15, fontweight='bold')
        ax3.legend(fontsize=12, loc='best')
        ax3.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
        ax3.tick_params(labelsize=11)
    
    # ========== 4. 线性拟合图 - 右下 ==========
    if 'log_inv_epsilon' in result and 'log_N' in result:
        ax4 = fig.add_subplot(224)
        
        # 绘制数据点
        ax4.plot(result['log_inv_epsilon'], result['log_N'], '^', 
                color='crimson', markersize=9, markeredgewidth=1.5,
                markerfacecolor='white', label='Data points', zorder=3)
        
        # 绘制拟合线
        if 'coefficients' in result:
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = a * result['log_inv_epsilon'] + b
            ax4.plot(result['log_inv_epsilon'], fit_line, '-', 
                    color='darkorange', linewidth=2.5, label='Linear fit', zorder=2)
            
            # 添加拟合方程
            x_min, x_max = np.min(result['log_inv_epsilon']), np.max(result['log_inv_epsilon'])
            y_min, y_max = np.min(result['log_N']), np.max(result['log_N'])
            
            equation_text = (
                r'$\ln(N) = {:.4f} \ln(1/\epsilon) + {:.4f}$'.format(a, b) + '\n' +
                r'$R^2 = {:.6f}$'.format(result["R2"]) + '\n' +
                r'$D = {:.4f}$'.format(D)
            )
            
            ax4.text(0.05, 0.95, equation_text,
                    transform=ax4.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', 
                             alpha=0.8, edgecolor='blue', linewidth=1.5))
        
        ax4.set_xlabel(r'$\ln(1/\epsilon)$', fontsize=14)
        ax4.set_ylabel(r'$\ln(N(\epsilon))$', fontsize=14)
        ax4.set_title('Linear Fit', fontsize=15, fontweight='bold')
        ax4.legend(fontsize=12, loc='lower right')
        ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax4.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    # 保存为多种格式
    for ext in ['eps', 'pdf', 'png']:
        output_file = os.path.join(current_dir, f"takagi_surface_boxcounting.{ext}")
        if ext == 'eps':
            # EPS不支持透明度
            fig.savefig(output_file, format='eps', dpi=300, bbox_inches='tight')
        else:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   已保存: takagi_surface_boxcounting.{ext}")
    
    plt.close(fig)


def main():
    """主函数"""
    print("="*70)
    print("Takagi曲面 Box-Counting 分形维数分析")
    print("="*70)
    
    # ==========  参数设置  ==========
    THEORETICAL_DIMENSION = 2.1  # 理论分形维数 (2, 3)
    ITERATION_LEVEL = 12         # 迭代层数
    SURFACE_SIZE = 256           # 曲面尺寸
    CALCULATION_METHOD = 2       # Box-counting方法 (SCCM - 最佳方法)
    # ===============================
    
    print(f"\n>>> 参数设置:")
    print(f">>> 理论分形维数: D = {THEORETICAL_DIMENSION}")
    print(f">>> 曲面尺寸: {SURFACE_SIZE} × {SURFACE_SIZE}")
    print(f">>> 迭代层数: {ITERATION_LEVEL}")
    print(f">>> 计算方法: method={CALCULATION_METHOD} (SCCM - Simplified Cubic Cover)")
    
    # 分析Takagi曲面
    surface, D, result = analyze_takagi_surface(
        dimension=THEORETICAL_DIMENSION,
        level=ITERATION_LEVEL,
        size=SURFACE_SIZE,
        method=CALCULATION_METHOD
    )
    
    # 生成期刊级别矢量图
    create_publication_figure(surface, THEORETICAL_DIMENSION, D, result)
    
    print("\n" + "="*70)
    print("分析完成！")
    print("="*70)
    print(f"理论分形维数: D = {THEORETICAL_DIMENSION}")
    print(f"测量分形维数: D = {D:.4f}")
    print(f"绝对误差: ΔD = {abs(D - THEORETICAL_DIMENSION):.4f}")
    print(f"相对误差: {abs(D - THEORETICAL_DIMENSION) / THEORETICAL_DIMENSION * 100:.2f}%")
    print(f"拟合优度: R² = {result['R2']:.6f}")
    print("="*70)
    print("\n生成的文件:")
    print("  - takagi_surface_boxcounting.eps (矢量图)")
    print("  - takagi_surface_boxcounting.pdf (矢量图)")
    print("  - takagi_surface_boxcounting.png (位图)")
    print("\n期刊论文可用级别的矢量图已生成！")


if __name__ == '__main__':
    main()

