#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Weierstrass-Mandelbrot曲面生成示例
=================================

本示例演示如何使用fracDimPy生成Weierstrass-Mandelbrot（WM）曲面。
WM曲面是WM曲线的二维推广，是一种连续但处处不可微的分形曲面，
通过叠加不同频率和振幅的二维波形而生成，可用于模拟粗糙表面。

主要功能：
- 生成不同分形维数的WM曲面
- 以3D曲面和2D热图两种方式可视化
- 展示分形维数对曲面粗糙度的影响
- 统计曲面的高度分布特征

理论背景：
- WM曲面的分形维数D ∈ (2, 3)
- D越大，曲面越粗糙、起伏越大
- 通过傅里叶级数形式构造：叠加不同频率的正弦波
- 参数λ控制频率的增长速度，通常λ > 1
- 是确定性的分形曲面（与FBM的随机性不同）
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

def main():
    print("="*60)
    print("Weierstrass-Mandelbrot曲面生成示例")
    print("="*60)
    
    from fracDimPy import generate_wm_surface
    
    # 生成WM曲面
    print("\n1. 正在生成Weierstrass-Mandelbrot曲面...")
    dimensions = [2.1, 2.5, 2.9]  # 不同的分形维数
    level = 15                     # 迭代层数
    size = 256                     # 曲面尺寸
    lambda_param = 1.5            # 频率增长参数
    
    fig = plt.figure(figsize=(15, 10))
    
    for idx, dimension in enumerate(dimensions):
        surface = generate_wm_surface(dimension=dimension, level=level, 
                                      size=size, lambda_param=lambda_param)
        
        print(f"   分形维数 D={dimension}: 曲面尺寸{size}x{size}")
        print(f"   数值范围: {surface.min():.4f} ~ {surface.max():.4f}")
        print(f"   标准差: {surface.std():.4f}")
        
        # 3D曲面视图
        ax1 = fig.add_subplot(2, 3, idx+1, projection='3d')
        
        # 创建坐标网格
        x = np.linspace(0, 6, size)
        y = np.linspace(0, 6, size)
        X, Y = np.meshgrid(x, y)
        
        # 降采样以提高渲染速度
        step = max(1, size // 50)
        ax1.plot_surface(X[::step, ::step], Y[::step, ::step], 
                        surface[::step, ::step], cmap='viridis', 
                        linewidth=0, antialiased=True)
        
        ax1.set_title(f'WM曲面 3D视图\n(D={dimension}, λ={lambda_param})')
        ax1.set_xlabel('X坐标')
        ax1.set_ylabel('Y坐标')
        ax1.set_zlabel('高度值')
        ax1.view_init(elev=30, azim=45)
        
        # 2D热图视图
        ax2 = fig.add_subplot(2, 3, idx+4)
        im = ax2.imshow(surface, cmap='viridis', origin='lower', 
                       extent=[0, 6, 0, 6], aspect='auto')
        ax2.set_title(f'WM曲面 2D视图\n(标准差={surface.std():.4f})')
        ax2.set_xlabel('X坐标')
        ax2.set_ylabel('Y坐标')
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, label='高度值')
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "result_wm_surface.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n2. 可视化结果已保存: {output_file}")
    plt.show()
    
    print("\n" + "="*60)
    print("理论知识补充")
    print("="*60)
    print("   Weierstrass-Mandelbrot曲面是确定性分形")
    print("   通过傅里叶级数叠加不同频率的波形构造")
    print(f"   Z(x,y) = Σ [C[n] * λ^((D-3)*n) * sin(...)]")
    print(f"   频率增长参数λ={lambda_param}, 迭代层数level={level}")
    print("\n示例运行完成！")


if __name__ == '__main__':
    main()
