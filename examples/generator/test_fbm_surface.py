#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分数布朗运动曲面生成示例
==============

本示例演示如何使用fracDimPy生成分数布朗运动（FBM）曲面。
FBM曲面是二维的随机分形表面，具有自相似性和各向同性，
广泛应用于地形生成、纹理合成、高程模拟等领域。

主要功能：
- 生成指定分形维数的FBM曲面
- 以2D热图、3D曲面、等高线图三种方式可视化
- 保存结果图像

理论背景：
- FBM曲面的分形维数D与Hurst指数H的关系：D = 3 - H
- D ∈ (2, 3)，D越大表示曲面越粗糙
- H ∈ (0, 1)，H越大表示曲面越平滑
"""

import numpy as np
import os
import matplotlib.pyplot as plt
#  scienceplots 
try:
    import scienceplots
    plt.style.use(['science','no-latex'])  # 
except ImportError:
    pass
# Microsoft YaHeiTimes New Roman
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 
def main():
    print("="*60)
    print("分数布朗运动曲面生成示例")
    print("="*60)
    
    try:
        from fracDimPy import generate_fbm_surface
        
        # 1. 生成FBM曲面
        print("\n1. 正在生成FBM曲面...")
        dimension = 2.3  # 目标分形维数
        size = 256       # 曲面尺寸（像素数）
        
        surface = generate_fbm_surface(dimension=dimension, size=size)
        
        H = 3 - dimension  # 计算Hurst指数
        print(f"   分形维数 D: {dimension}")
        print(f"   Hurst指数 H: {H:.2f}")
        print(f"   曲面尺寸: {size} x {size}")
        print(f"   数值范围: {surface.min():.4f} ~ {surface.max():.4f}")
        
        # 2. 可视化FBM曲面
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            
            fig = plt.figure(figsize=(15, 5))
            
            # 2D热图视图
            ax1 = fig.add_subplot(131)
            im = ax1.imshow(surface, cmap='terrain')
            ax1.set_title(f'FBM曲面-2D视图 (D={dimension}, H={H:.2f})')
            ax1.set_xlabel('X坐标')
            ax1.set_ylabel('Y坐标')
            plt.colorbar(im, ax=ax1, label='高度值')
            
            # 3D曲面视图
            ax2 = fig.add_subplot(132, projection='3d')
            X, Y = np.meshgrid(range(size), range(size))
            # 降采样以提高渲染速度
            step = max(1, size // 50)
            ax2.plot_surface(X[::step, ::step], Y[::step, ::step], 
                           surface[::step, ::step], cmap='terrain', alpha=0.9)
            ax2.set_title('FBM曲面-3D视图')
            ax2.set_xlabel('X坐标')
            ax2.set_ylabel('Y坐标')
            ax2.set_zlabel('高度值')
            
            # 等高线图视图
            ax3 = fig.add_subplot(133)
            # 绘制等高线（黑色线条）
            contour_lines = ax3.contour(surface, levels=15, colors='black', 
                                       linewidths=0.8, alpha=0.6)
            # 添加等高线标签
            ax3.clabel(contour_lines, inline=True, fontsize=8, fmt='%.2f')
            # 绘制填充等高线（彩色区域）
            contour_filled = ax3.contourf(surface, levels=15, cmap='terrain', alpha=0.7)
            ax3.set_title('FBM曲面-等高线图')
            ax3.set_xlabel('X坐标')
            ax3.set_ylabel('Y坐标')
            plt.colorbar(contour_filled, ax=ax3, label='高度值')
            
            plt.tight_layout()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_file = os.path.join(current_dir, "result_fbm_surface.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\n2. 可视化结果已保存: {output_file}")
            plt.show()
            
        except ImportError:
            print("\n2. 可视化失败: 需要安装matplotlib库")
        
    except ImportError:
        print("\n错误: 需要安装相关库来生成FBM曲面")
    
    print("\n示例运行完成！")


if __name__ == '__main__':
    main()

