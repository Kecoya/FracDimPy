#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Menger海绵生成示例
=================

本示例演示如何使用fracDimPy生成Menger海绵（Menger sponge）。
Menger海绵是Sierpinski地毯的三维推广，是一种经典的三维分形结构，
通过递归地从立方体中移除中心部分而生成，具有无限的表面积和零体积的特性。

主要功能：
- 生成不同层级的Menger海绵
- 以3D体素图形式可视化分形结构
- 统计体积填充率的变化
- 展示理论分形维数

理论背景：
- Menger海绵的分形维数: D = log(20)/log(3) ≈ 2.727
- 将立方体27等分（3×3×3），移除中心和各面中心共7个，保留20个
- 随着迭代次数增加，体积趋向于0，表面积趋向于无穷
- 是Sierpinski地毯的三维类比
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
    print("Menger海绵生成示例")
    print("="*60)
    
    from fracDimPy import generate_menger_sponge
    
    # 生成Menger海绵
    print("\n1. 正在生成Menger海绵...")
    levels = [1, 2, 3]  # 不同的迭代层级
    
    fig = plt.figure(figsize=(15, 5))
    
    for idx, level in enumerate(levels):
        size = 3 ** level
        sponge = generate_menger_sponge(level=level, size=size)
        
        total_voxels = size ** 3
        filled_voxels = np.sum(sponge)
        fill_ratio = filled_voxels / total_voxels * 100
        
        print(f"   层级={level}: 网格尺寸{size}x{size}x{size}")
        print(f"   填充体素数: {filled_voxels}/{total_voxels} ({fill_ratio:.2f}%)")
        
        # 3D可视化Menger海绵
        try:
            from mpl_toolkits.mplot3d import Axes3D
            
            ax = fig.add_subplot(1, len(levels), idx+1, projection='3d')
            
            # 创建颜色数组
            colors = np.empty(sponge.shape, dtype=object)
            colors[sponge == 1] = 'cyan'
            
            # 使用voxels函数绘制3D体素
            ax.voxels(sponge, facecolors=colors, edgecolors='gray', 
                     linewidth=0.1, alpha=0.9)
            
            ax.set_title(f'Menger海绵\n(迭代层级={level})')
            ax.set_xlabel('X轴')
            ax.set_ylabel('Y轴')
            ax.set_zlabel('Z轴')
            
            # 设置相等的坐标轴比例
            ax.set_box_aspect([1,1,1])
            
        except ImportError:
            print("   警告: 需要mpl_toolkits.mplot3d来进行3D可视化")
    
    # 添加理论说明
    D_theoretical = np.log(20) / np.log(3)
    fig.text(0.5, 0.02, 
            f'理论分形维数: D = log(20)/log(3) ≈ {D_theoretical:.3f}',
            ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "result_menger_sponge.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n2. 可视化结果已保存: {output_file}")
    plt.show()
    
    print("\n" + "="*60)
    print("理论知识补充")
    print("="*60)
    print("   Menger海绵是Sierpinski地毯的三维推广")
    print("   是Sierpinski三角形（二维）和Cantor集（一维）在三维空间的类比")
    print(f"   分形维数: D = log(20)/log(3) ≈ {D_theoretical:.4f}")
    print("\n示例运行完成！")


if __name__ == '__main__':
    main()
