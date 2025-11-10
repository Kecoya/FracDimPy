#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sierpinski三角形生成示例
=======================

本示例演示如何使用fracDimPy生成Sierpinski三角形。
Sierpinski三角形是一种经典的二维分形图案，通过递归地移除三角形的中心部分而生成，
具有完美的自相似性和精美的几何结构。

主要功能：
- 生成不同层级的Sierpinski三角形
- 可视化分形的逐层构造过程
- 展示理论分形维数和填充率

理论背景：
- Sierpinski三角形的分形维数: D = log(3)/log(2) ≈ 1.585
- 每次迭代保留3个小三角形，移除中间的1个
- 随着迭代次数增加，填充率逐渐降低趋向于0
- 具有完美的三重对称性
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
    print("Sierpinski三角形生成示例")
    print("="*60)
    
    from fracDimPy import generate_sierpinski
    
    # 生成Sierpinski三角形
    print("\n1. 正在生成Sierpinski三角形...")
    levels = [3, 5, 7]  # 不同的迭代层级
    size = 512          # 图像尺寸
    
    fig, axes = plt.subplots(1, len(levels), figsize=(15, 5))
    
    for idx, level in enumerate(levels):
        triangle = generate_sierpinski(level=level, size=size)
        
        fill_ratio = np.sum(triangle) / (size * size) * 100
        print(f"   层级={level}: 图像尺寸{size}x{size}")
        print(f"   填充率: {fill_ratio:.2f}%")
        
        # 显示Sierpinski三角形
        axes[idx].imshow(triangle, cmap='binary', origin='upper')
        axes[idx].set_title(f'Sierpinski三角形\n(迭代层级={level})')
        axes[idx].axis('off')
        
        # 添加理论分形维数标注
        D_theoretical = np.log(3) / np.log(2)
        axes[idx].text(0.5, 0.02, 
                      f'理论分形维数: D ≈ {D_theoretical:.3f}',
                      transform=axes[idx].transAxes,
                      ha='center',
                      bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "result_sierpinski.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n2. 可视化结果已保存: {output_file}")
    plt.show()
    
    print("\n示例运行完成！")


if __name__ == '__main__':
    main()
