#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sierpinski地毯生成示例
====================

本示例演示如何使用fracDimPy生成Sierpinski地毯（Sierpinski carpet）。
Sierpinski地毯是Sierpinski三角形的二维推广版本，也是Menger海绵的二维类比，
通过递归地移除正方形网格的中心部分而生成。

主要功能：
- 生成不同层级的Sierpinski地毯
- 可视化分形的逐层构造过程
- 展示理论分形维数和填充率变化

理论背景：
- Sierpinski地毯的分形维数: D = log(8)/log(3) ≈ 1.8928
- 将正方形9等分，移除中间1个，保留周围8个
- 是Menger海绵的二维类比
- 随着迭代次数增加，填充率指数下降
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
    print("Sierpinski地毯生成示例")
    print("="*60)
    
    from fracDimPy import generate_sierpinski_carpet
    
    # 生成Sierpinski地毯
    print("\n1. 正在生成Sierpinski地毯...")
    levels = [3, 4, 5]  # 不同的迭代层级
    
    fig, axes = plt.subplots(1, len(levels), figsize=(15, 5))
    
    for idx, level in enumerate(levels):
        size = 3 ** level
        carpet = generate_sierpinski_carpet(level=level, size=size)
        
        total_pixels = size ** 2
        filled_pixels = np.sum(carpet)
        fill_ratio = filled_pixels / total_pixels * 100
        
        print(f"   层级={level}: 图像尺寸{size}x{size}")
        print(f"   填充率: {fill_ratio:.2f}%")
        
        # 显示Sierpinski地毯
        axes[idx].imshow(carpet, cmap='binary', origin='upper')
        axes[idx].set_title(f'Sierpinski地毯\n(迭代层级={level})')
        axes[idx].axis('off')
        
        # 添加理论分形维数和填充率标注
        D_theoretical = np.log(8) / np.log(3)
        axes[idx].text(0.5, -0.05, 
                      f'理论分形维数: D ≈ {D_theoretical:.3f}\n填充率: {fill_ratio:.1f}%',
                      transform=axes[idx].transAxes,
                      ha='center', va='top',
                      fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "result_sierpinski_carpet.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n2. 可视化结果已保存: {output_file}")
    plt.show()
    
    print("\n" + "="*60)
    print("理论知识补充")
    print("="*60)
    print("   Sierpinski地毯是Menger海绵的二维类比")
    print("   每次迭代将正方形9等分，移除中心的1个，保留周围8个")
    print(f"   分形维数: D = log(8)/log(3) ≈ {np.log(8)/np.log(3):.4f}")
    print("\n示例运行完成！")


if __name__ == '__main__':
    main()
