#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Vicsek分形生成示例
=================

本示例演示如何使用fracDimPy生成Vicsek分形。
Vicsek分形也称为"维奇克分形"，是一种类似十字形的分形图案，
通过递归地在正方形的中心和四个角上放置小正方形而生成。

主要功能：
- 生成不同层级的Vicsek分形
- 可视化分形的逐层构造过程
- 展示理论分形维数和填充率

理论背景：
- Vicsek分形的分形维数: D = log(5)/log(3) ≈ 1.465
- 将正方形9等分，保留中心和四个角的5个部分
- 结构类似十字形或加号形状
- 与Sierpinski地毯相关，但保留模式不同
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
    print("Vicsek分形生成示例")
    print("="*60)
    
    from fracDimPy import generate_vicsek_fractal
    
    # 生成Vicsek分形
    print("\n1. 正在生成Vicsek分形...")
    levels = [2, 4, 6]  # 不同的迭代层级
    
    fig, axes = plt.subplots(1, len(levels), figsize=(15, 5))
    
    for idx, level in enumerate(levels):
        size = 3 ** level
        vicsek = generate_vicsek_fractal(level=level, size=size)
        
        total_pixels = size ** 2
        filled_pixels = np.sum(vicsek)
        fill_ratio = filled_pixels / total_pixels * 100
        
        print(f"   层级={level}: 图像尺寸{size}x{size}")
        print(f"   填充率: {fill_ratio:.2f}%")
        
        # 显示Vicsek分形
        axes[idx].imshow(vicsek, cmap='binary', origin='upper')
        axes[idx].set_title(f'Vicsek分形\n(迭代层级={level})')
        axes[idx].axis('off')
        
        # 添加理论分形维数和填充率标注
        D_theoretical = np.log(5) / np.log(3)
        axes[idx].text(0.5, -0.05, 
                      f'理论分形维数: D ≈ {D_theoretical:.4f}\n填充率: {fill_ratio:.1f}%',
                      transform=axes[idx].transAxes,
                      ha='center', va='top',
                      fontsize=9,
                      bbox=dict(boxstyle='round', facecolor='cyan', alpha=0.7))
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "result_vicsek.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n2. 可视化结果已保存: {output_file}")
    plt.show()
    
    print("\n" + "="*60)
    print("理论知识补充")
    print("="*60)
    print("   Vicsek分形呈现十字形或加号形状")
    print("   将正方形3×3等分，保留中心和四个角，共5个部分")
    print(f"   分形维数: D = log(5)/log(3) ≈ {np.log(5)/np.log(3):.4f}")
    print("   与Sierpinski地毯类似但保留模式不同")
    print("\n示例运行完成！")


if __name__ == '__main__':
    main()
