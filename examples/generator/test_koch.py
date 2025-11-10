#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Koch曲线与雪花生成示例
===============

本示例演示如何使用fracDimPy生成Koch曲线和Koch雪花。
Koch曲线是最著名的分形曲线之一，通过递归地将线段替换为特定形状而生成，
Koch雪花则是由三条Koch曲线组成的封闭图形。

主要功能：
- 生成不同层级的Koch曲线
- 生成不同层级的Koch雪花
- 可视化分形的逐层构造过程
- 展示理论分形维数

理论背景：
- Koch曲线的分形维数: D = log(4)/log(3) ≈ 1.2619
- 每次迭代将一条线段替换为4条，长度变为原来的1/3
- Koch雪花由3条Koch曲线围成，具有无限长的周长和有限的面积
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
    print("Koch曲线与雪花生成示例")
    print("="*60)
    
    from fracDimPy import generate_koch_curve, generate_koch_snowflake
    
    # 1. 生成Koch曲线
    print("\n1. 正在生成Koch曲线...")
    levels_curve = [1, 3, 5]  # 不同的迭代层级
    size = 512                # 图像尺寸
    
    fig1, axes1 = plt.subplots(1, len(levels_curve), figsize=(15, 5))
    
    for idx, level in enumerate(levels_curve):
        points, image = generate_koch_curve(level=level, size=size)
        
        print(f"   层级={level}: 生成了{len(points)}个点")
        
        # 显示Koch曲线
        axes1[idx].imshow(image, cmap='binary', origin='upper')
        axes1[idx].set_title(f'Koch曲线\n(迭代层级={level})')
        axes1[idx].axis('off')
        
        # 添加理论分形维数标注
        D_theoretical = np.log(4) / np.log(3)
        axes1[idx].text(0.5, -0.05, 
                       f'理论分形维数: D ≈ {D_theoretical:.4f}',
                       transform=axes1[idx].transAxes,
                       ha='center', va='top',
                       fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file1 = os.path.join(current_dir, "result_koch_curve.png")
    plt.savefig(output_file1, dpi=300, bbox_inches='tight')
    print(f"\n2. Koch曲线图像已保存: {output_file1}")
    
    # 2. 生成Koch雪花
    print("\n3. 正在生成Koch雪花...")
    levels_snow = [1, 3, 5]  # 不同的迭代层级
    
    fig2, axes2 = plt.subplots(1, len(levels_snow), figsize=(15, 5))
    
    for idx, level in enumerate(levels_snow):
        snowflake = generate_koch_snowflake(level=level, size=size)
        
        print(f"   层级={level}: 图像尺寸{size}x{size}")
        
        # 显示Koch雪花
        axes2[idx].imshow(snowflake, cmap='binary', origin='upper')
        axes2[idx].set_title(f'Koch雪花\n(迭代层级={level})')
        axes2[idx].axis('off')
        
        # 添加理论分形维数标注
        D_theoretical = np.log(4) / np.log(3)
        axes2[idx].text(0.5, -0.05, 
                       f'理论分形维数: D ≈ {D_theoretical:.4f}',
                       transform=axes2[idx].transAxes,
                       ha='center', va='top',
                       fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    output_file2 = os.path.join(current_dir, "result_koch_snowflake.png")
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"\n4. Koch雪花图像已保存: {output_file2}")
    
    plt.show()
    
    print("\n" + "="*60)
    print("理论知识补充")
    print("="*60)
    print("   Koch曲线是经典的分形曲线")
    print("   Koch雪花由3条Koch曲线围成，形成封闭图形")
    print(f"   分形维数: D = log(4)/log(3) ≈ {np.log(4)/np.log(3):.4f}")
    print("   特点: 周长无限，面积有限")
    print("\n示例运行完成！")


if __name__ == '__main__':
    main()

