#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Weierstrass-Mandelbrot曲线生成示例
=================================

本示例演示如何使用fracDimPy生成Weierstrass-Mandelbrot（WM）曲线。
WM曲线是Weierstrass函数的推广，是一种经典的连续但处处不可微的分形曲线，
通过叠加无穷多个不同频率和振幅的正弦波而生成。

主要功能：
- 生成不同分形维数的WM曲线
- 可视化曲线的形态变化
- 展示分形维数对曲线粗糙度的影响

理论背景：
- WM曲线的分形维数D ∈ (1, 2)
- D越大，曲线越粗糙、越不规则
- 通过傅里叶级数形式构造：叠加不同频率的正弦波
- 每个正弦波的振幅随频率的增加而按幂律衰减
- 是确定性的分形曲线（与FBM的随机性不同）
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
    print("Weierstrass-Mandelbrot曲线生成示例")
    print("="*60)
    
    from fracDimPy import generate_wm_curve
    
    # 生成WM曲线
    print("\n1. 正在生成Weierstrass-Mandelbrot曲线...")
    dimensions = [1.2, 1.5, 1.8]  # 不同的分形维数
    length = 2048                  # 采样点数
    
    fig, axes = plt.subplots(len(dimensions), 1, figsize=(12, 10))
    
    for idx, dimension in enumerate(dimensions):
        x, y = generate_wm_curve(dimension=dimension, length=length)
        
        print(f"   分形维数 D={dimension}: 生成了{len(x)}个采样点")
        
        # 绘制WM曲线
        axes[idx].plot(x, y, linewidth=0.5, color='steelblue')
        axes[idx].set_title(f'Weierstrass-Mandelbrot曲线 (分形维数 D={dimension})')
        axes[idx].set_xlabel('x坐标')
        axes[idx].set_ylabel('y坐标')
        axes[idx].grid(True, alpha=0.3)
        
        # 添加数值范围信息
        axes[idx].text(0.02, 0.95, 
                      f'数值范围: [{y.min():.3f}, {y.max():.3f}]',
                      transform=axes[idx].transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "result_wm_curve.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n2. 可视化结果已保存: {output_file}")
    plt.show()
    
    print("\n示例运行完成！")


if __name__ == '__main__':
    main()
