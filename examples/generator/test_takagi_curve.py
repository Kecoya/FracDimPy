#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Takagi曲线生成示例
=================

本示例演示如何使用fracDimPy生成Takagi曲线（也称为Blancmange曲线）。
Takagi曲线是一种连续但处处不可微的分形曲线，具有自仿射性质，
通过叠加不同尺度的三角波而生成。

主要功能：
- 生成不同分形维数的Takagi曲线
- 可视化曲线的形态变化
- 展示分形维数对曲线粗糙度的影响

理论背景：
- Takagi曲线的分形维数D ∈ (1, 2)
- D越大，曲线越粗糙、越不规则
- 通过迭代叠加不同振幅和频率的三角波形成
- 在[0,1]区间上定义，具有周期性
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
    print("Takagi曲线生成示例")
    print("="*60)
    
    from fracDimPy import generate_takagi_curve
    
    # 生成Takagi曲线
    print("\n1. 正在生成Takagi曲线...")
    dimensions = [1.2, 1.5, 1.8]  # 不同的分形维数
    level = 10                     # 迭代层数
    length = 2048                  # 采样点数
    
    fig, axes = plt.subplots(len(dimensions), 1, figsize=(12, 10))
    
    for idx, dimension in enumerate(dimensions):
        x, y = generate_takagi_curve(dimension=dimension, level=level, length=length)
        
        print(f"   分形维数 D={dimension}: 生成了{len(x)}个采样点")
        
        # 绘制Takagi曲线
        axes[idx].plot(x, y, linewidth=0.8, color='steelblue')
        axes[idx].set_title(f'Takagi曲线 (分形维数 D={dimension}, 迭代层数={level})')
        axes[idx].set_xlabel('x坐标')
        axes[idx].set_ylabel('y坐标')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].set_xlim(0, 1)
        
        # 添加数值范围信息
        axes[idx].text(0.02, 0.95, 
                      f'分形维数: D = {dimension}\n数值范围: [{y.min():.4f}, {y.max():.4f}]',
                      transform=axes[idx].transAxes,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "result_takagi_curve.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n2. 可视化结果已保存: {output_file}")
    plt.show()
    
    print("\n示例运行完成！")


if __name__ == '__main__':
    main()
