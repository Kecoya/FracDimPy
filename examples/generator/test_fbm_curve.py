#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分数布朗运动曲线生成示例
==============

本示例演示如何使用fracDimPy生成分数布朗运动（FBM）曲线。
FBM是一种重要的分形曲线，具有自相似性和长程相关性，
广泛应用于金融时间序列、地形建模等领域。

主要功能：
- 生成指定分形维数的FBM曲线
- 可视化生成的曲线
- 保存结果图像

理论背景：
- FBM的分形维数D与Hurst指数H的关系：D = 2 - H
- D ∈ (1, 2)，D越大表示曲线越不规则
- H ∈ (0, 1)，H越大表示曲线越平滑
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
    print("分数布朗运动曲线生成示例")
    print("="*60)
    
    try:
        from fracDimPy import generate_fbm_curve
        
        # 1. 生成FBM曲线
        print("\n1. 正在生成FBM曲线...")
        target_D = 1.5  # 目标分形维数
        length = 2048   # 曲线长度（采样点数）
        
        curve, D_set = generate_fbm_curve(dimension=target_D, length=length)
        
        print(f"   目标分形维数: {target_D}")
        print(f"   曲线长度: {length}")
        print(f"   数值范围: {curve.min():.4f} ~ {curve.max():.4f}")
        
        # 2. 可视化FBM曲线
        try:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            # 绘制FBM曲线
            ax.plot(curve, linewidth=0.8, color='steelblue')
            ax.set_title(f'分数布朗运动曲线 (分形维数 D={target_D})')
            ax.set_xlabel('索引位置')
            ax.set_ylabel('曲线值')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            current_dir = os.path.dirname(os.path.abspath(__file__))
            output_file = os.path.join(current_dir, "result_fbm_curve.png")
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"\n2. 可视化结果已保存: {output_file}")
            plt.show()
            
        except ImportError:
            print("\n2. 可视化失败: 需要安装matplotlib库")
        
    except ImportError:
        print("\n错误: 需要安装fbm库来生成分数布朗运动")
        print("安装命令: pip install fbm")
    
    print("\n示例运行完成！")


if __name__ == '__main__':
    main()

