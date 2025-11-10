#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DLA扩散限制凝聚生成示例
==============

本示例演示如何使用fracDimPy生成扩散限制凝聚（DLA）结构。
DLA是一种模拟粒子通过布朗运动随机游走并附着形成聚集体的过程，
广泛应用于晶体生长、电沉积、雪花形成等自然现象的模拟。

主要功能：
- 生成指定尺寸和粒子数的DLA结构
- 可视化完整的DLA形态和局部放大区域
- 统计粒子附着情况

理论背景：
- DLA结构具有分形特征，呈现出树枝状或雪花状的形态
- 粒子从远处随机游走，一旦接触到已有的聚集体就会附着
- DLA的分形维数约为1.71（2D情况）
- 具有强烈的各向异性和随机性
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
    print("扩散限制凝聚（DLA）生成示例")
    print("="*60)
    
    from fracDimPy import generate_dla
    
    # 1. 生成DLA结构
    print("\n1. 正在生成DLA结构...")
    size = 200           # 网格尺寸
    particles = 100000   # 粒子总数
    
    print(f"   网格尺寸: {size} x {size}")
    print(f"   粒子总数: {particles}")
    print("   生成过程中，请耐心等待...")
    
    dla = generate_dla(size=size, num_particles=particles)
    
    occupied = np.sum(dla > 0)
    print(f"   成功附着粒子数: {occupied}")
    print(f"   附着成功率: {occupied/particles*100:.2f}%")
    
    # 2. 可视化DLA结构
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图: 显示完整的DLA结构
        axes[0].imshow(dla, cmap='hot', interpolation='nearest')
        axes[0].set_title(f'DLA完整结构 (粒子数={particles})')
        axes[0].axis('off')
        
        # 右图: 显示中心区域的放大视图
        center = size // 2
        zoom_size = size // 4
        dla_zoom = dla[center-zoom_size:center+zoom_size, 
                       center-zoom_size:center+zoom_size]
        axes[1].imshow(dla_zoom, cmap='hot', interpolation='nearest')
        axes[1].set_title('DLA中心区域放大视图')
        axes[1].axis('off')
        
        plt.tight_layout()
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(current_dir, "result_dla.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n2. 可视化结果已保存: {output_file}")
        plt.show()
        
    except ImportError:
        print("\n2. 可视化失败: 需要安装matplotlib库")
    
    print("\n示例运行完成！")


if __name__ == '__main__':
    main()

