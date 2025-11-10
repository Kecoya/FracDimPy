#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多重分形分析 - 图像数据
==================

本示例演示如何使用fracDimPy对二维图像数据进行多重分形分析。
图像的多重分形特征能够揭示纹理的复杂性、非均匀性和多尺度特性，
在医学影像分析、材料科学、地质勘探等领域有广泛应用。

主要功能：
- 加载并预处理图像数据
- 计算图像的多重分形谱
- 提取多重分形特征参数
- 可视化分析结果

理论背景：
- 图像多重分形分析基于盒计数法的推广
- 通过不同q阶矩描述图像灰度分布的多样性
- 多重分形谱宽度反映图像的非均匀程度
- D(0): 容量维数, D(1): 信息维数, D(2): 关联维数
"""

import numpy as np
import os
from fracDimPy import multifractal_image
import matplotlib.pyplot as plt

# 设置中文字体
import scienceplots 
plt.style.use(['science', 'no-latex'])
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "mf_image_shale.png")

def main():
    print("="*60)
    print("多重分形分析 - 图像数据")
    print("="*60)
    
    # 1. 加载图像数据
    print(f"\n1. 正在加载图像: {data_file}")
    try:
        from PIL import Image
        img = Image.open(data_file)
        img_array = np.array(img)
        
        # 如果是彩色图像，转换为灰度
        if len(img_array.shape) == 3:
            img_gray = np.mean(img_array, axis=2)
        else:
            img_gray = img_array
        
        print(f"   图像尺寸: {img_gray.shape}")
        print(f"   像素范围: {img_gray.min():.1f} ~ {img_gray.max():.1f}")
        
    except Exception as e:
        print(f"   加载失败: {e}")
        return
    
    # 2. 多重分形分析
    print("\n2. 正在进行多重分形分析...")
    try:
        metrics, figure_data = multifractal_image(img_gray)
        
        # 3. 显示计算结果
        print("\n3. 多重分形特征参数:")
        print(f"   容量维数 D(0): {metrics['容量维数 D(0)'][0]:.4f}")
        print(f"   信息维数 D(1): {metrics['信息维数 D(1)'][0]:.4f}")
        print(f"   关联维数 D(2): {metrics['关联维数 D(2)'][0]:.4f}")
        print(f"   Hurst指数 H: {metrics['Hurst指数 H'][0]:.4f}")
        print(f"   谱宽度: {metrics['谱宽度'][0]:.4f}")
        
    except Exception as e:
        print(f"\n分析失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 4. 可视化结果
    print("\n4. 正在生成可视化图表...")
    try:
        fig = plt.figure(figsize=(16, 10))
        
        # 原始图像
        ax1 = fig.add_subplot(2, 3, 1)
        ax1.imshow(img_gray, cmap='gray')
        ax1.set_title('(a) 原始图像')
        ax1.axis('off')
        
        # 提取分析结果
        ql = figure_data['q值']
        tau_q = figure_data['质量指数τ(q)']
        alpha_q = figure_data['奇异性指数α(q)']
        f_alpha = figure_data['多重分形谱f(α)']
        D_q = figure_data['广义维数D(q)']
        
        # τ(q) 曲线
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.plot(ql, tau_q, 'o-', color='darkgreen', linewidth=2, markersize=4)
        ax2.set_xlabel(r'$q$ - 统计矩阶数', fontsize=10)
        ax2.set_ylabel(r'$\tau(q)$ - 质量指数', fontsize=10)
        ax2.set_title('(b) 质量指数函数')
        ax2.grid(True, alpha=0.3)
        
        # α(q) 曲线
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(ql, alpha_q, 's-', color='crimson', linewidth=2, markersize=4)
        ax3.set_xlabel(r'$q$ - 统计矩阶数', fontsize=10)
        ax3.set_ylabel(r'$\alpha(q)$ - Hölder指数', fontsize=10)
        ax3.set_title(r'(c) Hölder指数函数')
        ax3.grid(True, alpha=0.3)
        
        # f(α) 多重分形谱
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(alpha_q, f_alpha, '^-', color='darkorange', linewidth=2, markersize=4)
        ax4.set_xlabel(r'$\alpha$ - 奇异性指数', fontsize=10)
        ax4.set_ylabel(r'$f(\alpha)$ - 多重分形谱', fontsize=10)
        ax4.set_title('(d) 多重分形谱')
        ax4.grid(True, alpha=0.3)
        
        # D(q) 曲线
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.plot(ql, D_q, 'd-', color='mediumpurple', linewidth=2, markersize=4)
        ax5.set_xlabel(r'$q$ - 统计矩阶数', fontsize=10)
        ax5.set_ylabel(r'$D(q)$ - 广义维数', fontsize=10)
        ax5.set_title('(e) 广义维数谱')
        ax5.grid(True, alpha=0.3)
        
        # 关键参数对比
        ax6 = fig.add_subplot(2, 3, 6)
        params = ['D(0)', 'D(1)', 'D(2)', 'H', '谱宽度']
        values = [
            metrics['容量维数 D(0)'][0],
            metrics['信息维数 D(1)'][0],
            metrics['关联维数 D(2)'][0],
            metrics['Hurst指数 H'][0],
            metrics['谱宽度'][0]
        ]
        colors = ['green', 'blue', 'red', 'orange', 'purple']
        bars = ax6.bar(params, values, color=colors, alpha=0.7)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax6.set_ylabel('参数值')
        ax6.set_title('(f) 多重分形特征参数')
        ax6.tick_params(axis='x', labelsize=8, rotation=15)
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        output_file = os.path.join(current_dir, "result_mf_image.png")
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n可视化结果已保存: result_mf_image.png")
        plt.show()
        
    except Exception as e:
        print(f"\n可视化失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n示例运行完成！")


if __name__ == '__main__':
    main()
