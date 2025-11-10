#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
变差函数法测试示例
==================

本示例演示如何使用fracDimPy的变差函数法（Variogram）来估计数据的分形维数。
变差函数法通过分析数据在不同尺度上的空间变异性来估计分形特征，
广泛应用于地质统计学、地形分析等领域。

主要功能：
- 支持一维序列和二维曲面数据
- 计算变差函数并拟合幂律关系
- 从幂律指数估计分形维数
- 可视化变差函数的尺度关系

理论背景：
- 变差函数γ(h)描述距离h处的空间变异性
- 对于分形数据：γ(h) ∝ h^(2H)
- H是Hurst指数，反映数据的平滑程度
- 分形维数: D = E + 1 - H（E是嵌入维数）
- 一维: D = 2 - H，二维: D = 3 - H
"""

import numpy as np
import os
from fracDimPy import variogram
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 数据文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file_1d = os.path.join(current_dir, "variogram_1d_data.npy")
data_file_2d = os.path.join(current_dir, "variogram_surface_data.tif")

def main():
    print("="*60)
    print("变差函数法测试示例")
    print("="*60)
    
    # ========== 测试1: 一维数据 ==========
    print("\n【测试1: 一维时间序列】")
    print(f"正在加载数据: {data_file_1d}")
    try:
        data_1d = np.load(data_file_1d)
        print(f"   数据长度: {len(data_1d)}个点")
        print(f"   数值范围: {data_1d.min():.4f} ~ {data_1d.max():.4f}")
        
        print("\n正在计算变差函数...")
        D_1d, result_1d = variogram(data_1d)
        
        print("\n计算结果:")
        print(f"   分形维数 D: {D_1d:.4f}")
        print(f"   Hurst指数 H: {result_1d['hurst']:.4f}")
        print(f"   拟合优度 R²: {result_1d['R2']:.4f}")
        
    except Exception as e:
        print(f"   错误: {e}")
        data_1d = None
        result_1d = None
    
    # ========== 测试2: 二维曲面 ==========
    print("\n【测试2: 二维曲面数据】")
    print(f"正在加载数据: {data_file_2d}")
    try:
        from PIL import Image
        img = Image.open(data_file_2d)
        data_2d = np.array(img)
        
        # 如果是多通道图像，转换为灰度
        if len(data_2d.shape) > 2:
            data_2d = np.mean(data_2d, axis=2)
        
        print(f"   数据尺寸: {data_2d.shape}")
        print(f"   数值范围: {data_2d.min():.1f} ~ {data_2d.max():.1f}")
        
        print("\n正在计算变差函数...")
        D_2d, result_2d = variogram(data_2d)
        
        print("\n计算结果:")
        print(f"   分形维数 D: {D_2d:.4f}")
        print(f"   Hurst指数 H: {result_2d['hurst']:.4f}")
        print(f"   拟合优度 R²: {result_2d['R2']:.4f}")
        
    except Exception as e:
        print(f"   错误: {e}")
        data_2d = None
        result_2d = None
    
    # ========== 可视化结果 ==========
    print("\n正在生成可视化图表...")
    
    # 确定要绘制的子图数量
    n_plots = sum([data_1d is not None, data_2d is not None])
    if n_plots == 0:
        print("没有成功加载的数据，跳过可视化。")
        return
    
    fig = plt.figure(figsize=(15, 5*n_plots))
    plot_idx = 1
    
    # 可视化一维结果
    if data_1d is not None and result_1d is not None:
        # 原始数据
        ax1 = fig.add_subplot(n_plots, 3, plot_idx)
        ax1.plot(data_1d, linewidth=0.6, color='steelblue')
        ax1.set_title('一维时间序列')
        ax1.set_xlabel('时间索引')
        ax1.set_ylabel('数值')
        ax1.grid(True, alpha=0.3)
        
        # log-log图
        ax2 = fig.add_subplot(n_plots, 3, plot_idx+1)
        if 'log_lags' in result_1d and 'log_variogram' in result_1d:
            ax2.plot(result_1d['log_lags'], result_1d['log_variogram'], 
                    'o', label='观测数据', markersize=6, color='blue')
            
            # 拟合直线
            if 'coefficients' in result_1d:
                a, b = result_1d['coefficients']
                fit_line = a * result_1d['log_lags'] + b
                ax2.plot(result_1d['log_lags'], fit_line, 'r-', 
                        linewidth=2, label=f'拟合 (斜率={a:.4f})')
            
            ax2.set_xlabel('log(lag) - 滞后距离对数')
            ax2.set_ylabel('log(γ) - 变差函数对数')
            ax2.set_title('变差函数分析（一维）')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 参数对比
        ax3 = fig.add_subplot(n_plots, 3, plot_idx+2)
        params = ['分形维数D', 'Hurst指数H']
        values = [D_1d, result_1d['hurst']]
        colors = ['green', 'orange']
        bars = ax3.bar(params, values, color=colors, alpha=0.7)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom')
        
        ax3.set_ylabel('参数值')
        ax3.set_title(f'分形参数\nR²={result_1d["R2"]:.4f}')
        ax3.grid(True, alpha=0.3, axis='y')
        
        plot_idx += 3
    
    # 可视化二维结果
    if data_2d is not None and result_2d is not None:
        # 原始数据
        ax1 = fig.add_subplot(n_plots, 3, plot_idx)
        im = ax1.imshow(data_2d, cmap='terrain')
        ax1.set_title('二维曲面数据')
        ax1.axis('off')
        plt.colorbar(im, ax=ax1, fraction=0.046)
        
        # log-log图
        ax2 = fig.add_subplot(n_plots, 3, plot_idx+1)
        if 'log_lags' in result_2d and 'log_variogram' in result_2d:
            ax2.plot(result_2d['log_lags'], result_2d['log_variogram'], 
                    'o', label='观测数据', markersize=6, color='blue')
            
            # 拟合直线
            if 'coefficients' in result_2d:
                a, b = result_2d['coefficients']
                fit_line = a * result_2d['log_lags'] + b
                ax2.plot(result_2d['log_lags'], fit_line, 'r-', 
                        linewidth=2, label=f'拟合 (斜率={a:.4f})')
            
            ax2.set_xlabel('log(lag) - 滞后距离对数')
            ax2.set_ylabel('log(γ) - 变差函数对数')
            ax2.set_title('变差函数分析（二维）')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 参数对比
        ax3 = fig.add_subplot(n_plots, 3, plot_idx+2)
        params = ['分形维数D', 'Hurst指数H']
        values = [D_2d, result_2d['hurst']]
        colors = ['green', 'orange']
        bars = ax3.bar(params, values, color=colors, alpha=0.7)
        
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{val:.4f}', ha='center', va='bottom')
        
        ax3.set_ylabel('参数值')
        ax3.set_title(f'分形参数\nR²={result_2d["R2"]:.4f}')
        ax3.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = os.path.join(current_dir, "result_variogram.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n可视化结果已保存: {output_file}")
    plt.show()
    
    print("\n示例运行完成！")


if __name__ == '__main__':
    main()
