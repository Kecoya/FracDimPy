#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hurst指数(R/S分析)测试示例
=====================

本示例演示如何使用fracDimPy的Hurst指数方法（R/S分析）来估计时间序列的分形维数。
Hurst指数反映了时间序列的长程相关性和记忆效应，是分析分形时间序列的重要工具。

主要功能：
- 加载并分析时间序列数据
- 使用R/S分析法计算Hurst指数
- 从Hurst指数计算分形维数
- 可视化原始序列和R/S分析结果

理论背景：
- Hurst指数H ∈ (0, 1)
- H < 0.5: 反持续性（均值回归）
- H = 0.5: 随机游走（无记忆）
- H > 0.5: 持续性（趋势延续）
- 分形维数D = 2 - H（一维信号）
- R/S分析通过rescaled range统计量估计H
"""

import numpy as np
import os
from fracDimPy import hurst_dimension
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 
# 数据文件路径
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "hurst_data.npy")

def main():
    print("="*60)
    print("Hurst指数(R/S分析)测试示例")
    print("="*60)
    
    # 1. 加载测试数据
    print(f"\n1. 正在加载数据: {data_file}")
    data = np.load(data_file)
    print(f"   数据长度: {len(data)}个点")
    print(f"   数值范围: {data.min():.4f} ~ {data.max():.4f}")
    
    # 2. 计算Hurst指数和分形维数
    print("\n2. 正在计算Hurst指数...")
    D, result = hurst_dimension(data)
    
    # 3. 显示计算结果
    print("\n3. 计算结果:")
    print(f"   分形维数 D: {D:.4f}")
    print(f"   Hurst指数 H: {result['hurst']:.4f}")
    print(f"   拟合优度 R²: {result['R2']:.4f}")
    
    # 4. 可视化结果
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 左图：显示原始时间序列
        axes[0].plot(data, linewidth=0.6, color='steelblue')
        axes[0].set_title('原始时间序列')
        axes[0].set_xlabel('时间索引')
        axes[0].set_ylabel('数值')
        axes[0].grid(True, alpha=0.3)
        
        # 右图：R/S分析的log-log图
        if 'log_r' in result and 'log_rs' in result:
            axes[1].plot(result['log_r'], result['log_rs'], 'o', 
                        label='观测数据点', markersize=6, color='blue')
            # 绘制拟合直线
            coeffs = result.get('coefficients', [result['hurst'], 0])
            fit_line = coeffs[0] * result['log_r'] + coeffs[1]
            axes[1].plot(result['log_r'], fit_line, 'r-', linewidth=2,
                        label=f'线性拟合 (H={result["hurst"]:.4f})')
            axes[1].set_xlabel('log(r) - 时间尺度对数')
            axes[1].set_ylabel('log(R/S) - 重标极差对数')
            axes[1].set_title('R/S分析结果')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(current_dir, "result_hurst.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n4. 可视化结果已保存: {output_file}")
        plt.show()
        
    except ImportError:
        print("\n4. 可视化失败: 需要安装matplotlib库")
    
    print("\n示例运行完成！")


if __name__ == '__main__':
    main()
