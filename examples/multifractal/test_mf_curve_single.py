#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多重分形分析 - 单列曲线数据
=====================================

本示例演示如何使用fracDimPy对单列曲线数据进行多重分形分析。
多重分形分析能够揭示数据在不同尺度和不同统计矩上的复杂分形特征，
相比单分形分析能够更全面地描述复杂系统的多样性和异质性。

主要功能：
- 加载单列时间序列数据
- 计算配分函数和多重分形谱
- 生成τ(q)、α(q)、f(α)、D(q)等关键曲线
- 提取D(0)、D(1)、D(2)等特征参数
- 综合可视化分析结果

理论背景：
- 配分函数X(ε,q)描述不同q阶矩的标度行为
- 质量指数τ(q)通过配分函数的幂律标度获得
- Hölder指数α(q)描述局部奇异性
- 多重分形谱f(α)描述不同奇异性的分布
- 广义维数D(q)是对不同统计矩的分形维数推广
- D(0): 容量维数, D(1): 信息维数, D(2): 关联维数
"""

import numpy as np
import os
from fracDimPy import multifractal_curve
import random
from matplotlib.lines import Line2D

import matplotlib.pyplot as plt
# SciencePlots
import scienceplots 
plt.style.use(['science', 'no-latex'])
# scienceplots
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 
mkl = [mk[0] for mk in Line2D.filled_markers]
# 
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "mf_curve_single_test.txt")

def main():
    print("="*60)
    print("多重分形分析 - 单列曲线数据")
    print("="*60)
    
    # 1. 加载数据
    print(f"\n1. 正在加载数据: {data_file}")
    data = np.loadtxt(data_file)
    print(f"   数据长度: {len(data)}个点")
    print(f"   数值范围: {data.min():.4f} ~ {data.max():.4f}")
    
    # 2. 多重分形分析
    print("\n2. 正在进行多重分形分析...")
    metrics, figure_data = multifractal_curve(
        data,
        use_multiprocessing=False,
        data_type='single'
    )
    
    # 3. 显示计算结果
    print("\n3. 多重分形特征参数:")
    print(f"   容量维数 D(0): {metrics['容量维数 D(0)'][0]:.4f}")
    print(f"   信息维数 D(1): {metrics['信息维数 D(1)'][0]:.4f}")
    print(f"   关联维数 D(2): {metrics['关联维数 D(2)'][0]:.4f}")
    print(f"   Hurst指数 H: {metrics['Hurst指数 H'][0]:.4f}")
    print(f"   谱宽度: {metrics['谱宽度'][0]:.4f}")
    print(f"   最大奇异性指数: {metrics['最大奇异性指数'][0]:.4f}")
    print(f"   最小奇异性指数: {metrics['最小奇异性指数'][0]:.4f}")
    
    # 4. 可视化多重分形分析结果
    try:
        print("\n4. 正在生成可视化图表...")
        
        # 提取分析结果
        ql = figure_data['q值']
        tau_q = figure_data['质量指数τ(q)']
        alpha_q = figure_data['奇异性指数α(q)']
        f_alpha = figure_data['多重分形谱f(α)']
        D_q = figure_data['广义维数D(q)']
        
        # 创建2行3列的综合图表
        fig = plt.figure(figsize=(18, 12))
        
        # ========== 子图1: 原始数据 ==========
        ax1 = plt.subplot(2, 3, 1)
        ax1.plot(data, linewidth=1, color='steelblue')
        ax1.set_xlabel('数据索引', fontsize=11)
        ax1.set_ylabel('数值', fontsize=11)
        ax1.set_title('(a) 原始时间序列', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # ========== 子图2: 配分函数 X vs ln(ε) ==========
        ax2 = plt.subplot(2, 3, 2)
        temp_q_n = max(1, int(len(ql) / 20))  # 每20个q值选1个显示
        plotted_count = 0
        for i, q_val in enumerate(ql):
            key = f'q={q_val}_配分函数X'
            key_r = f'q={q_val}_尺度r'
            
            if key in figure_data and key_r in figure_data:
                if i % temp_q_n == 0:
                    colors = np.random.rand(3,)
                    log_r = figure_data[key_r]
                    log_X = figure_data[key]
                    
                    ax2.plot(log_r, log_X, 
                            marker=random.choice(mkl),
                            label=f'$q={q_val:.2f}$',
                            linestyle='none',
                            color=colors,
                            markersize=6)
                    
                    # 绘制拟合直线
                    coeffs = np.polyfit(log_r, log_X, 1)
                    fit_line = np.poly1d(coeffs)
                    ax2.plot(log_r, fit_line(log_r), color=colors, linewidth=1.5)
                    
                    plotted_count += 1
        
        if plotted_count > 0:
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
        ax2.set_xlabel(r'$\ln(\epsilon)$ - 尺度对数', fontsize=11)
        ax2.set_ylabel(r'$\ln(X)$ - 配分函数对数', fontsize=11)
        ax2.set_title('(b) 配分函数标度关系', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # ========== 子图3: 质量指数 τ(q) vs q ==========
        ax3 = plt.subplot(2, 3, 3)
        ax3.plot(ql, tau_q, 'o-', color='darkgreen', linewidth=2, markersize=4)
        ax3.set_xlabel(r'$q$ - 统计矩阶数', fontsize=11)
        ax3.set_ylabel(r'$\tau(q)$ - 质量指数', fontsize=11)
        ax3.set_title('(c) 质量指数函数', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # ========== 子图4: Hölder指数 α(q) vs q ==========
        ax4 = plt.subplot(2, 3, 4)
        ax4.plot(ql, alpha_q, 's-', color='crimson', linewidth=2, markersize=4)
        ax4.set_xlabel(r'$q$ - 统计矩阶数', fontsize=11)
        ax4.set_ylabel(r'$\alpha(q)$ - Hölder指数', fontsize=11)
        ax4.set_title(r'(d) Hölder指数函数', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # ========== 子图5: 多重分形谱 f(α) vs α ==========
        ax5 = plt.subplot(2, 3, 5)
        ax5.plot(alpha_q, f_alpha, '^-', color='darkorange', linewidth=2, markersize=4)
        ax5.set_xlabel(r'$\alpha$ - 奇异性指数', fontsize=11)
        ax5.set_ylabel(r'$f(\alpha)$ - 多重分形谱', fontsize=11)
        ax5.set_title('(e) 多重分形谱', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 标记q=0点
        idx_0 = ql.index(0) if 0 in ql else len(ql)//2
        if idx_0 < len(alpha_q):
            ax5.plot(alpha_q[idx_0], f_alpha[idx_0], 'ro', markersize=8, label='q=0点')
            ax5.legend(fontsize=9)
        
        # ========== 子图6: 广义维数 D(q) vs q ==========
        ax6 = plt.subplot(2, 3, 6)
        ax6.plot(ql, D_q, 'd-', color='mediumpurple', linewidth=2, markersize=4)
        ax6.set_xlabel(r'$q$ - 统计矩阶数', fontsize=11)
        ax6.set_ylabel(r'$D(q)$ - 广义维数', fontsize=11)
        ax6.set_title('(f) 广义维数谱', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # 标注关键点: D(0), D(1), D(2)
        for q_val in [0, 1, 2]:
            if q_val in ql:
                idx = ql.index(q_val)
                ax6.plot(q_val, D_q[idx], 'o', markersize=8)
                ax6.text(q_val, D_q[idx], f'  D({q_val})={D_q[idx]:.3f}', 
                        fontsize=8, verticalalignment='bottom')
        
        plt.tight_layout()
        output_file = os.path.join(current_dir, "result_mf_comprehensive.png")
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n   可视化结果已保存: result_mf_comprehensive.png")
        plt.close(fig)
        
    except Exception as e:
        print(f"\n4. 可视化失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n示例运行完成！")


if __name__ == '__main__':
    main()

