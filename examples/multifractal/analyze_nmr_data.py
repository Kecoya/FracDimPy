#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核磁数据多重分形分析
========================================
使用box-counting方法对1.xlsx、2.xlsx、3.xlsx三个核磁数据文件进行多重分形分析
生成期刊论文可用级别的矢量图像
"""

import numpy as np
import pandas as pd
import os
from fracDimPy import multifractal_curve
import matplotlib.pyplot as plt

# 使用SciencePlots样式
import scienceplots 
plt.style.use(['science', 'no-latex'])
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_nmr_data(file_path):
    """加载核磁数据文件"""
    print(f"正在读取: {file_path}")
    df = pd.read_excel(file_path)
    print(f"  数据形状: {df.shape}")
    print(f"  列名: {df.columns.tolist()}")
    
    # 读取第一列和第二列作为X和Y
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    
    print(f"  X范围: {x.min():.4f} ~ {x.max():.4f}")
    print(f"  Y范围: {y.min():.4f} ~ {y.max():.4f}")
    
    return x, y

def analyze_single_file(file_path, file_name):
    """分析单个文件并返回结果"""
    print("="*60)
    print(f"分析文件: {file_name}")
    print("="*60)
    
    # 加载数据
    x, y = load_nmr_data(file_path)
    
    # 进行多重分形分析
    print("\n执行多重分形分析...")
    metrics, figure_data = multifractal_curve(
        (x, y),
        use_multiprocessing=False,
        data_type='dual'
    )
    
    # 打印关键指标
    print("\n关键指标:")
    print(f"  容量维数 D(0): {metrics[' D(0)'][0]:.4f}")
    print(f"  信息维数 D(1): {metrics[' D(1)'][0]:.4f}")
    print(f"  关联维数 D(2): {metrics[' D(2)'][0]:.4f}")
    print(f"  Hurst指数 H: {metrics['H'][0]:.4f}")
    print(f"  多重分形强度: {metrics[''][0]:.4f}")
    
    return x, y, metrics, figure_data, file_name

def create_comprehensive_figure(results_list, output_dir):
    """创建综合对比图"""
    
    # 创建2×2画布
    fig = plt.figure(figsize=(16, 14))
    
    colors = ['steelblue', 'crimson', 'darkgreen']
    
    for idx, (x, y, metrics, figure_data, file_name) in enumerate(results_list):
        row = idx
        color = colors[idx % len(colors)]
        
        # 提取数据
        ql = figure_data['q']
        alpha_q = figure_data['(q)']  # α(q)
        f_alpha = figure_data['f()']   # f(α)
        D_q = figure_data['D(q)']
        
        # 1. 原始曲线（对数显示）- 左上
        ax1 = plt.subplot(2, 2, 1)
        ax1.semilogx(x, y, linewidth=2, color=color, alpha=0.8)
        ax1.set_xlabel(r'$T_2$ Signal (ms)', fontsize=14)
        ax1.set_ylabel('Porosity Increment', fontsize=14)
        ax1.set_title('NMR T2 Distribution', fontsize=15, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')
        ax1.tick_params(labelsize=12)
        
        # 2. 配分函数 ln(X) vs ln(ε) - 右上
        ax2 = plt.subplot(2, 2, 2)
        
        # 获取所有包含'_X'的键
        partition_keys = [k for k in figure_data.keys() if '_X' in k]
        
        if partition_keys:
            # 绘制配分函数
            temp_q_n = max(1, len(partition_keys) // 10)  # 显示约10条曲线
            
            for i, key in enumerate(partition_keys):
                if i % temp_q_n == 0:
                    key_r = key.replace('_X', '_r')
                    if key_r in figure_data:
                        log_r = figure_data[key_r]
                        log_X = figure_data[key]
                        
                        # 从键名中提取q值
                        q_str = key.replace('q=', '').replace('_X', '')
                        
                        # 使用不同颜色
                        colors_rand = plt.cm.viridis(i / len(partition_keys))
                        
                        ax2.scatter(log_r, log_X, s=30, alpha=0.6, color=colors_rand)
                        
                        # 拟合线
                        if len(log_r) > 1:
                            coeffs = np.polyfit(log_r, log_X, 1)
                            fit_line = np.poly1d(coeffs)
                            ax2.plot(log_r, fit_line(log_r), color=colors_rand, 
                                   linewidth=1.5, alpha=0.8, label=f'q={q_str}')
            
            # 只显示部分图例
            handles, labels = ax2.get_legend_handles_labels()
            if len(handles) > 0:
                ax2.legend(handles[::max(1, len(handles)//5)], labels[::max(1, len(handles)//5)], 
                          fontsize=10, loc='best', ncol=2)
        else:
            ax2.text(0.5, 0.5, 'No partition function data', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12, color='gray')
        
        ax2.set_xlabel(r'$\ln(\epsilon)$', fontsize=14)
        ax2.set_ylabel(r'$\ln(X_q)$', fontsize=14)
        ax2.set_title('Partition Function', fontsize=15, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax2.tick_params(labelsize=12)
        
        # 3. 多重分形谱 f(α) vs α - 左下
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(alpha_q, f_alpha, '^-', color='darkorange', linewidth=2.5, markersize=6, alpha=0.8)
        ax3.set_xlabel(r'$\alpha$', fontsize=14)
        ax3.set_ylabel(r'$f(\alpha)$', fontsize=14)
        ax3.set_title('Multifractal Spectrum', fontsize=15, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax3.tick_params(labelsize=12)
        
        # 标记q=0的位置
        idx_0 = ql.index(0) if 0 in ql else len(ql)//2
        if idx_0 < len(alpha_q):
            ax3.plot(alpha_q[idx_0], f_alpha[idx_0], 'ro', markersize=10, label='q=0', zorder=10)
            ax3.legend(fontsize=12)
        
        # 4. 广义维数 D(q) vs q - 右下
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(ql, D_q, 'd-', color='mediumpurple', linewidth=2.5, markersize=6, alpha=0.8)
        ax4.set_xlabel(r'$q$', fontsize=14)
        ax4.set_ylabel(r'$D(q)$', fontsize=14)
        ax4.set_title('Generalized Dimension', fontsize=15, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax4.tick_params(labelsize=12)
        
        # 标记D(0), D(1), D(2)
        for q_val in [0, 1, 2]:
            if q_val in ql:
                idx_q = ql.index(q_val)
                ax4.plot(q_val, D_q[idx_q], 'o', markersize=10, zorder=10)
                ax4.text(q_val, D_q[idx_q], f'  D({q_val})={D_q[idx_q]:.3f}', 
                        fontsize=11, verticalalignment='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存为多种格式
    for ext in ['eps', 'pdf', 'png']:
        output_file = os.path.join(output_dir, f"nmr_multifractal_analysis.{ext}")
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"已保存: nmr_multifractal_analysis.{ext}")
    
    plt.close(fig)

def create_metrics_comparison(results_list, output_dir):
    """创建指标对比图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    file_names = [r[4] for r in results_list]
    
    # 提取指标
    D0_list = [r[2][' D(0)'][0] for r in results_list]
    D1_list = [r[2][' D(1)'][0] for r in results_list]
    D2_list = [r[2][' D(2)'][0] for r in results_list]
    H_list = [r[2]['H'][0] for r in results_list]
    intensity_list = [r[2][''][0] for r in results_list]
    
    # 1. D(0), D(1), D(2)对比
    ax1 = axes[0, 0]
    x_pos = np.arange(len(file_names))
    width = 0.25
    ax1.bar(x_pos - width, D0_list, width, label='D(0)', color='steelblue', alpha=0.8)
    ax1.bar(x_pos, D1_list, width, label='D(1)', color='crimson', alpha=0.8)
    ax1.bar(x_pos + width, D2_list, width, label='D(2)', color='darkgreen', alpha=0.8)
    ax1.set_xlabel('Sample', fontsize=12)
    ax1.set_ylabel('Dimension', fontsize=12)
    ax1.set_title('(a) Generalized Dimensions', fontsize=13, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(file_names)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 2. Hurst指数对比
    ax2 = axes[0, 1]
    ax2.bar(file_names, H_list, color='darkorange', alpha=0.8)
    ax2.set_xlabel('Sample', fontsize=12)
    ax2.set_ylabel('Hurst Exponent', fontsize=12)
    ax2.set_title('(b) Hurst Exponent', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 3. 多重分形强度对比
    ax3 = axes[0, 2]
    ax3.bar(file_names, intensity_list, color='mediumpurple', alpha=0.8)
    ax3.set_xlabel('Sample', fontsize=12)
    ax3.set_ylabel('Multifractal Intensity', fontsize=12)
    ax3.set_title('(c) Multifractal Intensity', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 4. D(q)曲线对比
    ax4 = axes[1, 0]
    colors = ['steelblue', 'crimson', 'darkgreen']
    for idx, (x, y, metrics, figure_data, file_name) in enumerate(results_list):
        ql = figure_data['q']
        D_q = figure_data['D(q)']
        ax4.plot(ql, D_q, 'o-', color=colors[idx], linewidth=2, markersize=4, 
                label=file_name, alpha=0.8)
    ax4.set_xlabel(r'$q$', fontsize=12)
    ax4.set_ylabel(r'$D(q)$', fontsize=12)
    ax4.set_title('(d) D(q) Comparison', fontsize=13, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, linestyle='--')
    
    # 5. 多重分形谱对比
    ax5 = axes[1, 1]
    for idx, (x, y, metrics, figure_data, file_name) in enumerate(results_list):
        alpha_q = figure_data['(q)']
        f_alpha = figure_data['f()']
        ax5.plot(alpha_q, f_alpha, 'o-', color=colors[idx], linewidth=2, markersize=4,
                label=file_name, alpha=0.8)
    ax5.set_xlabel(r'$\alpha$', fontsize=12)
    ax5.set_ylabel(r'$f(\alpha)$', fontsize=12)
    ax5.set_title(r'(e) f($\alpha$) Comparison', fontsize=13, fontweight='bold')
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3, linestyle='--')
    
    # 6. τ(q)曲线对比
    ax6 = axes[1, 2]
    for idx, (x, y, metrics, figure_data, file_name) in enumerate(results_list):
        ql = figure_data['q']
        tau_q = figure_data['(q)']
        ax6.plot(ql, tau_q, 'o-', color=colors[idx], linewidth=2, markersize=4,
                label=file_name, alpha=0.8)
    ax6.set_xlabel(r'$q$', fontsize=12)
    ax6.set_ylabel(r'$\tau(q)$', fontsize=12)
    ax6.set_title(r'(f) $\tau(q)$ Comparison', fontsize=13, fontweight='bold')
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # 保存为多种格式
    for ext in ['eps', 'pdf', 'png']:
        output_file = os.path.join(output_dir, f"nmr_metrics_comparison.{ext}")
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"已保存: nmr_metrics_comparison.{ext}")
    
    plt.close(fig)

def main():
    # 获取根目录（上两级目录）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))
    
    # 数据文件（只保留Sample 3）
    data_files = [
        (os.path.join(root_dir, "3.xlsx"), "Sample 3")
    ]
    
    # 分析所有文件
    results_list = []
    for file_path, file_name in data_files:
        if os.path.exists(file_path):
            result = analyze_single_file(file_path, file_name)
            results_list.append(result)
            print()
        else:
            print(f"警告: 文件不存在 - {file_path}")
    
    if not results_list:
        print("错误: 没有找到可分析的数据文件！")
        return
    
    print("="*60)
    print("生成综合分析图像")
    print("="*60)
    
    # 生成综合对比图
    create_comprehensive_figure(results_list, current_dir)
    
    # 生成指标对比图
    create_metrics_comparison(results_list, current_dir)
    
    # 生成指标汇总表
    print("\n指标汇总表:")
    print("-"*80)
    print(f"{'样本':<15} {'D(0)':<10} {'D(1)':<10} {'D(2)':<10} {'Hurst':<10} {'强度':<10}")
    print("-"*80)
    for x, y, metrics, figure_data, file_name in results_list:
        print(f"{file_name:<15} "
              f"{metrics[' D(0)'][0]:<10.4f} "
              f"{metrics[' D(1)'][0]:<10.4f} "
              f"{metrics[' D(2)'][0]:<10.4f} "
              f"{metrics['H'][0]:<10.4f} "
              f"{metrics[''][0]:<10.4f}")
    print("-"*80)
    
    print("\n所有分析完成！")
    print("生成的文件:")
    print("  - nmr_multifractal_analysis.eps/pdf/png (综合分析图)")
    print("  - nmr_metrics_comparison.eps/pdf/png (指标对比图)")

if __name__ == '__main__':
    main()

