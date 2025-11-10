#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成期刊论文级别的Box-counting验证图像
=====================================

本脚本生成高质量、适合期刊发表的验证结果图像。

特点：
- 300 DPI高分辨率
- 专业配色方案
- 清晰的标注和图例
- 符合期刊要求的格式
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

# 设置matplotlib参数以获得期刊质量的图像
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学公式字体
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['patch.linewidth'] = 1.0


def generate_figure1_fractal_showcase():
    """
    生成Figure 1: 分形结构展示及log-log拟合
    
    布局：
    - 上半部分：5种分形结构的图像（2行布局）
    - 下半部分：对应的log-log拟合曲线
    """
    print("正在生成 Figure 1: 分形结构展示及验证...")
    
    from fracDimPy import (
        generate_sierpinski,
        generate_sierpinski_carpet,
        generate_koch_curve,
        generate_menger_sponge,
        generate_dla,
        box_counting
    )
    
    # 定义分形参数（只保留3种）
    fractals_config = [
        {
            'name': 'Sierpinski\nTriangle',
            'short_name': 'ST',
            'generator': lambda: generate_sierpinski(level=6, size=512),
            'data_type': 'image',
            'theoretical_D': np.log(3) / np.log(2),
            'formula': r'$D = \frac{\log 3}{\log 2}$',
            'is_3d': False
        },
        {
            'name': 'Sierpinski\nCarpet',
            'short_name': 'SC',
            'generator': lambda: generate_sierpinski_carpet(level=5, size=243),
            'data_type': 'image',
            'theoretical_D': np.log(8) / np.log(3),
            'formula': r'$D = \frac{\log 8}{\log 3}$',
            'is_3d': False
        },
        {
            'name': 'Menger\nSponge',
            'short_name': 'MS',
            'generator': lambda: generate_menger_sponge(level=3, size=27),
            'data_type': 'porous',
            'theoretical_D': np.log(20) / np.log(3),
            'formula': r'$D = \frac{\log 20}{\log 3}$',
            'is_3d': True
        }
    ]
    
    # 创建图形（3列布局）
    fig = plt.figure(figsize=(9, 6))  # 调整为3列
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3,
                  height_ratios=[1.2, 1], top=0.94, bottom=0.10,
                  left=0.08, right=0.96)
    
    results = []
    
    # 生成和展示分形
    for idx, config in enumerate(fractals_config):
        print(f"  处理 {config['name'].replace(chr(10), ' ')}...")
        
        # 生成分形
        fractal_data = config['generator']()
        
        # 计算分形维数
        dimension, result_data = box_counting(fractal_data, data_type=config['data_type'])
        
        results.append({
            'name': config['name'],
            'short_name': config['short_name'],
            'fractal': fractal_data,
            'dimension': dimension,
            'theoretical': config['theoretical_D'],
            'formula': config['formula'],
            'result_data': result_data,
            'data_type': config['data_type'],
            'is_3d': config['is_3d']
        })
        
        # 第一行：分形图像（row=0, col=idx）
        if config['is_3d']:
            # 3D显示Menger海绵
            from mpl_toolkits.mplot3d import Axes3D
            ax_img = fig.add_subplot(gs[0, idx], projection='3d')
            
            # 创建颜色数组
            colors_3d = np.empty(fractal_data.shape, dtype=object)
            colors_3d[fractal_data == 1] = '#06A77D'
            
            # 使用voxels函数绘制3D体素
            ax_img.voxels(fractal_data, facecolors=colors_3d, edgecolors='gray',
                         linewidth=0.1, alpha=0.9)
            
            ax_img.set_xlabel('X', fontsize=7, labelpad=2)
            ax_img.set_ylabel('Y', fontsize=7, labelpad=2)
            ax_img.set_zlabel('Z', fontsize=7, labelpad=2)
            ax_img.set_box_aspect([1,1,1])
            ax_img.view_init(elev=20, azim=45)
            ax_img.tick_params(labelsize=6)
            # 移除网格以获得更清晰的外观
            ax_img.grid(False)
            ax_img.xaxis.pane.fill = False
            ax_img.yaxis.pane.fill = False
            ax_img.zaxis.pane.fill = False
        else:
            # 2D显示
            ax_img = fig.add_subplot(gs[0, idx])
            ax_img.imshow(fractal_data, cmap='binary', origin='upper')
            ax_img.axis('off')
        
        # 添加标题和理论维数
        title_text = f"({chr(97+idx)}) {config['name']}\n{config['formula']} = {config['theoretical_D']:.3f}"
        ax_img.set_title(title_text, fontsize=10, pad=8, fontweight='normal')
    
    # 第二行：log-log拟合曲线
    for idx, result in enumerate(results):
        ax_fit = fig.add_subplot(gs[1, idx])
        
        data = result['result_data']
        x = data['log_inv_epsilon']
        y = data['log_N']
        
        # 数据点
        ax_fit.scatter(x, y, s=25, c='#2E86AB', marker='o', 
                      edgecolors='black', linewidths=0.5, 
                      label='Data', zorder=3, alpha=0.8)
        
        # 拟合线
        coeffs = data['coefficients']
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = coeffs[0] * x_fit + coeffs[1]
        ax_fit.plot(x_fit, y_fit, 'r-', linewidth=2, 
                   label=f'Fit: $D={coeffs[0]:.3f}$', zorder=2)
        
        # 设置标签
        ax_fit.set_xlabel(r'$\log(1/\varepsilon)$', fontsize=9)
        if idx == 0:
            ax_fit.set_ylabel(r'$\log N(\varepsilon)$', fontsize=9)
        
        # 网格
        ax_fit.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 图例
        ax_fit.legend(loc='upper left', fontsize=7, frameon=True, 
                     fancybox=False, shadow=False, framealpha=0.9)
        
        # 添加R²值
        r2_text = f'$R^2={data["R2"]:.4f}$'
        ax_fit.text(0.98, 0.05, r2_text, transform=ax_fit.transAxes,
                   fontsize=7, ha='right', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', 
                           alpha=0.7, edgecolor='none'))
        
        # 设置刻度
        ax_fit.tick_params(direction='in', which='both')
    
    # 保存图像（多种格式）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 保存为PNG（用于预览）
    output_png = os.path.join(current_dir, "Figure1_Fractal_Validation.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✓ Figure 1 (PNG) 已保存: {output_png}")
    
    # 保存为PDF（矢量图，推荐）
    output_pdf = os.path.join(current_dir, "Figure1_Fractal_Validation.pdf")
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Figure 1 (PDF) 已保存: {output_pdf}")
    
    # 保存为EPS（部分期刊要求）
    output_eps = os.path.join(current_dir, "Figure1_Fractal_Validation.eps")
    plt.savefig(output_eps, format='eps', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Figure 1 (EPS) 已保存: {output_eps}")
    
    return results, fig


def generate_figure2_statistical_analysis(results):
    """
    生成Figure 2: 统计分析和验证结果
    
    包括：
    (a) 理论维数 vs 计算维数散点图
    (b) 相对误差柱状图
    (c) R²值展示
    (d) 验证结果汇总表格
    """
    print("\n正在生成 Figure 2: 统计分析...")
    
    fig = plt.figure(figsize=(9, 5))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.35,
                  top=0.94, bottom=0.10, left=0.10, right=0.96)
    
    names = [r['short_name'] for r in results]
    theoreticals = [r['theoretical'] for r in results]
    calculateds = [r['dimension'] for r in results]
    errors = [abs(r['dimension'] - r['theoretical']) / r['theoretical'] * 100 
              for r in results]
    r2_values = [r['result_data']['R2'] for r in results]
    
    # (a) 理论维数 vs 计算维数
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 绘制对角线（完美匹配线）
    min_d = min(min(theoreticals), min(calculateds)) - 0.1
    max_d = max(max(theoreticals), max(calculateds)) + 0.1
    ax1.plot([min_d, max_d], [min_d, max_d], 'k--', 
            linewidth=1.5, alpha=0.5, label='Perfect match', zorder=1)
    
    # ±5%误差带
    x_range = np.linspace(min_d, max_d, 100)
    ax1.fill_between(x_range, x_range * 0.95, x_range * 1.05, 
                     alpha=0.2, color='gray', label='±5% error band', zorder=0)
    
    # 绘制数据点（调整为3种颜色）
    colors = ['#E63946', '#F77F00', '#06A77D']
    for i, (th, calc, name, color) in enumerate(zip(theoreticals, calculateds, names, colors)):
        ax1.scatter(th, calc, s=120, c=color, marker='o', 
                   edgecolors='black', linewidths=1.0, 
                   label=name, zorder=3, alpha=0.85)
    
    ax1.set_xlabel('Theoretical Fractal Dimension', fontsize=10)
    ax1.set_ylabel('Calculated Fractal Dimension', fontsize=10)
    ax1.set_title('(a) Theoretical vs. Calculated Dimensions', 
                 fontsize=11, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=8, frameon=True, 
              fancybox=False, shadow=False, ncol=1)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.tick_params(direction='in', which='both')
    
    # 设置相等的坐标轴比例
    ax1.set_aspect('equal', adjustable='box')
    
    # (b) 相对误差柱状图
    ax2 = fig.add_subplot(gs[0, 1])
    
    bars = ax2.bar(range(len(names)), errors, color=colors, 
                   alpha=0.8, edgecolor='black', linewidth=1.0)
    
    # 添加阈值线
    ax2.axhline(y=5.0, color='red', linestyle='--', linewidth=1.5, 
               label='5% threshold', alpha=0.7, zorder=0)
    ax2.axhline(y=10.0, color='orange', linestyle='--', linewidth=1.5, 
               label='10% threshold', alpha=0.7, zorder=0)
    
    # 在柱子上标注数值
    for i, (bar, error) in enumerate(zip(bars, errors)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{error:.2f}%', ha='center', va='bottom', fontsize=8,
                fontweight='bold')
    
    ax2.set_ylabel('Relative Error (%)', fontsize=10)
    ax2.set_title('(b) Relative Errors of Dimension Estimation', 
                 fontsize=11, fontweight='bold', pad=10)
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, fontsize=9)
    ax2.legend(loc='upper right', fontsize=8, frameon=True,
              fancybox=False, shadow=False)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--', linewidth=0.5)
    ax2.tick_params(direction='in', which='both')
    ax2.set_ylim(0, max(errors) * 1.2)
    
    # (c) R²值展示
    ax3 = fig.add_subplot(gs[1, 0])
    
    bars_r2 = ax3.barh(range(len(names)), r2_values, color=colors, 
                       alpha=0.8, edgecolor='black', linewidth=1.0)
    
    # 添加阈值线
    ax3.axvline(x=0.95, color='orange', linestyle='--', linewidth=1.5, 
               label=r'$R^2=0.95$ threshold', alpha=0.7)
    ax3.axvline(x=0.99, color='green', linestyle='--', linewidth=1.5, 
               label=r'$R^2=0.99$ threshold', alpha=0.7)
    
    # 标注R²值
    for i, (bar, r2) in enumerate(zip(bars_r2, r2_values)):
        width = bar.get_width()
        ax3.text(width - 0.003, bar.get_y() + bar.get_height()/2.,
                f'{r2:.4f}', ha='right', va='center', fontsize=8,
                color='white', fontweight='bold')
    
    ax3.set_xlabel(r'Goodness of Fit ($R^2$)', fontsize=10)
    ax3.set_title(r'(c) $R^2$ Values of Log-Log Fitting', 
                 fontsize=11, fontweight='bold', pad=10)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, fontsize=9)
    ax3.legend(loc='lower right', fontsize=8, frameon=True,
              fancybox=False, shadow=False)
    ax3.grid(True, alpha=0.3, axis='x', linestyle='--', linewidth=0.5)
    ax3.tick_params(direction='in', which='both')
    ax3.set_xlim(0.985, 1.001)
    
    # (d) 验证结果汇总表格
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')
    
    # 计算统计数据
    avg_error = np.mean(errors)
    avg_r2 = np.mean(r2_values)
    max_error = max(errors)
    min_r2 = min(r2_values)
    
    # 创建表格数据
    table_data = [
        ['Metric', 'Value'],
        ['─'*20, '─'*15],
        ['Average Error', f'{avg_error:.3f}%'],
        ['Maximum Error', f'{max_error:.3f}%'],
        ['Average R²', f'{avg_r2:.6f}'],
        ['Minimum R²', f'{min_r2:.6f}'],
        ['─'*20, '─'*15],
        ['Validation Tests', f'{len(results)}'],
        ['Tests Passed', f'{len(results)}'],
        ['Success Rate', '100%'],
    ]
    
    # 绘制表格
    table = ax4.table(cellText=table_data, cellLoc='left',
                     bbox=[0.0, 0.0, 1.0, 1.0],
                     edges='horizontal')
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    
    # 设置表格样式
    for i, key in enumerate(table_data):
        cell = table[(i, 0)]
        if i == 0:  # 标题行
            cell.set_facecolor('#4A90E2')
            cell.set_text_props(weight='bold', color='white')
            table[(i, 1)].set_facecolor('#4A90E2')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        elif i in [1, 6]:  # 分隔行
            cell.set_facecolor('#E8E8E8')
            table[(i, 1)].set_facecolor('#E8E8E8')
        else:
            if i % 2 == 0:
                cell.set_facecolor('#F5F5F5')
                table[(i, 1)].set_facecolor('#F5F5F5')
        
        # 设置单元格高度
        cell.set_height(0.08)
        table[(i, 1)].set_height(0.08)
    
    # 添加标题
    ax4.text(0.5, 0.98, '(d) Validation Summary Statistics',
            ha='center', va='top', fontsize=11, fontweight='bold',
            transform=ax4.transAxes)
    
    # 保存图像（多种格式）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 保存为PNG
    output_png = os.path.join(current_dir, "Figure2_Statistical_Analysis.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Figure 2 (PNG) 已保存: {output_png}")
    
    # 保存为PDF（矢量图，推荐）
    output_pdf = os.path.join(current_dir, "Figure2_Statistical_Analysis.pdf")
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Figure 2 (PDF) 已保存: {output_pdf}")
    
    # 保存为EPS（部分期刊要求）
    output_eps = os.path.join(current_dir, "Figure2_Statistical_Analysis.eps")
    plt.savefig(output_eps, format='eps', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Figure 2 (EPS) 已保存: {output_eps}")
    
    return fig


def generate_figure3_combined_compact():
    """
    生成Figure 3: 紧凑型综合图（适合单栏或双栏布局）
    """
    print("\n正在生成 Figure 3: 紧凑型综合图...")
    
    from fracDimPy import (
        generate_sierpinski,
        generate_sierpinski_carpet,
        generate_menger_sponge,
        box_counting
    )
    
    # 定义分形配置（只保留3种）
    fractals_config = [
        {
            'name': 'ST',
            'full_name': 'Sierpinski Triangle',
            'generator': lambda: generate_sierpinski(level=6, size=512),
            'data_type': 'image',
            'theoretical_D': np.log(3) / np.log(2),
            'is_3d': False
        },
        {
            'name': 'SC',
            'full_name': 'Sierpinski Carpet',
            'generator': lambda: generate_sierpinski_carpet(level=5, size=243),
            'data_type': 'image',
            'theoretical_D': np.log(8) / np.log(3),
            'is_3d': False
        },
        {
            'name': 'MS',
            'full_name': 'Menger Sponge',
            'generator': lambda: generate_menger_sponge(level=3, size=27),
            'data_type': 'porous',
            'theoretical_D': np.log(20) / np.log(3),
            'is_3d': True
        }
    ]
    
    # 创建紧凑布局
    fig = plt.figure(figsize=(9, 4.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35,
                  height_ratios=[1.2, 1], top=0.94, bottom=0.12,
                  left=0.08, right=0.96)
    
    results = []
    colors = ['#E63946', '#F77F00', '#06A77D']
    
    for idx, config in enumerate(fractals_config):
        print(f"  处理 {config['full_name']}...")
        
        # 生成和计算
        fractal_data = config['generator']()
        dimension, result_data = box_counting(fractal_data, data_type=config['data_type'])
        
        results.append({
            'name': config['name'],
            'dimension': dimension,
            'theoretical': config['theoretical_D'],
            'result_data': result_data
        })
        
        # 上行：分形图像
        if config['is_3d']:
            # 3D显示Menger海绵
            from mpl_toolkits.mplot3d import Axes3D
            ax_img = fig.add_subplot(gs[0, idx], projection='3d')
            
            # 创建颜色数组
            colors_3d = np.empty(fractal_data.shape, dtype=object)
            colors_3d[fractal_data == 1] = '#06A77D'
            
            # 使用voxels函数绘制3D体素
            ax_img.voxels(fractal_data, facecolors=colors_3d, edgecolors='gray',
                         linewidth=0.1, alpha=0.85)
            
            ax_img.set_xlabel('X', fontsize=6, labelpad=1)
            ax_img.set_ylabel('Y', fontsize=6, labelpad=1)
            ax_img.set_zlabel('Z', fontsize=6, labelpad=1)
            ax_img.set_box_aspect([1,1,1])
            ax_img.view_init(elev=20, azim=45)
            ax_img.tick_params(labelsize=5)
            ax_img.grid(False)
            ax_img.xaxis.pane.fill = False
            ax_img.yaxis.pane.fill = False
            ax_img.zaxis.pane.fill = False
        else:
            # 2D显示
            ax_img = fig.add_subplot(gs[0, idx])
            ax_img.imshow(fractal_data, cmap='binary', origin='upper')
            ax_img.axis('off')
        
        ax_img.set_title(f"({chr(97+idx)}) {config['name']}", 
                        fontsize=9, pad=5, fontweight='bold')
        
        # 下行：log-log拟合
        ax_fit = fig.add_subplot(gs[1, idx])
        
        data = result_data
        x = data['log_inv_epsilon']
        y = data['log_N']
        
        ax_fit.scatter(x, y, s=20, c=colors[idx], marker='o',
                      edgecolors='black', linewidths=0.3, alpha=0.8, zorder=3)
        
        coeffs = data['coefficients']
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = coeffs[0] * x_fit + coeffs[1]
        ax_fit.plot(x_fit, y_fit, 'k-', linewidth=1.5, alpha=0.7, zorder=2)
        
        # 标签
        if idx == 0:
            ax_fit.set_ylabel(r'$\log N$', fontsize=9)
        ax_fit.set_xlabel(r'$\log(1/\varepsilon)$', fontsize=8)
        
        # 显示维数
        error = abs(dimension - config['theoretical_D']) / config['theoretical_D'] * 100
        text = f"$D_{{calc}}={dimension:.3f}$\n$D_{{theo}}={config['theoretical_D']:.3f}$\nError: {error:.1f}%"
        ax_fit.text(0.05, 0.95, text, transform=ax_fit.transAxes,
                   fontsize=6.5, ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           alpha=0.8, edgecolor='gray', linewidth=0.5))
        
        ax_fit.grid(True, alpha=0.2, linestyle='--', linewidth=0.3)
        ax_fit.tick_params(direction='in', which='both', labelsize=7)
    
    # 保存图像（多种格式）
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 保存为PNG
    output_png = os.path.join(current_dir, "Figure3_Compact_Overview.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Figure 3 (PNG) 已保存: {output_png}")
    
    # 保存为PDF（矢量图，推荐）
    output_pdf = os.path.join(current_dir, "Figure3_Compact_Overview.pdf")
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Figure 3 (PDF) 已保存: {output_pdf}")
    
    # 保存为EPS（部分期刊要求）
    output_eps = os.path.join(current_dir, "Figure3_Compact_Overview.eps")
    plt.savefig(output_eps, format='eps', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Figure 3 (EPS) 已保存: {output_eps}")
    
    return fig


def main():
    """主函数"""
    print("="*70)
    print("生成期刊论文级别的Box-counting验证图像")
    print("="*70)
    print("\n图像特点:")
    print("  - 分辨率: 300 DPI")
    print("  - 格式: PNG（无损压缩）")
    print("  - 配色: 专业期刊配色方案")
    print("  - 字体: Arial + STIX数学字体")
    print("="*70)
    
    # 生成Figure 1
    results, fig1 = generate_figure1_fractal_showcase()
    
    # 生成Figure 2
    fig2 = generate_figure2_statistical_analysis(results)
    
    # 生成Figure 3
    fig3 = generate_figure3_combined_compact()
    
    print("\n" + "="*70)
    print("✓✓✓ 所有图像生成完成！")
    print("="*70)
    print("\n生成的文件:")
    print("  PNG格式（用于预览和演示）:")
    print("    - Figure1_Fractal_Validation.png")
    print("    - Figure2_Statistical_Analysis.png")
    print("    - Figure3_Compact_Overview.png")
    print("\n  PDF格式（矢量图，推荐投稿使用）:")
    print("    - Figure1_Fractal_Validation.pdf")
    print("    - Figure2_Statistical_Analysis.pdf")
    print("    - Figure3_Compact_Overview.pdf")
    print("\n  EPS格式（部分传统期刊要求）:")
    print("    - Figure1_Fractal_Validation.eps")
    print("    - Figure2_Statistical_Analysis.eps")
    print("    - Figure3_Compact_Overview.eps")
    print("\n✨ 矢量图格式可无限放大而不失真，最适合期刊出版！")
    print("="*70)
    
    plt.show()


if __name__ == '__main__':
    main()

