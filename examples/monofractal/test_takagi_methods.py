#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Different Box-Counting Methods on Takagi Surface
======================================================

This example tests and compares 6 different box-counting methods
for calculating the fractal dimension of Takagi surfaces.

The Takagi surface is a theoretical fractal surface with known
fractal dimension, making it ideal for method validation.
"""

import numpy as np
import matplotlib.pyplot as plt
from fracDimPy import generate_takagi_surface, box_counting

# 使用SciencePlots样式
try:
    import scienceplots 
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass

plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def test_methods_on_takagi():
    """测试所有6种方法在不同理论维数的Takagi曲面上"""
    
    print("="*80)
    print("Takagi曲面 Box-Counting 方法比较测试")
    print("="*80)
    
    # 理论维数列表
    theoretical_dims = [2.1, 2.3, 2.5, 2.7, 2.9]
    
    # 方法列表
    methods = {
        0: "RDCCM (Relative Differential)",
        1: "DCCM (Differential)",
        2: "CCM (Cubic Cover)",
        3: "ICCM (Interpolated)",
        5: "SCCM (Simplified)",
        6: "SDCCM (Simplified Differential)"
    }
    
    # 存储结果
    results = {method_id: [] for method_id in methods.keys()}
    
    # 参数设置
    level = 12
    size = 256
    
    print(f"\n参数设置: 尺寸={size}x{size}, 迭代层数={level}\n")
    
    # 测试每个理论维数
    for theo_D in theoretical_dims:
        print(f"\n{'='*80}")
        print(f"理论分形维数 D = {theo_D}")
        print(f"{'='*80}")
        
        # 生成Takagi曲面
        surface = generate_takagi_surface(dimension=theo_D, level=level, size=size)
        print(f"曲面生成完成: 高度范围 {surface.min():.4f} ~ {surface.max():.4f}, 标准差 {surface.std():.4f}")
        
        # 测试每种方法
        for method_id, method_name in methods.items():
            try:
                print(f"\n--- 方法 {method_id}: {method_name} ---")
                D_measured, result = box_counting(surface, data_type='surface', method=method_id)
                
                error = abs(D_measured - theo_D)
                rel_error = error / theo_D * 100
                
                print(f"测量维数: D = {D_measured:.4f}")
                print(f"绝对误差: ΔD = {error:.4f}")
                print(f"相对误差: {rel_error:.2f}%")
                print(f"拟合优度: R² = {result['R2']:.6f}")
                
                results[method_id].append({
                    'theoretical': theo_D,
                    'measured': D_measured,
                    'error': error,
                    'rel_error': rel_error,
                    'R2': result['R2']
                })
                
            except Exception as e:
                print(f"方法 {method_id} 失败: {e}")
                results[method_id].append({
                    'theoretical': theo_D,
                    'measured': np.nan,
                    'error': np.nan,
                    'rel_error': np.nan,
                    'R2': np.nan
                })
    
    # 可视化结果
    print(f"\n{'='*80}")
    print("生成对比图...")
    print(f"{'='*80}\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 测量值 vs 理论值
    ax1 = axes[0, 0]
    ax1.plot(theoretical_dims, theoretical_dims, 'k--', linewidth=2, label='理想线 (y=x)')
    
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', '^', 'v', 'd', 'p']
    
    for idx, (method_id, method_name) in enumerate(methods.items()):
        measured_vals = [r['measured'] for r in results[method_id]]
        ax1.plot(theoretical_dims, measured_vals, 
                marker=markers[idx], color=colors[idx], 
                linewidth=2, markersize=8, 
                label=f'{method_name}', alpha=0.8)
    
    ax1.set_xlabel('Theoretical Fractal Dimension', fontsize=13)
    ax1.set_ylabel('Measured Fractal Dimension', fontsize=13)
    ax1.set_title('Measured vs Theoretical Dimension', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=9, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=11)
    
    # 2. 绝对误差
    ax2 = axes[0, 1]
    for idx, (method_id, method_name) in enumerate(methods.items()):
        errors = [r['error'] for r in results[method_id]]
        ax2.plot(theoretical_dims, errors, 
                marker=markers[idx], color=colors[idx], 
                linewidth=2, markersize=8, 
                label=f'{method_name}', alpha=0.8)
    
    ax2.set_xlabel('Theoretical Fractal Dimension', fontsize=13)
    ax2.set_ylabel('Absolute Error |D_measured - D_theo|', fontsize=13)
    ax2.set_title('Absolute Error Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=11)
    
    # 3. 相对误差
    ax3 = axes[1, 0]
    for idx, (method_id, method_name) in enumerate(methods.items()):
        rel_errors = [r['rel_error'] for r in results[method_id]]
        ax3.plot(theoretical_dims, rel_errors, 
                marker=markers[idx], color=colors[idx], 
                linewidth=2, markersize=8, 
                label=f'{method_name}', alpha=0.8)
    
    ax3.set_xlabel('Theoretical Fractal Dimension', fontsize=13)
    ax3.set_ylabel('Relative Error (%)', fontsize=13)
    ax3.set_title('Relative Error Comparison', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9, loc='best')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(labelsize=11)
    
    # 4. R²拟合优度
    ax4 = axes[1, 1]
    for idx, (method_id, method_name) in enumerate(methods.items()):
        R2_vals = [r['R2'] for r in results[method_id]]
        ax4.plot(theoretical_dims, R2_vals, 
                marker=markers[idx], color=colors[idx], 
                linewidth=2, markersize=8, 
                label=f'{method_name}', alpha=0.8)
    
    ax4.set_xlabel('Theoretical Fractal Dimension', fontsize=13)
    ax4.set_ylabel('R² (Goodness of Fit)', fontsize=13)
    ax4.set_title('Fitting Quality Comparison', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=9, loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.tick_params(labelsize=11)
    ax4.set_ylim([0.95, 1.0])
    
    plt.tight_layout()
    
    # 保存图像
    for ext in ['pdf', 'png']:
        output_file = f"takagi_methods_comparison.{ext}"
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"已保存: {output_file}")
    
    plt.close()
    
    # 打印汇总表
    print(f"\n{'='*80}")
    print("结果汇总表")
    print(f"{'='*80}\n")
    
    for method_id, method_name in methods.items():
        print(f"\n{method_name} (Method {method_id}):")
        print(f"{'理论D':<10} {'测量D':<12} {'绝对误差':<12} {'相对误差(%)':<15} {'R²':<10}")
        print("-" * 70)
        for r in results[method_id]:
            print(f"{r['theoretical']:<10.1f} {r['measured']:<12.4f} {r['error']:<12.4f} "
                  f"{r['rel_error']:<15.2f} {r['R2']:<10.6f}")
    
    # 计算平均误差
    print(f"\n{'='*80}")
    print("平均误差统计")
    print(f"{'='*80}\n")
    print(f"{'方法':<35} {'平均绝对误差':<18} {'平均相对误差(%)':<18}")
    print("-" * 70)
    
    for method_id, method_name in methods.items():
        avg_abs_error = np.nanmean([r['error'] for r in results[method_id]])
        avg_rel_error = np.nanmean([r['rel_error'] for r in results[method_id]])
        print(f"{method_name:<35} {avg_abs_error:<18.4f} {avg_rel_error:<18.2f}")
    
    print(f"\n{'='*80}")
    print("测试完成！")
    print(f"{'='*80}")


if __name__ == '__main__':
    test_methods_on_takagi()

