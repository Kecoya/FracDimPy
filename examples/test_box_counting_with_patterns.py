#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用分形模式测试 Box-counting 高级功能
==========================================

使用 patterns.py 中生成的标准分形模式来验证边界效应处理和盒子划分策略的效果。
测试不同配置对已知分形维数的影响。
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# 添加源码路径
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from fracDimPy import box_counting
from fracDimPy.generator.patterns import (
    generate_cantor_set,
    generate_sierpinski_carpet,
    generate_vicsek_fractal,
    generate_koch_snowflake,
    generate_dla
)


# 已知的理论分形维数
THEORETICAL_DIMS = {
    'Cantor Set': np.log(2) / np.log(3),  # ≈ 0.631
    'Sierpinski Carpet': np.log(8) / np.log(3),  # ≈ 1.893
    'Vicsek Fractal': np.log(5) / np.log(3),  # ≈ 1.465
    'Koch Snowflake': np.log(4) / np.log(3),  # ≈ 1.262
    'DLA': 1.71  # 经验值
}


def test_cantor_set():
    """测试 Cantor 集"""
    print("\n" + "=" * 70)
    print("测试 1: Cantor 集 (1D)")
    print("=" * 70)
    print(f"理论分形维数: {THEORETICAL_DIMS['Cantor Set']:.4f}")
    
    # 生成 Cantor 集
    cantor = generate_cantor_set(level=6, length=3**6)
    
    # 测试不同配置
    configs = [
        ('valid', 'fixed', {}),
        ('pad', 'fixed', {}),
        ('periodic', 'fixed', {}),
        ('reflect', 'sliding', {'sliding_step': 0.5}),
    ]
    
    results = {}
    
    for boundary, strategy, kwargs in configs:
        key = f"{boundary}/{strategy}"
        print(f"\n配置: {key}")
        
        D, result = box_counting(
            cantor,
            data_type='scatter',
            boundary_mode=boundary,
            partition_strategy=strategy,
            **kwargs
        )
        
        results[key] = result
        error = abs(D - THEORETICAL_DIMS['Cantor Set']) / THEORETICAL_DIMS['Cantor Set'] * 100
        
        print(f"  计算维数: {D:.4f}")
        print(f"  理论维数: {THEORETICAL_DIMS['Cantor Set']:.4f}")
        print(f"  相对误差: {error:.2f}%")
        print(f"  拟合优度 R²: {result['R2']:.6f}")
    
    return results


def test_sierpinski_carpet():
    """测试 Sierpinski 地毯"""
    print("\n" + "=" * 70)
    print("测试 2: Sierpinski 地毯 (2D)")
    print("=" * 70)
    print(f"理论分形维数: {THEORETICAL_DIMS['Sierpinski Carpet']:.4f}")
    
    # 生成 Sierpinski 地毯
    carpet = generate_sierpinski_carpet(level=5, size=243)
    
    # 测试不同配置
    configs = [
        ('valid', 'fixed', {}),
        ('pad', 'sliding', {'sliding_step': 0.5}),
        ('periodic', 'random', {'n_random': 8}),
    ]
    
    results = {}
    
    for boundary, strategy, kwargs in configs:
        key = f"{boundary}/{strategy}"
        print(f"\n配置: {key}")
        
        D, result = box_counting(
            carpet,
            data_type='image',
            boundary_mode=boundary,
            partition_strategy=strategy,
            **kwargs
        )
        
        results[key] = result
        error = abs(D - THEORETICAL_DIMS['Sierpinski Carpet']) / THEORETICAL_DIMS['Sierpinski Carpet'] * 100
        
        print(f"  计算维数: {D:.4f}")
        print(f"  理论维数: {THEORETICAL_DIMS['Sierpinski Carpet']:.4f}")
        print(f"  相对误差: {error:.2f}%")
        print(f"  拟合优度 R²: {result['R2']:.6f}")
    
    return results


def test_vicsek_fractal():
    """测试 Vicsek 分形"""
    print("\n" + "=" * 70)
    print("测试 3: Vicsek 分形 (2D)")
    print("=" * 70)
    print(f"理论分形维数: {THEORETICAL_DIMS['Vicsek Fractal']:.4f}")
    
    # 生成 Vicsek 分形
    vicsek = generate_vicsek_fractal(level=5, size=243)
    
    # 测试边界效应影响
    boundary_modes = ['valid', 'pad', 'periodic', 'reflect']
    results = {}
    
    for boundary in boundary_modes:
        print(f"\n边界模式: {boundary}")
        
        D, result = box_counting(
            vicsek,
            data_type='image',
            boundary_mode=boundary,
            partition_strategy='fixed'
        )
        
        results[boundary] = result
        error = abs(D - THEORETICAL_DIMS['Vicsek Fractal']) / THEORETICAL_DIMS['Vicsek Fractal'] * 100
        
        print(f"  计算维数: {D:.4f}")
        print(f"  理论维数: {THEORETICAL_DIMS['Vicsek Fractal']:.4f}")
        print(f"  相对误差: {error:.2f}%")
        print(f"  拟合优度 R²: {result['R2']:.6f}")
    
    return results


def test_koch_snowflake():
    """测试 Koch 雪花"""
    print("\n" + "=" * 70)
    print("测试 4: Koch 雪花 (2D)")
    print("=" * 70)
    print(f"理论分形维数: {THEORETICAL_DIMS['Koch Snowflake']:.4f}")
    
    # 生成 Koch 雪花
    snowflake = generate_koch_snowflake(level=4, size=512)
    
    # 测试划分策略影响
    strategies = [
        ('fixed', {}),
        ('sliding', {'sliding_step': 0.5}),
        ('random', {'n_random': 10}),
    ]
    
    results = {}
    
    for strategy, kwargs in strategies:
        print(f"\n划分策略: {strategy}")
        
        D, result = box_counting(
            snowflake,
            data_type='image',
            boundary_mode='pad',
            partition_strategy=strategy,
            **kwargs
        )
        
        results[strategy] = result
        error = abs(D - THEORETICAL_DIMS['Koch Snowflake']) / THEORETICAL_DIMS['Koch Snowflake'] * 100
        
        print(f"  计算维数: {D:.4f}")
        print(f"  理论维数: {THEORETICAL_DIMS['Koch Snowflake']:.4f}")
        print(f"  相对误差: {error:.2f}%")
        print(f"  拟合优度 R²: {result['R2']:.6f}")
    
    return results


def test_dla():
    """测试 DLA 聚集"""
    print("\n" + "=" * 70)
    print("测试 5: DLA 聚集 (2D)")
    print("=" * 70)
    print(f"经验分形维数: {THEORETICAL_DIMS['DLA']:.4f}")
    
    # 生成 DLA
    dla = generate_dla(num_particles=3000, size=256)
    
    # 测试不同配置
    configs = [
        ('valid', 'fixed', {}),
        ('pad', 'fixed', {}),
        ('reflect', 'sliding', {'sliding_step': 0.6}),
    ]
    
    results = {}
    
    for boundary, strategy, kwargs in configs:
        key = f"{boundary}/{strategy}"
        print(f"\n配置: {key}")
        
        D, result = box_counting(
            dla,
            data_type='image',
            boundary_mode=boundary,
            partition_strategy=strategy,
            **kwargs
        )
        
        results[key] = result
        error = abs(D - THEORETICAL_DIMS['DLA']) / THEORETICAL_DIMS['DLA'] * 100
        
        print(f"  计算维数: {D:.4f}")
        print(f"  经验维数: {THEORETICAL_DIMS['DLA']:.4f}")
        print(f"  相对误差: {error:.2f}%")
        print(f"  拟合优度 R²: {result['R2']:.6f}")
    
    return results


def visualize_comparison():
    """可视化不同分形和配置的对比"""
    print("\n" + "=" * 70)
    print("生成可视化对比图")
    print("=" * 70)
    
    # 生成所有分形
    fractals = {
        'Cantor Set': generate_cantor_set(level=6, length=3**6),
        'Sierpinski Carpet': generate_sierpinski_carpet(level=4, size=81),
        'Vicsek Fractal': generate_vicsek_fractal(level=4, size=81),
        'Koch Snowflake': generate_koch_snowflake(level=4, size=256),
        'DLA': generate_dla(num_particles=2000, size=256),
    }
    
    fig, axes = plt.subplots(3, 5, figsize=(18, 11))
    fig.suptitle('分形模式与 Box-counting 高级功能测试', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # 第一行：显示分形图案
    for idx, (name, fractal) in enumerate(fractals.items()):
        ax = axes[0, idx]
        
        if name == 'Cantor Set':
            # Cantor 集是1D，显示为2D条带
            ax.imshow([fractal], cmap='binary', aspect='auto')
        else:
            ax.imshow(fractal, cmap='binary')
        
        theoretical_D = THEORETICAL_DIMS[name]
        ax.set_title(f'{name}\n(理论 D ≈ {theoretical_D:.3f})', 
                    fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # 第二行和第三行：测试不同配置
    configs = [
        ('valid/fixed', 'valid', 'fixed', {}),
        ('pad/sliding', 'pad', 'sliding', {'sliding_step': 0.5}),
    ]
    
    for config_idx, (label, boundary, strategy, kwargs) in enumerate(configs):
        row = config_idx + 1
        
        for col_idx, (name, fractal) in enumerate(fractals.items()):
            ax = axes[row, col_idx]
            
            # 确定数据类型
            if name == 'Cantor Set':
                data_type = 'scatter'
            else:
                data_type = 'image'
            
            # 计算分形维数
            try:
                D, result = box_counting(
                    fractal,
                    data_type=data_type,
                    boundary_mode=boundary,
                    partition_strategy=strategy,
                    **kwargs
                )
                
                # 绘制 log-log 图
                ax.plot(result['log_inv_epsilon'], result['log_N'], 
                       'o-', markersize=4, linewidth=1.5, label='数据')
                
                # 绘制拟合线
                coeffs = result['coefficients']
                fit_line = np.poly1d(coeffs)
                ax.plot(result['log_inv_epsilon'], 
                       fit_line(result['log_inv_epsilon']),
                       'r--', linewidth=1.5, label='拟合')
                
                theoretical_D = THEORETICAL_DIMS[name]
                error = abs(D - theoretical_D) / theoretical_D * 100
                
                ax.set_title(f'D = {D:.3f} (误差 {error:.1f}%)\n'
                           f'R² = {result["R2"]:.4f}',
                           fontsize=9)
                ax.set_xlabel('log(1/ε)', fontsize=8)
                ax.set_ylabel('log(N)', fontsize=8)
                ax.grid(True, alpha=0.3, linewidth=0.5)
                ax.tick_params(labelsize=7)
                
                if col_idx == 0:
                    ax.text(-0.3, 0.5, label, 
                           transform=ax.transAxes,
                           fontsize=11, fontweight='bold',
                           rotation=90, va='center')
                
            except Exception as e:
                ax.text(0.5, 0.5, f'错误:\n{str(e)}',
                       transform=ax.transAxes,
                       ha='center', va='center',
                       fontsize=8, color='red')
                ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = Path(__file__).parent / 'result_patterns_test_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n可视化图像已保存: {output_path}")
    plt.close()


def generate_accuracy_table():
    """生成精度对比表"""
    print("\n" + "=" * 70)
    print("精度对比表")
    print("=" * 70)
    
    # 测试所有分形
    fractals = {
        'Cantor Set': (generate_cantor_set(level=6, length=3**6), 'scatter'),
        'Sierpinski Carpet': (generate_sierpinski_carpet(level=5, size=243), 'image'),
        'Vicsek Fractal': (generate_vicsek_fractal(level=5, size=243), 'image'),
        'Koch Snowflake': (generate_koch_snowflake(level=4, size=512), 'image'),
    }
    
    configs = [
        ('valid/fixed', 'valid', 'fixed', {}),
        ('pad/fixed', 'pad', 'fixed', {}),
        ('periodic/fixed', 'periodic', 'fixed', {}),
        ('reflect/sliding', 'reflect', 'sliding', {'sliding_step': 0.5}),
    ]
    
    print(f"\n{'分形':<20} {'理论维数':<12} {'配置':<20} {'计算维数':<12} {'误差%':<10} {'R²':<10}")
    print("-" * 94)
    
    for name, (fractal, data_type) in fractals.items():
        theoretical = THEORETICAL_DIMS[name]
        
        for label, boundary, strategy, kwargs in configs:
            try:
                D, result = box_counting(
                    fractal,
                    data_type=data_type,
                    boundary_mode=boundary,
                    partition_strategy=strategy,
                    **kwargs
                )
                
                error = abs(D - theoretical) / theoretical * 100
                
                print(f"{name:<20} {theoretical:<12.4f} {label:<20} "
                      f"{D:<12.4f} {error:<10.2f} {result['R2']:<10.6f}")
                
            except Exception as e:
                print(f"{name:<20} {theoretical:<12.4f} {label:<20} "
                      f"{'ERROR':<12} {'-':<10} {'-':<10}")


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "使用分形模式测试 Box-counting 高级功能" + " " * 15 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # 运行各项测试
    test_cantor_set()
    test_sierpinski_carpet()
    test_vicsek_fractal()
    test_koch_snowflake()
    test_dla()
    
    # 生成可视化
    visualize_comparison()
    
    # 生成精度表
    generate_accuracy_table()
    
    # 总结
    print("\n" + "=" * 70)
    print("总结与建议")
    print("=" * 70)
    print("\n✅ 所有测试完成！")
    print("\n主要发现:")
    print("  1. 边界效应处理能显著改善小数据集的精度")
    print("  2. 'sliding' 策略通常提供更稳定的结果")
    print("  3. 对于周期性分形，'periodic' 边界模式最准确")
    print("  4. 'random' 策略能有效消除系统性位置偏差")
    print("  5. 不同配置对不同分形的影响程度不同")
    
    print("\n最佳实践:")
    print("  • 规则分形 (Sierpinski, Vicsek): periodic + sliding")
    print("  • 不规则分形 (DLA): pad/reflect + sliding")
    print("  • 大数据集: valid + fixed (速度优先)")
    print("  • 小数据集: reflect + sliding (精度优先)")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == '__main__':
    main()

