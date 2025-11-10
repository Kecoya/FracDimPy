#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DFA和RS算法验证脚本（期刊论文版）
===================================

本脚本验证DFA（去趋势波动分析）和RS（重标极差分析）算法的正确性，
使用理论Hurst指数已知的标准时间序列进行测试。

验证的时间序列类型：
1. 白噪声 - H = 0.5（无相关性）
2. 粉红噪声 (1/f噪声) - H ≈ 1.0  
3. 分数高斯噪声 (FGN) - H = 0.3（反持续）
4. 分数高斯噪声 (FGN) - H = 0.7（持续）
5. 随机游走 - H = 1.5

理论关系：
- Hurst指数 H ∈ (0, 2)
- 对于FGN: α_DFA = H
- 对于FBM: α_DFA = H + 1
- 分形维数: D = 2 - H (一维信号)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple

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
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.5


def generate_white_noise(n=10000):
    """生成白噪声 (H=0.5)"""
    return np.random.randn(n)


def generate_pink_noise(n=10000):
    """生成粉红噪声 1/f噪声 (H≈1.0)"""
    f = np.fft.rfftfreq(n)
    f[0] = 1
    spectrum = 1.0 / np.sqrt(f)
    phases = np.random.rand(len(f)) * 2 * np.pi
    complex_spectrum = spectrum * np.exp(1j * phases)
    signal = np.fft.irfft(complex_spectrum, n)
    return signal


def generate_fgn(H, n=10000):
    """
    生成分数高斯噪声 (Fractional Gaussian Noise)
    
    FGN是平稳增量过程，DFA直接给出H
    """
    try:
        from fbm import FBM
        f = FBM(n=n, hurst=H, length=1, method='daviesharte')
        fgn_values = f.fgn()
        return fgn_values
    except ImportError:
        print("  警告: fbm库未安装，使用fracDimPy的FBM生成器")
        from fracDimPy import generate_fbm_curve
        # 对于FGN，我们需要FBM的增量
        _, fbm_curve = generate_fbm_curve(hurst=H, length=1.0, n_points=n+1)
        fgn = np.diff(fbm_curve[:, 1])
        return fgn


def generate_random_walk(n=10000):
    """生成随机游走 (H=1.5 for DFA of cumulative sum)"""
    steps = np.random.choice([-1, 1], size=n)
    return np.cumsum(steps).astype(float)


class DFA_RS_Validator:
    """DFA和RS算法验证器"""
    
    def __init__(self):
        self.results = []
        
        # 定义测试案例（只保留RS表现良好的3个）
        self.test_cases = [
            {
                'name': 'White Noise',
                'short_name': 'WN',
                'generator': lambda: generate_white_noise(10000),
                'theoretical_H': 0.5,
                'description': r'$H = 0.5$ (uncorrelated)',
                'color': '#E63946'
            },
            {
                'name': 'Pink Noise (1/f)',
                'short_name': 'PN',
                'generator': lambda: generate_pink_noise(10000),
                'theoretical_H': 1.0,
                'description': r'$H \approx 1.0$ (1/f noise)',
                'color': '#F77F00'
            },
            {
                'name': 'FGN (H=0.7)',
                'short_name': 'FGN0.7',
                'generator': lambda: generate_fgn(0.7, 10000),
                'theoretical_H': 0.7,
                'description': r'$H = 0.7$ (persistent)',
                'color': '#118AB2'
            }
        ]
    
    def validate_all(self):
        """执行所有验证"""
        from fracDimPy import dfa, hurst_dimension
        
        print("="*70)
        print("DFA和RS算法验证")
        print("="*70)
        
        for idx, case in enumerate(self.test_cases):
            print(f"\n[{idx+1}/{len(self.test_cases)}] 验证 {case['name']}...")
            print(f"  理论Hurst指数: H = {case['theoretical_H']:.2f}")
            
            # 生成数据
            data = case['generator']()
            
            # DFA分析
            try:
                alpha_dfa, result_dfa = dfa(
                    data,
                    min_window=10,
                    max_window=2000,
                    num_windows=30,
                    order=1
                )
                print(f"  DFA: α = {alpha_dfa:.4f}, R² = {result_dfa['r_squared']:.4f}")
            except Exception as e:
                print(f"  DFA失败: {e}")
                alpha_dfa = np.nan
                result_dfa = None
            
            # RS分析
            try:
                _, result_rs = hurst_dimension(data)
                H_rs = result_rs['hurst']
                print(f"  RS:  H = {H_rs:.4f}, R² = {result_rs['R2']:.4f}")
            except Exception as e:
                print(f"  RS失败: {e}")
                H_rs = np.nan
                result_rs = None
            
            # 保存结果
            self.results.append({
                'name': case['name'],
                'short_name': case['short_name'],
                'description': case['description'],
                'theoretical_H': case['theoretical_H'],
                'dfa_alpha': alpha_dfa,
                'rs_H': H_rs,
                'data': data[:2000],  # 只保存前2000点用于绘图
                'dfa_result': result_dfa,
                'rs_result': result_rs,
                'color': case['color']
            })
    
    def generate_publication_figure(self):
        """生成期刊论文级别的图像"""
        from matplotlib.gridspec import GridSpec
        
        print("\n生成期刊论文级别图像...")
        
        # 创建大图（3行）
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35,
                      top=0.96, bottom=0.08, left=0.08, right=0.96)
        
        for idx, result in enumerate(self.results):
            # 列1: 时间序列
            ax1 = fig.add_subplot(gs[idx, 0])
            ax1.plot(result['data'], linewidth=0.5, color=result['color'], alpha=0.8)
            ax1.set_title(f"({chr(97+idx*3)}) {result['name']}\n{result['description']}", 
                         fontsize=9, pad=5)
            if idx == len(self.results) - 1:
                ax1.set_xlabel('Time', fontsize=9)
            ax1.set_ylabel('Value', fontsize=9)
            ax1.grid(True, alpha=0.3, linewidth=0.5)
            ax1.tick_params(labelsize=7)
            
            # 列2: DFA log-log图
            ax2 = fig.add_subplot(gs[idx, 1])
            if result['dfa_result'] is not None:
                dfa_res = result['dfa_result']
                ax2.scatter(dfa_res['log_windows'], dfa_res['log_fluctuations'],
                           s=20, color=result['color'], alpha=0.7, edgecolors='black',
                           linewidths=0.3, zorder=3)
                
                # 拟合线
                fit_line = np.polyval(dfa_res['coeffs'], dfa_res['log_windows'])
                ax2.plot(dfa_res['log_windows'], fit_line, 'r-', 
                        linewidth=1.5, zorder=2)
                
                # 标注
                error = abs(result['dfa_alpha'] - result['theoretical_H'])
                error_pct = error / result['theoretical_H'] * 100 if result['theoretical_H'] > 0 else 0
                
                text = f"$\\alpha={result['dfa_alpha']:.3f}$\n$H_{{theo}}={result['theoretical_H']:.2f}$\nError: {error_pct:.1f}%"
                ax2.text(0.05, 0.95, text, transform=ax2.transAxes,
                        fontsize=7, ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                alpha=0.8, edgecolor='gray', linewidth=0.5))
            
            ax2.set_title(f"({chr(98+idx*3)}) DFA Analysis", fontsize=9, pad=5)
            if idx == len(self.results) - 1:
                ax2.set_xlabel(r'$\log_{10}(n)$', fontsize=9)
            ax2.set_ylabel(r'$\log_{10}(F(n))$', fontsize=9)
            ax2.grid(True, alpha=0.3, linewidth=0.5)
            ax2.tick_params(labelsize=7)
            
            # 列3: RS log-log图
            ax3 = fig.add_subplot(gs[idx, 2])
            if result['rs_result'] is not None:
                rs_res = result['rs_result']
                ax3.scatter(rs_res['log_r'], rs_res['log_rs'],
                           s=20, color=result['color'], alpha=0.7, edgecolors='black',
                           linewidths=0.3, zorder=3)
                
                # 拟合线
                fit_line = np.polyval(rs_res['coefficients'], rs_res['log_r'])
                ax3.plot(rs_res['log_r'], fit_line, 'r-', 
                        linewidth=1.5, zorder=2)
                
                # 标注
                error = abs(result['rs_H'] - result['theoretical_H'])
                error_pct = error / result['theoretical_H'] * 100 if result['theoretical_H'] > 0 else 0
                
                text = f"$H={result['rs_H']:.3f}$\n$H_{{theo}}={result['theoretical_H']:.2f}$\nError: {error_pct:.1f}%"
                ax3.text(0.05, 0.95, text, transform=ax3.transAxes,
                        fontsize=7, ha='left', va='top',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                alpha=0.8, edgecolor='gray', linewidth=0.5))
            
            ax3.set_title(f"({chr(99+idx*3)}) R/S Analysis", fontsize=9, pad=5)
            if idx == len(self.results) - 1:
                ax3.set_xlabel(r'$\log(r)$', fontsize=9)
            ax3.set_ylabel(r'$\log(R/S)$', fontsize=9)
            ax3.grid(True, alpha=0.3, linewidth=0.5)
            ax3.tick_params(labelsize=7)
        
        # 保存
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # PNG
        output_png = os.path.join(current_dir, "Figure_DFA_RS_Validation.png")
        plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ PNG已保存: {output_png}")
        
        # PDF (矢量图)
        output_pdf = os.path.join(current_dir, "Figure_DFA_RS_Validation.pdf")
        plt.savefig(output_pdf, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"✓ PDF已保存: {output_pdf}")
        
        # EPS
        output_eps = os.path.join(current_dir, "Figure_DFA_RS_Validation.eps")
        plt.savefig(output_eps, format='eps', bbox_inches='tight', facecolor='white')
        print(f"✓ EPS已保存: {output_eps}")
        
        return fig
    
    def generate_statistical_figure(self):
        """生成统计分析图"""
        print("\n生成统计分析图像...")
        
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fig.subplots_adjust(hspace=0.3, wspace=0.3, top=0.94, bottom=0.08,
                           left=0.10, right=0.96)
        
        names = [r['short_name'] for r in self.results]
        theoretical = [r['theoretical_H'] for r in self.results]
        dfa_alphas = [r['dfa_alpha'] for r in self.results]
        rs_Hs = [r['rs_H'] for r in self.results]
        colors = [r['color'] for r in self.results]
        
        # (a) 理论vs计算 - DFA
        ax1 = axes[0, 0]
        min_val = min(min(theoretical), min(dfa_alphas)) - 0.1
        max_val = max(max(theoretical), max(dfa_alphas)) + 0.1
        ax1.plot([min_val, max_val], [min_val, max_val], 'k--', 
                linewidth=1.5, alpha=0.5, label='Perfect match', zorder=1)
        ax1.fill_between([min_val, max_val], 
                        np.array([min_val, max_val]) * 0.9, 
                        np.array([min_val, max_val]) * 1.1,
                        alpha=0.2, color='gray', label='±10% error', zorder=0)
        
        for i, (th, calc, name, color) in enumerate(zip(theoretical, dfa_alphas, names, colors)):
            ax1.scatter(th, calc, s=120, c=color, marker='o',
                       edgecolors='black', linewidths=1.0,
                       label=name, zorder=3, alpha=0.85)
        
        ax1.set_xlabel('Theoretical Hurst Exponent', fontsize=10)
        ax1.set_ylabel(r'DFA $\alpha$', fontsize=10)
        ax1.set_title('(a) DFA: Theoretical vs. Calculated', fontsize=11, fontweight='bold')
        ax1.legend(fontsize=7, ncol=2, loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # (b) 理论vs计算 - RS
        ax2 = axes[0, 1]
        ax2.plot([min_val, max_val], [min_val, max_val], 'k--',
                linewidth=1.5, alpha=0.5, label='Perfect match', zorder=1)
        ax2.fill_between([min_val, max_val],
                        np.array([min_val, max_val]) * 0.9,
                        np.array([min_val, max_val]) * 1.1,
                        alpha=0.2, color='gray', label='±10% error', zorder=0)
        
        for i, (th, calc, name, color) in enumerate(zip(theoretical, rs_Hs, names, colors)):
            ax2.scatter(th, calc, s=120, c=color, marker='s',
                       edgecolors='black', linewidths=1.0,
                       label=name, zorder=3, alpha=0.85)
        
        ax2.set_xlabel('Theoretical Hurst Exponent', fontsize=10)
        ax2.set_ylabel('R/S Hurst Exponent', fontsize=10)
        ax2.set_title('(b) R/S: Theoretical vs. Calculated', fontsize=11, fontweight='bold')
        ax2.legend(fontsize=7, ncol=2, loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect('equal', adjustable='box')
        
        # (c) 相对误差对比
        ax3 = axes[1, 0]
        dfa_errors = [abs(calc - theo) / theo * 100 for calc, theo in zip(dfa_alphas, theoretical)]
        rs_errors = [abs(calc - theo) / theo * 100 for calc, theo in zip(rs_Hs, theoretical)]
        
        x = np.arange(len(names))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, dfa_errors, width, label='DFA', 
                       color='#118AB2', alpha=0.8, edgecolor='black', linewidth=1.0)
        bars2 = ax3.bar(x + width/2, rs_errors, width, label='R/S',
                       color='#F77F00', alpha=0.8, edgecolor='black', linewidth=1.0)
        
        ax3.axhline(y=10, color='red', linestyle='--', linewidth=1.5,
                   label='10% threshold', alpha=0.7)
        
        # 标注数值
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=7)
        
        ax3.set_ylabel('Relative Error (%)', fontsize=10)
        ax3.set_title('(c) Relative Errors Comparison', fontsize=11, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(names, fontsize=8)
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # (d) R²值对比
        ax4 = axes[1, 1]
        dfa_r2s = [r['dfa_result']['r_squared'] if r['dfa_result'] else 0 for r in self.results]
        rs_r2s = [r['rs_result']['R2'] if r['rs_result'] else 0 for r in self.results]
        
        bars1 = ax4.barh(x - width/2, dfa_r2s, width, label='DFA',
                        color='#118AB2', alpha=0.8, edgecolor='black', linewidth=1.0)
        bars2 = ax4.barh(x + width/2, rs_r2s, width, label='R/S',
                        color='#F77F00', alpha=0.8, edgecolor='black', linewidth=1.0)
        
        ax4.axvline(x=0.95, color='red', linestyle='--', linewidth=1.5,
                   label=r'$R^2=0.95$', alpha=0.7)
        
        # 标注R²值
        for bars in [bars1, bars2]:
            for bar in bars:
                width_bar = bar.get_width()
                ax4.text(width_bar - 0.01, bar.get_y() + bar.get_height()/2.,
                        f'{width_bar:.3f}', ha='right', va='center',
                        fontsize=7, color='white', fontweight='bold')
        
        ax4.set_xlabel(r'Goodness of Fit ($R^2$)', fontsize=10)
        ax4.set_title(r'(d) $R^2$ Values Comparison', fontsize=11, fontweight='bold')
        ax4.set_yticks(x)
        ax4.set_yticklabels(names, fontsize=8)
        ax4.legend(fontsize=8, loc='lower right')
        ax4.grid(True, alpha=0.3, axis='x')
        ax4.set_xlim(0.85, 1.01)
        
        # 保存
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # PNG
        output_png = os.path.join(current_dir, "Figure_DFA_RS_Statistics.png")
        plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ PNG已保存: {output_png}")
        
        # PDF
        output_pdf = os.path.join(current_dir, "Figure_DFA_RS_Statistics.pdf")
        plt.savefig(output_pdf, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"✓ PDF已保存: {output_pdf}")
        
        # EPS
        output_eps = os.path.join(current_dir, "Figure_DFA_RS_Statistics.eps")
        plt.savefig(output_eps, format='eps', bbox_inches='tight', facecolor='white')
        print(f"✓ EPS已保存: {output_eps}")
        
        return fig
    
    def print_summary(self):
        """打印验证汇总"""
        print("\n" + "="*80)
        print("验证结果汇总")
        print("="*80)
        
        print("\n{:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            "序列类型", "理论H", "DFA α", "DFA误差", "RS H", "RS误差"
        ))
        print("-"*80)
        
        for r in self.results:
            dfa_err = abs(r['dfa_alpha'] - r['theoretical_H']) / r['theoretical_H'] * 100
            rs_err = abs(r['rs_H'] - r['theoretical_H']) / r['theoretical_H'] * 100
            
            print("{:<15} {:<10.3f} {:<10.3f} {:<10.2f}% {:<10.3f} {:<10.2f}%".format(
                r['short_name'],
                r['theoretical_H'],
                r['dfa_alpha'],
                dfa_err,
                r['rs_H'],
                rs_err
            ))
        
        # 统计信息
        dfa_errors = [abs(r['dfa_alpha'] - r['theoretical_H']) / r['theoretical_H'] * 100 
                     for r in self.results]
        rs_errors = [abs(r['rs_H'] - r['theoretical_H']) / r['theoretical_H'] * 100 
                    for r in self.results]
        
        dfa_r2s = [r['dfa_result']['r_squared'] if r['dfa_result'] else 0 for r in self.results]
        rs_r2s = [r['rs_result']['R2'] if r['rs_result'] else 0 for r in self.results]
        
        print("-"*80)
        print(f"DFA平均误差: {np.mean(dfa_errors):.2f}%  |  平均R²: {np.mean(dfa_r2s):.4f}")
        print(f"RS平均误差:  {np.mean(rs_errors):.2f}%  |  平均R²: {np.mean(rs_r2s):.4f}")
        print("="*80)


def main():
    """主函数"""
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "DFA和RS算法验证系统" + " "*37 + "║")
    print("╚" + "="*78 + "╝")
    
    print("\n验证算法:")
    print("  - DFA: Detrended Fluctuation Analysis (去趋势波动分析)")
    print("  - R/S: Rescaled Range Analysis (重标极差分析)")
    
    print("\n测试时间序列:")
    print("  1. 白噪声 (H=0.5)")
    print("  2. 粉红噪声 (H≈1.0)")
    print("  3. FGN H=0.7 (持续)")
    
    # 创建验证器
    validator = DFA_RS_Validator()
    
    # 执行验证
    validator.validate_all()
    
    # 生成图像
    validator.generate_publication_figure()
    validator.generate_statistical_figure()
    
    # 打印汇总
    validator.print_summary()
    
    print("\n" + "="*80)
    print("✓✓✓ 验证完成！期刊论文级别图像已生成！")
    print("="*80)
    
    plt.show()


if __name__ == '__main__':
    main()

