#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NMR Data Multifractal Analysis
========================================
Perform multifractal analysis on NMR data files (1.xlsx, 2.xlsx, 3.xlsx) 
using the box-counting method. Generate publication-quality vector graphics.
"""

import numpy as np
import pandas as pd
import os
from fracDimPy import multifractal_curve
import matplotlib.pyplot as plt

# Use SciencePlots style for publication-quality figures
import scienceplots 
plt.style.use(['science', 'no-latex'])
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def load_nmr_data(file_path):
    """
    Load NMR data from Excel file.
    
    Parameters
    ----------
    file_path : str
        Path to the Excel file containing NMR data.
        
    Returns
    -------
    x : np.ndarray
        First column data (T2 signal).
    y : np.ndarray
        Second column data (porosity increment).
    """
    print(f"正在读取: {file_path}")
    df = pd.read_excel(file_path)
    print(f"  数据形状: {df.shape}")
    print(f"  列名: {df.columns.tolist()}")
    
    # Read first and second columns as X and Y
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values
    
    print(f"  X范围: {x.min():.4f} ~ {x.max():.4f}")
    print(f"  Y范围: {y.min():.4f} ~ {y.max():.4f}")
    
    return x, y

def analyze_single_file(file_path, file_name):
    """
    Analyze a single NMR data file and return results.
    
    Parameters
    ----------
    file_path : str
        Path to the NMR data file.
    file_name : str
        Name identifier for the file.
        
    Returns
    -------
    x : np.ndarray
        T2 signal data.
    y : np.ndarray
        Porosity increment data.
    metrics : dict
        Multifractal metrics dictionary.
    figure_data : dict
        Data for plotting multifractal curves.
    file_name : str
        File name identifier.
    """
    print("="*60)
    print(f"分析文件: {file_name}")
    print("="*60)
    
    # Load data
    x, y = load_nmr_data(file_path)
    
    # Perform multifractal analysis
    print("\n执行多重分形分析...")
    metrics, figure_data = multifractal_curve(
        (x, y),
        use_multiprocessing=False,
        data_type='dual'
    )
    
    # Print key metrics
    print("\n关键指标:")
    print(f"  容量维数 D(0): {metrics[' D(0)'][0]:.4f}")
    print(f"  信息维数 D(1): {metrics[' D(1)'][0]:.4f}")
    print(f"  关联维数 D(2): {metrics[' D(2)'][0]:.4f}")
    print(f"  Hurst指数 H: {metrics['H'][0]:.4f}")
    print(f"  多重分形强度: {metrics[''][0]:.4f}")
    
    return x, y, metrics, figure_data, file_name

def create_comprehensive_figure(results_list, output_dir):
    """
    Create comprehensive comparison figure with multiple subplots.
    
    Parameters
    ----------
    results_list : list
        List of analysis results, each containing (x, y, metrics, figure_data, file_name).
    output_dir : str
        Directory to save the output figures.
    """
    # Create 2×2 subplot layout
    fig = plt.figure(figsize=(16, 14))
    
    colors = ['steelblue', 'crimson', 'darkgreen']
    
    for idx, (x, y, metrics, figure_data, file_name) in enumerate(results_list):
        row = idx
        color = colors[idx % len(colors)]
        
        # Extract data from figure_data
        ql = figure_data['q']
        alpha_q = figure_data['(q)']  # α(q) - singularity strength
        f_alpha = figure_data['f()']   # f(α) - multifractal spectrum
        D_q = figure_data['D(q)']      # D(q) - generalized dimension
        
        # 1. Original curve (log scale) - top left
        ax1 = plt.subplot(2, 2, 1)
        ax1.semilogx(x, y, linewidth=2, color=color, alpha=0.8)
        ax1.set_xlabel(r'$T_2$ Signal (ms)', fontsize=14)
        ax1.set_ylabel('Porosity Increment', fontsize=14)
        ax1.set_title('NMR T2 Distribution', fontsize=15, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5, which='both')
        ax1.tick_params(labelsize=12)
        
        # 2. Partition function ln(X) vs ln(ε) - top right
        ax2 = plt.subplot(2, 2, 2)
        
        # Get all keys containing '_X' (partition function data)
        partition_keys = [k for k in figure_data.keys() if '_X' in k]
        
        if partition_keys:
            # Plot partition function curves
            temp_q_n = max(1, len(partition_keys) // 10)  # Display approximately 10 curves
            
            for i, key in enumerate(partition_keys):
                if i % temp_q_n == 0:
                    key_r = key.replace('_X', '_r')
                    if key_r in figure_data:
                        log_r = figure_data[key_r]
                        log_X = figure_data[key]
                        
                        # Extract q value from key name
                        q_str = key.replace('q=', '').replace('_X', '')
                        
                        # Use different colors for different q values
                        colors_rand = plt.cm.viridis(i / len(partition_keys))
                        
                        ax2.scatter(log_r, log_X, s=30, alpha=0.6, color=colors_rand)
                        
                        # Fit line
                        if len(log_r) > 1:
                            coeffs = np.polyfit(log_r, log_X, 1)
                            fit_line = np.poly1d(coeffs)
                            ax2.plot(log_r, fit_line(log_r), color=colors_rand, 
                                   linewidth=1.5, alpha=0.8, label=f'q={q_str}')
            
            # Display only part of the legend
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
        
        # 3. Multifractal spectrum f(α) vs α - bottom left
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(alpha_q, f_alpha, '^-', color='darkorange', linewidth=2.5, markersize=6, alpha=0.8)
        ax3.set_xlabel(r'$\alpha$', fontsize=14)
        ax3.set_ylabel(r'$f(\alpha)$', fontsize=14)
        ax3.set_title('Multifractal Spectrum', fontsize=15, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax3.tick_params(labelsize=12)
        
        # Mark q=0 position
        idx_0 = ql.index(0) if 0 in ql else len(ql)//2
        if idx_0 < len(alpha_q):
            ax3.plot(alpha_q[idx_0], f_alpha[idx_0], 'ro', markersize=10, label='q=0', zorder=10)
            ax3.legend(fontsize=12)
        
        # 4. Generalized dimension D(q) vs q - bottom right
        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(ql, D_q, 'd-', color='mediumpurple', linewidth=2.5, markersize=6, alpha=0.8)
        ax4.set_xlabel(r'$q$', fontsize=14)
        ax4.set_ylabel(r'$D(q)$', fontsize=14)
        ax4.set_title('Generalized Dimension', fontsize=15, fontweight='bold')
        ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax4.tick_params(labelsize=12)
        
        # Mark D(0), D(1), D(2)
        for q_val in [0, 1, 2]:
            if q_val in ql:
                idx_q = ql.index(q_val)
                ax4.plot(q_val, D_q[idx_q], 'o', markersize=10, zorder=10)
                ax4.text(q_val, D_q[idx_q], f'  D({q_val})={D_q[idx_q]:.3f}', 
                        fontsize=11, verticalalignment='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    # Save in multiple formats
    for ext in ['eps', 'pdf', 'png']:
        output_file = os.path.join(output_dir, f"nmr_multifractal_analysis.{ext}")
        fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"已保存: nmr_multifractal_analysis.{ext}")
    
    plt.close(fig)

def main():
    """
    Main function to perform multifractal analysis on NMR data files.
    """
    data_files=[('nmr.xlsx','sample')]
    
    # Analyze all files
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
    
    # Generate comprehensive comparison figure
    create_comprehensive_figure(results_list, current_dir)
    
    # Generate metrics summary table
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

if __name__ == '__main__':
    main()

