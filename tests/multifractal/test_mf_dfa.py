#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MF-DFA (Multifractal Detrended Fluctuation Analysis)
====================================================

This example demonstrates how to use fracDimPy to perform Multifractal Detrended 
Fluctuation Analysis (MF-DFA) on time series data. MF-DFA is a powerful method for 
analyzing the multifractal properties of non-stationary time series by combining 
detrending and fluctuation analysis.

Main Features:
- Generate test data: white noise, fractional Gaussian noise (FGN), binomial cascade
- Perform MF-DFA analysis on different types of time series
- Calculate generalized Hurst exponent h(q) and multifractal spectrum
- Visualize analysis results including F_q(n) scaling, h(q) curve, and f(α) spectrum

Theoretical Background:
- MF-DFA extends DFA by analyzing q-order moments of fluctuations
- Generalized Hurst exponent h(q) describes scaling behavior for different q values
- Mass exponent τ(q) = qh(q) - 1
- Hölder exponent α(q) = dτ(q)/dq
- Multifractal spectrum f(α) describes the distribution of singularities
- Spectrum width Δα = α_max - α_min indicates the degree of multifractality
"""

import numpy as np
import os
import matplotlib.pyplot as plt
# SciencePlots style
try:
    import scienceplots
    plt.style.use(['science','no-latex'])
except ImportError:
    pass
# Set font family: Times New Roman and Microsoft YaHei
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue


def generate_binomial_cascade(n=8192, p=0.3):
    """Generate binomial cascade series for multifractal testing
    
    Args:
        n: Length of the series
        p: Probability parameter for cascade splitting
        
    Returns:
        Binomial cascade series with multifractal properties
    """
    levels = int(np.log2(n))
    series = np.ones(2**levels)
    
    for level in range(levels):
        step = 2**(levels - level)
        for i in range(0, 2**levels, step):
            # Randomly split each segment with probability p
            if np.random.rand() < p:
                series[i:i+step//2] *= 1.7
                series[i+step//2:i+step] *= 0.3
            else:
                series[i:i+step//2] *= 0.3
                series[i+step//2:i+step] *= 1.7
    
    return series[:n]


def generate_fgn_for_mfdfa(H, n=10000):
    """Generate Fractional Gaussian Noise (FGN) for MF-DFA testing
    
    Args:
        H: Hurst exponent (0 < H < 1)
        n: Length of the series
        
    Returns:
        FGN series with specified Hurst exponent
    """
    try:
        from fbm import FBM
        f = FBM(n=n, hurst=H, length=1, method='daviesharte')
        return f.fgn()
    except ImportError:
        print("Warning: fbm package not available, using random noise instead")
        return np.random.randn(n)


def main():
    print("="*60)
    print("MF-DFA")
    print("="*60)
    
    from fracDimPy import mf_dfa
    
    # q_list parameter: if None, automatically generates q values from -10 to 10 with 1000 points
    # You can also specify custom q values, e.g., q_list = [-5, -3, -1, 0, 1, 2, 3, 5]
    q_list = None  # Use default q range
    
    test_cases = []
    
    # 1. White noise test
    print("\n1. Testing white noise...")
    white_noise = np.random.randn(10000)
    try:
        hq_result_white, spectrum_white = mf_dfa(
            white_noise,
            q_list=q_list,
            min_window=10,
            max_window=1000,
            num_windows=25
        )
        
        h2 = hq_result_white['h_q'][hq_result_white['q_list'] == 2][0]
        width = spectrum_white['width']
        
        test_cases.append({
            'name': 'White Noise',
            'data': white_noise,
            'hq_result': hq_result_white,
            'spectrum': spectrum_white,
            'description': f'h(2)={h2:.3f}, width={width:.3f}'
        })
        
        print(f"   h(2) = {h2:.4f} (expected ~0.5)")
        print(f"   Spectrum width = {width:.4f}")
        print(f"   Result: {'Monofractal' if width < 0.3 else 'Multifractal'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. FGN (H=0.7, monofractal)
    print("\n2. Testing FGN (H=0.7, monofractal)...")
    fgn_07 = generate_fgn_for_mfdfa(H=0.7, n=10000)
    try:
        hq_result_fgn, spectrum_fgn = mf_dfa(
            fgn_07,
            q_list=q_list,
            min_window=10,
            max_window=1000,
            num_windows=25
        )
        
        h2 = hq_result_fgn['h_q'][hq_result_fgn['q_list'] == 2][0]
        width = spectrum_fgn['width']
        
        test_cases.append({
            'name': 'FGN (H=0.7)',
            'data': fgn_07,
            'hq_result': hq_result_fgn,
            'spectrum': spectrum_fgn,
            'description': f'h(2)={h2:.3f}, width={width:.3f}'
        })
        
        print(f"   h(2) = {h2:.4f} (expected ~0.7)")
        print(f"   Spectrum width = {width:.4f}")
        print(f"   Result: {'Monofractal' if width < 0.3 else 'Multifractal'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. Binomial cascade test
    print("\n3. Testing binomial cascade...")
    cascade = generate_binomial_cascade(n=8192, p=0.3)
    try:
        hq_result_cascade, spectrum_cascade = mf_dfa(
            cascade,
            q_list=q_list,
            min_window=10,
            max_window=1000,
            num_windows=25
        )
        
        h2 = hq_result_cascade['h_q'][hq_result_cascade['q_list'] == 2][0]
        width = spectrum_cascade['width']
        
        test_cases.append({
            'name': 'Binomial Cascade',
            'data': cascade,
            'hq_result': hq_result_cascade,
            'spectrum': spectrum_cascade,
            'description': f'h(2)={h2:.3f}, width={width:.3f}'
        })
        
        print(f"   h(2) = {h2:.4f}")
        print(f"   Spectrum width = {width:.4f}")
        print(f"   Result: {'Monofractal' if width < 0.3 else 'Multifractal'}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Visualize results
    if test_cases:
        n_cases = len(test_cases)
        fig = plt.figure(figsize=(18, 6*n_cases))
        
        for idx, case in enumerate(test_cases):
            # Subplot 1: Time series data
            ax1 = fig.add_subplot(n_cases, 4, idx*4 + 1)
            ax1.plot(case['data'][:2000], linewidth=0.6)
            ax1.set_title(f"{case['name']}\n{case['description']}", fontsize=10)
            ax1.set_xlabel('Time', fontsize=9)
            ax1.set_ylabel('Value', fontsize=9)
            ax1.tick_params(labelsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Subplot 2: F_q(n) vs n for different q values
            ax2 = fig.add_subplot(n_cases, 4, idx*4 + 2)
            hq_result = case['hq_result']
            
            # Select representative q values for plotting (if q_list has 1000 points, select key ones)
            q_all = hq_result['q_list']
            if len(q_all) > 20:
                # Select key q values: 0, 1, 2, and some negative/positive values
                q_to_plot_idx = []
                key_q_values = [-10, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 10]
                for key_q in key_q_values:
                    # Find closest q value
                    q_idx = np.argmin(np.abs(q_all - key_q))
                    if q_idx not in q_to_plot_idx:
                        q_to_plot_idx.append(q_idx)
                q_to_plot_idx = sorted(q_to_plot_idx)
            else:
                q_to_plot_idx = range(len(q_all))
            
            # Plot F_q(n) for selected q values
            colors = plt.cm.RdYlBu(np.linspace(0, 1, len(q_to_plot_idx)))
            for i, i_q in enumerate(q_to_plot_idx):
                q = q_all[i_q]
                Fq = hq_result['Fq_n'][i_q, :]
                valid = (Fq > 0) & np.isfinite(Fq)
                
                if np.sum(valid) > 0:
                    log_n = np.log10(hq_result['window_sizes'][valid])
                    log_Fq = np.log10(Fq[valid])
                    
                    ax2.plot(log_n, log_Fq, 'o-', 
                            color=colors[i], 
                            label=f'q={q:.1f}',
                            markersize=3,
                            linewidth=1.2,
                            alpha=0.8)
            
            ax2.set_xlabel('log(n)', fontsize=9)
            ax2.set_ylabel('log(F_q(n))', fontsize=9)
            ax2.set_title('Fluctuation Function', fontsize=10)
            ax2.legend(fontsize=7, ncol=2)
            ax2.tick_params(labelsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Subplot 3: h(q) vs q
            ax3 = fig.add_subplot(n_cases, 4, idx*4 + 3)
            q_vals = hq_result['q_list']
            h_vals = hq_result['h_q']
            
            valid = np.isfinite(h_vals)
            # Use line plot for many q values, markers for few
            if len(q_vals) > 50:
                ax3.plot(q_vals[valid], h_vals[valid], '-', 
                        color='blue', linewidth=2)
            else:
                ax3.plot(q_vals[valid], h_vals[valid], 'o-', 
                        color='blue', linewidth=2, markersize=6)
            
            ax3.set_xlabel('q', fontsize=9)
            ax3.set_ylabel('h(q)', fontsize=9)
            ax3.set_title('Generalized Hurst Exponent', fontsize=10)
            ax3.tick_params(labelsize=8)
            ax3.grid(True, alpha=0.3)
            
            # Mark h(2) value
            h2_idx = np.where(q_vals == 2)[0]
            if len(h2_idx) > 0:
                h2 = h_vals[h2_idx[0]]
                ax3.axhline(h2, color='red', linestyle='--', 
                           alpha=0.5, label=f'h(2)={h2:.3f}')
                ax3.legend(fontsize=8)
            
            # Subplot 4: f(α) multifractal spectrum
            ax4 = fig.add_subplot(n_cases, 4, idx*4 + 4)
            spectrum = case['spectrum']
            
            alpha_vals = spectrum['alpha']
            f_vals = spectrum['f_alpha']
            
            valid = np.isfinite(alpha_vals) & np.isfinite(f_vals)
            if np.sum(valid) > 0:
                ax4.plot(alpha_vals[valid], f_vals[valid], 'o-', 
                        color='green', linewidth=2, markersize=8)
                
                # Mark alpha_0 (most probable singularity)
                if np.isfinite(spectrum['alpha_0']):
                    ax4.axvline(spectrum['alpha_0'], color='red', 
                               linestyle='--', alpha=0.5, 
                               label=f"α₀={spectrum['alpha_0']:.3f}")
                
                # Annotate spectrum width
                alpha_min = np.min(alpha_vals[valid])
                alpha_max = np.max(alpha_vals[valid])
                ax4.annotate('', xy=(alpha_max, 0.1), xytext=(alpha_min, 0.1),
                            arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
                ax4.text((alpha_min + alpha_max)/2, 0.15, 
                        f'width={spectrum["width"]:.3f}',
                        ha='center', fontsize=8, color='blue')
            
            ax4.set_xlabel('α (Singularity Index)', fontsize=9)
            ax4.set_ylabel('f(α)', fontsize=9)
            ax4.set_title('Multifractal Spectrum', fontsize=10)
            ax4.tick_params(labelsize=8)
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(current_dir, "result_mf_dfa.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nResults saved to: {output_file}")
        plt.show()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("\n   MF-DFA Analysis Complete")
    print("   =========================")
    print("\n   Key Parameters:")
    print("   - q range: [-10, 10] with 1000 points (default)")
    print("   - Window sizes: logarithmically spaced")
    print("\n   Calculated Quantities:")
    print("   - h(q): Generalized Hurst exponent")
    print("   - τ(q): Mass exponent = qh(q) - 1")
    print("   - α(q): Hölder exponent = dτ(q)/dq")
    print("   - f(α): Multifractal spectrum")
    print("\n   Interpretation:")
    print("   - Monofractal: spectrum width < 0.3")
    print("   - Multifractal: h(q) varies significantly with q > 0.5")
    print("   - Spectrum width = α_max - α_min indicates multifractality strength")
    print("\n   Physical Meaning:")
    print("   - q > 0: Emphasizes large fluctuations")
    print("   - q < 0: Emphasizes small fluctuations")
    print("   - q = 2: Standard DFA analysis")


if __name__ == '__main__':
    main()

