#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MF-DFA
============

Multifractal Detrended Fluctuation Analysis
"""

import numpy as np
import os
import matplotlib.pyplot as plt
#  scienceplots 
try:
    import scienceplots
    plt.style.use(['science','no-latex'])
except ImportError:
    pass
# Microsoft YaHeiTimes New Roman
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 


def generate_binomial_cascade(n=8192, p=0.3):
    """"""
    levels = int(np.log2(n))
    series = np.ones(2**levels)
    
    for level in range(levels):
        step = 2**(levels - level)
        for i in range(0, 2**levels, step):
            # 
            if np.random.rand() < p:
                series[i:i+step//2] *= 1.7
                series[i+step//2:i+step] *= 0.3
            else:
                series[i:i+step//2] *= 0.3
                series[i+step//2:i+step] *= 1.7
    
    return series[:n]


def generate_fgn_for_mfdfa(H, n=10000):
    """FGNMF-DFA"""
    try:
        from fbm import FBM
        f = FBM(n=n, hurst=H, length=1, method='daviesharte')
        return f.fgn()
    except ImportError:
        print("fbm")
        return np.random.randn(n)


def main():
    print("="*60)
    print("MF-DFA")
    print("="*60)
    
    from fracDimPy import mf_dfa
    
    # q-10101000
    # 
    # q_list = [-5, -3, -1, 0, 1, 2, 3, 5]
    q_list = None  # 
    
    test_cases = []
    
    # 1. 
    print("\n1. ...")
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
            'name': '',
            'data': white_noise,
            'hq_result': hq_result_white,
            'spectrum': spectrum_white,
            'description': f'h(2)={h2:.3f}, ={width:.3f}'
        })
        
        print(f"   h(2) = {h2:.4f} (0.5)")
        print(f"     = {width:.4f}")
        print(f"   : {'' if width < 0.3 else ''}")
    except Exception as e:
        print(f"   : {e}")
    
    # 2. FGN (H=0.7, )
    print("\n2. FGN (H=0.7, )...")
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
            'description': f'h(2)={h2:.3f}, ={width:.3f}'
        })
        
        print(f"   h(2) = {h2:.4f} (0.7)")
        print(f"     = {width:.4f}")
        print(f"   : {'' if width < 0.3 else ''}")
    except Exception as e:
        print(f"   : {e}")
    
    # 3. 
    print("\n3. ...")
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
            'name': '',
            'data': cascade,
            'hq_result': hq_result_cascade,
            'spectrum': spectrum_cascade,
            'description': f'h(2)={h2:.3f}, ={width:.3f}'
        })
        
        print(f"   h(2) = {h2:.4f}")
        print(f"     = {width:.4f}")
        print(f"   : {'' if width < 0.3 else ''}")
    except Exception as e:
        print(f"   : {e}")
    
    # 
    if test_cases:
        n_cases = len(test_cases)
        fig = plt.figure(figsize=(18, 6*n_cases))
        
        for idx, case in enumerate(test_cases):
            # 1
            ax1 = fig.add_subplot(n_cases, 4, idx*4 + 1)
            ax1.plot(case['data'][:2000], linewidth=0.6)
            ax1.set_title(f"{case['name']}\n{case['description']}", fontsize=10)
            ax1.set_xlabel('', fontsize=9)
            ax1.set_ylabel('', fontsize=9)
            ax1.tick_params(labelsize=8)
            ax1.grid(True, alpha=0.3)
            
            # 2F_q(n) vs n (q)
            ax2 = fig.add_subplot(n_cases, 4, idx*4 + 2)
            hq_result = case['hq_result']
            
            # q(1000)q
            q_all = hq_result['q_list']
            if len(q_all) > 20:
                # q012
                q_to_plot_idx = []
                key_q_values = [-10, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 10]
                for key_q in key_q_values:
                    # q
                    q_idx = np.argmin(np.abs(q_all - key_q))
                    if q_idx not in q_to_plot_idx:
                        q_to_plot_idx.append(q_idx)
                q_to_plot_idx = sorted(q_to_plot_idx)
            else:
                q_to_plot_idx = range(len(q_all))
            
            # q
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
            ax2.set_title('', fontsize=10)
            ax2.legend(fontsize=7, ncol=2)
            ax2.tick_params(labelsize=8)
            ax2.grid(True, alpha=0.3)
            
            # 3h(q) vs q
            ax3 = fig.add_subplot(n_cases, 4, idx*4 + 3)
            q_vals = hq_result['q_list']
            h_vals = hq_result['h_q']
            
            valid = np.isfinite(h_vals)
            # qmarker
            if len(q_vals) > 50:
                ax3.plot(q_vals[valid], h_vals[valid], '-', 
                        color='blue', linewidth=2)
            else:
                ax3.plot(q_vals[valid], h_vals[valid], 'o-', 
                        color='blue', linewidth=2, markersize=6)
            
            ax3.set_xlabel('q', fontsize=9)
            ax3.set_ylabel('h(q)', fontsize=9)
            ax3.set_title('Hurst', fontsize=10)
            ax3.tick_params(labelsize=8)
            ax3.grid(True, alpha=0.3)
            
            # h(2)
            h2_idx = np.where(q_vals == 2)[0]
            if len(h2_idx) > 0:
                h2 = h_vals[h2_idx[0]]
                ax3.axhline(h2, color='red', linestyle='--', 
                           alpha=0.5, label=f'h(2)={h2:.3f}')
                ax3.legend(fontsize=8)
            
            # 4 f()
            ax4 = fig.add_subplot(n_cases, 4, idx*4 + 4)
            spectrum = case['spectrum']
            
            alpha_vals = spectrum['alpha']
            f_vals = spectrum['f_alpha']
            
            valid = np.isfinite(alpha_vals) & np.isfinite(f_vals)
            if np.sum(valid) > 0:
                ax4.plot(alpha_vals[valid], f_vals[valid], 'o-', 
                        color='green', linewidth=2, markersize=8)
                
                # _0
                if np.isfinite(spectrum['alpha_0']):
                    ax4.axvline(spectrum['alpha_0'], color='red', 
                               linestyle='--', alpha=0.5, 
                               label=f"={spectrum['alpha_0']:.3f}")
                
                # 
                alpha_min = np.min(alpha_vals[valid])
                alpha_max = np.max(alpha_vals[valid])
                ax4.annotate('', xy=(alpha_max, 0.1), xytext=(alpha_min, 0.1),
                            arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
                ax4.text((alpha_min + alpha_max)/2, 0.15, 
                        f'={spectrum["width"]:.3f}',
                        ha='center', fontsize=8, color='blue')
            
            ax4.set_xlabel(' ()', fontsize=9)
            ax4.set_ylabel('f()', fontsize=9)
            ax4.set_title('', fontsize=10)
            ax4.tick_params(labelsize=8)
            ax4.legend(fontsize=8)
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(current_dir, "result_mf_dfa.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n: {output_file}")
        plt.show()
    
    print("\n")
    print("\n")
    print("   MF-DFA")
    print("   ")
    print("\n   ")
    print("   - q: [-10, 10]1000")
    print("   - ")
    print("\n   ")
    print("   - h(q): Hurst")
    print("   - (q):  = qh(q) - 1")
    print("   - : Hlder")
    print("   - f(): ")
    print("\n   ")
    print("   - h(q)   < 0.3")
    print("   - h(q) q > 0.5")
    print("   -   = _max - _min ")
    print("\n   ")
    print("   - q > 0: ")
    print("   - q < 0: ")
    print("   - q = 2: DFA")


if __name__ == '__main__':
    main()

