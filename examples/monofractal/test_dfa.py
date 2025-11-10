#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DFA
==========

Detrended Fluctuation Analysis
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


def generate_fgn(H, n=10000):
    """FGNFBM
    
    DFAFGN   H
          DFAFBM   H + 1
    """
    try:
        from fbm import FBM
        f = FBM(n=n, hurst=H, length=1, method='daviesharte')
        # fgn()
        fgn_values = f.fgn()
        return fgn_values
    except ImportError:
        print("fbmfracDimPyFBM")
        from fracDimPy import generate_fbm_curve
        D = 2 - H
        fbm_curve, _ = generate_fbm_curve(dimension=D, length=n+1)
        # FGN = diff(FBM)
        fgn = np.diff(fbm_curve)
        return fgn


def generate_white_noise(n=10000):
    """"""
    return np.random.randn(n)


def generate_pink_noise(n=10000):
    """1/f"""
    # 
    f = np.fft.rfftfreq(n)
    f[0] = 1  # 0
    
    # 1/f 
    spectrum = 1.0 / np.sqrt(f)
    
    # 
    phases = np.random.rand(len(f)) * 2 * np.pi
    complex_spectrum = spectrum * np.exp(1j * phases)
    
    # FFT
    signal = np.fft.irfft(complex_spectrum, n)
    
    return signal


def generate_random_walk(n=10000):
    """"""
    steps = np.random.choice([-1, 1], size=n)
    return np.cumsum(steps)


def main():
    print("="*60)
    print("DFA")
    print("="*60)
    
    from fracDimPy import dfa
    
    # 
    test_cases = []
    
    # 1. =0.5
    print("\n1. ...")
    white_noise = generate_white_noise(n=10000)
    alpha_theory_white = 0.5
    try:
        alpha_white, result_white = dfa(
            white_noise,
            min_window=10,
            max_window=1000,
            num_windows=25,
            order=1
        )
        test_cases.append({
            'name': '',
            'data': white_noise,
            'alpha_measured': alpha_white,
            'alpha_theory': alpha_theory_white,
            'dimension': result_white['dimension'],
            'result': result_white,
            'description': '=0.5'
        })
        print(f"   : {alpha_theory_white:.2f}")
        print(f"   : {alpha_white:.4f}")
        print(f"   : {result_white['dimension']:.4f}")
        print(f"   R: {result_white['r_squared']:.4f}")
    except Exception as e:
        print(f"   : {e}")
    
    # 2. /1/f1.0
    print("\n2. 1/f...")
    pink_noise = generate_pink_noise(n=10000)
    alpha_theory_pink = 1.0
    try:
        alpha_pink, result_pink = dfa(
            pink_noise,
            min_window=10,
            max_window=1000,
            num_windows=25,
            order=1
        )
        test_cases.append({
            'name': ' (1/f)',
            'data': pink_noise,
            'alpha_measured': alpha_pink,
            'alpha_theory': alpha_theory_pink,
            'dimension': result_pink['dimension'],
            'result': result_pink,
            'description': '1.0'
        })
        print(f"   : ~{alpha_theory_pink:.2f}")
        print(f"   : {alpha_pink:.4f}")
        print(f"   : {result_pink['dimension']:.4f}")
        print(f"   R: {result_pink['r_squared']:.4f}")
    except Exception as e:
        print(f"   : {e}")
    
    # 3. 1.5
    print("\n3. ...")
    random_walk = generate_random_walk(n=10000)
    alpha_theory_rw = 1.5
    try:
        alpha_rw, result_rw = dfa(
            random_walk,
            min_window=10,
            max_window=1000,
            num_windows=25,
            order=1
        )
        test_cases.append({
            'name': '',
            'data': random_walk,
            'alpha_measured': alpha_rw,
            'alpha_theory': alpha_theory_rw,
            'dimension': result_rw['dimension'],
            'result': result_rw,
            'description': '1.5'
        })
        print(f"   : ~{alpha_theory_rw:.2f}")
        print(f"   : {alpha_rw:.4f}")
        print(f"   : {result_rw['dimension']:.4f}")
        print(f"   R: {result_rw['r_squared']:.4f}")
    except Exception as e:
        print(f"   : {e}")
    
    # 4. FGN (H=0.3, )
    print("\n4. FGNH=0.3...")
    fgn_03 = generate_fgn(H=0.3, n=10000)
    alpha_theory_fgn03 = 0.3
    try:
        alpha_fgn03, result_fgn03 = dfa(
            fgn_03,
            min_window=10,
            max_window=1000,
            num_windows=25,
            order=1
        )
        test_cases.append({
            'name': 'FGN (H=0.3)',
            'data': fgn_03,
            'alpha_measured': alpha_fgn03,
            'alpha_theory': alpha_theory_fgn03,
            'dimension': result_fgn03['dimension'],
            'result': result_fgn03,
            'description': ''
        })
        print(f"   : {alpha_theory_fgn03:.2f}")
        print(f"   : {alpha_fgn03:.4f}")
        print(f"   : {result_fgn03['dimension']:.4f}")
        print(f"   R: {result_fgn03['r_squared']:.4f}")
    except Exception as e:
        print(f"   : {e}")
    
    # 5. FGN (H=0.7, )
    print("\n5. FGNH=0.7...")
    fgn_07 = generate_fgn(H=0.7, n=10000)
    alpha_theory_fgn07 = 0.7
    try:
        alpha_fgn07, result_fgn07 = dfa(
            fgn_07,
            min_window=10,
            max_window=1000,
            num_windows=25,
            order=1
        )
        test_cases.append({
            'name': 'FGN (H=0.7)',
            'data': fgn_07,
            'alpha_measured': alpha_fgn07,
            'alpha_theory': alpha_theory_fgn07,
            'dimension': result_fgn07['dimension'],
            'result': result_fgn07,
            'description': ''
        })
        print(f"   : {alpha_theory_fgn07:.2f}")
        print(f"   : {alpha_fgn07:.4f}")
        print(f"   : {result_fgn07['dimension']:.4f}")
        print(f"   R: {result_fgn07['r_squared']:.4f}")
    except Exception as e:
        print(f"   : {e}")
    
    # 
    if test_cases:
        n_cases = len(test_cases)
        fig = plt.figure(figsize=(16, 5.5*n_cases))
        
        for idx, case in enumerate(test_cases):
            # 
            ax1 = fig.add_subplot(n_cases, 3, idx*3 + 1)
            ax1.plot(case['data'][:2000], linewidth=0.6)  # 2000
            ax1.set_title(f"{case['name']}\n{case['description']}", fontsize=10)
            ax1.set_xlabel('', fontsize=9)
            ax1.set_ylabel('', fontsize=9)
            ax1.tick_params(labelsize=8)
            ax1.grid(True, alpha=0.3)
            
            # log-log
            result = case['result']
            ax2 = fig.add_subplot(n_cases, 3, idx*3 + 2)
            ax2.plot(result['log_windows'], result['log_fluctuations'], 
                    'o', label='', markersize=6, color='blue')
            
            # 
            fit_line = np.polyval(result['coeffs'], result['log_windows'])
            ax2.plot(result['log_windows'], fit_line, 'r-', 
                    linewidth=2, label=f' = {case["alpha_measured"]:.4f}')
            
            ax2.set_xlabel('log( n)', fontsize=9)
            ax2.set_ylabel('log( F(n))', fontsize=9)
            ax2.set_title(f'DFA\nR = {result["r_squared"]:.4f}', fontsize=10)
            ax2.tick_params(labelsize=8)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # 
            ax3 = fig.add_subplot(n_cases, 3, idx*3 + 3)
            
            # D
            params = [' ()', ' ()', 'D ()']
            values = [case['alpha_theory'], case['alpha_measured'], case['dimension']]
            colors = ['blue', 'orange', 'green']
            bars = ax3.bar(params, values, color=colors, alpha=0.7)
            
            # 
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=8)
            
            error_alpha = abs(case['alpha_measured'] - case['alpha_theory'])
            error_pct = error_alpha / case['alpha_theory'] * 100 if case['alpha_theory'] > 0 else 0
            
            ax3.set_ylabel('', fontsize=9)
            ax3.set_title(f'\n: {error_alpha:.3f} ({error_pct:.1f}%)', fontsize=10)
            ax3.set_ylim([0, max(values) * 1.3])
            ax3.tick_params(axis='x', labelsize=7, rotation=15)
            ax3.tick_params(axis='y', labelsize=8)
            ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(current_dir, "result_dfa.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n: {output_file}")
        plt.show()
    
    print("\n")
    print("\n")
    print("   DFA")
    print("   Hurst")
    print("   -  < 0.5: ")
    print("   -  = 0.5: ")
    print("   -  > 0.5: ")
    print("   -   1.0: 1/f")
    print("   -  > 1.0: ")
    print("\n   D = 2 - 1D")
    print("   DFA")
    print("\n   ")
    print("   - FGN: DFA   H")
    print("   - FBM: DFA   H + 1")
    print("   - FBM = cumsum(FGN), FGN = diff(FBM)")


if __name__ == '__main__':
    main()

