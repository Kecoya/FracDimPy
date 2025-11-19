#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detrended Fluctuation Analysis (DFA) Test Example
=================================================

This example demonstrates how to use the Detrended Fluctuation Analysis (DFA)
method to calculate the scaling exponent α and fractal dimension of time series.

DFA is a method for determining the statistical self-affinity of a signal.
It is useful for analyzing time series that may be affected by trends.

Theoretical Background:
- For FGN (Fractional Gaussian Noise): DFA α = H
- For FBM (Fractional Brownian Motion): DFA α = H + 1
- Fractal dimension D = 2 - α (for 1D signals)
"""

import numpy as np
import os
import matplotlib.pyplot as plt

# Try to use scienceplots style
try:
    import scienceplots
    plt.style.use(['science','no-latex'])
except ImportError:
    pass

# Set font: Times New Roman for English, Microsoft YaHei for Chinese
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue


def generate_fgn(H, n=10000):
    """Generate Fractional Gaussian Noise (FGN) from FBM
    
    Note:
    - For FGN: DFA α = H
    - For FBM: DFA α = H + 1
    """
    try:
        from fbm import FBM
        f = FBM(n=n, hurst=H, length=1, method='daviesharte')
        # Generate FGN using fgn() method
        fgn_values = f.fgn()
        return fgn_values
    except ImportError:
        print("Warning: fbm library not installed, using fracDimPy FBM generator")
        from fracDimPy import generate_fbm_curve
        D = 2 - H
        fbm_curve, _ = generate_fbm_curve(dimension=D, length=n+1)
        # FGN = diff(FBM)
        fgn = np.diff(fbm_curve)
        return fgn


def generate_white_noise(n=10000):
    """Generate white noise (uncorrelated random signal)"""
    return np.random.randn(n)


def generate_pink_noise(n=10000):
    """Generate pink noise (1/f noise)"""
    # Generate frequency array
    f = np.fft.rfftfreq(n)
    f[0] = 1  # Avoid division by zero
    
    # Create 1/f power spectrum
    spectrum = 1.0 / np.sqrt(f)
    
    # Add random phases
    phases = np.random.rand(len(f)) * 2 * np.pi
    complex_spectrum = spectrum * np.exp(1j * phases)
    
    # Inverse FFT to get time domain signal
    signal = np.fft.irfft(complex_spectrum, n)
    
    return signal


def generate_random_walk(n=10000):
    """Generate random walk (cumulative sum of random steps)"""
    steps = np.random.choice([-1, 1], size=n)
    return np.cumsum(steps)


def main():
    print("="*60)
    print("Detrended Fluctuation Analysis (DFA) Test Example")
    print("="*60)
    
    from fracDimPy import dfa
    
    # Initialize test cases list
    test_cases = []
    
    # 1. White noise (α = 0.5)
    print("\n1. Testing white noise...")
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
            'name': 'White Noise',
            'data': white_noise,
            'alpha_measured': alpha_white,
            'alpha_theory': alpha_theory_white,
            'dimension': result_white['dimension'],
            'result': result_white,
            'description': 'α=0.5'
        })
        print(f"   Theoretical α: {alpha_theory_white:.2f}")
        print(f"   Measured α: {alpha_white:.4f}")
        print(f"   Fractal dimension: {result_white['dimension']:.4f}")
        print(f"   R²: {result_white['r_squared']:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 2. Pink noise / 1/f noise (α ≈ 1.0)
    print("\n2. Testing pink noise (1/f)...")
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
            'name': 'Pink Noise (1/f)',
            'data': pink_noise,
            'alpha_measured': alpha_pink,
            'alpha_theory': alpha_theory_pink,
            'dimension': result_pink['dimension'],
            'result': result_pink,
            'description': 'α≈1.0'
        })
        print(f"   Theoretical α: ~{alpha_theory_pink:.2f}")
        print(f"   Measured α: {alpha_pink:.4f}")
        print(f"   Fractal dimension: {result_pink['dimension']:.4f}")
        print(f"   R²: {result_pink['r_squared']:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. Random walk (α ≈ 1.5)
    print("\n3. Testing random walk...")
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
            'name': 'Random Walk',
            'data': random_walk,
            'alpha_measured': alpha_rw,
            'alpha_theory': alpha_theory_rw,
            'dimension': result_rw['dimension'],
            'result': result_rw,
            'description': 'α≈1.5'
        })
        print(f"   Theoretical α: ~{alpha_theory_rw:.2f}")
        print(f"   Measured α: {alpha_rw:.4f}")
        print(f"   Fractal dimension: {result_rw['dimension']:.4f}")
        print(f"   R²: {result_rw['r_squared']:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 4. FGN (H=0.3, anti-persistent)
    print("\n4. Testing FGN with H=0.3...")
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
            'description': 'Anti-persistent'
        })
        print(f"   Theoretical α: {alpha_theory_fgn03:.2f}")
        print(f"   Measured α: {alpha_fgn03:.4f}")
        print(f"   Fractal dimension: {result_fgn03['dimension']:.4f}")
        print(f"   R²: {result_fgn03['r_squared']:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 5. FGN (H=0.7, persistent)
    print("\n5. Testing FGN with H=0.7...")
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
            'description': 'Persistent'
        })
        print(f"   Theoretical α: {alpha_theory_fgn07:.2f}")
        print(f"   Measured α: {alpha_fgn07:.4f}")
        print(f"   Fractal dimension: {result_fgn07['dimension']:.4f}")
        print(f"   R²: {result_fgn07['r_squared']:.4f}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Visualize results
    if test_cases:
        n_cases = len(test_cases)
        fig = plt.figure(figsize=(16, 5.5*n_cases))
        
        for idx, case in enumerate(test_cases):
            # Left plot: Time series
            ax1 = fig.add_subplot(n_cases, 3, idx*3 + 1)
            ax1.plot(case['data'][:2000], linewidth=0.6)  # Show first 2000 points
            ax1.set_title(f"{case['name']}\n{case['description']}", fontsize=10)
            ax1.set_xlabel('Time', fontsize=9)
            ax1.set_ylabel('Value', fontsize=9)
            ax1.tick_params(labelsize=8)
            ax1.grid(True, alpha=0.3)
            
            # Middle plot: DFA log-log plot
            result = case['result']
            ax2 = fig.add_subplot(n_cases, 3, idx*3 + 2)
            ax2.plot(result['log_windows'], result['log_fluctuations'], 
                    'o', label='Data points', markersize=6, color='blue')
            
            # Draw fitting line
            fit_line = np.polyval(result['coeffs'], result['log_windows'])
            ax2.plot(result['log_windows'], fit_line, 'r-', 
                    linewidth=2, label=f'α = {case["alpha_measured"]:.4f}')
            
            ax2.set_xlabel('log(n)', fontsize=9)
            ax2.set_ylabel('log(F(n))', fontsize=9)
            ax2.set_title(f'DFA Analysis\nR² = {result["r_squared"]:.4f}', fontsize=10)
            ax2.tick_params(labelsize=8)
            ax2.legend(fontsize=8)
            ax2.grid(True, alpha=0.3)
            
            # Right plot: Comparison bar chart
            ax3 = fig.add_subplot(n_cases, 3, idx*3 + 3)
            
            # Display theoretical α, measured α, and dimension D
            params = ['α (theory)', 'α (measured)', 'D (dimension)']
            values = [case['alpha_theory'], case['alpha_measured'], case['dimension']]
            colors = ['blue', 'orange', 'green']
            bars = ax3.bar(params, values, color=colors, alpha=0.7)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.3f}',
                        ha='center', va='bottom', fontsize=8)
            
            error_alpha = abs(case['alpha_measured'] - case['alpha_theory'])
            error_pct = error_alpha / case['alpha_theory'] * 100 if case['alpha_theory'] > 0 else 0
            
            ax3.set_ylabel('Value', fontsize=9)
            ax3.set_title(f'Comparison\nError: {error_alpha:.3f} ({error_pct:.1f}%)', fontsize=10)
            ax3.set_ylim([0, max(values) * 1.3])
            ax3.tick_params(axis='x', labelsize=7, rotation=15)
            ax3.tick_params(axis='y', labelsize=8)
            ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout(pad=2.0, h_pad=3.0, w_pad=2.0)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        output_file = os.path.join(current_dir, "result_dfa.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: {output_file}")
        plt.show()
    
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    print("   DFA (Detrended Fluctuation Analysis)")
    print("   Scaling exponent α is related to Hurst exponent H:")
    print("   - α < 0.5: Anti-persistent (mean-reverting)")
    print("   - α = 0.5: Random walk (no memory)")
    print("   - α > 0.5: Persistent (trend-following)")
    print("   - α ≈ 1.0: 1/f noise (pink noise)")
    print("   - α > 1.0: Non-stationary processes")
    print("\n   Fractal dimension: D = 2 - α (for 1D signals)")
    print("   DFA is robust to trends in the data")
    print("\n   Important relationships:")
    print("   - FGN: DFA α = H")
    print("   - FBM: DFA α = H + 1")
    print("   - FBM = cumsum(FGN), FGN = diff(FBM)")


if __name__ == '__main__':
    main()

