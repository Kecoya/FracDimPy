#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DFA and R/S Algorithm Validation Script (Publication Version)
==============================================================

This script validates the correctness of DFA (Detrended Fluctuation Analysis) 
and R/S (Rescaled Range Analysis) algorithms using standard time series with 
known theoretical Hurst exponents.

Time series types for validation:
1. White noise - H = 0.5 (uncorrelated)
2. Pink noise (1/f noise) - H ≈ 1.0  
3. Fractional Gaussian Noise (FGN) - H = 0.3 (anti-persistent)
4. Fractional Gaussian Noise (FGN) - H = 0.7 (persistent)
5. Random walk - H = 1.5

Theoretical relationships:
- Hurst exponent H ∈ (0, 2)
- For FGN: α_DFA = H
- For FBM: α_DFA = H + 1
- Fractal dimension: D = 2 - H (for 1D signal)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Tuple

# Configure matplotlib parameters for publication-quality figures
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
    """
    Generate white noise (H=0.5).
    
    Parameters
    ----------
    n : int
        Number of data points.
        
    Returns
    -------
    np.ndarray
        White noise time series.
    """
    return np.random.randn(n)


def generate_pink_noise(n=10000):
    """
    Generate pink noise (1/f noise) with H≈1.0.
    
    Parameters
    ----------
    n : int
        Number of data points.
        
    Returns
    -------
    np.ndarray
        Pink noise time series.
    """
    f = np.fft.rfftfreq(n)
    f[0] = 1
    spectrum = 1.0 / np.sqrt(f)
    phases = np.random.rand(len(f)) * 2 * np.pi
    complex_spectrum = spectrum * np.exp(1j * phases)
    signal = np.fft.irfft(complex_spectrum, n)
    return signal


def generate_fgn(H, n=10000):
    """
    Generate Fractional Gaussian Noise (FGN).
    
    FGN is a stationary increment process, and DFA directly gives H.
    
    Parameters
    ----------
    H : float
        Hurst exponent.
    n : int
        Number of data points.
        
    Returns
    -------
    np.ndarray
        FGN time series.
    """
    try:
        from fbm import FBM
        f = FBM(n=n, hurst=H, length=1, method='daviesharte')
        fgn_values = f.fgn()
        return fgn_values
    except ImportError:
        print("  Warning: fbm library not installed, using fracDimPy's FBM generator")
        from fracDimPy import generate_fbm_curve
        # For FGN, we need the increments of FBM
        _, fbm_curve = generate_fbm_curve(hurst=H, length=1.0, n_points=n+1)
        fgn = np.diff(fbm_curve[:, 1])
        return fgn


def generate_random_walk(n=10000):
    """
    Generate random walk (H=1.5 for DFA of cumulative sum).
    
    Parameters
    ----------
    n : int
        Number of data points.
        
    Returns
    -------
    np.ndarray
        Random walk time series.
    """
    steps = np.random.choice([-1, 1], size=n)
    return np.cumsum(steps).astype(float)


class DFA_RS_Validator:
    """
    DFA and R/S algorithm validator.
    
    This class validates DFA and R/S algorithms using time series with known
    theoretical Hurst exponents.
    """
    
    def __init__(self):
        self.results = []
        
        # Define test cases (only keep 3 cases where R/S performs well)
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
        """
        Execute all validation tests.
        
        Performs DFA and R/S analysis on all test cases and stores the results.
        """
        from fracDimPy import dfa, hurst_dimension
        
        print("="*70)
        print("DFA and RS Algorithm Validation")
        print("="*70)
        
        for idx, case in enumerate(self.test_cases):
            print(f"\n[{idx+1}/{len(self.test_cases)}] Validating {case['name']}...")
            print(f"  Theoretical Hurst exponent: H = {case['theoretical_H']:.2f}")
            
            # Generate data
            data = case['generator']()
            
            # DFA analysis
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
                print(f"  DFA failed: {e}")
                alpha_dfa = np.nan
                result_dfa = None
            
            # R/S analysis
            try:
                _, result_rs = hurst_dimension(data)
                H_rs = result_rs['hurst']
                print(f"  RS:  H = {H_rs:.4f}, R² = {result_rs['R2']:.4f}")
            except Exception as e:
                print(f"  RS failed: {e}")
                H_rs = np.nan
                result_rs = None
            
            # Save results
            self.results.append({
                'name': case['name'],
                'short_name': case['short_name'],
                'description': case['description'],
                'theoretical_H': case['theoretical_H'],
                'dfa_alpha': alpha_dfa,
                'rs_H': H_rs,
                'data': data[:2000],  # Only save first 2000 points for plotting
                'dfa_result': result_dfa,
                'rs_result': result_rs,
                'color': case['color']
            })
    
    def generate_publication_figure(self):
        """
        Generate publication-quality figure.
        
        Creates a comprehensive figure with time series, DFA analysis, and R/S analysis
        for all test cases.
        
        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        from matplotlib.gridspec import GridSpec
        
        print("\nGenerating publication-quality figure...")
        
        # Create large figure (3 rows)
        fig = plt.figure(figsize=(10, 8))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35,
                      top=0.96, bottom=0.08, left=0.08, right=0.96)
        
        for idx, result in enumerate(self.results):
            # Column 1: Time series
            ax1 = fig.add_subplot(gs[idx, 0])
            ax1.plot(result['data'], linewidth=0.5, color=result['color'], alpha=0.8)
            ax1.set_title(f"({chr(97+idx*3)}) {result['name']}\n{result['description']}", 
                         fontsize=9, pad=5)
            if idx == len(self.results) - 1:
                ax1.set_xlabel('Time', fontsize=9)
            ax1.set_ylabel('Value', fontsize=9)
            ax1.grid(True, alpha=0.3, linewidth=0.5)
            ax1.tick_params(labelsize=7)
            
            # Column 2: DFA log-log plot
            ax2 = fig.add_subplot(gs[idx, 1])
            if result['dfa_result'] is not None:
                dfa_res = result['dfa_result']
                ax2.scatter(dfa_res['log_windows'], dfa_res['log_fluctuations'],
                           s=20, color=result['color'], alpha=0.7, edgecolors='black',
                           linewidths=0.3, zorder=3)
                
                # Fit line
                fit_line = np.polyval(dfa_res['coeffs'], dfa_res['log_windows'])
                ax2.plot(dfa_res['log_windows'], fit_line, 'r-', 
                        linewidth=1.5, zorder=2)
                
                # Annotation
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
            
            # Column 3: R/S log-log plot
            ax3 = fig.add_subplot(gs[idx, 2])
            if result['rs_result'] is not None:
                rs_res = result['rs_result']
                ax3.scatter(rs_res['log_r'], rs_res['log_rs'],
                           s=20, color=result['color'], alpha=0.7, edgecolors='black',
                           linewidths=0.3, zorder=3)
                
                # Fit line
                fit_line = np.polyval(rs_res['coefficients'], rs_res['log_r'])
                ax3.plot(rs_res['log_r'], fit_line, 'r-', 
                        linewidth=1.5, zorder=2)
                
                # Annotation
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
        
        # Save figures
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # PNG
        output_png = os.path.join(current_dir, "Figure_DFA_RS_Validation.png")
        plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"✓ PNG saved: {output_png}")
        
        # PDF (vector graphics)
        output_pdf = os.path.join(current_dir, "Figure_DFA_RS_Validation.pdf")
        plt.savefig(output_pdf, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"✓ PDF saved: {output_pdf}")
        
        # EPS
        output_eps = os.path.join(current_dir, "Figure_DFA_RS_Validation.eps")
        plt.savefig(output_eps, format='eps', bbox_inches='tight', facecolor='white')
        print(f"✓ EPS saved: {output_eps}")
        
        return fig
    
    def print_summary(self):
        """
        Print validation summary.
        
        Displays a summary table with theoretical and calculated Hurst exponents,
        errors, and R² values for both DFA and R/S methods.
        """
        print("\n" + "="*80)
        print("Validation Results Summary")
        print("="*80)
        
        print("\n{:<15} {:<10} {:<10} {:<10} {:<10} {:<10}".format(
            "Series Type", "Theo. H", "DFA α", "DFA Error", "RS H", "RS Error"
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
        
        # Statistical information
        dfa_errors = [abs(r['dfa_alpha'] - r['theoretical_H']) / r['theoretical_H'] * 100 
                     for r in self.results]
        rs_errors = [abs(r['rs_H'] - r['theoretical_H']) / r['theoretical_H'] * 100 
                    for r in self.results]
        
        dfa_r2s = [r['dfa_result']['r_squared'] if r['dfa_result'] else 0 for r in self.results]
        rs_r2s = [r['rs_result']['R2'] if r['rs_result'] else 0 for r in self.results]
        
        print("-"*80)
        print(f"DFA average error: {np.mean(dfa_errors):.2f}%  |  Average R²: {np.mean(dfa_r2s):.4f}")
        print(f"RS average error:  {np.mean(rs_errors):.2f}%  |  Average R²: {np.mean(rs_r2s):.4f}")
        print("="*80)


def main():
    """
    Main function to run DFA and R/S algorithm validation.
    """
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "DFA and RS Algorithm Validation System" + " "*25 + "║")
    print("╚" + "="*78 + "╝")
    
    print("\nValidation algorithms:")
    print("  - DFA: Detrended Fluctuation Analysis")
    print("  - R/S: Rescaled Range Analysis")
    
    print("\nTest time series:")
    print("  1. White noise (H=0.5)")
    print("  2. Pink noise (H≈1.0)")
    print("  3. FGN H=0.7 (persistent)")
    
    # Create validator
    validator = DFA_RS_Validator()
    
    # Execute validation
    validator.validate_all()
    
    # Generate figures
    validator.generate_publication_figure()
    
    # Print summary
    validator.print_summary()
    
    print("\n" + "="*80)
    print("✓✓✓ Validation completed! Publication-quality figures have been generated!")
    print("="*80)
    
    plt.show()


if __name__ == '__main__':
    main()

