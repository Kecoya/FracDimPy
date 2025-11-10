#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Takagi Surface Box-Counting Fractal Dimension Analysis
=======================================================
Calculate the fractal dimension of Takagi surfaces using the box-counting method.
Generate publication-quality vector graphics.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from fracDimPy import generate_takagi_surface, box_counting

# Use SciencePlots style
try:
    import scienceplots 
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass

plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))


def analyze_takagi_surface(dimension=2.5, level=12, size=256, method=2):
    """
    Analyze a single Takagi surface
    
    Parameters
    ----------
    dimension : float
        Theoretical fractal dimension (2, 3)
    level : int
        Iteration level
    size : int
        Surface size
    method : int
        Box-counting method (0-6)
        
    Returns
    -------
    surface : np.ndarray
        Generated Takagi surface
    D : float
        Calculated fractal dimension
    result : dict
        Analysis results
    """
    print("="*70)
    print(f"Analyzing Takagi surface (theoretical dimension D={dimension})")
    print("="*70)
    
    # 1. Generate Takagi surface
    print(f"\n1. Generating Takagi surface...")
    print(f"   Parameters: size={size}x{size}, iteration level={level}")
    surface = generate_takagi_surface(dimension=dimension, level=level, size=size)
    
    b = 2 ** dimension / 8
    print(f"   Theoretical fractal dimension: D = {dimension}")
    print(f"   Parameter b = {b:.4f}")
    print(f"   Height range: {surface.min():.4f} ~ {surface.max():.4f}")
    print(f"   Standard deviation: {surface.std():.4f}")
    
    # 2. Calculate fractal dimension
    print(f"\n2. Calculating fractal dimension using Box-Counting method...")
    print(f"   Calculation method: method={method}")
    
    D, result = box_counting(surface, data_type='surface', method=method)
    
    # 3. Display results
    print(f"\n3. Calculation results:")
    print(f"   Measured fractal dimension: D = {D:.4f}")
    print(f"   Goodness of fit: R² = {result['R2']:.6f}")
    print(f"   Relative error: {abs(D - dimension) / dimension * 100:.2f}%")
    
    return surface, D, result


def create_publication_figure(surface, dimension, D, result):
    """
    Create publication-quality vector graphics.
    
    Parameters
    ----------
    surface : np.ndarray
        Takagi surface data.
    dimension : float
        Theoretical fractal dimension.
    D : float
        Measured fractal dimension.
    result : dict
        Box-counting analysis results.
    """
    print("\n4. Generating publication-quality vector graphics...")
    
    # Create 2x2 layout
    fig = plt.figure(figsize=(16, 14))
    
    ny, nx = surface.shape
    X, Y = np.meshgrid(range(nx), range(ny))
    
    # ========== 1. 3D surface view - top left ==========
    ax1 = fig.add_subplot(221, projection='3d')
    
    # Downsample to improve rendering speed
    step = max(1, min(nx, ny) // 50)
    surf = ax1.plot_surface(X[::step, ::step], Y[::step, ::step], 
                            surface[::step, ::step], 
                            cmap='terrain', alpha=0.9, 
                            linewidth=0, antialiased=True, 
                            rcount=50, ccount=50)
    
    ax1.set_title('3D Surface View', fontsize=15, fontweight='bold', pad=15)
    ax1.set_xlabel('X', fontsize=13, labelpad=10)
    ax1.set_ylabel('Y', fontsize=13, labelpad=10)
    ax1.set_zlabel('Height', fontsize=13, labelpad=10)
    ax1.view_init(elev=25, azim=45)
    ax1.tick_params(labelsize=11)
    
    # Add information box
    info_text = f'Theoretical D = {dimension}\nMeasured D = {D:.4f}'
    ax1.text2D(0.02, 0.98, info_text, transform=ax1.transAxes,
              fontsize=11, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # ========== 2. 2D heatmap view - top right ==========
    ax2 = fig.add_subplot(222)
    
    im = ax2.imshow(surface, cmap='terrain', aspect='auto', origin='lower')
    ax2.set_title('2D Heatmap View', fontsize=15, fontweight='bold')
    ax2.set_xlabel('X', fontsize=13)
    ax2.set_ylabel('Y', fontsize=13)
    ax2.tick_params(labelsize=11)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('Height', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Add statistical information
    stats_text = f'Min: {surface.min():.4f}\nMax: {surface.max():.4f}\nStd: {surface.std():.4f}'
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========== 3. Log-log plot - bottom left ==========
    if 'epsilon_values' in result and 'N_values' in result:
        ax3 = fig.add_subplot(223)
        
        # Plot data points
        ax3.loglog(result['epsilon_values'], result['N_values'], 'o', 
                  color='steelblue', markersize=8, markeredgewidth=1.5,
                  markerfacecolor='white', label='Data points', zorder=3)
        
        # Plot fit line
        if 'coefficients' in result:
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = np.exp(b) * np.array(result['epsilon_values'])**(-a)
            ax3.loglog(result['epsilon_values'], fit_line, 'r-', 
                      linewidth=2.5, label=f'Fit line (D={D:.4f})', zorder=2)
        
        ax3.set_xlabel(r'Box size $\epsilon$', fontsize=14)
        ax3.set_ylabel(r'Number of boxes $N(\epsilon)$', fontsize=14)
        ax3.set_title('Log-Log Plot', fontsize=15, fontweight='bold')
        ax3.legend(fontsize=12, loc='best')
        ax3.grid(True, alpha=0.3, which='both', linestyle='--', linewidth=0.5)
        ax3.tick_params(labelsize=11)
    
    # ========== 4. Linear fit plot - bottom right ==========
    if 'log_inv_epsilon' in result and 'log_N' in result:
        ax4 = fig.add_subplot(224)
        
        # Plot data points
        ax4.plot(result['log_inv_epsilon'], result['log_N'], '^', 
                color='crimson', markersize=9, markeredgewidth=1.5,
                markerfacecolor='white', label='Data points', zorder=3)
        
        # Plot fit line
        if 'coefficients' in result:
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = a * result['log_inv_epsilon'] + b
            ax4.plot(result['log_inv_epsilon'], fit_line, '-', 
                    color='darkorange', linewidth=2.5, label='Linear fit', zorder=2)
            
            # Add fit equation
            x_min, x_max = np.min(result['log_inv_epsilon']), np.max(result['log_inv_epsilon'])
            y_min, y_max = np.min(result['log_N']), np.max(result['log_N'])
            
            equation_text = (
                r'$\ln(N) = {:.4f} \ln(1/\epsilon) + {:.4f}$'.format(a, b) + '\n' +
                r'$R^2 = {:.6f}$'.format(result["R2"]) + '\n' +
                r'$D = {:.4f}$'.format(D)
            )
            
            ax4.text(0.05, 0.95, equation_text,
                    transform=ax4.transAxes,
                    fontsize=12, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', 
                             alpha=0.8, edgecolor='blue', linewidth=1.5))
        
        ax4.set_xlabel(r'$\ln(1/\epsilon)$', fontsize=14)
        ax4.set_ylabel(r'$\ln(N(\epsilon))$', fontsize=14)
        ax4.set_title('Linear Fit', fontsize=15, fontweight='bold')
        ax4.legend(fontsize=12, loc='lower right')
        ax4.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax4.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    # Save in multiple formats
    for ext in ['eps', 'pdf', 'png']:
        output_file = os.path.join(current_dir, f"takagi_surface_boxcounting.{ext}")
        if ext == 'eps':
            # EPS does not support transparency
            fig.savefig(output_file, format='eps', dpi=300, bbox_inches='tight')
        else:
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   Saved: takagi_surface_boxcounting.{ext}")
    
    plt.close(fig)


def main():
    """
    Main function to perform Takagi surface box-counting fractal dimension analysis.
    """
    print("="*70)
    print("Takagi Surface Box-Counting Fractal Dimension Analysis")
    print("="*70)
    
    # ==========  Parameter settings  ==========
    THEORETICAL_DIMENSION = 2.1  # Theoretical fractal dimension (2, 3)
    ITERATION_LEVEL = 12         # Iteration level
    SURFACE_SIZE = 256           # Surface size
    CALCULATION_METHOD = 2       # Box-counting method (SCCM - best method)
    # ===========================================
    
    print(f"\n>>> Parameter settings:")
    print(f">>> Theoretical fractal dimension: D = {THEORETICAL_DIMENSION}")
    print(f">>> Surface size: {SURFACE_SIZE} × {SURFACE_SIZE}")
    print(f">>> Iteration level: {ITERATION_LEVEL}")
    print(f">>> Calculation method: method={CALCULATION_METHOD} (SCCM - Simplified Cubic Cover)")
    
    # Analyze Takagi surface
    surface, D, result = analyze_takagi_surface(
        dimension=THEORETICAL_DIMENSION,
        level=ITERATION_LEVEL,
        size=SURFACE_SIZE,
        method=CALCULATION_METHOD
    )
    
    # Generate publication-quality vector graphics
    create_publication_figure(surface, THEORETICAL_DIMENSION, D, result)
    
    print("\n" + "="*70)
    print("Analysis completed!")
    print("="*70)
    print(f"Theoretical fractal dimension: D = {THEORETICAL_DIMENSION}")
    print(f"Measured fractal dimension: D = {D:.4f}")
    print(f"Absolute error: ΔD = {abs(D - THEORETICAL_DIMENSION):.4f}")
    print(f"Relative error: {abs(D - THEORETICAL_DIMENSION) / THEORETICAL_DIMENSION * 100:.2f}%")
    print(f"Goodness of fit: R² = {result['R2']:.6f}")
    print("="*70)
    print("\nGenerated files:")
    print("  - takagi_surface_boxcounting.eps (vector graphics)")
    print("  - takagi_surface_boxcounting.pdf (vector graphics)")
    print("  - takagi_surface_boxcounting.png (raster graphics)")
    print("\nPublication-quality vector graphics have been generated!")


if __name__ == '__main__':
    main()

