#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate Publication-Quality Box-Counting Validation Figures
============================================================

This script generates high-quality validation result figures suitable for 
journal publication.

Features:
- 300 DPI high resolution
- Professional color scheme
- Clear annotations and legends
- Journal-compliant format
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import os

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
plt.rcParams['mathtext.fontset'] = 'stix'  # Math formula font
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['grid.linewidth'] = 0.5
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['patch.linewidth'] = 1.0


def generate_figure3_combined_compact():
    """
    Generate Figure 3: Compact comprehensive figure (suitable for single or double column layout).
    
    Returns
    -------
    matplotlib.figure.Figure
        The generated figure.
    """
    print("\nGenerating Figure 3: Compact comprehensive figure...")
    
    from fracDimPy import (
        generate_sierpinski,
        generate_sierpinski_carpet,
        generate_menger_sponge,
        box_counting
    )
    
    # Define fractal configurations (only keep 3 types)
    fractals_config = [
        {
            'name': 'ST',
            'full_name': 'Sierpinski Triangle',
            'generator': lambda: generate_sierpinski(level=6, size=512),
            'data_type': 'image',
            'theoretical_D': np.log(3) / np.log(2),
            'is_3d': False
        },
        {
            'name': 'SC',
            'full_name': 'Sierpinski Carpet',
            'generator': lambda: generate_sierpinski_carpet(level=5, size=243),
            'data_type': 'image',
            'theoretical_D': np.log(8) / np.log(3),
            'is_3d': False
        },
        {
            'name': 'MS',
            'full_name': 'Menger Sponge',
            'generator': lambda: generate_menger_sponge(level=3, size=27),
            'data_type': 'porous',
            'theoretical_D': np.log(20) / np.log(3),
            'is_3d': True
        }
    ]
    
    # Create compact layout
    fig = plt.figure(figsize=(9, 4.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35,
                  height_ratios=[1.2, 1], top=0.94, bottom=0.12,
                  left=0.08, right=0.96)
    
    results = []
    colors = ['#E63946', '#F77F00', '#06A77D']
    
    for idx, config in enumerate(fractals_config):
        print(f"  Processing {config['full_name']}...")
        
        # Generate and calculate
        fractal_data = config['generator']()
        dimension, result_data = box_counting(fractal_data, data_type=config['data_type'])
        
        results.append({
            'name': config['name'],
            'dimension': dimension,
            'theoretical': config['theoretical_D'],
            'result_data': result_data
        })
        
        # Top row: Fractal images
        if config['is_3d']:
            # 3D display for Menger sponge
            from mpl_toolkits.mplot3d import Axes3D
            ax_img = fig.add_subplot(gs[0, idx], projection='3d')
            
            # Create color array
            colors_3d = np.empty(fractal_data.shape, dtype=object)
            colors_3d[fractal_data == 1] = '#06A77D'
            
            # Use voxels function to draw 3D voxels
            ax_img.voxels(fractal_data, facecolors=colors_3d, edgecolors='gray',
                         linewidth=0.1, alpha=0.85)
            
            ax_img.set_xlabel('X', fontsize=6, labelpad=1)
            ax_img.set_ylabel('Y', fontsize=6, labelpad=1)
            ax_img.set_zlabel('Z', fontsize=6, labelpad=1)
            ax_img.set_box_aspect([1,1,1])
            ax_img.view_init(elev=20, azim=45)
            ax_img.tick_params(labelsize=5)
            ax_img.grid(False)
            ax_img.xaxis.pane.fill = False
            ax_img.yaxis.pane.fill = False
            ax_img.zaxis.pane.fill = False
        else:
            # 2D display
            ax_img = fig.add_subplot(gs[0, idx])
            ax_img.imshow(fractal_data, cmap='binary', origin='upper')
            ax_img.axis('off')
        
        ax_img.set_title(f"({chr(97+idx)}) {config['name']}", 
                        fontsize=9, pad=5, fontweight='bold')
        
        # Bottom row: Log-log fit
        ax_fit = fig.add_subplot(gs[1, idx])
        
        data = result_data
        x = data['log_inv_epsilon']
        y = data['log_N']
        
        ax_fit.scatter(x, y, s=20, c=colors[idx], marker='o',
                      edgecolors='black', linewidths=0.3, alpha=0.8, zorder=3)
        
        coeffs = data['coefficients']
        x_fit = np.linspace(x.min(), x.max(), 100)
        y_fit = coeffs[0] * x_fit + coeffs[1]
        ax_fit.plot(x_fit, y_fit, 'k-', linewidth=1.5, alpha=0.7, zorder=2)
        
        # Labels
        if idx == 0:
            ax_fit.set_ylabel(r'$\log N$', fontsize=9)
        ax_fit.set_xlabel(r'$\log(1/\varepsilon)$', fontsize=8)
        
        # Display dimension
        error = abs(dimension - config['theoretical_D']) / config['theoretical_D'] * 100
        text = f"$D_{{calc}}={dimension:.3f}$\n$D_{{theo}}={config['theoretical_D']:.3f}$\nError: {error:.1f}%"
        ax_fit.text(0.05, 0.95, text, transform=ax_fit.transAxes,
                   fontsize=6.5, ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                           alpha=0.8, edgecolor='gray', linewidth=0.5))
        
        ax_fit.grid(True, alpha=0.2, linestyle='--', linewidth=0.3)
        ax_fit.tick_params(direction='in', which='both', labelsize=7)
    
    # Save figures (multiple formats)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save as PNG
    output_png = os.path.join(current_dir, "Figure_Compact_Overview.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Figure 3 (PNG) saved: {output_png}")
    
    # Save as PDF (vector graphics, recommended)
    output_pdf = os.path.join(current_dir, "Figure_Compact_Overview.pdf")
    plt.savefig(output_pdf, format='pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Figure 3 (PDF) saved: {output_pdf}")
    
    # Save as EPS (required by some journals)
    output_eps = os.path.join(current_dir, "Figure_Compact_Overview.eps")
    plt.savefig(output_eps, format='eps', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"✓ Figure 3 (EPS) saved: {output_eps}")
    
    return fig


def main():
    """
    Main function to generate publication-quality box-counting validation figures.
    """
    print("="*70)
    print("Generating Publication-Quality Box-Counting Validation Figures")
    print("="*70)
    print("\nFigure features:")
    print("  - Resolution: 300 DPI")
    print("  - Format: PNG (lossless compression)")
    print("  - Color scheme: Professional journal color scheme")
    print("  - Font: Arial + STIX math font")
    print("="*70)
    
    # Generate Figure 3
    fig3 = generate_figure3_combined_compact()
    
    print("\n" + "="*70)
    print("✓✓✓ Figure generation completed!")
    print("="*70)
    print("\nGenerated files:")
    print("  PNG format (for preview and presentation):")
    print("    - Figure_Compact_Overview.png")
    print("\n  PDF format (vector graphics, recommended for submission):")
    print("    - Figure_Compact_Overview.pdf")
    print("\n  EPS format (required by some traditional journals):")
    print("    - Figure_Compact_Overview.eps")
    print("\n✨ Vector graphics can be zoomed infinitely without distortion, perfect for journal publication!")
    print("="*70)
    
    plt.show()


if __name__ == '__main__':
    main()

