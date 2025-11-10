#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 - 
================



6


- method=0: RDCCM - 
- method=1: DCCM  - 
- method=2: CCM   -  
- method=3: ICCM  - 

/
- method=5: SCCM  - /
- method=6: SDCCM - /


- 
- 
- 
- 
- /
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from fracDimPy import box_counting
from scipy.interpolate import interp2d

#  scienceplots 
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass

# 
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "box_counting_surface_data.csv")


def advise_mtepsilon(x, y):
    """
    
    
     BC-SURFACE\Coordinate_to_Matrix.py
    """
    xl = np.max(x) - np.min(x)
    yl = np.max(y) - np.min(y)
    num = len(x)
    return round(np.sqrt(xl * yl / num), 4)


def coordinate_to_matrix(x, y, z, epsilon=None):
    """
    XYZ
    
     BC-SURFACE\Coordinate_to_Matrix.py
    
    Parameters
    ----------
    x, y, z : np.ndarray
        
    epsilon : float, optional
        
    
    Returns
    -------
    matrix : np.ndarray
        
    epsilon : float
        
    """
    if epsilon is None:
        epsilon = advise_mtepsilon(x, y)
        print(f"   : {epsilon:.4f}")
    
    # 
    y_ = np.round((y - np.min(y)) / epsilon).astype(int)
    x_ = np.round((x - np.min(x)) / epsilon).astype(int)
    
    # 
    matrix = np.zeros((np.max(y_) + 1, np.max(x_) + 1))
    z = z - np.min(z)
    matrix[y_, x_] = z
    
    # 
    if matrix.shape[0] > 4 and matrix.shape[1] > 4:
        matrix = matrix[2:-2, 2:-2]
    
    return matrix, epsilon


def interpolate_surface(mt):
    """
    
    
     BC-SURFACE\Coordinate_to_Matrix.py
    """
    h, w = mt.shape[0], mt.shape[1]
    a, b = np.where(mt == 0)
    
    num = len(a)
    if num == 0:
        return mt
    
    print(f'   : {num}/{h*w} ({num/(h*w)*100:.2f}%)')
    
    # 0
    for i in range(len(a)):
        if a[i] == 0:
            a1, a2 = a[i], a[i] + 2
        elif a[i] == h - 1:
            a1, a2 = a[i] - 1, a[i] + 1
        else:
            a1, a2 = a[i] - 1, a[i] + 2
        
        if b[i] == 0:
            b1, b2 = b[i], b[i] + 2
        elif b[i] == w - 1:
            b1, b2 = b[i] - 1, b[i] + 1
        else:
            b1, b2 = b[i] - 1, b[i] + 2
        
        tempt = mt[a1:a2, b1:b2]
        c = np.sum(tempt) / np.sum(tempt != 0)
        if np.isnan(c):
            c = 0
        mt[a[i], b[i]] = c
    
    # 0
    r = max(int(min(h, w) / 10), 5)
    a, b = np.where(mt == 0)
    for i in range(len(a)):
        a1 = 0 if a[i] - r < 0 else a[i] - r
        a2 = h if a[i] + r + 1 > h else a[i] + r + 1
        b1 = 0 if b[i] - r < 0 else b[i] - r
        b2 = w if b[i] + r + 1 > w else b[i] + r + 1
        tempt = mt[a1:a2, b1:b2]
        c = np.sum(tempt) / np.sum(tempt != 0)
        if np.isnan(c):
            c = 0
        mt[a[i], b[i]] = c
    
    # 0
    mt[mt == 0] = np.mean(mt)
    
    return mt

def main():
    print("="*60)
    print(" - ")
    print("="*60)
    
    # ==========  ==========
    # 0(RDCCM), 1(DCCM), 2(CCM), 3(ICCM), 5(SCCM), 6(SDCCM)
    #  method=2 (CCM)
    CALCULATION_METHOD = 2
    
    method_info = {
        0: "RDCCM",
        1: "DCCM",
        2: "CCM",
        3: "ICCM",
        5: "SCCM/",
        6: "SDCCM/"
    }
    
    print(f"\n>>> : method={CALCULATION_METHOD}")
    print(f">>> {method_info.get(CALCULATION_METHOD, '')}")
    # =============================
    
    # 1. 
    print(f"\n1. : {data_file}")
    df = pd.read_csv(data_file, header=None)
    print(f"   : {df.shape}")
    
    # XYZ
    mt_epsilon_min = None
    
    if df.shape[1] == 3:
        # XYZ
        print("   : XYZ")
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        z = df.iloc[:, 2].values
        
        print(f"   : {len(x)}")
        print(f"   X: {x.min():.4f} ~ {x.max():.4f}")
        print(f"   Y: {y.min():.4f} ~ {y.max():.4f}")
        print(f"   Z: {z.min():.4f} ~ {z.max():.4f}")
        
        # 
        print("   ...")
        surface, mt_epsilon_min = coordinate_to_matrix(x, y, z)
        print(f"   : {surface.shape}")
        
        # 
        if np.any(surface == 0):
            print("   ...")
            surface = interpolate_surface(surface)
            print("   ")
        
        print(f"   : {np.min(surface):.4f} ~ {np.max(surface):.4f}")
        
    else:
        # 
        print("   : ")
        surface = df.values
        print(f"   : {surface.shape}")
        print(f"   : {np.nanmin(surface):.4f} ~ {np.nanmax(surface):.4f}")
        
        # NaN
        if np.any(np.isnan(surface)):
            print("   NaN...")
            surface = np.nan_to_num(surface, nan=0.0)
            if np.any(surface == 0):
                surface = interpolate_surface(surface)
            print("   ")
    
    # 2. 
    print("\n2. ...")
    if mt_epsilon_min is not None:
        # XYZ
        D, result = box_counting(surface, data_type='surface', 
                                method=CALCULATION_METHOD, 
                                mt_epsilon_min=mt_epsilon_min)
    else:
        # 
        D, result = box_counting(surface, data_type='surface', 
                                method=CALCULATION_METHOD)
    
    # 3. 
    print("\n3. :")
    print(f"    D: {D:.4f}")
    print(f"    R: {result['R2']:.4f}")
    
    # 4. 
    print("\n4. ...")
    
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(15, 10))
    
    # 3D
    ax1 = fig.add_subplot(221, projection='3d')
    ny, nx = surface.shape
    X, Y = np.meshgrid(range(nx), range(ny))
    # 
    step = max(1, min(nx, ny) // 50)
    surf = ax1.plot_surface(X[::step, ::step], Y[::step, ::step], 
                            surface[::step, ::step], cmap='terrain', alpha=0.8)
    ax1.set_title('3D', fontsize=12)
    ax1.set_xlabel('X', fontsize=10)
    ax1.set_ylabel('Y', fontsize=10)
    ax1.set_zlabel('Z ()', fontsize=10)
    
    # 2D
    ax2 = fig.add_subplot(222)
    im = ax2.imshow(surface, cmap='terrain', aspect='auto')
    ax2.set_title('', fontsize=12)
    ax2.set_xlabel('X', fontsize=10)
    ax2.set_ylabel('Y', fontsize=10)
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('', fontsize=10)
    
    # 
    if 'epsilon_values' in result and 'N_values' in result:
        ax3 = fig.add_subplot(223)
        ax3.loglog(result['epsilon_values'], result['N_values'], 'mo', 
                  markersize=6, label='')
        
        # 
        if 'coefficients' in result:
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = np.exp(b) * np.array(result['epsilon_values'])**(-a)
            ax3.loglog(result['epsilon_values'], fit_line, 'r-', linewidth=2,
                      label=f' (D={D:.4f})')
        
        ax3.set_xlabel(' ()', fontsize=12)
        ax3.set_ylabel('N() ()', fontsize=12)
        ax3.set_title('', fontsize=12)
        ax3.legend()
        ax3.grid(True, alpha=0.3, which='both')
    
    # 
    if 'log_inv_epsilon' in result and 'log_N' in result:
        ax4 = fig.add_subplot(224)
        
        # 
        ax4.plot(result['log_inv_epsilon'], result['log_N'], 'r^', 
                label='', markerfacecolor='white', markersize=8, markeredgewidth=1.5)
        
        # 
        if 'coefficients' in result:
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = a * result['log_inv_epsilon'] + b
            ax4.plot(result['log_inv_epsilon'], fit_line, 'c-', linewidth=2, label='')
            
            # 
            x_min, x_max = np.min(result['log_inv_epsilon']), np.max(result['log_inv_epsilon'])
            y_min, y_max = np.min(result['log_N']), np.max(result['log_N'])
            
            equation_text = (
                r'$\ln(N) = {:.4f} \ln(\frac{{1}}{{r}}) + {:.4f}$'.format(a, b) + '\n' +
                r'$R^2 = {:.6f} \qquad D = {:.4f}$'.format(result["R2"], a)
            )
            
            ax4.text(
                x_min + 0.05 * (x_max - x_min),
                y_max - 0.15 * (y_max - y_min),
                equation_text,
                fontsize=11,
                bbox={'facecolor': 'blue', 'alpha': 0.2}
            )
        
        # 
        ax4.set_xlabel(r'$ \ln ( \frac{1}{\epsilon} ) $', fontsize=12)
        ax4.set_ylabel(r'$ \ln ( N_{\epsilon} )$', fontsize=12)
        ax4.set_title('', fontsize=12)
        ax4.legend(loc='lower right')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(current_dir, "result_box_counting_surface.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   : {output_file}")
    plt.show()
    
    print("\n")


if __name__ == '__main__':
    main()
