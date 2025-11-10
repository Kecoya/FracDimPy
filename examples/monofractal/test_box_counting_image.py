#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 - 
================
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from fracDimPy import box_counting

#  scienceplots 
try:
    import scienceplots
    # plt.style.use(['ieee'])  # 
except ImportError:
    pass

# Microsoft YaHeiTimes New Roman
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 

# 
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "box_counting_image_data.png")

def main():
    print("="*60)
    print(" - ")
    print("="*60)
    
    # 1. 
    print(f"\n1. : {data_file}")
    try:
        from PIL import Image
        img = Image.open(data_file).convert('RGB')  # RGB
        img_array = np.array(img)
        
        # BC-Image
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        img_gray = 255 - (0.2989 * r + 0.5870 * g + 0.1140 * b)  # 
        
        print(f"   : {img_gray.shape}")
        print(f"   : {img_gray.min():.1f} ~ {img_gray.max():.1f}")
        
        # 0255BC-Image
        threshold = np.mean(img_gray)
        binary_img = np.where(img_gray > threshold, 255, 0).astype(np.uint8)
        print(f"   : {threshold:.2f}")
        print(f"   : {np.sum(binary_img == 255)}")
        
        # 
        img_data = np.array(Image.open(data_file).convert('L'))
        
    except Exception as e:
        print(f"   : {e}")
        print("   ...")
        binary_img = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
        img_data = binary_img
    
    # 2. 
    print("\n2. ...")
    D, result = box_counting(binary_img, data_type='image')
    
    # 3. 
    print("\n3. :")
    print(f"    D: {D:.4f}")
    print(f"    R: {result['R2']:.4f}")
    
    # 4. 
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 
    axes[0, 0].imshow(img_data, cmap='gray')
    axes[0, 0].set_title('')
    axes[0, 0].axis('off')
    
    # 
    axes[0, 1].imshow(binary_img, cmap='binary')
    axes[0, 1].set_title('')
    axes[0, 1].axis('off')
    
    # log-log
    if 'epsilon_values' in result and 'N_values' in result:
        axes[1, 0].loglog(result['epsilon_values'], result['N_values'], 'o', markersize=6, label='')
        # 
        if 'coefficients' in result:
            # : log(N) = a*log(1/) + b  =>  N = exp(b) * ^(-a)
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = np.exp(b) * np.array(result['epsilon_values'])**(-a)
            axes[1, 0].loglog(result['epsilon_values'], fit_line, 'r-', linewidth=2,
                            label=f' (D={D:.4f})')
            
            # 
            C = np.exp(b)
            equation_text = f'$N(\\varepsilon) = {C:.2e} \\cdot \\varepsilon^{{-{a:.4f}}}$\n$R^2 = {result["R2"]:.4f}$'
            axes[1, 0].text(0.05, 0.95, equation_text, transform=axes[1, 0].transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        axes[1, 0].set_xlabel(' ()')
        axes[1, 0].set_ylabel('N() ()')
        axes[1, 0].set_title(' ()')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 
    if 'log_inv_epsilon' in result and 'log_N' in result:
        axes[1, 1].plot(result['log_inv_epsilon'], result['log_N'], 'o', markersize=6, label='')
        if 'coefficients' in result:
            # 
            a, b = result['coefficients'][0], result['coefficients'][1]
            fit_line = a * result['log_inv_epsilon'] + b
            axes[1, 1].plot(result['log_inv_epsilon'], fit_line, 'r-', linewidth=2,
                          label=f' (={a:.4f})')
            
            # 
            equation_text = f'$\\log(N) = {a:.4f} \\cdot \\log(1/\\varepsilon) + {b:.4f}$\n$R^2 = {result["R2"]:.4f}$\n$D = {a:.4f}$'
            axes[1, 1].text(0.05, 0.95, equation_text, transform=axes[1, 1].transAxes,
                          fontsize=10, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        axes[1, 1].set_xlabel('log(1/)')
        axes[1, 1].set_ylabel('log(N)')
        axes[1, 1].set_title('')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_file = os.path.join(current_dir, "result_box_counting_image.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n4. : {output_file}")
    plt.show()
    
    print("\n")


if __name__ == '__main__':
    main()

