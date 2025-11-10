#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

=========


"""

import numpy as np
import os
from fracDimPy import sandbox_method
import matplotlib.pyplot as plt
# 
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 
# 
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "sandbox_data.png")

def main():
    print("="*60)
    print("")
    print("="*60)
    
    # 1. 
    print(f"\n1. : {data_file}")
    
    # 2. 
    print("\n2. ...")
    D, result = sandbox_method(data_file)
    
    # 3. 
    print("\n3. :")
    print(f"    D: {D:.4f}")
    print(f"    R: {result['R2']:.4f}")
    
    # 4. 
    try:
        import matplotlib.pyplot as plt
        from PIL import Image
        
        img = Image.open(data_file).convert('L')
        img_array = np.array(img)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 
        axes[0].imshow(img_array, cmap='gray')
        axes[0].set_title('')
        axes[0].axis('off')
        
        # Sandbox
        if 'r_values' in result and 'N_values' in result:
            axes[1].loglog(result['r_values'], result['N_values'], 'o', label='')
            # 
            if 'coefficients' in result:
                fit_line = np.exp(result['coefficients'][1]) * np.array(result['r_values'])**result['coefficients'][0]
                axes[1].loglog(result['r_values'], fit_line, 'r-', label=f' (D={D:.4f})')
            axes[1].set_xlabel(' r ()')
            axes[1].set_ylabel(' N(r) ()')
            axes[1].set_title(' ()')
            axes[1].legend()
            axes[1].grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(current_dir, "result_sandbox.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n4. : {output_file}")
        plt.show()
        
    except ImportError:
        print("\n4. matplotlibPIL")
    
    print("\n")


if __name__ == '__main__':
    main()

