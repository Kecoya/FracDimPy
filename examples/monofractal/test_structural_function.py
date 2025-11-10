#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

=============


"""

import numpy as np
import os
from fracDimPy import structural_function
import matplotlib.pyplot as plt
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass

# 
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "structural_function_data.txt")

def main():
    print("="*60)
    print("")
    print("="*60)
    
    # 1. 
    print(f"\n1. : {data_file}")
    
    # XY
    data = np.loadtxt(data_file)
    
    # 
    if data.ndim == 1:
        # 1
        y_data = data
        x_interval = 1.0
        print(f"   : =1.0")
        print(f"   : {len(y_data)}")
        print(f"   : {y_data.min():.4f} ~ {y_data.max():.4f}")
    elif data.ndim == 2 and data.shape[1] >= 2:
        # XY
        x_data = data[:, 0]
        y_data = data[:, 1]
        x_interval = float(x_data[1] - x_data[0])
        print(f"   : X, Y")
        print(f"   : {len(y_data)}")
        print(f"   X: {x_interval:.6f}")
        print(f"   Y: {y_data.min():.4f} ~ {y_data.max():.4f}")
    else:
        raise ValueError(f"shape={data.shape}")
    
    # 2. 
    print("\n2. ...")
    D, result = structural_function(y_data, x_interval=x_interval)
    
    # 3. 
    print("\n3. :")
    print(f"    D: {D:.4f}")
    print(f"    R: {result['R2']:.4f}")
    
    # 4. 
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 
        axes[0].plot(y_data)
        axes[0].set_title('')
        axes[0].set_xlabel(f' (={x_interval:.6f})')
        axes[0].set_ylabel('')
        axes[0].grid(True)
        
        # 
        if 'tau_values' in result and 'S_values' in result:
            axes[1].loglog(result['tau_values'], result['S_values'], 'o', label='')
            # 
            if 'coefficients' in result:
                fit_line = np.exp(result['coefficients'][1]) * np.array(result['tau_values'])**result['coefficients'][0]
                axes[1].loglog(result['tau_values'], fit_line, 'r-', label=f' (={result["slope"]:.4f})')
            axes[1].set_xlabel(' ()')
            axes[1].set_ylabel('S() ()')
            axes[1].set_title(f' (D={D:.4f})')
            axes[1].legend()
            axes[1].grid(True, which='both', alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(current_dir, "result_structural_function.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n4. : {output_file}")
        plt.show()
        
    except ImportError:
        print("\n4. matplotlib")
    
    print("\n")


if __name__ == '__main__':
    main()

