#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cantor集生成示例
===============

本示例演示如何使用fracDimPy生成经典的Cantor集（康托尔集）。
Cantor集是最早被研究的分形之一，通过递归地移除线段中间部分而生成，
具有自相似性和处处不连续的特点。

主要功能：
- 生成指定层级的Cantor集
- 可视化多个层级的Cantor集
- 将结果保存为CSV文件
- 展示分形的逐层构造过程

理论背景：
- Cantor集的分形维数: D = log(2)/log(3) ≈ 0.6309
- 每次迭代移除中间1/3，保留两端各1/3
- 经过无穷多次迭代后形成的集合是不可数的
"""

import numpy as np
import os

import matplotlib.pyplot as plt
#  scienceplots 
try:
    import scienceplots
    plt.style.use(['science','no-latex'])  # 
except ImportError:
    pass
# Microsoft YaHeiTimes New Roman
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # 

def main():
    print("="*60)
    print("Cantor集生成示例")
    print("="*60)
    
    from fracDimPy import generate_cantor_set
    
    # 1. 生成Cantor集
    print("\n1. 正在生成Cantor集...")
    level = 6  # 迭代层级
    
    cantor = generate_cantor_set(level=level)
    
    print(f"   迭代层级: {level}")
    print(f"   数组长度: {len(cantor)}")
    print(f"   理论长度: {3**level}")
    print(f"   保留点数: {np.sum(cantor)}")
    print(f"   理论分形维数: {np.log(2)/np.log(3):.4f} (≈ 0.6309)")
    
    # 2. 保存为CSV文件
    print("\n2. 正在将Cantor集保存为CSV...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(current_dir, f"cantor_set_level{level}.csv")
    np.savetxt(csv_file, cantor.astype(int), fmt='%d', delimiter=',', header='value', comments='')
    print(f"   文件已保存: {csv_file}")
    print(f"   数据说明: 使用0和1表示点的存在与否, 共{len(cantor)}个点")
    
    # 3. 可视化Cantor集
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        
        # 上图: 显示完整的Cantor集（最终层级）
        axes[0].imshow([cantor], cmap='binary', aspect='auto')
        axes[0].set_title(f'Cantor集完整视图 (迭代层级={level})')
        axes[0].set_yticks([])
        axes[0].set_xlabel('位置索引')
        
        # 下图: 显示Cantor集的逐层构造过程
        axes[1].set_title('Cantor集构造过程（逐层展示）')
        for i in range(1, min(level+1, 7)):
            c = generate_cantor_set(level=i)
            y_pos = i * 1.2
            # 绘制每个保留的线段
            for j, val in enumerate(c):
                if val > 0:
                    axes[1].plot([j, j], [y_pos-0.3, y_pos+0.3], 'k-', linewidth=1)
            axes[1].text(-len(c)*0.05, y_pos, f'层级 {i}', ha='right', va='center')
        
        axes[1].set_xlim(-len(c)*0.1, len(c)*1.1)
        axes[1].set_ylim(0, (level+1)*1.2 + 1)
        axes[1].set_xlabel('位置索引')
        axes[1].set_yticks([])
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = os.path.join(current_dir, "result_cantor_set.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"\n3. 可视化结果已保存: {output_file}")
        plt.show()
        
    except ImportError:
        print("\n3. 可视化失败: 需要安装matplotlib库")
    
    print("\n示例运行完成！")


if __name__ == '__main__':
    main()

