#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
随机游走生成示例
==================

本示例演示如何使用fracDimPy生成多种类型的随机游走轨迹。
包括布朗运动、Lévy飞行和自回避行走，这些都是重要的随机过程模型，
广泛应用于物理学、生物学、金融学等领域。

主要功能：
- 生成布朗运动轨迹（Brownian Motion）
- 生成不同参数的Lévy飞行轨迹（Lévy Flight）
- 生成自回避行走轨迹（Self-Avoiding Walk）
- 可视化轨迹路径和密度热图
- 对比不同随机游走的特征

理论背景：
布朗运动：
- 分形维数 D = 2（在2D平面上）
- 步长服从正态分布，无长程跳跃

Lévy飞行：
- 步长服从幂律分布，存在长程跳跃
- 参数α ∈ (0, 2]控制跳跃距离分布
- α = 2时退化为布朗运动
- α < 2时出现长程跳跃

自回避行走：
- 不允许访问已经访问过的位置
- 分形维数约为4/3 ≈ 1.333（在2D平面上）
- 模拟高分子链的行为
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

def main():
    print("="*60)
    print("随机游走生成示例")
    print("="*60)
    
    from fracDimPy import generate_brownian_motion, generate_levy_flight, generate_self_avoiding_walk
    
    # 1. 生成布朗运动
    print("\n1. 正在生成布朗运动...")
    steps_bm = 10000  # 步数
    num_paths = 3     # 路径数
    
    paths_bm, image_bm = generate_brownian_motion(
        steps=steps_bm, 
        size=512, 
        num_paths=num_paths
    )
    
    print(f"   步数: {steps_bm}, 路径数: {num_paths}")
    
    # 2. 生成Lévy飞行
    print("\n2. 正在生成Lévy飞行...")
    steps_levy = 5000
    alphas = [1.0, 1.5, 2.0]  # 不同的α参数
    
    # 3. 生成自回避行走
    print("\n3. 正在生成自回避行走...")
    steps_saw = 200   # 步数（自回避行走计算较慢，使用较小步数）
    num_saw = 5       # 尝试生成的路径数
    
    paths_saw, image_saw = generate_self_avoiding_walk(
        steps=steps_saw,
        size=512,
        num_attempts=num_saw,
        max_retries=5000  # 最大重试次数
    )
    
    print(f"   步数: {steps_saw}, 成功生成路径数: {len(paths_saw)}")
    
    # 创建2行5列的综合可视化图表
    fig = plt.figure(figsize=(20, 8))
    
    # 第1列：布朗运动轨迹
    ax1 = fig.add_subplot(2, 5, 1)
    for i in range(num_paths):
        ax1.plot(paths_bm[i, :, 0], paths_bm[i, :, 1], 
                linewidth=0.5, alpha=0.7, label=f'路径 {i+1}')
    ax1.set_title('布朗运动\n(轨迹图)')
    ax1.set_xlabel('X坐标')
    ax1.set_ylabel('Y坐标')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_aspect('equal', adjustable='datalim')
    
    # 第6列：布朗运动密度热图
    ax2 = fig.add_subplot(2, 5, 6)
    ax2.imshow(image_bm, cmap='hot', origin='upper')
    ax2.set_title('布朗运动\n(密度热图)')
    ax2.axis('off')
    
    # 第2-4列：不同α参数的Lévy飞行轨迹
    for idx, alpha in enumerate(alphas):
        paths_levy, image_levy = generate_levy_flight(
            steps=steps_levy,
            size=512,
            alpha=alpha,
            num_paths=1
        )
        
        print(f"   Lévy飞行 α={alpha}: 生成了{steps_levy}步")
        
        # 第1行：轨迹图
        ax_traj = fig.add_subplot(2, 5, idx+2)
        ax_traj.plot(paths_levy[0, :, 0], paths_levy[0, :, 1], 
                    linewidth=0.5, alpha=0.8, color='blue')
        ax_traj.set_title(f'Lévy飞行\n(α={alpha})')
        ax_traj.set_xlabel('X坐标')
        ax_traj.set_ylabel('Y坐标')
        ax_traj.grid(True, alpha=0.3)
        ax_traj.set_aspect('equal', adjustable='datalim')
        
        # 第2行：密度热图
        ax_img = fig.add_subplot(2, 5, idx+7)
        ax_img.imshow(image_levy, cmap='hot', origin='upper')
        ax_img.set_title(f'Lévy飞行\n(α={alpha})')
        ax_img.axis('off')
    
    # 第5列：自回避行走轨迹
    ax_saw_traj = fig.add_subplot(2, 5, 5)
    for i, path in enumerate(paths_saw[:3]):  # 只显示前3条路径
        ax_saw_traj.plot(path[:, 0], path[:, 1], 
                        linewidth=0.8, alpha=0.7, label=f'路径 {i+1}')
    ax_saw_traj.set_title(f'自回避行走\n(步数={steps_saw})')
    ax_saw_traj.set_xlabel('X坐标')
    ax_saw_traj.set_ylabel('Y坐标')
    ax_saw_traj.grid(True, alpha=0.3)
    ax_saw_traj.legend()
    ax_saw_traj.set_aspect('equal', adjustable='datalim')
    
    # 第10列：自回避行走密度热图
    ax_saw_img = fig.add_subplot(2, 5, 10)
    ax_saw_img.imshow(image_saw, cmap='hot', origin='upper')
    ax_saw_img.set_title(f'自回避行走\n(共{len(paths_saw)}条路径)')
    ax_saw_img.axis('off')
    
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "result_random_walk.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n4. 可视化结果已保存: {output_file}")
    plt.show()
    
    print("\n" + "="*60)
    print("理论知识补充")
    print("="*60)
    print("   布朗运动：")
    print("   - 标准的随机游走，分形维数 D = 2")
    print("   - 步长服从正态分布")
    print("\n   Lévy飞行：")
    print("   - 参数α ∈ (0, 2]控制步长分布，Lévy飞行的特点")
    print("   - α = 2时退化为布朗运动")
    print("   - α < 2时出现长程跳跃")
    print("   - 广泛应用于动物觅食行为等领域")
    print("\n   自回避行走：")
    print("   - 分形维数约为 4/3 ≈ 1.333")
    print("   - 不允许重复访问已访问位置（\"不会走回头路\"）")
    print("   - 模拟高分子链、蛋白质折叠等")
    print("   - 计算复杂度高，生成速度较慢")
    print("\n示例运行完成！")


if __name__ == '__main__':
    main()
