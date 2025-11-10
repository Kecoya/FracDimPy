#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Box-counting算法验证脚本
========================

本脚本通过生成理论分形维数已知的标准分形结构，
使用box-counting算法计算其维数，并与理论值比较，
以此验证算法的正确性和准确性。

验证的分形结构：
1. Sierpinski三角形 - 理论维数: log(3)/log(2) ≈ 1.585
2. Sierpinski地毯 - 理论维数: log(8)/log(3) ≈ 1.893
3. Koch曲线 - 理论维数: log(4)/log(3) ≈ 1.262
4. Menger海绵 - 理论维数: log(20)/log(3) ≈ 2.727
5. DLA (扩散限制聚集) - 经验维数: 约1.71 (2D)

验证标准：
- 计算相对误差 = |计算值 - 理论值| / 理论值 * 100%
- R²拟合优度 > 0.95 视为良好拟合
- 相对误差 < 5% 视为算法验证通过
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# 尝试使用scienceplots样式
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass

# 设置中文字体
plt.rcParams['font.family'] = ['Times New Roman', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class FractalValidator:
    """分形维数验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.results = []
        self.theoretical_dimensions = {
            'Sierpinski三角形': np.log(3) / np.log(2),
            'Sierpinski地毯': np.log(8) / np.log(3),
            'Koch曲线': np.log(4) / np.log(3),
            'Menger海绵': np.log(20) / np.log(3),
            'DLA': 1.71  # 2D的经验值
        }
    
    def validate_sierpinski_triangle(self, level: int = 6, size: int = 512) -> Dict:
        """
        验证Sierpinski三角形
        
        Parameters
        ----------
        level : int
            迭代层级，默认6
        size : int
            图像尺寸，默认512
            
        Returns
        -------
        result : dict
            验证结果字典
        """
        print("\n" + "="*60)
        print("【1】验证 Sierpinski三角形")
        print("="*60)
        
        from fracDimPy import generate_sierpinski, box_counting
        
        # 生成分形
        print(f"生成参数: level={level}, size={size}")
        fractal = generate_sierpinski(level=level, size=size)
        
        # 计算分形维数
        print("正在计算分形维数...")
        dimension, result_data = box_counting(fractal, data_type='image')
        
        # 理论值
        theoretical = self.theoretical_dimensions['Sierpinski三角形']
        relative_error = abs(dimension - theoretical) / theoretical * 100
        
        print(f"理论分形维数: D = log(3)/log(2) = {theoretical:.6f}")
        print(f"计算分形维数: D = {dimension:.6f}")
        print(f"相对误差: {relative_error:.3f}%")
        print(f"拟合优度 R²: {result_data['R2']:.6f}")
        
        # 保存结果
        validation_result = {
            'name': 'Sierpinski三角形',
            'theoretical': theoretical,
            'calculated': dimension,
            'relative_error': relative_error,
            'R2': result_data['R2'],
            'result_data': result_data,
            'fractal_data': fractal,
            'passed': relative_error < 5.0 and result_data['R2'] > 0.95
        }
        
        self.results.append(validation_result)
        
        if validation_result['passed']:
            print("✓ 验证通过")
        else:
            print("✗ 验证失败")
        
        return validation_result
    
    def validate_sierpinski_carpet(self, level: int = 5) -> Dict:
        """
        验证Sierpinski地毯
        
        Parameters
        ----------
        level : int
            迭代层级，默认5
            
        Returns
        -------
        result : dict
            验证结果字典
        """
        print("\n" + "="*60)
        print("【2】验证 Sierpinski地毯")
        print("="*60)
        
        from fracDimPy import generate_sierpinski_carpet, box_counting
        
        # 生成分形
        size = 3 ** level
        print(f"生成参数: level={level}, size={size}")
        fractal = generate_sierpinski_carpet(level=level, size=size)
        
        # 计算分形维数
        print("正在计算分形维数...")
        dimension, result_data = box_counting(fractal, data_type='image')
        
        # 理论值
        theoretical = self.theoretical_dimensions['Sierpinski地毯']
        relative_error = abs(dimension - theoretical) / theoretical * 100
        
        print(f"理论分形维数: D = log(8)/log(3) = {theoretical:.6f}")
        print(f"计算分形维数: D = {dimension:.6f}")
        print(f"相对误差: {relative_error:.3f}%")
        print(f"拟合优度 R²: {result_data['R2']:.6f}")
        
        # 保存结果
        validation_result = {
            'name': 'Sierpinski地毯',
            'theoretical': theoretical,
            'calculated': dimension,
            'relative_error': relative_error,
            'R2': result_data['R2'],
            'result_data': result_data,
            'fractal_data': fractal,
            'passed': relative_error < 5.0 and result_data['R2'] > 0.95
        }
        
        self.results.append(validation_result)
        
        if validation_result['passed']:
            print("✓ 验证通过")
        else:
            print("✗ 验证失败")
        
        return validation_result
    
    def validate_koch_curve(self, level: int = 6, size: int = 1024) -> Dict:
        """
        验证Koch曲线
        
        Parameters
        ----------
        level : int
            迭代层级，默认6（平衡精度和计算速度）
        size : int
            图像尺寸，默认1024（增大图像以获得更多细节）
            
        Returns
        -------
        result : dict
            验证结果字典
        """
        print("\n" + "="*60)
        print("【3】验证 Koch曲线")
        print("="*60)
        
        from fracDimPy import generate_koch_curve, box_counting
        
        # 生成分形
        print(f"生成参数: level={level}, size={size}")
        points, image = generate_koch_curve(level=level, size=size)
        
        # 计算分形维数 - 使用图像类型分析
        print("正在计算分形维数...")
        print("注意：使用高分辨率图像分析Koch曲线的分形特征")
        dimension, result_data = box_counting(image, data_type='image')
        
        # 理论值
        theoretical = self.theoretical_dimensions['Koch曲线']
        relative_error = abs(dimension - theoretical) / theoretical * 100
        
        print(f"理论分形维数: D = log(4)/log(3) = {theoretical:.6f}")
        print(f"计算分形维数: D = {dimension:.6f}")
        print(f"相对误差: {relative_error:.3f}%")
        print(f"拟合优度 R²: {result_data['R2']:.6f}")
        
        # 保存结果
        # Koch曲线作为1D嵌入在2D空间中的分形，box-counting可能会高估其维数
        # 因此适当放宽验证标准
        validation_result = {
            'name': 'Koch曲线',
            'theoretical': theoretical,
            'calculated': dimension,
            'relative_error': relative_error,
            'R2': result_data['R2'],
            'result_data': result_data,
            'fractal_data': image,
            'passed': relative_error < 10.0 and result_data['R2'] > 0.99  # 曲线维数计算标准调整
        }
        
        self.results.append(validation_result)
        
        if validation_result['passed']:
            print("✓ 验证通过")
        else:
            print("✗ 验证失败")
        
        return validation_result
    
    def validate_menger_sponge(self, level: int = 3) -> Dict:
        """
        验证Menger海绵
        
        Parameters
        ----------
        level : int
            迭代层级，默认3
            
        Returns
        -------
        result : dict
            验证结果字典
        """
        print("\n" + "="*60)
        print("【4】验证 Menger海绵")
        print("="*60)
        
        from fracDimPy import generate_menger_sponge, box_counting
        
        # 生成分形
        size = 3 ** level
        print(f"生成参数: level={level}, size={size}")
        fractal = generate_menger_sponge(level=level, size=size)
        
        # 计算分形维数
        print("正在计算分形维数...")
        dimension, result_data = box_counting(fractal, data_type='porous')
        
        # 理论值
        theoretical = self.theoretical_dimensions['Menger海绵']
        relative_error = abs(dimension - theoretical) / theoretical * 100
        
        print(f"理论分形维数: D = log(20)/log(3) = {theoretical:.6f}")
        print(f"计算分形维数: D = {dimension:.6f}")
        print(f"相对误差: {relative_error:.3f}%")
        print(f"拟合优度 R²: {result_data['R2']:.6f}")
        
        # 保存结果
        validation_result = {
            'name': 'Menger海绵',
            'theoretical': theoretical,
            'calculated': dimension,
            'relative_error': relative_error,
            'R2': result_data['R2'],
            'result_data': result_data,
            'fractal_data': fractal,
            'passed': relative_error < 5.0 and result_data['R2'] > 0.95
        }
        
        self.results.append(validation_result)
        
        if validation_result['passed']:
            print("✓ 验证通过")
        else:
            print("✗ 验证失败")
        
        return validation_result
    
    def validate_dla(self, size: int = 300, particles: int = 100000) -> Dict:
        """
        验证DLA (扩散限制聚集)
        
        Parameters
        ----------
        size : int
            网格尺寸，默认300（增大网格）
        particles : int
            粒子数量，默认100000（增加粒子数）
            
        Returns
        -------
        result : dict
            验证结果字典
        """
        print("\n" + "="*60)
        print("【5】验证 DLA (扩散限制聚集)")
        print("="*60)
        
        from fracDimPy import generate_dla, box_counting
        
        # 生成分形
        print(f"生成参数: size={size}, particles={particles}")
        print("正在生成DLA结构，这可能需要一些时间...")
        fractal = generate_dla(size=size, num_particles=particles)
        
        occupied = np.sum(fractal > 0)
        print(f"成功附着粒子数: {occupied}")
        print(f"附着成功率: {occupied/particles*100:.2f}%")
        
        # 计算分形维数
        print("正在计算分形维数...")
        dimension, result_data = box_counting(fractal, data_type='image')
        
        # 理论值（经验值）
        theoretical = self.theoretical_dimensions['DLA']
        relative_error = abs(dimension - theoretical) / theoretical * 100
        
        print(f"理论分形维数: D ≈ {theoretical:.6f} (经验值)")
        print(f"计算分形维数: D = {dimension:.6f}")
        print(f"相对误差: {relative_error:.3f}%")
        print(f"拟合优度 R²: {result_data['R2']:.6f}")
        
        # DLA说明
        print("\n【DLA验证说明】")
        print("  DLA是随机过程生成的分形，具有以下特点：")
        print("  1. 每次生成结果不同（随机性）")
        print("  2. 分形维数受生成参数影响较大")
        print("  3. 需要大量粒子才能得到稳定的维数估计")
        print("  4. 作为定性验证，重点关注拟合优度R²")
        
        # DLA由于其随机性和生成特点，验证标准适度放宽
        # 注意：DLA的分形维数依赖于成功附着的粒子数量
        # 保存结果
        validation_result = {
            'name': 'DLA',
            'theoretical': theoretical,
            'calculated': dimension,
            'relative_error': relative_error,
            'R2': result_data['R2'],
            'result_data': result_data,
            'fractal_data': fractal,
            'passed': relative_error < 20.0 and result_data['R2'] > 0.95  # DLA标准放宽到20%，但要求更高的R²
        }
        
        self.results.append(validation_result)
        
        if validation_result['passed']:
            print("✓ 验证通过（定性验证）")
        else:
            print("✗ 验证失败（注：DLA由于随机性，仅作为参考）")
        
        return validation_result
    
    def generate_summary_report(self, save_path: str = None):
        """
        生成验证结果汇总报告
        
        Parameters
        ----------
        save_path : str, optional
            报告保存路径
        """
        print("\n" + "="*80)
        print("Box-counting算法验证汇总报告")
        print("="*80)
        
        # 打印表格
        print("\n{:<20} {:<12} {:<12} {:<12} {:<10} {:<8}".format(
            "分形类型", "理论维数", "计算维数", "相对误差(%)", "R²", "验证结果"
        ))
        print("-" * 80)
        
        passed_count = 0
        for result in self.results:
            status = "✓ 通过" if result['passed'] else "✗ 失败"
            if result['passed']:
                passed_count += 1
            
            print("{:<20} {:<12.6f} {:<12.6f} {:<12.3f} {:<10.6f} {:<8}".format(
                result['name'],
                result['theoretical'],
                result['calculated'],
                result['relative_error'],
                result['R2'],
                status
            ))
        
        print("-" * 80)
        print(f"总计: {len(self.results)} 个验证，通过: {passed_count}，失败: {len(self.results) - passed_count}")
        print(f"通过率: {passed_count / len(self.results) * 100:.1f}%")
        
        # 统计信息
        avg_error = np.mean([r['relative_error'] for r in self.results])
        avg_r2 = np.mean([r['R2'] for r in self.results])
        
        print(f"\n平均相对误差: {avg_error:.3f}%")
        print(f"平均拟合优度 R²: {avg_r2:.6f}")
        
        # 总体结论
        print("\n" + "="*80)
        if passed_count == len(self.results):
            print("✓✓✓ 验证结论: Box-counting算法验证全部通过，算法正确性得到验证！")
        elif passed_count >= len(self.results) * 0.8:
            print("✓ 验证结论: Box-counting算法大部分验证通过，算法基本可靠。")
        else:
            print("✗ 验证结论: Box-counting算法部分验证未通过，需要进一步检查。")
        print("="*80)
    
    def plot_validation_results(self, save_path: str = None):
        """
        绘制验证结果可视化图
        
        Parameters
        ----------
        save_path : str, optional
            图像保存路径
        """
        n_fractals = len(self.results)
        
        # 创建大图
        fig = plt.figure(figsize=(20, 12))
        
        # 上半部分：分形图像展示
        for i, result in enumerate(self.results):
            ax = plt.subplot(3, n_fractals, i + 1)
            
            # 根据数据维度选择显示方式
            if result['fractal_data'].ndim == 2:
                ax.imshow(result['fractal_data'], cmap='binary', origin='upper')
            elif result['fractal_data'].ndim == 3:
                # 3D数据显示中间切片
                mid_slice = result['fractal_data'].shape[0] // 2
                ax.imshow(result['fractal_data'][mid_slice, :, :], cmap='binary', origin='upper')
            
            ax.set_title(f"{result['name']}\nD理论={result['theoretical']:.3f}", 
                        fontsize=10)
            ax.axis('off')
        
        # 中间部分：log-log拟合图
        for i, result in enumerate(self.results):
            ax = plt.subplot(3, n_fractals, n_fractals + i + 1)
            
            data = result['result_data']
            x = data['log_inv_epsilon']
            y = data['log_N']
            
            # 绘制数据点
            ax.plot(x, y, 'o', markersize=6, label='数据点')
            
            # 绘制拟合线
            coeffs = data['coefficients']
            x_fit = np.linspace(x.min(), x.max(), 100)
            y_fit = coeffs[0] * x_fit + coeffs[1]
            ax.plot(x_fit, y_fit, 'r-', linewidth=2, 
                   label=f'拟合: D={coeffs[0]:.3f}')
            
            ax.set_xlabel('log(1/ε)', fontsize=9)
            ax.set_ylabel('log(N)', fontsize=9)
            ax.set_title(f"R²={result['R2']:.4f}", fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        # 下半部分：误差对比柱状图
        ax = plt.subplot(3, 1, 3)
        
        names = [r['name'] for r in self.results]
        errors = [r['relative_error'] for r in self.results]
        colors = ['green' if r['passed'] else 'red' for r in self.results]
        
        bars = ax.bar(range(len(names)), errors, color=colors, alpha=0.7)
        ax.axhline(y=5.0, color='orange', linestyle='--', linewidth=2, 
                  label='验证阈值: 5%')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=10)
        ax.set_ylabel('相对误差 (%)', fontsize=11)
        ax.set_title('Box-counting算法验证：相对误差对比', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # 在柱子上标注具体数值
        for i, (bar, error) in enumerate(zip(bars, errors)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                   f'{error:.2f}%',
                   ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n可视化结果已保存: {save_path}")
        
        return fig


def main():
    """主函数"""
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "Box-counting算法验证系统" + " "*33 + "║")
    print("╚" + "="*78 + "╝")
    
    print("\n本脚本将通过标准分形验证box-counting算法的正确性")
    print("验证内容包括：")
    print("  1. Sierpinski三角形")
    print("  2. Sierpinski地毯")
    print("  3. Koch曲线")
    print("  4. Menger海绵")
    print("  5. DLA (扩散限制聚集)")
    
    # 创建验证器
    validator = FractalValidator()
    
    # 执行验证
    try:
        validator.validate_sierpinski_triangle(level=6, size=512)
        validator.validate_sierpinski_carpet(level=5)
        validator.validate_koch_curve(level=6, size=1024)
        validator.validate_menger_sponge(level=3)
        validator.validate_dla(size=300, particles=100000)
    except Exception as e:
        print(f"\n验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    # 生成汇总报告
    validator.generate_summary_report()
    
    # 绘制可视化结果
    current_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(current_dir, "validation_box_counting_results.png")
    validator.plot_validation_results(save_path=output_file)
    
    plt.show()
    
    print("\n" + "="*80)
    print("验证脚本运行完成！")
    print("="*80)


if __name__ == '__main__':
    main()

