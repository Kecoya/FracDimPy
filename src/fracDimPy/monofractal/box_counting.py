#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Box-counting Method
===================

Implements box-counting algorithm for calculating fractal dimension.

Supports multiple data types:
- 1D curves
- 2D images  
- 3D surfaces
- Scatter data
- Porous media

The box-counting dimension is estimated from the scaling relationship:
log(N) ~ D * log(1/ε)
where N is the number of boxes and ε is the box size.

New Features:
- Boundary effect handling: 'valid', 'pad', 'periodic', 'reflect'
- Box partitioning strategies: 'fixed', 'sliding', 'random'
"""

import numpy as np
from scipy import interpolate
from typing import Tuple, Optional, Union, Literal


def box_counting(
    data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    data_type: str = 'curve',
    boundary_mode: str = 'valid',
    partition_strategy: str = 'fixed',
    **kwargs
) -> Tuple[float, dict]:
    """
    计算分形维数的盒计数方法
    
    Parameters
    ----------
    data : np.ndarray or tuple
        输入数据
        - 'curve': (x, y) 元组或单独的 y 数组
        - 'image': 2D 灰度图像数组
        - 'surface': 2D 高程数组
        - 'scatter': (x, y, z) 元组或散点位置数组
        - 'porous': 3D 二值数组
    data_type : str
        数据类型: 'curve', 'image', 'surface', 'scatter', 'porous'
    boundary_mode : str, optional
        边界效应处理模式:
        - 'valid': 仅使用完整盒子（默认）
        - 'pad': 零填充到完整盒子
        - 'periodic': 周期性边界条件
        - 'reflect': 镜像边界条件
    partition_strategy : str, optional
        盒子划分策略:
        - 'fixed': 固定网格（默认）
        - 'sliding': 滑动窗口，盒子部分重叠
        - 'random': 随机位置采样（多次平均）
    **kwargs : dict
        其他参数（传递给特定数据类型的处理函数）
        - n_random: 随机策略的采样次数（默认5）
        - sliding_step: 滑动步长因子（默认0.5）
        
    Returns
    -------
    dimension : float
        分形维数
    result : dict
        详细结果字典
        
    Examples
    --------
    >>> import numpy as np
    >>> from fracDimPy import box_counting
    >>> # 曲线数据
    >>> x = np.linspace(0, 1, 1000)
    >>> y = np.random.randn(1000)
    >>> D, result = box_counting((x, y), data_type='curve')
    >>> print(f"分形维数: {D:.4f}")
    >>> 
    >>> # 使用滑动窗口策略
    >>> D, result = box_counting((x, y), data_type='curve', 
    ...                          partition_strategy='sliding')
    >>> 
    >>> # 使用周期性边界
    >>> D, result = box_counting((x, y), data_type='curve',
    ...                          boundary_mode='periodic')
    """
    if data_type == 'curve':
        return _box_counting_curve(data, boundary_mode, partition_strategy, **kwargs)
    elif data_type == 'image':
        return _box_counting_image(data, boundary_mode, partition_strategy, **kwargs)
    elif data_type == 'surface':
        return _box_counting_surface(data, boundary_mode, partition_strategy, **kwargs)
    elif data_type == 'scatter':
        return _box_counting_scatter(data, boundary_mode, partition_strategy, **kwargs)
    elif data_type == 'porous':
        return _box_counting_porous(data, boundary_mode, partition_strategy, **kwargs)
    else:
        raise ValueError(f"不支持的数据类型: {data_type}")


def _apply_boundary_condition(data: np.ndarray, epsilon: int, 
                              boundary_mode: str = 'valid') -> np.ndarray:
    """
    应用边界条件处理
    
    Parameters
    ----------
    data : np.ndarray
        输入数组（1D、2D或3D）
    epsilon : int
        当前盒子大小
    boundary_mode : str
        边界处理模式
        
    Returns
    -------
    padded_data : np.ndarray
        处理后的数组
    """
    if boundary_mode == 'valid':
        # 不做处理，在计数时只使用完整盒子
        return data
    
    elif boundary_mode == 'pad':
        # 零填充到能被epsilon整除
        pad_widths = []
        for dim_size in data.shape:
            remainder = dim_size % epsilon
            if remainder != 0:
                pad_width = epsilon - remainder
            else:
                pad_width = 0
            pad_widths.append((0, pad_width))
        
        if data.ndim == 1:
            return np.pad(data, pad_widths[0], mode='constant', constant_values=0)
        elif data.ndim == 2:
            return np.pad(data, pad_widths, mode='constant', constant_values=0)
        elif data.ndim == 3:
            return np.pad(data, pad_widths, mode='constant', constant_values=0)
    
    elif boundary_mode == 'periodic':
        # 周期性边界 - 用wrap模式填充
        pad_widths = []
        for dim_size in data.shape:
            remainder = dim_size % epsilon
            if remainder != 0:
                pad_width = epsilon - remainder
            else:
                pad_width = 0
            pad_widths.append((0, pad_width))
        
        if data.ndim == 1:
            return np.pad(data, pad_widths[0], mode='wrap')
        elif data.ndim == 2:
            return np.pad(data, pad_widths, mode='wrap')
        elif data.ndim == 3:
            return np.pad(data, pad_widths, mode='wrap')
    
    elif boundary_mode == 'reflect':
        # 镜像边界
        pad_widths = []
        for dim_size in data.shape:
            remainder = dim_size % epsilon
            if remainder != 0:
                pad_width = epsilon - remainder
            else:
                pad_width = 0
            pad_widths.append((0, pad_width))
        
        if data.ndim == 1:
            return np.pad(data, pad_widths[0], mode='reflect')
        elif data.ndim == 2:
            return np.pad(data, pad_widths, mode='reflect')
        elif data.ndim == 3:
            return np.pad(data, pad_widths, mode='reflect')
    
    else:
        raise ValueError(f"不支持的边界模式: {boundary_mode}")


def _get_box_positions(data_shape: tuple, epsilon: int, 
                       strategy: str = 'fixed',
                       n_random: int = 5,
                       sliding_step: float = 0.5) -> list:
    """
    根据策略获取盒子位置
    
    Parameters
    ----------
    data_shape : tuple
        数据形状
    epsilon : int
        盒子大小
    strategy : str
        划分策略
    n_random : int
        随机策略的采样次数
    sliding_step : float
        滑动步长因子（相对于epsilon）
        
    Returns
    -------
    positions : list
        盒子起始位置列表，每个元素是坐标元组
    """
    if strategy == 'fixed':
        # 固定网格，不重叠
        if len(data_shape) == 1:
            positions = [(i,) for i in range(0, data_shape[0], epsilon)]
        elif len(data_shape) == 2:
            positions = [(i, j) 
                        for i in range(0, data_shape[0], epsilon)
                        for j in range(0, data_shape[1], epsilon)]
        elif len(data_shape) == 3:
            positions = [(i, j, k)
                        for i in range(0, data_shape[0], epsilon)
                        for j in range(0, data_shape[1], epsilon)
                        for k in range(0, data_shape[2], epsilon)]
        return positions
    
    elif strategy == 'sliding':
        # 滑动窗口，部分重叠
        step = max(1, int(epsilon * sliding_step))
        
        if len(data_shape) == 1:
            positions = [(i,) for i in range(0, data_shape[0] - epsilon + 1, step)]
        elif len(data_shape) == 2:
            positions = [(i, j)
                        for i in range(0, data_shape[0] - epsilon + 1, step)
                        for j in range(0, data_shape[1] - epsilon + 1, step)]
        elif len(data_shape) == 3:
            positions = [(i, j, k)
                        for i in range(0, data_shape[0] - epsilon + 1, step)
                        for j in range(0, data_shape[1] - epsilon + 1, step)
                        for k in range(0, data_shape[2] - epsilon + 1, step)]
        return positions
    
    elif strategy == 'random':
        # 随机位置采样
        positions = []
        np.random.seed(42)  # 保证可重复性
        
        if len(data_shape) == 1:
            max_pos = data_shape[0] - epsilon
            for _ in range(n_random):
                i = np.random.randint(0, max(1, max_pos))
                positions.append((i,))
        elif len(data_shape) == 2:
            max_i = data_shape[0] - epsilon
            max_j = data_shape[1] - epsilon
            for _ in range(n_random):
                i = np.random.randint(0, max(1, max_i))
                j = np.random.randint(0, max(1, max_j))
                positions.append((i, j))
        elif len(data_shape) == 3:
            max_i = data_shape[0] - epsilon
            max_j = data_shape[1] - epsilon
            max_k = data_shape[2] - epsilon
            for _ in range(n_random):
                i = np.random.randint(0, max(1, max_i))
                j = np.random.randint(0, max(1, max_j))
                k = np.random.randint(0, max(1, max_k))
                positions.append((i, j, k))
        return positions
    
    else:
        raise ValueError(f"不支持的划分策略: {strategy}")


def _box_counting_curve(
    data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    boundary_mode: str = 'valid',
    partition_strategy: str = 'fixed',
    **kwargs
) -> Tuple[float, dict]:
    """
    曲线数据的盒计数分形维数
    
    Parameters
    ----------
    data : tuple or np.ndarray
        (x, y) 元组或单独的 y 数组
    boundary_mode : str
        边界处理模式
    partition_strategy : str
        盒子划分策略
        
    Returns
    -------
    dimension : float
        分形维数
    result : dict
        详细结果
    """
    # 解析数据
    if isinstance(data, tuple):
        x, y = data
    else:
        y = data
        x = np.arange(len(y))
    
    # 归一化
    x = (x - np.min(x)) / np.ptp(x)
    y = (y - np.min(y)) / np.ptp(y)
    
    # 插值到2的幂次
    x_new = np.linspace(np.min(x), np.max(x), 2 ** (int(np.log2(len(x))) + 1))
    f_linear = interpolate.interp1d(x, y, kind='linear')
    y_new = f_linear(x_new)
    
    # 转换为矩阵
    mt_epsilon = x_new[1] - x_new[0]
    mt = _curve_to_matrix(x_new, y_new, epsilon=mt_epsilon)
    
    # 获取参数
    n_random = kwargs.get('n_random', 5)
    sliding_step = kwargs.get('sliding_step', 0.5)
    
    # 计算盒子数量
    height, width = mt.shape
    M = min(height, width)
    
    Nl = []
    epsilonl = []
    
    for i in range(1, int(np.log(M) / np.log(2)) + 1):
        epsilon = 2 ** i
        
        # 应用边界条件
        mt_processed = _apply_boundary_condition(mt, epsilon, boundary_mode)
        
        # 根据策略计数
        N = _count_boxes_2d_advanced(mt_processed, epsilon, 
                                     partition_strategy, 
                                     n_random, sliding_step)
        if N == 0:
            N = 1  # 避免log(0)
        Nl.append(N)
        epsilonl.append(epsilon * mt_epsilon)
    
    # 线性拟合
    x_fit = np.log(np.array([1 / epsilon for epsilon in epsilonl]))
    y_fit = np.log(Nl)
    
    coefficients = np.polyfit(x_fit, y_fit, 1)
    f = np.poly1d(coefficients)
    
    dimension = coefficients[0]
    R2 = np.corrcoef(y_fit, f(x_fit))[0, 1] ** 2
    
    result = {
        'dimension': dimension,
        'N_values': Nl,
        'epsilon_values': epsilonl,
        'log_inv_epsilon': x_fit,
        'log_N': y_fit,
        'R2': R2,
        'coefficients': coefficients,
        'method': f'Box-counting (Curve) - {boundary_mode}/{partition_strategy}',
        'matrix_shape': mt.shape,
        'boundary_mode': boundary_mode,
        'partition_strategy': partition_strategy
    }
    
    return dimension, result


def _curve_to_matrix(x: np.ndarray, y: np.ndarray, epsilon: float) -> np.ndarray:
    """
    
    
    Parameters
    ----------
    x, y : np.ndarray
        
    epsilon : float
        
        
    Returns
    -------
    matrix : np.ndarray
        
    """
    y_ = np.round((y - np.min(y)) / epsilon).astype(int)
    x_ = np.round((x - np.min(x)) / epsilon).astype(int)
    
    matrix = np.zeros((np.max(y_) + 1, np.max(x_) + 1))
    matrix[y_, x_] = 1
    
    return np.array(matrix, dtype=np.int8)


def _count_boxes_2d(MT: np.ndarray, EPSILON: int) -> int:
    """
    2D盒计数（固定网格）
    
    Parameters
    ----------
    MT : np.ndarray
        二维矩阵
    EPSILON : int
        盒子大小
        
    Returns
    -------
    count : int
        有效盒子数量
    """
    # 沿第0轴合并EPSILON个单元
    MT_BOX_0 = np.add.reduceat(
        MT,
        np.arange(0, MT.shape[0], EPSILON),
        axis=0
    )
    # 沿第1轴合并EPSILON个单元
    MT_BOX_1 = np.add.reduceat(
        MT_BOX_0,
        np.arange(0, MT.shape[1], EPSILON),
        axis=1
    )
    
    # 计数非空盒子
    return len(np.where((MT_BOX_1 > 0) & (MT_BOX_1 <= EPSILON ** 2 * 1))[0])


def _count_boxes_2d_advanced(MT: np.ndarray, EPSILON: int,
                             strategy: str = 'fixed',
                             n_random: int = 5,
                             sliding_step: float = 0.5) -> float:
    """
    2D盒计数（支持多种策略）
    
    Parameters
    ----------
    MT : np.ndarray
        二维矩阵
    EPSILON : int
        盒子大小
    strategy : str
        划分策略: 'fixed', 'sliding', 'random'
    n_random : int
        随机策略的采样次数
    sliding_step : float
        滑动步长因子
        
    Returns
    -------
    count : float
        有效盒子数量（对于随机策略可能是平均值）
    """
    if strategy == 'fixed':
        return _count_boxes_2d(MT, EPSILON)
    
    elif strategy == 'sliding':
        # 滑动窗口策略
        step = max(1, int(EPSILON * sliding_step))
        count = 0
        boxes_checked = 0
        
        for i in range(0, MT.shape[0] - EPSILON + 1, step):
            for j in range(0, MT.shape[1] - EPSILON + 1, step):
                box = MT[i:i+EPSILON, j:j+EPSILON]
                if np.sum(box) > 0:
                    count += 1
                boxes_checked += 1
        
        # 对重叠部分进行归一化
        if boxes_checked > 0:
            # 计算重叠因子并调整计数
            overlap_factor = (step / EPSILON) ** 2
            return count * overlap_factor
        return count
    
    elif strategy == 'random':
        # 随机采样策略
        counts = []
        for _ in range(n_random):
            positions = _get_box_positions(MT.shape, EPSILON, 'random', 
                                          n_random=1, sliding_step=sliding_step)
            count = 0
            for pos in positions:
                i, j = pos
                box = MT[i:i+EPSILON, j:j+EPSILON]
                if np.sum(box) > 0:
                    count += 1
            
            # 估算总盒子数
            total_boxes_possible = (MT.shape[0] // EPSILON) * (MT.shape[1] // EPSILON)
            counts.append(count * total_boxes_possible)
        
        return np.mean(counts)
    
    else:
        return _count_boxes_2d(MT, EPSILON)


def _box_counting_image(image: np.ndarray,
                         boundary_mode: str = 'valid',
                         partition_strategy: str = 'fixed',
                         **kwargs) -> Tuple[float, dict]:
    """
    图像数据的盒计数分形维数
    
    Parameters
    ----------
    image : np.ndarray
        二维灰度图像数组
    boundary_mode : str
        边界处理模式
    partition_strategy : str
        盒子划分策略
        
    Returns
    -------
    dimension : float
        分形维数
    result : dict
        详细结果
    """
    # 检查2D
    if image.ndim != 2:
        raise ValueError(f"期望2D数组，得到: {image.ndim}维")
    
    # 转换0-1到0-255
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)
    
    mt = image
    height, width = mt.shape
    M = min(height, width)
    
    # 获取参数
    n_random = kwargs.get('n_random', 5)
    sliding_step = kwargs.get('sliding_step', 0.5)
    
    print(f'图像形状 (height, width): {height}, {width}')
    print(f'像素值范围: {mt.min()} ~ {mt.max()}')
    print(f'边界模式: {boundary_mode}, 划分策略: {partition_strategy}')
    
    # 计算盒子
    Nl = []
    # 尺度列表
    epsilonl = [2**i for i in range(1, int(np.log(M) / np.log(2)) + 1)]
    
    for epsilon in epsilonl:
        # 应用边界条件
        mt_processed = _apply_boundary_condition(mt, epsilon, boundary_mode)
        
        # 根据策略计数
        N = _count_boxes_image_advanced(mt_processed, epsilon,
                                        partition_strategy,
                                        n_random, sliding_step)
        if N == 0:
            N = 1  # 避免log(0)
        Nl.append(N)
        print(f'N:{N}  盒子大小:{epsilon}       网格数: {height/epsilon} x {width/epsilon}')
    
    print('盒子数 N: ', Nl)
    print('盒子尺度: ', epsilonl)
    
    # 线性拟合
    x_fit = np.log(np.array([1 / epsilon for epsilon in epsilonl]))
    y_fit = np.log(Nl)
    
    coefficients = np.polyfit(x_fit, y_fit, 1)
    f = np.poly1d(coefficients)
    
    dimension = coefficients[0]
    R2 = np.corrcoef(y_fit, f(x_fit))[0, 1] ** 2
    
    print(f'拟合函数: {f}')
    print(f'拟合度 R²: {R2:.6f}')
    
    result = {
        'dimension': dimension,
        'N_values': Nl,
        'epsilon_values': epsilonl,
        'log_inv_epsilon': x_fit,
        'log_N': y_fit,
        'R2': R2,
        'coefficients': coefficients,
        'method': f'Box-counting (Image) - {boundary_mode}/{partition_strategy}',
        'image_shape': mt.shape,
        'boundary_mode': boundary_mode,
        'partition_strategy': partition_strategy
    }
    
    return dimension, result


def _count_boxes_image(MT: np.ndarray, EPSILON: int) -> int:
    """
    图像盒计数（固定网格）
    
    Parameters
    ----------
    MT : np.ndarray
        图像矩阵
    EPSILON : int
        盒子大小
        
    Returns
    -------
    count : int
        有效盒子数量
    """
    # 沿第0轴合并EPSILON个单元
    MT_BOX_0 = np.add.reduceat(MT,
                               np.arange(0, MT.shape[0], EPSILON),
                               axis=0)
    # 沿第1轴合并EPSILON个单元
    MT_BOX_1 = np.add.reduceat(MT_BOX_0,
                               np.arange(0, MT.shape[1], EPSILON),
                               axis=1)
    # 计数非空盒子
    return len(
        np.where((MT_BOX_1 > 0) & (MT_BOX_1 <= EPSILON ** 2 * 255))[0]
    )


def _count_boxes_image_advanced(MT: np.ndarray, EPSILON: int,
                                strategy: str = 'fixed',
                                n_random: int = 5,
                                sliding_step: float = 0.5) -> float:
    """
    图像盒计数（支持多种策略）
    
    Parameters
    ----------
    MT : np.ndarray
        图像矩阵
    EPSILON : int
        盒子大小
    strategy : str
        划分策略
    n_random : int
        随机策略的采样次数
    sliding_step : float
        滑动步长因子
        
    Returns
    -------
    count : float
        有效盒子数量
    """
    if strategy == 'fixed':
        return _count_boxes_image(MT, EPSILON)
    
    elif strategy == 'sliding':
        # 滑动窗口策略
        step = max(1, int(EPSILON * sliding_step))
        count = 0
        boxes_checked = 0
        
        for i in range(0, MT.shape[0] - EPSILON + 1, step):
            for j in range(0, MT.shape[1] - EPSILON + 1, step):
                box = MT[i:i+EPSILON, j:j+EPSILON]
                if np.sum(box) > 0:
                    count += 1
                boxes_checked += 1
        
        # 归一化
        if boxes_checked > 0:
            overlap_factor = (step / EPSILON) ** 2
            return count * overlap_factor
        return count
    
    elif strategy == 'random':
        # 随机采样策略
        counts = []
        for _ in range(n_random):
            positions = _get_box_positions(MT.shape, EPSILON, 'random',
                                          n_random=1, sliding_step=sliding_step)
            count = 0
            for pos in positions:
                i, j = pos
                box = MT[i:i+EPSILON, j:j+EPSILON]
                if np.sum(box) > 0:
                    count += 1
            
            total_boxes_possible = (MT.shape[0] // EPSILON) * (MT.shape[1] // EPSILON)
            counts.append(count * total_boxes_possible)
        
        return np.mean(counts)
    
    else:
        return _count_boxes_image(MT, EPSILON)


def _box_counting_surface(surface: np.ndarray, 
                          boundary_mode: str = 'valid',
                          partition_strategy: str = 'fixed',
                          method: int = 2, 
                          mt_epsilon_min: float = None, 
                          n_scales: int = 5,
                          **kwargs) -> Tuple[float, dict]:
    """
    曲面数据的盒计数分形维数（支持多种计数方法）
    
    Parameters
    ----------
    surface : np.ndarray
        二维高程数组
    boundary_mode : str
        边界处理模式（注意：surface使用特殊的边界处理）
    partition_strategy : str
        盒子划分策略
    method : int, optional
        默认2 (CCM):
        0: RDCCM - 范围差分立方体计数
        1: DCCM - 差分立方体计数
        2: CCM - 立方体计数（默认）
        3: ICCM - 插值立方体计数
        5: SCCM - 简化立方体计数（按行）
        6: SDCCM - 简化差分立方体计数（按行）
    mt_epsilon_min : float, optional
        最小网格间距
    n_scales : int, optional
        尺度数量，默认5
        
    Returns
    -------
    dimension : float
        分形维数
    result : dict
        详细结果
    """
    # 检查2D
    if surface.ndim != 2:
        raise ValueError(f"期望2D数组，得到: {surface.ndim}维")
    
    # 设置最小间距
    if mt_epsilon_min is None:
        mt_epsilon_min = 1.0
    
    mt = surface - np.min(surface)  # 归零
    
    # 获取参数
    n_random = kwargs.get('n_random', 5)
    sliding_step = kwargs.get('sliding_step', 0.5)
    
    y_lenth = mt.shape[0] * mt_epsilon_min
    x_lenth = mt.shape[1] * mt_epsilon_min
    M = min(x_lenth, y_lenth)
    
    method_names = {
        0: "RDCCM", 1: "DCCM", 2: "CCM", 
        3: "ICCM", 5: "SCCM", 6: "SDCCM"
    }
    
    print(f'曲面形状: {mt.shape}')
    print(f'X长度: {x_lenth:.4f}, Y长度: {y_lenth:.4f}')
    print(f'最小尺度: {M:.4f}, 网格间距: {mt_epsilon_min}')
    print(f'计数方法: {method_names.get(method, "Unknown")} (method={method})')
    print(f'边界模式: {boundary_mode}, 划分策略: {partition_strategy}')
    
    # 生成尺度列表
    epsilon_min = mt_epsilon_min * 2
    epsilonl = [i * epsilon_min for i in range(1, int(min(mt.shape) / 2)) if i < n_scales + 1]
    
    # 计算盒子
    N_list, epsilon_list = _count_boxes_surface(mt, epsilonl, mt_epsilon_min, method,
                                                boundary_mode, partition_strategy,
                                                n_random, sliding_step)
    
    print('盒子数 N: ', N_list)
    print('盒子尺度: ', epsilon_list)
    
    # 线性拟合
    x_fit = np.log(np.array([1 / epsilon for epsilon in epsilon_list]))
    y_fit = np.log(N_list)
    
    coefficients = np.polyfit(x_fit, y_fit, 1)
    f = np.poly1d(coefficients)
    
    dimension = coefficients[0]
    R2 = np.corrcoef(y_fit, f(x_fit))[0, 1] ** 2
    
    print(f'拟合函数: {f}')
    print(f'拟合度 R²: {R2:.6f}')
    
    result = {
        'dimension': dimension,
        'N_values': N_list,
        'epsilon_values': epsilon_list,
        'log_inv_epsilon': x_fit,
        'log_N': y_fit,
        'R2': R2,
        'coefficients': coefficients,
        'method': f'Box-counting (Surface) - {method_names.get(method, "Unknown")} - {boundary_mode}/{partition_strategy}',
        'surface_shape': mt.shape,
        'grid_size': mt_epsilon_min,
        'boundary_mode': boundary_mode,
        'partition_strategy': partition_strategy
    }
    
    return dimension, result


def _count_boxes_surface(mt: np.ndarray, epsilonl: list, 
                         mt_epsilon_min: float, method: int,
                         boundary_mode: str = 'valid',
                         partition_strategy: str = 'fixed',
                         n_random: int = 5,
                         sliding_step: float = 0.5) -> Tuple[list, list]:
    """
    曲面盒子计数（支持多种策略）
    
    Parameters
    ----------
    mt : np.ndarray
        曲面高程矩阵
    epsilonl : list
        盒子尺度列表
    mt_epsilon_min : float
        最小网格间距
    method : int
        计数方法
    boundary_mode : str
        边界处理模式
    partition_strategy : str
        划分策略
    n_random : int
        随机采样次数
    sliding_step : float
        滑动步长因子
        
    Returns
    -------
    N_list : list
        盒子数量列表
    epsilon_list : list
        实际使用的尺度列表
    """
    from itertools import product
    
    y_lenth = mt.shape[0] * mt_epsilon_min
    x_lenth = mt.shape[1] * mt_epsilon_min
    
    N_list = []
    epsilon_list = []
    
    # RDCCM: 
    if method == 0:
        for epsilon in epsilonl:
            N = 0
            item = int(epsilon / mt_epsilon_min)
            for j, k in product(range(int(y_lenth // epsilon)), 
                               range(int(x_lenth // epsilon))):
                z_list = mt[j * item: j * item + item,
                           k * item: k * item + item]
                N += np.ceil(np.ptp(z_list) / epsilon)
            N_list.append(N)
            epsilon_list.append(epsilon)
    
    # DCCM: 
    elif method == 1:
        for epsilon in epsilonl:
            N = 0
            item = int(epsilon / mt_epsilon_min)
            for j, k in product(range(int(y_lenth // epsilon)), 
                               range(int(x_lenth // epsilon))):
                z_list = mt[j * item: j * item + item,
                           k * item: k * item + item]
                N += np.ceil(np.max(z_list) / epsilon) - np.ceil(np.min(z_list) / epsilon)
            N_list.append(N)
            epsilon_list.append(epsilon)
    
    # CCM: 
    elif method == 2:
        for epsilon in epsilonl:
            N = 0
            item = int(epsilon / mt_epsilon_min)
            for j, k in product(range(int(y_lenth // epsilon)), 
                               range(int(x_lenth // epsilon))):
                z_list = mt[j * item: j * item + item,
                           k * item: k * item + item]
                
                # 
                z_corners = [
                    z_list[0, 0],
                    z_list[-1, 0],
                    z_list[-1, -1],
                    z_list[0, -1]
                ]
                
                # 0
                if 0 in z_corners:
                    continue
                
                N += np.ceil(np.ptp(z_corners) / epsilon)
            N_list.append(N)
            epsilon_list.append(epsilon)
    
    # ICCM: 
    elif method == 3:
        for epsilon in epsilonl:
            N = 0
            item = int(epsilon / mt_epsilon_min)
            for j, k in product(range(int(y_lenth // epsilon)), 
                               range(int(x_lenth // epsilon))):
                z_list = mt[j * item: j * item + item,
                           k * item: k * item + item]
                
                # 
                z_corners = [
                    z_list[0, 0],
                    z_list[-1, 0],
                    z_list[-1, -1],
                    z_list[0, -1]
                ]
                
                N += np.ceil(np.max(z_corners) / epsilon) - np.ceil(np.min(z_corners) / epsilon)
            N_list.append(N)
            epsilon_list.append(epsilon)
    
    # SCCM: /
    elif method == 5:
        for epsilon in epsilonl:
            N = 0
            item = int(epsilon / mt_epsilon_min)
            # 
            for j in range(int(y_lenth // epsilon)):
                N_row = 0  # 
                for k in range(int(x_lenth // epsilon)):
                    z_list = mt[j * item: j * item + item,
                               k * item: k * item + item]
                    
                    # 
                    z_corners = [
                        z_list[0, 0],
                        z_list[-1, 0],
                        z_list[-1, -1],
                        z_list[0, -1]
                    ]
                    
                    # 
                    N_row += np.ptp(z_corners) / epsilon
                
                # 
                N += np.ceil(N_row)
            
            N_list.append(int(np.ceil(N)))
            epsilon_list.append(epsilon)
    
    # SDCCM: /
    elif method == 6:
        for epsilon in epsilonl:
            N = 0
            item = int(epsilon / mt_epsilon_min)
            # 
            for j in range(int(y_lenth // epsilon)):
                N_row = 0  # 
                for k in range(int(x_lenth // epsilon)):
                    z_list = mt[j * item: j * item + item,
                               k * item: k * item + item]
                    
                    # 
                    N_row += np.ptp(z_list) / epsilon
                
                # 
                N += np.ceil(N_row)
            
            N_list.append(int(N))
            epsilon_list.append(epsilon)
    
    else:
        raise ValueError(f"method: {method}: 0,1,2,3,5,6")
    
    return N_list, epsilon_list


def _box_counting_scatter(scatter: Union[np.ndarray, Tuple],
                          boundary_mode: str = 'valid',
                          partition_strategy: str = 'fixed',
                          **kwargs) -> Tuple[float, dict]:
    """
    散点数据的盒计数分形维数
    
    Parameters
    ----------
    scatter : np.ndarray or tuple
        散点数据:
        1. 二值数组 0/1 [0,1,0,0,1,1,0,...]
        2. 坐标数组 [1.5, 3.2, 7.8, ...]
           （自动转换为二值矩阵）
    boundary_mode : str
        边界处理模式
    partition_strategy : str
        盒子划分策略
        
    Returns
    -------
    dimension : float
        分形维数
    result : dict
        详细结果
    """
    # 解析输入
    if isinstance(scatter, tuple):
        # 元组形式，取第一个元素
        scatter = scatter[0]
    
    # 展平为1D
    scatter = np.asarray(scatter).flatten()
    
    # 转换为二值矩阵
    if not np.all(np.isin(scatter, [0, 1])):
        # 坐标数据，转换为二值矩阵
        mt, mt_epsilon = _coordinate_to_matrix(scatter)
    else:
        # 已经是二值数据
        mt = scatter.astype(np.int8)
        mt_epsilon = 1.0
    
    # 获取参数
    n_random = kwargs.get('n_random', 5)
    sliding_step = kwargs.get('sliding_step', 0.5)
    
    print(f'数据长度: {len(mt)}')
    print(f'散点数量: {np.sum(mt)}')
    print(f'最小间距: {mt_epsilon}')
    print(f'边界模式: {boundary_mode}, 划分策略: {partition_strategy}')
    
    # 计算盒子
    Nl = []
    # 尺度列表
    epsilonl = []
    
    for i in range(1, int(np.log(len(mt)) / np.log(2)) + 1):
        epsilon = 2 ** i
        
        # 应用边界条件
        mt_processed = _apply_boundary_condition(mt, epsilon, boundary_mode)
        
        # 根据策略计数
        N = _count_boxes_1d_advanced(mt_processed, epsilon,
                                     partition_strategy,
                                     n_random, sliding_step)
        if N == 0:
            N = 1  # 避免log(0)
        Nl.append(N)
        epsilonl.append(epsilon * mt_epsilon)
    
    print('盒子数 N: ', Nl)
    print('盒子尺度: ', epsilonl)
    
    # 线性拟合
    x_fit = np.log(np.array([1 / epsilon for epsilon in epsilonl]))
    y_fit = np.log(Nl)
    
    coefficients = np.polyfit(x_fit, y_fit, 1)
    f = np.poly1d(coefficients)
    
    dimension = coefficients[0]
    R2 = np.corrcoef(y_fit, f(x_fit))[0, 1] ** 2
    
    print(f'拟合函数: {f}')
    print(f'拟合度 R²: {R2:.6f}')
    
    result = {
        'dimension': dimension,
        'N_values': Nl,
        'epsilon_values': epsilonl,
        'log_inv_epsilon': x_fit,
        'log_N': y_fit,
        'R2': R2,
        'coefficients': coefficients,
        'method': f'Box-counting (1D Scatter) - {boundary_mode}/{partition_strategy}',
        'data_length': len(mt),
        'scatter_count': int(np.sum(mt)),
        'min_epsilon': mt_epsilon,
        'boundary_mode': boundary_mode,
        'partition_strategy': partition_strategy
    }
    
    return dimension, result


def _coordinate_to_matrix(x: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    
    
    Parameters
    ----------
    x : np.ndarray
        
        
    Returns
    -------
    matrix : np.ndarray
        0/1
    epsilon : float
        
    """
    # 
    x_range = np.max(x) - np.min(x)
    num = 2 ** (int(np.log2(len(x))) + 1)
    epsilon = x_range / num
    
    # 
    x_indices = np.round((x - np.min(x)) / epsilon).astype(int)
    
    # 
    matrix = np.zeros(np.max(x_indices) + 1, dtype=np.int8)
    matrix[x_indices] = 1
    
    return matrix, epsilon


def _count_boxes_1d(MT: np.ndarray, EPSILON: int) -> int:
    """
    1D盒计数（固定网格）
    
    Parameters
    ----------
    MT : np.ndarray
        一维数组
    EPSILON : int
        盒子大小
        
    Returns
    -------
    count : int
        有效盒子数量
    """
    # 沿第0轴合并EPSILON个单元
    MT_BOX = np.add.reduceat(MT,
                             np.arange(0, MT.shape[0], EPSILON),
                             axis=0)
    # 计数非空盒子: 0 < 值 <= EPSILON   
    return len(np.where((MT_BOX > 0) & (MT_BOX <= EPSILON * 1))[0])


def _count_boxes_1d_advanced(MT: np.ndarray, EPSILON: int,
                             strategy: str = 'fixed',
                             n_random: int = 5,
                             sliding_step: float = 0.5) -> float:
    """
    1D盒计数（支持多种策略）
    
    Parameters
    ----------
    MT : np.ndarray
        一维数组
    EPSILON : int
        盒子大小
    strategy : str
        划分策略
    n_random : int
        随机策略的采样次数
    sliding_step : float
        滑动步长因子
        
    Returns
    -------
    count : float
        有效盒子数量
    """
    if strategy == 'fixed':
        return _count_boxes_1d(MT, EPSILON)
    
    elif strategy == 'sliding':
        # 滑动窗口策略
        step = max(1, int(EPSILON * sliding_step))
        count = 0
        boxes_checked = 0
        
        for i in range(0, len(MT) - EPSILON + 1, step):
            box = MT[i:i+EPSILON]
            if np.sum(box) > 0:
                count += 1
            boxes_checked += 1
        
        # 归一化
        if boxes_checked > 0:
            overlap_factor = step / EPSILON
            return count * overlap_factor
        return count
    
    elif strategy == 'random':
        # 随机采样策略
        counts = []
        for _ in range(n_random):
            positions = _get_box_positions((len(MT),), EPSILON, 'random',
                                          n_random=1, sliding_step=sliding_step)
            count = 0
            for pos in positions:
                i = pos[0]
                box = MT[i:i+EPSILON]
                if np.sum(box) > 0:
                    count += 1
            
            total_boxes_possible = len(MT) // EPSILON
            counts.append(count * total_boxes_possible)
        
        return np.mean(counts)
    
    else:
        return _count_boxes_1d(MT, EPSILON)


def _box_counting_porous(porous: np.ndarray,
                         boundary_mode: str = 'valid',
                         partition_strategy: str = 'fixed',
                         **kwargs) -> Tuple[float, dict]:
    """
    多孔介质数据的盒计数分形维数
    
    Parameters
    ----------
    porous : np.ndarray
        三维二值数组 (0, 1)
        1表示孔隙，0表示固体
    boundary_mode : str
        边界处理模式
    partition_strategy : str
        盒子划分策略
        
    Returns
    -------
    dimension : float
        分形维数
    result : dict
        详细结果
    """
    # 检查3D
    if porous.ndim != 3:
        raise ValueError(f"期望3D数组，得到: {porous.ndim}维")
    
    # 转换为二值
    if not np.all(np.isin(porous, [0, 1])):
        # 非二值数据，阈值化
        porous = (porous > 0).astype(np.int8)
    
    mt = porous.astype(np.int8)
    depth, height, width = mt.shape
    M = np.min(mt.shape)
    
    # 获取参数
    n_random = kwargs.get('n_random', 5)
    sliding_step = kwargs.get('sliding_step', 0.5)
    
    print(f'多孔介质形状 (depth, height, width): {depth}, {height}, {width}')
    print(f'孔隙体素: {np.sum(mt == 1)}, 固体体素: {np.sum(mt == 0)}')
    print(f'边界模式: {boundary_mode}, 划分策略: {partition_strategy}')
    
    # 计算盒子
    Nl = []
    # 尺度列表
    epsilonl = [2**i for i in range(1, int(np.log(M) / np.log(2)) + 1)]

    for epsilon in epsilonl:
        # 应用边界条件
        mt_processed = _apply_boundary_condition(mt, epsilon, boundary_mode)
        
        # 根据策略计数
        N = _count_boxes_3d_advanced(mt_processed, epsilon,
                                     partition_strategy,
                                     n_random, sliding_step)
        if N == 0:
            N = 1  # 避免log(0)
        Nl.append(N)
        print(f'R:{epsilon}  N_R:{N}')
    
    print('盒子数 N: ', Nl)
    print('盒子尺度: ', epsilonl)
    
    # 线性拟合
    x_fit = np.log(np.array([1 / epsilon for epsilon in epsilonl]))
    y_fit = np.log(Nl)
    
    coefficients = np.polyfit(x_fit, y_fit, 1)
    f = np.poly1d(coefficients)
    
    dimension = coefficients[0]
    R2 = np.corrcoef(y_fit, f(x_fit))[0, 1] ** 2
    
    print(f'拟合函数: {f}')
    print(f'拟合度 R²: {R2:.6f}')
    
    result = {
        'dimension': dimension,
        'N_values': Nl,
        'epsilon_values': epsilonl,
        'log_inv_epsilon': x_fit,
        'log_N': y_fit,
        'R2': R2,
        'coefficients': coefficients,
        'method': f'Box-counting (Porous Media) - {boundary_mode}/{partition_strategy}',
        'data_shape': mt.shape,
        'boundary_mode': boundary_mode,
        'partition_strategy': partition_strategy
    }
    
    return dimension, result


def _count_boxes_3d(MT: np.ndarray, EPSILON: int) -> int:
    """
    3D盒计数（固定网格）
    
    Parameters
    ----------
    MT : np.ndarray
        三维数组
    EPSILON : int
        盒子大小
        
    Returns
    -------
    count : int
        有效盒子数量
    """
    # 沿第0轴合并EPSILON个单元
    MT_BOX_0 = np.add.reduceat(MT,
                               np.arange(0, MT.shape[0], EPSILON),
                               axis=0)
    # 沿第1轴合并EPSILON个单元
    MT_BOX_1 = np.add.reduceat(MT_BOX_0,
                               np.arange(0, MT.shape[1], EPSILON),
                               axis=1)
    # 沿第2轴合并EPSILON个单元
    MT_BOX_2 = np.add.reduceat(MT_BOX_1,
                               np.arange(0, MT.shape[2], EPSILON),
                               axis=2)
    # 计数非空盒子: 值 > 0 且 <= EPSILON^3 
    return len(
        np.where((MT_BOX_2 > 0) & (MT_BOX_2 <= EPSILON ** 3))[0]
    )


def _count_boxes_3d_advanced(MT: np.ndarray, EPSILON: int,
                             strategy: str = 'fixed',
                             n_random: int = 5,
                             sliding_step: float = 0.5) -> float:
    """
    3D盒计数（支持多种策略）
    
    Parameters
    ----------
    MT : np.ndarray
        三维数组
    EPSILON : int
        盒子大小
    strategy : str
        划分策略
    n_random : int
        随机策略的采样次数
    sliding_step : float
        滑动步长因子
        
    Returns
    -------
    count : float
        有效盒子数量
    """
    if strategy == 'fixed':
        return _count_boxes_3d(MT, EPSILON)
    
    elif strategy == 'sliding':
        # 滑动窗口策略
        step = max(1, int(EPSILON * sliding_step))
        count = 0
        boxes_checked = 0
        
        for i in range(0, MT.shape[0] - EPSILON + 1, step):
            for j in range(0, MT.shape[1] - EPSILON + 1, step):
                for k in range(0, MT.shape[2] - EPSILON + 1, step):
                    box = MT[i:i+EPSILON, j:j+EPSILON, k:k+EPSILON]
                    if np.sum(box) > 0:
                        count += 1
                    boxes_checked += 1
        
        # 归一化
        if boxes_checked > 0:
            overlap_factor = (step / EPSILON) ** 3
            return count * overlap_factor
        return count
    
    elif strategy == 'random':
        # 随机采样策略
        counts = []
        for _ in range(n_random):
            positions = _get_box_positions(MT.shape, EPSILON, 'random',
                                          n_random=1, sliding_step=sliding_step)
            count = 0
            for pos in positions:
                i, j, k = pos
                box = MT[i:i+EPSILON, j:j+EPSILON, k:k+EPSILON]
                if np.sum(box) > 0:
                    count += 1
            
            total_boxes_possible = (MT.shape[0] // EPSILON) * \
                                   (MT.shape[1] // EPSILON) * \
                                   (MT.shape[2] // EPSILON)
            counts.append(count * total_boxes_possible)
        
        return np.mean(counts)
    
    else:
        return _count_boxes_3d(MT, EPSILON)

