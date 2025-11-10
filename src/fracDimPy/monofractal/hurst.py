#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hurst Exponent via R/S Analysis
================================

Rescaled Range (R/S) analysis for computing the Hurst exponent.

Fractal dimension D is related to Hurst exponent H by: D = 2 - H

Reference:
    Hurst, H. E. (1951). Long-term storage capacity of reservoirs. 
    Transactions of the American Society of Civil Engineers, 116, 770-799.
"""

import numpy as np
from typing import Tuple, Optional


def hurst_dimension(
    timeseries: np.ndarray,
    method: str = 'rs'
) -> Tuple[float, dict]:
    """
    R/SHurst
    
    Parameters
    ----------
    timeseries : np.ndarray
        
    method : str, optional
        'rs' - R/S (Rescaled Range)
        
    Returns
    -------
    dimension : float
         (D = 2 - H)
    result : dict
        
        - 'dimension': 
        - 'hurst': Hurst
        - 'R2': R
        - 'r_values': r
        - 'rs_values': R/S
        - 'coefficients': 
        
    Examples
    --------
    >>> import numpy as np
    >>> from fracDimPy import hurst_dimension
    >>> # 
    >>> t = np.random.randn(1000)
    >>> D, result = hurst_dimension(t)
    >>> print(f"Hurst: {result['hurst']:.4f}")
    >>> print(f": {D:.4f}")
    """
    return _rs_method(timeseries)


def _rs_method(mt: np.ndarray) -> Tuple[float, dict]:
    """
    R/S分析计算Hurst指数
    
    Parameters
    ----------
    mt : np.ndarray
        时间序列数据
        
    Returns
    -------
    dimension : float
        分形维数
    result : dict
        包含详细结果的字典
    """
    N = mt.shape[0]
    ars = []  # R/S值列表
    rl = []   # 窗口尺寸列表
    
    # 使用对数尺度选择窗口尺寸，更合理
    min_window = 10
    max_window = N // 4
    num_windows = min(50, max_window - min_window)
    
    # 对数均匀分布的窗口尺寸
    Rl = np.unique(np.logspace(
        np.log10(min_window),
        np.log10(max_window),
        num_windows
    ).astype(int))
    
    for r in Rl:
        if r < 4:  # 跳过太小的窗口
            continue
            
        # 计算这个尺度下有多少完整的段
        num_segments = N // r
        if num_segments < 2:  # 至少需要2个段
            continue
        
        rs_values = []
        
        # 对每个段分别计算R/S
        for i in range(num_segments):
            segment = mt[i*r : (i+1)*r]
            
            if len(segment) < 4:  # 段太短，跳过
                continue
            
            # 1. 计算段的均值
            mean_seg = np.mean(segment)
            
            # 2. 计算偏离均值的累积和
            Y = np.cumsum(segment - mean_seg)
            
            # 3. 计算极差R
            R = np.max(Y) - np.min(Y)
            
            # 4. 计算标准差S
            S = np.std(segment, ddof=1)
            
            # 5. 计算R/S（避免除以0）
            if S > 1e-10 and R > 0:
                rs_values.append(R / S)
        
        # 对所有段的R/S取平均
        if len(rs_values) > 0:
            RS = np.mean(rs_values)
            
            if not (np.isnan(RS) or np.isinf(RS)) and RS > 0:
                ars.append(RS)
                rl.append(r)
    
    if len(rl) < 3:
        raise ValueError("R/S分析失败：有效数据点不足，请提供更长的时间序列")
    
    # 在log-log空间进行线性拟合以估计Hurst指数
    x = np.log10(rl)
    y = np.log10(ars)
    coeff = np.polyfit(x, y, 1)
    f = np.poly1d(coeff)
    
    hurst_exponent = coeff[0]
    dimension = 2 - hurst_exponent
    
    # 计算拟合优度
    y_fit = f(x)
    ss_res = np.sum((y - y_fit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    result = {
        'dimension': dimension,
        'hurst': hurst_exponent,
        'R2': R2,
        'r_values': rl,
        'rs_values': ars,
        'log_r': x,
        'log_rs': y,
        'coefficients': coeff,
        'method': 'R/S Analysis'
    }
    
    return dimension, result



