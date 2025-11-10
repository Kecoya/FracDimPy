#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detrended Fluctuation Analysis (DFA)
=====================================

DFA is used to detect long-range correlations in time series.

References:
    Peng, C. K., et al. (1994). Mosaic organization of DNA nucleotides. 
    Physical review e, 49(2), 1685.
"""

import numpy as np
from typing import Tuple, Dict, Optional


def dfa(
    data: np.ndarray,
    min_window: int = 10,
    max_window: Optional[int] = None,
    num_windows: int = 20,
    order: int = 1
) -> Tuple[float, Dict]:
    """
    Detrended Fluctuation Analysis, DFA
    
    Hurst
    
    Parameters
    ----------
    data : np.ndarray
        1D
    min_window : int, optional
        10
    max_window : int, optional
        1/4
    num_windows : int, optional
        20
    order : int, optional
        1
        - order=1: DFA1
        - order=2: DFA2
        
    Returns
    -------
    alpha : float
        Hurst = H
        -  < 0.5: 
        -  = 0.5: 
        -  > 0.5: 
        -  = 1.0: 1/f
        -  > 1.0: 
    result : dict
        
        - 'alpha': Hurst
        - 'dimension':  D = 2 - 
        - 'window_sizes': 
        - 'fluctuations':  F(n)
        - 'log_windows': log()
        - 'log_fluctuations': log(F(n))
        - 'coeffs':  [=, ]
        - 'r_squared': 
        
    Examples
    --------
    >>> import numpy as np
    >>> from fracDimPy import dfa
    >>> # 
    >>> H = 0.7  # Hurst
    >>> n = 10000
    >>> noise = np.random.randn(n)
    >>> fbm = np.cumsum(noise)
    >>> alpha, result = dfa(fbm)
    >>> print(f": {alpha:.4f}")
    >>> print(f": {result['dimension']:.4f}")
    
    Notes
    -----
    DFA
    1. Y(i) = [x(k) - <x>]
    2. Yn
    3. ordery_fit
    4. F(n) = (1/N) [Y(i) - y_fit(i)]
    5. nlog(F(n)) vs log(n)
    6. Hurst
    7. D = 2 - 1D
    
    References
    ----------
    Peng, C. K., et al. (1994). Mosaic organization of DNA nucleotides. 
    Physical review e, 49(2), 1685.
    
    Kantelhardt, J. W., et al. (2001). Detecting long-range correlations 
    with detrended fluctuation analysis. Physica A, 295(3-4), 441-454.
    """
    # 
    if data.ndim != 1:
        data = data.flatten()
    
    N = len(data)
    
    if N < 100:
        raise ValueError("100")
    
    # 1
    mean = np.mean(data)
    Y = np.cumsum(data - mean)
    
    # 
    if max_window is None:
        max_window = N // 4
    
    # 
    min_window = max(min_window, order + 2)
    max_window = min(max_window, N // 4)
    
    if min_window >= max_window:
        raise ValueError(f"min_window={min_window}, max_window={max_window}")
    
    # 
    window_sizes = np.unique(np.logspace(
        np.log10(min_window),
        np.log10(max_window),
        num_windows
    ).astype(int))
    
    fluctuations = []
    
    for n in window_sizes:
        # 2
        # 
        num_segments = N // n
        
        if num_segments == 0:
            continue
        
        # 
        segment_fluctuations = []
        
        # 
        for i in range(num_segments):
            start = i * n
            end = start + n
            segment = Y[start:end]
            
            # 3
            x = np.arange(n)
            coeffs = np.polyfit(x, segment, order)
            fit = np.polyval(coeffs, x)
            
            # 4
            detrended = segment - fit
            variance = np.mean(detrended ** 2)
            segment_fluctuations.append(variance)
        
        # 
        remainder = N % n
        if remainder > n // 2:  # 
            for i in range(num_segments):
                start = N - (i + 1) * n
                end = start + n
                if start >= 0:
                    segment = Y[start:end]
                    
                    x = np.arange(n)
                    coeffs = np.polyfit(x, segment, order)
                    fit = np.polyval(coeffs, x)
                    
                    detrended = segment - fit
                    variance = np.mean(detrended ** 2)
                    segment_fluctuations.append(variance)
        
        # 
        if len(segment_fluctuations) > 0:
            F_n = np.sqrt(np.mean(segment_fluctuations))
            fluctuations.append(F_n)
        else:
            fluctuations.append(np.nan)
    
    fluctuations = np.array(fluctuations)
    
    # 
    valid = np.isfinite(fluctuations) & (fluctuations > 0)
    window_sizes = window_sizes[valid]
    fluctuations = fluctuations[valid]
    
    if len(window_sizes) < 3:
        raise ValueError("DFA")
    
    # 5log-log
    log_n = np.log10(window_sizes)
    log_F = np.log10(fluctuations)
    
    # log(F) =  * log(n) + const
    coeffs = np.polyfit(log_n, log_F, 1)
    alpha = coeffs[0]  # Hurst
    
    # 
    fit_values = np.polyval(coeffs, log_n)
    ss_res = np.sum((log_F - fit_values) ** 2)
    ss_tot = np.sum((log_F - np.mean(log_F)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # 
    # 1DD = 2 - 
    dimension = 2 - alpha
    
    result = {
        'alpha': alpha,
        'dimension': dimension,
        'window_sizes': window_sizes,
        'fluctuations': fluctuations,
        'log_windows': log_n,
        'log_fluctuations': log_F,
        'coeffs': coeffs,
        'r_squared': r_squared
    }
    
    return alpha, result

