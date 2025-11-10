#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multifractal Detrended Fluctuation Analysis (MF-DFA)
=====================================================

Extends DFA to multifractal analysis for detecting multi-scale
correlations and singularity structures.

Reference:
    Kantelhardt, J. W., et al. (2002). Multifractal detrended fluctuation 
    analysis of nonstationary time series. Physica A, 316(1-4), 87-114.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List


def mf_dfa(
    data: np.ndarray,
    q_list: Optional[List[float]] = None,
    min_window: int = 10,
    max_window: Optional[int] = None,
    num_windows: int = 20,
    order: int = 1
) -> Tuple[Dict, Dict]:
    """
    Multifractal DFA
    
    Hursth(q)f()
    
    Parameters
    ----------
    data : np.ndarray
        1D
    q_list : list of float, optional
        q[-10, 10]1000
        - q > 0: 
        - q < 0: 
        - q = 0: 
        - q = 2: DFA
    min_window : int, optional
        10
    max_window : int, optional
        1/4
    num_windows : int, optional
        20
    order : int, optional
        1
        
    Returns
    -------
    hq_result : dict
        Hurst
        - 'q_list': q
        - 'h_q': Hursth(q)
        - 'tau_q': (q) = qh(q) - 1
        - 'window_sizes': 
        - 'Fq_n': F_q(n) [len(q_list), len(window_sizes)]
    spectrum : dict
        
        - 'alpha': 
        - 'f_alpha': f()
        - 'alpha_0': f()
        - 'width':   = _max - _min
        
    Examples
    --------
    >>> import numpy as np
    >>> from fracDimPy import mf_dfa
    >>> # 
    >>> data = np.random.randn(10000)
    >>> hq_result, spectrum = mf_dfa(data)
    >>> print(f"h(2) = {hq_result['h_q'][hq_result['q_list'] == 2][0]:.4f}")
    >>> print(f"  = {spectrum['width']:.4f}")
    
    Notes
    -----
    MF-DFA
    1. Y(i) = [x(k) - <x>]
    2. Yn
    3. order
    4. q
       F_q(n) = {(1/N_s) [F()]^(q/2)}^(1/q)  for q  0
       F_0(n) = exp{(1/N_s)  ln[F()]}          for q = 0
    5. nlog(F_q(n)) vs log(n)
    6. Hursth(q)
    7. 
       (q) = qh(q) - 1
        = d/dq = h(q) + q*dh/dq
       f() = q - (q)
    
    
    - h(q) = q
    - h(q) q
    - 
    
    References
    ----------
    Kantelhardt, J. W., et al. (2002). Multifractal detrended fluctuation 
    analysis of nonstationary time series. Physica A, 316(1-4), 87-114.
    """
    # 
    if data.ndim != 1:
        data = data.flatten()
    
    N = len(data)
    
    if N < 100:
        raise ValueError("100")
    
    # qmf_curve.py
    if q_list is None:
        q_min = -10
        q_max = 10
        # 10000,1,2
        q_list = np.unique(np.append(
            np.round(np.linspace(q_min, q_max, 1000), 2),
            [0, 1, 2]
        ))
    else:
        q_list = np.array(q_list)
    
    # 1
    mean = np.mean(data)
    Y = np.cumsum(data - mean)
    
    # 
    if max_window is None:
        max_window = N // 4
    
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
    
    # 
    Fq_n = np.zeros((len(q_list), len(window_sizes)))
    
    # 
    for i_n, n in enumerate(window_sizes):
        # 2
        num_segments = N // n
        
        if num_segments == 0:
            continue
        
        # 
        F2_segments = []
        
        # 
        for v in range(num_segments):
            start = v * n
            end = start + n
            segment = Y[start:end]
            
            # 3
            x = np.arange(n)
            coeffs = np.polyfit(x, segment, order)
            fit = np.polyval(coeffs, x)
            
            # 
            detrended = segment - fit
            variance = np.mean(detrended ** 2)
            F2_segments.append(variance)
        
        # 
        remainder = N % n
        if remainder > n // 2:
            for v in range(num_segments):
                start = N - (v + 1) * n
                end = start + n
                if start >= 0:
                    segment = Y[start:end]
                    
                    x = np.arange(n)
                    coeffs = np.polyfit(x, segment, order)
                    fit = np.polyval(coeffs, x)
                    
                    detrended = segment - fit
                    variance = np.mean(detrended ** 2)
                    F2_segments.append(variance)
        
        F2_segments = np.array(F2_segments)
        
        # 4q
        for i_q, q in enumerate(q_list):
            if q == 0:
                # q=0: 
                # F_0(n) = exp{(1/N_s)  ln[F()]}
                log_F = np.log(F2_segments + 1e-10) / 2  # F = sqrt(F)
                Fq = np.exp(np.mean(log_F))
            else:
                # q0: 
                # F_q(n) = {(1/N_s) [F()]^(q/2)}^(1/q)
                Fq_power = np.mean(F2_segments ** (q / 2.0))
                if Fq_power > 0:
                    Fq = Fq_power ** (1.0 / q)
                else:
                    Fq = 0
            
            Fq_n[i_q, i_n] = Fq
    
    # 5h(q) - Hurst
    h_q = np.zeros(len(q_list))
    
    for i_q in range(len(q_list)):
        Fq = Fq_n[i_q, :]
        
        # 
        valid = (Fq > 0) & np.isfinite(Fq)
        if np.sum(valid) < 3:
            h_q[i_q] = np.nan
            continue
        
        log_n = np.log10(window_sizes[valid])
        log_Fq = np.log10(Fq[valid])
        
        # 
        coeffs = np.polyfit(log_n, log_Fq, 1)
        h_q[i_q] = coeffs[0]
    
    # 6(q)
    tau_q = q_list * h_q - 1
    
    # 7 f()
    # 
    alpha = np.zeros_like(q_list)
    f_alpha = np.zeros_like(q_list)
    
    # q
    for i in range(len(q_list)):
        if i == 0:
            # 
            dh_dq = (h_q[i+1] - h_q[i]) / (q_list[i+1] - q_list[i])
        elif i == len(q_list) - 1:
            # 
            dh_dq = (h_q[i] - h_q[i-1]) / (q_list[i] - q_list[i-1])
        else:
            # 
            dh_dq = (h_q[i+1] - h_q[i-1]) / (q_list[i+1] - q_list[i-1])
        
        #  = h(q) + q * dh/dq
        alpha[i] = h_q[i] + q_list[i] * dh_dq
        
        # f() = q *  - (q) = q *  - (q*h(q) - 1)
        f_alpha[i] = q_list[i] * alpha[i] - tau_q[i]
    
    # 
    valid_spectrum = np.isfinite(alpha) & np.isfinite(f_alpha)
    alpha_valid = alpha[valid_spectrum]
    f_alpha_valid = f_alpha[valid_spectrum]
    
    # 
    if len(alpha_valid) > 0:
        width = np.max(alpha_valid) - np.min(alpha_valid)
        # f()
        max_idx = np.argmax(f_alpha_valid)
        alpha_0 = alpha_valid[max_idx]
    else:
        width = 0
        alpha_0 = np.nan
    
    # 
    hq_result = {
        'q_list': q_list,
        'h_q': h_q,
        'tau_q': tau_q,
        'window_sizes': window_sizes,
        'Fq_n': Fq_n
    }
    
    spectrum = {
        'alpha': alpha,
        'f_alpha': f_alpha,
        'alpha_0': alpha_0,
        'width': width
    }
    
    return hq_result, spectrum

