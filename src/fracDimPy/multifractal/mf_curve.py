#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multifractal Analysis for 1D Curves
====================================

Implements partition function method for multifractal analysis of curves.

Computes:
- Mass exponent τ(q)
- Hölder exponent α(q)
- Multifractal spectrum f(α)
- Generalized dimensions D(q)
"""

import numpy as np
from numpy import polyfit
import multiprocessing
from typing import Tuple, List, Optional, Dict, Union
from .custom_epsilon import custom_epsilon, is_power_of_two


def multifractal_curve(
    data: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    q_list: Optional[List[float]] = None,
    epsilon_grid: Optional[List[int]] = None,
    epsilonl: Optional[List[float]] = None,
    use_multiprocessing: bool = True,
    data_type: str = 'single',
    interpolate_to_power2: bool = True,
    remove_zeros: bool = True
) -> Tuple[dict, dict]:
    """
    
    
    Parameters
    ----------
    data : np.ndarray or tuple
        - data_type='single': (np.ndarray)
        - data_type='dual': (x, y) 
    q_list : list of float, optional
        q[-10, -9, ..., 9, 10]
    epsilon_grid : list of int, optional
        [2, 4, 8, ..., len(data)//2]
    epsilonl : list of float, optional
        None
    use_multiprocessing : bool, optional
        True
    data_type : str, optional
        'single'()  'dual'(XY)'single'
    interpolate_to_power2 : bool, optional
        2True
    remove_zeros : bool, optional
        0dualTrue
        
    Returns
    -------
    metrics : dict
        
        - '(q=0)', '(q=1)', '(q=2)': q
        - 'f(q=0)', 'f(q=1)', 'f(q=2)': q
        - 'D(0)', 'D(1)', 'D(2)': 
        - 'H': Hurst
        - '', '', '': 
    figure_data : dict
        
        - 'q': q
        - '(q)': 
        - '(q)': 
        - 'f()': 
        - 'D(q)': 
        
    Examples
    --------
    >>> import numpy as np
    >>> from fracDimPy import multifractal_curve
    >>> # 
    >>> data = np.random.randn(1024)
    >>> metrics, figure_data = multifractal_curve(data, data_type='single')
    >>> print(f" D(0): {metrics[' D(0)'][0]:.4f}")
    >>> print(f" D(1): {metrics[' D(1)'][0]:.4f}")
    >>> 
    >>> # XY
    >>> x = np.linspace(0, 10, 1024)
    >>> y = np.random.randn(1024)
    >>> metrics, figure_data = multifractal_curve((x, y), data_type='dual')
    >>> print(f" D(0): {metrics[' D(0)'][0]:.4f}")
    
    Notes
    -----
    
    -f
    
    data_type
    - 'single': 
    - 'dual': XY
    """
    # 
    if data_type == 'dual':
        if isinstance(data, tuple) and len(data) == 2:
            x_data, y_data = data
            
            print(f': {len(y_data)}')
            
            # epsilonRun_CustomEpsilon.py
            # 2
            if remove_zeros:
                # 0
                non_zero_indices = np.where(y_data != 0)[0]
                current_len = len(non_zero_indices)
            else:
                current_len = len(y_data)
            
            if interpolate_to_power2:
                if is_power_of_two(current_len):
                    target_length = current_len
                else:
                    target_length = 2 ** (int(np.log2(current_len)) + 1)
            else:
                target_length = None
            
            # epsilon
            epsilon_physical, mt, x_grid = custom_epsilon(
                x_data, y_data,
                interpolate_to_power2=interpolate_to_power2,
                target_length=target_length,
                remove_zeros=remove_zeros
            )
            
            if interpolate_to_power2:
                print(f': {len(mt)} (2)')
            
        else:
            raise ValueError(" (x, y) ")
    else:
        mt = data
        epsilon_physical = 1.0
    
    # qMF_BC1D.py
    if q_list is None:
        q_min = -10
        q_max = 10
        # 10000,1,2
        q_list = np.unique(np.append(
            np.round(np.linspace(q_min, q_max, 1000), 2),
            [0, 1, 2]
        ))
        q_list = q_list.tolist()
    
    # Run_CustomEpsilon.pyepsilonl_grid
    if epsilon_grid is None:
        # epsilon_grid = [2 ** i for i in range(1, int(np.log(len(mt)) / np.log(2)) + 1)]
        epsilon_grid = [2 ** i for i in range(1, int(np.log2(len(mt))) + 1)]
    
    # epsilon_grid
    if epsilonl is None and data_type == 'dual':
        epsilonl = [2 ** i * epsilon_physical for i in range(1, int(np.log2(len(mt))) + 1)]
    elif epsilonl is None:
        epsilonl = epsilon_grid
    
    print(f'{epsilon_grid}')
    print(f'q: {q_list}')
    
    q_min = min(q_list)
    q_max = max(q_list)
    
    # 
    xl = []   # 
    tl = []   # 
    al = []   # 
    fl = []   # 
    dl = []   # 
    
    Pill = []
    
    # 
    if use_multiprocessing and __name__ != '__main__':
        try:
            pool = multiprocessing.Pool()
            result_list = []
            for epsilon in epsilon_grid:
                PACK = (epsilon, mt)
                result_list.append(pool.apply_async(_compute_probability, (PACK,)))
            pool.close()
            pool.join()
            for item_return in result_list:
                Pill.append(item_return.get())
        except:
            # 
            use_multiprocessing = False
    
    if not use_multiprocessing:
        for epsilon in epsilon_grid:
            PACK = (epsilon, mt)
            Pill.append(_compute_probability(PACK))
    
    # q
    for q in q_list:
        xl_t = []
        xl_a = []
        xl_d = []
        
        for Pil in Pill:
            # 
            temp = np.power(Pil, q)
            X_t = np.sum(temp)
            xl_t.append(X_t)
            xl_a.append(np.sum(temp / X_t * np.log(Pil)))
            
            if q == 1:
                xl_d.append(np.sum(Pil * np.log(Pil)))
        
        #  (q)MF_BC1D.py
        t = polyfit(np.log(epsilonl), np.log(xl_t), 1)[0]
        X = [np.log(epsilonl), np.log(xl_t), q]
        
        #  (q)
        a = polyfit(np.log(epsilonl), xl_a, 1)[0]
        
        #  f()
        f = q * a - t
        
        #  D(q)
        if q == 1:
            D = polyfit(np.log(epsilonl), xl_d, 1)[0]
        else:
            D = t / (q - 1)
        
        tl.append(t)
        xl.append(X)
        al.append(a)
        fl.append(f)
        dl.append(D)
    
    # 
    al = list(al)
    q_list = list(q_list)
    dl = list(dl)
    
    coeff = polyfit(al, fl, 2)
    print(f'f- : f = {coeff[0]:.4f} + {coeff[1]:.4f} + {coeff[2]:.4f}')
    
    # 
    W = max(al) - min(al)
    W_l = al[q_list.index(0)] - min(al)
    W_r = max(al) - al[q_list.index(0)]
    
    # 
    metrics = {
        # 
        'lenth': [np.sum(mt)],
        'mean': [np.mean(mt)],
        # 
        '(q=0)': [al[q_list.index(0)]],
        '(q=1)': [al[q_list.index(1)]],
        '(q=2)': [al[q_list.index(2)]],
        f'(q={q_min})': [al[q_list.index(q_min)]],
        f'(q={q_max})': [al[q_list.index(q_max)]],
        f'(q={q_min}) - (q=0)': [al[q_list.index(q_min)] - al[q_list.index(0)]],
        f'(q=0) - (q={q_max})': [al[q_list.index(0)] - al[q_list.index(q_max)]],
        f'(q={q_min}) - (q={q_max})': [al[q_list.index(q_min)] - al[q_list.index(q_max)]],
        # 
        '': [coeff[0]],
        '': [coeff[1]],
        '': [coeff[2]],
        'f(q=0)': [fl[q_list.index(0)]],
        'f(q=1)': [fl[q_list.index(1)]],
        'f(q=2)': [fl[q_list.index(2)]],
        f'f(q={q_min})': [fl[q_list.index(q_min)]],
        f'f(q={q_max})': [fl[q_list.index(q_max)]],
        '': [W_l],
        '': [W_r],
        '': [W],
        # 
        'H': [(1 + dl[q_list.index(2)]) / 2],
        ' D(0)': [dl[q_list.index(0)]],
        ' D(1)': [dl[q_list.index(1)]],
        ' D(2)': [dl[q_list.index(2)]],
        'D(0)-D(1)': [dl[q_list.index(0)] - dl[q_list.index(1)]],
        f'D({q_min})': [dl[q_list.index(q_min)]],
        f'D({q_max})': [dl[q_list.index(q_max)]],
        f'D({q_min})-D({q_max})': [dl[q_list.index(q_min)] - dl[q_list.index(q_max)]],
    }
    
    figure_data = {
        'q': q_list,
        '(q)': tl,
        '(q)': al,
        'f()': fl,
        'D(q)': dl,
    }
    
    # 
    temp_q_n = max(1, int(len(q_list) / 20))
    for i, item in enumerate(q_list):
        if i != 0 and i % temp_q_n == 0:
            figure_data.update({
                f'q={item}_X': list(xl[i][1]),
                f'q={item}_r': list(xl[i][0]),
            })
    
    return metrics, figure_data


def _compute_probability(PACK: Tuple) -> np.ndarray:
    """
    
    
    Parameters
    ----------
    PACK : tuple
        (epsilon, data) 
        
    Returns
    -------
    Pil : np.ndarray
        
    """
    epsilon, mt = PACK
    
    # 
    temp_mt = _box_counting_1d(mt, epsilon)
    
    # 0
    temp_mt = np.delete(temp_mt, np.where(temp_mt == 0)[0])
    
    # 
    N_sum = np.sum(temp_mt)
    Pil = temp_mt / N_sum
    
    return Pil


def _box_counting_1d(MT: np.ndarray, EPSILON: int) -> np.ndarray:
    """
    
    
    Parameters
    ----------
    MT : np.ndarray
        
    EPSILON : int
        
        
    Returns
    -------
    boxes : np.ndarray
        
    """
    MT_BOX_0 = np.add.reduceat(
        MT,
        np.arange(0, MT.shape[0], EPSILON),
        axis=0
    )
    return MT_BOX_0

