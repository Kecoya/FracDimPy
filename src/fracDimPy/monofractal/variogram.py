#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Variogram Method
================

Variogram (semi-variogram) analysis for fractal dimension estimation.

Definition: γ(h) = E[(Z(x+h) - Z(x))²]

The variogram scales as γ(h) ~ h^(2H) where H is the Hurst exponent.
Dimension: D = 2 - H (for curves) or D = 3 - H (for surfaces)
"""

import numpy as np
from typing import Tuple, Optional


def variogram_method(
    data: np.ndarray, rate: float = 0.05, dimension_type: str = "1d"
) -> Tuple[float, dict]:
    """


    Parameters
    ----------
    data : np.ndarray

    rate : float, optional
        0.550%lag
    dimension_type : str, optional
        '1d''surface''1d'

    Returns
    -------
    dimension : float

    result : dict

        - 'dimension':
        - 'hurst': Hurst
        - 'lag_values': h
        - 'variogram_values':
        - 'R2': R
        - 'coefficients':

    Examples
    --------
    >>> import numpy as np
    >>> from fracDimPy import variogram_method
    >>> #
    >>> t = np.random.randn(1000)
    >>> D, result = variogram_method(t)
    >>> print(f": {D:.4f}")
    >>> print(f"Hurst: {result['hurst']:.4f}")

    Notes
    -----

    """
    mt = data
    M = mt.shape[0]

    #
    Ekl = []
    #
    Kl = []

    for k in range(1, int(M * rate) + 1, 1):
        # k
        n = M - k
        COL = M - k
        COL_ = M

        #
        variogram_value = np.sum(np.square(mt[0:COL] - mt[k:COL_])) / n

        Ekl.append(variogram_value)
        Kl.append(k)

    #
    y = np.array(Ekl)
    x = np.array(Kl)

    log_x = np.log(x)
    log_y = np.log(y)

    coefficients = np.polyfit(log_x, log_y, 1)
    f = np.poly1d(coefficients)

    # Hurst
    # log((h)) = log(C) + 2H*log(h)
    #  = 2H
    # H = /2
    hurst_exponent = coefficients[0] / 2

    # Variogram_method_1d_meansquare.py
    # D = 2 - H
    # D = 3 - H
    if dimension_type == "1d":
        dimension = 2 - hurst_exponent
    else:  # surface
        dimension = 3 - hurst_exponent

    #
    R2 = np.corrcoef(log_y, f(log_x))[0, 1] ** 2

    result = {
        "dimension": dimension,
        "hurst": hurst_exponent,
        "lag_values": Kl,
        "variogram_values": Ekl,
        "log_lag": log_x,
        "log_variogram": log_y,
        "slope": coefficients[0],
        "R2": R2,
        "coefficients": coefficients,
        "method": "Variogram",
        "dimension_type": dimension_type,
    }

    return dimension, result
