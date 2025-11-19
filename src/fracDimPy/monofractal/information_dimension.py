#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Information Dimension
=====================

Calculates information dimension using Shannon entropy.

The information dimension is based on the box-counting method with 
probability weighting via Shannon entropy.
"""

import numpy as np
from typing import Tuple, Dict, Optional


def information_dimension(
    data: np.ndarray, min_boxes: int = 4, max_boxes: Optional[int] = None, num_points: int = 10
) -> Tuple[float, Dict]:
    """
    Information Dimension

    Parameters
    ----------
    data : np.ndarray
        1D2D
        - 1D: (N,)  (N, 1)
        - 2D: (H, W)  (N, 2)
    min_boxes : int, optional
        4
    max_boxes : int, optional

    num_points : int, optional
        log-log10

    Returns
    -------
    D_info : float

    result : dict

        - 'dimension':
        - 'box_sizes':  ()
        - 'information':  I()
        - 'log_inv_epsilon': log(1/)
        - 'coeffs':  [=D_I, ]
        - 'r_squared':

    Examples
    --------
    >>> import numpy as np
    >>> from fracDimPy import information_dimension
    >>> # Cantor
    >>> from fracDimPy import generate_cantor_set
    >>> cantor = generate_cantor_set(level=6)
    >>> D, result = information_dimension(cantor)
    >>> print(f": {D:.4f}")

    Notes
    -----
    Shannon
    I() = - p_i * log(p_i)
     p_i i


    D_I = lim(0) I() / log(1/)


    -
    -
    - D_I  D_0 ()
    """
    #
    if data.ndim == 1:
        # 1D
        data = data.flatten()
        points = np.column_stack([np.arange(len(data)), data])
        #
        if np.all(data >= 0) and np.all(data <= 1):
            points = points[data > 0]
    elif data.ndim == 2:
        if data.shape[1] == 2:
            #
            points = data
        else:
            # 2D
            y, x = np.where(data > 0)
            points = np.column_stack([x, y])
    else:
        raise ValueError("1D2D")

    if len(points) == 0:
        raise ValueError("")

    #  [0, 1]
    points_min = points.min(axis=0)
    points_max = points.max(axis=0)
    points_range = points_max - points_min
    points_range[points_range == 0] = 1  # 0
    points_norm = (points - points_min) / points_range

    #
    if max_boxes is None:
        max_boxes = min(len(points) // 10, 100)

    #
    box_counts = np.unique(
        np.logspace(np.log10(min_boxes), np.log10(max_boxes), num_points).astype(int)
    )

    box_sizes = []
    informations = []

    for n_boxes in box_counts:
        #
        box_size = 1.0 / n_boxes

        #
        box_indices = (points_norm / box_size).astype(int)
        box_indices = np.clip(box_indices, 0, n_boxes - 1)

        #
        if points.shape[1] == 2:
            # 2D
            box_id = box_indices[:, 0] * n_boxes + box_indices[:, 1]
        else:
            # 1D
            box_id = box_indices[:, 0]

        #
        unique_boxes, counts = np.unique(box_id, return_counts=True)

        #
        probabilities = counts / len(points)

        #
        information = -np.sum(probabilities * np.log(probabilities))

        box_sizes.append(box_size)
        informations.append(information)

    box_sizes = np.array(box_sizes)
    informations = np.array(informations)

    #
    valid = (informations > 0) & (box_sizes > 0)
    box_sizes = box_sizes[valid]
    informations = informations[valid]

    if len(box_sizes) < 3:
        raise ValueError("")

    # I() vs log(1/)
    # log(1/) = -log()
    log_inv_epsilon = -np.log(box_sizes)  # log(1/)

    # : I() = D_I * log(1/) + const
    #
    coeffs = np.polyfit(log_inv_epsilon, informations, 1)
    D_info = coeffs[0]  #

    #
    fit_values = np.polyval(coeffs, log_inv_epsilon)
    ss_res = np.sum((informations - fit_values) ** 2)
    ss_tot = np.sum((informations - np.mean(informations)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    result = {
        "dimension": D_info,
        "box_sizes": box_sizes,
        "information": informations,
        "log_inv_epsilon": log_inv_epsilon,  # log(1/)
        "coeffs": coeffs,
        "r_squared": r_squared,
    }

    return D_info, result
