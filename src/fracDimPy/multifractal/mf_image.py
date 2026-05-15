#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multifractal Analysis for 2D Images
====================================

Implements multifractal analysis for image data using partition function method.
"""

import numpy as np
from numpy import polyfit
from typing import Tuple, List, Optional

# type: ignore

from ..utils.multifractal_common import default_q_list, compute_partition, build_figure_data
from ..utils.scales import power_of_two_scales
from ..utils.box_counting_core import count_boxes_fixed


def multifractal_image(
    image: np.ndarray, q_list: Optional[List[float]] = None
) -> Tuple[dict, dict]:
    """


    Parameters
    ----------
    image : np.ndarray
        (H x W)
    q_list : list of float, optional
        q1000

    Returns
    -------
    metrics : dict

    figure_data : dict


    Examples
    --------
    >>> import numpy as np
    >>> from fracDimPy import multifractal_image
    >>> #
    >>> img = np.random.randint(0, 256, (256, 256))
    >>> metrics, figure_data = multifractal_image(img)
    >>> print(f" D(0): {metrics[' D(0)'][0]:.4f}")

    Notes
    -----

    """
    mt = image
    height, width = mt.shape
    print(f"height,width: {height}, {width}")

    # q0,1,2
    if q_list is None:
        q_min = -10
        q_max = 10
        q_list = default_q_list(q_min, q_max)

    q_min = min(q_list)  # type: ignore
    q_max = max(q_list)  # type: ignore
    print(f"q: {len(q_list)}, : [{q_min}, {q_max}]")

    #
    xl = []  #
    tl = []  #
    al = []  # Holder
    fl = []  #
    dl = []  #
    Pill = []  #

    #
    M = min(height, width)
    epsilonl = power_of_two_scales(M)
    print(f"{epsilonl}")

    #
    for epsilon in epsilonl:
        Pill.append(_compute_probability_image(mt, epsilon))

    # q
    tl, al, fl, dl, xl = compute_partition(q_list, Pill, epsilonl)

    al = list(al)
    q_list = list(q_list)
    dl = list(dl)

    #  f = a*^2 + b* + c
    coeff = polyfit(al, fl, 2)

    print(f"\nf- " f"\nf = {coeff[0]:.4f} + {coeff[1]:.4f} + {coeff[2]:.4f}")

    #
    W = max(al) - min(al)  #
    W_l = al[q_list.index(0)] - min(al)  #
    W_r = max(al) - al[q_list.index(0)]  #

    # metricsMFBC2D.py
    metrics = {
        # Holder
        "(q=0)": [al[q_list.index(0)]],
        "(q=1)": [al[q_list.index(1)]],
        "(q=2)": [al[q_list.index(2)]],
        f"(q={q_min})": [al[q_list.index(q_min)]],
        f"(q=+{q_max})": [al[q_list.index(q_max)]],
        f"(q={q_min}) - (q=0)": [al[q_list.index(q_min)] - al[q_list.index(0)]],
        f"(q=0) - (q=+{q_max})": [al[q_list.index(0)] - al[q_list.index(q_max)]],
        f"(q={q_min}) - (q=+{q_max})": [al[q_list.index(q_min)] - al[q_list.index(q_max)]],
        #
        "quad_coeff": [coeff[0]],
        "linear_coeff": [coeff[1]],
        "const_coeff": [coeff[2]],
        #
        "f(q=0)": [fl[q_list.index(0)]],
        "f(q=1)": [fl[q_list.index(1)]],
        "f(q=2)": [fl[q_list.index(2)]],
        f"f(q={q_min})": [fl[q_list.index(q_min)]],
        f"f(q=+{q_max})": [fl[q_list.index(q_max)]],
        "f(q=0)-f(q=1)": [fl[q_list.index(0)] - fl[q_list.index(1)]],
        f"f(q={q_min})-f(q=+{q_max})": [fl[q_list.index(q_min)] - fl[q_list.index(q_max)]],
        #
        "width_left": [W_l],
        "width_right": [W_r],
        "width_total": [W],
        #
        "H": [(1 + dl[q_list.index(2)]) / 2],
        " D(0)": [dl[q_list.index(0)]],
        " D(1)": [dl[q_list.index(1)]],
        " D(2)": [dl[q_list.index(2)]],
        "D(0)-D(1)": [dl[q_list.index(0)] - dl[q_list.index(1)]],
        f"D({q_min})": [dl[q_list.index(q_min)]],
        f"D(+{q_max})": [dl[q_list.index(q_max)]],
        f"D({q_min})-D(+{q_max})": [dl[q_list.index(q_min)] - dl[q_list.index(q_max)]],
    }

    #
    figure_data = build_figure_data(q_list, tl, al, fl, dl, xl)

    print("\n:")
    for key in [" D(0)", " D(1)", " D(2)", "H", "width_total", "width_left", "width_right"]:
        print(f"  {key}: {metrics[key][0]:.4f}")

    return metrics, figure_data


def _compute_probability_image(mt: np.ndarray, epsilon: int) -> np.ndarray:
    """


    Parameters
    ----------
    mt : np.ndarray

    epsilon : int


    Returns
    -------
    Pil : np.ndarray

    """
    #
    temp_mt = _box_counting_2d(mt, epsilon)
    temp_mt = temp_mt.flatten()

    #
    N_sum = np.sum(temp_mt)
    Pil = temp_mt / N_sum

    return Pil


def _box_counting_2d(MT: np.ndarray, EPSILON: int) -> np.ndarray:
    """Box counting for 2D data using shared utility."""
    return count_boxes_fixed(MT, EPSILON)
