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
        # 10000,1,2
        q_list = np.unique(
            np.append(np.round(np.linspace(q_min, q_max, 1000), 2), [0, 1, 2])
        ).tolist()

    q_min = min(q_list)
    q_max = max(q_list)
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
    epsilonl = [2**i for i in range(1, int(np.log(M) / np.log(2)) + 1)]
    print(f"{epsilonl}")

    #
    for epsilon in epsilonl:
        Pill.append(_compute_probability_image(mt, epsilon))

    # q
    for q in q_list:
        xl_t = []  #
        xl_a = []  #
        xl_d = []  # q=1

        for Pil in Pill:
            temp = np.power(Pil, q)
            X_t = np.sum(temp)
            xl_t.append(X_t)
            xl_a.append(np.sum(temp / X_t * np.log(Pil)))

            if q == 1:
                xl_d.append(np.sum(Pil * np.log(Pil)))

        #  (q)
        t = polyfit(np.log(epsilonl), np.log(xl_t), 1)[0]

        #
        X = [np.log(epsilonl), np.log(xl_t), q]

        #  (q) = d/dq
        a = polyfit(np.log(epsilonl), xl_a, 1)[0]

        #  f() = q -
        f = q * a - t

        #  D(q) = /(q-1), q1; D(1)
        if q == 1:
            D = polyfit(np.log(epsilonl), xl_d, 1)[0]
        else:
            D = t / (q - 1)

        tl.append(t)
        xl.append(X)
        al.append(a)
        fl.append(f)
        dl.append(D)

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
        "": [coeff[0]],
        "": [coeff[1]],
        "": [coeff[2]],
        #
        "f(q=0)": [fl[q_list.index(0)]],
        "f(q=1)": [fl[q_list.index(1)]],
        "f(q=2)": [fl[q_list.index(2)]],
        f"f(q={q_min})": [fl[q_list.index(q_min)]],
        f"f(q=+{q_max})": [fl[q_list.index(q_max)]],
        "f(q=0)-f(q=1)": [fl[q_list.index(0)] - fl[q_list.index(1)]],
        f"f(q={q_min})-f(q=+{q_max})": [fl[q_list.index(q_min)] - fl[q_list.index(q_max)]],
        #
        "": [W_l],
        "": [W_r],
        "": [W],
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
    figure_data = {
        "q": q_list,
        "(q)": tl,
        "(q)": al,
        "f()": fl,
        "D(q)": dl,
    }

    # 20q
    temp_q_n = max(1, int(len(q_list) / 20))
    for i, item in enumerate(q_list):
        if i != 0 and i % temp_q_n == 0:
            figure_data.update(
                {
                    f"q={item}_X": list(xl[i][1]),
                    f"q={item}_r": list(xl[i][0]),
                }
            )

    #
    print("\n:")
    for key in [" D(0)", " D(1)", " D(2)", "H", "", "", ""]:
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
    """


    Parameters
    ----------
    MT : np.ndarray

    EPSILON : int


    Returns
    -------
    boxes : np.ndarray

    """
    # EPSILON
    MT_BOX_0 = np.add.reduceat(MT, np.arange(0, MT.shape[0], EPSILON), axis=0)
    # EPSILON
    MT_BOX_1 = np.add.reduceat(MT_BOX_0, np.arange(0, MT.shape[1], EPSILON), axis=1)

    return MT_BOX_1
