#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Scale generation utilities for fractal analysis.

Provides shared functions for generating power-of-two epsilon/grid scales,
replacing duplicated scale-generation patterns across modules.
"""

import numpy as np


def power_of_two_scales(M):
    """Generate power-of-two scales up to M.

    Parameters
    ----------
    M : int or float
        Maximum value. Scales are [2, 4, 8, ...] up to M.

    Returns
    -------
    list of int
        Power-of-two scale values.
    """
    return [2 ** i for i in range(1, int(np.log2(M)) + 1)]


def power_of_two_scales_generator(M):
    """Generator version yielding power-of-two scales up to M.

    Parameters
    ----------
    M : int or float
        Maximum value.

    Yields
    ------
    int
        Each power-of-two scale value.
    """
    for i in range(1, int(np.log2(M)) + 1):
        yield 2 ** i
