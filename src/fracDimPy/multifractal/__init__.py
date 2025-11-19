#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multifractal Analysis Module
============================

This module provides comprehensive tools for multifractal analysis of
1D curves, 2D images, and time series data. Multifractal analysis extends
monofractal analysis by characterizing the full spectrum of scaling exponents
present in complex systems.

Main Functions:
- multifractal_curve: Multifractal analysis of 1D curves/series
- multifactal_image: Multifractal analysis of 2D images
- mf_dfa: Multifractal Detrended Fluctuation Analysis for non-stationary time series

Utility Functions:
- custom_epsilon: Custom scaling sequences for optimized analysis
- advise_mtepsilon: Automatic scaling sequence recommendation
- coordinate_to_matrix: Convert point coordinates to matrix format
- fill_vacancy: Fill missing data in matrices
- is_power_of_two: Check if number is power of two

The module supports various multifractal methods including:
- Partition function analysis
- Singularity spectrum calculation
- Generalized dimension estimation
- Hurst exponent spectrum
"""

from .mf_curve import multifractal_curve
from .mf_image import multifractal_image
from .mf_dfa import mf_dfa
from .custom_epsilon import (
    custom_epsilon,
    advise_mtepsilon,
    coordinate_to_matrix,
    fill_vacancy,
    is_power_of_two
)

__all__ = [
    'multifractal_curve',
    'multifractal_image',
    'mf_dfa',
    'custom_epsilon',
    'advise_mtepsilon',
    'coordinate_to_matrix',
    'fill_vacancy',
    'is_power_of_two',
]

