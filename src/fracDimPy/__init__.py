#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FracDimPy - 
==============================

Python
- 
-   
- 

Author: Zhile Han
Link: https://www.zhihu.com/people/xiao-xue-sheng-ye-xiang-xie-shu/posts
"""

__version__ = "0.1.1"
__author__ = "Zhile Han"

# 
from . import monofractal
from . import multifractal
from . import generator
from . import utils

# 
from .monofractal import (
    hurst_dimension,
    structural_function,
    variogram_method,
    box_counting,
    sandbox_method,
    information_dimension,
    correlation_dimension,
    dfa
)

from .multifractal import (
    multifractal_curve,
    multifractal_image,
    mf_dfa
)

from .generator import (
    # 
    generate_fbm_curve,
    generate_wm_curve,
    generate_takagi_curve,
    # 
    generate_fbm_surface,
    generate_wm_surface,
    generate_takagi_surface,
    # 
    generate_cantor_set,
    generate_sierpinski,
    generate_sierpinski_carpet,
    generate_vicsek_fractal,
    generate_koch_curve,
    generate_koch_snowflake,
    generate_brownian_motion,
    generate_levy_flight,
    generate_self_avoiding_walk,
    generate_dla,
    generate_menger_sponge
)

__all__ = [
    # 
    'hurst_dimension', 
    'structural_function',
    'variogram_method',
    'box_counting',
    'sandbox_method',
    'information_dimension',
    'correlation_dimension',
    'dfa',
    # 
    'multifractal_curve',
    'multifractal_image',
    'mf_dfa',
    #  - 
    'generate_fbm_curve',
    'generate_wm_curve',
    'generate_takagi_curve',
    #  - 
    'generate_fbm_surface',
    'generate_wm_surface',
    'generate_takagi_surface',
    #  - 
    'generate_cantor_set',
    'generate_sierpinski',
    'generate_sierpinski_carpet',
    'generate_vicsek_fractal',
    'generate_koch_curve',
    'generate_koch_snowflake',
    'generate_brownian_motion',
    'generate_levy_flight',
    'generate_self_avoiding_walk',
    'generate_dla',
    'generate_menger_sponge',
    # 
    'monofractal',
    'multifractal',
    'generator',
    'utils',
]

