#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

==============


- : FBM, Weierstrass-Mandelbrot, Takagi
- : FBM, WM, Takagi
- : Cantor, Sierpinski, DLA, Menger
"""

from .curves import (
    generate_fbm_curve,
    generate_wm_curve,
    generate_takagi_curve
)

from .surfaces import (
    generate_fbm_surface,
    generate_wm_surface,
    generate_takagi_surface
)

from .patterns import (
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
    'generate_fbm_curve',
    'generate_wm_curve',
    'generate_takagi_curve',
    # 
    'generate_fbm_surface',
    'generate_wm_surface',
    'generate_takagi_surface',
    # 
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
]

