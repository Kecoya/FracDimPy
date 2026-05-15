#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility Module
==============

This module provides utility functions for data input/output, visualization,
and shared computation primitives to support the fractal analysis workflows.

Data I/O Functions:
- load_data: Load data from various file formats
- save_results: Save analysis results to different formats

Visualization Functions:
- plot_fractal_analysis: Create plots for monofractal analysis results
- plot_multifractal_spectrum: Generate multifractal spectrum visualizations

Shared Computation Utilities:
- fitting: Log-log linear regression and R-squared computation
- scales: Power-of-two scale generation
- box_counting_core: Dimension-agnostic box counting primitives
- multifractal_common: Shared multifractal partition/metrics computation
- image_drawing: Bresenham line drawing and coordinate normalization
- conversion: Coordinate-to-matrix, grayscale, boundary padding
"""

from .data_io import load_data, save_results
from .plotting import plot_fractal_analysis, plot_multifractal_spectrum

from .fitting import log_log_fit, linear_fit
from .scales import power_of_two_scales
from .box_counting_core import (
    count_boxes_fixed,
    count_nonempty,
    count_boxes_advanced,
)
from .multifractal_common import (
    default_q_list,
    compute_partition,
    build_metrics,
    build_figure_data,
)
from .image_drawing import bresenham_line, normalize_to_pixel_coords
from .conversion import (
    rgb_to_grayscale,
    coordinate_to_matrix_1d,
    coordinate_to_matrix_2d,
    apply_boundary_padding,
)

__all__ = [
    # Data I/O
    "load_data",
    "save_results",
    # Plotting
    "plot_fractal_analysis",
    "plot_multifractal_spectrum",
    # Fitting
    "log_log_fit",
    "linear_fit",
    # Scales
    "power_of_two_scales",
    # Box counting
    "count_boxes_fixed",
    "count_nonempty",
    "count_boxes_advanced",
    # Multifractal
    "default_q_list",
    "compute_partition",
    "build_metrics",
    "build_figure_data",
    # Image drawing
    "bresenham_line",
    "normalize_to_pixel_coords",
    # Conversion
    "rgb_to_grayscale",
    "coordinate_to_matrix_1d",
    "coordinate_to_matrix_2d",
    "apply_boundary_padding",
]
