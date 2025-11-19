#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility Module
==============

This module provides utility functions for data input/output and visualization
to support the fractal analysis workflows.

Data I/O Functions:
- load_data: Load data from various file formats (CSV, Excel, TXT, NPY, images)
- save_results: Save analysis results to different formats

Visualization Functions:
- plot_fractal_analysis: Create plots for monofractal analysis results
- plot_multifractal_spectrum: Generate multifractal spectrum visualizations

These utilities help streamline the data processing and result presentation
workflow for fractal analysis applications.
"""

from .data_io import load_data, save_results
from .plotting import plot_fractal_analysis, plot_multifractal_spectrum

__all__ = [
    'load_data',
    'save_results',
    'plot_fractal_analysis',
    'plot_multifractal_spectrum',
]

