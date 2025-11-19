#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Box-counting Method for Curve Data Tests
========================================

Test suite for the box-counting method to calculate
the fractal dimension of 2D curve data (X, Y coordinates).

The box-counting method works by covering the curve with boxes
of different sizes and counting how many boxes are needed.
"""

import numpy as np
import pandas as pd
import os
import pytest
from fracDimPy import box_counting

# Data file path - use robust path construction
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "box_counting_curve_data.xlsx")


def test_box_counting_curve_basic():
    """Test basic box-counting functionality for curve data."""
    # Load data
    df = pd.read_excel(data_file)

    # Extract X and Y coordinates
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values

    # Basic data validation
    assert len(x) > 0, "X data should not be empty"
    assert len(y) > 0, "Y data should not be empty"
    assert len(x) == len(y), "X and Y data should have same length"

    # Calculate fractal dimension using box-counting
    D, result = box_counting((x, y), data_type='curve')

    # Basic result validation
    assert isinstance(D, (float, np.floating)), "Fractal dimension should be a number"
    assert isinstance(result, dict), "Result should be a dictionary"
    assert 0 < D < 3, f"Fractal dimension {D} should be in reasonable range for 2D curve"

    # Check for required result keys
    required_keys = ['R2']
    for key in required_keys:
        assert key in result, f"Result should contain '{key}' key"


def test_box_counting_curve_goodness_of_fit():
    """Test that the box-counting method provides good fit quality."""
    # Load data
    df = pd.read_excel(data_file)
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values

    # Calculate fractal dimension
    D, result = box_counting((x, y), data_type='curve')

    # Check goodness of fit - should be reasonably high for good data
    r_squared = result['R2']
    assert r_squared > 0.8, f"R² should be > 0.8 for good fit, got {r_squared}"
    assert r_squared <= 1.0, f"R² should not exceed 1.0, got {r_squared}"


def test_box_counting_curve_consistency():
    """Test that box-counting produces consistent results."""
    # Load data
    df = pd.read_excel(data_file)
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values

    # Calculate fractal dimension multiple times
    results = []
    for _ in range(3):
        D, result = box_counting((x, y), data_type='curve')
        results.append(D)

    # Results should be consistent (small variation due to potential randomness)
    std_dev = np.std(results)
    mean_D = np.mean(results)

    # Relative standard deviation should be small
    relative_std = std_dev / mean_D if mean_D != 0 else std_dev
    assert relative_std < 0.01, f"Results should be consistent, relative std: {relative_std}"


def test_box_counting_curve_theoretical_range():
    """Test that fractal dimension is within theoretical bounds."""
    # Load data
    df = pd.read_excel(data_file)
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values

    # Calculate fractal dimension
    D, result = box_counting((x, y), data_type='curve')

    # For a 2D curve, fractal dimension should be between 1 (smooth line) and 2 (space-filling)
    # We use a slightly broader range to account for noise and discrete sampling
    assert 0.9 < D < 2.1, f"Fractal dimension {D} should be between 0.9 and 2.1 for 2D curve"


def test_box_counting_curve_different_parameters():
    """Test box-counting with different parameter combinations."""
    # Load data
    df = pd.read_excel(data_file)
    x = df.iloc[:, 0].values
    y = df.iloc[:, 1].values

    # Test with different parameter combinations
    parameter_sets = [
        {},  # default parameters
        {'min_box_size': 0.001},
        {'max_box_size': 1.0},
        {'num_boxes': 20},
        {'min_box_size': 0.001, 'max_box_size': 1.0, 'num_boxes': 25}
    ]

    results = []
    for params in parameter_sets:
        D, result = box_counting((x, y), data_type='curve', **params)
        results.append(D)

        # Each calculation should produce valid results
        assert isinstance(D, (float, np.floating)), f"Result with params {params} should be numeric"
        assert 0 < D < 3, f"Fractal dimension {D} should be in reasonable range"
        assert result['R2'] > 0.5, f"R² should be reasonable for params {params}"

    # Results should be relatively consistent across parameter variations
    mean_D = np.mean(results)
    std_D = np.std(results)

    # Allow for some variation but not too much
    assert std_D / mean_D < 0.2, f"Results should be relatively stable across parameters"
