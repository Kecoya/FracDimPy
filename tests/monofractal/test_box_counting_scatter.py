#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Box-counting Method for Scatter Data Tests
==========================================

Test suite for the box-counting method to calculate
the fractal dimension of 1D scatter data.

The scatter data can be:
1. Binary data: 0/1 values [0,1,0,0,1,1,0,...]
2. Continuous values: [1.5, 3.2, 7.8, ...]

The box-counting method works by dividing the data space into boxes of
different sizes and counting how many boxes contain at least one data point.
The fractal dimension is estimated from the scaling relationship between
box size and the number of occupied boxes.
"""

import numpy as np
import pandas as pd
import os
import pytest
from fracDimPy import box_counting

# Data file path - use robust path construction
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "box_counting_scatter_data.xlsx")


def test_box_counting_scatter_basic():
    """Test basic box-counting functionality for scatter data."""
    # Load data
    df = pd.read_excel(data_file, header=None)

    # Extract scatter data from first column
    scatter_data = df.iloc[:, 0].values

    # Basic data validation
    assert len(scatter_data) > 0, "Scatter data should not be empty"
    assert np.isfinite(scatter_data).all(), "Scatter data should contain only finite values"

    # Calculate fractal dimension using box-counting
    D, result = box_counting(scatter_data, data_type='scatter')

    # Basic result validation
    assert isinstance(D, (float, np.floating)), "Fractal dimension should be a number"
    assert isinstance(result, dict), "Result should be a dictionary"
    assert 0 < D < 2, f"Fractal dimension {D} should be in reasonable range for 1D scatter data"

    # Check for required result keys
    required_keys = ['R2']
    for key in required_keys:
        assert key in result, f"Result should contain '{key}' key"


def test_box_counting_scatter_goodness_of_fit():
    """Test that the box-counting method provides good fit quality for scatter data."""
    # Load data
    df = pd.read_excel(data_file, header=None)
    scatter_data = df.iloc[:, 0].values

    # Calculate fractal dimension
    D, result = box_counting(scatter_data, data_type='scatter')

    # Check goodness of fit - should be reasonably high for good data
    r_squared = result['R2']
    assert r_squared > 0.7, f"R² should be > 0.7 for good fit, got {r_squared}"
    assert r_squared <= 1.0, f"R² should not exceed 1.0, got {r_squared}"


def test_box_counting_scatter_consistency():
    """Test that box-counting produces consistent results for scatter data."""
    # Load data
    df = pd.read_excel(data_file, header=None)
    scatter_data = df.iloc[:, 0].values

    # Calculate fractal dimension multiple times
    results = []
    for _ in range(3):
        D, result = box_counting(scatter_data, data_type='scatter')
        results.append(D)

    # Results should be consistent (small variation due to potential randomness)
    std_dev = np.std(results)
    mean_D = np.mean(results)

    # Relative standard deviation should be small
    relative_std = std_dev / mean_D if mean_D != 0 else std_dev
    assert relative_std < 0.01, f"Results should be consistent, relative std: {relative_std}"


def test_box_counting_scatter_data_types():
    """Test box-counting with different types of scatter data."""
    # Load real data first
    df = pd.read_excel(data_file, header=None)
    real_data = df.iloc[:, 0].values

    # Test with different data types
    test_cases = [
        # Original real data
        real_data,
        # Binary data
        np.random.choice([0, 1], size=1000, p=[0.7, 0.3]),
        # Continuous random data
        np.random.randn(1000),
        # Uniform data
        np.random.uniform(0, 10, 1000),
    ]

    results = []
    for i, test_data in enumerate(test_cases):
        D, result = box_counting(test_data, data_type='scatter')
        results.append(D)

        # Each calculation should produce valid results
        assert isinstance(D, (float, np.floating)), f"Test case {i} should produce numeric result"
        assert isinstance(result, dict), f"Test case {i} should produce dictionary result"
        assert 0 < D < 2, f"Test case {i}: Fractal dimension {D} should be between 0 and 2 for 1D data"
        assert result['R2'] > 0.5, f"Test case {i}: R² should be reasonable"


def test_box_counting_scatter_different_parameters():
    """Test box-counting with different parameter combinations for scatter data."""
    # Load data
    df = pd.read_excel(data_file, header=None)
    scatter_data = df.iloc[:, 0].values

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
        D, result = box_counting(scatter_data, data_type='scatter', **params)
        results.append(D)

        # Each calculation should produce valid results
        assert isinstance(D, (float, np.floating)), f"Result with params {params} should be numeric"
        assert 0 < D < 2, f"Fractal dimension {D} should be in reasonable range for 1D scatter data"
        assert result['R2'] > 0.5, f"R² should be reasonable for params {params}"

    # Results should be relatively consistent across parameter variations
    mean_D = np.mean(results)
    std_D = np.std(results)

    # Allow for some variation but not too much
    assert std_D / mean_D < 0.2, f"Results should be relatively stable across parameters"


def test_box_counting_scatter_theoretical_bounds():
    """Test that fractal dimension is within theoretical bounds for 1D scatter data."""
    # Load data
    df = pd.read_excel(data_file, header=None)
    scatter_data = df.iloc[:, 0].values

    # Calculate fractal dimension
    D, result = box_counting(scatter_data, data_type='scatter')

    # For 1D scatter data, fractal dimension should be between 0 (isolated points) and 1 (continuous)
    # We use a slightly broader range to account for noise and discrete sampling
    assert 0 < D < 1.5, f"Fractal dimension {D} should be between 0 and 1.5 for 1D scatter data"


def test_box_counting_scatter_scaling_invariance():
    """Test that box-counting is scale-invariant for scatter data."""
    # Load data
    df = pd.read_excel(data_file, header=None)
    scatter_data = df.iloc[:, 0].values

    # Test scaling invariance
    scales = [0.5, 1.0, 2.0, 5.0]
    results = []

    for scale in scales:
        scaled_data = scatter_data * scale
        D, result = box_counting(scaled_data, data_type='scatter')
        results.append(D)

        # Each calculation should produce valid results
        assert isinstance(D, (float, np.floating)), f"Scaled data (scale={scale}) should produce numeric result"
        assert 0 < D < 2, f"Scaled data: Fractal dimension {D} should be between 0 and 2"
        assert result['R2'] > 0.5, f"Scaled data: R² should be reasonable"

    # Results should be very consistent across scales (fractal dimension is scale-invariant)
    std_D = np.std(results)
    mean_D = np.mean(results)
    relative_std = std_D / mean_D if mean_D != 0 else std_D

    assert relative_std < 0.05, f"Fractal dimension should be scale-invariant, relative std: {relative_std}"
