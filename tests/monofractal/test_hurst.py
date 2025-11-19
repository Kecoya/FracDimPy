#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Hurst Exponent (R/S Analysis) Tests
===================================

Test suite for fracDimPy's Hurst exponent method (R/S analysis)
to estimate the fractal dimension of time series data.
The Hurst exponent reflects the long-range correlation and memory effects of time series.

Theoretical Background:
- Hurst exponent H ∈ (0, 1)
- H < 0.5: Anti-persistent (mean-reverting)
- H = 0.5: Random walk (no memory)
- H > 0.5: Persistent (trend-following)
- Fractal dimension D = 2 - H (for 1D signals)
"""

import numpy as np
import os
import pytest
from fracDimPy import hurst_dimension

# Data file path - use robust path construction
current_dir = os.path.dirname(os.path.abspath(__file__))
data_file = os.path.join(current_dir, "hurst_data.npy")


def test_hurst_basic():
    """Test basic Hurst exponent functionality."""
    # Load test data
    data = np.load(data_file)

    # Basic data validation
    assert len(data) > 0, "Data should not be empty"
    assert np.isfinite(data).all(), "Data should contain only finite values"

    # Calculate Hurst exponent and fractal dimension
    D, result = hurst_dimension(data)

    # Basic result validation
    assert isinstance(D, (float, np.floating)), "Fractal dimension should be a number"
    assert isinstance(result, dict), "Result should be a dictionary"
    assert 1 < D < 2, f"Fractal dimension {D} should be between 1 and 2 for 1D signal"

    # Check for required result keys
    required_keys = ['hurst', 'R2']
    for key in required_keys:
        assert key in result, f"Result should contain '{key}' key"


def test_hurst_theoretical_relationships():
    """Test theoretical relationships between Hurst exponent and fractal dimension."""
    # Load test data
    data = np.load(data_file)

    # Calculate Hurst exponent and fractal dimension
    D, result = hurst_dimension(data)
    H = result['hurst']

    # Test theoretical relationship: D = 2 - H
    D_calculated = 2 - H
    assert pytest.approx(D, rel=1e-10) == D_calculated, \
        f"Fractal dimension should satisfy D = 2 - H, got D={D}, H={H}"

    # Test Hurst exponent bounds
    assert 0 < H < 1, f"Hurst exponent {H} should be between 0 and 1"


def test_hurst_goodness_of_fit():
    """Test that Hurst analysis provides good fit quality."""
    # Load test data
    data = np.load(data_file)

    # Calculate Hurst exponent
    D, result = hurst_dimension(data)

    # Check goodness of fit - should be reasonably high for good data
    r_squared = result['R2']
    assert r_squared > 0.7, f"R² should be > 0.7 for good fit, got {r_squared}"
    assert r_squared <= 1.0, f"R² should not exceed 1.0, got {r_squared}"


def test_hurst_consistency():
    """Test that Hurst analysis produces consistent results."""
    # Load test data
    data = np.load(data_file)

    # Calculate Hurst exponent multiple times
    results = []
    hurst_values = []
    for _ in range(3):
        D, result = hurst_dimension(data)
        results.append(D)
        hurst_values.append(result['hurst'])

    # Results should be consistent (small variation due to potential randomness)
    std_dev = np.std(results)
    mean_D = np.mean(results)

    # Relative standard deviation should be small
    relative_std = std_dev / mean_D if mean_D != 0 else std_dev
    assert relative_std < 0.01, f"Results should be consistent, relative std: {relative_std}"

    # Hurst values should also be consistent
    std_hurst = np.std(hurst_values)
    mean_hurst = np.mean(hurst_values)
    relative_std_hurst = std_hurst / mean_hurst if mean_hurst != 0 else std_hurst
    assert relative_std_hurst < 0.01, f"Hurst values should be consistent, relative std: {relative_std_hurst}"


def test_hurst_data_length_sensitivity():
    """Test Hurst analysis with different data lengths."""
    # Load full test data
    full_data = np.load(data_file)

    # Test with different data lengths
    test_lengths = [500, 1000, min(len(full_data), 5000)]

    results = []
    for length in test_lengths:
        if length <= len(full_data):
            data_subset = full_data[:length]
            D, result = hurst_dimension(data_subset)
            results.append((D, result['hurst'], result['R2']))

            # Each calculation should produce valid results
            assert isinstance(D, (float, np.floating)), f"Result with length {length} should be numeric"
            assert 1 < D < 2, f"Fractal dimension {D} should be between 1 and 2"
            assert 0 < result['hurst'] < 1, f"Hurst exponent {result['hurst']} should be between 0 and 1"
            assert result['R2'] > 0.5, f"R² should be reasonable for length {length}"

    # If we have multiple results, they should be relatively consistent
    if len(results) > 1:
        D_values = [r[0] for r in results]
        mean_D = np.mean(D_values)
        std_D = np.std(D_values)

        # Allow for some variation with different lengths but not too much
        assert std_D / mean_D < 0.15, f"Results should be relatively stable across different lengths"


def test_hurst_input_validation():
    """Test Hurst analysis with different input validations."""
    # Load test data
    data = np.load(data_file)

    # Test with different input scenarios
    test_cases = [
        # Normal case
        data,
        # Add small noise
        data + np.random.normal(0, 0.01, len(data)),
        # Scaled data
        data * 10,
        # Offset data
        data + 100,
    ]

    for i, test_data in enumerate(test_cases):
        D, result = hurst_dimension(test_data)

        # Each test case should produce valid results
        assert isinstance(D, (float, np.floating)), f"Test case {i} should produce numeric result"
        assert isinstance(result, dict), f"Test case {i} should produce dictionary result"
        assert 1 < D < 2, f"Test case {i}: Fractal dimension {D} should be between 1 and 2"
        assert 0 < result['hurst'] < 1, f"Test case {i}: Hurst exponent should be between 0 and 1"
        assert result['R2'] > 0.5, f"Test case {i}: R² should be reasonable"

        # The Hurst exponent should be relatively stable across scaling/offset
        if i > 0:
            hurst_diff = abs(result['hurst'] - test_cases[0][1]['hurst'])
            assert hurst_diff < 0.1, f"Hurst exponent should be stable under scaling/offset"
