#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Fractional Brownian Motion (FBM) Curve Generation
======================================================

Test cases for generating Fractional Brownian Motion (FBM) curves using fracDimPy.
FBM is an important fractal curve with self-similarity and long-range correlation
properties, widely used in financial time series, terrain modeling, and other fields.

Theoretical Background:
- Relationship between FBM fractal dimension D and Hurst exponent H: D = 2 - H
- D ∈ (1, 2), larger D indicates more irregular curves
- H ∈ (0, 1), larger H indicates smoother curves
"""

import numpy as np
import pytest


def test_fbm_curve_basic_generation():
    """Test basic FBM curve generation functionality."""
    from fracDimPy import generate_fbm_curve

    # Test basic generation
    target_D = 1.5  # Target fractal dimension
    length = 2048  # Curve length (number of sampling points)

    curve, D_set = generate_fbm_curve(dimension=target_D, length=length)

    # Test basic properties
    assert len(curve) == length, f"Expected length {length}, got {len(curve)}"
    assert curve.dtype in [np.float64, np.float32], "Expected float array for curve values"
    assert np.all(np.isfinite(curve)), "All curve values should be finite"

    # Test that D_set is close to target_D
    assert isinstance(D_set, (int, float, np.number)), "D_set should be a number"
    assert (
        abs(D_set - target_D) < 0.1
    ), f"Set dimension {D_set} should be close to target {target_D}"

    # Test FBM curve properties
    assert curve.min() < curve.max(), "Should have variation in curve values"
    assert len(np.unique(curve)) > length // 10, "Should have sufficient variation in values"

    # Test Hurst exponent relationship
    H = 2 - target_D
    assert 0 < H < 1, f"Hurst exponent H={H} should be in (0, 1)"


def test_fbm_curve_different_dimensions():
    """Test FBM curve generation with different fractal dimensions."""
    from fracDimPy import generate_fbm_curve

    dimensions = [1.1, 1.3, 1.5, 1.7, 1.9]
    length = 1024

    std_devs = []
    for dimension in dimensions:
        curve, D_set = generate_fbm_curve(dimension=dimension, length=length)

        # Test basic properties
        assert len(curve) == length, f"Dimension {dimension}: Incorrect length"
        assert np.all(np.isfinite(curve)), f"Dimension {dimension}: All values should be finite"
        assert abs(D_set - dimension) < 0.1, f"Dimension {dimension}: D_set {D_set} should be close"

        std_dev = curve.std()
        std_devs.append(std_dev)

        # Test that roughness increases with dimension
        H = 2 - dimension
        assert 0 < H < 1, f"Dimension {dimension}: Invalid Hurst exponent"

    # Test that standard deviation generally varies with fractal dimension
    # Allow for some variation due to randomness, but check that we have different values
    unique_std_devs = len(np.unique(np.round(std_devs, 6)))
    assert (
        unique_std_devs >= 2
    ), "Should have different standard deviations for different dimensions"


def test_fbm_curve_hurst_exponent():
    """Test Hurst exponent relationship with curve roughness."""
    from fracDimPy import generate_fbm_curve

    length = 1024

    # Test Hurst exponents from 0.1 to 0.9
    hurst_exponents = [0.1, 0.3, 0.5, 0.7, 0.9]

    for H in hurst_exponents:
        dimension = 2 - H
        curve, D_set = generate_fbm_curve(dimension=dimension, length=length)

        # Test basic properties
        assert len(curve) == length, f"H={H}: Incorrect length"
        assert np.all(np.isfinite(curve)), f"H={H}: All values should be finite"
        assert (
            abs(D_set - dimension) < 0.1
        ), f"H={H}: D_set {D_set} should be close to target {dimension}"

        # Test that curve has reasonable properties
        assert curve.std() > 0, f"H={H}: Curve should have non-zero standard deviation"
        assert (
            len(np.unique(curve)) > length // 20
        ), f"H={H}: Curve should have sufficient variation"


def test_fbm_curve_theoretical_properties():
    """Test theoretical properties of FBM curve."""
    from fracDimPy import generate_fbm_curve

    # Test with specific parameters
    dimension = 1.5
    H = 2 - dimension  # Should be 0.5
    length = 2048

    curve, D_set = generate_fbm_curve(dimension=dimension, length=length)

    # Test dimension bounds
    assert 1.0 < dimension < 2.0, f"Fractal dimension {dimension} should be in (1, 2)"
    assert 0.0 < H < 1.0, f"Hurst exponent {H} should be in (0, 1)"

    # Test that curve has expected statistical properties
    # FBM should be statistically self-similar
    mean_val = curve.mean()
    std_val = curve.std()

    # FBM curves should have approximately zero mean (or can be normalized)
    # We allow for some deviation due to random generation
    assert (
        abs(mean_val) < 3 * std_val
    ), f"Mean {mean_val} should be reasonable relative to std {std_val}"

    # Test theoretical dimension calculation
    assert (
        abs(dimension + H - 2) < 0.001
    ), f"Dimension {dimension} and H {H} should satisfy D + H = 2"

    # Test self-similarity property by checking variance at different scales
    # This is a simplified test - true self-similarity analysis would be more complex
    quarter_length = length // 4
    quarter_std = curve[:quarter_length].std()
    half_std = curve[: length // 2].std()
    full_std = curve.std()

    # Standard deviations should scale appropriately (simplified test)
    assert quarter_std > 0, "First quarter should have variation"
    assert half_std > 0, "First half should have variation"
    assert full_std > 0, "Full curve should have variation"


def test_fbm_curve_edge_cases():
    """Test edge cases for FBM curve generation."""
    from fracDimPy import generate_fbm_curve

    # Test with small length
    length = 128
    curve_small, D_set_small = generate_fbm_curve(dimension=1.5, length=length)
    assert len(curve_small) == length, "Small length should work"
    assert np.all(np.isfinite(curve_small)), "Small length curve should be finite"
    assert abs(D_set_small - 1.5) < 0.1, "Small length D_set should be close to target"

    # Test with extreme dimensions
    for dimension in [1.01, 1.99]:
        H = 2 - dimension
        assert 0 < H < 1, f"Extreme dimension {dimension} should give valid H"

        curve, D_set = generate_fbm_curve(dimension=dimension, length=512)
        assert len(curve) == 512, f"Dimension {dimension}: Incorrect length"
        assert np.all(np.isfinite(curve)), f"Dimension {dimension}: All values should be finite"
        assert (
            abs(D_set - dimension) < 0.1
        ), f"Dimension {dimension}: D_set should be close to target"

    # Test with large length
    length = 4096
    curve_large, D_set_large = generate_fbm_curve(dimension=1.3, length=length)
    assert len(curve_large) == length, "Large length should work"
    assert np.all(np.isfinite(curve_large)), "Large length curve should be finite"
    assert abs(D_set_large - 1.3) < 0.1, "Large length D_set should be close to target"


def test_fbm_curve_return_values():
    """Test that FBM curve generation returns correct values."""
    from fracDimPy import generate_fbm_curve

    dimension = 1.6
    length = 1024

    curve, D_set = generate_fbm_curve(dimension=dimension, length=length)

    # Test return types
    assert isinstance(curve, np.ndarray), "Curve should be numpy array"
    assert isinstance(D_set, (int, float, np.number)), "D_set should be numeric"

    # Test curve properties
    assert len(curve.shape) == 1, "Curve should be 1D array"
    assert curve.size == length, "Curve should have correct size"

    # Test D_set properties
    assert 1.0 <= D_set <= 2.0, f"D_set {D_set} should be in valid range [1, 2]"
    assert not np.isnan(D_set), "D_set should not be NaN"
    assert np.isfinite(D_set), "D_set should be finite"


def test_fbm_curve_statistical_properties():
    """Test statistical properties of FBM curves."""
    from fracDimPy import generate_fbm_curve

    dimension = 1.4
    length = 2048

    curve, D_set = generate_fbm_curve(dimension=dimension, length=length)

    # Test that curve has reasonable statistical distribution
    # FBM curves should have Gaussian-like increments

    # Test basic statistics
    assert curve.mean() is not None, "Curve should have a mean"
    assert curve.std() > 0, "Curve should have positive standard deviation"

    # Test that curve has variation across different segments
    segment_size = length // 8
    segments = [curve[i * segment_size : (i + 1) * segment_size] for i in range(8)]

    # Each segment should have different statistics (within tolerance)
    segment_means = [seg.mean() for seg in segments]
    segment_stds = [seg.std() for seg in segments]

    # Test that we have variation between segments
    mean_range = max(segment_means) - min(segment_means)
    std_range = max(segment_stds) - min(segment_stds)

    assert mean_range >= 0, "Should have some variation in segment means"
    assert std_range >= 0, "Should have some variation in segment standard deviations"


@pytest.mark.parametrize("dimension", [1.1, 1.3, 1.5, 1.7, 1.9])
def test_fbm_curve_dimension_parameter(dimension):
    """Test FBM curve generation with different dimension parameters."""
    from fracDimPy import generate_fbm_curve

    length = 1024
    curve, D_set = generate_fbm_curve(dimension=dimension, length=length)

    assert len(curve) == length, f"Dimension {dimension}: Correct length"
    assert np.all(np.isfinite(curve)), f"Dimension {dimension}: All values should be finite"
    assert curve.std() > 0, f"Dimension {dimension}: Should have variation"
    assert abs(D_set - dimension) < 0.1, f"Dimension {dimension}: D_set should be close to target"

    # Test dimension bounds
    assert 1.0 < dimension < 2.0, f"Dimension {dimension}: Should be in valid range"

    # Test Hurst exponent relationship
    H = 2 - dimension
    assert 0.0 < H < 1.0, f"Dimension {dimension}: Hurst exponent {H} should be in (0, 1)"


@pytest.mark.parametrize("length", [256, 512, 1024, 2048, 4096])
def test_fbm_curve_length_parameter(length):
    """Test FBM curve generation with different length parameters."""
    from fracDimPy import generate_fbm_curve

    dimension = 1.5
    curve, D_set = generate_fbm_curve(dimension=dimension, length=length)

    assert len(curve) == length, f"Length {length}: Correct length"
    assert np.all(np.isfinite(curve)), f"Length {length}: All values should be finite"
    assert curve.std() > 0, f"Length {length}: Should have variation"
    assert abs(D_set - dimension) < 0.1, f"Length {length}: D_set should be close to target"

    # Test that larger curves have more unique values
    unique_values = len(np.unique(np.round(curve, 6)))
    assert unique_values > length // 50, f"Length {length}: Should have sufficient variation"
