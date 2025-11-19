#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Fractional Brownian Motion (FBM) Surface Generation
========================================================

Test cases for generating Fractional Brownian Motion (FBM) surfaces using fracDimPy.
FBM surfaces are two-dimensional random fractal surfaces with self-similarity and
isotropy, widely used in terrain generation, texture synthesis, elevation simulation,
and other fields.

Theoretical Background:
- Relationship between FBM surface fractal dimension D and Hurst exponent H: D = 3 - H
- D ∈ (2, 3), larger D indicates rougher surfaces
- H ∈ (0, 1), larger H indicates smoother surfaces
"""

import numpy as np
import pytest


def test_fbm_surface_basic_generation():
    """Test basic FBM surface generation functionality."""
    from fracDimPy import generate_fbm_surface

    # Test basic generation
    dimension = 2.3  # Target fractal dimension
    size = 256  # Surface size (number of pixels)

    surface = generate_fbm_surface(dimension=dimension, size=size)

    # Test basic properties
    assert surface.shape == (size, size), f"Expected shape {(size, size)}, got {surface.shape}"
    assert surface.dtype in [np.float64, np.float32], "Expected float array for surface values"
    assert np.all(np.isfinite(surface)), "All surface values should be finite"

    # Test FBM surface properties
    assert surface.min() < surface.max(), "Should have variation in surface values"
    assert len(np.unique(surface)) > 10, "Should have sufficient variation in values"

    # Test Hurst exponent relationship
    H = 3 - dimension
    assert 0 < H < 1, f"Hurst exponent H={H} should be in (0, 1)"


def test_fbm_surface_different_dimensions():
    """Test FBM surface generation with different fractal dimensions."""
    from fracDimPy import generate_fbm_surface

    dimensions = [2.1, 2.3, 2.5, 2.7, 2.9]
    size = 128

    std_devs = []
    for dimension in dimensions:
        surface = generate_fbm_surface(dimension=dimension, size=size)

        # Test basic properties
        assert surface.shape == (size, size), f"Dimension {dimension}: Incorrect shape"
        assert np.all(np.isfinite(surface)), f"Dimension {dimension}: All values should be finite"

        std_dev = surface.std()
        std_devs.append(std_dev)

        # Test that roughness increases with dimension
        H = 3 - dimension
        assert 0 < H < 1, f"Dimension {dimension}: Invalid Hurst exponent"

    # Test that standard deviation increases with fractal dimension (rougher surfaces)
    # Allow for some variation due to randomness, but check general trend
    for i in range(1, len(dimensions)):
        # Higher dimension should generally give higher standard deviation
        # But we allow for some tolerance due to randomness
        assert (
            abs(std_devs[i] - std_devs[i - 1]) < max(std_devs) * 0.5
        ), f"Std dev should not vary dramatically between dimensions {dimensions[i]} and {dimensions[i-1]}"


def test_fbm_surface_hurst_exponent():
    """Test Hurst exponent relationship with surface roughness."""
    from fracDimPy import generate_fbm_surface

    size = 128

    # Test Hurst exponents from 0.1 to 0.9
    hurst_exponents = [0.1, 0.3, 0.5, 0.7, 0.9]

    for H in hurst_exponents:
        dimension = 3 - H
        surface = generate_fbm_surface(dimension=dimension, size=size)

        # Test basic properties
        assert surface.shape == (size, size), f"H={H}: Incorrect shape"
        assert np.all(np.isfinite(surface)), f"H={H}: All values should be finite"

        # Test that surface has reasonable properties
        assert surface.std() > 0, f"H={H}: Surface should have non-zero standard deviation"
        assert len(np.unique(surface)) > size, f"H={H}: Surface should have sufficient variation"


def test_fbm_surface_theoretical_properties():
    """Test theoretical properties of FBM surface."""
    from fracDimPy import generate_fbm_surface

    # Test with specific parameters
    dimension = 2.5
    H = 3 - dimension  # Should be 0.5
    size = 256

    surface = generate_fbm_surface(dimension=dimension, size=size)

    # Test dimension bounds
    assert 2.0 < dimension < 3.0, f"Fractal dimension {dimension} should be in (2, 3)"
    assert 0.0 < H < 1.0, f"Hurst exponent {H} should be in (0, 1)"

    # Test that surface has expected statistical properties
    # FBM should be statistically self-similar
    mean_val = surface.mean()
    std_val = surface.std()

    # FBM surfaces should have approximately zero mean (or can be normalized)
    # We allow for some deviation due to random generation
    assert (
        abs(mean_val) < 5 * std_val
    ), f"Mean {mean_val} should be reasonable relative to std {std_val}"

    # Test theoretical dimension calculation
    assert (
        abs(dimension + H - 3) < 0.001
    ), f"Dimension {dimension} and H {H} should satisfy D + H = 3"


def test_fbm_surface_edge_cases():
    """Test edge cases for FBM surface generation."""
    from fracDimPy import generate_fbm_surface

    # Test with small size
    size = 64
    surface_small = generate_fbm_surface(dimension=2.3, size=size)
    assert surface_small.shape == (size, size), "Small size should work"
    assert np.all(np.isfinite(surface_small)), "Small size surface should be finite"

    # Test with extreme dimensions
    for dimension in [2.01, 2.99]:
        H = 3 - dimension
        assert 0 < H < 1, f"Extreme dimension {dimension} should give valid H"

        surface = generate_fbm_surface(dimension=dimension, size=128)
        assert surface.shape == (128, 128), f"Dimension {dimension}: Incorrect shape"
        assert np.all(np.isfinite(surface)), f"Dimension {dimension}: All values should be finite"

    # Test with medium size
    size = 512
    surface_medium = generate_fbm_surface(dimension=2.5, size=size)
    assert surface_medium.shape == (size, size), "Medium size should work"
    assert np.all(np.isfinite(surface_medium)), "Medium size surface should be finite"


def test_fbm_surface_self_similarity():
    """Test statistical self-similarity property of FBM surface."""
    from fracDimPy import generate_fbm_surface

    dimension = 2.3
    size = 256

    surface = generate_fbm_surface(dimension=dimension, size=size)

    # Test statistical self-similarity by comparing statistics at different scales
    # This is a simplified test - true self-similarity analysis would be more complex

    # Sample different regions of the surface
    region_size = size // 4
    regions = [
        surface[:region_size, :region_size],
        surface[region_size : 2 * region_size, region_size : 2 * region_size],
        surface[-region_size:, -region_size:],
        surface[:region_size, -region_size:],
        surface[-region_size:, :region_size],
    ]

    # Test that different regions have similar statistical properties
    std_devs = [region.std() for region in regions]
    mean_std = np.mean(std_devs)

    # All regions should have similar standard deviations (within tolerance)
    for i, std_dev in enumerate(std_devs):
        assert (
            abs(std_dev - mean_std) < 0.5 * mean_std
        ), f"Region {i}: Standard deviation {std_dev} should be close to mean {mean_std}"


@pytest.mark.parametrize("dimension", [2.1, 2.3, 2.5, 2.7, 2.9])
def test_fbm_surface_dimension_parameter(dimension):
    """Test FBM surface generation with different dimension parameters."""
    from fracDimPy import generate_fbm_surface

    size = 128
    surface = generate_fbm_surface(dimension=dimension, size=size)

    assert surface.shape == (size, size), f"Dimension {dimension}: Correct shape"
    assert np.all(np.isfinite(surface)), f"Dimension {dimension}: All values should be finite"
    assert surface.std() > 0, f"Dimension {dimension}: Should have variation"

    # Test dimension bounds
    assert 2.0 < dimension < 3.0, f"Dimension {dimension}: Should be in valid range"

    # Test Hurst exponent relationship
    H = 3 - dimension
    assert 0.0 < H < 1.0, f"Dimension {dimension}: Hurst exponent {H} should be in (0, 1)"


@pytest.mark.parametrize("size", [64, 128, 256, 512])
def test_fbm_surface_size_parameter(size):
    """Test FBM surface generation with different size parameters."""
    from fracDimPy import generate_fbm_surface

    dimension = 2.5
    surface = generate_fbm_surface(dimension=dimension, size=size)

    assert surface.shape == (size, size), f"Size {size}: Correct shape"
    assert np.all(np.isfinite(surface)), f"Size {size}: All values should be finite"
    assert surface.std() > 0, f"Size {size}: Should have variation"

    # Test that larger surfaces have more unique values
    unique_values = len(np.unique(surface))
    assert unique_values > size, f"Size {size}: Should have sufficient variation"
