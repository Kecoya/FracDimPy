#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Information Dimension Method Tests
=================================

Test suite for information dimension method applied to point set data.

The information dimension is based on information entropy and measures
the amount of information needed to specify a point in the set. It is
particularly useful for analyzing chaotic systems and time series data.

Theoretical Background:
- Uses Shannon entropy: I(epsilon) = -Σ p_i log(p_i)
- p_i is the probability of finding a point in box i
- Information dimension D_I is the slope of I(epsilon) vs log(1/epsilon)
- Key properties:
  - D_I <= D_0 (capacity dimension)
  - D_I = D_0 for uniform distributions
  - D_I < D_0 for non-uniform distributions
"""

import numpy as np
import os
import pytest
from fracDimPy import information_dimension


def logistic_map(r, x0=0.1, num_steps=5000, transient=1000):
    """Generate trajectory from logistic map

    The logistic map is a polynomial mapping that exhibits chaotic behavior.
    """
    x = x0
    trajectory = []

    # Transient period (discard initial points)
    for _ in range(transient):
        x = r * x * (1 - x)

    # Collect trajectory points
    for _ in range(num_steps):
        x = r * x * (1 - x)
        trajectory.append(x)

    return np.array(trajectory)


def tent_map(mu, x0=0.1, num_steps=5000, transient=1000):
    """Generate trajectory from tent map

    The tent map is a piecewise linear map that exhibits chaotic behavior.
    """
    x = x0
    trajectory = []

    # Transient period
    for _ in range(transient):
        if x < 0.5:
            x = mu * x
        else:
            x = mu * (1 - x)

    # Collect trajectory points
    for _ in range(num_steps):
        if x < 0.5:
            x = mu * x
        else:
            x = mu * (1 - x)
        trajectory.append(x)

    return np.array(trajectory)


def henon_map_1d(num_steps=5000, a=1.4, b=0.3, transient=1000):
    """Generate 1D trajectory from Henon map (x component only)

    The Henon map is a discrete-time dynamical system that exhibits chaotic behavior.
    """
    x, y = 0, 0
    trajectory = []

    # Transient period
    for _ in range(transient):
        x_new = 1 - a * x**2 + y
        y_new = b * x
        x, y = x_new, y_new

    # Collect trajectory points
    for _ in range(num_steps):
        x_new = 1 - a * x**2 + y
        y_new = b * x
        x, y = x_new, y_new
        trajectory.append(x)

    return np.array(trajectory)


def generate_multifractal_series(n=5000, p=0.3):
    """Generate multifractal time series using multiplicative cascade"""
    # Calculate number of levels
    levels = int(np.log2(n))
    series = np.ones(2**levels)

    for level in range(levels):
        step = 2 ** (levels - level)
        for i in range(0, 2**levels, step):
            # Apply random multiplicative factor
            if np.random.rand() < p:
                series[i : i + step // 2] *= 1.5
                series[i + step // 2 : i + step] *= 0.5
            else:
                series[i : i + step // 2] *= 0.5
                series[i + step // 2 : i + step] *= 1.5

    return series[:n]


class TestInformationDimension:
    """Test suite for information dimension method."""

    def test_logistic_map_dimension(self):
        """Test information dimension on logistic map data."""
        # Generate logistic map data
        data = logistic_map(r=3.9, num_steps=2000)  # Reduced for faster testing

        D, result = information_dimension(
            data, num_points=15, min_boxes=4, max_boxes=30  # Reduced for faster testing
        )

        # Validate results
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert 0 < D < 2  # For 1D chaotic time series
        assert "r_squared" in result
        assert 0 < result["r_squared"] <= 1

        # Logistic map should have information dimension between 0.9 and 1.0
        assert 0.8 < D < 1.2

    def test_tent_map_dimension(self):
        """Test information dimension on tent map data."""
        # Generate tent map data
        data = tent_map(mu=1.9, num_steps=2000)

        D, result = information_dimension(data, num_points=15, min_boxes=4, max_boxes=30)

        # Validate results
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert 0 < D < 2
        assert "r_squared" in result
        assert 0 < result["r_squared"] <= 1

        # Tent map should have information dimension close to 1
        assert 0.8 < D < 1.2

    def test_henon_map_dimension(self):
        """Test information dimension on Henon map data."""
        # Generate Henon map data
        data = henon_map_1d(num_steps=2000)

        D, result = information_dimension(data, num_points=15, min_boxes=4, max_boxes=30)

        # Validate results
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert 0 < D < 2
        assert "r_squared" in result
        assert 0 < result["r_squared"] <= 1

        # Henon map should have information dimension around 1.2
        assert 1.0 < D < 1.5

    def test_multifractal_series_dimension(self):
        """Test information dimension on multifractal series."""
        # Generate multifractal data
        data = generate_multifractal_series(n=2048)  # Reduced for faster testing

        D, result = information_dimension(data, num_points=15, min_boxes=4, max_boxes=40)

        # Validate results
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert 0 < D < 2
        assert "r_squared" in result
        assert 0 < result["r_squared"] <= 1

        # Multifractal series should have D between 0.5 and 1.5
        assert 0.5 < D < 1.5

    def test_uniform_random_data(self):
        """Test information dimension on uniform random data."""
        # Generate uniform random data
        data = np.random.rand(2000)

        D, result = information_dimension(data, num_points=15, min_boxes=4, max_boxes=30)

        # For uniform distribution, information dimension should be close to 1
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert pytest.approx(D, rel=0.2) == 1.0
        assert result["r_squared"] > 0.8

    def test_gaussian_random_data(self):
        """Test information dimension on Gaussian random data."""
        # Generate Gaussian random data
        data = np.random.randn(2000)

        D, result = information_dimension(data, num_points=15, min_boxes=4, max_boxes=30)

        # For Gaussian distribution, information dimension should be close to 1
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert pytest.approx(D, rel=0.3) == 1.0
        assert result["r_squared"] > 0.7

    def test_different_parameters(self):
        """Test information dimension with different parameters."""
        data = logistic_map(r=3.9, num_steps=1000)

        # Test different parameter combinations
        param_sets = [(10, 3, 20), (15, 4, 30), (20, 5, 40)]

        for num_points, min_boxes, max_boxes in param_sets:
            D, result = information_dimension(
                data, num_points=num_points, min_boxes=min_boxes, max_boxes=max_boxes
            )

            assert isinstance(D, (int, float))
            assert isinstance(result, dict)
            assert 0 < D < 2
            assert "r_squared" in result
            assert 0 < result["r_squared"] <= 1

    def test_different_data_lengths(self):
        """Test information dimension with different data lengths."""
        lengths = [500, 1000, 2000]

        for length in lengths:
            data = logistic_map(r=3.9, num_steps=length)

            D, result = information_dimension(data, num_points=12, min_boxes=3, max_boxes=25)

            assert isinstance(D, (int, float))
            assert isinstance(result, dict)
            assert 0 < D < 2
            assert "r_squared" in result
            assert 0 < result["r_squared"] <= 1

            # Longer data should generally give better R²
            min_r2 = 0.6 if length < 1000 else 0.7
            assert result["r_squared"] > min_r2

    def test_result_structure_information(self):
        """Test that result dictionary contains expected structure."""
        data = logistic_map(r=3.9, num_steps=1000)
        D, result = information_dimension(data, num_points=10, min_boxes=3, max_boxes=20)

        # Check required keys
        required_keys = ["r_squared"]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Check data consistency if box sizes are present
        if "box_sizes" in result and "information_values" in result:
            assert len(result["box_sizes"]) == len(result["information_values"])
            assert all(x > 0 for x in result["box_sizes"])
            assert all(
                x >= 0 for x in result["information_values"]
            )  # Information should be non-negative

        # Check coefficients if present
        if "coefficients" in result:
            assert len(result["coefficients"]) >= 2
            assert all(isinstance(c, (int, float)) for c in result["coefficients"])

    def test_theoretical_constraints(self):
        """Test results against theoretical constraints."""
        data = logistic_map(r=3.9, num_steps=1000)
        D, result = information_dimension(data, num_points=10, min_boxes=3, max_boxes=20)

        # For 1D time series, information dimension should satisfy:
        # - Lower bound: 0 (completely deterministic)
        # - Upper bound: 1 (completely random)
        # However, chaotic systems can exceed 1 slightly due to attractor structure
        assert 0 < D < 2  # Allow some tolerance

        # R² should indicate reasonable fit
        assert result["r_squared"] > 0.5

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very short data
        short_data = np.random.rand(100)
        try:
            D, result = information_dimension(short_data, num_points=5, min_boxes=2, max_boxes=10)
            assert isinstance(D, (int, float))
            assert isinstance(result, dict)
        except (ValueError, RuntimeError):
            # It's acceptable if very short data raises an error
            pass

        # Test with constant data
        constant_data = np.ones(500) * 0.5
        try:
            D, result = information_dimension(
                constant_data, num_points=5, min_boxes=2, max_boxes=10
            )
            # Constant data should give D = 0 or raise error
            if isinstance(D, (int, float)):
                assert D <= 0.5  # Should be very low
        except (ValueError, RuntimeError):
            pass

    def test_parameter_validation(self):
        """Test parameter validation."""
        data = logistic_map(r=3.9, num_steps=500)

        # Test invalid parameters
        with pytest.raises((ValueError, TypeError)):
            information_dimension(data, num_points=0)

        with pytest.raises((ValueError, TypeError)):
            information_dimension(data, min_boxes=0)

        with pytest.raises((ValueError, TypeError)):
            information_dimension(data, max_boxes=1, min_boxes=5)  # max < min

        # Test invalid data
        with pytest.raises((ValueError, TypeError)):
            information_dimension([], num_points=5)

        with pytest.raises((ValueError, TypeError)):
            information_dimension(np.array([]), num_points=5)
