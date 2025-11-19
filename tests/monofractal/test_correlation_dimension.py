#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Correlation Dimension Method Tests
=================================

Test suite for the correlation dimension method
to calculate the fractal dimension of point set data.

The correlation dimension is based on the correlation integral, which measures
the probability that two points are within a certain distance of each other.
It is particularly useful for analyzing chaotic attractors and time series data.

Theoretical Background:
- Uses Grassberger-Procaccia algorithm
- Correlation integral: C(r) = (1/N^2) Σ Θ(r - |x_i - x_j|)
- For fractal sets: C(r) ∝ r^D
- Correlation dimension D is the slope of log(C(r)) vs log(r)
- Known theoretical values:
  - Lorenz attractor: D ~= 2.06
  - Henon map: D ~= 1.26
"""

import numpy as np
import pytest
from fracDimPy import correlation_dimension


def lorenz_attractor(num_steps=8000, dt=0.01):
    """Generate trajectory from Lorenz attractor

    The Lorenz attractor is a set of chaotic solutions to the Lorenz system
    of differential equations, which exhibits a strange attractor.
    """

    def lorenz_deriv(state, sigma=10, rho=28, beta=8 / 3):
        x, y, z = state
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    # Initial state
    state = np.array([1.0, 1.0, 1.0])
    trajectory = [state]

    # 4th-order Runge-Kutta integration
    for _ in range(num_steps):
        k1 = lorenz_deriv(state)
        k2 = lorenz_deriv(state + 0.5 * dt * k1)
        k3 = lorenz_deriv(state + 0.5 * dt * k2)
        k4 = lorenz_deriv(state + dt * k3)
        state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        trajectory.append(state)

    return np.array(trajectory)


def henon_map(num_steps=4000, a=1.4, b=0.3):
    """Generate trajectory from Henon map

    The Henon map is a discrete-time dynamical system that exhibits chaotic behavior.
    """
    x, y = 0, 0
    trajectory = []

    for _ in range(num_steps):
        x_new = 1 - a * x**2 + y
        y_new = b * x
        x, y = x_new, y_new
        trajectory.append([x, y])

    return np.array(trajectory)


def test_correlation_dimension_lorenz():
    """Test correlation dimension on Lorenz attractor."""
    # Generate Lorenz trajectory
    lorenz_traj = lorenz_attractor(num_steps=6000, dt=0.01)
    # Remove transient period
    lorenz_traj = lorenz_traj[800:]

    # Known theoretical value for Lorenz attractor
    D_theory = 2.06

    # Calculate correlation dimension
    D, result = correlation_dimension(lorenz_traj, num_points=20, max_samples=2000)

    # Basic result validation
    assert isinstance(D, (float, np.floating)), "Correlation dimension should be a number"
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "r_squared" in result, "Result should contain 'r_squared'"
    assert 0 < D < 4, f"Correlation dimension {D} should be in reasonable range"

    # Test against theoretical value
    assert (
        pytest.approx(D, abs=0.3) == D_theory
    ), f"Lorenz correlation dimension {D} should be close to theoretical {D_theory}"

    # Check goodness of fit
    assert result["r_squared"] > 0.8, f"R² should be > 0.8 for good fit, got {result['r_squared']}"


def test_correlation_dimension_henon():
    """Test correlation dimension on Henon map."""
    # Generate Henon trajectory
    henon_traj = henon_map(num_steps=3000)

    # Known theoretical value for Henon map
    D_theory = 1.26

    # Calculate correlation dimension
    D, result = correlation_dimension(henon_traj, num_points=20, max_samples=2000)

    # Basic result validation
    assert isinstance(D, (float, np.floating)), "Correlation dimension should be a number"
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "r_squared" in result, "Result should contain 'r_squared'"
    assert 0 < D < 3, f"Correlation dimension {D} should be in reasonable range"

    # Test against theoretical value
    assert (
        pytest.approx(D, abs=0.3) == D_theory
    ), f"Henon correlation dimension {D} should be close to theoretical {D_theory}"

    # Check goodness of fit
    assert result["r_squared"] > 0.7, f"R² should be > 0.7 for good fit, got {result['r_squared']}"


def test_correlation_dimension_consistency():
    """Test that correlation dimension produces consistent results."""
    # Generate smaller dataset for efficiency
    henon_traj = henon_map(num_steps=2000)

    # Calculate correlation dimension multiple times
    results = []
    for _ in range(3):
        D, result = correlation_dimension(henon_traj, num_points=15, max_samples=1500)
        results.append(D)

    # Results should be consistent (small variation due to potential randomness)
    std_dev = np.std(results)
    mean_D = np.mean(results)

    # Relative standard deviation should be small
    relative_std = std_dev / mean_D if mean_D != 0 else std_dev
    assert relative_std < 0.05, f"Results should be consistent, relative std: {relative_std}"


def test_correlation_dimension_different_parameters():
    """Test correlation dimension with different parameter combinations."""
    # Generate test data
    henon_traj = henon_map(num_steps=2500)

    # Test with different parameter combinations
    parameter_sets = [
        {},  # default parameters
        {"num_points": 15, "max_samples": 1500},
        {"num_points": 25, "max_samples": 2000},
        {"num_points": 18, "max_samples": 1800},
    ]

    results = []
    for params in parameter_sets:
        D, result = correlation_dimension(henon_traj, **params)
        results.append(D)

        # Each calculation should produce valid results
        assert isinstance(D, (float, np.floating)), f"Result with params {params} should be numeric"
        assert 0 < D < 3, f"Correlation dimension {D} should be in reasonable range"
        assert result["r_squared"] > 0.6, f"R² should be reasonable for params {params}"

    # Results should be relatively consistent across parameter variations
    mean_D = np.mean(results)
    std_D = np.std(results)

    # Allow for some variation but not too much
    assert std_D / mean_D < 0.2, f"Results should be relatively stable across parameters"


def test_correlation_dimension_data_sizes():
    """Test correlation dimension with different data sizes."""
    # Test with different data sizes
    test_sizes = [1000, 2000, 3000]

    results = []
    for size in test_sizes:
        # Generate test data
        henon_traj = henon_map(num_steps=size)

        D, result = correlation_dimension(
            henon_traj, num_points=15, max_samples=min(1500, size // 2)
        )
        results.append(D)

        # Each calculation should produce valid results
        assert isinstance(D, (float, np.floating)), f"Result with size {size} should be numeric"
        assert 0 < D < 3, f"Correlation dimension {D} should be in reasonable range for size {size}"
        assert result["r_squared"] > 0.5, f"R² should be reasonable for size {size}"

    # Results should be relatively consistent across different sizes
    mean_D = np.mean(results)
    std_D = np.std(results)

    # Allow for some variation with different sizes but not too much
    assert std_D / mean_D < 0.3, f"Results should be relatively stable across different data sizes"


def test_correlation_dimension_input_validation():
    """Test correlation dimension with different input validations."""
    # Generate base test data
    base_traj = henon_map(num_steps=2000)

    # Test with different input scenarios
    test_cases = [
        # Normal case
        base_traj,
        # Scaled data
        base_traj * 5,
        # Translated data
        base_traj + np.array([1, 1]),
        # Combined scaling and translation
        base_traj * 2 + np.array([0.5, -0.5]),
    ]

    for i, test_data in enumerate(test_cases):
        D, result = correlation_dimension(test_data, num_points=15, max_samples=1500)

        # Each test case should produce valid results
        assert isinstance(D, (float, np.floating)), f"Test case {i} should produce numeric result"
        assert isinstance(result, dict), f"Test case {i} should produce dictionary result"
        assert 0 < D < 3, f"Test case {i}: Correlation dimension {D} should be between 0 and 3"
        assert result["r_squared"] > 0.5, f"Test case {i}: R² should be reasonable"

        # The correlation dimension should be invariant under scaling/translation
        if i > 0:
            D_diff = abs(D - results[0])
            assert (
                D_diff < 0.3
            ), f"Correlation dimension should be invariant under scaling/translation"


def test_correlation_dimension_dimensional_bounds():
    """Test that correlation dimension respects dimensional bounds."""
    # Test on different dimensional systems
    test_cases = [
        # 1D data (should give D close to 1)
        (np.random.randn(2000).reshape(-1, 1), (0.8, 1.2)),
        # 2D data (should give D between 1 and 2)
        (henon_map(num_steps=2000), (1.0, 2.0)),
        # 3D data (should give D between 2 and 3)
        (lorenz_attractor(num_steps=3000)[500:], (1.5, 2.8)),
    ]

    for i, (data, expected_range) in enumerate(test_cases):
        D, result = correlation_dimension(
            data, num_points=15, max_samples=min(1500, len(data) // 2)
        )

        # Check that dimension is within expected bounds
        min_expected, max_expected = expected_range
        assert (
            min_expected <= D <= max_expected
        ), f"Test case {i}: Correlation dimension {D} should be between {min_expected} and {max_expected}"

        # Basic validation
        assert isinstance(D, (float, np.floating)), f"Test case {i} should produce numeric result"
        assert result["r_squared"] > 0.5, f"Test case {i}: R² should be reasonable"
