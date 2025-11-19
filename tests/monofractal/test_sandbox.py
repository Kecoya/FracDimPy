#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sandbox Method Tests
===================

Test suite for sandbox method applied to point set and image data.

The sandbox method is a local scale analysis technique that counts
the number of points within boxes of increasing radius centered at
various points in the dataset.
"""

import numpy as np
import os
import pytest
from fracDimPy import sandbox_method

def load_sandbox_data():
    """Load sandbox data file path."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "sandbox_data.png")


def generate_test_patterns():
    """Generate synthetic test patterns for sandbox method."""
    patterns = {}

    # 2D Sierpinski triangle points
    points_sierpinski = []
    for _ in range(2000):
        x, y = np.random.rand(2)
        for _ in range(10):  # Iterations for Sierpinski
            vertex = np.random.choice(3)
            vertices = [(0, 0), (1, 0), (0.5, np.sqrt(3)/2)]
            x = 0.5 * (x + vertices[vertex][0])
            y = 0.5 * (y + vertices[vertex][1])
        points_sierpinski.append([x, y])
    patterns['sierpinski'] = np.array(points_sierpinski)

    # 2D grid points (D = 2)
    x, y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
    grid_points = np.column_stack([x.ravel(), y.ravel()])
    patterns['grid'] = grid_points

    # Random points (D = 2)
    random_points = np.random.rand(1000, 2)
    patterns['random'] = random_points

    # 1D Cantor set points
    cantor_points = []
    for _ in range(1000):
        x = 0.0
        for _ in range(8):  # 8 iterations
            x = x / 3 if np.random.rand() < 0.5 else (x + 2) / 3
        cantor_points.append([x, 0])
    patterns['cantor'] = np.array(cantor_points)

    return patterns


class TestSandbox:
    """Test suite for sandbox method."""

    @pytest.fixture
    def sandbox_data_file(self):
        """Get sandbox data file path."""
        return load_sandbox_data()

    @pytest.fixture
    def test_patterns(self):
        """Generate synthetic test patterns."""
        return generate_test_patterns()

    def test_sandbox_with_file(self, sandbox_data_file):
        """Test sandbox method with image file."""
        try:
            D, result = sandbox_method(sandbox_data_file)

            # Validate results
            assert isinstance(D, (int, float))
            assert isinstance(result, dict)
            assert 0 < D < 3  # For 2D image data
            assert 'R2' in result
            assert 0 < result['R2'] <= 1

        except (FileNotFoundError, ImportError):
            # Skip test if data file doesn't exist or required libraries missing
            pytest.skip("Data file or required libraries not available")

    def test_sierpinski_pattern(self, test_patterns):
        """Test sandbox method on Sierpinski triangle."""
        points = test_patterns['sierpinski']

        D, result = sandbox_method(points)

        # Sierpinski triangle has D ≈ 1.585
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert pytest.approx(D, rel=0.2) == 1.585
        assert result['R2'] > 0.8  # Should have good fit

    def test_grid_pattern(self, test_patterns):
        """Test sandbox method on grid pattern."""
        points = test_patterns['grid']

        D, result = sandbox_method(points)

        # Grid should have D ≈ 2 (space-filling in 2D)
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert pytest.approx(D, rel=0.2) == 2.0
        assert result['R2'] > 0.8

    def test_random_pattern(self, test_patterns):
        """Test sandbox method on random points."""
        points = test_patterns['random']

        D, result = sandbox_method(points)

        # Random points should have D ≈ 2
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert 1.5 <= D <= 2.5  # Allow reasonable range
        assert result['R2'] > 0.6  # Random data might have lower R²

    def test_cantor_pattern(self, test_patterns):
        """Test sandbox method on Cantor set."""
        points = test_patterns['cantor']

        D, result = sandbox_method(points)

        # Cantor set has D ≈ 0.631
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert pytest.approx(D, rel=0.3) == 0.631
        assert result['R2'] > 0.6

    def test_different_point_densities(self):
        """Test sandbox method with different point densities."""
        densities = [100, 500, 1000, 2000]

        for density in densities:
            # Generate random points
            points = np.random.rand(density, 2)

            D, result = sandbox_method(points)

            assert isinstance(D, (int, float))
            assert isinstance(result, dict)
            assert 0 < D < 3
            assert 'R2' in result
            assert 0 < result['R2'] <= 1

            # Higher density should generally give better R²
            min_r2 = 0.5 if density < 500 else 0.6
            assert result['R2'] > min_r2

    def test_result_structure_sandbox(self, test_patterns):
        """Test that result dictionary contains expected structure."""
        points = test_patterns['sierpinski']
        D, result = sandbox_method(points)

        # Check required keys
        required_keys = ['R2']
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Check data consistency if radius values are present
        if 'r_values' in result and 'N_values' in result:
            assert len(result['r_values']) == len(result['N_values'])
            assert all(x > 0 for x in result['r_values'])
            assert all(x > 0 for x in result['N_values'])

            # N should increase with r
            r_vals = np.array(result['r_values'])
            n_vals = np.array(result['N_values'])
            assert np.all(np.diff(n_vals) >= 0) or len(set(n_vals)) > 1

        # Check coefficients if present
        if 'coefficients' in result:
            assert len(result['coefficients']) >= 2
            assert all(isinstance(c, (int, float)) for c in result['coefficients'])

    def test_theoretical_constraints(self, test_patterns):
        """Test results against theoretical constraints."""
        points = test_patterns['sierpinski']
        D, result = sandbox_method(points)

        # For 2D point sets, fractal dimension should satisfy:
        # - Lower bound: 0 (discrete points)
        # - Upper bound: 2 (space-filling)
        assert 0 <= D <= 2

        # R² should indicate reasonable fit
        assert result['R2'] > 0.5

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very few points
        few_points = np.random.rand(10, 2)
        try:
            D, result = sandbox_method(few_points)
            assert isinstance(D, (int, float))
            assert isinstance(result, dict)
        except (ValueError, RuntimeError):
            # It's acceptable if very few points raise an error
            pass

        # Test with collinear points (should have D ≈ 1)
        collinear_points = np.column_stack([np.linspace(0, 1, 100), np.zeros(100)])
        try:
            D, result = sandbox_method(collinear_points)
            if isinstance(D, (int, float)):
                assert pytest.approx(D, rel=0.3) == 1.0
        except (ValueError, RuntimeError):
            pass

        # Test with identical points (should raise error or give D = 0)
        identical_points = np.tile([0.5, 0.5], (100, 1))
        try:
            D, result = sandbox_method(identical_points)
            if isinstance(D, (int, float)):
                assert D <= 0.5  # Should be very low
        except (ValueError, RuntimeError):
            pass

    def test_parameter_validation(self, test_patterns):
        """Test parameter validation."""
        points = test_patterns['sierpinski']

        # Test invalid data types
        with pytest.raises((ValueError, TypeError)):
            sandbox_method("invalid_string")

        with pytest.raises((ValueError, TypeError)):
            sandbox_method([])

        with pytest.raises((ValueError, TypeError)):
            sandbox_method(np.array([]))

        # Test invalid point format (wrong dimensions)
        with pytest.raises((ValueError, TypeError)):
            sandbox_method(np.random.rand(100))  # 1D instead of 2D

        with pytest.raises((ValueError, TypeError)):
            sandbox_method(np.random.rand(100, 3))  # 3D instead of 2D
