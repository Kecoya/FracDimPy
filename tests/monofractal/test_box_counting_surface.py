#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Box-counting Method for Surface Data Tests
==========================================

Test suite for box-counting method applied to 2D surface data (height field).

Tests 6 different box-counting methods:
- method=0: RDCCM - Relative Differential Cubic Cover Method
- method=1: DCCM  - Differential Cubic Cover Method
- method=2: CCM   - Cubic Cover Method (standard)
- method=3: ICCM  - Interpolated Cubic Cover Method
- method=5: SCCM  - Simplified Cubic Cover Method
- method=6: SDCCM - Simplified Differential Cubic Cover Method
"""

import numpy as np
import pandas as pd
import os
import pytest
from fracDimPy import box_counting


def advise_mtepsilon(x, y):
    """
    Calculate optimal epsilon for coordinate to matrix conversion.

    Parameters
    ----------
    x, y : np.ndarray
        Coordinate arrays

    Returns
    -------
    float
        Recommended epsilon value
    """
    xl = np.max(x) - np.min(x)
    yl = np.max(y) - np.min(y)
    num = len(x)
    return round(np.sqrt(xl * yl / num), 4)


def coordinate_to_matrix(x, y, z, epsilon=None):
    """
    Convert XYZ coordinate data to matrix format.

    Parameters
    ----------
    x, y, z : np.ndarray
        Coordinate arrays
    epsilon : float, optional
        Grid spacing. If None, will be calculated automatically.

    Returns
    -------
    matrix : np.ndarray
        2D height matrix
    epsilon : float
        Used epsilon value
    """
    if epsilon is None:
        epsilon = advise_mtepsilon(x, y)

    # Convert coordinates to grid indices
    y_ = np.round((y - np.min(y)) / epsilon).astype(int)
    x_ = np.round((x - np.min(x)) / epsilon).astype(int)

    # Create matrix
    matrix = np.zeros((np.max(y_) + 1, np.max(x_) + 1))
    z = z - np.min(z)
    matrix[y_, x_] = z

    # Remove boundary
    if matrix.shape[0] > 4 and matrix.shape[1] > 4:
        matrix = matrix[2:-2, 2:-2]

    return matrix, epsilon


def interpolate_surface(mt):
    """
    Interpolate missing values in surface matrix.

    Parameters
    ----------
    mt : np.ndarray
        Surface matrix with potential zeros

    Returns
    -------
    np.ndarray
        Interpolated surface matrix
    """
    h, w = mt.shape[0], mt.shape[1]
    a, b = np.where(mt == 0)

    num = len(a)
    if num == 0:
        return mt

    # Local interpolation for zero values
    for i in range(len(a)):
        if a[i] == 0:
            a1, a2 = a[i], a[i] + 2
        elif a[i] == h - 1:
            a1, a2 = a[i] - 1, a[i] + 1
        else:
            a1, a2 = a[i] - 1, a[i] + 2

        if b[i] == 0:
            b1, b2 = b[i], b[i] + 2
        elif b[i] == w - 1:
            b1, b2 = b[i] - 1, b[i] + 1
        else:
            b1, b2 = b[i] - 1, b[i] + 2

        tempt = mt[a1:a2, b1:b2]
        c = np.sum(tempt) / np.sum(tempt != 0)
        if np.isnan(c):
            c = 0
        mt[a[i], b[i]] = c

    # Global interpolation for remaining zeros
    r = max(int(min(h, w) / 10), 5)
    a, b = np.where(mt == 0)
    for i in range(len(a)):
        a1 = 0 if a[i] - r < 0 else a[i] - r
        a2 = h if a[i] + r + 1 > h else a[i] + r + 1
        b1 = 0 if b[i] - r < 0 else b[i] - r
        b2 = w if b[i] + r + 1 > w else b[i] + r + 1
        tempt = mt[a1:a2, b1:b2]
        c = np.sum(tempt) / np.sum(tempt != 0)
        if np.isnan(c):
            c = 0
        mt[a[i], b[i]] = c

    # Replace any remaining zeros with mean value
    mt[mt == 0] = np.mean(mt)

    return mt


def load_surface_data():
    """Load surface data from CSV file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "box_counting_surface_data.csv")
    df = pd.read_csv(data_file, header=None)

    mt_epsilon_min = None

    if df.shape[1] == 3:
        # XYZ coordinate format
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        z = df.iloc[:, 2].values

        surface, mt_epsilon_min = coordinate_to_matrix(x, y, z)

        # Interpolate if needed
        if np.any(surface == 0):
            surface = interpolate_surface(surface)
    else:
        # Matrix format
        surface = df.values

        # Handle NaN values
        if np.any(np.isnan(surface)):
            surface = np.nan_to_num(surface, nan=0.0)
            if np.any(surface == 0):
                surface = interpolate_surface(surface)

    return surface, mt_epsilon_min


class TestBoxCountingSurface:
    """Test suite for box-counting method on surface data."""

    @pytest.fixture
    def surface_data(self):
        """Load surface data for testing."""
        return load_surface_data()

    def test_box_counting_surface_ccm(self, surface_data):
        """Test Cubic Cover Method (CCM) on surface data."""
        surface, mt_epsilon_min = surface_data

        D, result = box_counting(surface, data_type="surface", method=2)

        # Validate results
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert 0 < D < 4  # Surface fractal dimension should be between 0 and 4
        assert "R2" in result
        assert 0 < result["R2"] <= 1  # R² should be between 0 and 1

        # For a typical surface, D should be around 2-3
        assert pytest.approx(D, rel=0.3) == 2.5

    def test_box_counting_surface_with_epsilon(self, surface_data):
        """Test box-counting with custom epsilon parameter."""
        surface, mt_epsilon_min = surface_data

        if mt_epsilon_min is not None:
            D, result = box_counting(
                surface, data_type="surface", method=2, mt_epsilon_min=mt_epsilon_min
            )

            assert isinstance(D, (int, float))
            assert isinstance(result, dict)
            assert 0 < D < 4
            assert "R2" in result
            assert 0 < result["R2"] <= 1

    @pytest.mark.parametrize("method", [0, 1, 2, 3])
    def test_box_counting_surface_different_methods(self, surface_data, method):
        """Test different box-counting methods on surface data."""
        surface, _ = surface_data

        D, result = box_counting(surface, data_type="surface", method=method)

        # Validate results for each method
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert 0 < D < 4
        assert "R2" in result
        assert 0 < result["R2"] <= 1

    def test_surface_data_loading(self):
        """Test that surface data loads correctly."""
        surface, mt_epsilon_min = load_surface_data()

        # Validate surface data
        assert isinstance(surface, np.ndarray)
        assert surface.ndim == 2  # Should be 2D array
        assert surface.shape[0] > 0 and surface.shape[1] > 0
        assert not np.any(np.isnan(surface))  # No NaN values
        assert np.all(surface >= 0)  # All values should be non-negative

    def test_coordinate_to_matrix_conversion(self):
        """Test XYZ to matrix conversion."""
        # Create simple test data
        x = np.array([0, 1, 0, 1])
        y = np.array([0, 0, 1, 1])
        z = np.array([1, 2, 3, 4])

        matrix, epsilon = coordinate_to_matrix(x, y, z)

        assert isinstance(matrix, np.ndarray)
        assert matrix.ndim == 2
        assert isinstance(epsilon, (int, float))
        assert epsilon > 0

    def test_surface_interpolation(self):
        """Test surface interpolation function."""
        # Create test matrix with zeros
        test_matrix = np.array([[1, 0, 3], [0, 5, 0], [7, 0, 9]], dtype=float)

        interpolated = interpolate_surface(test_matrix.copy())

        # Should have no zeros after interpolation
        assert not np.any(interpolated == 0)
        assert interpolated.shape == test_matrix.shape

    def test_result_structure(self, surface_data):
        """Test that result dictionary contains expected structure."""
        surface, _ = surface_data

        D, result = box_counting(surface, data_type="surface", method=2)

        # Check required keys
        required_keys = ["R2"]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Check data consistency if epsilon values are present
        if "epsilon_values" in result and "N_values" in result:
            assert len(result["epsilon_values"]) == len(result["N_values"])
            assert all(x > 0 for x in result["epsilon_values"])
            assert all(x > 0 for x in result["N_values"])

        # Check log data consistency
        if "log_inv_epsilon" in result and "log_N" in result:
            assert len(result["log_inv_epsilon"]) == len(result["log_N"])
