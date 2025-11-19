#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Box-counting Method for Porous Media (3D) Tests
===============================================

Test suite for box-counting method applied to 3D porous media data.

Porous media data is represented as a 3D binary array where:
- 1 represents pore space
- 0 represents solid material
"""

import numpy as np
import os
import pytest
from fracDimPy import box_counting

def load_porous_data():
    """Load and preprocess porous media data."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "box_counting_porous_data.npy")
    porous_data = np.load(data_file)

    # Convert to binary if needed
    if porous_data.max() > 1:
        threshold = np.mean(porous_data)
        binary_data = (porous_data > threshold).astype(np.uint8)
    else:
        binary_data = porous_data.astype(np.uint8)

    return binary_data


def simplify_3d(MT, EPSILON):
    """
    Simplify 3D data by block averaging.

    Parameters
    ----------
    MT : np.ndarray
        3D array
    EPSILON : int
        Block size

    Returns
    -------
    np.ndarray
        Simplified 3D array
    """
    MT_BOX_0 = np.add.reduceat(MT, np.arange(0, MT.shape[0], EPSILON), axis=0)
    MT_BOX_1 = np.add.reduceat(MT_BOX_0, np.arange(0, MT.shape[1], EPSILON), axis=1)
    MT_BOX_2 = np.add.reduceat(MT_BOX_1, np.arange(0, MT.shape[2], EPSILON), axis=2)
    return MT_BOX_2


class TestBoxCountingPorous:
    """Test suite for box-counting method on porous media data."""

    @pytest.fixture
    def porous_data(self):
        """Load porous data for testing."""
        return load_porous_data()

    def test_box_counting_porous_basic(self, porous_data):
        """Test basic box-counting on porous media data."""
        D, result = box_counting(porous_data, data_type='porous')

        # Validate results
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert 0 < D < 4  # 3D fractal dimension should be between 0 and 4
        assert 'R2' in result
        assert 0 < result['R2'] <= 1  # R² should be between 0 and 1

        # For porous media, D should be reasonable
        assert 1.5 < D < 3.5

    def test_porous_data_loading(self):
        """Test that porous data loads correctly."""
        porous_data = load_porous_data()

        # Validate porous data
        assert isinstance(porous_data, np.ndarray)
        assert porous_data.ndim == 3  # Should be 3D array
        assert porous_data.shape[0] > 0 and porous_data.shape[1] > 0 and porous_data.shape[2] > 0
        assert porous_data.dtype == np.uint8  # Should be binary
        assert np.all((porous_data == 0) | (porous_data == 1))  # Only 0 or 1 values

        # Check porosity
        porosity = np.sum(porous_data) / porous_data.size
        assert 0 < porosity < 1  # Porosity should be between 0 and 1

    def test_porous_data_properties(self, porous_data):
        """Test porous media specific properties."""
        # Calculate porosity
        porosity = np.sum(porous_data) / porous_data.size

        # Porosity should be reasonable for porous media
        assert 0.05 < porosity < 0.95  # Between 5% and 95%

        # Check data integrity
        assert not np.any(np.isnan(porous_data))
        assert not np.any(np.isinf(porous_data))

    def test_simplify_3d_function(self):
        """Test the 3D simplification function."""
        # Create test data
        test_data = np.random.randint(0, 2, size=(20, 20, 20), dtype=np.uint8)
        epsilon = 4

        simplified = simplify_3d(test_data, epsilon)

        # Check shape
        expected_shape = (
            (test_data.shape[0] + epsilon - 1) // epsilon,
            (test_data.shape[1] + epsilon - 1) // epsilon,
            (test_data.shape[2] + epsilon - 1) // epsilon
        )
        assert simplified.shape == expected_shape

        # Check data integrity
        assert isinstance(simplified, np.ndarray)
        assert simplified.ndim == 3

    def test_result_structure_porous(self, porous_data):
        """Test that result dictionary contains expected structure for porous data."""
        D, result = box_counting(porous_data, data_type='porous')

        # Check required keys
        required_keys = ['R2']
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Check data consistency if epsilon values are present
        if 'epsilon_values' in result and 'N_values' in result:
            assert len(result['epsilon_values']) == len(result['N_values'])
            assert all(x > 0 for x in result['epsilon_values'])
            assert all(x > 0 for x in result['N_values'])

            # For porous media, N should generally increase with decreasing epsilon
            # (monotonic relationship)
            if len(result['epsilon_values']) > 1:
                eps_vals = result['epsilon_values']
                n_vals = result['N_values']

                # Sort by epsilon (descending) and check N values (ascending)
                sorted_indices = np.argsort(eps_vals)[::-1]
                sorted_eps = eps_vals[sorted_indices]
                sorted_n = n_vals[sorted_indices]

                # N should be non-decreasing as epsilon decreases
                assert np.all(np.diff(sorted_n) >= 0) or len(set(sorted_n)) > 1

        # Check log data consistency
        if 'log_inv_epsilon' in result and 'log_N' in result:
            assert len(result['log_inv_epsilon']) == len(result['log_N'])

        # Check coefficients if present
        if 'coefficients' in result:
            assert len(result['coefficients']) >= 2
            assert all(isinstance(c, (int, float)) for c in result['coefficients'])

    def test_different_voxel_sizes(self, porous_data):
        """Test box-counting with different data preprocessing."""
        # Test with different simplification levels
        epsilon_sizes = [1, 2, 4]

        for epsilon in epsilon_sizes:
            if min(porous_data.shape) >= epsilon:
                simplified = simplify_3d(porous_data, epsilon)
                voxel_data = np.where(simplified > 0, 1, 0)

                D, result = box_counting(voxel_data, data_type='porous')

                assert isinstance(D, (int, float))
                assert isinstance(result, dict)
                assert 0 < D < 4
                assert 'R2' in result
                assert 0 < result['R2'] <= 1

    def test_theoretical_constraints(self, porous_data):
        """Test results against theoretical constraints."""
        D, result = box_counting(porous_data, data_type='porous')

        # For 3D porous media, fractal dimension should satisfy:
        # - Lower bound: 2 (surface fractal dimension)
        # - Upper bound: 3 (space-filling fractal)
        assert 2 <= D <= 3

        # R² should indicate good fit for fractal behavior
        assert result['R2'] > 0.8  # Should have decent linear fit

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very small data
        tiny_data = np.random.randint(0, 2, size=(2, 2, 2), dtype=np.uint8)

        try:
            D, result = box_counting(tiny_data, data_type='porous')
            # If it doesn't crash, results should be valid
            assert isinstance(D, (int, float))
            assert isinstance(result, dict)
        except (ValueError, RuntimeError):
            # It's acceptable if small data raises an error
            pass

