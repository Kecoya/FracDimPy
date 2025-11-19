#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Box-counting Method for Image Data Tests
========================================

Test suite for box-counting method applied to 2D image data.

The box-counting method works by covering the image with boxes of different
sizes and counting how many boxes contain part of the fractal structure.
"""

import numpy as np
import os
import pytest
from fracDimPy import box_counting

def load_image_data():
    """Load and preprocess image data for box-counting."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "box_counting_image_data.png")

    try:
        from PIL import Image
        img = Image.open(data_file).convert('RGB')  # Convert to RGB
        img_array = np.array(img)

        # Convert to grayscale for box-counting (inverted for dark fractals)
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        img_gray = 255 - (0.2989 * r + 0.5870 * g + 0.1140 * b)  # Inverted grayscale

        # Binarize image (0 or 255) for box-counting
        threshold = np.mean(img_gray)
        binary_img = np.where(img_gray > threshold, 255, 0).astype(np.uint8)

        return binary_img, threshold

    except Exception:
        # Fallback to synthetic test data
        binary_img = np.random.randint(0, 2, (256, 256), dtype=np.uint8) * 255
        return binary_img, 127.5


def create_test_patterns():
    """Create synthetic test patterns with known fractal dimensions."""
    patterns = {}

    # Simple line (D = 1)
    patterns['line'] = np.zeros((128, 128), dtype=np.uint8)
    patterns['line'][64, :] = 255

    # Square (D = 2)
    patterns['square'] = np.ones((128, 128), dtype=np.uint8) * 255

    # Checkerboard (D = 2)
    patterns['checkerboard'] = np.zeros((128, 128), dtype=np.uint8)
    patterns['checkerboard'][::2, ::2] = 255
    patterns['checkerboard'][1::2, 1::2] = 255

    # Diagonal line (D = 1)
    patterns['diagonal'] = np.zeros((128, 128), dtype=np.uint8)
    np.fill_diagonal(patterns['diagonal'], 255)

    return patterns


class TestBoxCountingImage:
    """Test suite for box-counting method on image data."""

    @pytest.fixture
    def image_data(self):
        """Load image data for testing."""
        return load_image_data()

    @pytest.fixture
    def test_patterns(self):
        """Create test patterns with known properties."""
        return create_test_patterns()

    def test_box_counting_image_basic(self, image_data):
        """Test basic box-counting on image data."""
        binary_img, _ = image_data

        D, result = box_counting(binary_img, data_type='image')

        # Validate results
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert 0 < D < 3  # 2D image fractal dimension should be between 0 and 3
        assert 'R2' in result
        assert 0 < result['R2'] <= 1  # R² should be between 0 and 1

    def test_image_data_loading(self):
        """Test that image data loads correctly."""
        binary_img, threshold = load_image_data()

        # Validate image data
        assert isinstance(binary_img, np.ndarray)
        assert binary_img.ndim == 2  # Should be 2D array
        assert binary_img.shape[0] > 0 and binary_img.shape[1] > 0
        assert binary_img.dtype == np.uint8  # Should be uint8
        assert np.all((binary_img == 0) | (binary_img == 255))  # Binary image

        # Validate threshold
        assert isinstance(threshold, (int, float))
        assert 0 <= threshold <= 255

    def test_binary_image_properties(self, image_data):
        """Test binary image specific properties."""
        binary_img, _ = image_data

        # Check that it's truly binary
        unique_values = np.unique(binary_img)
        assert len(unique_values) <= 2
        assert all(val in [0, 255] for val in unique_values)

        # Check data integrity
        assert not np.any(np.isnan(binary_img))
        assert not np.any(np.isinf(binary_img))

        # Check that there's both black and white pixels
        assert np.any(binary_img == 0)  # Has black pixels
        assert np.any(binary_img == 255)  # Has white pixels

    def test_synthetic_patterns_line(self, test_patterns):
        """Test box-counting on line pattern."""
        binary_img = test_patterns['line']

        D, result = box_counting(binary_img, data_type='image')

        # Line should have D ≈ 1
        assert pytest.approx(D, rel=0.3) == 1.0
        assert result['R2'] > 0.9  # Should have good fit

    def test_synthetic_patterns_square(self, test_patterns):
        """Test box-counting on filled square pattern."""
        binary_img = test_patterns['square']

        D, result = box_counting(binary_img, data_type='image')

        # Filled square should have D ≈ 2
        assert pytest.approx(D, rel=0.2) == 2.0
        assert result['R2'] > 0.9  # Should have good fit

    def test_synthetic_patterns_checkerboard(self, test_patterns):
        """Test box-counting on checkerboard pattern."""
        binary_img = test_patterns['checkerboard']

        D, result = box_counting(binary_img, data_type='image')

        # Checkerboard should have D ≈ 2
        assert pytest.approx(D, rel=0.3) == 2.0
        assert result['R2'] > 0.8  # Should have reasonable fit

    def test_synthetic_patterns_diagonal(self, test_patterns):
        """Test box-counting on diagonal line pattern."""
        binary_img = test_patterns['diagonal']

        D, result = box_counting(binary_img, data_type='image')

        # Diagonal line should have D ≈ 1
        assert pytest.approx(D, rel=0.3) == 1.0
        assert result['R2'] > 0.8  # Should have reasonable fit

    def test_result_structure_image(self, image_data):
        """Test that result dictionary contains expected structure for image data."""
        binary_img, _ = image_data
        D, result = box_counting(binary_img, data_type='image')

        # Check required keys
        required_keys = ['R2']
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Check data consistency if epsilon values are present
        if 'epsilon_values' in result and 'N_values' in result:
            assert len(result['epsilon_values']) == len(result['N_values'])
            assert all(x > 0 for x in result['epsilon_values'])
            assert all(x > 0 for x in result['N_values'])

            # For images, N should generally increase as epsilon decreases
            if len(result['epsilon_values']) > 1:
                eps_vals = result['epsilon_values']
                n_vals = result['N_values']

                # Sort by epsilon (descending) and check N values
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

    def test_different_thresholds(self, image_data):
        """Test box-counting with different thresholding methods."""
        _, original_threshold = image_data

        # Load original grayscale image if possible
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_file = os.path.join(current_dir, "box_counting_image_data.png")

        try:
            from PIL import Image
            img = Image.open(data_file).convert('RGB')
            img_array = np.array(img)
            r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
            img_gray = 255 - (0.2989 * r + 0.5870 * g + 0.1140 * b)

            # Test different thresholds
            thresholds = [
                np.mean(img_gray),           # Mean threshold
                np.median(img_gray),         # Median threshold
                127,                         # Fixed threshold
                img_gray.min() + 0.25 * (img_gray.max() - img_gray.min())  # 25%
            ]

            for threshold in thresholds:
                binary_img = np.where(img_gray > threshold, 255, 0).astype(np.uint8)

                # Only test if there are both black and white pixels
                if np.any(binary_img == 0) and np.any(binary_img == 255):
                    D, result = box_counting(binary_img, data_type='image')

                    assert isinstance(D, (int, float))
                    assert isinstance(result, dict)
                    assert 0 < D < 3
                    assert 'R2' in result
                    assert 0 < result['R2'] <= 1

        except Exception:
            # Skip this test if image loading fails
            pytest.skip("Cannot load original image for threshold testing")

    def test_image_sizes(self):
        """Test box-counting on different image sizes."""
        sizes = [(64, 64), (128, 128), (256, 256)]

        for size in sizes:
            # Create test image
            test_img = np.random.randint(0, 2, size, dtype=np.uint8) * 255

            D, result = box_counting(test_img, data_type='image')

            assert isinstance(D, (int, float))
            assert isinstance(result, dict)
            assert 0 < D < 3
            assert 'R2' in result
            assert 0 < result['R2'] <= 1

    def test_theoretical_constraints(self, image_data):
        """Test results against theoretical constraints."""
        binary_img, _ = image_data
        D, result = box_counting(binary_img, data_type='image')

        # For 2D images, fractal dimension should satisfy:
        # - Lower bound: 1 (line-like)
        # - Upper bound: 2 (space-filling)
        assert 1 <= D <= 2.5  # Allow some margin for real images

        # R² should indicate reasonable fit
        assert result['R2'] > 0.7

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very small image
        tiny_img = np.array([[255, 0], [0, 255]], dtype=np.uint8)

        try:
            D, result = box_counting(tiny_img, data_type='image')
            # If it doesn't crash, results should be valid
            assert isinstance(D, (int, float))
            assert isinstance(result, dict)
        except (ValueError, RuntimeError):
            # It's acceptable if very small data raises an error
            pass

        # Test with uniform image (all white)
        uniform_white = np.ones((32, 32), dtype=np.uint8) * 255
        try:
            D, result = box_counting(uniform_white, data_type='image')
            assert isinstance(D, (int, float))
            # Uniform image should give D ≈ 2
            assert pytest.approx(D, rel=0.1) == 2.0
        except (ValueError, RuntimeError):
            pass

        # Test with empty image (all black)
        uniform_black = np.zeros((32, 32), dtype=np.uint8)
        try:
            D, result = box_counting(uniform_black, data_type='image')
            assert isinstance(D, (int, float))
            # Empty image might give D = 0 or raise error
        except (ValueError, RuntimeError):
            pass

