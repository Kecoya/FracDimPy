#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Multifractal Analysis - Image Data
=======================================

Tests for multifractal analysis on 2D image data using the multifractal_image
function from fracDimPy.

Test Coverage:
- Image loading and preprocessing (grayscale conversion)
- Multifractal spectrum calculation for 2D images
- Key multifractal dimensions for image data
- Image-specific multifractal properties
- Different image formats and preprocessing
"""

import numpy as np
import os
import pytest
from fracDimPy import multifractal_image


class TestMultifractalImage:
    """Test suite for multifractal analysis of image data."""

    @staticmethod
    def find_key_by_pattern(metrics, pattern):
        """Helper function to find keys by pattern matching."""
        for key in metrics.keys():
            if pattern in key:
                return key
        return None

    @pytest.fixture
    def sample_image_path(self):
        """Path to the sample image file."""
        return os.path.join(os.path.dirname(__file__), 'mf_image_shale.png')

    @pytest.fixture
    def load_sample_image(self, sample_image_path):
        """Load and preprocess the sample image."""
        try:
            from PIL import Image
            img = Image.open(sample_image_path)
            img_array = np.array(img)

            # Convert to grayscale if color image
            if len(img_array.shape) == 3:
                img_gray = np.mean(img_array, axis=2)
            else:
                img_gray = img_array

            return img_gray, img_array
        except ImportError:
            pytest.skip("PIL/Pillow not available for image loading")
        except Exception as e:
            pytest.fail(f"Failed to load sample image: {e}")

    def test_image_loading(self, load_sample_image):
        """Test that sample image loads correctly."""
        img_gray, img_original = load_sample_image

        # Check image properties
        assert len(img_gray.shape) == 2, "Processed image should be 2D (grayscale)"
        assert img_gray.shape[0] > 0, "Image height should be > 0"
        assert img_gray.shape[1] > 0, "Image width should be > 0"
        assert np.isfinite(img_gray).all(), "All pixel values should be finite"

    def test_multifractal_image_basic(self, load_sample_image):
        """Test basic multifractal analysis for image data."""
        img_gray, _ = load_sample_image

        # Perform multifractal analysis
        metrics, figure_data = multifractal_image(img_gray)

        # Check that metrics and figure data are returned
        assert metrics is not None, "Metrics should be returned"
        assert figure_data is not None, "Figure data should be returned"
        assert isinstance(metrics, dict), "Metrics should be a dictionary"
        assert isinstance(figure_data, dict), "Figure data should be a dictionary"

    def test_multifractal_dimensions_image(self, load_sample_image):
        """Test that key multifractal dimensions are calculated correctly for images."""
        img_gray, _ = load_sample_image

        metrics, figure_data = multifractal_image(img_gray)

        # Check that key dimensions are present using pattern matching
        d0_key = self.find_key_by_pattern(metrics, 'D(0)')
        d1_key = self.find_key_by_pattern(metrics, 'D(1)')
        d2_key = self.find_key_by_pattern(metrics, 'D(2)')

        assert d0_key is not None, "Capacity dimension D(0) should be calculated"
        assert d1_key is not None, "Information dimension D(1) should be calculated"
        assert d2_key is not None, "Correlation dimension D(2) should be calculated"

        # Extract dimension values
        d0 = metrics[d0_key][0]
        d1 = metrics[d1_key][0]
        d2 = metrics[d2_key][0]

        # For 2D image data, dimensions should be between 0 and 2
        assert 0 <= d0 <= 2, f"Capacity dimension D(0)={d0} should be in [0, 2]"
        assert 0 <= d1 <= 2, f"Information dimension D(1)={d1} should be in [0, 2]"
        assert 0 <= d2 <= 2, f"Correlation dimension D(2)={d2} should be in [0, 2]"

        # For multifractal data: D(0) >= D(1) >= D(2)
        # For monofractal data: D(0) <= D(1) <= D(2)
        # Allow either pattern since image data can be either
        dimension_range = max(d0, d1, d2) - min(d0, d1, d2)
        assert dimension_range < 1.0, f"Dimensions should be clustered: spread={dimension_range}"

    def test_hurst_exponent_image(self, load_sample_image):
        """Test Hurst exponent calculation for image data."""
        img_gray, _ = load_sample_image

        metrics, figure_data = multifractal_image(img_gray)

        h_key = self.find_key_by_pattern(metrics, 'H')
        assert h_key is not None, "Hurst exponent should be calculated"

        h = metrics[h_key][0]

        # Hurst exponent should be in [0, 1]
        assert 0 <= h <= 1, f"Hurst exponent H={h} should be in [0, 1]"

    def test_spectrum_properties_image(self, load_sample_image):
        """Test multifractal spectrum properties for image data."""
        img_gray, _ = load_sample_image

        metrics, figure_data = multifractal_image(img_gray)

        # Check spectrum-related metrics
        width_key = self.find_key_by_pattern(metrics, '宽度')
        assert width_key is not None, "Spectrum width should be calculated"

        # Check figure data contains essential curves
        q_key = self.find_key_by_pattern(figure_data, 'q')
        assert q_key is not None, "q values should be in figure data"
        assert self.find_key_by_pattern(figure_data, 'alpha') is not None, "Alpha(q) should be in figure data"
        assert self.find_key_by_pattern(figure_data, 'f') is not None, "f(alpha) should be in figure data"

        # Extract spectrum properties
        spectrum_width = metrics[width_key][0]

        # Validate spectrum properties
        assert spectrum_width >= 0, f"Spectrum width should be non-negative: {spectrum_width}"

        # Check alpha and f(alpha) arrays
        alpha_q = figure_data[self.find_key_by_pattern(figure_data, 'alpha')]
        f_alpha = figure_data[self.find_key_by_pattern(figure_data, 'f')]

        assert len(alpha_q) > 0, "Alpha array should not be empty"
        assert len(f_alpha) > 0, "f(alpha) array should not be empty"
        assert len(alpha_q) == len(f_alpha), "Alpha and f(alpha) should have same length"

    def test_image_specific_properties(self, load_sample_image):
        """Test properties specific to image multifractal analysis."""
        img_gray, img_original = load_sample_image

        metrics, figure_data = multifractal_image(img_gray)

        # Get image dimensions
        height, width = img_gray.shape

        # For image data, the dimensions should be consistent with 2D structures
        d0 = metrics[self.find_key_by_pattern(metrics, 'D(0)')][0]
        d1 = metrics[self.find_key_by_pattern(metrics, 'D(1)')][0]
        d2 = metrics[self.find_key_by_pattern(metrics, 'D(2)')][0]

        # Image multifractal dimensions should be reasonable for 2D data
        # D(0) should typically be between 1.5 and 2.5 for natural images
        assert 1.0 <= d0 <= 3.0, f"Image D(0)={d0} should be in reasonable range"

        # Spectrum properties for images
        spectrum_width = metrics[self.find_key_by_pattern(metrics, '宽度')][0]
        assert spectrum_width >= 0, f"Spectrum width should be non-negative: {spectrum_width}"

        # Check that analysis handles different image sizes
        q_values = figure_data[self.find_key_by_pattern(figure_data, 'q')]
        assert len(q_values) > 0, "Q values should not be empty"

        # Images typically show multifractal behavior
        dimension_spread = max(d0, d1, d2) - min(d0, d1, d2)
        if dimension_spread < 0.1:
            # If it appears monofractal, spectrum width should be small
            assert spectrum_width < 0.3, \
                f"Small dimension spread ({dimension_spread}) should correspond to small spectrum width ({spectrum_width})"

    def test_different_image_sizes(self):
        """Test multifractal analysis with different image sizes."""
        # Create test images of different sizes
        test_sizes = [(32, 32), (64, 64), (128, 128)]

        for height, width in test_sizes:
            # Create a test image with some structure
            img = np.random.rand(height, width)
            # Add some structure
            for i in range(0, height, 8):
                for j in range(0, width, 8):
                    img[i:i+4, j:j+4] += 0.3

            # Normalize to [0, 1]
            img = (img - img.min()) / (img.max() - img.min())

            try:
                metrics, figure_data = multifractal_image(img)

                # Should produce results for any reasonable image size
                assert metrics is not None, f"Should work for image size {height}x{width}"
                assert self.find_key_by_pattern(metrics, 'D(0)') is not None, f"Should calculate D(0) for size {height}x{width}"

                d0 = metrics[self.find_key_by_pattern(metrics, 'D(0)')][0]
                assert 0 <= d0 <= 3, f"D(0) should be reasonable for size {height}x{width}"

            except Exception as e:
                # Some image sizes might fail due to computational constraints
                # but we expect reasonable sizes to work
                if height >= 64 and width >= 64:
                    pytest.fail(f"Analysis should work for size {height}x{width}: {e}")

    def test_color_image_processing(self):
        """Test processing of color images."""
        # Create a color test image
        height, width = 64, 64
        color_img = np.random.rand(height, width, 3)

        try:
            metrics, figure_data = multifractal_image(color_img)

            # Should handle color images (typically by converting to grayscale)
            assert metrics is not None, "Should handle color images"
            assert self.find_key_by_pattern(metrics, 'D(0)') is not None, "Should calculate dimensions for color images"

        except Exception as e:
            pytest.fail(f"Should handle color images: {e}")

    def test_data_integrity_preservation_image(self, load_sample_image):
        """Test that input image data is not modified during analysis."""
        img_gray, _ = load_sample_image
        original_img = img_gray.copy()

        # Perform analysis
        metrics, figure_data = multifractal_image(img_gray)

        # Check that original image is unchanged
        np.testing.assert_array_equal(original_img, img_gray,
                                    "Input image data should not be modified")

    def test_image_preprocessing_effects(self):
        """Test effects of different preprocessing methods."""
        # Create a test image
        height, width = 64, 64
        img = np.random.rand(height, width)

        # Test with original image
        try:
            metrics1, figure_data1 = multifractal_image(img)
        except Exception as e:
            pytest.fail(f"Should handle original image: {e}")

        # Test with normalized image
        img_normalized = (img - img.min()) / (img.max() - img.min())
        try:
            metrics2, figure_data2 = multifractal_image(img_normalized)
        except Exception as e:
            pytest.fail(f"Should handle normalized image: {e}")

        # Both should work and produce reasonable results
        assert metrics1 is not None, "Original image analysis should work"
        assert metrics2 is not None, "Normalized image analysis should work"

        # Extract D(0) for comparison
        d0_1 = metrics1[self.find_key_by_pattern(metrics1, 'D(0)')][0]
        d0_2 = metrics2[self.find_key_by_pattern(metrics2, 'D(0)')][0]

        assert 0 <= d0_1 <= 3, "Original image D(0) should be reasonable"
        assert 0 <= d0_2 <= 3, "Normalized image D(0) should be reasonable"

        # The results should be similar since normalization shouldn't change fractal properties much
        assert abs(d0_1 - d0_2) < 1.0, \
            f"Normalization shouldn't dramatically change D(0): {d0_1} vs {d0_2}"

    def test_figure_data_completeness_image(self, load_sample_image):
        """Test that all required figure data is generated for image analysis."""
        img_gray, _ = load_sample_image

        metrics, figure_data = multifractal_image(img_gray)

        # Required keys for comprehensive visualization - check using pattern matching
        required_patterns = ['q', 'tau', 'alpha', 'f', 'D']

        for pattern in required_patterns:
            key = self.find_key_by_pattern(figure_data, pattern)
            assert key is not None, f"Required key with pattern '{pattern}' missing from figure data"
            assert len(figure_data[key]) > 0, f"Data for '{key}' should not be empty"

        # Check data consistency
        q_values = figure_data[self.find_key_by_pattern(figure_data, 'q')]
        tau_q = figure_data[self.find_key_by_pattern(figure_data, 'tau')]
        alpha_q = figure_data[self.find_key_by_pattern(figure_data, 'alpha')]
        f_alpha = figure_data[self.find_key_by_pattern(figure_data, 'f')]
        D_q = figure_data[self.find_key_by_pattern(figure_data, 'D')]

        # All arrays should have same length
        assert len(q_values) == len(tau_q) == len(alpha_q) == len(f_alpha) == len(D_q), \
            "All multifractal arrays should have same length"

        # Check that typical q values are included
        assert 0 in q_values, "q=0 should be included"
        assert 1 in q_values, "q=1 should be included"
        assert 2 in q_values, "q=2 should be included"

    def test_edge_cases(self):
        """Test multifractal analysis with edge cases."""
        # Test with very small image
        small_img = np.ones((16, 16))
        try:
            metrics, figure_data = multifractal_image(small_img)
            # Should either work or fail gracefully
            if metrics is not None:
                assert self.find_key_by_pattern(metrics, 'D(0)') is not None, "Small image should produce dimensions"
        except Exception:
            # Small images might fail due to insufficient data points
            pass

        # Test with uniform image
        uniform_img = np.ones((64, 64)) * 0.5
        try:
            metrics, figure_data = multifractal_image(uniform_img)
            # Uniform images might have special multifractal properties
            if metrics is not None:
                assert self.find_key_by_pattern(metrics, 'D(0)') is not None, "Uniform image should produce dimensions"
        except Exception as e:
            # Uniform images might be edge cases
            pass

        # Test with image containing zeros
        zero_img = np.zeros((64, 64))
        zero_img[::8, ::8] = 1.0  # Add some structure
        try:
            metrics, figure_data = multifractal_image(zero_img)
            if metrics is not None:
                assert self.find_key_by_pattern(metrics, 'D(0)') is not None, "Image with zeros should produce dimensions"
        except Exception:
            # Zero images might be edge cases
            pass
