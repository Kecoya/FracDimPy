#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Variogram Method Tests
======================

Test suite for variogram method applied to 1D and 2D data.

The variogram method estimates fractal characteristics by analyzing spatial
variability at different scales, and is widely used in geostatistics,
terrain analysis, and other fields.

Theoretical Background:
- Variogram gamma(h) describes spatial variability at distance h
- For fractal data: gamma(h) ∝ h^(2H)
- H is the Hurst exponent, reflecting data smoothness
- Fractal dimension: D = E + 1 - H (E is embedding dimension)
- 1D: D = 2 - H, 2D: D = 3 - H
"""

import numpy as np
import os
import pytest
from fracDimPy import variogram_method


def load_variogram_data():
    """Load variogram test data files."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file_1d = os.path.join(current_dir, "variogram_1d_data.npy")
    data_file_2d = os.path.join(current_dir, "variogram_surface_data.tif")

    return data_file_1d, data_file_2d


def generate_test_signals():
    """Generate synthetic test signals with known properties."""
    signals = {}

    # Smooth signal (H ≈ 1, D ≈ 1)
    t = np.linspace(0, 10, 1000)
    signals["smooth"] = np.sin(2 * np.pi * t)

    # White noise (H ≈ 0.5, D ≈ 1.5)
    signals["white_noise"] = np.random.randn(1000)

    # Brownian motion (H ≈ 0, D ≈ 2)
    signals["brownian"] = np.cumsum(np.random.randn(1000) * 0.01)

    # Fractional Brownian motion with H=0.7
    # Simple approximation using filtering
    white = np.random.randn(1000)
    # Apply power-law filter 1/f^(2H-1) with H=0.7
    fft = np.fft.fft(white)
    freqs = np.fft.fftfreq(len(white))
    with np.errstate(divide="ignore", invalid="ignore"):
        filter_ = 1.0 / np.abs(freqs) ** (2 * 0.7 - 1)
        filter_[0] = 1  # Set DC component to 1
    fft_filtered = fft * filter_
    signals["fbm"] = np.real(np.fft.ifft(fft_filtered))

    return signals


def generate_test_surfaces():
    """Generate synthetic test surfaces with known properties."""
    surfaces = {}

    # Smooth surface (H ≈ 1, D ≈ 2)
    x, y = np.meshgrid(np.linspace(0, 4 * np.pi, 64), np.linspace(0, 4 * np.pi, 64))
    surfaces["smooth"] = np.sin(x) + np.sin(y)

    # Rough surface (H ≈ 0.5, D ≈ 2.5)
    surfaces["rough"] = np.random.randn(64, 64)

    # Fractional Brownian surface (simplified)
    # Generate 2D noise with power-law characteristics
    noise_2d = np.random.randn(64, 64)
    fft_2d = np.fft.fft2(noise_2d)
    freqs_x = np.fft.fftfreq(64)
    freqs_y = np.fft.fftfreq(64)
    freq_x, freq_y = np.meshgrid(freqs_x, freqs_y)
    freq_magnitude = np.sqrt(freq_x**2 + freq_y**2)

    with np.errstate(divide="ignore", invalid="ignore"):
        filter_2d = 1.0 / (freq_magnitude + 1e-10) ** (0.7)
        filter_2d[0, 0] = 1
    fft_filtered_2d = fft_2d * filter_2d
    surfaces["fbm_2d"] = np.real(np.fft.ifft2(fft_filtered_2d))

    return surfaces


class TestVariogram:
    """Test suite for variogram method."""

    @pytest.fixture
    def variogram_files(self):
        """Get variogram data file paths."""
        return load_variogram_data()

    @pytest.fixture
    def test_signals(self):
        """Generate synthetic test signals."""
        return generate_test_signals()

    @pytest.fixture
    def test_surfaces(self):
        """Generate synthetic test surfaces."""
        return generate_test_surfaces()

    def test_variogram_1d_file(self, variogram_files):
        """Test variogram method on 1D data file."""
        data_file_1d, _ = variogram_files

        try:
            data = np.load(data_file_1d)
            D, result = variogram_method(data)

            # Validate results
            assert isinstance(D, (int, float))
            assert isinstance(result, dict)
            assert 1 <= D <= 2  # For 1D time series
            assert "hurst" in result
            assert "R2" in result
            assert 0 <= result["hurst"] <= 1  # Hurst should be between 0 and 1
            assert 0 < result["R2"] <= 1

            # Check D-H relationship: D = 2 - H
            expected_D = 2 - result["hurst"]
            assert abs(D - expected_D) < 0.1

        except (FileNotFoundError, ImportError):
            pytest.skip("Data file or required libraries not available")

    def test_variogram_2d_file(self, variogram_files):
        """Test variogram method on 2D data file."""
        _, data_file_2d = variogram_files

        try:
            from PIL import Image

            img = Image.open(data_file_2d)
            data = np.array(img)

            # Convert to grayscale if multi-channel image
            if len(data.shape) > 2:
                data = np.mean(data, axis=2)

            D, result = variogram_method(data)

            # Validate results
            assert isinstance(D, (int, float))
            assert isinstance(result, dict)
            assert 2 <= D <= 3  # For 2D surfaces
            assert "hurst" in result
            assert "R2" in result
            assert 0 <= result["hurst"] <= 1  # Hurst should be between 0 and 1
            assert 0 < result["R2"] <= 1

            # Check D-H relationship: D = 3 - H
            expected_D = 3 - result["hurst"]
            assert abs(D - expected_D) < 0.1

        except (FileNotFoundError, ImportError):
            pytest.skip("Data file or required libraries not available")

    def test_smooth_signal(self, test_signals):
        """Test variogram on smooth signal."""
        data = test_signals["smooth"]

        D, result = variogram_method(data)

        # Smooth signal should have H ≈ 1, D ≈ 1
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert pytest.approx(result["hurst"], rel=0.2) == 1.0
        assert pytest.approx(D, rel=0.2) == 1.0
        assert result["R2"] > 0.8

    def test_white_noise_signal(self, test_signals):
        """Test variogram on white noise."""
        data = test_signals["white_noise"]

        D, result = variogram_method(data)

        # White noise should have H ≈ 0.5, D ≈ 1.5
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert pytest.approx(result["hurst"], rel=0.3) == 0.5
        assert pytest.approx(D, rel=0.2) == 1.5
        assert result["R2"] > 0.7

    def test_brownian_motion_signal(self, test_signals):
        """Test variogram on Brownian motion."""
        data = test_signals["brownian"]

        D, result = variogram_method(data)

        # Brownian motion should have H ≈ 0, D ≈ 2
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert pytest.approx(result["hurst"], rel=0.3) == 0.0
        assert pytest.approx(D, rel=0.2) == 2.0
        assert result["R2"] > 0.7

    def test_fbm_signal(self, test_signals):
        """Test variogram on fractional Brownian motion."""
        data = test_signals["fbm"]

        D, result = variogram_method(data)

        # FBM with H=0.7 should have D ≈ 1.3
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert pytest.approx(result["hurst"], rel=0.3) == 0.7
        assert pytest.approx(D, rel=0.3) == 1.3
        assert result["R2"] > 0.6

    def test_smooth_surface(self, test_surfaces):
        """Test variogram on smooth surface."""
        data = test_surfaces["smooth"]

        D, result = variogram_method(data)

        # Smooth surface should have H ≈ 1, D ≈ 2
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert pytest.approx(result["hurst"], rel=0.3) == 1.0
        assert pytest.approx(D, rel=0.2) == 2.0
        assert result["R2"] > 0.7

    def test_rough_surface(self, test_surfaces):
        """Test variogram on rough surface."""
        data = test_surfaces["rough"]

        D, result = variogram_method(data)

        # Rough surface should have H ≈ 0.5, D ≈ 2.5
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert pytest.approx(result["hurst"], rel=0.3) == 0.5
        assert pytest.approx(D, rel=0.2) == 2.5
        assert result["R2"] > 0.6

    def test_fbm_surface(self, test_surfaces):
        """Test variogram on fractional Brownian surface."""
        data = test_surfaces["fbm_2d"]

        D, result = variogram_method(data)

        # FBM surface should have intermediate H and D
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert 0.3 < result["hurst"] < 0.8  # Reasonable H range
        assert 2.2 < D < 2.7  # Corresponding D range
        assert result["R2"] > 0.5

    def test_different_signal_lengths(self):
        """Test variogram on different signal lengths."""
        lengths = [256, 512, 1024]

        for length in lengths:
            # Generate Brownian motion
            data = np.cumsum(np.random.randn(length) * 0.01)

            D, result = variogram_method(data)

            assert isinstance(D, (int, float))
            assert isinstance(result, dict)
            assert 1 <= D <= 2
            assert "hurst" in result
            assert "R2" in result
            assert 0 <= result["hurst"] <= 1
            assert 0 < result["R2"] <= 1

    def test_different_surface_sizes(self):
        """Test variogram on different surface sizes."""
        sizes = [32, 64]

        for size in sizes:
            # Generate random surface
            data = np.random.randn(size, size)

            D, result = variogram_method(data)

            assert isinstance(D, (int, float))
            assert isinstance(result, dict)
            assert 2 <= D <= 3
            assert "hurst" in result
            assert "R2" in result
            assert 0 <= result["hurst"] <= 1
            assert 0 < result["R2"] <= 1

    def test_result_structure_variogram(self, test_signals):
        """Test that result dictionary contains expected structure."""
        data = test_signals["white_noise"]
        D, result = variogram_method(data)

        # Check required keys
        required_keys = ["hurst", "R2"]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Check data consistency if lag values are present
        if "log_lags" in result and "log_variogram" in result:
            assert len(result["log_lags"]) == len(result["log_variogram"])
            assert all(
                x >= 0 for x in result["log_lags"]
            )  # log(lags) should be non-negative for positive lags

        # Check coefficients if present
        if "coefficients" in result:
            assert len(result["coefficients"]) >= 2
            assert all(isinstance(c, (int, float)) for c in result["coefficients"])

    def test_theoretical_constraints(self, test_signals):
        """Test results against theoretical constraints."""
        data = test_signals["brownian"]
        D, result = variogram_method(data)

        # For 1D time series: D = 2 - H
        expected_D = 2 - result["hurst"]
        assert abs(D - expected_D) < 0.05  # Should be very close

        # Hurst exponent should be between 0 and 1
        assert 0 <= result["hurst"] <= 1

        # R² should indicate reasonable fit
        assert result["R2"] > 0.5

    def test_d_h_relationship(self, test_surfaces):
        """Test D-H relationship for 2D surfaces: D = 3 - H."""
        data = test_surfaces["rough"]
        D, result = variogram_method(data)

        # For 2D surfaces: D = 3 - H
        expected_D = 3 - result["hurst"]
        assert abs(D - expected_D) < 0.05  # Should be very close

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very short signal
        short_data = np.random.randn(50)
        try:
            D, result = variogram_method(short_data)
            assert isinstance(D, (int, float))
            assert isinstance(result, dict)
        except (ValueError, RuntimeError):
            # It's acceptable if very short data raises an error
            pass

        # Test with constant signal
        constant_data = np.ones(200) * 0.5
        try:
            D, result = variogram_method(constant_data)
            # Constant signal might give H = 1 (perfectly smooth), D = 1
            if isinstance(D, (int, float)):
                assert pytest.approx(result["hurst"], rel=0.1) == 1.0
                assert pytest.approx(D, rel=0.1) == 1.0
        except (ValueError, RuntimeError):
            pass

    def test_parameter_validation(self, test_signals):
        """Test parameter validation."""
        data = test_signals["white_noise"]

        # Test invalid data
        with pytest.raises((ValueError, TypeError)):
            variogram_method([])

        with pytest.raises((ValueError, TypeError)):
            variogram_method(np.array([]))

        with pytest.raises((ValueError, TypeError)):
            variogram_method("invalid_string")
