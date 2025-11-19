#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Structural Function Method Tests
================================

Test suite for structural function method applied to 1D curve data.

The structural function method is suitable for self-affine curves
and analyzes the scaling behavior of the structure function.
"""

import numpy as np
import os
import pytest
from fracDimPy import structural_function


def load_structural_function_data():
    """Load structural function data from text file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, "structural_function_data.txt")

    # Load X and Y coordinates
    data = np.loadtxt(data_file)

    # Process data format
    if data.ndim == 1:
        # 1D data (only Y values)
        y_data = data
        x_interval = 1.0
    elif data.ndim == 2 and data.shape[1] >= 2:
        # 2D data (X and Y coordinates)
        x_data = data[:, 0]
        y_data = data[:, 1]
        x_interval = float(x_data[1] - x_data[0])
    else:
        raise ValueError(f"Unsupported data format, shape={data.shape}")

    return y_data, x_interval


def generate_test_signals():
    """Generate synthetic test signals with known properties."""
    signals = {}

    # Smooth sine wave (D ≈ 1)
    t = np.linspace(0, 10, 1000)
    signals["sine"] = np.sin(2 * np.pi * t)

    # White noise (D ≈ 2)
    signals["white_noise"] = np.random.randn(1000)

    # Brownian motion (D ≈ 1.5)
    signals["brownian"] = np.cumsum(np.random.randn(1000) * 0.01)

    # Pink noise (1/f noise, D ≈ 1.7)
    # Generate pink noise using simple method
    white = np.random.randn(1000)
    # Apply 1/f filter in frequency domain
    fft = np.fft.fft(white)
    freqs = np.fft.fftfreq(len(white))
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        filter_ = 1.0 / np.sqrt(np.abs(freqs))
        filter_[0] = 0  # Set DC component to 0
    fft_filtered = fft * filter_
    signals["pink_noise"] = np.real(np.fft.ifft(fft_filtered))

    return signals


class TestStructuralFunction:
    """Test suite for structural function method."""

    @pytest.fixture
    def structural_data(self):
        """Load structural function data for testing."""
        return load_structural_function_data()

    @pytest.fixture
    def test_signals(self):
        """Generate synthetic test signals."""
        return generate_test_signals()

    def test_structural_function_basic(self, structural_data):
        """Test basic structural function calculation."""
        y_data, x_interval = structural_data

        D, result = structural_function(y_data, x_interval=x_interval)

        # Validate results
        assert isinstance(D, (int, float))
        assert isinstance(result, dict)
        assert 1 <= D <= 2  # For 1D signals, D should be between 1 and 2
        assert "R2" in result
        assert 0 < result["R2"] <= 1  # R² should be between 0 and 1

    def test_structural_data_loading(self):
        """Test that structural function data loads correctly."""
        y_data, x_interval = load_structural_function_data()

        # Validate data
        assert isinstance(y_data, np.ndarray)
        assert y_data.ndim == 1  # Should be 1D array
        assert len(y_data) > 0
        assert isinstance(x_interval, (int, float))
        assert x_interval > 0

        # Check data integrity
        assert not np.any(np.isnan(y_data))
        assert not np.any(np.isinf(y_data))

    def test_sine_signal(self, test_signals):
        """Test structural function on sine wave signal."""
        y_data = test_signals["sine"]
        x_interval = 10.0 / len(y_data)  # Total time / number of points

        D, result = structural_function(y_data, x_interval=x_interval)

        # Sine wave should have D ≈ 1 (smooth, differentiable)
        assert pytest.approx(D, rel=0.3) == 1.0
        assert result["R2"] > 0.8  # Should have good fit

    def test_white_noise_signal(self, test_signals):
        """Test structural function on white noise signal."""
        y_data = test_signals["white_noise"]
        x_interval = 1.0

        D, result = structural_function(y_data, x_interval=x_interval)

        # White noise should have D ≈ 2 (completely rough)
        assert pytest.approx(D, rel=0.3) == 2.0
        assert result["R2"] > 0.7  # Should have reasonable fit

    def test_brownian_motion_signal(self, test_signals):
        """Test structural function on Brownian motion signal."""
        y_data = test_signals["brownian"]
        x_interval = 1.0

        D, result = structural_function(y_data, x_interval=x_interval)

        # Brownian motion should have D ≈ 1.5
        assert 1.3 <= D <= 1.7  # Allow some tolerance
        assert result["R2"] > 0.7  # Should have reasonable fit

    def test_pink_noise_signal(self, test_signals):
        """Test structural function on pink noise signal."""
        y_data = test_signals["pink_noise"]
        x_interval = 1.0

        D, result = structural_function(y_data, x_interval=x_interval)

        # Pink noise should have D between 1 and 2, typically around 1.7
        assert 1.5 <= D <= 2.0  # Allow reasonable range
        assert result["R2"] > 0.6  # Pink noise might have lower R²

    def test_different_x_intervals(self, structural_data):
        """Test structural function with different x intervals."""
        y_data, _ = structural_data

        x_intervals = [0.1, 0.5, 1.0, 2.0]

        for x_interval in x_intervals:
            D, result = structural_function(y_data, x_interval=x_interval)

            assert isinstance(D, (int, float))
            assert isinstance(result, dict)
            assert 1 <= D <= 2
            assert "R2" in result
            assert 0 < result["R2"] <= 1

    def test_signal_length_effects(self):
        """Test structural function on different signal lengths."""
        lengths = [128, 256, 512, 1024]

        for length in lengths:
            # Generate test signal
            y_data = np.cumsum(np.random.randn(length) * 0.01)
            x_interval = 1.0

            D, result = structural_function(y_data, x_interval=x_interval)

            assert isinstance(D, (int, float))
            assert isinstance(result, dict)
            assert 1 <= D <= 2
            assert "R2" in result
            assert 0 < result["R2"] <= 1

            # Longer signals should generally give better R²
            min_r2 = 0.6 if length < 256 else 0.7
            assert result["R2"] > min_r2

    def test_result_structure_structural(self, structural_data):
        """Test that result dictionary contains expected structure."""
        y_data, x_interval = structural_data
        D, result = structural_function(y_data, x_interval=x_interval)

        # Check required keys
        required_keys = ["R2"]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Check data consistency if tau values are present
        if "tau_values" in result and "S_values" in result:
            assert len(result["tau_values"]) == len(result["S_values"])
            assert all(x > 0 for x in result["tau_values"])
            assert all(
                x >= 0 for x in result["S_values"]
            )  # Structure function should be non-negative

        # Check coefficients if present
        if "coefficients" in result:
            assert len(result["coefficients"]) >= 2
            assert all(isinstance(c, (int, float)) for c in result["coefficients"])

        # Check slope if present
        if "slope" in result:
            assert isinstance(result["slope"], (int, float))
            assert 0 <= result["slope"] <= 2  # Slope should be reasonable

    def test_theoretical_constraints(self, structural_data):
        """Test results against theoretical constraints."""
        y_data, x_interval = structural_data
        D, result = structural_function(y_data, x_interval=x_interval)

        # For 1D signals, fractal dimension should satisfy:
        # - Lower bound: 1 (smooth, differentiable signal)
        # - Upper bound: 2 (completely rough signal)
        assert 1 <= D <= 2

        # R² should indicate reasonable fit
        assert result["R2"] > 0.5

        # The relationship between slope and D: D = 2 - slope/2
        if "slope" in result:
            expected_D = 2 - result["slope"] / 2
            assert abs(D - expected_D) < 0.1  # Should be close

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with very short signal
        short_signal = np.random.randn(50)
        try:
            D, result = structural_function(short_signal, x_interval=1.0)
            assert isinstance(D, (int, float))
            assert isinstance(result, dict)
        except (ValueError, RuntimeError):
            # It's acceptable if very short data raises an error
            pass

        # Test with constant signal
        constant_signal = np.ones(200)
        try:
            D, result = structural_function(constant_signal, x_interval=1.0)
            # Constant signal might give D = 1 or raise error
            if isinstance(D, (int, float)):
                assert D <= 1.5  # Should be low
        except (ValueError, RuntimeError):
            pass

        # Test with linear trend
        linear_signal = np.linspace(0, 10, 200)
        try:
            D, result = structural_function(linear_signal, x_interval=1.0)
            # Linear signal should have D ≈ 1
            if isinstance(D, (int, float)):
                assert pytest.approx(D, rel=0.2) == 1.0
        except (ValueError, RuntimeError):
            pass

    def test_parameter_validation(self, structural_data):
        """Test parameter validation."""
        y_data, _ = structural_data

        # Test invalid x_interval
        with pytest.raises((ValueError, TypeError)):
            structural_function(y_data, x_interval=0)

        with pytest.raises((ValueError, TypeError)):
            structural_function(y_data, x_interval=-1)

        # Test invalid y_data
        with pytest.raises((ValueError, TypeError)):
            structural_function([], x_interval=1.0)

        with pytest.raises((ValueError, TypeError)):
            structural_function(np.array([]), x_interval=1.0)
