#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pytest configuration file for headless testing.

This file sets up the testing environment to work properly in CI/CD
environments without display capabilities.
"""

import os
import sys

import numpy as np
import pytest

# Set matplotlib to use non-interactive backend for CI
try:
    import matplotlib

    matplotlib.use("Agg")
except ImportError:
    pass  # matplotlib not available

# Set environment variables for headless operation
os.environ["MPLBACKEND"] = "Agg"

# Print configuration for debugging
print("Pytest configuration loaded:")
print(
    f"- Matplotlib backend: {matplotlib.get_backend() if 'matplotlib' in sys.modules else 'not imported'}"
)
print("- Headless mode enabled for CI/CD")


# ============================================================
# Shared fixtures
# ============================================================


@pytest.fixture
def random_seed():
    """Set and restore numpy random seed for reproducibility."""
    np.random.seed(42)
    yield
    np.random.seed(None)


@pytest.fixture
def white_noise():
    """Standard white noise signal (5000 samples)."""
    np.random.seed(42)
    return np.random.randn(5000)


@pytest.fixture
def random_walk():
    """Standard random walk signal (5000 samples)."""
    np.random.seed(42)
    return np.cumsum(np.random.randn(5000))


@pytest.fixture
def fbm_curve():
    """Standard FBM curve for testing."""
    from fracDimPy import generate_fbm_curve

    return generate_fbm_curve(dimension=1.5, length=2048)


# ============================================================
# Shared signal generation helpers
# ============================================================


def generate_white_noise(n=5000, seed=42):
    """Generate white noise signal."""
    np.random.seed(seed)
    return np.random.randn(n)


def generate_random_walk(n=5000, seed=42):
    """Generate random walk (cumulative sum of white noise)."""
    return np.cumsum(generate_white_noise(n, seed))


def generate_pink_noise(n=5000, seed=42):
    """Generate pink (1/f) noise using FFT filtering."""
    np.random.seed(seed)
    white = np.random.randn(n)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1  # avoid division by zero
    pink_spectrum = np.fft.rfft(white) / np.sqrt(freqs)
    return np.fft.irfft(pink_spectrum, n)


def generate_fgn(H=0.7, n=5000, seed=42):
    """Generate fractional Gaussian noise."""
    np.random.seed(seed)
    # Autocovariance function
    k = np.arange(n)
    gamma = 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k + 1) ** (2 * H))
    gamma = np.concatenate([[1], gamma[1:]])

    # Build Toeplitz covariance matrix (use circulant approximation for speed)
    from scipy.linalg import toeplitz
    cov = toeplitz(gamma)
    cov = (cov + cov.T) / 2

    # Cholesky decomposition
    try:
        L = np.linalg.cholesky(cov)
        return L @ np.random.randn(n)
    except np.linalg.LinAlgError:
        return np.random.randn(n)


def generate_lorenz(n=10000, dt=0.01, sigma=10, rho=28, beta=8 / 3, seed=42):
    """Generate Lorenz attractor time series."""
    np.random.seed(seed)
    x, y, z = 1.0, 1.0, 1.0
    data = np.zeros((n, 3))
    for i in range(n):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        data[i] = [x, y, z]
    return data


def generate_henon_map(n=5000, a=1.4, b=0.3, seed=42):
    """Generate Henon map time series."""
    np.random.seed(seed)
    x, y = 0.1, 0.1
    data = np.zeros((n, 2))
    for i in range(n):
        x_new = 1 - a * x ** 2 + y
        y_new = b * x
        x, y = x_new, y_new
        data[i] = [x, y]
    return data


def generate_binomial_cascade(n=512, seed=42):
    """Generate binomial multiplicative cascade."""
    np.random.seed(seed)
    cascade = np.array([1.0])
    m = 0.6  # multiplicative factor
    while len(cascade) < n:
        cascade = np.column_stack([cascade * m, cascade * (1 - m)]).flatten()
    return cascade[:n]


# ============================================================
# Shared assertion helpers
# ============================================================


def assert_valid_dimension(D, min_val=0, max_val=3):
    """Assert fractal dimension is a valid number in range."""
    assert isinstance(D, (int, float)), f"Expected numeric dimension, got {type(D)}"
    assert min_val <= D <= max_val, f"Dimension {D} outside expected range [{min_val}, {max_val}]"
    assert np.isfinite(D), f"Dimension should be finite, got {D}"


def assert_valid_result_dict(result, required_keys=None):
    """Assert result dict has expected structure."""
    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    if required_keys is None:
        required_keys = ["R2"]
    for key in required_keys:
        assert key in result, f"Missing key '{key}' in result dict"


def assert_result_consistency(func, data, n_runs=3, max_relative_std=0.05, **kwargs):
    """Assert function produces consistent results across runs."""
    results = []
    for _ in range(n_runs):
        D, result = func(data, **kwargs)
        results.append(D)
    std_dev = np.std(results)
    mean_D = np.mean(results)
    if mean_D != 0:
        relative_std = std_dev / abs(mean_D)
        assert relative_std < max_relative_std, (
            f"Results not consistent: std={std_dev:.4f}, relative_std={relative_std:.4f}"
        )
