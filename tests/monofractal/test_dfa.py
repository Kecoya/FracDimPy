#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Detrended Fluctuation Analysis (DFA) Tests
==========================================

Test suite for the Detrended Fluctuation Analysis (DFA)
method to calculate the scaling exponent alpha and fractal dimension of time series.

DFA is a method for determining the statistical self-affinity of a signal.
It is useful for analyzing time series that may be affected by trends.

Theoretical Background:
- For FGN (Fractional Gaussian Noise): DFA alpha = H
- For FBM (Fractional Brownian Motion): DFA alpha = H + 1
- Fractal dimension D = 2 - alpha (for 1D signals)
"""

import numpy as np
import os
import pytest
from fracDimPy import dfa


def generate_fgn(H, n=10000):
    """Generate Fractional Gaussian Noise (FGN) from FBM

    Note:
    - For FGN: DFA alpha = H
    - For FBM: DFA alpha = H + 1
    """
    try:
        from fbm import FBM
        f = FBM(n=n, hurst=H, length=1, method='daviesharte')
        # Generate FGN using fgn() method
        fgn_values = f.fgn()
        return fgn_values
    except ImportError:
        # Use fracDimPy FBM generator as fallback
        from fracDimPy import generate_fbm_curve
        D = 2 - H
        fbm_curve, _ = generate_fbm_curve(dimension=D, length=n+1)
        # FGN = diff(FBM)
        fgn = np.diff(fbm_curve)
        return fgn


def generate_white_noise(n=10000):
    """Generate white noise (uncorrelated random signal)"""
    return np.random.randn(n)


def generate_pink_noise(n=10000):
    """Generate pink noise (1/f noise)"""
    # Generate frequency array
    f = np.fft.rfftfreq(n)
    f[0] = 1  # Avoid division by zero

    # Create 1/f power spectrum
    spectrum = 1.0 / np.sqrt(f)

    # Add random phases
    phases = np.random.rand(len(f)) * 2 * np.pi
    complex_spectrum = spectrum * np.exp(1j * phases)

    # Inverse FFT to get time domain signal
    signal = np.fft.irfft(complex_spectrum, n)

    return signal


def generate_random_walk(n=10000):
    """Generate random walk (cumulative sum of random steps)"""
    steps = np.random.choice([-1, 1], size=n)
    return np.cumsum(steps)


def test_dfa_white_noise():
    """Test DFA on white noise (should give alpha ≈ 0.5)."""
    # Set random seed for reproducibility
    np.random.seed(42)
    white_noise = generate_white_noise(n=5000)
    alpha_theory = 0.5

    alpha, result = dfa(
        white_noise,
        min_window=10,
        max_window=500,
        num_windows=20,
        order=1
    )

    # Basic result validation
    assert isinstance(alpha, (float, np.floating)), "Alpha should be a number"
    assert isinstance(result, dict), "Result should be a dictionary"
    assert 'dimension' in result, "Result should contain 'dimension'"
    assert 'r_squared' in result, "Result should contain 'r_squared'"

    # Test theoretical expectation
    assert pytest.approx(alpha, abs=0.15) == alpha_theory, \
        f"White noise DFA alpha {alpha} should be close to theoretical {alpha_theory}"

    # Test fractal dimension relationship: D = 2 - alpha
    D_expected = 2 - alpha_theory
    assert pytest.approx(result['dimension'], abs=0.15) == D_expected, \
        f"Fractal dimension {result['dimension']} should be close to 2 - alpha"

    # Goodness of fit should be reasonable
    assert result['r_squared'] > 0.7, f"R² should be > 0.7 for good fit, got {result['r_squared']}"


def test_dfa_random_walk():
    """Test DFA on random walk (should give alpha ≈ 1.5)."""
    # Set random seed for reproducibility
    np.random.seed(42)
    random_walk = generate_random_walk(n=5000)
    alpha_theory = 1.5

    alpha, result = dfa(
        random_walk,
        min_window=10,
        max_window=500,
        num_windows=20,
        order=1
    )

    # Basic result validation
    assert isinstance(alpha, (float, np.floating)), "Alpha should be a number"
    assert isinstance(result, dict), "Result should be a dictionary"
    assert 'dimension' in result, "Result should contain 'dimension'"
    assert 'r_squared' in result, "Result should contain 'r_squared'"

    # Test theoretical expectation
    assert pytest.approx(alpha, abs=0.2) == alpha_theory, \
        f"Random walk DFA alpha {alpha} should be close to theoretical {alpha_theory}"

    # Test fractal dimension relationship: D = 2 - alpha
    D_expected = 2 - alpha_theory
    assert pytest.approx(result['dimension'], abs=0.2) == D_expected, \
        f"Fractal dimension {result['dimension']} should be close to 2 - alpha"

    # Goodness of fit should be reasonable
    assert result['r_squared'] > 0.7, f"R² should be > 0.7 for good fit, got {result['r_squared']}"


def test_dfa_fgn_anti_persistent():
    """Test DFA on FGN with H=0.3 (anti-persistent, should give alpha ≈ 0.3)."""
    # Set random seed for reproducibility
    np.random.seed(42)
    fgn_03 = generate_fgn(H=0.3, n=5000)
    alpha_theory = 0.3

    alpha, result = dfa(
        fgn_03,
        min_window=10,
        max_window=500,
        num_windows=20,
        order=1
    )

    # Basic result validation
    assert isinstance(alpha, (float, np.floating)), "Alpha should be a number"
    assert isinstance(result, dict), "Result should be a dictionary"

    # Test theoretical expectation
    assert pytest.approx(alpha, abs=0.15) == alpha_theory, \
        f"FGN(H=0.3) DFA alpha {alpha} should be close to theoretical {alpha_theory}"

    # For FGN, alpha should be < 0.5 (anti-persistent)
    assert alpha < 0.5, f"Alpha {alpha} should be < 0.5 for anti-persistent FGN"

    # Goodness of fit should be reasonable
    assert result['r_squared'] > 0.7, f"R² should be > 0.7 for good fit, got {result['r_squared']}"


def test_dfa_fgn_persistent():
    """Test DFA on FGN with H=0.7 (persistent, should give alpha ≈ 0.7)."""
    # Set random seed for reproducibility
    np.random.seed(42)
    fgn_07 = generate_fgn(H=0.7, n=5000)
    alpha_theory = 0.7

    alpha, result = dfa(
        fgn_07,
        min_window=10,
        max_window=500,
        num_windows=20,
        order=1
    )

    # Basic result validation
    assert isinstance(alpha, (float, np.floating)), "Alpha should be a number"
    assert isinstance(result, dict), "Result should be a dictionary"

    # Test theoretical expectation
    assert pytest.approx(alpha, abs=0.15) == alpha_theory, \
        f"FGN(H=0.7) DFA alpha {alpha} should be close to theoretical {alpha_theory}"

    # For FGN, alpha should be > 0.5 (persistent)
    assert alpha > 0.5, f"Alpha {alpha} should be > 0.5 for persistent FGN"

    # Goodness of fit should be reasonable
    assert result['r_squared'] > 0.7, f"R² should be > 0.7 for good fit, got {result['r_squared']}"


def test_dfa_pink_noise():
    """Test DFA on pink noise (should give alpha ≈ 1.0)."""
    # Set random seed for reproducibility
    np.random.seed(42)
    pink_noise = generate_pink_noise(n=5000)
    alpha_theory = 1.0

    alpha, result = dfa(
        pink_noise,
        min_window=10,
        max_window=500,
        num_windows=20,
        order=1
    )

    # Basic result validation
    assert isinstance(alpha, (float, np.floating)), "Alpha should be a number"
    assert isinstance(result, dict), "Result should be a dictionary"

    # Test theoretical expectation (1/f noise should have alpha ≈ 1.0)
    assert pytest.approx(alpha, abs=0.2) == alpha_theory, \
        f"Pink noise DFA alpha {alpha} should be close to theoretical {alpha_theory}"

    # Goodness of fit should be reasonable
    assert result['r_squared'] > 0.6, f"R² should be > 0.6 for good fit, got {result['r_squared']}"


def test_dfa_consistency():
    """Test that DFA produces consistent results."""
    # Set random seed for reproducibility
    np.random.seed(42)
    white_noise = generate_white_noise(n=2000)

    # Calculate DFA multiple times
    results = []
    for _ in range(3):
        alpha, result = dfa(
            white_noise,
            min_window=10,
            max_window=200,
            num_windows=15,
            order=1
        )
        results.append(alpha)

    # Results should be consistent (small variation due to potential randomness)
    std_dev = np.std(results)
    mean_alpha = np.mean(results)

    # Relative standard deviation should be small
    relative_std = std_dev / mean_alpha if mean_alpha != 0 else std_dev
    assert relative_std < 0.05, f"Results should be consistent, relative std: {relative_std}"


def test_dfa_different_parameters():
    """Test DFA with different parameter combinations."""
    # Set random seed for reproducibility
    np.random.seed(42)
    white_noise = generate_white_noise(n=3000)

    # Test with different parameter combinations
    parameter_sets = [
        {},  # default parameters
        {'min_window': 5, 'max_window': 300, 'num_windows': 20, 'order': 1},
        {'min_window': 8, 'max_window': 400, 'num_windows': 25, 'order': 2},
        {'min_window': 15, 'max_window': 500, 'num_windows': 30, 'order': 1},
    ]

    results = []
    for params in parameter_sets:
        alpha, result = dfa(white_noise, **params)
        results.append(alpha)

        # Each calculation should produce valid results
        assert isinstance(alpha, (float, np.floating)), f"Result with params {params} should be numeric"
        assert 0 < alpha < 2, f"Alpha {alpha} should be in reasonable range"
        assert result['r_squared'] > 0.5, f"R² should be reasonable for params {params}"

    # Results should be relatively consistent across parameter variations
    mean_alpha = np.mean(results)
    std_alpha = np.std(results)

    # Allow for some variation but not too much
    assert std_alpha / mean_alpha < 0.3, f"Results should be relatively stable across parameters"


def test_dfa_input_validation():
    """Test DFA with different input validations."""
    # Set random seed for reproducibility
    np.random.seed(42)
    base_data = generate_white_noise(n=1000)

    # Test with different input scenarios
    test_cases = [
        # Normal case
        base_data,
        # Scaled data
        base_data * 10,
        # Offset data
        base_data + 100,
        # Smaller dataset
        base_data[:500],
    ]

    for i, test_data in enumerate(test_cases):
        alpha, result = dfa(
            test_data,
            min_window=5,
            max_window=min(len(test_data) // 4, 100),
            num_windows=15,
            order=1
        )

        # Each test case should produce valid results
        assert isinstance(alpha, (float, np.floating)), f"Test case {i} should produce numeric result"
        assert isinstance(result, dict), f"Test case {i} should produce dictionary result"
        assert 0 < alpha < 2, f"Test case {i}: Alpha {alpha} should be between 0 and 2"
        assert result['r_squared'] > 0.3, f"Test case {i}: R² should be reasonable"


def test_dfa_theoretical_relationships():
    """Test theoretical relationships in DFA results."""
    # Set random seed for reproducibility
    np.random.seed(42)
    white_noise = generate_white_noise(n=2000)

    alpha, result = dfa(
        white_noise,
        min_window=10,
        max_window=200,
        num_windows=15,
        order=1
    )

    # Test fractal dimension relationship: D = 2 - alpha
    D_expected = 2 - alpha
    assert pytest.approx(result['dimension'], rel=1e-10) == D_expected, \
        f"Fractal dimension should satisfy D = 2 - alpha, got D={result['dimension']}, alpha={alpha}"

    # Test bounds on alpha and dimension
    assert 0 < alpha < 2, f"Alpha {alpha} should be between 0 and 2"
    assert 0 < result['dimension'] < 2, f"Dimension {result['dimension']} should be between 0 and 2"

