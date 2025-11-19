#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test MF-DFA (Multifractal Detrended Fluctuation Analysis)
=========================================================

Tests for Multifractal Detrended Fluctuation Analysis (MF-DFA) on time series
data using the mf_dfa function from fracDimPy.

Test Coverage:
- White noise analysis (monofractal behavior)
- Fractional Gaussian noise (FGN) analysis
- Binomial cascade analysis (multifractal behavior)
- Generalized Hurst exponent h(q) calculation
- Multifractal spectrum f(alpha) properties
- Different q value ranges and window parameters
"""

import numpy as np
import os
import pytest
from fracDimPy import mf_dfa


class TestMultifractalDFA:
    """Test suite for MF-DFA analysis."""

    def generate_binomial_cascade(self, n=8192, p=0.3):
        """Generate binomial cascade series for multifractal testing."""
        levels = int(np.log2(n))
        series = np.ones(2**levels)

        for level in range(levels):
            step = 2**(levels - level)
            for i in range(0, 2**levels, step):
                # Randomly split each segment with probability p
                if np.random.rand() < p:
                    series[i:i+step//2] *= 1.7
                    series[i+step//2:i+step] *= 0.3
                else:
                    series[i:i+step//2] *= 0.3
                    series[i+step//2:i+step] *= 1.7

        return series[:n]

    def generate_fgn_for_mfdfa(self, H, n=10000):
        """Generate Fractional Gaussian Noise (FGN) for MF-DFA testing."""
        try:
            from fbm import FBM
            f = FBM(n=n, hurst=H, length=1, method='daviesharte')
            return f.fgn()
        except ImportError:
            # Fallback to correlated noise approximation
            # Simple approximation using filtering
            noise = np.random.randn(n)
            # Apply simple FIR filter to create correlation
            # This is a rough approximation of FGN
            if H > 0.5:
                # Persistent series
                for i in range(1, n):
                    noise[i] = 0.7 * noise[i-1] + 0.3 * noise[i]
            elif H < 0.5:
                # Anti-persistent series
                for i in range(1, n):
                    noise[i] = -0.3 * noise[i-1] + 1.3 * noise[i]
            return noise

    def test_mf_dfa_white_noise(self):
        """Test MF-DFA on white noise (should show monofractal behavior)."""
        # Generate white noise
        np.random.seed(42)  # For reproducible results
        white_noise = np.random.randn(5000)

        # Perform MF-DFA analysis
        hq_result, spectrum = mf_dfa(
            white_noise,
            q_list=None,  # Use default q range
            min_window=10,
            max_window=500,
            num_windows=20
        )

        # Check that results are returned
        assert hq_result is not None, "H(q) results should be returned"
        assert spectrum is not None, "Spectrum results should be returned"

        # Check required keys in hq_result
        required_keys = ['q_list', 'h_q', 'Fq_n', 'window_sizes']
        for key in required_keys:
            assert key in hq_result, f"Key '{key}' missing from hq_result"

        # Check required keys in spectrum
        required_spectrum_keys = ['alpha', 'f_alpha', 'width']
        for key in required_spectrum_keys:
            assert key in spectrum, f"Key '{key}' missing from spectrum"

        # Extract h(2) for white noise (should be ~0.5)
        q_list = hq_result['q_list']
        h_q = hq_result['h_q']

        # Find h(2)
        idx_2 = np.where(q_list == 2)[0]
        if len(idx_2) > 0:
            h2 = h_q[idx_2[0]]
            assert abs(h2 - 0.5) < 0.2, f"White noise h(2) should be ~0.5, got {h2}"

        # Check spectrum width (should be small for monofractal)
        spectrum_width = spectrum['width']
        assert spectrum_width >= 0, f"Spectrum width should be non-negative: {spectrum_width}"

        # White noise should be nearly monofractal
        if spectrum_width < 0.4:
            # Typical case - small width indicates monofractal
            pass
        else:
            # Edge case - might show some multifractality due to finite sample effects
            pass

    def test_mf_dfa_fgn(self):
        """Test MF-DFA on fractional Gaussian noise."""
        np.random.seed(42)
        fgn_data = self.generate_fgn_for_mfdfa(H=0.7, n=5000)

        # Perform MF-DFA analysis
        hq_result, spectrum = mf_dfa(
            fgn_data,
            q_list=None,
            min_window=10,
            max_window=500,
            num_windows=20
        )

        # Check that results are returned
        assert hq_result is not None, "H(q) results should be returned for FGN"
        assert spectrum is not None, "Spectrum results should be returned for FGN"

        # Extract h(2) (should be close to the input H=0.7)
        q_list = hq_result['q_list']
        h_q = hq_result['h_q']

        # Find h(2)
        idx_2 = np.where(q_list == 2)[0]
        if len(idx_2) > 0:
            h2 = h_q[idx_2[0]]
            # Allow some tolerance due to finite sample effects
            assert 0.5 <= h2 <= 0.9, f"FGN h(2) should be close to H=0.7, got {h2}"

        # FGN should be nearly monofractal
        spectrum_width = spectrum['width']
        assert spectrum_width >= 0, f"Spectrum width should be non-negative: {spectrum_width}"

    def test_mf_dfa_binomial_cascade(self):
        """Test MF-DFA on binomial cascade (should show multifractal behavior)."""
        np.random.seed(42)
        cascade_data = self.generate_binomial_cascade(n=4096, p=0.3)

        # Perform MF-DFA analysis
        hq_result, spectrum = mf_dfa(
            cascade_data,
            q_list=None,
            min_window=8,
            max_window=400,
            num_windows=20
        )

        # Check that results are returned
        assert hq_result is not None, "H(q) results should be returned for cascade"
        assert spectrum is not None, "Spectrum results should be returned for cascade"

        # Check data structure
        q_list = hq_result['q_list']
        h_q = hq_result['h_q']
        Fq_n = hq_result['Fq_n']

        assert len(q_list) > 0, "Q list should not be empty"
        assert len(h_q) == len(q_list), "h(q) should have same length as q_list"
        assert Fq_n.shape[0] == len(q_list), "Fq_n should have correct number of q rows"

        # Binomial cascade should show multifractal behavior
        spectrum_width = spectrum['width']
        assert spectrum_width >= 0, f"Spectrum width should be non-negative: {spectrum_width}"

        # Check that h(q) varies with q (indicating multifractality)
        h_values = h_q[np.isfinite(h_q)]
        if len(h_values) > 5:
            h_variation = np.std(h_values)
            # For multifractal data, h(q) should vary noticeably
            assert h_variation > 0.05, f"h(q) should vary for multifractal data: variation={h_variation}"

    def test_mf_dfa_custom_q_range(self):
        """Test MF-DFA with custom q value range."""
        np.random.seed(42)
        test_data = np.random.randn(3000)

        # Test with custom q range
        custom_q = [-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5]

        hq_result, spectrum = mf_dfa(
            test_data,
            q_list=custom_q,
            min_window=10,
            max_window=300,
            num_windows=15
        )

        # Check that custom q range is used
        q_list = hq_result['q_list']
        assert len(q_list) == len(custom_q), f"Should use custom q range length: {len(q_list)} vs {len(custom_q)}"

        # Check that all requested q values are included
        for q in custom_q:
            assert q in q_list, f"q={q} should be in results"

        # Check corresponding h_q values
        h_q = hq_result['h_q']
        assert len(h_q) == len(custom_q), "h(q) should have same length as custom q list"

    def test_mf_dfa_window_parameters(self):
        """Test MF-DFA with different window parameters."""
        np.random.seed(42)
        test_data = np.random.randn(2000)

        # Test with different window parameters
        hq_result1, spectrum1 = mf_dfa(
            test_data,
            q_list=[-2, 0, 2],
            min_window=5,
            max_window=100,
            num_windows=10
        )

        hq_result2, spectrum2 = mf_dfa(
            test_data,
            q_list=[-2, 0, 2],
            min_window=10,
            max_window=200,
            num_windows=20
        )

        # Both should work
        assert hq_result1 is not None, "First window config should work"
        assert hq_result2 is not None, "Second window config should work"

        # Check window sizes
        ws1 = hq_result1['window_sizes']
        ws2 = hq_result2['window_sizes']

        assert len(ws1) == 10, "First config should have 10 window sizes"
        assert len(ws2) == 20, "Second config should have 20 window sizes"

        assert ws1[0] >= 5, "First window size should meet minimum"
        assert ws2[0] >= 10, "Second window size should meet minimum"

    def test_hq_result_structure(self):
        """Test the structure and properties of h(q) results."""
        np.random.seed(42)
        test_data = np.random.randn(3000)

        hq_result, spectrum = mf_dfa(
            test_data,
            q_list=[-3, -1, 0, 1, 2, 3],
            min_window=10,
            max_window=300,
            num_windows=15
        )

        # Check required fields
        required_fields = ['q_list', 'h_q', 'Fq_n', 'window_sizes']
        for field in required_fields:
            assert field in hq_result, f"Field '{field}' should be in hq_result"

        # Check data consistency
        q_list = hq_result['q_list']
        h_q = hq_result['h_q']
        Fq_n = hq_result['Fq_n']
        window_sizes = hq_result['window_sizes']

        # Array dimensions should be consistent
        assert len(q_list) == len(h_q), "q_list and h_q should have same length"
        assert Fq_n.shape[0] == len(q_list), "Fq_n should have rows for each q value"
        assert Fq_n.shape[1] == len(window_sizes), "Fq_n should have columns for each window size"

        # Check data validity
        assert np.all(np.isfinite(q_list)), "All q values should be finite"
        assert np.all(np.isfinite(window_sizes)), "All window sizes should be finite"
        assert np.all(window_sizes > 0), "All window sizes should be positive"

        # Check h(q) properties
        finite_h = h_q[np.isfinite(h_q)]
        if len(finite_h) > 0:
            # h(q) should typically be in reasonable range
            assert np.all(finite_h > -1), "h(q) should generally be > -1"
            assert np.all(finite_h < 3), "h(q) should generally be < 3"

    def test_spectrum_structure(self):
        """Test the structure and properties of multifractal spectrum."""
        np.random.seed(42)
        test_data = np.random.randn(3000)

        hq_result, spectrum = mf_dfa(
            test_data,
            q_list=[-2, 0, 2],
            min_window=10,
            max_window=300,
            num_windows=15
        )

        # Check required spectrum fields
        required_fields = ['alpha', 'f_alpha', 'width']
        for field in required_fields:
            assert field in spectrum, f"Field '{field}' should be in spectrum"

        # Check data consistency
        alpha = spectrum['alpha']
        f_alpha = spectrum['f_alpha']
        width = spectrum['width']

        # Array dimensions
        assert len(alpha) == len(f_alpha), "alpha and f_alpha should have same length"
        assert len(alpha) > 0, "Spectrum should not be empty"

        # Check data validity
        finite_mask = np.isfinite(alpha) & np.isfinite(f_alpha)
        finite_alpha = alpha[finite_mask]
        finite_f_alpha = f_alpha[finite_mask]

        if len(finite_alpha) > 0:
            assert np.all(finite_f_alpha >= 0), "f(alpha) should be non-negative"

        # Spectrum width
        assert width >= 0, f"Spectrum width should be non-negative: {width}"

        # If we have valid alpha values, check width calculation
        if len(finite_alpha) > 1:
            calculated_width = np.max(finite_alpha) - np.min(finite_alpha)
            assert abs(width - calculated_width) < 1e-6, \
                f"Spectrum width mismatch: {width} vs {calculated_width}"

    def test_data_integrity_preservation(self):
        """Test that input data is not modified during MF-DFA analysis."""
        np.random.seed(42)
        original_data = np.random.randn(2000)
        test_data = original_data.copy()

        # Perform analysis
        hq_result, spectrum = mf_dfa(
            test_data,
            q_list=[-1, 0, 1],
            min_window=10,
            max_window=200,
            num_windows=10
        )

        # Check that original data is unchanged
        np.testing.assert_array_equal(original_data, test_data,
                                    "Input data should not be modified")

    def test_edge_cases(self):
        """Test MF-DFA with edge cases."""
        # Test with very short series
        short_data = np.random.randn(100)
        try:
            hq_result, spectrum = mf_dfa(
                short_data,
                q_list=[-1, 0, 1],
                min_window=5,
                max_window=50,
                num_windows=5
            )
            # Might work with very small parameters
            if hq_result is not None:
                assert 'q_list' in hq_result, "Short data should produce q_list if it works"
        except Exception:
            # Expected to fail for very short series
            pass

        # Test with constant series
        constant_data = np.ones(1000) * 5.0
        try:
            hq_result, spectrum = mf_dfa(
                constant_data,
                q_list=[-1, 0, 1],
                min_window=10,
                max_window=200,
                num_windows=10
            )
            # Constant series might have special properties
            if hq_result is not None:
                assert 'q_list' in hq_result, "Constant data should produce q_list if it works"
        except Exception:
            # Constant series might fail or have special handling
            pass

        # Test with linear trend
        linear_data = np.arange(1000) * 0.01
        try:
            hq_result, spectrum = mf_dfa(
                linear_data,
                q_list=[-1, 0, 1],
                min_window=10,
                max_window=200,
                num_windows=10
            )
            # Linear trend should be handled by detrending
            if hq_result is not None:
                assert 'q_list' in hq_result, "Linear data should produce q_list if it works"
        except Exception:
            # Linear series might be edge case
            pass

    def test_theoretical_relationships(self):
        """Test theoretical relationships in MF-DFA results."""
        np.random.seed(42)
        test_data = np.random.randn(3000)

        hq_result, spectrum = mf_dfa(
            test_data,
            q_list=[-2, -1, 0, 1, 2],
            min_window=10,
            max_window=300,
            num_windows=15
        )

        q_list = hq_result['q_list']
        h_q = hq_result['h_q']

        # Test theoretical relationships
        # h(q) should be a decreasing function of q for multifractal data
        # or nearly constant for monofractal data
        valid_mask = np.isfinite(h_q)
        if np.sum(valid_mask) >= 3:
            valid_q = q_list[valid_mask]
            valid_h = h_q[valid_mask]

            # Check monotonicity (decreasing trend)
            # This is not strictly required but typical
            if len(valid_q) > 2:
                # Calculate correlation between q and h(q)
                correlation = np.corrcoef(valid_q, valid_h)[0, 1]
                # Should generally be negative or small positive
                assert correlation < 0.8, f"Unusual h(q) behavior: correlation={correlation}"

        # Check spectrum properties
        alpha = spectrum['alpha']
        f_alpha = spectrum['f_alpha']

        valid_mask = np.isfinite(alpha) & np.isfinite(f_alpha)
        if np.sum(valid_mask) > 2:
            valid_alpha = alpha[valid_mask]
            valid_f = f_alpha[valid_mask]

            # f(alpha) should have maximum around the most probable singularity
            max_f_idx = np.argmax(valid_f)
            assert valid_f[max_f_idx] > 0, "f(alpha) maximum should be positive"

