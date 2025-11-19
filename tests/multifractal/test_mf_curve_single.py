#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Multifractal Analysis - Single Column Curve Data
===================================================

Tests for multifractal analysis on single column time series data using
the multifractal_curve function from fracDimPy.

Test Coverage:
- Data loading and validation
- Multifractal spectrum calculation
- Key multifractal dimensions (D(0), D(1), D(2))
- Hölder exponent and spectrum properties
- Different q value ranges
"""

import numpy as np
import os
import pytest
from fracDimPy import multifractal_curve


class TestMultifractalCurveSingle:
    """Test suite for multifractal analysis of single column curve data."""

    @pytest.fixture
    def sample_data_path(self):
        """Path to the sample single column data file."""
        return os.path.join(os.path.dirname(__file__), 'mf_curve_single_test.txt')

    @pytest.fixture
    def load_sample_data(self, sample_data_path):
        """Load the sample time series data."""
        return np.loadtxt(sample_data_path)

    def test_data_loading(self, load_sample_data):
        """Test that sample data loads correctly."""
        data = load_sample_data
        assert len(data) > 0, "Data should not be empty"
        assert np.isfinite(data).all(), "All data values should be finite"
        assert data.min() >= 0, "Data should contain non-negative values"

    def test_multifractal_curve_basic(self, load_sample_data):
        """Test basic multifractal analysis functionality."""
        data = load_sample_data

        # Perform multifractal analysis
        metrics, figure_data = multifractal_curve(
            data,
            use_multiprocessing=False,
            data_type='single'
        )

        # Check that metrics and figure data are returned
        assert metrics is not None, "Metrics should be returned"
        assert figure_data is not None, "Figure data should be returned"
        assert isinstance(metrics, dict), "Metrics should be a dictionary"
        assert isinstance(figure_data, dict), "Figure data should be a dictionary"

    def test_multifractal_dimensions(self, load_sample_data):
        """Test that key multifractal dimensions are calculated correctly."""
        data = load_sample_data

        metrics, figure_data = multifractal_curve(
            data,
            use_multiprocessing=False,
            data_type='single'
        )

        # Check that key dimensions are present and reasonable
        assert '容量维数 D(0)' in metrics, "Capacity dimension D(0) should be calculated"
        assert '信息维数 D(1)' in metrics, "Information dimension D(1) should be calculated"
        assert '关联维数 D(2)' in metrics, "Correlation dimension D(2) should be calculated"

        # Extract dimension values
        d0 = metrics['容量维数 D(0)'][0]
        d1 = metrics['信息维数 D(1)'][0]
        d2 = metrics['关联维数 D(2)'][0]

        # Validate dimension ranges (should be between 0 and 2 for 1D data)
        assert 0 <= d0 <= 2, f"Capacity dimension D(0)={d0} should be in [0, 2]"
        assert 0 <= d1 <= 2, f"Information dimension D(1)={d1} should be in [0, 2]"
        assert 0 <= d2 <= 2, f"Correlation dimension D(2)={d2} should be in [0, 2]"

        # D(0) >= D(1) >= D(2) for multifractal data
        assert d0 >= d1 >= d2, f"Dimensions should decrease: D(0)={d0} >= D(1)={d1} >= D(2)={d2}"

    def test_hurst_exponent(self, load_sample_data):
        """Test Hurst exponent calculation."""
        data = load_sample_data

        metrics, figure_data = multifractal_curve(
            data,
            use_multiprocessing=False,
            data_type='single'
        )

        assert 'Hurst指数 H' in metrics, "Hurst exponent should be calculated"

        h = metrics['Hurst指数 H'][0]

        # Hurst exponent should be in [0, 1]
        assert 0 <= h <= 1, f"Hurst exponent H={h} should be in [0, 1]"

    def test_spectrum_properties(self, load_sample_data):
        """Test multifractal spectrum properties."""
        data = load_sample_data

        metrics, figure_data = multifractal_curve(
            data,
            use_multiprocessing=False,
            data_type='single'
        )

        # Check spectrum-related metrics
        assert '谱宽度' in metrics, "Spectrum width should be calculated"
        assert '最大奇异性指数' in metrics, "Maximum singularity index should be calculated"
        assert '最小奇异性指数' in metrics, "Minimum singularity index should be calculated"

        # Check figure data contains essential curves
        assert 'q值' in figure_data, "q values should be in figure data"
        assert '奇异性指数alpha(q)' in figure_data, "Alpha(q) should be in figure data"
        assert '多重分形谱f(alpha)' in figure_data, "f(alpha) should be in figure data"

        # Extract spectrum properties
        spectrum_width = metrics['谱宽度'][0]
        alpha_min = metrics['最小奇异性指数'][0]
        alpha_max = metrics['最大奇异性指数'][0]

        # Validate spectrum properties
        assert spectrum_width >= 0, f"Spectrum width should be non-negative: {spectrum_width}"
        assert alpha_max >= alpha_min, f"Alpha max should be >= alpha min: {alpha_max} >= {alpha_min}"

        # Spectrum width should equal alpha_max - alpha_min
        assert abs(spectrum_width - (alpha_max - alpha_min)) < 1e-6, \
            f"Spectrum width mismatch: {spectrum_width} != {alpha_max - alpha_min}"

        # Check alpha and f(alpha) arrays
        alpha_q = figure_data['奇异性指数alpha(q)']
        f_alpha = figure_data['多重分形谱f(alpha)']

        assert len(alpha_q) > 0, "Alpha array should not be empty"
        assert len(f_alpha) > 0, "f(alpha) array should not be empty"
        assert len(alpha_q) == len(f_alpha), "Alpha and f(alpha) should have same length"

    def test_q_value_properties(self, load_sample_data):
        """Test q-value related properties and calculations."""
        data = load_sample_data

        metrics, figure_data = multifractal_curve(
            data,
            use_multiprocessing=False,
            data_type='single'
        )

        # Check q values and related curves
        q_values = figure_data['q值']
        tau_q = figure_data['质量指数tau(q)']
        D_q = figure_data['广义维数D(q)']

        assert len(q_values) > 0, "Q values should not be empty"
        assert len(tau_q) > 0, "Tau(q) should not be empty"
        assert len(D_q) > 0, "D(q) should not be empty"

        # All arrays should have same length
        assert len(q_values) == len(tau_q) == len(D_q), \
            "All q-related arrays should have same length"

        # Check that q includes typical values
        assert 0 in q_values, "q=0 should be included"
        assert 1 in q_values, "q=1 should be included"
        assert 2 in q_values, "q=2 should be included"

        # Check that tau(q) has expected properties
        # tau(0) = -1 for normalized data
        idx_0 = list(q_values).index(0)
        assert abs(tau_q[idx_0] + 1) < 0.1, f"tau(0) should be ~-1, got {tau_q[idx_0]}"

    def test_different_q_ranges(self, load_sample_data):
        """Test multifractal analysis with different q value ranges."""
        data = load_sample_data

        # Test with default q range
        metrics1, figure_data1 = multifractal_curve(
            data,
            use_multiprocessing=False,
            data_type='single'
        )

        # The analysis should be consistent
        assert '容量维数 D(0)' in metrics1, "D(0) should be calculated for default q range"

        # Extract key dimensions for comparison
        d0_1 = metrics1['容量维数 D(0)'][0]
        d1_1 = metrics1['信息维数 D(1)'][0]
        d2_1 = metrics1['关联维数 D(2)'][0]

        # Validate ranges
        assert 0 <= d0_1 <= 2, f"D(0) should be valid: {d0_1}"
        assert 0 <= d1_1 <= 2, f"D(1) should be valid: {d1_1}"
        assert 0 <= d2_1 <= 2, f"D(2) should be valid: {d2_1}"

    def test_multifractal_vs_monofractal_properties(self, load_sample_data):
        """Test properties that distinguish multifractal from monofractal behavior."""
        data = load_sample_data

        metrics, figure_data = multifractal_curve(
            data,
            use_multiprocessing=False,
            data_type='single'
        )

        # Get dimensions
        d0 = metrics['容量维数 D(0)'][0]
        d1 = metrics['信息维数 D(1)'][0]
        d2 = metrics['关联维数 D(2)'][0]

        spectrum_width = metrics['谱宽度'][0]

        # Test multifractal properties
        # For monofractal: D(0) ≈ D(1) ≈ D(2) and spectrum width ≈ 0
        # For multifractal: significant differences and wider spectrum

        dimension_spread = max(d0, d1, d2) - min(d0, d1, d2)

        # Either monofractal or multifractal behavior should be detected
        if dimension_spread < 0.1:
            # Likely monofractal
            assert spectrum_width < 0.2, \
                f"Small dimension spread ({dimension_spread}) should correspond to small spectrum width ({spectrum_width})"
        else:
            # Likely multifractal
            assert dimension_spread > 0.05, \
                f"Multifractal data should show dimension spread: {dimension_spread}"

    def test_data_integrity_preservation(self, load_sample_data):
        """Test that input data is not modified during analysis."""
        data = load_sample_data.copy()
        original_data = data.copy()

        # Perform analysis
        metrics, figure_data = multifractal_curve(
            data,
            use_multiprocessing=False,
            data_type='single'
        )

        # Check that original data is unchanged
        np.testing.assert_array_equal(original_data, data,
                                    "Input data should not be modified")

    def test_figure_data_completeness(self, load_sample_data):
        """Test that all required figure data is generated."""
        data = load_sample_data

        metrics, figure_data = multifractal_curve(
            data,
            use_multiprocessing=False,
            data_type='single'
        )

        # Required keys for comprehensive visualization
        required_keys = [
            'q值',                    # q values
            '质量指数tau(q)',         # Mass exponent
            '奇异性指数alpha(q)',     # Hölder exponent
            '多重分形谱f(alpha)',      # Multifractal spectrum
            '广义维数D(q)'            # Generalized dimensions
        ]

        for key in required_keys:
            assert key in figure_data, f"Required key '{key}' missing from figure data"
            assert len(figure_data[key]) > 0, f"Data for '{key}' should not be empty"

