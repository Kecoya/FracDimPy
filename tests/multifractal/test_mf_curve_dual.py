#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Multifractal Analysis - X-Y Curve Data
===========================================

Tests for multifractal analysis on dual column X-Y curve data using
the multifractal_curve function from fracDimPy.

Test Coverage:
- X-Y coordinate data loading and validation
- Multifractal curve analysis for spatial data
- Key multifractal dimensions for 2D curves
- Hölder exponent and spectrum properties for spatial data
- Data integrity and error handling
"""

import numpy as np
import pandas as pd
import os
import pytest
from fracDimPy import multifractal_curve


class TestMultifractalCurveDual:
    """Test suite for multifractal analysis of dual column X-Y curve data."""

    @pytest.fixture
    def sample_data_path(self):
        """Path to the sample dual column data file."""
        return os.path.join(os.path.dirname(__file__), "mf_curve_dual_1.xlsx")

    @pytest.fixture
    def load_sample_data(self, sample_data_path):
        """Load the sample X-Y curve data."""
        df = pd.read_excel(sample_data_path)
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        return (x, y), df

    def test_data_loading(self, load_sample_data):
        """Test that sample X-Y data loads correctly."""
        (x, y), df = load_sample_data

        # Check DataFrame
        assert len(df) > 0, "DataFrame should not be empty"
        assert df.shape[1] >= 2, "DataFrame should have at least 2 columns"

        # Check coordinate arrays
        assert len(x) > 0, "X coordinates should not be empty"
        assert len(y) > 0, "Y coordinates should not be empty"
        assert len(x) == len(y), "X and Y should have same length"

        # Check data validity
        assert np.isfinite(x).all(), "All X values should be finite"
        assert np.isfinite(y).all(), "All Y values should be finite"

    def test_multifractal_curve_dual_basic(self, load_sample_data):
        """Test basic multifractal analysis for dual column data."""
        (x, y), _ = load_sample_data

        # Perform multifractal analysis on X-Y curve
        metrics, figure_data = multifractal_curve(
            (x, y), use_multiprocessing=False, data_type="dual"
        )

        # Check that metrics and figure data are returned
        assert metrics is not None, "Metrics should be returned"
        assert figure_data is not None, "Figure data should be returned"
        assert isinstance(metrics, dict), "Metrics should be a dictionary"
        assert isinstance(figure_data, dict), "Figure data should be a dictionary"

    def test_multifractal_dimensions_dual(self, load_sample_data):
        """Test that key multifractal dimensions are calculated correctly for curves."""
        (x, y), _ = load_sample_data

        metrics, figure_data = multifractal_curve(
            (x, y), use_multiprocessing=False, data_type="dual"
        )

        # Check that key dimensions are present (keys might have different names for dual data)
        dimension_keys = [key for key in metrics.keys() if "D(" in key]
        assert len(dimension_keys) > 0, "At least one dimension should be calculated"

        # Extract dimension values (try different possible key formats)
        d0 = d1 = d2 = None
        for key, value in metrics.items():
            if "D(0)" in key or "0" in key:
                d0 = value[0]
            elif "D(1)" in key or "1" in key:
                d1 = value[0]
            elif "D(2)" in key or "2" in key:
                d2 = value[0]

        # For curve data, dimensions should be between 1 and 2
        if d0 is not None:
            assert 1 <= d0 <= 2, f"Capacity dimension D(0)={d0} should be in [1, 2] for curves"
        if d1 is not None:
            assert 1 <= d1 <= 2, f"Information dimension D(1)={d1} should be in [1, 2] for curves"
        if d2 is not None:
            assert 1 <= d2 <= 2, f"Correlation dimension D(2)={d2} should be in [1, 2] for curves"

        # D(0) >= D(1) >= D(2) for multifractal data (if all available)
        if d0 is not None and d1 is not None and d2 is not None:
            assert (
                d0 >= d1 >= d2
            ), f"Dimensions should decrease: D(0)={d0} >= D(1)={d1} >= D(2)={d2}"

    def test_hurst_exponent_dual(self, load_sample_data):
        """Test Hurst exponent calculation for dual column data."""
        (x, y), _ = load_sample_data

        metrics, figure_data = multifractal_curve(
            (x, y), use_multiprocessing=False, data_type="dual"
        )

        # Check for Hurst exponent (key might vary)
        h_keys = [key for key in metrics.keys() if "H" in key.lower() or "hurst" in key.lower()]
        if h_keys:
            h = metrics[h_keys[0]][0]
            # Hurst exponent should be in [0, 1]
            assert 0 <= h <= 1, f"Hurst exponent H={h} should be in [0, 1]"

    def test_spectrum_properties_dual(self, load_sample_data):
        """Test multifractal spectrum properties for curve data."""
        (x, y), _ = load_sample_data

        metrics, figure_data = multifractal_curve(
            (x, y), use_multiprocessing=False, data_type="dual"
        )

        # Check for spectrum-related keys
        spectrum_keys = [
            key
            for key in metrics.keys()
            if "width" in key.lower() or "谱" in key or "alpha" in key.lower()
        ]
        if spectrum_keys:
            # Should have at least some spectrum information
            assert len(spectrum_keys) > 0, "Spectrum information should be calculated"

        # Check figure data contains essential curves
        # Keys might vary in format, so look for key patterns
        q_keys = [
            key for key in figure_data.keys() if "q" in key.lower() and len(figure_data[key]) > 0
        ]
        alpha_keys = [key for key in figure_data.keys() if "alpha" in key.lower()]
        f_keys = [key for key in figure_data.keys() if "f(" in key or "f_alpha" in key]

        assert len(q_keys) > 0, "q values should be in figure data"
        assert len(alpha_keys) > 0, "Alpha values should be in figure data"

        # Get actual data
        q_key = q_keys[0]  # Use first available q key
        q_values = figure_data[q_key]
        assert len(q_values) > 0, "Q values should not be empty"

        # Check that typical q values are included
        assert 0 in q_values or any(abs(q - 0) < 1e-10 for q in q_values), "q=0 should be included"
        assert 1 in q_values or any(abs(q - 1) < 1e-10 for q in q_values), "q=1 should be included"
        assert 2 in q_values or any(abs(q - 2) < 1e-10 for q in q_values), "q=2 should be included"

    def test_curve_data_specific_properties(self, load_sample_data):
        """Test properties specific to X-Y curve multifractal analysis."""
        (x, y), _ = load_sample_data

        metrics, figure_data = multifractal_curve(
            (x, y), use_multiprocessing=False, data_type="dual"
        )

        # For curve data, the analysis should treat the 2D coordinates as a single curve
        # The dimensions should be consistent with a 1D curve embedded in 2D space

        # Get available dimension keys
        dim_keys = [key for key in metrics.keys() if "D(" in key]
        assert len(dim_keys) > 0, "Should calculate fractal dimensions for curve"

        # Test data format consistency
        q_keys = [
            key for key in figure_data.keys() if "q" in key.lower() and len(figure_data[key]) > 0
        ]
        tau_keys = [key for key in figure_data.keys() if "tau" in key.lower()]
        d_keys = [key for key in figure_data.keys() if "d(" in key.lower() or "d(q)" in key.lower()]

        if len(q_keys) > 0:
            q_key = q_keys[0]
            q_values = figure_data[q_key]

            # All curve arrays should have same length
            if len(tau_keys) > 0:
                tau_key = tau_keys[0]
                assert len(q_values) == len(
                    figure_data[tau_key]
                ), "q and tau(q) should have same length"

            if len(d_keys) > 0:
                d_key = d_keys[0]
                assert len(q_values) == len(
                    figure_data[d_key]
                ), "q and D(q) should have same length"

    def test_data_integrity_preservation_dual(self, load_sample_data):
        """Test that input curve data is not modified during analysis."""
        (x, y), _ = load_sample_data
        original_x = x.copy()
        original_y = y.copy()

        # Perform analysis
        metrics, figure_data = multifractal_curve(
            (x, y), use_multiprocessing=False, data_type="dual"
        )

        # Check that original data is unchanged
        np.testing.assert_array_equal(original_x, x, "Input X data should not be modified")
        np.testing.assert_array_equal(original_y, y, "Input Y data should not be modified")

    def test_different_data_formats(self, load_sample_data):
        """Test multifractal analysis with different input data formats."""
        (x, y), _ = load_sample_data

        # Test with tuple format
        metrics1, figure_data1 = multifractal_curve(
            (x, y), use_multiprocessing=False, data_type="dual"
        )

        # Test with numpy array format
        xy_array = np.column_stack([x, y])
        metrics2, figure_data2 = multifractal_curve(
            xy_array, use_multiprocessing=False, data_type="dual"
        )

        # Both should work and produce results
        assert metrics1 is not None, "Tuple format should work"
        assert metrics2 is not None, "Array format should work"

        # Results should be consistent
        # Compare available dimension keys
        dim_keys1 = [key for key in metrics1.keys() if "D(" in key]
        dim_keys2 = [key for key in metrics2.keys() if "D(" in key]

        assert len(dim_keys1) > 0, "First format should produce dimensions"
        assert len(dim_keys2) > 0, "Second format should produce dimensions"

    def test_curve_vs_time_series_properties(self, load_sample_data):
        """Test differences between curve data and time series data analysis."""
        (x, y), _ = load_sample_data

        # Analyze as curve data (dual)
        metrics_dual, figure_data_dual = multifractal_curve(
            (x, y), use_multiprocessing=False, data_type="dual"
        )

        # Analyze as time series (using only x component)
        metrics_single, figure_data_single = multifractal_curve(
            x, use_multiprocessing=False, data_type="single"
        )

        # Both should produce results
        assert metrics_dual is not None, "Dual analysis should work"
        assert metrics_single is not None, "Single analysis should work"

        # Check that we get meaningful results from both
        dim_keys_dual = [key for key in metrics_dual.keys() if "D(" in key]
        dim_keys_single = [key for key in metrics_single.keys() if "D(" in key]

        assert len(dim_keys_dual) > 0, "Dual should produce dimensions"
        assert len(dim_keys_single) > 0, "Single should produce dimensions"

    def test_figure_data_completeness_dual(self, load_sample_data):
        """Test that all required figure data is generated for curve analysis."""
        (x, y), _ = load_sample_data

        metrics, figure_data = multifractal_curve(
            (x, y), use_multiprocessing=False, data_type="dual"
        )

        # Should have essential multifractal analysis data
        # Keys might vary, so check for patterns
        q_keys = [
            key for key in figure_data.keys() if "q" in key.lower() and len(figure_data[key]) > 0
        ]
        tau_keys = [key for key in figure_data.keys() if "tau" in key.lower()]
        alpha_keys = [key for key in figure_data.keys() if "alpha" in key.lower()]
        f_keys = [key for key in figure_data.keys() if "f(" in key or "f_alpha" in key]
        d_keys = [key for key in figure_data.keys() if "d(" in key.lower() or "d(q)" in key.lower()]

        assert len(q_keys) > 0, "q values should be in figure data"
        assert len(tau_keys) > 0, "tau(q) should be in figure data"
        assert len(alpha_keys) > 0, "alpha(q) should be in figure data"
        assert len(f_keys) > 0, "f(alpha) should be in figure data"
        assert len(d_keys) > 0, "D(q) should be in figure data"

        # Check data consistency
        q_key = q_keys[0]
        q_values = figure_data[q_key]

        # All arrays should have same length as q_values
        for key in tau_keys + alpha_keys + f_keys + d_keys:
            assert len(figure_data[key]) == len(
                q_values
            ), f"{key} should have same length as q values"
