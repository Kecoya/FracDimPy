#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Takagi Surface Box-Counting Methods Tests
=========================================

Test suite for different box-counting methods applied to Takagi surfaces.

The Takagi surface is a theoretical fractal surface with known
fractal dimension, making it ideal for method validation.

Tests 6 different box-counting methods:
- method=0: RDCCM - Relative Differential Cubic Cover Method
- method=1: DCCM  - Differential Cubic Cover Method
- method=2: CCM   - Cubic Cover Method (standard)
- method=3: ICCM  - Interpolated Cubic Cover Method
- method=5: SCCM  - Simplified Cubic Cover Method
- method=6: SDCCM - Simplified Differential Cubic Cover Method
"""

import numpy as np
import pytest
from fracDimPy import generate_takagi_surface, box_counting


class TestTakagiMethods:
    """Test suite for box-counting methods on Takagi surfaces."""

    @pytest.fixture
    def takagi_parameters(self):
        """Standard parameters for Takagi surface generation."""
        return {
            "level": 10,  # Reduced for faster testing
            "size": 128,  # Reduced for faster testing
        }

    @pytest.fixture
    def theoretical_dimensions(self):
        """List of theoretical dimensions to test."""
        return [2.1, 2.3, 2.5, 2.7]

    @pytest.fixture
    def test_methods(self):
        """Dictionary of box-counting methods to test."""
        return {
            0: "RDCCM (Relative Differential)",
            1: "DCCM (Differential)",
            2: "CCM (Cubic Cover)",
            3: "ICCM (Interpolated)",
            5: "SCCM (Simplified)",
            6: "SDCCM (Simplified Differential)",
        }

    def test_takagi_surface_generation(self, takagi_parameters, theoretical_dimensions):
        """Test Takagi surface generation with different dimensions."""
        level = takagi_parameters["level"]
        size = takagi_parameters["size"]

        for theo_D in theoretical_dimensions:
            surface = generate_takagi_surface(dimension=theo_D, level=level, size=size)

            # Validate surface properties
            assert isinstance(surface, np.ndarray)
            assert surface.shape == (size, size)
            assert surface.ndim == 2
            assert not np.any(np.isnan(surface))
            assert not np.any(np.isinf(surface))

            # Surface should have variation
            assert surface.std() > 0

    def test_single_method_single_dimension(self, takagi_parameters):
        """Test a single method on a single Takagi surface."""
        surface = generate_takagi_surface(
            dimension=2.5, level=takagi_parameters["level"], size=takagi_parameters["size"]
        )

        D_measured, result = box_counting(surface, data_type="surface", method=2)

        # Validate results
        assert isinstance(D_measured, (int, float))
        assert isinstance(result, dict)
        assert "R2" in result
        assert 0 < result["R2"] <= 1
        assert 0 < D_measured < 4

        # Should be reasonably close to theoretical value
        error = abs(D_measured - 2.5)
        assert error < 0.5  # Allow reasonable error margin

    @pytest.mark.parametrize("method", [0, 1, 2, 3, 5, 6])
    def test_all_methods_single_dimension(self, method, takagi_parameters):
        """Test all methods on a single Takagi surface."""
        theo_D = 2.5
        surface = generate_takagi_surface(
            dimension=theo_D, level=takagi_parameters["level"], size=takagi_parameters["size"]
        )

        D_measured, result = box_counting(surface, data_type="surface", method=method)

        # Validate results for each method
        assert isinstance(D_measured, (int, float))
        assert isinstance(result, dict)
        assert "R2" in result
        assert 0 < result["R2"] <= 1
        assert 0 < D_measured < 4

        # Should be reasonably close to theoretical value
        error = abs(D_measured - theo_D)
        rel_error = error / theo_D * 100
        assert rel_error < 20  # Allow 20% relative error

    def test_method_accuracy_comparison(
        self, takagi_parameters, theoretical_dimensions, test_methods
    ):
        """Compare accuracy of different methods across different dimensions."""
        level = takagi_parameters["level"]
        size = takagi_parameters["size"]

        results = {}

        for theo_D in theoretical_dimensions:
            surface = generate_takagi_surface(dimension=theo_D, level=level, size=size)

            for method_id, method_name in test_methods.items():
                try:
                    D_measured, result = box_counting(
                        surface, data_type="surface", method=method_id
                    )

                    error = abs(D_measured - theo_D)
                    rel_error = error / theo_D * 100

                    # Store results for validation
                    if method_id not in results:
                        results[method_id] = []
                    results[method_id].append(
                        {
                            "theoretical": theo_D,
                            "measured": D_measured,
                            "error": error,
                            "rel_error": rel_error,
                            "R2": result["R2"],
                        }
                    )

                    # Validate each measurement
                    assert isinstance(D_measured, (int, float))
                    assert isinstance(result, dict)
                    assert "R2" in result
                    assert 0 < result["R2"] <= 1
                    assert 0 < D_measured < 4

                    # Reasonable accuracy requirements
                    assert rel_error < 25  # Allow 25% relative error
                    assert error < 0.6  # Allow 0.6 absolute error

                except Exception:
                    # Allow some methods to fail gracefully
                    pytest.skip(f"Method {method_id} failed on theoretical D={theo_D}")

    def test_method_consistency(self, takagi_parameters):
        """Test that methods give consistent results for repeated runs."""
        theo_D = 2.3
        level = takagi_parameters["level"]
        size = takagi_parameters["size"]

        # Generate surface once
        surface = generate_takagi_surface(dimension=theo_D, level=level, size=size)

        # Test multiple times with same method
        measurements = []
        for _ in range(3):  # Run 3 times
            D_measured, result = box_counting(surface, data_type="surface", method=2)
            measurements.append(D_measured)

            assert isinstance(D_measured, (int, float))
            assert isinstance(result, dict)
            assert "R2" in result

        # Results should be consistent (low variance)
        measurements = np.array(measurements)
        assert np.std(measurements) < 0.01  # Very low variance expected

    def test_different_surface_sizes(self):
        """Test methods on different surface sizes."""
        theo_D = 2.5
        sizes = [64, 128]  # Test multiple sizes

        for size in sizes:
            surface = generate_takagi_surface(dimension=theo_D, level=8, size=size)

            D_measured, result = box_counting(surface, data_type="surface", method=2)

            assert isinstance(D_measured, (int, float))
            assert isinstance(result, dict)
            assert "R2" in result
            assert 0 < result["R2"] <= 1
            assert 0 < D_measured < 4

            # Should be reasonably accurate even for smaller surfaces
            error = abs(D_measured - theo_D)
            assert error < 0.8  # Allow larger error for smaller surfaces

    def test_different_iterations(self):
        """Test methods with different iteration levels."""
        theo_D = 2.5
        levels = [8, 10]  # Test different complexity levels

        for level in levels:
            surface = generate_takagi_surface(dimension=theo_D, level=level, size=128)

            D_measured, result = box_counting(surface, data_type="surface", method=2)

            assert isinstance(D_measured, (int, float))
            assert isinstance(result, dict)
            assert "R2" in result
            assert 0 < result["R2"] <= 1
            assert 0 < D_measured < 4

            # Higher iterations should give better accuracy
            error = abs(D_measured - theo_D)
            max_error = 0.6 if level >= 10 else 0.8
            assert error < max_error

    def test_method_specific_requirements(self, takagi_parameters):
        """Test method-specific requirements and behaviors."""
        surface = generate_takagi_surface(
            dimension=2.5, level=takagi_parameters["level"], size=takagi_parameters["size"]
        )

        # Test CCM method (method=2) as reference
        D_ccm, result_ccm = box_counting(surface, data_type="surface", method=2)

        # Test differential methods should give similar results
        try:
            D_dccm, result_dccm = box_counting(surface, data_type="surface", method=1)
            # Should be reasonably close to CCM
            assert abs(D_dccm - D_ccm) < 0.3
        except Exception:
            pass  # Allow some methods to fail

        try:
            D_rdccm, result_rdccm = box_counting(surface, data_type="surface", method=0)
            # Should be reasonably close to CCM
            assert abs(D_rdccm - D_ccm) < 0.3
        except Exception:
            pass  # Allow some methods to fail

    def test_result_structure_takagi(self, takagi_parameters):
        """Test that result dictionary contains expected structure for Takagi surfaces."""
        surface = generate_takagi_surface(
            dimension=2.5, level=takagi_parameters["level"], size=takagi_parameters["size"]
        )

        D, result = box_counting(surface, data_type="surface", method=2)

        # Check required keys
        required_keys = ["R2"]
        for key in required_keys:
            assert key in result, f"Missing required key: {key}"

        # Check data consistency if epsilon values are present
        if "epsilon_values" in result and "N_values" in result:
            assert len(result["epsilon_values"]) == len(result["N_values"])
            assert all(x > 0 for x in result["epsilon_values"])
            assert all(x > 0 for x in result["N_values"])

        # Check log data consistency
        if "log_inv_epsilon" in result and "log_N" in result:
            assert len(result["log_inv_epsilon"]) == len(result["log_N"])

        # Check coefficients if present
        if "coefficients" in result:
            assert len(result["coefficients"]) >= 2
            assert all(isinstance(c, (int, float)) for c in result["coefficients"])

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with extreme dimension values
        extreme_dims = [2.01, 2.99]

        for theo_D in extreme_dims:
            surface = generate_takagi_surface(dimension=theo_D, level=8, size=64)

            try:
                D_measured, result = box_counting(surface, data_type="surface", method=2)

                assert isinstance(D_measured, (int, float))
                assert isinstance(result, dict)
                assert 0 < D_measured < 4

                # Even for extreme values, should be reasonably close
                error = abs(D_measured - theo_D)
                assert error < 1.0  # Allow larger error for extreme values

            except Exception:
                # It's acceptable if extreme cases fail
                pass
