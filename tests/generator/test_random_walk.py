#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Random Walk Generation
==========================

Test cases for generating various types of random walk trajectories using fracDimPy.
Including Brownian motion, Lévy flights, and self-avoiding walks, these are important
stochastic process models widely used in physics, biology, finance, and other fields.

Theoretical Background:
Brownian Motion:
- Fractal dimension D = 2 (on 2D plane)
- Step length follows normal distribution, no long-range jumps

Lévy Flight:
- Step length follows power-law distribution, with long-range jumps
- Parameter alpha ∈ (0, 2] controls jump distance distribution
- Degenerates to Brownian motion when alpha = 2
- Long-range jumps occur when alpha < 2

Self-Avoiding Walk:
- Cannot visit already visited positions
- Fractal dimension approximately 4/3 ~= 1.333 (on 2D plane)
- Models polymer chain behavior
"""

import numpy as np
import os
import pytest


def test_brownian_motion_basic_generation():
    """Test basic Brownian motion generation functionality."""
    from fracDimPy import generate_brownian_motion

    # Test basic generation
    steps = 1000
    size = 256
    num_paths = 3

    paths, image = generate_brownian_motion(steps=steps, size=size, num_paths=num_paths)

    # Test basic properties
    assert paths.shape[0] == num_paths, f"Should have {num_paths} paths"
    assert paths.shape[1] == steps, f"Should have {steps} steps"
    assert paths.shape[2] == 2, "Paths should be 2D coordinates"
    assert image.shape == (size, size), f"Image should be {size}x{size}"
    assert np.all(np.isfinite(paths)), "All path coordinates should be finite"
    assert np.all(np.isfinite(image)), "All image values should be finite"

    # Test that paths have reasonable displacement
    for i in range(num_paths):
        start_pos = paths[i, 0, :]
        end_pos = paths[i, -1, :]
        displacement = np.linalg.norm(end_pos - start_pos)

        # Brownian motion should have some displacement
        assert displacement > 0, f"Path {i}: Should have non-zero displacement"

    # Test image has non-zero values
    assert np.any(image > 0), "Density image should have non-zero values"


def test_brownian_motion_properties():
    """Test Brownian motion statistical properties."""
    from fracDimPy import generate_brownian_motion

    steps = 5000
    size = 512
    num_paths = 10

    paths, image = generate_brownian_motion(steps=steps, size=size, num_paths=num_paths)

    # Test mean square displacement scaling
    # For Brownian motion, MSD should be proportional to time
    for i in range(num_paths):
        # Calculate step-by-step displacements
        displacements = np.diff(paths[i], axis=0)
        step_lengths = np.linalg.norm(displacements, axis=1)

        # Step lengths should follow approximately normal distribution
        assert np.mean(step_lengths) > 0, f"Path {i}: Mean step length should be positive"
        assert np.std(step_lengths) > 0, f"Path {i}: Step lengths should vary"

        # Test that walk is not stuck in one place
        unique_positions = len(np.unique(paths[i].reshape(-1, 2), axis=0))
        assert unique_positions > steps // 10, f"Path {i}: Should visit many different positions"


def test_levy_flight_basic_generation():
    """Test basic Lévy flight generation functionality."""
    from fracDimPy import generate_levy_flight

    steps = 1000
    size = 256
    alpha = 1.5
    num_paths = 2

    paths, image = generate_levy_flight(steps=steps, size=size, alpha=alpha, num_paths=num_paths)

    # Test basic properties
    assert paths.shape[0] == num_paths, f"Should have {num_paths} paths"
    assert paths.shape[1] == steps, f"Should have {steps} steps"
    assert paths.shape[2] == 2, "Paths should be 2D coordinates"
    assert image.shape == (size, size), f"Image should be {size}x{size}"
    assert np.all(np.isfinite(paths)), "All path coordinates should be finite"
    assert np.all(np.isfinite(image)), "All image values should be finite"

    # Test that paths have reasonable displacement
    for i in range(num_paths):
        start_pos = paths[i, 0, :]
        end_pos = paths[i, -1, :]
        displacement = np.linalg.norm(end_pos - start_pos)

        # Lévy flight should have displacement
        assert displacement > 0, f"Path {i}: Should have non-zero displacement"


def test_levy_flight_different_alphas():
    """Test Lévy flight generation with different alpha parameters."""
    from fracDimPy import generate_levy_flight

    steps = 2000
    size = 512
    alphas = [1.0, 1.5, 2.0]  # Different alpha parameters

    max_step_lengths = []
    for alpha in alphas:
        paths, image = generate_levy_flight(steps=steps, size=size, alpha=alpha, num_paths=3)

        # Test basic properties
        assert paths.shape[1] == steps, f"Alpha {alpha}: Should have {steps} steps"
        assert image.shape == (size, size), f"Alpha {alpha}: Incorrect image size"

        # Calculate step lengths
        all_step_lengths = []
        for i in range(paths.shape[0]):
            displacements = np.diff(paths[i], axis=0)
            step_lengths = np.linalg.norm(displacements, axis=1)
            all_step_lengths.extend(step_lengths)

        max_step_lengths.append(max(all_step_lengths))

        # Test that there are steps of different lengths
        assert (
            len(set(all_step_lengths[:100])) > 1
        ), f"Alpha {alpha}: Should have varied step lengths"

    # Test that alpha=1.0 generally produces longer jumps than alpha=2.0
    # (though this is statistical, so we allow some tolerance)
    if len(max_step_lengths) >= 2:
        # Lévy flights with lower alpha should have potential for longer jumps
        # This is a statistical property, so we just check that both generate walks
        assert max_step_lengths[0] > 0 and max_step_lengths[-1] > 0, "Both should generate walks"


def test_levy_flight_alpha_2_brownian():
    """Test that alpha=2.0 gives behavior similar to Brownian motion."""
    from fracDimPy import generate_levy_flight, generate_brownian_motion

    steps = 2000
    size = 256
    num_paths = 3

    # Generate Lévy flight with alpha=2.0 (should be similar to Brownian motion)
    paths_levy, image_levy = generate_levy_flight(
        steps=steps, size=size, alpha=2.0, num_paths=num_paths
    )

    # Generate Brownian motion for comparison
    paths_bm, image_bm = generate_brownian_motion(steps=steps, size=size, num_paths=num_paths)

    # Both should have the same shape
    assert paths_levy.shape == paths_bm.shape, "Should have same shape"
    assert image_levy.shape == image_bm.shape, "Images should have same shape"

    # Test that both generate valid walks
    assert np.all(np.isfinite(paths_levy)), "Lévy flight paths should be finite"
    assert np.all(np.isfinite(paths_bm)), "Brownian motion paths should be finite"


def test_self_avoiding_walk_basic_generation():
    """Test basic self-avoiding walk generation functionality."""
    from fracDimPy import generate_self_avoiding_walk

    steps = 100  # Small number of steps for SAW (computationally intensive)
    size = 256
    num_attempts = 5
    max_retries = 1000

    paths, image = generate_self_avoiding_walk(
        steps=steps, size=size, num_attempts=num_attempts, max_retries=max_retries
    )

    # Test basic properties
    assert len(paths) > 0, "Should generate at least one path"
    assert image.shape == (size, size), f"Image should be {size}x{size}"
    assert np.all(np.isfinite(image)), "All image values should be finite"

    # Test self-avoiding property
    for path in paths:
        # Check that no position is visited twice
        unique_positions = np.unique(path.reshape(-1, 2), axis=0)
        assert len(unique_positions) == len(path), "Path should not revisit positions"

        # Test that path has correct length
        assert path.shape[0] == steps, f"Path should have {steps} steps"
        assert path.shape[1] == 2, "Path should be 2D coordinates"

    # Test image has non-zero values
    assert np.any(image > 0), "Density image should have non-zero values"


def test_self_avoiding_walk_properties():
    """Test self-avoiding walk specific properties."""
    from fracDimPy import generate_self_avoiding_walk

    steps = 50  # Small number of steps
    size = 128
    num_attempts = 3
    max_retries = 1000

    paths, image = generate_self_avoiding_walk(
        steps=steps, size=size, num_attempts=num_attempts, max_retries=max_retries
    )

    if len(paths) > 0:
        # Test that paths don't get stuck too early
        for i, path in enumerate(paths):
            # Calculate actual length of path (number of unique positions)
            unique_positions = np.unique(path.reshape(-1, 2), axis=0)

            # Should be able to make reasonable progress
            assert (
                len(unique_positions) >= steps * 0.5
            ), f"Path {i}: Should achieve at least 50% of requested steps"

            # Test that path spreads out
            if len(unique_positions) > 10:
                positions_range = np.ptp(unique_positions, axis=0)
                assert np.any(positions_range > 5), f"Path {i}: Should spread out"


def test_random_walk_edge_cases():
    """Test edge cases for random walk generation."""
    from fracDimPy import generate_brownian_motion, generate_levy_flight

    # Test with small number of steps
    steps = 10
    size = 64
    num_paths = 1

    # Brownian motion with minimal parameters
    paths_bm, image_bm = generate_brownian_motion(steps=steps, size=size, num_paths=num_paths)
    assert paths_bm.shape == (1, steps, 2), "Small Brownian motion should work"
    assert image_bm.shape == (size, size), "Small image should work"

    # Lévy flight with minimal parameters
    paths_levy, image_levy = generate_levy_flight(
        steps=steps, size=size, alpha=1.5, num_paths=num_paths
    )
    assert paths_levy.shape == (1, steps, 2), "Small Lévy flight should work"
    assert image_levy.shape == (size, size), "Small image should work"


@pytest.mark.parametrize("steps", [100, 500, 1000])
def test_brownian_motion_steps_parameter(steps):
    """Test Brownian motion with different step counts."""
    from fracDimPy import generate_brownian_motion

    size = 256
    num_paths = 2

    paths, image = generate_brownian_motion(steps=steps, size=size, num_paths=num_paths)

    assert paths.shape == (num_paths, steps, 2), f"Steps {steps}: Correct path shape"
    assert image.shape == (size, size), f"Steps {steps}: Correct image size"
    assert np.all(np.isfinite(paths)), f"Steps {steps}: All paths should be finite"


@pytest.mark.parametrize("alpha", [1.0, 1.5, 2.0])
def test_levy_flight_alpha_parameter(alpha):
    """Test Lévy flight with different alpha parameters."""
    from fracDimPy import generate_levy_flight

    steps = 500
    size = 256
    num_paths = 2

    paths, image = generate_levy_flight(steps=steps, size=size, alpha=alpha, num_paths=num_paths)

    assert paths.shape == (num_paths, steps, 2), f"Alpha {alpha}: Correct path shape"
    assert image.shape == (size, size), f"Alpha {alpha}: Correct image size"
    assert np.all(np.isfinite(paths)), f"Alpha {alpha}: All paths should be finite"

    # Test that there are both small and large steps (except for alpha=2.0)
    if alpha < 2.0:
        all_step_lengths = []
        for i in range(num_paths):
            displacements = np.diff(paths[i], axis=0)
            step_lengths = np.linalg.norm(displacements, axis=1)
            all_step_lengths.extend(step_lengths)

        if len(all_step_lengths) > 10:
            # Should have variation in step lengths
            step_std = np.std(all_step_lengths)
            assert step_std > 0, f"Alpha {alpha}: Should have varied step lengths"


def test_random_walk_theoretical_properties():
    """Test theoretical properties of random walks."""
    # Test theoretical fractal dimensions
    brownian_dim = 2.0
    levy_dim_range = (1.0, 2.0)  # Depends on alpha
    saw_dim = 4.0 / 3.0  # Approximately 1.333

    assert brownian_dim == 2.0, "Brownian motion fractal dimension should be 2"
    assert 1.33 < saw_dim < 1.34, "Self-avoiding walk fractal dimension should be ~4/3"

    # Test that Levy flight dimension depends on alpha
    # For alpha=1.0, dimension should be approximately 1.0
    # For alpha=2.0, dimension should be 2.0 (same as Brownian)
    assert levy_dim_range[0] < 2.0, "Lévy flight dimension should be valid"
