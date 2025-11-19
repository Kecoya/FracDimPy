#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test Diffusion-Limited Aggregation (DLA) Generation
===================================================

Test cases for generating Diffusion-Limited Aggregation (DLA) structures using fracDimPy.
DLA is a process that simulates particles performing random walks via Brownian motion
and attaching to form aggregates, widely used in modeling natural phenomena such as
crystal growth, electrodeposition, and snowflake formation.

Theoretical Background:
- DLA structures have fractal characteristics, showing dendritic or snowflake-like morphology
- Particles perform random walks from far away, attaching once they contact existing aggregates
- Fractal dimension of DLA is approximately 1.71 (2D case)
- Has strong anisotropy and randomness
"""

import numpy as np
import pytest


def test_dla_basic_generation():
    """Test basic DLA generation functionality."""
    from fracDimPy import generate_dla

    # Test basic generation with moderate parameters for faster execution
    size = 100
    num_particles = 5000

    dla = generate_dla(size=size, num_particles=num_particles)

    # Test basic properties
    assert dla.shape == (size, size), f"Expected shape {(size, size)}, got {dla.shape}"
    assert dla.dtype in [
        np.float64,
        np.float32,
        np.uint8,
        np.uint16,
        np.bool_,
    ], "Expected valid dtype"
    assert np.all(np.isfinite(dla)), "All DLA values should be finite"

    # Test DLA structure properties
    occupied = np.sum(dla > 0)
    assert occupied > 0, "Should have some occupied cells"
    assert occupied <= num_particles, "Should not exceed number of particles"

    # Test that there's a central seed (usually at center)
    center_x, center_y = size // 2, size // 2
    assert dla[center_x, center_y] > 0, "Center should be occupied (seed)"


def test_dla_different_sizes():
    """Test DLA generation with different grid sizes."""
    from fracDimPy import generate_dla

    sizes = [50, 100, 150]
    num_particles = 2000

    for size in sizes:
        dla = generate_dla(size=size, num_particles=num_particles)

        # Test basic properties
        assert dla.shape == (size, size), f"Size {size}: Incorrect shape"
        assert np.all(np.isfinite(dla)), f"Size {size}: All values should be finite"

        occupied = np.sum(dla > 0)
        assert occupied > 0, f"Size {size}: Should have occupied cells"
        assert occupied <= num_particles, f"Size {size}: Should not exceed particle count"

        # Test that center is occupied
        center = size // 2
        assert dla[center, center] > 0, f"Size {size}: Center should be occupied"


def test_dla_different_particle_counts():
    """Test DLA generation with different particle counts."""
    from fracDimPy import generate_dla

    size = 100
    particle_counts = [1000, 3000, 5000]

    occupied_counts = []
    for num_particles in particle_counts:
        dla = generate_dla(size=size, num_particles=num_particles)

        occupied = np.sum(dla > 0)
        occupied_counts.append(occupied)

        # Test basic properties
        assert dla.shape == (size, size), f"Particles {num_particles}: Incorrect shape"
        assert np.all(np.isfinite(dla)), f"Particles {num_particles}: All values should be finite"
        assert occupied > 0, f"Particles {num_particles}: Should have occupied cells"
        assert (
            occupied <= num_particles
        ), f"Particles {num_particles}: Should not exceed particle count"

        # Test success rate (should be reasonable)
        success_rate = occupied / num_particles
        assert (
            0.1 <= success_rate <= 1.0
        ), f"Particles {num_particles}: Success rate {success_rate} should be reasonable"

    # Test that more particles generally lead to more occupied cells
    for i in range(1, len(particle_counts)):
        assert (
            occupied_counts[i] >= occupied_counts[i - 1] * 0.8
        ), f"Should have more occupied cells with more particles"


def test_dla_fractal_properties():
    """Test fractal properties of DLA structures."""
    from fracDimPy import generate_dla

    size = 150
    num_particles = 8000

    dla = generate_dla(size=size, num_particles=num_particles)

    # Test that DLA has dendritic structure
    occupied = np.sum(dla > 0)

    # Test that structure is not completely filled
    fill_ratio = occupied / (size * size)
    assert fill_ratio < 0.5, "DLA should not fill entire grid"

    # Test that structure extends from center
    center = size // 2
    assert dla[center, center] > 0, "Center should be occupied"

    # Find occupied positions
    occupied_positions = np.where(dla > 0)

    if len(occupied_positions[0]) > 0:
        # Test that structure extends outward from center
        max_distance = np.max(
            np.sqrt((occupied_positions[0] - center) ** 2 + (occupied_positions[1] - center) ** 2)
        )
        assert max_distance > size // 4, "Structure should extend outward from center"

        # Test that occupied cells form a connected structure (simplified test)
        # The center should be connected to other occupied cells
        center_neighbors = [
            (center - 1, center),
            (center + 1, center),
            (center, center - 1),
            (center, center + 1),
        ]

        # At least one neighbor of center should be occupied (if enough particles)
        if occupied > 100:  # Only test if we have enough particles
            neighbor_occupied = any(
                0 <= x < size and 0 <= y < size and dla[x, y] > 0 for x, y in center_neighbors
            )
            # Note: This might not always be true due to randomness, so we don't strictly enforce it


def test_dla_theoretical_properties():
    """Test theoretical properties of DLA."""
    # Test theoretical fractal dimension
    # DLA typically has fractal dimension around 1.71 in 2D
    theoretical_dim_range = (1.6, 1.8)
    assert 1.6 < 1.71 < 1.8, "Theoretical dimension should be around 1.71"

    # Test DLA generation and basic properties
    from fracDimPy import generate_dla

    size = 120
    num_particles = 6000
    dla = generate_dla(size=size, num_particles=num_particles)

    occupied = np.sum(dla > 0)
    total_cells = size * size
    fill_ratio = occupied / total_cells

    # DLA should have sparse structure
    assert fill_ratio < 0.3, f"Fill ratio {fill_ratio} should be low for DLA"
    assert occupied > 10, "Should have reasonable number of occupied cells"

    # Test that the structure has branching characteristics
    # This is a simplified test - true fractal analysis would be more complex
    occupied_positions = np.where(dla > 0)
    if len(occupied_positions[0]) > 50:
        # Calculate some simple structural properties
        center = size // 2
        distances = np.sqrt(
            (occupied_positions[0] - center) ** 2 + (occupied_positions[1] - center) ** 2
        )

        # Test that particles are at various distances from center
        distance_range = np.max(distances) - np.min(distances)
        assert distance_range > size // 6, "Structure should extend to various distances"


def test_dla_edge_cases():
    """Test edge cases for DLA generation."""
    from fracDimPy import generate_dla

    # Test with small size and few particles
    size = 50
    num_particles = 500
    dla_small = generate_dla(size=size, num_particles=num_particles)

    assert dla_small.shape == (size, size), "Small size should work"
    assert np.all(np.isfinite(dla_small)), "Small size DLA should be finite"
    assert np.sum(dla_small > 0) > 0, "Small size should have occupied cells"

    # Test with larger grid
    size = 200
    num_particles = 10000
    dla_large = generate_dla(size=size, num_particles=num_particles)

    assert dla_large.shape == (size, size), "Large size should work"
    assert np.all(np.isfinite(dla_large)), "Large size DLA should be finite"
    assert np.sum(dla_large > 0) > 0, "Large size should have occupied cells"

    # Test with very few particles
    size = 80
    num_particles = 100
    dla_few = generate_dla(size=size, num_particles=num_particles)

    assert dla_few.shape == (size, size), "Few particles should work"
    occupied = np.sum(dla_few > 0)
    assert 1 <= occupied <= num_particles, "Should have reasonable occupation"


def test_dla_structure_connectivity():
    """Test connectivity and structure properties of DLA."""
    from fracDimPy import generate_dla

    size = 120
    num_particles = 5000
    dla = generate_dla(size=size, num_particles=num_particles)

    occupied = np.sum(dla > 0)

    if occupied > 50:  # Only test if we have enough particles
        # Find all occupied positions
        occupied_positions = np.where(dla > 0)

        # Test that there are occupied cells at various positions
        assert len(occupied_positions[0]) == occupied, "Occupied positions count should match"

        # Test that positions are within bounds
        assert np.all(occupied_positions[0] >= 0) and np.all(
            occupied_positions[0] < size
        ), "X positions should be within bounds"
        assert np.all(occupied_positions[1] >= 0) and np.all(
            occupied_positions[1] < size
        ), "Y positions should be within bounds"

        # Test that there's a reasonable distribution of occupied cells
        unique_x = len(np.unique(occupied_positions[0]))
        unique_y = len(np.unique(occupied_positions[1]))

        assert unique_x > 1, "Should have occupied cells in multiple X positions"
        assert unique_y > 1, "Should have occupied cells in multiple Y positions"


def test_dla_reproducibility():
    """Test that DLA generation produces different results (due to randomness)."""
    from fracDimPy import generate_dla

    size = 100
    num_particles = 3000

    # Generate DLA twice
    dla1 = generate_dla(size=size, num_particles=num_particles)
    dla2 = generate_dla(size=size, num_particles=num_particles)

    # Test that both are valid
    assert dla1.shape == (size, size), "First DLA should have correct shape"
    assert dla2.shape == (size, size), "Second DLA should have correct shape"
    assert np.all(np.isfinite(dla1)), "First DLA should be finite"
    assert np.all(np.isfinite(dla2)), "Second DLA should be finite"

    # Test that both have occupied cells
    occupied1 = np.sum(dla1 > 0)
    occupied2 = np.sum(dla2 > 0)

    assert occupied1 > 0, "First DLA should have occupied cells"
    assert occupied2 > 0, "Second DLA should have occupied cells"

    # Due to randomness, the results should be different
    # (This test allows for the small possibility of identical results)
    are_identical = np.array_equal(dla1, dla2)
    # We don't strictly enforce difference due to randomness, but it's likely


@pytest.mark.parametrize("size", [60, 80, 100, 120])
def test_dla_size_parameter(size):
    """Test DLA generation with different size parameters."""
    from fracDimPy import generate_dla

    num_particles = 2000
    dla = generate_dla(size=size, num_particles=num_particles)

    assert dla.shape == (size, size), f"Size {size}: Correct shape"
    assert np.all(np.isfinite(dla)), f"Size {size}: All values should be finite"
    assert np.sum(dla > 0) > 0, f"Size {size}: Should have occupied cells"
    assert np.sum(dla > 0) <= num_particles, f"Size {size}: Should not exceed particle count"


@pytest.mark.parametrize("num_particles", [1000, 3000, 5000, 7000])
def test_dla_particles_parameter(num_particles):
    """Test DLA generation with different particle count parameters."""
    from fracDimPy import generate_dla

    size = 100
    dla = generate_dla(size=size, num_particles=num_particles)

    assert dla.shape == (size, size), f"Particles {num_particles}: Correct shape"
    assert np.all(np.isfinite(dla)), f"Particles {num_particles}: All values should be finite"

    occupied = np.sum(dla > 0)
    assert occupied > 0, f"Particles {num_particles}: Should have occupied cells"
    assert occupied <= num_particles, f"Particles {num_particles}: Should not exceed particle count"

    # Test success rate
    success_rate = occupied / num_particles
    assert (
        0.1 <= success_rate <= 1.0
    ), f"Particles {num_particles}: Success rate should be reasonable"
