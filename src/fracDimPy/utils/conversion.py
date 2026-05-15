#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data conversion utilities for fractal analysis.

Provides coordinate-to-matrix conversion, RGB-to-grayscale, and
boundary padding functions shared across analysis modules.
"""

import numpy as np


def rgb_to_grayscale(img_array, invert=False):
    """Convert RGB image array to grayscale.

    Parameters
    ----------
    img_array : numpy.ndarray
        RGB image array with shape (H, W, 3).
    invert : bool, optional
        If True, invert the grayscale values (255 - gray).

    Returns
    -------
    numpy.ndarray
        Grayscale image with shape (H, W).
    """
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    if invert:
        gray = 255 - gray
    return gray


def coordinate_to_matrix_1d(x, epsilon=None):
    """Convert 1D coordinate data to a binary matrix.

    Parameters
    ----------
    x : numpy.ndarray
        1D coordinate values.
    epsilon : float, optional
        Grid resolution. If None, auto-computed from data.

    Returns
    -------
    matrix : numpy.ndarray
        Binary matrix with 1s at coordinate positions.
    epsilon : float
        Grid resolution used.
    """
    if epsilon is None:
        x_range = np.max(x) - np.min(x)
        num = 2 ** (int(np.log2(len(x))) + 1)
        epsilon = x_range / num

    x_indices = np.round((x - np.min(x)) / epsilon).astype(int)
    matrix = np.zeros(np.max(x_indices) + 1, dtype=np.int8)
    matrix[x_indices] = 1
    return matrix, epsilon


def coordinate_to_matrix_2d(x, y, epsilon):
    """Convert 2D coordinate data to a binary matrix.

    Parameters
    ----------
    x, y : numpy.ndarray
        Coordinate arrays.
    epsilon : float
        Grid resolution.

    Returns
    -------
    numpy.ndarray
        Binary 2D matrix with 1s at coordinate positions.
    """
    y_idx = np.round((y - np.min(y)) / epsilon).astype(int)
    x_idx = np.round((x - np.min(x)) / epsilon).astype(int)
    matrix = np.zeros((np.max(y_idx) + 1, np.max(x_idx) + 1), dtype=np.int8)
    matrix[y_idx, x_idx] = 1
    return matrix


def apply_boundary_padding(data, epsilon, mode="valid"):
    """Apply boundary condition padding to data.

    Parameters
    ----------
    data : numpy.ndarray
        Input data array (1D, 2D, or 3D).
    epsilon : int
        Box size used for padding computation.
    mode : str, optional
        Boundary mode: 'valid' (no padding), 'pad' (zero padding),
        'periodic' (wrap), 'reflect' (mirror).

    Returns
    -------
    numpy.ndarray
        Padded data array.
    """
    if mode == "valid":
        return data

    pad_widths = []
    for s in data.shape:
        remainder = s % epsilon
        if remainder > 0:
            pad = epsilon - remainder
            pad_widths.append((0, pad))
        else:
            pad_widths.append((0, 0))

    np_mode = {
        "pad": "constant",
        "periodic": "wrap",
        "reflect": "reflect",
    }.get(mode, "constant")

    return np.pad(data, pad_widths, mode=np_mode)
