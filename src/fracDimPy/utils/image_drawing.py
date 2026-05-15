#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Image drawing utilities for fractal pattern generation.

Provides Bresenham line algorithm and coordinate normalization for
rasterizing fractal curves onto images.
"""

import numpy as np


def bresenham_line(image, x0, y0, x1, y1, increment=1, max_value=255):
    """Draw a line on an image using Bresenham's algorithm.

    Parameters
    ----------
    image : numpy.ndarray
        2D image array to draw on.
    x0, y0 : int
        Start pixel coordinates.
    x1, y1 : int
        End pixel coordinates.
    increment : int, optional
        Value to add at each pixel (default 1). Use 0 for simple binary drawing.
    max_value : int, optional
        Maximum pixel value clamp (default 255).
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    size = image.shape[0]

    while True:
        if 0 <= x0 < size and 0 <= y0 < size:
            if increment == 0:
                image[y0, x0] = 1
            else:
                image[y0, x0] = min(max_value, image[y0, x0] + increment)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def normalize_to_pixel_coords(points, size, margin=50):
    """Normalize point coordinates to pixel coordinates within image bounds.

    Parameters
    ----------
    points : numpy.ndarray
        (N, 2) array of (x, y) coordinates.
    size : int
        Image size (width and height).
    margin : int, optional
        Pixel margin from image edge (default 50).

    Returns
    -------
    x_img : numpy.ndarray
        Pixel x-coordinates (int).
    y_img : numpy.ndarray
        Pixel y-coordinates (int), y-axis flipped for image convention.
    """
    x = points[:, 0]
    y = points[:, 1]
    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)

    x_norm = (x - x_min) / (x_max - x_min) if x_max > x_min else np.zeros_like(x)
    y_norm = (y - y_min) / (y_max - y_min) if y_max > y_min else np.zeros_like(y)

    x_img = (margin + x_norm * (size - 2 * margin)).astype(int)
    y_img = (size - margin - y_norm * (size - 2 * margin)).astype(int)

    return x_img, y_img
