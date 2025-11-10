#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sandbox Method
==============

The sandbox method measures fractal dimension by counting pixels
within circles of increasing radius centered on the pattern.

Scaling relationship: log(N) ~ D * log(r)
where N is the pixel count and r is the radius.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from PIL import Image


def sandbox_method(
    image_path: Optional[str] = None,
    image_array: Optional[np.ndarray] = None,
    threshold: Optional[float] = None
) -> Tuple[float, dict]:
    """
    
    
    Parameters
    ----------
    image_path : str, optional
        
    image_array : np.ndarray, optional
        RGB
    threshold : float, optional
        None
        
    Returns
    -------
    dimension : float
        
    result : dict
        
        - 'dimension': 
        - 'r_values': r
        - 'N_values': N
        - 'R2': R
        - 'coefficients': 
        - 'binary_image': 
        
    Examples
    --------
    >>> from fracDimPy import sandbox_method
    >>> # 
    >>> D, result = sandbox_method(image_path='fractal.png')
    >>> print(f": {D:.4f}")
    
    >>> # 
    >>> import numpy as np
    >>> img = np.random.rand(256, 256)
    >>> D, result = sandbox_method(image_array=img)
    
    Notes
    -----
    
    """
    # 
    if image_path is not None:
        # numpy
        img_buffer = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f": {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_array = np.array(img)
    elif image_array is not None:
        img_array = image_array
        if len(img_array.shape) == 3:
            # RGB
            if img_array.shape[2] == 3:
                r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
                img_array = 0.2989 * r + 0.5870 * g + 0.1140 * b
    else:
        raise ValueError("image_pathimage_array")
    
    # 
    if len(img_array.shape) == 3:
        r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
        img_array = 255 - (0.2989 * r + 0.5870 * g + 0.1140 * b)
    else:
        img_array = 255 - img_array
    
    # 
    if threshold is None:
        threshold = np.mean(img_array)
    
    height, width = img_array.shape[:2]
    
    for i in range(height):
        for j in range(width):
            if img_array[i, j] > threshold:
                img_array[i, j] = 255
            else:
                img_array[i, j] = 0
    
    # 
    coordinate_centre = [int(height / 2), int(width / 2)]
    
    # 
    minimum_length = min(height, width)
    length_rate = height / width
    
    if height <= width:
        hang = 1  # 
    else:
        hang = 0  # 
    
    Nl = []  # 
    rl = []  # 
    
    # 2
    for i in range(1, int(np.log(minimum_length) / np.log(2))):
        r = 2 ** i
        
        if hang == 1:
            # 
            r2 = int(r / length_rate)
            pixel_count = np.sum(
                img_array[
                    coordinate_centre[0] - int(r/2) : coordinate_centre[0] - int(r/2) + r + 1,
                    coordinate_centre[1] - int(r2/2) : coordinate_centre[1] - int(r2/2) + r2 + 1,
                ] > 0
            )
        else:
            # 
            r2 = int(r * length_rate)
            pixel_count = np.sum(
                img_array[
                    coordinate_centre[0] - int(r2/2) : coordinate_centre[0] - int(r2/2) + r2 + 1,
                    coordinate_centre[1] - int(r/2) : coordinate_centre[1] - int(r/2) + r + 1,
                ] > 0
            )
        
        Nl.append(pixel_count)
        rl.append(r)
    
    # 
    x = np.log(rl)
    y = np.log(Nl)
    
    coefficients = np.polyfit(x, y, 1)
    f = np.poly1d(coefficients)
    
    dimension = coefficients[0]
    R2 = np.corrcoef(y, f(x))[0, 1] ** 2
    
    result = {
        'dimension': dimension,
        'r_values': rl,
        'N_values': Nl,
        'log_r': x,
        'log_N': y,
        'R2': R2,
        'coefficients': coefficients,
        'method': 'Sandbox',
        'threshold': threshold,
        'image_shape': (height, width),
        'binary_image': img_array
    }
    
    return dimension, result

