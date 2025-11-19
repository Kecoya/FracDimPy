#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pytest configuration file for headless testing.

This file sets up the testing environment to work properly in CI/CD
environments without display capabilities.
"""

import os
import sys

# Set matplotlib to use non-interactive backend for CI
try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass  # matplotlib not available

# Set OpenCV to run in headless mode
try:
    import cv2
    # Disable OpenCV GUI threading
    cv2.setNumThreads(1)
except ImportError:
    pass  # opencv not available

# Set environment variables for headless operation
os.environ['MPLBACKEND'] = 'Agg'

# Print configuration for debugging
print("Pytest configuration loaded:")
print(f"- Matplotlib backend: {matplotlib.get_backend() if 'matplotlib' in sys.modules else 'not imported'}")
print(f"- OpenCV threads: {cv2.getNumThreads() if 'cv2' in sys.modules else 'not imported'}")
print("- Headless mode enabled for CI/CD")