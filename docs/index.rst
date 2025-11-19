FracDimPy Documentation
=======================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   quickstart
   api/modules
   examples
   contributing

Introduction
============

FracDimPy is a comprehensive Python package for fractal analysis that provides both monofractal and multifractal analysis tools. The package implements classical monofractal methods (Box-Counting, Hurst exponent, Detrended Fluctuation Analysis) and advanced multifractal techniques (MF-DFA, multifractal spectrum).

Key Features
------------

* **Comprehensive Coverage**: Monofractal, multifractal, and fractal generation methods
* **Multiple Data Types**: Support for 1D time series, 2D images, and 3D surfaces
* **Validation Framework**: Built-in generators for synthetic data with known properties
* **Consistent API**: Unified interface across different methods

Installation
============

From PyPI (recommended):

.. code-block:: bash

   pip install FracDimPy

From source:

.. code-block:: bash

   git clone https://github.com/Kecoya/FracDimPy.git
   cd FracDimPy
   pip install -e .

For development:

.. code-block:: bash

   pip install -e .[dev]

Quick Start
===========

Basic Usage
-----------

.. code-block:: python

   import numpy as np
   from fracdimpy.monofractal import box_counting, hurst_dimension
   from fracdimpy.generator import fractional_brownian_motion

   # Generate fractal data
   signal = fractional_brownian_motion(1000, hurst=0.7)

   # Calculate box-counting dimension
   result = box_counting(signal)
   print(f"Box-counting dimension: {result['dimension']:.3f}")

   # Calculate Hurst exponent
   hurst_result = hurst_dimension(signal)
   print(f"Hurst exponent: {hurst_result['hurst']:.3f}")

Multifractal Analysis
---------------------

.. code-block:: python

   from fracdimpy.multifractal import mf_dfa

   # Multifractal analysis
   mf_result = mf_dfa(signal, q_values=np.linspace(-5, 5, 21))
   print(f"Spectrum width: {mf_result['spectrum_width']:.3f}")

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`