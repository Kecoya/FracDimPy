---
title: 'FracDimPy: A Comprehensive Python Package for Fractal Dimension Calculation and Multifractal Analysis'
tags:
  - Python
  - fractal geometry
  - multifractal analysis
  - computational physics
  - data analysis
  - scientific computing
authors:
  - name: Zhile Han
    orcid: 0009-0002-6306-1452
    affiliation: "1"
  - name: Cong Lu
    affiliation: "1"
  - name: Shouxin Wang
    orcid: 0009-0003-1583-4999
    affiliation: "1"
affiliations:
  - index: 1
    name: State Key Laboratory of Oil and Gas Reservoir Geology and Exploitation, Southwest Petroleum University, Chengdu 610500, China
    ror: 05a1rry66
date: 19 November 2025
bibliography: paper.bib
linenumbers: false
numbersections: false
header-includes: |
  \usepackage{lineno}
  \DisableLinenumbers
---

# Summary

FracDimPy is a comprehensive Python package for fractal dimension calculation and multifractal analysis that provides researchers with a unified toolkit for analyzing complex, self-similar patterns in data. Fractal geometry has become essential for understanding irregular structures across numerous scientific disciplines, from geological formations and material surfaces to biological systems and financial markets [@mandelbrotFractalGeometryNature1982]. FracDimPy offers a complete suite of computational tools including eight different monofractal analysis methods, comprehensive multifractal analysis capabilities, and built-in fractal generation tools. The package is designed for both research applications and educational purposes, featuring an intuitive API, extensive documentation, and rich visualization capabilities that make advanced fractal analysis accessible to researchers across disciplines.

# Statement of Need

The analysis of fractal patterns has become increasingly critical across scientific research, yet researchers face significant challenges when accessing appropriate computational tools. Existing fractal analysis software often suffers from several limitations: fragmented implementations across multiple packages, restrictive licensing, steep learning curves, or limited methodological coverage. Commercial alternatives can be prohibitively expensive, while many academic tools lack comprehensive documentation or maintenance.

FracDimPy addresses these challenges by providing a unified, open-source solution that combines multiple fractal analysis methodologies within a single, well-documented package. While existing tools like Nolds, Fathon, and FracLab provide excellent implementations of specific methods [@nolds; @fathon; @fraclab], FracDimPy offers a comprehensive suite that integrates both monofractal and multifractal analysis methods with extensive visualization capabilities. The software includes eight different monofractal analysis methods (Hurst exponent, box-counting, information dimension, correlation dimension, structural function, variogram, sandbox, and detrended fluctuation analysis), comprehensive multifractal analysis with partition function and spectrum calculation, and fractal generation tools for testing and educational purposes.

The package fills a critical gap in the Python scientific computing ecosystem by offering fractal analysis capabilities that integrate seamlessly with popular libraries like NumPy, SciPy, and Matplotlib [@harrisArrayProgrammingNumPy2020; @virtanenSciPy1.0Fundamental2020; @hunterMatplotlib2D2007]. This integration enables researchers to incorporate fractal analysis into existing data analysis workflows without learning entirely new software environments. FracDimPy has already been applied across diverse fields including geoscience (terrain analysis, seismic data), materials science (surface roughness, porous media), biomedicine (DNA sequences, medical imaging), and financial analysis (market volatility, risk assessment).

The software's permissive GPL-3.0 license ensures accessibility for both academic and commercial applications, while its active development and comprehensive documentation support long-term research projects and educational initiatives. By providing a complete fractal analysis toolkit, FracDimPy enables researchers to extract meaningful insights from complex data patterns that traditional linear analysis methods might overlook.

# Methods

## Monofractal Analysis

FracDimPy implements diverse monofractal methods:

- **Box-Counting Dimension**: Robust implementation with configurable boundary handling and partitioning strategies [@foroutan-pourAdvancesImplementationBoxcounting1999; @liebovitchFastAlgorithmDetermine1989]
- **Hurst Exponent**: R/S analysis with logarithmic window spacing for improved accuracy [@hurstLongTermStorageCapacity1951]
- **Detrended Fluctuation Analysis**: Configurable detrending order with two-pass processing [@pengMosaicOrganizationDNA1994]
- **Correlation Dimension**: Grassberger-Procaccia algorithm with automatic scaling range selection [@grassbergerMeasuringStrangenessAttractors1983]

## Multifractal Analysis

Advanced multifractal techniques include:

- **MF-DFA**: Multifractal Detrended Fluctuation Analysis for non-stationary time series [@kantelhardtMultifractalDetrendedFluctuation2002]
- **Multifractal Spectrum**: Box-counting based formalism for 1D curves and 2D images
- **Custom Epsilon Sequences**: Flexible scaling range optimization

## Fractal Generation

The generator module creates synthetic fractals with known dimensions:

- Deterministic fractals: Sierpinski triangle, Koch curve, Cantor set
- Stochastic fractals: Fractional Brownian motion, Weierstrass function
- Validation datasets for algorithm testing

# Validation and Applications

## Algorithm Validation

We validated FracDimPy's implementations using canonical fractals with known theoretical dimensions:

- **Sierpinski Triangle**: Box-counting dimension $D = \log(3)/\log(2) \approx 1.585$
- **Koch Curve**: Box-counting dimension $D = \log(4)/\log(3) \approx 1.262$
- **Cantor Set**: Box-counting dimension $D = \log(2)/\log(3) \approx 0.631$

Our implementations recover these theoretical values with relative errors less than 2%.

## Case Study 1: NMR Data Analysis

We applied FracDimPy to Nuclear Magnetic Resonance (NMR) data from petroleum engineering research. The multifractal spectrum analysis revealed heterogeneous pore distributions that correlate with rock permeability measurements.

## Case Study 2: Surface Roughness Analysis

Using Takagi surface data, we demonstrated the package's ability to characterize surface roughness through multiple fractal dimensions, providing insights into material properties.

# Performance

FracDimPy is optimized for computational efficiency:
- Vectorized operations using NumPy
- Optional parallel processing for large datasets
- Memory-efficient algorithms for high-dimensional data

Benchmarks show linear scaling with data size up to $10^6$ points.

# Acknowledgements

We acknowledge the support of Southwest Petroleum University and the State Key Laboratory of Oil and Gas Reservoir Geology and Exploitation. We are grateful to the open source community for providing the essential scientific computing ecosystem that makes this work possible, including NumPy, SciPy, Matplotlib, and the broader Python scientific community. Special thanks to the contributors to existing fractal analysis software packages such as Nolds, Fathon, and FracLab, which have informed and inspired the development of FracDimPy. This work was inspired by the need for accessible fractal analysis tools in petroleum engineering research and aims to contribute back to the scientific computing community.

# References