#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Setup script for FracDimPy
"""

from setuptools import setup, find_packages
from pathlib import Path

# README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="FracDimPy",
    version="0.1.3",
    author="Zhile Han",
    author_email="2667032759@qq.com",
    description="A comprehensive Python package for fractal dimension calculation and multifractal analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="GPL-3.0-or-later",
    url="https://github.com/Kecoya/FracDimPy",
    project_urls={
        "Bug Tracker": "https://github.com/Kecoya/FracDimPy/issues",
        "Documentation": "https://github.com/Kecoya/FracDimPy/blob/main/README.md",
        "Source Code": "https://github.com/Kecoya/FracDimPy",
    },
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
        "pandas>=1.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "black>=22.0",
            "flake8>=5.0",
            "mypy>=0.990",
        ],
        "full": [
            "opencv-python>=4.5.0",
            "Pillow>=9.0.0",
        ],
    },
    keywords="fractal fractal-dimension multifractal box-counting hurst r-s-analysis",
)

