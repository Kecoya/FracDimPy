"""
Sphinx configuration file for FracDimPy documentation
"""

import os
import sys

# Add src directory to path for autodoc
sys.path.insert(0, os.path.abspath('../src'))

# Project information
project = 'FracDimPy'
copyright = '2025, Zhile Han, Cong Lu, Shouxin Wang'
author = 'Zhile Han, Cong Lu, Shouxin Wang'
release = '0.1.3'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
}

# HTML output settings
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = []

# The master toctree document.
master_doc = 'index'

# Output file base name for HTML help builder.
htmlhelp_basename = 'FracDimPydoc'