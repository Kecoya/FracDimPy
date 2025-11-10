# FracDimPy

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/songLe/FracDimPy)

**A Comprehensive Python Package for Fractal Dimension Calculation and Multifractal Analysis**

[English](README_EN.md) | 简体中文

</div>

---

## 📖 Introduction

FracDimPy is a powerful and easy-to-use Python package designed for fractal dimension calculation and multifractal analysis. Whether you are a researcher studying fractal geometry or an engineer analyzing complex data, FracDimPy provides professional and accurate analysis tools.

### ✨ Key Features

- **🔢 Multiple Monofractal Methods**

  - Hurst Exponent Method (R/S Analysis)
  - Box-counting Method
  - Information Dimension Method
  - Correlation Dimension Method
  - Structure Function Method
  - Variogram Method
  - Sandbox Method
  - Detrended Fluctuation Analysis (DFA)
- **📊 Multifractal Analysis**

  - One-dimensional curve multifractal analysis
  - Two-dimensional image multifractal analysis
  - Multifractal Detrended Fluctuation Analysis (MF-DFA)
  - Custom scale sequences
- **🎨 Fractal Generator**

  - Classical fractals: Cantor set, Sierpinski triangle/carpet, Koch curve, Menger sponge, etc.
  - Random fractals: Brownian motion, Lévy flight, self-avoiding walk, Diffusion-Limited Aggregation (DLA)
  - Fractal curves: FBM curve, Weierstrass-Mandelbrot function, Takagi curve
  - Fractal surfaces: FBM surface, Weierstrass-Mandelbrot surface, Takagi surface
- **📈 Rich Visualization**

  - Automatic generation of professional charts
  - Log-log plot fitting
  - Multifractal spectrum display
  - Customizable plotting options
- **💾 Flexible Data Processing**

  - Support for multiple data formats (CSV, Excel, TXT, NPY, images, etc.)
  - Automatic data preprocessing
  - Result export functionality

---

## 🚀 Quick Start

### Installation

#### Install from PyPI (Recommended)

```bash
# Basic installation
pip install FracDimPy

# Full installation (with all optional dependencies)
pip install FracDimPy[full]
```

#### Install from Source

```bash
git clone https://github.com/songLe/FracDimPy.git
cd FracDimPy
pip install -e .
```

For detailed installation instructions, please refer to [INSTALLATION.md](INSTALLATION.md)

### Basic Usage

#### Example 1: Calculate Hurst Exponent of Time Series

```python
import numpy as np
from fracDimPy import hurst_dimension

# Generate test data
data = np.random.randn(1000).cumsum()

# Calculate Hurst exponent
D, result = hurst_dimension(data, method='RS', show_plot=True)

print(f"Fractal dimension: {D:.4f}")
print(f"Hurst exponent: {result['H']:.4f}")
```

#### Example 2: Box-counting Analysis of Images

```python
from fracDimPy import box_counting
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Read image
img = mpimg.imread('fractal_image.png')

# Calculate fractal dimension
D, result = box_counting(img, show_plot=True)

print(f"Box-counting dimension: {D:.4f}")
print(f"Goodness of fit R²: {result['r_squared']:.4f}")
```

#### Example 3: Multifractal Analysis

```python
from fracDimPy import multifractal_curve
import numpy as np

# Read data
data = np.loadtxt('curve_data.txt')

# Multifractal analysis
result = multifractal_curve(
    data,
    q_range=(-10, 10),
    show_plot=True,
    save_path='mf_result.png'
)

print(f"Multifractal spectrum width Δα: {result['delta_alpha']:.4f}")
print(f"Singularity strength Δf: {result['delta_f']:.4f}")
```

#### Example 4: Generate Fractal Patterns

```python
from fracDimPy import generate_sierpinski, generate_koch_snowflake
import matplotlib.pyplot as plt

# Generate Sierpinski triangle
sierpinski = generate_sierpinski(level=6)

# Generate Koch snowflake
snowflake = generate_koch_snowflake(level=5)

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.scatter(sierpinski[:, 0], sierpinski[:, 1], s=1, c='blue')
ax1.set_title('Sierpinski Triangle')
ax1.axis('equal')

ax2.plot(snowflake[:, 0], snowflake[:, 1], 'r-', linewidth=0.5)
ax2.set_title('Koch Snowflake')
ax2.axis('equal')

plt.show()
```

---

## 📚 Documentation

- [Installation Guide](INSTALLATION.md) - Detailed installation instructions
- [Build Guide](BUILD.md) - Instructions for building from source
- [Contributing Guide](CONTRIBUTING.md) - How to contribute to the project
- [Changelog](CHANGELOG.md) - Version update records
- [Example Code](examples/) - Rich usage examples

---

## 📦 Module Description

### 1. Monofractal Module (`monofractal`)

Provides various monofractal dimension calculation methods:

| Method                | Function Name               | Data Type       | Description                                             |
| --------------------- | --------------------------- | --------------- | ------------------------------------------------------- |
| Hurst Exponent        | `hurst_dimension()`       | 1D time series  | R/S analysis, modified R/S, DFA                         |
| Box-counting          | `box_counting()`          | 1D/2D/3D        | Most commonly used fractal dimension calculation method |
| Information Dimension | `information_dimension()` | Point set data  | Dimension based on information entropy                  |
| Correlation Dimension | `correlation_dimension()` | Point set data  | Based on correlation integral                           |
| Structure Function    | `structural_function()`   | 1D curve        | Suitable for self-affine curves                         |
| Variogram             | `variogram_method()`      | 1D/2D           | Geostatistical method                                   |
| Sandbox               | `sandbox_method()`        | Point set/image | Local scale analysis                                    |
| DFA                   | `dfa()`                   | 1D time series  | Detrended Fluctuation Analysis                          |

### 2. Multifractal Module (`multifractal`)

Provides multifractal analysis tools:

| Function                 | Description                                 | Output                                                           |
| ------------------------ | ------------------------------------------- | ---------------------------------------------------------------- |
| `multifractal_curve()` | One-dimensional curve multifractal analysis | Partition function, generalized dimension, multifractal spectrum |
| `multifractal_image()` | Two-dimensional image multifractal analysis | Singularity index, multifractal characteristics                  |
| `mf_dfa()`             | Multifractal DFA                            | Fluctuation function, Hurst exponent spectrum                    |

### 3. Fractal Generator (`generator`)

Generates various theoretical and random fractals:

**Curve Class** (1D):

- `generate_fbm_curve()` - Fractional Brownian Motion curve
- `generate_wm_curve()` - Weierstrass-Mandelbrot function
- `generate_takagi_curve()` - Takagi curve
- `generate_koch_curve()` - Koch curve
- `generate_brownian_motion()` - Brownian motion
- `generate_levy_flight()` - Lévy flight

**Surface Class** (2D):

- `generate_fbm_surface()` - Fractional Brownian Motion surface
- `generate_wm_surface()` - WM surface
- `generate_takagi_surface()` - Takagi surface

**Pattern Class** (Geometric fractals):

- `generate_cantor_set()` - Cantor set
- `generate_sierpinski()` - Sierpinski triangle
- `generate_sierpinski_carpet()` - Sierpinski carpet
- `generate_vicsek_fractal()` - Vicsek fractal
- `generate_koch_snowflake()` - Koch snowflake
- `generate_dla()` - Diffusion-Limited Aggregation
- `generate_menger_sponge()` - Menger sponge (3D)

### 4. Utility Module (`utils`)

- Data I/O (`data_io`)
- Visualization tools (`plotting`)

---

## 🔬 Application Areas

FracDimPy can be applied to multiple scientific and engineering fields:

- **Earth Sciences**: Terrain analysis, seismic data, fracture networks
- **Materials Science**: Porous media, surface roughness, nanostructures
- **Biomedical**: DNA sequences, protein folding, medical imaging
- **Financial Analysis**: Stock prices, market volatility, risk assessment
- **Image Processing**: Texture analysis, pattern recognition, image segmentation
- **Environmental Science**: River networks, cloud pattern analysis, pollution diffusion
- **Physics**: Turbulence, phase transitions, chaotic systems

---

## 📊 Examples and Data

The [examples](examples/) directory contains rich example code and test data:

```
examples/
├── monofractal/          # Monofractal method examples
│   ├── test_hurst.py
│   ├── test_box_counting_*.py
│   └── ...
├── multifractal/         # Multifractal examples
│   ├── test_mf_curve_*.py
│   ├── test_mf_image.py
│   └── ...
└── generator/            # Fractal generation examples
    ├── test_koch.py
    ├── test_dla.py
    └── ...
```

Run examples:

```bash
cd examples/monofractal
python test_hurst.py
```

For more details, see [examples/README.md](examples/README.md)

---

## 🛠️ Dependencies

### Core Dependencies

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.3.0
- Pandas >= 1.3.0

### Optional Dependencies

- `opencv-python` - Advanced image processing
- `Pillow` - Image I/O

For the complete dependency list, please refer to [pyproject.toml](pyproject.toml)

---

## 🤝 Contributing

Contributions of all kinds are welcome! Whether it's reporting bugs, suggesting new features, or submitting code improvements.

Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

### Contributors

- **Zhile Han** - *Main Developer* - [Zhihu Profile](https://www.zhihu.com/people/xiao-xue-sheng-ye-xiang-xie-shu/posts)

---

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details

---

## 📮 Contact

- **Author**: Zhile Han
- **Email**: 2667032759@qq.com
- **Address**: State Key Laboratory of Oil and Gas Reservoir Geology and Exploitation, Southwest Petroleum University, Chengdu 610500, China
- **Zhihu**: [小学生也想写书](https://www.zhihu.com/people/xiao-xue-sheng-ye-xiang-xie-shu/posts)
- **GitHub**: [https://github.com/Kecoya/FracDimPy](https://github.com/Kecoya/FracDimPy)

---

## 📝 Citation

If you use FracDimPy in your research, please cite:

```bibtex
@software{fracdimpy2024,
  author = {Zhile Han},
  title = {FracDimPy: A Comprehensive Python Package for Fractal Dimension Calculation and Multifractal Analysis},
  year = {2024},
  url = {https://github.com/Kecoya/FracDimPy},
  version = {0.1.0}
}
```

---

## 🙏 Acknowledgments

Thanks to all researchers and open-source community members who have contributed to fractal theory and algorithm implementation.

---

## ⭐ Star History

If this project is helpful to you, please give it a ⭐️!

---

## 🔗 Related Projects

- [NumPy](https://numpy.org/) - Numerical computing foundation
- [SciPy](https://scipy.org/) - Scientific computing tools
- [Matplotlib](https://matplotlib.org/) - Data visualization

---

<div align="center">

**[⬆ Back to Top](#fracdimpy)**

Made with ❤️ by Zhile Han

</div>
