# FracDimPy

<div align="center">

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](https://github.com/songLe/FracDimPy)

**一个全面的Python分形维数计算与多重分形分析工具包**

[English](https://github.com/Kecoya/FracDimPy/blob/main/README_EN.md) | 简体中文

</div>

---

## 📖 简介

FracDimPy 是一个功能强大、易于使用的Python软件包，专门用于分形维数计算和多重分形分析。无论您是研究分形几何的科研人员，还是需要分析复杂数据的工程师，FracDimPy都能为您提供专业、准确的分析工具。

### ✨ 主要特性

- **🔢 多种单分形方法**

  - Hurst指数法 (R/S分析)
  - 盒计数法 (Box-counting)
  - 信息维数法 (Information Dimension)
  - 关联维数法 (Correlation Dimension)
  - 结构函数法 (Structure Function)
  - 变差函数法 (Variogram)
  - 沙盒法 (Sandbox)
  - 去趋势波动分析 (DFA)
- **📊 多重分形分析**

  - 一维曲线多重分形分析
  - 二维图像多重分形分析
  - 多重分形去趋势波动分析 (MF-DFA)
  - 自定义尺度序列
- **🎨 分形生成器**

  - 经典分形：Cantor集、Sierpinski三角形/地毯、Koch曲线、Menger海绵等
  - 随机分形：布朗运动、Lévy飞行、自回避行走、扩散限制聚集(DLA)
  - 分形曲线：FBM曲线、Weierstrass-Mandelbrot函数、Takagi曲线
  - 分形曲面：FBM曲面、Weierstrass-Mandelbrot曲面、Takagi曲面
- **📈 丰富的可视化**

  - 自动生成专业图表
  - 双对数图拟合
  - 多重分形谱展示
  - 可定制的绘图选项
- **💾 灵活的数据处理**

  - 支持多种数据格式 (CSV, Excel, TXT, NPY, 图像等)
  - 自动数据预处理
  - 结果导出功能

---

## 🚀 快速开始

### 安装

#### 从PyPI安装（推荐）

```bash
# 基础安装
pip install FracDimPy

# 完整安装（包含所有可选依赖）
pip install FracDimPy[full]
```

## 📦 模块说明

### 1. 单分形模块 (`monofractal`)

提供多种单分形维数计算方法：

| 方法      | 函数名                      | 适用数据类型 | 说明                     |
| --------- | --------------------------- | ------------ | ------------------------ |
| Hurst指数 | `hurst_dimension()`       | 1D时间序列   | R/S分析、修正R/S、DFA    |
| 盒计数法  | `box_counting()`          | 1D/2D/3D     | 最常用的分形维数计算方法 |
| 信息维数  | `information_dimension()` | 点集数据     | 基于信息熵的维数         |
| 关联维数  | `correlation_dimension()` | 点集数据     | 基于关联积分             |
| 结构函数  | `structural_function()`   | 1D曲线       | 适用于自仿射曲线         |
| 变差函数  | `variogram_method()`      | 1D/2D        | 地统计学方法             |
| 沙盒法    | `sandbox_method()`        | 点集/图像    | 局部尺度分析             |
| DFA       | `dfa()`                   | 1D时间序列   | 去趋势波动分析           |

### 2. 多重分形模块 (`multifractal`)

提供多重分形分析工具：

| 函数                     | 说明                 | 输出                           |
| ------------------------ | -------------------- | ------------------------------ |
| `multifractal_curve()` | 一维曲线多重分形分析 | 配分函数、广义维数、多重分形谱 |
| `multifractal_image()` | 二维图像多重分形分析 | 奇异性指数、多重分形特征       |
| `mf_dfa()`             | 多重分形DFA          | 波动函数、Hurst指数谱          |

### 3. 分形生成器 (`generator`)

生成各种理论和随机分形：

**曲线类** (1D):

- `generate_fbm_curve()` - 分数布朗运动曲线
- `generate_wm_curve()` - Weierstrass-Mandelbrot函数
- `generate_takagi_curve()` - Takagi曲线
- `generate_koch_curve()` - Koch曲线
- `generate_brownian_motion()` - 布朗运动
- `generate_levy_flight()` - Lévy飞行

**曲面类** (2D):

- `generate_fbm_surface()` - 分数布朗运动曲面
- `generate_wm_surface()` - WM曲面
- `generate_takagi_surface()` - Takagi曲面

**图案类** (几何分形):

- `generate_cantor_set()` - Cantor集
- `generate_sierpinski()` - Sierpinski三角形
- `generate_sierpinski_carpet()` - Sierpinski地毯
- `generate_vicsek_fractal()` - Vicsek分形
- `generate_koch_snowflake()` - Koch雪花
- `generate_dla()` - 扩散限制聚集
- `generate_menger_sponge()` - Menger海绵（3D）

### 4. 工具模块 (`utils`)

- 数据读写 (`data_io`)
- 可视化工具 (`plotting`)

---

## 🔬 应用领域

FracDimPy可应用于多个科学和工程领域：

- **地球科学**：地形分析、地震数据、裂缝网络
- **材料科学**：多孔介质、表面粗糙度、纳米结构
- **生物医学**：DNA序列、蛋白质折叠、医学影像
- **金融分析**：股票价格、市场波动、风险评估
- **图像处理**：纹理分析、模式识别、图像分割
- **环境科学**：河流网络、云图分析、污染扩散
- **物理学**：湍流、相变、混沌系统

---

## 📊 示例与数据

[examples](examples/) 目录包含丰富的示例代码和测试数据：

```
examples/
├── monofractal/          # 单分形方法示例
│   ├── test_hurst.py
│   ├── test_box_counting_*.py
│   └── ...
├── multifractal/         # 多重分形示例
│   ├── test_mf_curve_*.py
│   ├── test_mf_image.py
│   └── ...
└── generator/            # 分形生成示例
    ├── test_koch.py
    ├── test_dla.py
    └── ...
```

运行示例：

```bash
cd examples/monofractal
python test_hurst.py
```

详见 [examples/README.md](examples/README.md)

---

## 🛠️ 依赖项

### 核心依赖

- Python >= 3.8
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Matplotlib >= 3.3.0
- Pandas >= 1.3.0

### 可选依赖

- `opencv-python` - 高级图像处理
- `Pillow` - 图像读写

完整依赖列表请参阅 [pyproject.toml](pyproject.toml)

---

## 🤝 贡献

欢迎各种形式的贡献！无论是报告bug、提出新功能建议，还是提交代码改进。

请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解详细的贡献指南。

### 贡献者

- **Zhile Han** - *主要开发者* - [知乎主页](https://www.zhihu.com/people/xiao-xue-sheng-ye-xiang-xie-shu/posts)

---

## 📄 许可证

本项目采用 GNU General Public License v3.0 许可证 - 详见 [LICENSE](LICENSE) 文件

---

## 📮 联系方式

- **作者**: Zhile Han
- **邮箱**: 2667032759@qq.com
- **地址**: 油气藏地质及开发工程全国重点实验室，西南石油大学，成都610500，中国
- **知乎**: [小学生也想写书](https://www.zhihu.com/people/xiao-xue-sheng-ye-xiang-xie-shu/posts)
- **GitHub**: [https://github.com/Kecoya/FracDimPy](https://github.com/Kecoya/FracDimPy)

---

## 📝 引用

如果您在研究中使用了FracDimPy，请引用：

```bibtex
@software{fracdimpy2024,
  author = {Zhile Han},
  title = {FracDimPy: A Comprehensive Python Package for Fractal Dimension Calculation and Multifractal Analysis},
  year = {2024},
  url = {https://github.com/Kecoya/FracDimPy},
  version = {0.1.2}
}
```

---

## 🙏 致谢

感谢所有为分形理论和算法实现做出贡献的研究者和开源社区成员。

---

## ⭐ Star History

如果这个项目对您有帮助，请给它一个⭐️！

---

## 🔗 相关项目

- [NumPy](https://numpy.org/) - 数值计算基础
- [SciPy](https://scipy.org/) - 科学计算工具
- [Matplotlib](https://matplotlib.org/) - 数据可视化

---

<div align="center">

**[⬆ 返回顶部](#fracdimpy)**

Made with ❤️ by Zhile Han

</div>
