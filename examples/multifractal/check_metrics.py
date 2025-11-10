#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查返回的指标键名"""

import pandas as pd
import os
from fracDimPy import multifractal_curve

# 获取根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
data_file = os.path.join(root_dir, "1.xlsx")

# 读取数据
df = pd.read_excel(data_file)
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

# 运行分析
metrics, figure_data = multifractal_curve(
    (x, y),
    use_multiprocessing=False,
    data_type='dual'
)

print("返回的metrics键名：")
for key in metrics.keys():
    print(f"  '{key}': {metrics[key]}")

print("\n返回的figure_data键名：")
for key in figure_data.keys():
    if isinstance(figure_data[key], list) and len(figure_data[key]) > 5:
        print(f"  '{key}': [list with {len(figure_data[key])} elements]")
    else:
        print(f"  '{key}': {figure_data[key]}")

