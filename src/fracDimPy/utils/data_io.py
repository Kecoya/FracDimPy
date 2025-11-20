#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Data I/O Utilities
==================

Functions for loading and saving data
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from pathlib import Path


def load_data(file_path: str, data_type: str = "auto") -> np.ndarray:
    """
    Load data from various file formats.

    Parameters
    ----------
    file_path : str
        Path to data file
    data_type : str, optional
        File format: 'auto', 'txt', 'csv', 'xlsx', 'npy'
        Default is 'auto' (auto-detect from extension)

    Returns
    -------
    data : np.ndarray
        Loaded data as numpy array

    Examples
    --------
    >>> from fracDimPy.utils import load_data
    >>> data = load_data('data.txt')
    >>> data = load_data('data.csv', data_type='csv')
    """
    path_obj = Path(file_path)

    if data_type == "auto":
        suffix = path_obj.suffix.lower()
        if suffix == ".txt":
            data_type = "txt"
        elif suffix == ".csv":
            data_type = "csv"
        elif suffix in [".xlsx", ".xls"]:
            data_type = "xlsx"
        elif suffix == ".npy":
            data_type = "npy"
        else:
            raise ValueError(f"Unsupported file extension: {suffix}")

    if data_type == "txt":
        data = np.loadtxt(file_path)
    elif data_type == "csv":
        data = pd.read_csv(file_path).values
    elif data_type == "xlsx":
        data = pd.read_excel(file_path).values
    elif data_type == "npy":
        data = np.load(file_path)
    else:
        raise ValueError(f"Unknown data type: {data_type}")

    return data


def save_results(results: Dict[str, Any], output_path: str, format: str = "xlsx") -> None:
    """
    Save analysis results to file.

    Parameters
    ----------
    results : dict
        Results dictionary to save
    output_path : str
        Output file path
    format : str, optional
        Output format: 'xlsx', 'csv', 'json'
        Default is 'xlsx'

    Examples
    --------
    >>> from fracDimPy.utils import save_results
    >>> results = {'dimension': [1.5], 'R2': [0.99]}
    >>> save_results(results, 'results.xlsx')
    >>> save_results(results, 'results.json', format='json')
    """
    output_obj = Path(output_path)

    if format == "xlsx":
        df = pd.DataFrame(results)
        df.to_excel(output_obj, index=False)
    elif format == "csv":
        df = pd.DataFrame(results)
        df.to_csv(output_obj, index=False)
    elif format == "json":
        import json

        # Convert numpy types to Python types for JSON serialization
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                serializable_results[key] = value.item()
            else:
                serializable_results[key] = value

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Results saved to: {output_path}")
