#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Linear fitting utilities for fractal analysis.

Provides shared functions for log-log linear regression and R-squared
computation, replacing duplicated polyfit+R² patterns across modules.
"""

import numpy as np


def log_log_fit(x, y):
    """Perform linear regression on log-log scale.

    Parameters
    ----------
    x : array-like
        Independent variable values (will be log-transformed).
    y : array-like
        Dependent variable values (will be log-transformed).

    Returns
    -------
    slope : float
        Slope of the log-log linear fit.
    intercept : float
        Intercept of the log-log linear fit.
    r_squared : float
        R² (coefficient of determination) of the fit.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    r_squared = _r_squared_corrcoef(x, y, coeffs)
    return slope, intercept, r_squared


def linear_fit(x, y):
    """Perform linear regression and return results.

    Parameters
    ----------
    x : array-like
        Independent variable values.
    y : array-like
        Dependent variable values.

    Returns
    -------
    slope : float
        Slope of the linear fit.
    intercept : float
        Intercept of the linear fit.
    r_squared : float
        R² of the fit (residual method).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    coeffs = np.polyfit(x, y, 1)
    slope = coeffs[0]
    intercept = coeffs[1]
    r_squared = _r_squared_residual(y, np.polyval(coeffs, x))
    return slope, intercept, r_squared


def _r_squared_corrcoef(x, y, coeffs):
    """R² via Pearson correlation coefficient squared."""
    f = np.poly1d(coeffs)
    r = np.corrcoef(y, f(x))[0, 1]
    return r ** 2


def _r_squared_residual(y_actual, y_predicted):
    """R² via residual sum of squares."""
    ss_res = np.sum((y_actual - y_predicted) ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
