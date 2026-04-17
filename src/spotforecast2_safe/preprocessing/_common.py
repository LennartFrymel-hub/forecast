# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

"""
Common preprocessing functions and utilities.
"""

import functools
from typing import Any, Callable

import numpy as np
from numba import njit


def _check_X_numpy_ndarray_1d(ensure_1d: bool = True):
    """
    Decorator to check if argument `X` is a 1D numpy ndarray.

    Args:
        ensure_1d : bool, default True
            If True, ensure X is a 1D array.

    Returns:
        wrapper : Callable
            Decorated function.
    """

    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # args[0] is self, args[1] is X (if passed positional)
            # kwargs might contain X
            X = kwargs.get("X")
            if X is None and len(args) > 0:
                X = args[0]

            if X is not None:
                if not isinstance(X, np.ndarray):
                    raise TypeError(f"`X` must be a numpy ndarray. Got {type(X)}.")
                if ensure_1d and X.ndim != 1:
                    raise ValueError(f"`X` must be a 1D numpy ndarray. Got {X.ndim}D.")

            return func(self, *args, **kwargs)

        return wrapper

    return decorator


@njit(cache=True)
def _np_mean_jit(x: np.ndarray) -> float:
    """
    Numba optimized mean function.
    """
    return np.nanmean(x)


@njit(cache=True)
def _np_std_jit(x: np.ndarray) -> float:
    """
    Numba optimized std function (ddof=1).
    """
    return np.nanstd(x)


@njit(cache=True)
def _np_min_jit(x: np.ndarray) -> float:
    """
    Numba optimized min function.
    """
    return np.nanmin(x)


@njit(cache=True)
def _np_max_jit(x: np.ndarray) -> float:
    """
    Numba optimized max function.
    """
    return np.nanmax(x)


@njit(cache=True)
def _np_sum_jit(x: np.ndarray) -> float:
    """
    Numba optimized sum function.
    """
    return np.nansum(x)


@njit(cache=True)
def _np_median_jit(x: np.ndarray) -> float:
    """
    Numba optimized median function.
    """
    return np.nanmedian(x)


@njit(cache=True)
def _np_min_max_ratio_jit(x: np.ndarray) -> float:
    """
    NumPy min-max ratio function implemented with Numba JIT.
    """
    return np.nanmin(x) / np.nanmax(x)


@njit(cache=True)
def _np_cv_jit(x: np.ndarray) -> float:
    """
    Coefficient of variation function implemented with Numba JIT.
    If the array has only one element, the function returns 0.
    """
    if len(x) == 1:
        return 0.0

    a_a, b_b = 0.0, 0.0
    for i in x:
        if not np.isnan(i):
            a_a = a_a + i
            b_b = b_b + i * i

    n = np.sum(~np.isnan(x))
    if n <= 1:
        return 0.0

    var = b_b / n - ((a_a / n) ** 2)
    var = var * (n / (n - 1))
    std = np.sqrt(var)

    return std / (a_a / n)


@njit(cache=True)
def _ewm_jit(x: np.ndarray, alpha: float = 0.3) -> float:
    """
    Calculate the exponentially weighted mean of an array.
    """
    if not (0 < alpha <= 1):
        # Numba njit doesn't support f-strings or complex error messages easily in all versions
        # so we keep it simple.
        return np.nan

    n = len(x)
    weights = 0.0
    sum_weights = 0.0
    for i in range(n):
        if not np.isnan(x[i]):
            weight = (1 - alpha) ** (n - 1 - i)
            weights += x[i] * weight
            sum_weights += weight

    if sum_weights == 0:
        return np.nan

    return weights / sum_weights


def check_valid_quantile(quantile: float | list[float] | tuple[float]) -> None:
    """
    Check if quantile is valid (0 <= quantile <= 1).
    """
    if isinstance(quantile, (float, int)):
        if not (0 <= quantile <= 1):
            raise ValueError(f"Quantile must be between 0 and 1. Got {quantile}.")
    elif isinstance(quantile, (list, tuple, np.ndarray)):
        for q in quantile:
            if not (0 <= q <= 1):
                raise ValueError(f"Quantiles must be between 0 and 1. Got {q}.")
    else:
        raise TypeError(
            f"Quantile must be a float, list, tuple or numpy array. Got {type(quantile)}."
        )


def check_is_fitted(estimator: Any, attributes: list[str] | None = None) -> None:
    """
    Check if estimator is fitted by verifying if attributes exist.
    """
    if attributes is None:
        attributes = []

    for attr in attributes:
        if not hasattr(estimator, attr):
            raise ValueError(
                f"This {type(estimator).__name__} instance is not fitted yet. "
                f"Call 'fit' with appropriate arguments before using this estimator."
            )
