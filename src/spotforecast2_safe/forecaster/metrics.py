# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

"""Metrics for evaluating forecasting models.

This module provides various metric functions for evaluating forecasting performance,
including custom metrics like MASE, RMSSE, and probabilistic metrics like CRPS.

Examples:
    Using standard metrics::

        import numpy as np
        from spotforecast2.forecaster.metrics import _get_metric

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        # Get a metric function
        mse = _get_metric('mean_squared_error')
        error = mse(y_true, y_pred)

    Using scaled metrics::

        from spotforecast2.forecaster.metrics import mean_absolute_scaled_error

        y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        y_true = np.array([9, 10, 11])
        y_pred = np.array([8.8, 10.2, 10.9])

        mase = mean_absolute_scaled_error(y_true, y_pred, y_train)
"""

from __future__ import annotations

import inspect
from functools import wraps
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_pinball_loss,
    mean_squared_error,
    mean_squared_log_error,
    median_absolute_error,
    precision_score,
    recall_score,
)


def _get_metric(metric: str) -> Callable:
    """Get the corresponding scikit-learn function to calculate the metric.

    Args:
        metric: Metric used to quantify the goodness of fit of the model.

    Returns:
        scikit-learn function to calculate the desired metric.

    Examples:
        >>> from spotforecast2.forecaster.metrics import _get_metric
        >>> mse_func = _get_metric('mean_squared_error')
        >>> y_true = np.array([1, 2, 3])
        >>> y_pred = np.array([1.1, 1.9, 3.2])
        >>> error = mse_func(y_true, y_pred)
        >>> error > 0
        True
    """

    allowed_metrics = [
        # Regression metrics
        "mean_squared_error",
        "mean_absolute_error",
        "mean_absolute_percentage_error",
        "mean_squared_log_error",
        "mean_absolute_scaled_error",
        "root_mean_squared_scaled_error",
        "median_absolute_error",
        "symmetric_mean_absolute_percentage_error",
        # Classification metrics
        "accuracy_score",
        "balanced_accuracy_score",
        "f1_score",
        "precision_score",
        "recall_score",
    ]

    if metric not in allowed_metrics:
        raise ValueError(f"Allowed metrics are: {allowed_metrics}. Got {metric}.")

    metrics = {
        "mean_squared_error": mean_squared_error,
        "mean_absolute_error": mean_absolute_error,
        "mean_absolute_percentage_error": mean_absolute_percentage_error,
        "mean_squared_log_error": mean_squared_log_error,
        "mean_absolute_scaled_error": mean_absolute_scaled_error,
        "root_mean_squared_scaled_error": root_mean_squared_scaled_error,
        "median_absolute_error": median_absolute_error,
        "symmetric_mean_absolute_percentage_error": symmetric_mean_absolute_percentage_error,
        "accuracy_score": accuracy_score,
        "balanced_accuracy_score": balanced_accuracy_score,
        "f1_score": f1_score,
        "precision_score": precision_score,
        "recall_score": recall_score,
    }

    metric = add_y_train_argument(metrics[metric])

    return metric


def add_y_train_argument(func: Callable) -> Callable:
    """Add `y_train` argument to a function if it is not already present.

    Args:
        func: Function to which the argument is added.

    Returns:
        Function with `y_train` argument added.

    Examples:
        >>> def my_metric(y_true, y_pred):
        ...     return np.mean(np.abs(y_true - y_pred))
        >>> enhanced_metric = add_y_train_argument(my_metric)
        >>> # Now the function accepts y_train parameter
        >>> result = enhanced_metric(np.array([1,2,3]), np.array([1,2,3]), y_train=None)
    """

    sig = inspect.signature(func)

    if "y_train" in sig.parameters:
        return func

    new_params = list(sig.parameters.values()) + [
        inspect.Parameter("y_train", inspect.Parameter.KEYWORD_ONLY, default=None)
    ]
    new_sig = sig.replace(parameters=new_params)

    @wraps(func)
    def wrapper(*args, y_train=None, **kwargs):
        return func(*args, **kwargs)

    wrapper.__signature__ = new_sig

    return wrapper


def mean_absolute_scaled_error(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_train: list[float] | np.ndarray | pd.Series,
) -> float:
    """Mean Absolute Scaled Error (MASE).

    MASE is a scale-independent error metric that measures the accuracy of
    a forecast. It is the mean absolute error of the forecast divided by the
    mean absolute error of a naive forecast in the training set. The naive
    forecast is the one obtained by shifting the time series by one period.
    If y_train is a list of numpy arrays or pandas Series, it is considered
    that each element is the true value of the target variable in the training
    set for each time series. In this case, the naive forecast is calculated
    for each time series separately.

    Args:
        y_true: True values of the target variable.
        y_pred: Predicted values of the target variable.
        y_train: True values of the target variable in the training set. If `list`, it
            is consider that each element is the true value of the target variable
            in the training set for each time series.

    Returns:
        MASE value.

    Examples:
        >>> from spotforecast2.forecaster.metrics import mean_absolute_scaled_error
        >>> y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        >>> y_true = np.array([9, 10, 11])
        >>> y_pred = np.array([8.8, 10.2, 10.9])
        >>> mase = mean_absolute_scaled_error(y_true, y_pred, y_train)
        >>> mase < 1.0  # Good forecast
        True
    """

    # NOTE: When using this metric in validation, `y_train` doesn't include
    # the first window_size observations used to create the predictors and/or
    # rolling features.

    if not isinstance(y_true, (pd.Series, np.ndarray)):
        raise TypeError("`y_true` must be a pandas Series or numpy ndarray.")
    if not isinstance(y_pred, (pd.Series, np.ndarray)):
        raise TypeError("`y_pred` must be a pandas Series or numpy ndarray.")
    if not isinstance(y_train, (list, pd.Series, np.ndarray)):
        raise TypeError("`y_train` must be a list, pandas Series or numpy ndarray.")
    if isinstance(y_train, list):
        for x in y_train:
            if not isinstance(x, (pd.Series, np.ndarray)):
                raise TypeError(
                    "When `y_train` is a list, each element must be a pandas Series "
                    "or numpy ndarray."
                )
    if len(y_true) != len(y_pred):
        raise ValueError("`y_true` and `y_pred` must have the same length.")
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("`y_true` and `y_pred` must have at least one element.")

    if isinstance(y_train, list):
        # Flatten list of arrays for naive forecast if meaningful, but MASE usually assumes
        # naive forecast on single series. If list, we might be doing something else.
        # Original code does: np.concatenate([np.diff(x) for x in y_train])
        # This assumes independent series and we average error over all of them.
        naive_forecast = np.concatenate([np.diff(x) for x in y_train])
    else:
        naive_forecast = np.diff(y_train)

    mase = np.mean(np.abs(y_true - y_pred)) / np.nanmean(np.abs(naive_forecast))

    return mase


def root_mean_squared_scaled_error(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_train: list[float] | np.ndarray | pd.Series,
) -> float:
    """Root Mean Squared Scaled Error (RMSSE).

    RMSSE is a scale-independent error metric that measures the accuracy of
    a forecast. It is the root mean squared error of the forecast divided by
    the root mean squared error of a naive forecast in the training set. The
    naive forecast is the one obtained by shifting the time series by one period.
    If y_train is a list of numpy arrays or pandas Series, it is considered
    that each element is the true value of the target variable in the training
    set for each time series. In this case, the naive forecast is calculated
    for each time series separately.

    Args:
        y_true: True values of the target variable.
        y_pred: Predicted values of the target variable.
        y_train: True values of the target variable in the training set. If list, it
            is consider that each element is the true value of the target variable
            in the training set for each time series.

    Returns:
        RMSSE value.

    Examples:
        >>> from spotforecast2.forecaster.metrics import root_mean_squared_scaled_error
        >>> y_train = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        >>> y_true = np.array([9, 10, 11])
        >>> y_pred = np.array([8.8, 10.2, 10.9])
        >>> rmsse = root_mean_squared_scaled_error(y_true, y_pred, y_train)
        >>> rmsse < 1.0  # Good forecast
        True
    """

    # NOTE: When using this metric in validation, `y_train` doesn't include
    # the first window_size observations used to create the predictors and/or
    # rolling features.

    if not isinstance(y_true, (pd.Series, np.ndarray)):
        raise TypeError("`y_true` must be a pandas Series or numpy ndarray.")
    if not isinstance(y_pred, (pd.Series, np.ndarray)):
        raise TypeError("`y_pred` must be a pandas Series or numpy ndarray.")
    if not isinstance(y_train, (list, pd.Series, np.ndarray)):
        raise TypeError("`y_train` must be a list, pandas Series or numpy ndarray.")
    if isinstance(y_train, list):
        for x in y_train:
            if not isinstance(x, (pd.Series, np.ndarray)):
                raise TypeError(
                    "When `y_train` is a list, each element must be a pandas Series "
                    "or numpy ndarray."
                )
    if len(y_true) != len(y_pred):
        raise ValueError("`y_true` and `y_pred` must have the same length.")
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("`y_true` and `y_pred` must have at least one element.")

    if isinstance(y_train, list):
        naive_forecast = np.concatenate([np.diff(x) for x in y_train])
    else:
        naive_forecast = np.diff(y_train)

    rmsse = np.sqrt(np.mean((y_true - y_pred) ** 2)) / np.sqrt(
        np.nanmean(naive_forecast**2)
    )

    return rmsse


def crps_from_predictions(y_true: float, y_pred: np.ndarray) -> float:
    """Compute the Continuous Ranked Probability Score (CRPS) from predictions.

    The CRPS compares the empirical distribution of a set of forecasted values
    to a scalar observation. The smaller the CRPS, the better.

    Args:
        y_true: The true value of the random variable.
        y_pred: The predicted values of the random variable. These are the multiple
            forecasted values for a single observation.

    Returns:
        The CRPS score.

    Examples:
        >>> from spotforecast2.forecaster.metrics import crps_from_predictions
        >>> y_true = 5.0
        >>> y_pred = np.array([4.5, 5.1, 4.9, 5.3, 4.7])
        >>> crps = crps_from_predictions(y_true, y_pred)
        >>> crps >= 0
        True
    """
    if not isinstance(y_pred, np.ndarray) or y_pred.ndim != 1:
        raise TypeError("`y_pred` must be a 1D numpy array.")

    if not isinstance(y_true, (float, int)):
        raise TypeError("`y_true` must be a float or integer.")

    y_pred = np.sort(y_pred)
    # Define the grid for integration including the true value
    grid = np.concatenate(([y_true], y_pred))
    grid = np.sort(grid)
    cdf_values = np.searchsorted(y_pred, grid, side="right") / len(y_pred)
    indicator = grid >= y_true
    diffs = np.diff(grid)
    crps = np.sum(diffs * (cdf_values[:-1] - indicator[:-1]) ** 2)

    return crps


def crps_from_quantiles(
    y_true: float,
    pred_quantiles: np.ndarray,
    quantile_levels: np.ndarray,
) -> float:
    """Calculate the Continuous Ranked Probability Score (CRPS) from quantiles.

    The empirical cdf is approximated using linear interpolation
    between the predicted quantiles.

    Args:
        y_true: The true value of the random variable.
        pred_quantiles: The predicted quantile values.
        quantile_levels: The quantile levels corresponding to the predicted quantiles.

    Returns:
        The CRPS score.

    Examples:
        >>> from spotforecast2.forecaster.metrics import crps_from_quantiles
        >>> y_true = 5.0
        >>> pred_quantiles = np.array([4.0, 4.5, 5.0, 5.5, 6.0])
        >>> quantile_levels = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
        >>> crps = crps_from_quantiles(y_true, pred_quantiles, quantile_levels)
        >>> crps >= 0
        True
    """
    if not isinstance(y_true, (float, int)):
        raise TypeError("`y_true` must be a float or integer.")

    if not isinstance(pred_quantiles, np.ndarray) or pred_quantiles.ndim != 1:
        raise TypeError("`pred_quantiles` must be a 1D numpy array.")

    if not isinstance(quantile_levels, np.ndarray) or quantile_levels.ndim != 1:
        raise TypeError("`quantile_levels` must be a 1D numpy array.")

    if len(pred_quantiles) != len(quantile_levels):
        raise ValueError(
            "The number of predicted quantiles and quantile levels must be equal."
        )

    sorted_indices = np.argsort(pred_quantiles)
    pred_quantiles = pred_quantiles[sorted_indices]
    quantile_levels = quantile_levels[sorted_indices]

    # Define the empirical CDF function using interpolation
    def empirical_cdf(x):
        return np.interp(x, pred_quantiles, quantile_levels, left=0.0, right=1.0)

    # Define the CRPS integrand
    def crps_integrand(x):
        return (empirical_cdf(x) - (x >= y_true)) ** 2

    # Integration bounds: Extend slightly beyond predicted quantiles
    xmin = np.min(pred_quantiles) * 0.9
    xmax = np.max(pred_quantiles) * 1.1

    # Create a fine grid of x values for integration
    x_values = np.linspace(xmin, xmax, 1000)

    # Compute the integrand values and integrate using the trapezoidal rule
    integrand_values = crps_integrand(x_values)
    if np.__version__ >= "2.0.0":
        crps = np.trapezoid(integrand_values, x=x_values)
    else:
        crps = np.trapz(integrand_values, x_values)

    return crps


def calculate_coverage(
    y_true: np.ndarray | pd.Series,
    lower_bound: np.ndarray | pd.Series,
    upper_bound: np.ndarray | pd.Series,
) -> float:
    """Calculate coverage of a given interval.

    Coverage is the proportion of true values that fall within the interval.

    Args:
        y_true: True values of the target variable.
        lower_bound: Lower bound of the interval.
        upper_bound: Upper bound of the interval.

    Returns:
        Coverage of the interval.

    Examples:
        >>> from spotforecast2.forecaster.metrics import calculate_coverage
        >>> y_true = np.array([1, 2, 3, 4, 5])
        >>> lower_bound = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
        >>> upper_bound = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
        >>> coverage = calculate_coverage(y_true, lower_bound, upper_bound)
        >>> coverage == 1.0  # All values within bounds
        True
    """
    if not isinstance(y_true, (np.ndarray, pd.Series)) or y_true.ndim != 1:
        raise TypeError("`y_true` must be a 1D numpy array or pandas Series.")

    if not isinstance(lower_bound, (np.ndarray, pd.Series)) or lower_bound.ndim != 1:
        raise TypeError("`lower_bound` must be a 1D numpy array or pandas Series.")

    if not isinstance(upper_bound, (np.ndarray, pd.Series)) or upper_bound.ndim != 1:
        raise TypeError("`upper_bound` must be a 1D numpy array or pandas Series.")

    y_true = np.asarray(y_true)
    lower_bound = np.asarray(lower_bound)
    upper_bound = np.asarray(upper_bound)

    if y_true.shape != lower_bound.shape or y_true.shape != upper_bound.shape:
        raise ValueError(
            "`y_true`, `lower_bound` and `upper_bound` must have the same shape."
        )

    coverage = np.mean(np.logical_and(y_true >= lower_bound, y_true <= upper_bound))

    return coverage


def create_mean_pinball_loss(alpha: float) -> callable:
    """Create pinball loss for a given quantile.

    Also known as quantile loss. Internally, it uses the `mean_pinball_loss`
    function from scikit-learn.

    Args:
        alpha: Quantile for which the Pinball loss is calculated.
            Must be between 0 and 1, inclusive.

    Returns:
        Mean Pinball loss function for the given quantile.

    Examples:
        >>> from spotforecast2.forecaster.metrics import create_mean_pinball_loss
        >>> pinball_loss_50 = create_mean_pinball_loss(alpha=0.5)
        >>> y_true = np.array([1, 2, 3, 4, 5])
        >>> y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])
        >>> loss = pinball_loss_50(y_true, y_pred)
        >>> loss >= 0
        True
    """
    if not (0 <= alpha <= 1):
        raise ValueError("alpha must be between 0 and 1, both inclusive.")

    def mean_pinball_loss_q(y_true, y_pred):
        return mean_pinball_loss(y_true, y_pred, alpha=alpha)

    return mean_pinball_loss_q


def symmetric_mean_absolute_percentage_error(
    y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series
) -> float:
    """Compute the Symmetric Mean Absolute Percentage Error (SMAPE).

    SMAPE is a relative error metric used to measure the accuracy
    of forecasts. Unlike MAPE, it is symmetric and prevents division
    by zero by averaging the absolute values of actual and predicted values.

    The result is expressed as a percentage and ranges from 0%
    (perfect prediction) to 200% (maximum error).

    Args:
        y_true: True values of the target variable.
        y_pred: Predicted values of the target variable.

    Returns:
        SMAPE value as a percentage.

    Examples:
        >>> from spotforecast2.forecaster.metrics import symmetric_mean_absolute_percentage_error
        >>> y_true = np.array([100, 200, 0])
        >>> y_pred = np.array([110, 180, 10])
        >>> result = symmetric_mean_absolute_percentage_error(y_true, y_pred)
        >>> 0 <= result <= 200
        True
    """

    if not isinstance(y_true, (pd.Series, np.ndarray)):
        raise TypeError("`y_true` must be a pandas Series or numpy ndarray.")
    if not isinstance(y_pred, (pd.Series, np.ndarray)):
        raise TypeError("`y_pred` must be a pandas Series or numpy ndarray.")
    if len(y_true) != len(y_pred):
        raise ValueError("`y_true` and `y_pred` must have the same length.")
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("`y_true` and `y_pred` must have at least one element.")

    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # NOTE: Avoid division by zero
    mask = denominator != 0
    smape_values = np.zeros_like(denominator)
    smape_values[mask] = numerator[mask] / denominator[mask]

    smape = 100 * np.mean(smape_values)

    return smape
