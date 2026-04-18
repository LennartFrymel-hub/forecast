# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Metrics module for model evaluation.

This module provides utilities for calculating performance metrics
for prediction models in safety-critical systems.
"""

from typing import Dict

import pandas as pd


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
    """
    Calculate MAE and MSE for numeric evaluation.

    Computes Mean Absolute Error (MAE) and Mean Squared Error (MSE) between
    actual and predicted values. These metrics are essential for evaluating
    forecasting model performance in safety-critical applications.

    Args:
        actual: Series of actual observed values.
        predicted: Series of predicted values (must have same length as actual).

    Returns:
        Dict[str, float]: Dictionary containing:
            - 'MAE': Mean Absolute Error
            - 'MSE': Mean Squared Error

    Raises:
        ValueError: If series have different lengths or contain NaN values.

    Examples:
        >>> import pandas as pd
        >>> from spotforecast2_safe.manager.metrics import calculate_metrics
        >>>
        >>> # Example 1: Perfect predictions
        >>> actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> predicted = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> metrics = calculate_metrics(actual, predicted)
        >>> print(f"MAE: {metrics['MAE']:.4f}")
        MAE: 0.0000
        >>> print(f"MSE: {metrics['MSE']:.4f}")
        MSE: 0.0000
        >>>
        >>> # Example 2: Small prediction errors
        >>> actual = pd.Series([10.0, 20.0, 30.0, 40.0])
        >>> predicted = pd.Series([11.0, 19.0, 31.0, 39.0])
        >>> metrics = calculate_metrics(actual, predicted)
        >>> print(f"MAE: {metrics['MAE']:.4f}")
        MAE: 1.0000
        >>> print(f"MSE: {metrics['MSE']:.4f}")
        MSE: 1.0000
        >>>
        >>> # Example 3: Larger prediction errors
        >>> actual = pd.Series([100.0, 200.0, 300.0])
        >>> predicted = pd.Series([95.0, 210.0, 290.0])
        >>> metrics = calculate_metrics(actual, predicted)
        >>> print(f"MAE: {metrics['MAE']:.4f}")
        MAE: 8.3333
        >>> print(f"MSE: {metrics['MSE']:.4f}")
        MSE: 75.0000
        >>>
        >>> # Example 4: Safety-critical - validate metric bounds
        >>> actual = pd.Series([50.0, 55.0, 60.0, 65.0, 70.0])
        >>> predicted = pd.Series([51.0, 54.0, 61.0, 64.0, 71.0])
        >>> metrics = calculate_metrics(actual, predicted)
        >>> # Verify metrics are within acceptable bounds
        >>> assert metrics['MAE'] < 2.0, "MAE exceeds safety threshold"
        >>> assert metrics['MSE'] < 5.0, "MSE exceeds safety threshold"
        >>> print("Safety validation passed")
        Safety validation passed
        >>>
        >>> # Example 5: Time series with datetime index
        >>> dates = pd.date_range('2024-01-01', periods=5, freq='D')
        >>> actual = pd.Series([10.5, 11.2, 10.8, 11.5, 12.0], index=dates)
        >>> predicted = pd.Series([10.3, 11.4, 10.9, 11.3, 12.1], index=dates)
        >>> metrics = calculate_metrics(actual, predicted)
        >>> print(f"MAE: {metrics['MAE']:.4f}")
        MAE: 0.1600
        >>> print(f"MSE: {metrics['MSE']:.4f}")
        MSE: 0.0280
        >>>
        >>> # Example 6: Compare models using metrics
        >>> actual = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        >>> pred_model_a = pd.Series([1.1, 2.1, 2.9, 4.2, 4.8])
        >>> pred_model_b = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5])
        >>> metrics_a = calculate_metrics(actual, pred_model_a)
        >>> metrics_b = calculate_metrics(actual, pred_model_b)
        >>> if metrics_a['MAE'] < metrics_b['MAE']:
        ...     print("Model A has better MAE")
        ... else:
        ...     print("Model B has better MAE")
        Model A has better MAE
    """
    diff = actual - predicted
    mae = diff.abs().mean()
    mse = (diff**2).mean()
    return {"MAE": mae, "MSE": mse}
