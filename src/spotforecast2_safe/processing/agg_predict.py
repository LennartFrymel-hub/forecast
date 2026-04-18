# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


def agg_predict(
    predictions: pd.DataFrame,
    weights: Optional[Union[Dict[str, float], List[float], np.ndarray]] = None,
) -> pd.Series:
    """Aggregates multiple prediction columns into a single combined prediction series.

    The combination is a weighted sum of the prediction columns. If no weights are provided,
    a default weighting scheme based on specific predefined columns is used.

    Args:
        predictions (pd.DataFrame): DataFrame containing the prediction columns.
        weights (Optional[Union[Dict[str, float], List[float], np.ndarray]]):
            Dictionary mapping column names to their weights, or a list/array of weights
            corresponding to the order of columns in `predictions`.
            If None, defaults to summing all columns (weight=1.0 for each column).

    Returns:
        pd.Series: A Series containing the aggregated values.

    Raises:
        ValueError: If a column specified in weights (or default weights) is missing from predictions.
        ValueError: If weights is a list/array and its length does not match the number of columns in predictions.

    Examples:
        >>> from spotforecast2_safe.processing import agg_predict
        >>> import pandas as pd
        >>> df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        >>> agg_predict(df, weights={"A": 1.0, "B": -1.0})
        0   -2.0
        1   -2.0
        dtype: float64
        >>> agg_predict(df, weights=[0.5, 2.0])
        0    6.5
        1    9.0
        dtype: float64
    """
    if weights is None:
        # Default to summing all columns
        weights = {col: 1.0 for col in predictions.columns}

    if isinstance(weights, (list, np.ndarray)):
        if len(weights) != len(predictions.columns):
            raise ValueError(
                f"Length of weights ({len(weights)}) does not match number of columns in predictions ({len(predictions.columns)})"
            )
        # Convert to dictionary using column order
        weights = dict(zip(predictions.columns, weights))

    combined = pd.Series(0.0, index=predictions.index)

    missing_cols = [col for col in weights.keys() if col not in predictions.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in predictions dataframe: {missing_cols}")

    for col, weight in weights.items():
        combined += predictions[col] * weight

    return combined
