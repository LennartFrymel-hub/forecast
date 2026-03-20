# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import logging
import pandas as pd
from typing import Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class WeightFunction:
    """Callable class for sample weights that can be pickled.

    This class wraps the weights_series and provides a callable interface
    compatible with ForecasterRecursive's weight_func parameter. Unlike
    local functions with closures, instances of this class can be pickled
    using standard pickle/joblib.

    When all weights for the requested index sum to zero (e.g. the entire
    training window falls inside gap-penalty zones after outlier detection),
    ``__call__`` returns ``None`` instead of an all-zero array.
    ``ForecasterRecursive.create_sample_weights`` treats a ``None`` return as
    "no weighting", avoiding a ``ValueError`` while still applying the
    weighted imputation for windows that do contain positive weights.

    Args:
        weights_series: Series containing weight values for each index.

    Examples:
        >>> import pandas as pd
        >>> import pickle
        >>> weights = pd.Series([1.0, 0.9, 0.8], index=[0, 1, 2])
        >>> weight_func = WeightFunction(weights)
        >>> weight_func(pd.Index([0, 1]))
        array([1. , 0.9])
        >>> # Returns None when all weights in the window are zero
        >>> zero_weights = pd.Series([0.0, 0.0, 0.0], index=[0, 1, 2])
        >>> wf_zero = WeightFunction(zero_weights)
        >>> wf_zero(pd.Index([0, 1])) is None
        True
        >>> # Can be pickled
        >>> pickled = pickle.dumps(weight_func)
        >>> unpickled = pickle.loads(pickled)
        >>> unpickled(pd.Index([0, 1]))
        array([1. , 0.9])
    """

    def __init__(self, weights_series: pd.Series):
        """Initialize with a weights series.

        Args:
            weights_series: Series containing weight values for each index.
        """
        self.weights_series = weights_series

    def __call__(
        self, index: Union[pd.Index, np.ndarray, list]
    ) -> Optional[np.ndarray]:
        """Return sample weights for the given index, or None if all weights are zero.

        Computes the weights via :func:`custom_weights`.  When every weight in
        the result sums to zero — which happens when the entire requested window
        falls within gap-penalty zones created by outlier detection — the method
        logs a warning and returns ``None``.  ``ForecasterRecursive`` interprets
        a ``None`` return as "use uniform weights", preventing a
        ``ValueError: sample_weight cannot be normalized`` crash.

        Args:
            index: Index or indices to get weights for.

        Returns:
            Numpy array of weight values, or ``None`` when the sum of all
            weights is zero (degenerate window).

        Examples:
            >>> import pandas as pd
            >>> weights = pd.Series([1.0, 0.5, 0.0], index=[0, 1, 2])
            >>> wf = WeightFunction(weights)
            >>> wf(pd.Index([0, 1]))
            array([1. , 0.5])
            >>> # Degenerate window → None
            >>> wf(pd.Index([2]))  is None
            True
        """
        result = custom_weights(index, self.weights_series)
        if np.sum(result) == 0:
            logger.warning(
                "WeightFunction: all sample weights for the requested index are "
                "zero (the window falls entirely within gap-penalty zones). "
                "Returning None so ForecasterRecursive uses uniform weighting."
            )
            return None
        return result

    def __repr__(self):
        """String representation."""
        return f"WeightFunction(weights_series with {len(self.weights_series)} entries)"


def custom_weights(index, weights_series: pd.Series) -> float:
    """
    Return 0 if index is in or near any gap.

    Args:
        index (pd.Index):
            The index to check.
        weights_series (pd.Series):
            Series containing weights.

    Returns:
        float: The weight corresponding to the index.

    Examples:
        >>> from spotforecast2_safe.data.fetch_data import fetch_data
        >>> from spotforecast2_safe.preprocessing.imputation import custom_weights
        >>> data = fetch_data()
        >>> _, missing_weights = get_missing_weights(data, window_size=72, verbose=False)
        >>> for idx in data.index[:5]:
        ...     weight = custom_weights(idx, missing_weights)
        ...     print(f"Index: {idx}, Weight: {weight}")
    """
    # do plausibility check
    if isinstance(index, pd.Index):
        if not index.isin(weights_series.index).all():
            raise ValueError("Index not found in weights_series.")
        return weights_series.loc[index].values

    if index not in weights_series.index:
        raise ValueError("Index not found in weights_series.")
    return weights_series.loc[index]


def get_missing_weights(
    data: pd.DataFrame, window_size: int = 72, verbose: bool = False
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return imputed DataFrame and a series indicating missing weights.

    Args:
        data (pd.DataFrame):
            The input dataset.
        window_size (int):
            The size of the rolling window to consider for missing values.
        verbose (bool):
            Whether to print additional information.

    Returns:
        Tuple[pd.DataFrame, pd.Series]:
            A tuple containing the forward and backward filled DataFrame and a numeric series (0.0 or 1.0) where 0.0 indicates a weight for missing values/gaps.

    Examples:
        >>> from spotforecast2_safe.data.fetch_data import fetch_data
        >>> from spotforecast2_safe.preprocessing.imputation import get_missing_weights
        >>> data = fetch_data()
        >>> filled_data, missing_weights = get_missing_weights(data, window_size=72, verbose=True)

    """
    # first perform some checks if dataframe has enough data and if window_size is appropriate
    if data.shape[0] == 0:
        raise ValueError("Input data is empty.")
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")
    if window_size >= data.shape[0]:
        raise ValueError("window_size must be smaller than the number of rows in data.")

    missing_indices = data.index[data.isnull().any(axis=1)]
    n_missing = len(missing_indices)
    if verbose:
        pct_missing = (n_missing / len(data)) * 100
        print(f"Number of rows with missing values: {n_missing}")
        print(f"Percentage of rows with missing values: {pct_missing:.2f}%")
        print(f"missing_indices: {missing_indices}")
    data = data.ffill()
    data = data.bfill()

    is_missing = pd.Series(0, index=data.index)
    is_missing.loc[missing_indices] = 1
    # weights_series: 0 if in/near gap, 1 otherwise
    weights_series = (
        1.0 - is_missing.rolling(window=window_size + 1, min_periods=1).max()
    )

    if verbose:
        n_missing_after = (weights_series == 0).sum()
        pct_missing_after = (n_missing_after / len(data)) * 100
        print(
            f"Number of rows with missing weights after processing: {n_missing_after}"
        )
        print(
            f"Percentage of rows with missing weights after processing: {pct_missing_after:.2f}%"
        )
    return data, weights_series
