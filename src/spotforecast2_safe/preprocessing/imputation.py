# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

import logging
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from spotforecast2_safe.preprocessing.linearly_interpolate_ts import (
    LinearlyInterpolateTS,
)

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
        >>> zero_weights = pd.Series([0.0, 0.0, 0.0], index=[0, 1, 2])
        >>> wf_zero = WeightFunction(zero_weights)
        >>> wf_zero(pd.Index([0, 1])) is None
        True
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
        self,
        index: Union[pd.Index, np.ndarray, list],
    ) -> Optional[np.ndarray]:
        """Return sample weights for the given index, or None if all weights are zero.

        Computes the weights via :func:`custom_weights`. When every weight in
        the result sums to zero, the method logs a warning and returns ``None``.
        ``ForecasterRecursive`` interprets this as "use uniform weights".

        Args:
            index: Index or indices to get weights for.

        Returns:
            Numpy array of weight values, or ``None`` when the sum of all
            weights is zero for a pandas Index.

        Examples:
            >>> import pandas as pd
            >>> weights = pd.Series([1.0, 0.5, 0.0], index=[0, 1, 2])
            >>> wf = WeightFunction(weights)
            >>> wf(pd.Index([0, 1]))
            array([1. , 0.5])
            >>> wf(2)
            0.0
            >>> wf(pd.Index([2])) is None
            True
        """
        result = custom_weights(index, self.weights_series)

        if isinstance(index, pd.Index) and np.sum(result) == 0:
            logger.warning(
                "WeightFunction: all sample weights for the requested index are "
                "zero. Returning None so ForecasterRecursive uses uniform "
                "weighting."
            )
            return None

        return result

    def __repr__(self):
        """String representation."""
        return (
            f"WeightFunction(weights_series with "
            f"{len(self.weights_series)} entries)"
        )


def custom_weights(index, weights_series: pd.Series) -> Union[float, np.ndarray]:
    """Return sample weights for a scalar index or a pandas Index.

    Args:
        index:
            Scalar index value or pandas Index.
        weights_series:
            Series containing weights.

    Returns:
        Weight value for scalar input, or numpy array for pandas Index input.

    Raises:
        ValueError:
            If requested index values are not present in ``weights_series``.

    Examples:
        >>> import pandas as pd
        >>> weights = pd.Series([1.0, 0.5, 0.0], index=[0, 1, 2])
        >>> custom_weights(1, weights)
        0.5
        >>> custom_weights(pd.Index([0, 2]), weights)
        array([1., 0.])
    """
    if isinstance(index, pd.Index):
        if not index.isin(weights_series.index).all():
            raise ValueError("Index not found in weights_series.")
        return weights_series.loc[index].values

    if index not in weights_series.index:
        raise ValueError("Index not found in weights_series.")

    return weights_series.loc[index]


def get_missing_weights(
    data: pd.DataFrame,
    window_size: Union[int, np.integer, list[int], tuple[int, ...]] = 72,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return imputed DataFrame and a series indicating missing-value weights.

    The function first forward-/backward-fills missing values and then builds
    a weight series. Rows in or near gaps receive weight 0.0; all other rows
    receive weight 1.0.

    ``window_size`` may be either a single integer or a list/tuple of integers.
    If a list/tuple is passed, the largest value is used for the imputation
    weighting context. This keeps the function compatible with pipelines that
    use multiple rolling-window feature sizes such as ``[72, 168, 720]``.

    Args:
        data:
            Input DataFrame.
        window_size:
            Rolling window size used to mark rows near missing values.
            May be a single integer or a list/tuple of integers.
        verbose:
            Whether to print diagnostics.

    Returns:
        Tuple containing:

        - imputed DataFrame
        - weight series with 0.0 around gaps and 1.0 elsewhere

    Raises:
        ValueError:
            If data is empty, window size is invalid, or window size is too
            large for the data.

    Examples:
        >>> idx = pd.date_range("2024-01-01", periods=5, freq="h")
        >>> data = pd.DataFrame({"load": [1.0, None, 3.0, 4.0, 5.0]}, index=idx)
        >>> filled, weights = get_missing_weights(data, window_size=2)
        >>> filled.isna().sum().sum()
        0
        >>> len(weights)
        5
        >>> filled2, weights2 = get_missing_weights(data, window_size=[2, 3])
        >>> len(weights2)
        5
    """
    if data.shape[0] == 0:
        raise ValueError("Input data is empty.")

    if isinstance(window_size, (int, np.integer)):
        window_size = int(window_size)

        if window_size <= 0:
            raise ValueError(
                f"window_size must be a positive integer, got {window_size}"
            )

    else:
        if not window_size:
            raise ValueError("window_size list must not be empty")

        window_sizes = [int(ws) for ws in window_size]

        if any(ws <= 0 for ws in window_sizes):
            raise ValueError(
                f"all window sizes must be positive, got {window_size}"
            )

        # get_missing_weights internally needs one scalar window size.
        # Use the largest window as conservative context.
        window_size = max(window_sizes)

    if window_size >= data.shape[0]:
        raise ValueError(
            "window_size must be smaller than the number of rows in data."
        )

    missing_indices = data.index[data.isnull().any(axis=1)]
    n_missing = len(missing_indices)

    if verbose:
        pct_missing = (n_missing / len(data)) * 100
        print(f"Number of rows with missing values: {n_missing}")
        print(f"Percentage of rows with missing values: {pct_missing:.2f}%")
        print(f"missing_indices: {missing_indices}")

    data = data.ffill().bfill()

    is_missing = pd.Series(0, index=data.index)
    is_missing.loc[missing_indices] = 1

    # weights_series: 0 if in/near gap, 1 otherwise
    weights_series = (
        1.0 - is_missing.rolling(window=window_size + 1, min_periods=1).max()
    )

    if verbose:
        n_missing_after = int((weights_series == 0).sum())
        pct_missing_after = (n_missing_after / len(data)) * 100
        print(
            f"Number of rows with missing weights after processing: "
            f"{n_missing_after}"
        )
        print(
            f"Percentage of rows with missing weights after processing: "
            f"{pct_missing_after:.2f}%"
        )

    return data, weights_series


def apply_imputation(
    df_pipeline: pd.DataFrame,
    config: Any,
    logger: logging.Logger,
    verbose: bool = False,
) -> tuple[pd.DataFrame, "WeightFunction | None"]:
    """Apply imputation to a DataFrame based on the method specified in config.

    Supports two strategies:

    - ``"weighted"``: forward-fill then backward-fill gaps, then build a
      :class:`WeightFunction` that down-weights training rows near any gap.
      Rows inside or near a gap receive weight 0. The rolling window
      ``config.window_size`` controls how far the penalty extends.
    - ``"linear"``: apply :class:`LinearlyInterpolateTS` column-by-column.

    A diagnostic summary is always written to the logger.

    Args:
        df_pipeline:
            DataFrame to impute.
        config:
            Configuration object that must expose ``imputation_method``,
            ``targets`` and ``window_size``.
        logger:
            Logger used to emit diagnostics.
        verbose:
            Whether to print additional diagnostics.

    Returns:
        Tuple containing the imputed DataFrame and either a WeightFunction or
        ``None`` when linear interpolation is used.

    Raises:
        ValueError:
            If ``config.imputation_method`` is neither ``"weighted"`` nor
            ``"linear"``.

    Examples:
        >>> import logging
        >>> import pandas as pd
        >>> from types import SimpleNamespace
        >>> idx = pd.date_range("2024-01-01", periods=5, freq="h")
        >>> df = pd.DataFrame({"A": [1.0, None, 3.0, 4.0, 5.0]}, index=idx)
        >>> config = SimpleNamespace(
        ...     imputation_method="linear",
        ...     targets=["A"],
        ...     window_size=3,
        ... )
        >>> logger = logging.getLogger("demo")
        >>> imputed, weight_func = apply_imputation(df, config, logger)
        >>> imputed.isna().sum().sum()
        0
        >>> weight_func is None
        True
    """
    nan_before = int(df_pipeline.isnull().sum().sum())

    logger.info(
        "apply_imputation: NaN cells before imputation: %d "
        "(method=%r, shape=%s)",
        nan_before,
        config.imputation_method,
        df_pipeline.shape,
    )

    weight_func = None

    if config.imputation_method == "weighted":
        logger.info("Applying weighted imputation (n2n style)...")

        df_pipeline, weights_series = get_missing_weights(
            df_pipeline,
            window_size=config.window_size,
            verbose=verbose,
        )

        weight_func = WeightFunction(weights_series)

        logger.info(
            "Weight function created with %d entries.",
            len(weights_series),
        )

    elif config.imputation_method == "linear":
        logger.info("Applying linear interpolation...")

        # passthrough lets residual endpoint NaNs survive into the
        # post-imputation NaN count + WARNING below, matching the
        # surrounding logger.warning contract.
        interpolator = LinearlyInterpolateTS(on_missing="passthrough")

        # LinearlyInterpolateTS expects a Series; apply per column.
        for col in config.targets:
            series = df_pipeline[col]
            df_pipeline[col] = interpolator.fit_transform(series)

    else:
        raise ValueError(
            f"Unknown imputation_method: {config.imputation_method!r}. "
            "Expected one of: 'weighted', 'linear'."
        )

    nan_after = int(df_pipeline.isnull().sum().sum())

    logger.info(
        "apply_imputation: NaN cells after imputation: %d",
        nan_after,
    )

    if nan_after > 0:
        logger.warning(
            "apply_imputation: %d NaN cell(s) remain after imputation. "
            "Consider reviewing the data or adjusting the imputation method.",
            nan_after,
        )

    return df_pipeline, weight_func
