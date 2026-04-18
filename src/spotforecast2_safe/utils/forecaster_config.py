# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

"""
Forecaster configuration utilities.

This module provides functions for initializing and validating forecaster
configuration parameters like lags and weights.
"""

from typing import Any, List, Optional, Tuple, Union

import numpy as np


def initialize_lags(
    forecaster_name: str, lags: Any
) -> Tuple[Optional[np.ndarray], Optional[List[str]], Optional[int]]:
    """
    Validate and normalize lag specification for forecasting.

    This function converts various lag specifications (int, list, tuple, range, ndarray)
    into a standardized format: sorted numpy array, lag names, and maximum lag value.

    Args:
        forecaster_name: Name of the forecaster class for error messages.
        lags: Lag specification in one of several formats:
            - int: Creates lags from 1 to lags (e.g., 5 → [1,2,3,4,5])
            - list/tuple/range: Converted to numpy array
            - numpy.ndarray: Validated and used directly
            - None: Returns (None, None, None)

    Returns:
        Tuple containing:
        - lags: Sorted numpy array of lag values (or None)
        - lags_names: List of lag names like ['lag_1', 'lag_2', ...] (or None)
        - max_lag: Maximum lag value (or None)

    Raises:
        ValueError: If lags < 1, empty array, or not 1-dimensional.
        TypeError: If lags is not an integer, not in the right format for the forecaster,
            or array contains non-integer values.

    Examples:
        >>> import numpy as np
        >>> from spotforecast2_safe.utils.forecaster_config import initialize_lags
        >>>
        >>> # Integer input
        >>> lags, names, max_lag = initialize_lags("ForecasterRecursive", 3)
        >>> lags
        array([1, 2, 3])
        >>> names
        ['lag_1', 'lag_2', 'lag_3']
        >>> max_lag
        3
        >>>
        >>> # List input
        >>> lags, names, max_lag = initialize_lags("ForecasterRecursive", [1, 3, 5])
        >>> lags
        array([1, 3, 5])
        >>> names
        ['lag_1', 'lag_3', 'lag_5']
        >>>
        >>> # Range input
        >>> lags, names, max_lag = initialize_lags("ForecasterRecursive", range(1, 4))
        >>> lags
        array([1, 2, 3])
        >>>
        >>> # None input
        >>> lags, names, max_lag = initialize_lags("ForecasterRecursive", None)
        >>> lags is None
        True
        >>>
        >>> # Invalid: lags < 1
        >>> try:
        ...     initialize_lags("ForecasterRecursive", 0)
        ... except ValueError as e:
        ...     print("Error: Minimum value of lags allowed is 1")
        Error: Minimum value of lags allowed is 1
        >>>
        >>> # Invalid: negative lags
        >>> try:
        ...     initialize_lags("ForecasterRecursive", [1, -2, 3])
        ... except ValueError as e:
        ...     print("Error: Minimum value of lags allowed is 1")
        Error: Minimum value of lags allowed is 1
    """
    lags_names = None
    max_lag = None

    if lags is not None:
        if isinstance(lags, int):
            if lags < 1:
                raise ValueError("Minimum value of lags allowed is 1.")
            lags = np.arange(1, lags + 1)

        if isinstance(lags, (list, tuple, range)):
            lags = np.array(lags)

        if isinstance(lags, np.ndarray):
            if lags.size == 0:
                return None, None, None
            if lags.ndim != 1:
                raise ValueError("`lags` must be a 1-dimensional array.")
            if not np.issubdtype(lags.dtype, np.integer):
                raise TypeError("All values in `lags` must be integers.")
            if np.any(lags < 1):
                raise ValueError("Minimum value of lags allowed is 1.")
        else:
            if forecaster_name == "ForecasterDirectMultiVariate":
                raise TypeError(
                    f"`lags` argument must be a dict, int, 1d numpy ndarray, range, "
                    f"tuple or list. Got {type(lags)}."
                )
            else:
                raise TypeError(
                    f"`lags` argument must be an int, 1d numpy ndarray, range, "
                    f"tuple or list. Got {type(lags)}."
                )

        lags = np.sort(lags)
        lags_names = [f"lag_{i}" for i in lags]
        max_lag = int(max(lags))

    return lags, lags_names, max_lag


def initialize_weights(
    forecaster_name: str, estimator: Any, weight_func: Any, series_weights: Any
) -> Tuple[Any, Optional[Union[str, dict]], Any]:
    """
    Validate and initialize weight function configuration for forecasting.

    This function validates weight_func and series_weights, extracts source code
    from weight functions for serialization, and checks if the estimator supports
    sample weights in its fit method.

    Args:
        forecaster_name: Name of the forecaster class.
        estimator: Scikit-learn compatible estimator or pipeline.
        weight_func: Weight function specification:
            - Callable: Single weight function
            - dict: Dictionary of weight functions (for MultiSeries forecasters)
            - None: No weighting
        series_weights: Dictionary of series-level weights (for MultiSeries forecasters).
            - dict: Maps series names to weight values
            - None: No series weighting

    Returns:
        Tuple containing:
        - weight_func: Validated weight function (or None if invalid)
        - source_code_weight_func: Source code of weight function(s) for serialization (or None)
        - series_weights: Validated series weights (or None if invalid)

    Raises:
        TypeError: If weight_func is not Callable/dict (depending on forecaster type),
            or if series_weights is not a dict.

    Warnings:
        IgnoredArgumentWarning: If estimator doesn't support sample_weight.

    Examples:
        >>> import numpy as np
        >>> from sklearn.linear_model import Ridge
        >>> from spotforecast2_safe.utils.forecaster_config import initialize_weights
        >>>
        >>> # Simple weight function
        >>> def custom_weights(index):
        ...     return np.ones(len(index))
        >>>
        >>> estimator = Ridge()
        >>> wf, source, sw = initialize_weights(
        ...     "ForecasterRecursive", estimator, custom_weights, None
        ... )
        >>> wf is not None
        True
        >>> isinstance(source, str)
        True
        >>>
        >>> # No weight function
        >>> wf, source, sw = initialize_weights(
        ...     "ForecasterRecursive", estimator, None, None
        ... )
        >>> wf is None
        True
        >>> source is None
        True
        >>>
        >>> # Invalid type for non-MultiSeries forecaster
        >>> try:
        ...     initialize_weights("ForecasterRecursive", estimator, "invalid", None)
        ... except TypeError as e:
        ...     print("Error: weight_func must be Callable")
        Error: weight_func must be Callable
    """
    import inspect
    import warnings
    from collections.abc import Callable

    # Import IgnoredArgumentWarning if available, otherwise define locally
    try:
        from spotforecast2_safe.exceptions import IgnoredArgumentWarning
    except ImportError:

        class IgnoredArgumentWarning(UserWarning):
            """Warning for ignored arguments."""

            pass

    source_code_weight_func = None

    if weight_func is not None:
        if forecaster_name in ["ForecasterRecursiveMultiSeries"]:
            if not isinstance(weight_func, (Callable, dict)):
                raise TypeError(
                    f"Argument `weight_func` must be a Callable or a dict of "
                    f"Callables. Got {type(weight_func)}."
                )
        elif not isinstance(weight_func, Callable):
            raise TypeError(
                f"Argument `weight_func` must be a Callable. Got {type(weight_func)}."
            )

        if isinstance(weight_func, dict):
            source_code_weight_func = {}
            for key in weight_func:
                try:
                    source_code_weight_func[key] = inspect.getsource(weight_func[key])
                except (OSError, TypeError):
                    # OSError: source not available, TypeError: callable class instance
                    source_code_weight_func[key] = (
                        f"<source unavailable: {weight_func[key]!r}>"
                    )
        else:
            try:
                source_code_weight_func = inspect.getsource(weight_func)
            except (OSError, TypeError):
                # OSError: source not available (e.g., built-in, lambda in REPL)
                # TypeError: callable class instance (e.g., WeightFunction)
                # In these cases, we can't get source but the object can still be pickled
                source_code_weight_func = f"<source unavailable: {weight_func!r}>"

        if "sample_weight" not in inspect.signature(estimator.fit).parameters:
            warnings.warn(
                f"Argument `weight_func` is ignored since estimator {estimator} "
                f"does not accept `sample_weight` in its `fit` method.",
                IgnoredArgumentWarning,
            )
            weight_func = None
            source_code_weight_func = None

    if series_weights is not None:
        if not isinstance(series_weights, dict):
            raise TypeError(
                f"Argument `series_weights` must be a dict of floats or ints."
                f"Got {type(series_weights)}."
            )
        if "sample_weight" not in inspect.signature(estimator.fit).parameters:
            warnings.warn(
                f"Argument `series_weights` is ignored since estimator {estimator} "
                f"does not accept `sample_weight` in its `fit` method.",
                IgnoredArgumentWarning,
            )
            series_weights = None

    return weight_func, source_code_weight_func, series_weights


def check_select_fit_kwargs(estimator: Any, fit_kwargs: Optional[dict] = None) -> dict:
    """
    Check if `fit_kwargs` is a dict and select only keys used by estimator's `fit`.

    This function validates that fit_kwargs is a dictionary, warns about unused arguments,
    removes 'sample_weight' (which should be handled via weight_func), and returns
    a dictionary containing only the arguments accepted by the estimator's fit method.

    Args:
        estimator: Scikit-learn compatible estimator.
        fit_kwargs: Dictionary of arguments to pass to the estimator's fit method.

    Returns:
        Dictionary with only the arguments accepted by the estimator's fit method.

    Raises:
        TypeError: If fit_kwargs is not a dict.

    Warnings:
        IgnoredArgumentWarning: If fit_kwargs contains keys not used by fit method,
            or if 'sample_weight' is present (it gets removed).

    Examples:
        >>> from sklearn.linear_model import Ridge
        >>> from spotforecast2_safe.utils.forecaster_config import check_select_fit_kwargs
        >>>
        >>> estimator = Ridge()
        >>> # Valid argument for Ridge.fit
        >>> kwargs = {"sample_weight": [1, 1], "invalid_arg": 10}
        >>> # sample_weight is removed (should be passed via weight_func in forecaster)
        >>> # invalid_arg is ignored
        >>> filtered = check_select_fit_kwargs(estimator, kwargs)
        >>> filtered
        {}
    """
    import inspect
    import warnings

    # Import IgnoredArgumentWarning if available, otherwise define locally
    try:
        from spotforecast2_safe.exceptions import IgnoredArgumentWarning
    except ImportError:

        class IgnoredArgumentWarning(UserWarning):
            """Warning for ignored arguments."""

            pass

    if fit_kwargs is None:
        fit_kwargs = {}
    else:
        if not isinstance(fit_kwargs, dict):
            raise TypeError(
                f"Argument `fit_kwargs` must be a dict. Got {type(fit_kwargs)}."
            )

        # Get parameters accepted by estimator.fit
        fit_params = inspect.signature(estimator.fit).parameters

        # Identify unused keys
        non_used_keys = [k for k in fit_kwargs.keys() if k not in fit_params]
        if non_used_keys:
            warnings.warn(
                f"Argument/s {non_used_keys} ignored since they are not used by the "
                f"estimator's `fit` method.",
                IgnoredArgumentWarning,
            )

        # Handle sample_weight specially
        if "sample_weight" in fit_kwargs.keys():
            warnings.warn(
                "The `sample_weight` argument is ignored. Use `weight_func` to pass "
                "a function that defines the individual weights for each sample "
                "based on its index.",
                IgnoredArgumentWarning,
            )
            del fit_kwargs["sample_weight"]

        # Select only the keyword arguments allowed by the estimator's `fit` method.
        # Note: We need to re-check keys because sample_weight might have been deleted but it might be in fit_params
        # If it was deleted, it is no longer in fit_kwargs, so this comprehension is safe
        fit_kwargs = {k: v for k, v in fit_kwargs.items() if k in fit_params}

    return fit_kwargs
