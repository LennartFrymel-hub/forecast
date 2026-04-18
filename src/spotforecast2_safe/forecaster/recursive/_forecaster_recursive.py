# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause

from __future__ import annotations

import inspect
import sys
import warnings
from copy import copy
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.linear_model._base import LinearModel
from sklearn.pipeline import Pipeline

from spotforecast2_safe.exceptions import (
    DataTransformationWarning,
    NotFittedError,
    ResidualsUsageWarning,
    set_skforecast_warnings,
)
from spotforecast2_safe.forecaster.base import ForecasterBase
from spotforecast2_safe.forecaster.utils import (
    check_extract_values_and_index,
    get_style_repr_html,
    initialize_estimator,
    initialize_window_features,
    transform_numpy,
)
from spotforecast2_safe.preprocessing import QuantileBinner, TimeSeriesDifferentiator
from spotforecast2_safe.utils import (
    check_exog,
    check_exog_dtypes,
    check_interval,
    check_predict_input,
    check_residuals_input,
    check_select_fit_kwargs,
    check_y,
    date_to_index_position,
    expand_index,
    get_exog_dtypes,
    initialize_lags,
    initialize_weights,
    input_to_frame,
    set_cpu_gpu_device,
    transform_dataframe,
)

# from spotforecast2_safe import __version__  # Removed to avoid circular import


class ForecasterRecursive(ForecasterBase):
    """
    Recursive autoregressive forecaster for scikit-learn compatible estimators.

    This class turns any estimator compatible with the scikit-learn API into a
    recursive autoregressive (multi-step) forecaster. The forecaster learns to predict
    future values by using lagged values of the target variable and optional exogenous
    features. Predictions are made iteratively, where each step uses previous predictions
    as input for the next step (recursive strategy).

    Args:
        estimator: Scikit-learn compatible estimator for regression. If None, a default
            estimator will be initialized. Can also be passed via regressor parameter.
        lags: Lagged values of the target variable to use as predictors. Can be an
            integer (uses lags from 1 to lags), list of integers, numpy array, or range.
            At least one of lags or window_features must be provided. Defaults to None.
        window_features: List of window feature objects to compute features from the
            target variable. Each object must implement transform_batch() method.
            At least one of lags or window_features must be provided. Defaults to None.
        transformer_y: Transformer object for the target variable. Must implement fit()
            and transform() methods. Applied before training and predictions.
            Defaults to None.
        transformer_exog: Transformer object for exogenous variables. Must implement
            fit() and transform() methods. Applied before training and predictions.
            Defaults to None.
        weight_func: Function to compute sample weights for training. Must accept an
            index and return an array of weights. Defaults to None.
        differentiation: Order of differencing to apply to the target variable.
            Must be a positive integer. Differencing is applied before creating lags.
            Defaults to None.
        fit_kwargs: Dictionary of additional keyword arguments to pass to the estimator's
            fit() method. Defaults to None.
        binner_kwargs: Dictionary of keyword arguments for QuantileBinner used in
            probabilistic predictions. Defaults to {'n_bins': 10, 'method': 'linear'}.
        forecaster_id: Identifier for the forecaster instance. Can be a string or
            integer. Used for tracking and logging purposes. Defaults to None.
        regressor: Alternative parameter name for estimator. If provided, used instead
            of estimator. Defaults to None.

    Attributes:
        estimator: Fitted scikit-learn estimator.
        lags: Lag indices used in the model.
        lags_names: Names of lag features (e.g., ['lag_1', 'lag_2']).
        window_features: List of window feature transformers.
        window_features_names: Names of window features.
        window_size: Maximum window size needed (max of lags and window features).
        transformer_y: Transformer for target variable.
        transformer_exog: Transformer for exogenous variables.
        weight_func: Function for sample weighting.
        differentiation: Order of differencing applied.
        differentiator: TimeSeriesDifferentiator instance if differencing is used.
        is_fitted: Boolean indicating if forecaster has been fitted.
        fit_date: Timestamp of the last fit operation.
        last_window_: Last window_size observations from training data.
        index_type_: Type of index in training data (RangeIndex or DatetimeIndex).
        index_freq_: Frequency of DatetimeIndex if applicable.
        training_range_: First and last index values of training data.
        series_name_in_: Name of the target series.
        exog_in_: Boolean indicating if exogenous variables were used in training.
        exog_names_in_: Names of exogenous variables.
        exog_type_in_: Type of exogenous input (Series or DataFrame).
        X_train_features_names_out_: Names of all training features.
        in_sample_residuals_: Residuals from training set.
        in_sample_residuals_by_bin_: Residuals grouped by bins for probabilistic pred.
        forecaster_id: Identifier for the forecaster instance.

    Note:
        - Either lags or window_features (or both) must be provided during initialization.
        - The forecaster uses a recursive strategy where each multi-step prediction
          depends on previous predictions within the same forecast horizon.
        - Exogenous variables must have the same index as the target variable and must
          be available for the entire prediction horizon.
        - The forecaster supports point predictions, prediction intervals, bootstrapping,
          quantile predictions, and probabilistic forecasts via conformal methods.

    Examples:
        Create a basic forecaster with lags:

        >>> import numpy as np
        >>> import pandas as pd
        >>> from sklearn.linear_model import LinearRegression
        >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        >>> y = pd.Series(np.random.randn(100), name='y')
        >>> forecaster = ForecasterRecursive(
        ...     estimator=LinearRegression(),
        ...     lags=10
        ... )
        >>> forecaster.fit(y)
        >>> predictions = forecaster.predict(steps=5)

        Create a forecaster with window features and transformations:

        >>> from sklearn.ensemble import RandomForestRegressor
        >>> from sklearn.preprocessing import StandardScaler
        >>> from spotforecast2_safe.preprocessing import RollingFeatures
        >>> import pandas as pd
        >>> y = pd.Series(np.random.randn(100), name='y')
        >>> forecaster = ForecasterRecursive(
        ...     estimator=RandomForestRegressor(n_estimators=100),
        ...     lags=[1, 7, 30],
        ...     window_features=[RollingFeatures(stats='mean', window_sizes=7)],
        ...     transformer_y=StandardScaler(),
        ...     differentiation=1
        ... )
        >>> forecaster.fit(y)
        >>> predictions = forecaster.predict(steps=10)

        Create a forecaster with exogenous variables:

        >>> import pandas as pd
        >>> from sklearn.linear_model import Ridge
        >>> y = pd.Series(np.random.randn(100), name='target')
        >>> exog = pd.DataFrame({'temp': np.random.randn(100)}, index=y.index)
        >>> forecaster = ForecasterRecursive(
        ...     estimator=Ridge(),
        ...     lags=7,
        ...     forecaster_id='my_forecaster'
        ... )
        >>> forecaster.fit(y, exog)
        >>> exog_future = pd.DataFrame(
        ...     {'temp': np.random.randn(5)},
        ...     index=pd.RangeIndex(start=100, stop=105)
        ... )
        >>> predictions = forecaster.predict(steps=5, exog=exog_future)

        Create a forecaster with probabilistic prediction configuration:

        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> import pandas as pd
        >>> y = pd.Series(np.random.randn(100), name='y')
        >>> forecaster = ForecasterRecursive(
        ...     estimator=GradientBoostingRegressor(),
        ...     lags=14,
        ...     binner_kwargs={'n_bins': 15, 'method': 'linear'}
        ... )
        >>> forecaster.fit(y, store_in_sample_residuals=True)
        >>> predictions = forecaster.predict(steps=5)
    """

    def __init__(
        self,
        estimator: object = None,
        lags: Union[int, List[int], np.ndarray, range, None] = None,
        window_features: Union[object, List[object], None] = None,
        transformer_y: Optional[object] = None,
        transformer_exog: Optional[object] = None,
        weight_func: Optional[Callable] = None,
        differentiation: Optional[int] = None,
        fit_kwargs: Optional[Dict[str, object]] = None,
        binner_kwargs: Optional[Dict[str, object]] = None,
        forecaster_id: Union[str, int, None] = None,
        regressor: object = None,
    ) -> None:

        self.estimator = copy(initialize_estimator(estimator, regressor))
        self.transformer_y = transformer_y
        self.transformer_exog = transformer_exog
        self.weight_func = weight_func
        self.source_code_weight_func = None
        self.differentiation = differentiation
        self.differentiation_max = None
        self.differentiator = None
        self.last_window_ = None
        self.index_type_ = None
        self.index_freq_ = None
        self.training_range_ = None
        self.series_name_in_ = None
        self.exog_in_ = False
        self.exog_names_in_ = None
        self.exog_type_in_ = None
        self.exog_dtypes_in_ = None
        self.exog_dtypes_out_ = None
        self.X_train_window_features_names_out_ = None
        self.X_train_exog_names_out_ = None
        self.X_train_features_names_out_ = None
        self.in_sample_residuals_ = None
        self.out_sample_residuals_ = None
        self.in_sample_residuals_by_bin_ = None
        self.out_sample_residuals_by_bin_ = None
        self.creation_date = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M:%S")
        self.is_fitted = False
        self.fit_date = None
        try:
            from spotforecast2_safe import __version__

            self.spotforecast_version = __version__
        except ImportError:
            self.spotforecast_version = "unknown"
        self.python_version = sys.version.split(" ")[0]
        self.forecaster_id = forecaster_id
        self._probabilistic_mode = "binned"

        (
            self.lags,
            self.lags_names,
            self.max_lag,
        ) = initialize_lags(type(self).__name__, lags)
        (
            self.window_features,
            self.window_features_names,
            self.max_size_window_features,
        ) = initialize_window_features(window_features)
        if self.window_features is None and self.lags is None:
            raise ValueError(
                "At least one of the arguments `lags` or `window_features` "
                "must be different from None. This is required to create the "
                "predictors used in training the forecaster."
            )

        self.window_size = max(
            [
                ws
                for ws in [self.max_lag, self.max_size_window_features]
                if ws is not None
            ]
        )
        self.window_features_class_names = None
        if window_features is not None:
            self.window_features_class_names = [
                type(wf).__name__ for wf in self.window_features
            ]

        self.weight_func, self.source_code_weight_func, _ = initialize_weights(
            forecaster_name=type(self).__name__,
            estimator=estimator,
            weight_func=weight_func,
            series_weights=None,
        )

        if differentiation is not None:
            if not isinstance(differentiation, int) or differentiation < 1:
                raise ValueError(
                    f"Argument `differentiation` must be an integer equal to or "
                    f"greater than 1. Got {differentiation}."
                )
            self.differentiation = differentiation
            self.differentiation_max = differentiation
            self.window_size += differentiation
            self.differentiator = TimeSeriesDifferentiator(
                order=differentiation  # , window_size=self.window_size # TODO: TimeSeriesDifferentiator in preprocessing created only takes order, add window_size if needed
            )

        self.fit_kwargs = check_select_fit_kwargs(
            estimator=estimator, fit_kwargs=fit_kwargs
        )

        self.binner_kwargs = binner_kwargs
        if binner_kwargs is None:
            self.binner_kwargs = {
                "n_bins": 10,
                "method": "linear",
            }
        self.binner = QuantileBinner(**self.binner_kwargs)
        self.binner_intervals_ = None

        self.__spotforecast_tags__ = {
            "library": "spotforecast",
            "forecaster_name": "ForecasterRecursive",
            "forecaster_task": "regression",
            "forecasting_scope": "single-series",  # single-series | global
            "forecasting_strategy": "recursive",  # recursive | direct | deep_learning
            "index_types_supported": ["pandas.RangeIndex", "pandas.DatetimeIndex"],
            "requires_index_frequency": True,
            "allowed_input_types_series": ["pandas.Series"],
            "supports_exog": True,
            "allowed_input_types_exog": ["pandas.Series", "pandas.DataFrame"],
            "handles_missing_values_series": False,
            "handles_missing_values_exog": True,
            "supports_lags": True,
            "supports_window_features": True,
            "supports_transformer_series": True,
            "supports_transformer_exog": True,
            "supports_weight_func": True,
            "supports_differentiation": True,
            "prediction_types": [
                "point",
                "interval",
                "bootstrapping",
                "quantiles",
                "distribution",
            ],
            "supports_probabilistic": True,
            "probabilistic_methods": ["bootstrapping", "conformal"],
            "handles_binned_residuals": True,
        }

    def __repr__(self) -> str:
        """
        Information displayed when a ForecasterRecursive object is printed.

        Returns:
            str: String representation of the forecaster with key information about its configuration and state.

        Examples:
            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
            >>> print(forecaster)  # doctest: +ELLIPSIS
            =========================
            ForecasterRecursive
            =========================
            Estimator: LinearRegression
            Lags: [1, 2, 3]
            Window features: []
            Window size: 3
            Series name: None
            Exogenous included: False
            Exogenous names: None
            Transformer for y: None
            Transformer for exog: None
            Weight function included: False
            Differentiation order: None
            Training range: None
            Training index type: None
            Training index frequency: None
            Estimator parameters: {...}
            fit_kwargs: {...}
            Creation date: ...
            Last fit date: None
            spotforecast version: ...
            Python version: ...
            Forecaster id: None

        """

        params = (
            self.estimator.get_params() if hasattr(self.estimator, "get_params") else {}
        )
        exog_names_in_ = self.exog_names_in_ if self.exog_in_ else None

        info = (
            f"{'=' * len(type(self).__name__)} \n"
            f"{type(self).__name__} \n"
            f"{'=' * len(type(self).__name__)} \n"
            f"Estimator: {type(self.estimator).__name__} \n"
            f"Lags: {self.lags} \n"
            f"Window features: {self.window_features_names} \n"
            f"Window size: {self.window_size} \n"
            f"Series name: {self.series_name_in_} \n"
            f"Exogenous included: {self.exog_in_} \n"
            f"Exogenous names: {exog_names_in_} \n"
            f"Transformer for y: {self.transformer_y} \n"
            f"Transformer for exog: {self.transformer_exog} \n"
            f"Weight function included: {True if self.weight_func is not None else False} \n"
            f"Differentiation order: {self.differentiation} \n"
            f"Training range: {self.training_range_.to_list() if self.is_fitted else None} \n"
            f"Training index type: {str(self.index_type_).split('.')[-1][:-2] if self.is_fitted else None} \n"
            f"Training index frequency: {self.index_freq_ if self.is_fitted else None} \n"
            f"Estimator parameters: {params} \n"
            f"fit_kwargs: {self.fit_kwargs} \n"
            f"Creation date: {self.creation_date} \n"
            f"Last fit date: {self.fit_date} \n"
            f"spotforecast version: {self.spotforecast_version} \n"
            f"Python version: {self.python_version} \n"
            f"Forecaster id: {self.forecaster_id} \n"
        )

        return info

    def _repr_html_(self) -> str:
        """
        HTML representation of the object.
        The "General Information" section is expanded by default.

        Returns:
            HTML string representation of the forecaster.

        Examples:
            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
            >>> forecaster._repr_html_()  # doctest: +ELLIPSIS
            '<div class="container-...">...</div>'
        """

        params = (
            self.estimator.get_params() if hasattr(self.estimator, "get_params") else {}
        )
        exog_names_in_ = self.exog_names_in_ if self.exog_in_ else None

        style, unique_id = get_style_repr_html(self.is_fitted)

        content = f"""
        <div class="container-{unique_id}">
            <p style="font-size: 1.5em; font-weight: bold; margin-block-start: 0.83em; margin-block-end: 0.83em;">{type(self).__name__}</p>
            <details open>
                <summary>General Information</summary>
                <ul>
                    <li><strong>Estimator:</strong> {type(self.estimator).__name__}</li>
                    <li><strong>Lags:</strong> {self.lags}</li>
                    <li><strong>Window features:</strong> {self.window_features_names}</li>
                    <li><strong>Window size:</strong> {self.window_size}</li>
                    <li><strong>Series name:</strong> {self.series_name_in_}</li>
                    <li><strong>Exogenous included:</strong> {self.exog_in_}</li>
                    <li><strong>Weight function included:</strong> {self.weight_func is not None}</li>
                    <li><strong>Differentiation order:</strong> {self.differentiation}</li>
                    <li><strong>Creation date:</strong> {self.creation_date}</li>
                    <li><strong>Last fit date:</strong> {self.fit_date}</li>
                    <li><strong>spotforecast version:</strong> {self.spotforecast_version}</li>
                    <li><strong>Python version:</strong> {self.python_version}</li>
                    <li><strong>Forecaster id:</strong> {self.forecaster_id}</li>
                </ul>
            </details>
            <details>
                <summary>Exogenous Variables</summary>
                <ul>
                    {exog_names_in_}
                </ul>
            </details>
            <details>
                <summary>Data Transformations</summary>
                <ul>
                    <li><strong>Transformer for y:</strong> {self.transformer_y}</li>
                    <li><strong>Transformer for exog:</strong> {self.transformer_exog}</li>
                </ul>
            </details>
            <details>
                <summary>Training Information</summary>
                <ul>
                    <li><strong>Training range:</strong> {self.training_range_.to_list() if self.is_fitted else 'Not fitted'}</li>
                    <li><strong>Training index type:</strong> {str(self.index_type_).split('.')[-1][:-2] if self.is_fitted else 'Not fitted'}</li>
                    <li><strong>Training index frequency:</strong> {self.index_freq_ if self.is_fitted else 'Not fitted'}</li>
                </ul>
            </details>
            <details>
                <summary>Estimator Parameters</summary>
                <ul>
                    {params}
                </ul>
            </details>
            <details>
                <summary>Fit Kwargs</summary>
                <ul>
                    {self.fit_kwargs}
                </ul>
            </details>
        </div>
        """

        return style + content

    def __setstate__(self, state: dict) -> None:
        """
        Custom __setstate__ to ensure backward compatibility when unpickling.
        Only sets __spotforecast_tags__ if not present, preserving custom tags.
        """
        super().__setstate__(state)
        if not hasattr(self, "__spotforecast_tags__"):
            self.__spotforecast_tags__ = {
                "library": "spotforecast",
                "forecaster_name": "ForecasterRecursive",
                "forecaster_task": "regression",
                "forecasting_scope": "single-series",
                "forecasting_strategy": "recursive",
                "index_types_supported": ["pandas.RangeIndex", "pandas.DatetimeIndex"],
                "requires_index_frequency": True,
                "allowed_input_types_series": ["pandas.Series"],
                "supports_exog": True,
                "allowed_input_types_exog": ["pandas.Series", "pandas.DataFrame"],
                "handles_missing_values_series": False,
                "handles_missing_values_exog": True,
                "supports_lags": True,
                "supports_window_features": True,
                "supports_transformer_series": True,
                "supports_transformer_exog": True,
                "supports_weight_func": True,
                "supports_differentiation": True,
                "prediction_types": [
                    "point",
                    "interval",
                    "bootstrapping",
                    "quantiles",
                    "distribution",
                ],
                "supports_probabilistic": True,
                "probabilistic_methods": ["bootstrapping", "conformal"],
                "handles_binned_residuals": True,
            }

    def _create_lags(
        self,
        y: np.ndarray,
        X_as_pandas: bool = False,
        train_index: Optional[pd.Index] = None,
    ) -> Tuple[Optional[Union[np.ndarray, pd.DataFrame]], np.ndarray]:
        """
        Create lagged predictors and aligned target values.

        Args:
            y: Target values used to build lag features. Expected shape is
                (n_samples,) or (n_samples, 1).
            X_as_pandas: If True, returns lagged features as a pandas DataFrame.
            train_index: Index to use for the lagged feature DataFrame when
                `X_as_pandas` is True.

        Returns:
            Tuple containing:
                - X_data: Lagged predictors with shape (n_rows, n_lags) or None
                  if no lags are configured.
                - y_data: Target values aligned to the lagged predictors with
                  shape (n_rows,).

        Raises:
            ValueError: If `X_as_pandas` is True but `train_index` is not provided.
            ValueError: If the length of `y` is not sufficient to create the
                specified lags.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> forecaster = ForecasterRecursive(lags=3)
            >>> y = np.arange(10)
            >>> train_index = pd.RangeIndex(start=3, stop=10)
            >>> X_data, y_data = forecaster._create_lags(y=y, X_as_pandas=True, train_index=train_index)
            >>> isinstance(X_data, pd.DataFrame)
            True
            >>> X_data.shape
            (7, 3)
            >>> y_data.shape
            (7,)
        """
        if X_as_pandas and train_index is None:
            raise ValueError(
                "If `X_as_pandas` is True, `train_index` must be provided."
            )

        if len(y) <= self.window_size:
            raise ValueError(
                f"Length of `y` must be greater than the maximum window size "
                f"needed by the forecaster.\n"
                f"    Length `y`: {len(y)}.\n"
                f"    Max window size: {self.window_size}."
            )

        X_data = None
        if self.lags is not None:
            # y = y.ravel() # Assuming y is already raveled
            # Using stride_tricks for sliding window
            y_strided = np.lib.stride_tricks.sliding_window_view(y, self.window_size)[
                :-1
            ]
            X_data = y_strided[:, self.window_size - self.lags]

            if X_as_pandas:
                X_data = pd.DataFrame(
                    data=X_data, columns=self.lags_names, index=train_index
                )

        y_data = y[self.window_size :]

        return X_data, y_data

    def _create_window_features(
        self,
        y: pd.Series,
        train_index: pd.Index,
        X_as_pandas: bool = False,
    ) -> Tuple[List[Union[np.ndarray, pd.DataFrame]], List[str]]:
        """
        Generate window features from the target series.

        Args:
            y: Target series used to compute window features. Must be a pandas
                Series with an index aligned to `train_index` after trimming.
            train_index: Index for the training rows to align the window features.
            X_as_pandas: If True, keeps each window feature matrix as a pandas
                DataFrame; otherwise converts to NumPy arrays.

        Returns:
            Tuple containing:
                - X_train_window_features: List of window feature matrices, one
                  per window feature transformer.
                - X_train_window_features_names_out_: List of feature names for
                  all generated window features.

        Raises:
            TypeError: If any window feature's `transform_batch` method does not
                return a pandas DataFrame.
            ValueError: If the output DataFrame from any window feature does not
                have the same number of rows as `train_index` or if the index
                does not match `train_index`.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from spotforecast2_safe.preprocessing import RollingFeatures
            >>> y = pd.Series(np.arange(30), name='y')
            >>> forecaster = ForecasterRecursive(
            ...     estimator=LinearRegression(),
            ...     window_features=[RollingFeatures(stats='mean', window_sizes=3)]
            ... )
            >>> train_index = y.index[3:]  # Assuming window_size is 3
            >>> X_train_window_features, feature_names = forecaster._create_window_features(
            ...     y=y,
            ...     train_index=train_index,
            ...     X_as_pandas=True
            ... )
            >>> isinstance(X_train_window_features[0], pd.DataFrame)
            True
            >>> X_train_window_features[0].shape[0] == len(train_index)
            True
            >>> (X_train_window_features[0].index == train_index).all()
            True

        """

        len_train_index = len(train_index)
        X_train_window_features = []
        X_train_window_features_names_out_ = []
        for wf in self.window_features:
            X_train_wf = wf.transform_batch(y)
            if not isinstance(X_train_wf, pd.DataFrame):
                raise TypeError(
                    f"The method `transform_batch` of {type(wf).__name__} "
                    f"must return a pandas DataFrame."
                )
            X_train_wf = X_train_wf.iloc[-len_train_index:]
            if not len(X_train_wf) == len_train_index:
                raise ValueError(
                    f"The method `transform_batch` of {type(wf).__name__} "
                    f"must return a DataFrame with the same number of rows as "
                    f"the input time series - `window_size`: {len_train_index}."
                )
            if not (X_train_wf.index == train_index).all():
                raise ValueError(
                    f"The method `transform_batch` of {type(wf).__name__} "
                    f"must return a DataFrame with the same index as "
                    f"the input time series - `window_size`."
                )

            X_train_window_features_names_out_.extend(X_train_wf.columns)
            if not X_as_pandas:
                X_train_wf = X_train_wf.to_numpy()
            X_train_window_features.append(X_train_wf)

        return X_train_window_features, X_train_window_features_names_out_

    def _create_train_X_y(
        self, y: pd.Series, exog: Union[pd.Series, pd.DataFrame, None] = None
    ) -> Tuple[
        pd.DataFrame,
        pd.Series,
        List[str],
        List[str],
        List[str],
        List[str],
        Dict[str, type],
        Dict[str, type],
    ]:
        """Create training predictors and target values.

        Args:
            y: Target series for training. Must be a pandas Series.
            exog:
                Optional exogenous variables for training. Can be a pandas Series or DataFrame.
                Must have the same index as `y` and cover the same time range.

        Returns:
            Tuple containing:
                - X_train: DataFrame of training predictors including lags, window features, and exogenous variables (if provided).
                - y_train: Series of target values aligned with the predictors.
                - X_train_features_names_out_: List of all predictor feature names.
                - lags_names: List of lag feature names.
                - window_features_names: List of window feature names.
                - exog_names_in_: List of exogenous variable names (if exogenous variables are used).
                - exog_dtypes_in_: Dictionary of input data types for exogenous variables.
                - exog_dtypes_out_: Dictionary of output data types for exogenous variables after transformation (if exogenous variables are used).

        Raises:
            ValueError: If the length of `y` is not sufficient to create the specified lags and window features.
            ValueError: If `exog` is provided but does not have the same index as `y` or does not cover the same time range.
            ValueError: If `exog` is provided but contains data types that are not supported after transformation.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from spotforecast2_safe.preprocessing import RollingFeatures
            >>> y = pd.Series(np.arange(30), name='y')
            >>> exog = pd.DataFrame({'temp': np.random.randn(30)}, index=y.index)
            >>> forecaster = ForecasterRecursive(
            ...     estimator=LinearRegression(),
            ...     lags=3,
            ...     window_features=[RollingFeatures(stats='mean', window_sizes=3)]
            ... )
            >>> (X_train, y_train, exog_names_in_, window_features_names,
            ...  exog_names_out, feature_names, exog_dtypes_in_,
            ...  exog_dtypes_out_) = forecaster._create_train_X_y(y=y, exog=exog)
            >>> isinstance(X_train, pd.DataFrame)
            True
            >>> isinstance(y_train, pd.Series)
            True
            >>> feature_names == forecaster.lags_names + window_features_names + exog_names_out
            True
        """
        check_y(y=y)
        y = input_to_frame(data=y, input_name="y")

        if len(y) <= self.window_size:
            raise ValueError(
                f"Length of `y` must be greater than the maximum window size "
                f"needed by the forecaster.\n"
                f"    Length `y`: {len(y)}.\n"
                f"    Max window size: {self.window_size}.\n"
                f"    Lags window size: {self.max_lag}.\n"
                f"    Window features window size: {self.max_size_window_features}."
            )

        fit_transformer = False if self.is_fitted else True
        y = transform_dataframe(
            df=y,
            transformer=self.transformer_y,
            fit=fit_transformer,
            inverse_transform=False,
        )
        y_values, y_index = check_extract_values_and_index(data=y, data_label="`y`")
        if y_values.ndim == 2 and y_values.shape[1] == 1:
            y_values = y_values.ravel()
        train_index = y_index[self.window_size :]

        if self.differentiation is not None:
            if not self.is_fitted:
                y_values = self.differentiator.fit_transform(y_values)
            else:
                differentiator = copy(self.differentiator)
                y_values = differentiator.fit_transform(y_values)

        exog_names_in_ = None
        exog_dtypes_in_ = None
        exog_dtypes_out_ = None
        X_as_pandas = False
        if exog is not None:
            check_exog(exog=exog, allow_nan=True)
            exog = input_to_frame(data=exog, input_name="exog")
            _, exog_index = check_extract_values_and_index(
                data=exog, data_label="`exog`", ignore_freq=True, return_values=False
            )

            len_y_original = len(y)
            len_train = len(train_index)
            len_exog = len(exog)

            if not len_exog == len_y_original and not len_exog == len_train:
                raise ValueError(
                    f"Length mismatch for exogenous variables. Expected either:\n"
                    f"  - Full length matching `y`: {len_y_original} observations, OR\n"
                    f"  - Pre-aligned length: {len_train} observations (y length - window_size)\n"
                    f"Got: {len_exog} observations.\n"
                    f"Window size: {self.window_size}"
                )

            if len_exog == len_y_original:
                if not (exog_index == y_index).all():
                    raise ValueError(
                        "When `exog` has the same length as `y`, the index of "
                        "`exog` must be aligned with the index of `y` "
                        "to ensure the correct alignment of values."
                    )
                # Standard case: exog covers full y range, trim by window_size
                exog = exog.iloc[self.window_size :, :]
            else:
                if not (exog_index == train_index).all():
                    raise ValueError(
                        "When `exog` already starts after the first `window_size` "
                        "observations, its index must be aligned with the index "
                        "of `y` starting from `window_size`."
                    )

            exog_names_in_ = exog.columns.to_list()
            exog_dtypes_in_ = get_exog_dtypes(exog=exog)

            exog = transform_dataframe(
                df=exog,
                transformer=self.transformer_exog,
                fit=fit_transformer,
                inverse_transform=False,
            )

            check_exog_dtypes(exog, call_check_exog=True)
            exog_dtypes_out_ = get_exog_dtypes(exog=exog)
            X_as_pandas = any(
                not pd.api.types.is_numeric_dtype(dtype)
                or pd.api.types.is_bool_dtype(dtype)
                for dtype in set(exog.dtypes)
            )

        X_train = []
        X_train_features_names_out_ = []

        # Create lags
        # Note: y_values might have NaNs from differentiation.
        # TODO: check if _create_lags handles this!
        X_train_lags, y_train = self._create_lags(
            y=y_values, X_as_pandas=X_as_pandas, train_index=train_index
        )
        if X_train_lags is not None:
            X_train.append(X_train_lags)
            X_train_features_names_out_.extend(self.lags_names)

        X_train_window_features_names_out_ = None
        if self.window_features is not None:
            n_diff = 0 if self.differentiation is None else self.differentiation
            if isinstance(y_values, pd.Series):
                y_vals_for_wf = y_values.iloc[n_diff:]
                y_index_for_wf = y_index[n_diff:]
            else:
                y_vals_for_wf = y_values[n_diff:]
                y_index_for_wf = y_index[n_diff:]

            y_window_features = pd.Series(y_vals_for_wf, index=y_index_for_wf)
            X_train_window_features, X_train_window_features_names_out_ = (
                self._create_window_features(
                    y=y_window_features,
                    X_as_pandas=X_as_pandas,
                    train_index=train_index,
                )
            )
            X_train.extend(X_train_window_features)
            X_train_features_names_out_.extend(X_train_window_features_names_out_)

        X_train_exog_names_out_ = None
        if exog is not None:
            X_train_exog_names_out_ = exog.columns.to_list()
            if not X_as_pandas:
                exog = exog.to_numpy()
            X_train_features_names_out_.extend(X_train_exog_names_out_)
            X_train.append(exog)

        if len(X_train) == 1:
            X_train = X_train[0]
        else:
            if X_as_pandas:
                X_train = pd.concat(X_train, axis=1)
            else:
                X_train = np.concatenate(X_train, axis=1)

        if X_as_pandas:
            X_train.index = train_index
        else:
            X_train = pd.DataFrame(
                data=X_train, index=train_index, columns=X_train_features_names_out_
            )

        y_train = pd.Series(data=y_train, index=train_index, name="y")

        return (
            X_train,
            y_train,
            exog_names_in_,
            X_train_window_features_names_out_,
            X_train_exog_names_out_,
            X_train_features_names_out_,
            exog_dtypes_in_,
            exog_dtypes_out_,
        )

    def create_train_X_y(
        self, y: pd.Series, exog: Union[pd.Series, pd.DataFrame, None] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Public method to create training predictors and target values.

        This method is a public wrapper around the internal method `_create_train_X_y`,
        which generates the training predictors and target values based on the provided time series and exogenous variables.
        It ensures that the necessary transformations and feature engineering steps are applied to prepare the data for training the forecaster.

        Args:
            y: Target series for training. Must be a pandas Series.
            exog: Optional exogenous variables for training. Can be a pandas Series or DataFrame. Must have the same index as `y` and cover the same time range. Defaults to None.

        Returns:
            Tuple containing:
                - X_train: DataFrame of training predictors including lags, window features, and exogenous variables (if provided).
                - y_train: Series of target values aligned with the predictors.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from spotforecast2_safe.preprocessing import RollingFeatures
            >>> y = pd.Series(np.arange(30), name='y')
            >>> exog = pd.DataFrame({'temp': np.random.randn(30)}, index=y.index)
            >>> forecaster = ForecasterRecursive(
            ...     estimator=LinearRegression(),
            ...     lags=3,
            ...     window_features=[RollingFeatures(stats='mean', window_sizes=3)]
            ... )
            >>> X_train, y_train = forecaster.create_train_X_y(y=y, exog=exog)
            >>> isinstance(X_train, pd.DataFrame)
            True
            >>> isinstance(y_train, pd.Series)
            True

        """
        output = self._create_train_X_y(y=y, exog=exog)

        return output[0], output[1]

    def _train_test_split_one_step_ahead(
        self,
        y: pd.Series,
        initial_train_size: int,
        exog: Union[pd.Series, pd.DataFrame, None] = None,
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Create matrices needed to train and test the forecaster for one-step-ahead
        predictions.

        Args:
            y: Training time series.
            initial_train_size: Initial size of the training set. It is the number of
                observations used to train the forecaster before making the first
                prediction.
            exog: Exogenous variable/s included as predictor/s. Must have the same
                number of observations as y and their indexes must be aligned.
                Defaults to None.

        Returns:
            Tuple containing:
                - X_train: Predictor values used to train the model as pandas DataFrame.
                - y_train: Target values related to each row of X_train as pandas Series.
                - X_test: Predictor values used to test the model as pandas DataFrame.
                - y_test: Target values related to each row of X_test as pandas Series.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from spotforecast2_safe.preprocessing import RollingFeatures
            >>> y = pd.Series(np.arange(30), name='y')
            >>> exog = pd.DataFrame({'temp': np.random.randn(30)}, index=y.index)
            >>> forecaster = ForecasterRecursive(
            ...     estimator=LinearRegression(),
            ...     lags=3,
            ...     window_features=[RollingFeatures(stats='mean', window_sizes=3)]
            ... )
            >>> X_train, y_train, X_test, y_test = forecaster._train_test_split_one_step_ahead(y=y, initial_train_size=20, exog=exog)
            >>> isinstance(X_train, pd.DataFrame)
            True
            >>> isinstance(y_train, pd.Series)
            True
            >>> isinstance(X_test, pd.DataFrame)
            True
            >>> isinstance(y_test, pd.Series)
            True
        """

        is_fitted = self.is_fitted
        self.is_fitted = False
        X_train, y_train, *_ = self._create_train_X_y(
            y=y.iloc[:initial_train_size],
            exog=exog.iloc[:initial_train_size] if exog is not None else None,
        )

        test_init = initial_train_size - self.window_size
        self.is_fitted = True
        X_test, y_test, *_ = self._create_train_X_y(
            y=y.iloc[test_init:],
            exog=exog.iloc[test_init:] if exog is not None else None,
        )

        self.is_fitted = is_fitted

        return X_train, y_train, X_test, y_test

    def get_params(self, deep: bool = True) -> Dict[str, object]:
        """
        Get parameters for this forecaster.

        Args:
            deep: If True, will return the parameters for this forecaster and
                contained sub-objects that are estimators.

        Returns:
            params: Dictionary of parameter names mapped to their values.

        Examples:
            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
            >>> forecaster.get_params()  # doctest: +ELLIPSIS
            {
                'estimator': LinearRegression(), 'lags': 3, 'window_features': None,
                'transformer_y': None, 'transformer_exog': None, 'weight_func': None,
                'differentiation': None, 'fit_kwargs': {}, 'binner_kwargs': None, 'forecaster_id': '...'}
        """
        params = {}
        for key in [
            "estimator",
            "lags",
            "window_features",
            "transformer_y",
            "transformer_exog",
            "weight_func",
            "differentiation",
            "fit_kwargs",
            "binner_kwargs",
            "forecaster_id",
        ]:
            if hasattr(self, key):
                params[key] = getattr(self, key)

        if not deep:
            return params

        if hasattr(self, "estimator") and self.estimator is not None:
            if hasattr(self.estimator, "get_params"):
                for key, value in self.estimator.get_params(deep=True).items():
                    params[f"estimator__{key}"] = value

        return params

    def set_params(
        self, params: Dict[str, object] = None, **kwargs: object
    ) -> "ForecasterRecursive":
        """
        Set the parameters of this forecaster.

        Args:
            params: Optional dictionary of parameter names mapped to their new values.
                If provided, these parameters are set first.
            **kwargs: Dictionary of parameter names mapped to their new values.
                Parameters can be for the forecaster itself or for the contained estimator (using the `estimator__` prefix).

        Returns:
            self: The forecaster instance with updated parameters.

        Examples:
            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
            >>> forecaster.set_params(estimator__fit_intercept=False)
            >>> forecaster.estimator.get_params()["fit_intercept"]
            False
        """

        # Merge params dict and kwargs
        all_params = {}
        if params is not None:
            all_params.update(params)
        all_params.update(kwargs)

        if not all_params:
            return self

        valid_params = self.get_params(deep=True)
        nested_params = {}

        for key, value in all_params.items():
            if key not in valid_params and "__" not in key:
                # Relaxed check for now
                pass

            if "__" in key:
                obj_name, param_name = key.split("__", 1)
                if obj_name not in nested_params:
                    nested_params[obj_name] = {}
                nested_params[obj_name][param_name] = value
            else:
                setattr(self, key, value)

        for obj_name, obj_params in nested_params.items():
            if hasattr(self, obj_name):
                obj = getattr(self, obj_name)
                if hasattr(obj, "set_params"):
                    obj.set_params(**obj_params)
                else:
                    for param_name, value in obj_params.items():
                        setattr(obj, param_name, value)

        return self

    def fit(
        self,
        y: pd.Series,
        exog: Union[pd.Series, pd.DataFrame, None] = None,
        store_last_window: bool = True,
        store_in_sample_residuals: bool = False,
        random_state: int = 123,
        suppress_warnings: bool = False,
    ) -> None:
        """
        Fit the forecaster to the training data.

        Args:
            y:
                  Target series for training. Must be a pandas Series.
            exog:
                  Optional exogenous variables for training. Can be a pandas Series or DataFrame.Must have the same index as `y` and cover the same time range. Defaults to None.
            store_last_window:
                  Whether to store the last window of the training series for use in prediction. Defaults to True.
            store_in_sample_residuals:
                  Whether to store in-sample residuals after fitting, which can be used for certain probabilistic prediction methods. Defaults to False.
            random_state:
                  Random seed for reproducibility when sampling residuals if `store_in_sample_residuals` is True. Defaults to 123.
            suppress_warnings:
                  Whether to suppress warnings during fitting, such as those related to insufficient data length for lags or window features. Defaults to False.

        Returns:
            None

        Examples:
                 >>> import numpy as np
                 >>> import pandas as pd
                 >>> from sklearn.linear_model import LinearRegression
                 >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
                 >>> from spotforecast2_safe.preprocessing import RollingFeatures
                 >>> y = pd.Series(np.arange(30), name='y')
                 >>> exog = pd.DataFrame({'temp': np.random.randn(30)}, index=y.index)
                 >>> forecaster = ForecasterRecursive(
                 ...     estimator=LinearRegression(),
                 ...     lags=3,
                 ...     window_features=[RollingFeatures(stats='mean', window_sizes=3)]
                 ... )
                 >>> forecaster.fit(y=y, exog=exog, store_in_sample_residuals=True)
        """

        set_skforecast_warnings(suppress_warnings, action="ignore")

        # Reset values in case the forecaster has already been fitted.
        self.last_window_ = None
        self.index_type_ = None
        self.index_freq_ = None
        self.training_range_ = None
        self.series_name_in_ = None
        self.exog_in_ = False
        self.exog_names_in_ = None
        self.exog_type_in_ = None
        self.exog_dtypes_in_ = None
        self.exog_dtypes_out_ = None
        self.X_train_window_features_names_out_ = None
        self.X_train_exog_names_out_ = None
        self.X_train_features_names_out_ = None
        self.in_sample_residuals_ = None
        self.in_sample_residuals_by_bin_ = None
        self.binner_intervals_ = None
        self.is_fitted = False
        self.fit_date = None

        (
            X_train,
            y_train,
            exog_names_in_,
            X_train_window_features_names_out_,
            X_train_exog_names_out_,
            X_train_features_names_out_,
            exog_dtypes_in_,
            exog_dtypes_out_,
        ) = self._create_train_X_y(y=y, exog=exog)

        sample_weight = self.create_sample_weights(X_train=X_train)

        if sample_weight is not None:
            self.estimator.fit(
                X=X_train,
                y=y_train,
                sample_weight=sample_weight,
                **self.fit_kwargs,
            )
        else:
            self.estimator.fit(X=X_train, y=y_train, **self.fit_kwargs)

        self.X_train_window_features_names_out_ = X_train_window_features_names_out_
        self.X_train_features_names_out_ = X_train_features_names_out_

        self.is_fitted = True
        self.fit_date = pd.Timestamp.today().strftime("%Y-%m-%d %H:%M:%S")
        self.training_range_ = y.index[[0, -1]]
        self.index_type_ = type(y.index)
        if isinstance(y.index, pd.DatetimeIndex):
            self.index_freq_ = y.index.freqstr
        else:
            try:
                self.index_freq_ = y.index.step
            except AttributeError:
                self.index_freq_ = None

        if exog is not None:
            self.exog_in_ = True
            self.exog_type_in_ = type(exog)
            self.exog_names_in_ = exog_names_in_
            self.exog_dtypes_in_ = exog_dtypes_in_
            self.exog_dtypes_out_ = exog_dtypes_out_
            self.X_train_exog_names_out_ = X_train_exog_names_out_

        self.series_name_in_ = y.name if y.name is not None else "y"

        # NOTE: This is done to save time during fit in functions such as backtesting()
        if self._probabilistic_mode is not False:
            self._binning_in_sample_residuals(
                y_true=y_train.to_numpy(),
                y_pred=self.estimator.predict(X_train).ravel(),
                store_in_sample_residuals=store_in_sample_residuals,
                random_state=random_state,
            )

        if store_last_window:
            self.last_window_ = (
                y.iloc[-self.window_size :]
                .copy()
                .to_frame(name=y.name if y.name is not None else "y")
            )

        set_skforecast_warnings(suppress_warnings, action="default")

    def create_sample_weights(self, X_train: pd.DataFrame) -> np.ndarray:
        """
        Create weights for each observation according to the forecaster's attribute
        `weight_func`.

        Args:
            X_train: Dataframe created with the `create_train_X_y` method, first return.

        Returns:
            Weights to use in `fit` method.
        """

        sample_weight = None

        if self.weight_func is not None:
            sample_weight = self.weight_func(X_train.index)

        if sample_weight is not None:
            if np.isnan(sample_weight).any():
                raise ValueError(
                    "The resulting `sample_weight` cannot have NaN values."
                )
            if np.any(sample_weight < 0):
                raise ValueError(
                    "The resulting `sample_weight` cannot have negative values."
                )
            if np.sum(sample_weight) == 0:
                raise ValueError(
                    "The resulting `sample_weight` cannot be normalized because "
                    "the sum of the weights is zero."
                )

        return sample_weight

    def _create_predict_inputs(
        self,
        steps: int | str | pd.Timestamp,
        last_window: Union[pd.Series, pd.DataFrame, None] = None,
        exog: Union[pd.Series, pd.DataFrame, None] = None,
        predict_probabilistic: bool = False,
        use_in_sample_residuals: bool = True,
        use_binned_residuals: bool = True,
        check_inputs: bool = True,
    ) -> Tuple[np.ndarray, Union[np.ndarray, None], pd.Index, int]:
        """
        Create the inputs needed for the first iteration of the prediction
        process. As this is a recursive process, the last window is updated at
        each iteration of the prediction process.

        Args:
            steps: Number of steps to predict.
                - If steps is int, number of steps to predict.
                - If str or pandas Datetime, the prediction will be up to that date.
            last_window: Series values used to create the predictors (lags) needed in the
                first iteration of the prediction (t + 1).
                If `last_window = None`, the values stored in `self.last_window_` are
                used to calculate the initial predictors, and the predictions start
                right after training data.
            exog: Exogenous variable/s included as predictor/s.
            predict_probabilistic: If `True`, the necessary checks for probabilistic predictions will be
                performed.
            use_in_sample_residuals: If `True`, residuals from the training data are used as proxy of
                prediction error to create predictions.
                If `False`, out of sample residuals (calibration) are used.
                Out-of-sample residuals must be precomputed using Forecaster's
                `set_out_sample_residuals()` method.
            use_binned_residuals: If `True`, residuals are selected based on the predicted values
                (binned selection).
                If `False`, residuals are selected randomly.
            check_inputs: If `True`, the input is checked for possible warnings and errors
                with the `check_predict_input` function. This argument is created
                for internal use and is not recommended to be changed.

        Returns:
            - last_window_values:
                Numpy array of the last window values to use for prediction,
                transformed and ready for input into the prediction method.
            - exog_values:
                Numpy array of exogenous variable values for prediction,
                transformed and ready for input into the prediction method,
                or None if no exogenous variables are used.
            - prediction_index:
                Pandas Index for the predicted values, constructed based on the
                last window index and the number of steps to predict.
            - steps:
                Number of future steps predicted.
        """

        if last_window is None:
            last_window = self.last_window_

        if self.is_fitted:
            steps = date_to_index_position(
                index=last_window.index,
                date_input=steps,
                method="prediction",
                date_literal="steps",
            )

        if check_inputs:
            check_predict_input(
                forecaster_name=type(self).__name__,
                steps=steps,
                is_fitted=self.is_fitted,
                exog_in_=self.exog_in_,
                index_type_=self.index_type_,
                index_freq_=self.index_freq_,
                window_size=self.window_size,
                last_window=last_window,
                last_window_exog=None,
                exog=exog,
                exog_names_in_=self.exog_names_in_,
                interval=None,
            )

            if predict_probabilistic:
                check_residuals_input(
                    forecaster_name=type(self).__name__,
                    use_in_sample_residuals=use_in_sample_residuals,
                    in_sample_residuals_=self.in_sample_residuals_,
                    out_sample_residuals_=self.out_sample_residuals_,
                    use_binned_residuals=use_binned_residuals,
                    in_sample_residuals_by_bin_=self.in_sample_residuals_by_bin_,
                    out_sample_residuals_by_bin_=self.out_sample_residuals_by_bin_,
                )

        last_window_values = (
            last_window.iloc[-self.window_size :].to_numpy(copy=True).ravel()
        )
        last_window_values = transform_numpy(
            array=last_window_values,
            transformer=self.transformer_y,
            fit=False,
            inverse_transform=False,
        )
        if self.differentiation is not None:
            last_window_values = self.differentiator.fit_transform(last_window_values)

        if exog is not None:
            exog = input_to_frame(data=exog, input_name="exog")
            if exog.columns.tolist() != self.exog_names_in_:
                exog = exog[self.exog_names_in_]

            exog = transform_dataframe(
                df=exog,
                transformer=self.transformer_exog,
                fit=False,
                inverse_transform=False,
            )

            if not exog.dtypes.to_dict() == self.exog_dtypes_out_:
                check_exog_dtypes(exog=exog)
            else:
                check_exog(exog=exog, allow_nan=False)

            exog_values = exog.to_numpy()[:steps]
        else:
            exog_values = None

        prediction_index = expand_index(index=last_window.index, steps=steps)

        if self.transformer_y is not None or self.differentiation is not None:
            warnings.warn(
                "The output matrix is in the transformed scale due to the "
                "inclusion of transformations or differentiation in the Forecaster. "
                "As a result, any predictions generated using this matrix will also "
                "be in the transformed scale. Please refer to the documentation "
                "for more details: "
                "https://skforecast.org/latest/user_guides/training-and-prediction-matrices.html",
                DataTransformationWarning,
            )

        return last_window_values, exog_values, prediction_index, steps

    def _recursive_predict(
        self,
        steps: int,
        last_window_values: np.ndarray,
        exog_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predict n steps ahead. It is an iterative process in which, each prediction,
        is used as a predictor for the next step.

        Args:
            steps:
                Number of steps to predict.
            last_window_values:
                Series values used to create the predictors needed in the first
                iteration of the prediction (t + 1).
            exog_values:
                Exogenous variable/s included as predictor/s.

        Returns:
            Predicted values.
        """

        original_device = set_cpu_gpu_device(estimator=self.estimator, device="cpu")

        n_lags = len(self.lags) if self.lags is not None else 0
        n_window_features = (
            len(self.X_train_window_features_names_out_)
            if self.window_features is not None
            else 0
        )
        n_exog = exog_values.shape[1] if exog_values is not None else 0

        X = np.full(
            shape=(n_lags + n_window_features + n_exog), fill_value=np.nan, dtype=float
        )
        predictions = np.full(shape=steps, fill_value=np.nan, dtype=float)
        last_window = np.concatenate((last_window_values, predictions))

        estimator_name = type(self.estimator).__name__
        is_linear = isinstance(self.estimator, LinearModel)
        is_lightgbm = estimator_name == "LGBMRegressor"
        is_xgboost = estimator_name == "XGBRegressor"

        if is_linear:
            coef = self.estimator.coef_
            intercept = self.estimator.intercept_
        elif is_lightgbm:
            booster = self.estimator.booster_
        elif is_xgboost:
            booster = self.estimator.get_booster()

        has_lags = self.lags is not None
        has_window_features = self.window_features is not None
        has_exog = exog_values is not None

        for i in range(steps):

            if has_lags:
                X[:n_lags] = last_window[-self.lags - (steps - i)]
            if has_window_features:
                window_data = last_window[i : -(steps - i)]
                X[n_lags : n_lags + n_window_features] = np.concatenate(
                    [wf.transform(window_data) for wf in self.window_features]
                )
            if has_exog:
                X[n_lags + n_window_features :] = exog_values[i]

            if is_linear:
                pred = np.dot(X, coef) + intercept
            elif is_lightgbm:
                pred = booster.predict(X.reshape(1, -1))
            elif is_xgboost:
                pred = booster.inplace_predict(X.reshape(1, -1))
            else:
                pred = self.estimator.predict(X.reshape(1, -1)).ravel()

            pred = pred.item()
            predictions[i] = pred

            # Update `last_window` values. The first position is discarded and
            # the new prediction is added at the end.
            last_window[-(steps - i)] = pred

        set_cpu_gpu_device(estimator=self.estimator, device=original_device)

        return predictions

    def _recursive_predict_bootstrapping(
        self,
        steps: int,
        last_window_values: np.ndarray,
        sampled_residuals: np.ndarray,
        use_binned_residuals: bool,
        n_boot: int,
        exog_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Vectorized bootstrap prediction - predict all n_boot iterations per step.
        Instead of running n_boot sequential predictions, this method predicts
        all bootstrap samples at once per step, significantly reducing overhead.

        Args:
            steps:
                Number of steps to predict.
            last_window_values:
                Series values used to create the predictors needed in the first
                iteration of the prediction (t + 1).
            sampled_residuals:
                Pre-sampled residuals for all bootstrap iterations.
                - If `use_binned_residuals=True`: 3D array of shape (n_bins, steps, n_boot)
                - If `use_binned_residuals=False`: 2D array of shape (steps, n_boot)
            use_binned_residuals:
                If `True`, residuals are selected based on the predicted values.
                If `False`, residuals are selected randomly.
            n_boot:
                Number of bootstrap iterations.
            exog_values:
                Exogenous variable/s included as predictor/s. Defaults to None.

        Returns:
            Numpy ndarray with the predicted values. Shape (steps, n_boot).

        Raises:
            ValueError:
                If `sampled_residuals` does not match the expected shape/dimensions.
            IndexError:
                If `last_window_values` or `exog_values` are not of expected lengths.

        Examples:
            >>> import numpy as np
            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=2)
            >>> _ = forecaster.fit(y=pd.Series(np.arange(10)))
            >>> last_window = np.array([8, 9])
            >>> residuals = np.random.normal(size=(3, 5)) # 3 steps, 5 boots
            >>> preds = forecaster._recursive_predict_bootstrapping(
            ...     steps=3,
            ...     last_window_values=last_window,
            ...     sampled_residuals=residuals,
            ...     use_binned_residuals=False,
            ...     n_boot=5
            ... )
            >>> preds.shape
            (3, 5)
        """

        original_device = set_cpu_gpu_device(estimator=self.estimator, device="cpu")

        n_lags = len(self.lags) if self.lags is not None else 0
        n_window_features = (
            len(self.X_train_window_features_names_out_)
            if self.window_features is not None
            else 0
        )
        n_exog = exog_values.shape[1] if exog_values is not None else 0
        n_features = n_lags + n_window_features + n_exog

        # Input matrix for prediction: shape (n_boot, n_features)
        X = np.full((n_boot, n_features), fill_value=np.nan, dtype=float)

        # Output predictions: shape (steps, n_boot)
        predictions = np.full((steps, n_boot), fill_value=np.nan, dtype=float)

        # Expand last_window to 2D: (window_size + steps, n_boot)
        # Each column represents a separate bootstrap trajectory
        last_window = np.tile(last_window_values[:, np.newaxis], (1, n_boot))
        last_window = np.vstack([last_window, np.full((steps, n_boot), np.nan)])

        estimator_name = type(self.estimator).__name__
        is_linear = isinstance(self.estimator, LinearModel)
        is_lightgbm = estimator_name == "LGBMRegressor"
        is_xgboost = estimator_name == "XGBRegressor"

        if is_linear:
            coef = self.estimator.coef_
            intercept = self.estimator.intercept_
        elif is_lightgbm:
            booster = self.estimator.booster_
        elif is_xgboost:
            booster = self.estimator.get_booster()

        has_lags = self.lags is not None
        has_window_features = self.window_features is not None
        has_exog = exog_values is not None

        for i in range(steps):

            if has_lags:
                for j, lag in enumerate(self.lags):
                    X[:, j] = last_window[-(lag + steps - i), :]

            if has_window_features:
                window_data = last_window[: -(steps - i), :]
                # transform accepts 2D: (window_length, n_boot) -> (n_boot, n_stats)
                # and concatenate along axis=1: (n_boot, total_window_features)
                X[:, n_lags : n_lags + n_window_features] = np.concatenate(
                    [wf.transform(window_data) for wf in self.window_features], axis=1
                )

            if has_exog:
                X[:, n_lags + n_window_features :] = exog_values[i]

            if is_linear:
                pred = np.dot(X, coef) + intercept
            elif is_lightgbm:
                pred = booster.predict(X)
            elif is_xgboost:
                pred = booster.inplace_predict(X)
            else:
                pred = self.estimator.predict(X).ravel()

            if use_binned_residuals:
                # sampled_residuals is a 3D array: (n_bins, steps, n_boot)
                boot_indices = np.arange(n_boot)
                pred_bins = self.binner.transform(pred).astype(int)
                pred += sampled_residuals[pred_bins, i, boot_indices]
            else:
                pred += sampled_residuals[i, :]

            predictions[i, :] = pred
            last_window[-(steps - i), :] = pred

        set_cpu_gpu_device(estimator=self.estimator, device=original_device)

        return predictions

    def create_predict_X(
        self,
        steps: int,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        check_inputs: bool = True,
    ) -> pd.DataFrame:
        """
        Create the predictors needed to predict `steps` ahead. As it is a recursive
        process, the predictors are created at each iteration of the prediction
        process.

        Args:
            steps:
                Number of steps to predict.
                - If steps is int, number of steps to predict.
                - If str or pandas Datetime, the prediction will be up to that date.
            last_window:
                Series values used to create the predictors (lags) needed in the
                first iteration of the prediction (t + 1).
                If `last_window = None`, the values stored in `self.last_window_` are
                used to calculate the initial predictors, and the predictions start
                right after training data. Defaults to None.
            exog:
                Exogenous variable/s included as predictor/s. Defaults to None.
            check_inputs:
                If `True`, the input is checked for possible warnings and errors
                with the `check_predict_input` function. This argument is created
                for internal use and is not recommended to be changed.
                Defaults to True.

        Returns:
            Pandas DataFrame with the predictors for each step. The index
            is the same as the prediction index.
        """

        (
            last_window_values,
            exog_values,
            prediction_index,
            steps,
        ) = self._create_predict_inputs(
            steps=steps,
            last_window=last_window,
            exog=exog,
            check_inputs=check_inputs,
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            predictions = self._recursive_predict(
                steps=steps,
                last_window_values=last_window_values,
                exog_values=exog_values,
            )

        X_predict = []
        full_predictors = np.concatenate((last_window_values, predictions))

        if self.lags is not None:
            idx = np.arange(-steps, 0)[:, None] - self.lags
            X_lags = full_predictors[idx + len(full_predictors)]
            X_predict.append(X_lags)

        if self.window_features is not None:
            X_window_features = np.full(
                shape=(steps, len(self.X_train_window_features_names_out_)),
                fill_value=np.nan,
                order="C",
                dtype=float,
            )
            for i in range(steps):
                X_window_features[i, :] = np.concatenate(
                    [
                        wf.transform(full_predictors[i : -(steps - i)])
                        for wf in self.window_features
                    ]
                )
            X_predict.append(X_window_features)

        if exog is not None:
            X_predict.append(exog_values)

        X_predict = pd.DataFrame(
            data=np.concatenate(X_predict, axis=1),
            columns=self.X_train_features_names_out_,
            index=prediction_index,
        )

        if self.exog_in_:
            categorical_features = any(
                not pd.api.types.is_numeric_dtype(dtype)
                or pd.api.types.is_bool_dtype(dtype)
                for dtype in set(self.exog_dtypes_out_.values())
            )
            if categorical_features:
                X_predict = X_predict.astype(self.exog_dtypes_out_)

        if self.transformer_y is not None or self.differentiation is not None:
            warnings.warn(
                "The output matrix is in the transformed scale due to the "
                "inclusion of transformations or differentiation in the Forecaster. "
                "As a result, any predictions generated using this matrix will also "
                "be in the transformed scale. Please refer to the documentation "
                "for more details: "
                "https://skforecast.org/latest/user_guides/training-and-prediction-matrices.html",
                DataTransformationWarning,
            )

        return X_predict

    def predict(
        self,
        steps: int | str | pd.Timestamp,
        last_window: Union[pd.Series, pd.DataFrame, None] = None,
        exog: Union[pd.Series, pd.DataFrame, None] = None,
        check_inputs: bool = True,
    ) -> pd.Series:
        """
        Predict future values recursively for the specified number of steps.

        Args:
            steps:
                Number of future steps to predict.
            last_window:
                Optional last window of observed values to use for prediction. If None, uses the last window from training.
                Must be a pandas Series or DataFrame with the same structure as the training target series. Defaults to None.
            exog:
                Optional exogenous variables for prediction. Can be a pandas Series or DataFrame.
                Must have the same structure as the exogenous variables used in training. Defaults to None.
            check_inputs:
                Whether to perform input validation checks. Defaults to True.

        Returns:
            Pandas Series of predicted values for the specified number of steps,
            indexed according to the prediction index constructed from the last window and the number of steps.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from spotforecast2_safe.preprocessing import RollingFeatures
            >>> y = pd.Series(np.arange(30), name='y')
            >>> exog = pd.DataFrame({'temp': np.random.randn(30)}, index=y.index)
            >>> forecaster = ForecasterRecursive(
            ...     estimator=LinearRegression(),
            ...     lags=3,
            ...     window_features=[RollingFeatures(stats='mean', window_sizes=3)]
            ... )
            >>> forecaster.fit(y=y, exog=exog)
            >>> last_window = y.iloc[-3:]
            >>> exog_future = pd.DataFrame({'temp': np.random.randn(5)}, index=pd.RangeIndex(start=30, stop=35))
            >>> predictions = forecaster.predict(
            ...     steps=5, last_window=last_window, exog=exog_future, check_inputs=True
            ... )
            >>> isinstance(predictions, pd.Series)
            True
        """

        last_window_values, exog_values, prediction_index, steps = (
            self._create_predict_inputs(
                steps=steps,
                last_window=last_window,
                exog=exog,
                check_inputs=check_inputs,
            )
        )

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            predictions = self._recursive_predict(
                steps=steps,
                last_window_values=last_window_values,
                exog_values=exog_values,
            )

        if self.differentiation is not None:
            predictions = self.differentiator.inverse_transform_next_window(predictions)

        predictions = transform_numpy(
            array=predictions,
            transformer=self.transformer_y,
            fit=False,
            inverse_transform=True,
        )

        predictions = pd.Series(data=predictions, index=prediction_index, name="pred")

        return predictions

    def predict_bootstrapping(
        self,
        steps: int | str | pd.Timestamp,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        n_boot: int = 250,
        use_in_sample_residuals: bool = True,
        use_binned_residuals: bool = True,
        random_state: int = 123,
    ) -> pd.DataFrame:
        """
        Generate multiple forecasting predictions using a bootstrapping process.
        By sampling from a collection of past observed errors (the residuals),
        each iteration of bootstrapping generates a different set of predictions.
        See the References section for more information.

        Args:
            steps:
                Number of steps to predict.
                - If steps is int, number of steps to predict.
                - If str or pandas Datetime, the prediction will be up to that date.
            last_window:
                Series values used to create the predictors (lags) needed in the
                first iteration of the prediction (t + 1).
                If `last_window = None`, the values stored in `self.last_window_` are
                used to calculate the initial predictors, and the predictions start
                right after training data. Defaults to None.
            exog:
                Exogenous variable/s included as predictor/s. Defaults to None.
            n_boot:
                Number of bootstrapping iterations to perform when estimating prediction
                intervals. Defaults to 250.
            use_in_sample_residuals:
                If `True`, residuals from the training data are used as proxy of
                prediction error to create predictions.
                If `False`, out of sample residuals (calibration) are used.
                Out-of-sample residuals must be precomputed using Forecaster's
                `set_out_sample_residuals()` method. Defaults to True.
            use_binned_residuals:
                If `True`, residuals are selected based on the predicted values
                (binned selection).
                If `False`, residuals are selected randomly. Defaults to True.
            random_state:
                Seed for the random number generator to ensure reproducibility. Defaults to 123.

        Returns:
            Pandas DataFrame with predictions generated by bootstrapping. Shape: (steps, n_boot).

        Raises:
            ValueError:
                If `steps` is not an integer or a valid date.
            ValueError:
                If `exog` is missing or has invalid shape.
            ValueError:
                If `n_boot` is not a positive integer.
            ValueError:
                If `use_in_sample_residuals=True` and `in_sample_residuals_` are not available.
            ValueError:
                If `use_in_sample_residuals=False` and `out_sample_residuals_` are not available.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> rng = np.random.default_rng(123)
            >>> y = pd.Series(rng.normal(size=100), name='y')
            >>> forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
            >>> _ = forecaster.fit(y=y)
            >>> boot_preds = forecaster.predict_bootstrapping(steps=3, n_boot=5)
            >>> boot_preds.shape
            (3, 5)

        References:
            .. [1] Forecasting: Principles and Practice (3rd ed) Rob J Hyndman and George Athanasopoulos.
                   https://otexts.com/fpp3/prediction-intervals.html
        """

        (
            last_window_values,
            exog_values,
            prediction_index,
            steps,
        ) = self._create_predict_inputs(
            steps=steps,
            last_window=last_window,
            exog=exog,
            predict_probabilistic=True,
            use_in_sample_residuals=use_in_sample_residuals,
            use_binned_residuals=use_binned_residuals,
            check_inputs=True,
        )

        if use_in_sample_residuals:
            residuals = self.in_sample_residuals_
            residuals_by_bin = self.in_sample_residuals_by_bin_
        else:
            residuals = self.out_sample_residuals_
            residuals_by_bin = self.out_sample_residuals_by_bin_

        rng = np.random.default_rng(seed=random_state)
        if use_binned_residuals:
            # Create 3D array with sampled residuals: (n_bins, steps, n_boot)
            n_bins = len(residuals_by_bin)
            sampled_residuals = np.stack(
                [
                    residuals_by_bin[k][
                        rng.integers(
                            low=0, high=len(residuals_by_bin[k]), size=(steps, n_boot)
                        )
                    ]
                    for k in range(n_bins)
                ],
                axis=0,
            )
        else:
            sampled_residuals = residuals[
                rng.integers(low=0, high=len(residuals), size=(steps, n_boot))
            ]

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            boot_predictions = self._recursive_predict_bootstrapping(
                steps=steps,
                last_window_values=last_window_values,
                exog_values=exog_values,
                sampled_residuals=sampled_residuals,
                use_binned_residuals=use_binned_residuals,
                n_boot=n_boot,
            )

        if self.differentiation is not None:
            boot_predictions = self.differentiator.inverse_transform_next_window(
                boot_predictions
            )

        if self.transformer_y:
            boot_predictions = transform_numpy(
                array=boot_predictions,
                transformer=self.transformer_y,
                fit=False,
                inverse_transform=True,
            )

        boot_columns = [f"pred_boot_{i}" for i in range(n_boot)]
        boot_predictions = pd.DataFrame(
            data=boot_predictions, index=prediction_index, columns=boot_columns
        )

        return boot_predictions

    def predict_quantiles(
        self,
        steps: int | str | pd.Timestamp,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        quantiles: list[float] | tuple[float] = [0.05, 0.5, 0.95],
        n_boot: int = 250,
        use_in_sample_residuals: bool = True,
        use_binned_residuals: bool = True,
        random_state: int = 123,
    ) -> pd.DataFrame:
        """
        Calculate the specified quantiles for each step. After generating
        multiple forecasting predictions through a bootstrapping process, each
        quantile is calculated for each step.

        Args:
            steps: Number of steps to predict.
                - If steps is int, number of steps to predict.
                - If str or pandas Datetime, the prediction will be up to that date.
            last_window: Series values used to create the predictors (lags) needed in the
                first iteration of the prediction (t + 1).
                If `last_window = None`, the values stored in` self.last_window_` are
                used to calculate the initial predictors, and the predictions start
                right after training data.
            exog: Exogenous variable/s included as predictor/s.
            quantiles: Sequence of quantiles to compute, which must be between 0 and 1
                inclusive. For example, quantiles of 0.05, 0.5 and 0.95 should be as
                `quantiles = [0.05, 0.5, 0.95]`.
            n_boot: Number of bootstrapping iterations to perform when estimating quantiles.
            use_in_sample_residuals: If `True`, residuals from the training data are used as proxy of
                prediction error to create predictions.
                If `False`, out of sample residuals (calibration) are used.
                Out-of-sample residuals must be precomputed using Forecaster's
                `set_out_sample_residuals()` method.
            use_binned_residuals: If `True`, residuals are selected based on the predicted values
                (binned selection).
                If `False`, residuals are selected randomly.
            random_state: Seed for the random number generator to ensure reproducibility.

        Returns:
            Quantiles predicted by the forecaster.
        """

        check_interval(quantiles=quantiles)

        boot_predictions = self.predict_bootstrapping(
            steps=steps,
            last_window=last_window,
            exog=exog,
            n_boot=n_boot,
            random_state=random_state,
            use_in_sample_residuals=use_in_sample_residuals,
            use_binned_residuals=use_binned_residuals,
        )

        predictions = boot_predictions.quantile(q=quantiles, axis=1).transpose()
        predictions.columns = [f"q_{q}" for q in quantiles]

        return predictions

    def predict_dist(
        self,
        steps: int | str | pd.Timestamp,
        distribution: object,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        n_boot: int = 250,
        use_in_sample_residuals: bool = True,
        use_binned_residuals: bool = True,
        random_state: int = 123,
    ) -> pd.DataFrame:
        """
        Fit a given probability distribution for each step. After generating
        multiple forecasting predictions through a bootstrapping process, each
        step is fitted to the given distribution.

        Args:
            steps: Number of steps to predict.
                - If steps is int, number of steps to predict.
                - If str or pandas Datetime, the prediction will be up to that date.
            distribution: A distribution object from scipy.stats with methods `_pdf` and `fit`.
                For example scipy.stats.norm.
            last_window: Series values used to create the predictors (lags) needed in the
                first iteration of the prediction (t + 1).
                If `last_window = None`, the values stored in` self.last_window_` are
                used to calculate the initial predictors, and the predictions start
                right after training data.
            exog: Exogenous variable/s included as predictor/s.
            n_boot: Number of bootstrapping iterations to perform when estimating prediction
                intervals.
            use_in_sample_residuals: If `True`, residuals from the training data are used as proxy of
                prediction error to create predictions.
                If `False`, out of sample residuals (calibration) are used.
                Out-of-sample residuals must be precomputed using Forecaster's
                `set_out_sample_residuals()` method.
            use_binned_residuals: If `True`, residuals are selected based on the predicted values
                (binned selection).
                If `False`, residuals are selected randomly.
            random_state: Seed for the random number generator to ensure reproducibility.

        Returns:
            Distribution parameters estimated for each step.
        """

        if not hasattr(distribution, "_pdf") or not callable(
            getattr(distribution, "fit", None)
        ):
            raise TypeError(
                "`distribution` must be a valid probability distribution object "
                "from scipy.stats, with methods `_pdf` and `fit`."
            )

        predictions = self.predict_bootstrapping(
            steps=steps,
            last_window=last_window,
            exog=exog,
            n_boot=n_boot,
            random_state=random_state,
            use_in_sample_residuals=use_in_sample_residuals,
            use_binned_residuals=use_binned_residuals,
        )

        param_names = [
            p for p in inspect.signature(distribution._pdf).parameters if not p == "x"
        ] + ["loc", "scale"]

        predictions[param_names] = predictions.apply(
            lambda x: distribution.fit(x), axis=1, result_type="expand"
        )
        predictions = predictions[param_names]

        return predictions

    def _predict_interval_conformal(
        self,
        steps: int | str | pd.Timestamp,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        nominal_coverage: float = 0.95,
        use_in_sample_residuals: bool = True,
        use_binned_residuals: bool = True,
    ) -> pd.DataFrame:
        """
        Generate prediction intervals using the conformal prediction
        split method [1]_.

        Args:
            steps:
                Number of steps to predict.
                - If steps is int, number of steps to predict.
                - If str or pandas Datetime, the prediction will be up to that date.
            last_window:
                Series values used to create the predictors (lags) needed in the
                first iteration of the prediction (t + 1).
                If `last_window = None`, the values stored in` self.last_window_` are
                used to calculate the initial predictors, and the predictions start
                right after training data. Defaults to None.
            exog:
                Exogenous variable/s included as predictor/s. Defaults to None.
            nominal_coverage:
                Nominal coverage, also known as expected coverage, of the prediction
                intervals. Must be between 0 and 1. Defaults to 0.95.
            use_in_sample_residuals:
                If `True`, residuals from the training data are used as proxy of
                prediction error to create predictions.
                If `False`, out of sample residuals (calibration) are used.
                Out-of-sample residuals must be precomputed using Forecaster's
                `set_out_sample_residuals()` method. Defaults to True.
            use_binned_residuals:
                If `True`, residuals are selected based on the predicted values
                (binned selection).
                If `False`, residuals are selected randomly. Defaults to True.

        Returns:
            Pandas DataFrame with values predicted by the forecaster and their estimated interval.
            - pred: predictions.
            - lower_bound: lower bound of the interval.
            - upper_bound: upper bound of the interval.

        Raises:
            ValueError:
                If `nominal_coverage` is not between 0 and 1.
            ValueError:
                If inputs are invalid (checked by `_create_predict_inputs`).

        Examples:
            >>> # Internal method, typically used via predict_interval(method='conformal')
            >>> import numpy as np
            >>> import pandas as pd
            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> rng = np.random.default_rng(123)
            >>> y = pd.Series(rng.normal(size=100), name='y')
            >>> forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
            >>> _ = forecaster.fit(y=y)
            >>> preds = forecaster._predict_interval_conformal(steps=3, nominal_coverage=0.9)
            >>> preds.columns.tolist()
            ['pred', 'lower_bound', 'upper_bound']

        References:
            .. [1] MAPIE - Model Agnostic Prediction Interval Estimator.
                   https://mapie.readthedocs.io/en/stable/theoretical_description_regression.html#the-split-method
        """

        last_window_values, exog_values, prediction_index, steps = (
            self._create_predict_inputs(
                steps=steps,
                last_window=last_window,
                exog=exog,
                predict_probabilistic=True,
                use_in_sample_residuals=use_in_sample_residuals,
                use_binned_residuals=use_binned_residuals,
                check_inputs=True,
            )
        )

        if use_in_sample_residuals:
            residuals = self.in_sample_residuals_
            residuals_by_bin = self.in_sample_residuals_by_bin_
        else:
            residuals = self.out_sample_residuals_
            residuals_by_bin = self.out_sample_residuals_by_bin_

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names",
                category=UserWarning,
            )
            predictions = self._recursive_predict(
                steps=steps,
                last_window_values=last_window_values,
                exog_values=exog_values,
            )

        if use_binned_residuals:
            # Fallback to global residuals if bin is empty
            if len(residuals) > 0:
                global_cf = np.quantile(np.abs(residuals), nominal_coverage)
            else:
                global_cf = np.nan

            correction_factor_by_bin = {}
            for k, v in residuals_by_bin.items():
                if len(v) > 0:
                    correction_factor_by_bin[k] = np.quantile(
                        np.abs(v), nominal_coverage
                    )
                else:
                    correction_factor_by_bin[k] = global_cf

            replace_func = np.vectorize(
                lambda x: correction_factor_by_bin.get(x, global_cf)
            )

            predictions_bin = self.binner.transform(predictions)
            correction_factor = replace_func(predictions_bin)
        else:
            correction_factor = np.quantile(np.abs(residuals), nominal_coverage)

        lower_bound = predictions - correction_factor
        upper_bound = predictions + correction_factor
        predictions = np.column_stack([predictions, lower_bound, upper_bound])

        if self.differentiation is not None:
            predictions = self.differentiator.inverse_transform_next_window(predictions)

        if self.transformer_y:
            predictions = transform_numpy(
                array=predictions,
                transformer=self.transformer_y,
                fit=False,
                inverse_transform=True,
            )

        predictions = pd.DataFrame(
            data=predictions,
            index=prediction_index,
            columns=["pred", "lower_bound", "upper_bound"],
        )

        return predictions

    def predict_interval(
        self,
        steps: int | str | pd.Timestamp,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
        method: str = "bootstrapping",
        interval: float | list[float] | tuple[float] = [5, 95],
        n_boot: int = 250,
        use_in_sample_residuals: bool = True,
        use_binned_residuals: bool = True,
        random_state: int = 123,
    ) -> pd.DataFrame:
        """
        Predict n steps ahead and estimate prediction intervals using either
        bootstrapping or conformal prediction methods. Refer to the References
        section for additional details on these methods.

        Args:
            steps:
                Number of steps to predict.
                - If steps is int, number of steps to predict.
                - If str or pandas Datetime, the prediction will be up to that date.
            last_window:
                Series values used to create the predictors (lags) needed in the
                first iteration of the prediction (t + 1).
                If `last_window = None`, the values stored in `self.last_window_` are
                used to calculate the initial predictors, and the predictions start
                right after training data. Defaults to None.
            exog:
                Exogenous variable/s included as predictor/s. Defaults to None.
            method:
                Technique used to estimate prediction intervals. Available options:
                - 'bootstrapping': Bootstrapping is used to generate prediction
                  intervals [1]_.
                - 'conformal': Employs the conformal prediction split method for
                  interval estimation [2]_.
                Defaults to 'bootstrapping'.
            interval:
                Confidence level of the prediction interval. Interpretation depends
                on the method used:
                - If `float`, represents the nominal (expected) coverage (between 0
                  and 1). For instance, `interval=0.95` corresponds to `[2.5, 97.5]`
                  percentiles.
                - If `list` or `tuple`, defines the exact percentiles to compute, which
                  must be between 0 and 100 inclusive. For example, interval
                  of 95% should be as `interval = [2.5, 97.5]`.
                - When using `method='conformal'`, the interval must be a float or
                  a list/tuple defining a symmetric interval.
                Defaults to [5, 95].
            n_boot:
                Number of bootstrapping iterations to perform when estimating prediction
                intervals. Defaults to 250.
            use_in_sample_residuals:
                If `True`, residuals from the training data are used as proxy of
                prediction error to create predictions.
                If `False`, out of sample residuals (calibration) are used.
                Out-of-sample residuals must be precomputed using Forecaster's
                `set_out_sample_residuals()` method. Defaults to True.
            use_binned_residuals:
                If `True`, residuals are selected based on the predicted values
                (binned selection).
                If `False`, residuals are selected randomly. Defaults to True.
            random_state:
                Seed for the random number generator to ensure reproducibility. Defaults to 123.

        Returns:
            Pandas DataFrame with values predicted by the forecaster and their estimated interval.
            - pred: predictions.
            - lower_bound: lower bound of the interval.
            - upper_bound: upper bound of the interval.

        Raises:
            ValueError:
                If `method` is not 'bootstrapping' or 'conformal'.
            ValueError:
                 If `interval` is invalid or not compatible with the chosen method.
            ValueError:
                If inputs (`steps`, `exog`, etc.) are invalid.

        Examples:
            >>> import numpy as np
            >>> import pandas as pd
            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> rng = np.random.default_rng(123)
            >>> y = pd.Series(rng.normal(size=100), name='y')
            >>> forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
            >>> _ = forecaster.fit(y=y)
            >>> # Bootstrapping method
            >>> intervals_boot = forecaster.predict_interval(
            ...     steps=3, method='bootstrapping', interval=[5, 95]
            ... )
            >>> intervals_boot.columns.tolist()
            ['pred', 'lower_bound', 'upper_bound']

            >>> # Conformal method
            >>> intervals_conf = forecaster.predict_interval(
            ...     steps=3, method='conformal', interval=0.95
            ... )
            >>> intervals_conf.columns.tolist()
            ['pred', 'lower_bound', 'upper_bound']

        References:
            .. [1] Forecasting: Principles and Practice (3rd ed) Rob J Hyndman and George Athanasopoulos.
                   https://otexts.com/fpp3/prediction-intervals.html
            .. [2] MAPIE - Model Agnostic Prediction Interval Estimator.
                   https://mapie.readthedocs.io/en/stable/theoretical_description_regression.html#the-split-method
        """

        if method == "bootstrapping":

            if isinstance(interval, (list, tuple)):
                check_interval(interval=interval, ensure_symmetric_intervals=False)
                interval = np.array(interval) / 100
            else:
                check_interval(alpha=interval, alpha_literal="interval")
                interval = np.array([0.5 - interval / 2, 0.5 + interval / 2])

            boot_predictions = self.predict_bootstrapping(
                steps=steps,
                last_window=last_window,
                exog=exog,
                n_boot=n_boot,
                random_state=random_state,
                use_in_sample_residuals=use_in_sample_residuals,
                use_binned_residuals=use_binned_residuals,
            )

            predictions = self.predict(
                steps=steps, last_window=last_window, exog=exog, check_inputs=False
            )

            predictions_interval = boot_predictions.quantile(
                q=interval, axis=1
            ).transpose()
            predictions_interval.columns = ["lower_bound", "upper_bound"]
            predictions = pd.concat((predictions, predictions_interval), axis=1)

        elif method == "conformal":

            if isinstance(interval, (list, tuple)):
                check_interval(interval=interval, ensure_symmetric_intervals=True)
                nominal_coverage = (interval[1] - interval[0]) / 100
            else:
                check_interval(alpha=interval, alpha_literal="interval")
                nominal_coverage = interval

            predictions = self._predict_interval_conformal(
                steps=steps,
                last_window=last_window,
                exog=exog,
                nominal_coverage=nominal_coverage,
                use_in_sample_residuals=use_in_sample_residuals,
                use_binned_residuals=use_binned_residuals,
            )
        else:
            raise ValueError(
                f"Invalid `method` '{method}'. Choose 'bootstrapping' or 'conformal'."
            )

        return predictions

    def _binning_in_sample_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        store_in_sample_residuals: bool = False,
        random_state: int = 123,
    ) -> None:
        """
        Bin residuals according to the predicted value each residual is
        associated with. First a `skforecast.preprocessing.QuantileBinner` object
        is fitted to the predicted values. Then, residuals are binned according
        to the predicted value each residual is associated with. Residuals are
        stored in the forecaster object as `in_sample_residuals_` and
        `in_sample_residuals_by_bin_`.

        `y_true` and `y_pred` assumed to be differentiated and or transformed
        according to the attributes `differentiation` and `transformer_y`.
        The number of residuals stored per bin is limited to
        `10_000 // self.binner.n_bins_`. The total number of residuals stored is
        `10_000`.

        Args:
            y_true: True values of the time series.
            y_pred: Predicted values of the time series.
            store_in_sample_residuals: If `True`, in-sample residuals will be stored in the forecaster object
                after fitting (`in_sample_residuals_` and `in_sample_residuals_by_bin_`
                attributes). If `False`, only the intervals of the bins are stored.
            random_state: Set a seed for the random generator so that the stored sample
                residuals are always deterministic.
        """

        residuals = y_true - y_pred

        if self._probabilistic_mode == "binned":
            data = pd.DataFrame({"prediction": y_pred, "residuals": residuals})
            self.binner.fit(y_pred)
            self.binner_intervals_ = self.binner.intervals_

        if store_in_sample_residuals:
            rng = np.random.default_rng(seed=random_state)
            if self._probabilistic_mode == "binned":
                data["bin"] = self.binner.transform(y_pred).astype(int)
                self.in_sample_residuals_by_bin_ = (
                    data.groupby("bin")["residuals"].apply(np.array).to_dict()
                )

                max_sample = 10_000 // self.binner.n_bins_
                for k, v in self.in_sample_residuals_by_bin_.items():
                    if len(v) > max_sample:
                        sample = v[rng.integers(low=0, high=len(v), size=max_sample)]
                        self.in_sample_residuals_by_bin_[k] = sample

            if len(residuals) > 10_000:
                residuals = residuals[
                    rng.integers(low=0, high=len(residuals), size=10_000)
                ]

            self.in_sample_residuals_ = residuals

    def set_fit_kwargs(self, fit_kwargs: dict[str, object]) -> None:
        """
        Set new values for the additional keyword arguments passed to the `fit`
        method of the estimator.

        Args:
            fit_kwargs: Dict of the form {"argument": new_value}.
        """

        self.fit_kwargs = check_select_fit_kwargs(self.estimator, fit_kwargs=fit_kwargs)

    def set_lags(
        self, lags: Union[int, List[int], np.ndarray, range, None] = None
    ) -> None:
        """
        Set new value to the attribute `lags`. Attributes `lags_names`,
        `max_lag` and `window_size` are also updated.

        Args:
            lags: Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
                - `int`: include lags from 1 to `lags` (included).
                - `list`, `1d numpy ndarray` or `range`: include only lags present in
                `lags`, all elements must be int.
                - `None`: no lags are included as predictors.
        """

        if self.window_features is None and lags is None:
            raise ValueError(
                "At least one of the arguments `lags` or `window_features` "
                "must be different from None. This is required to create the "
                "predictors used in training the forecaster."
            )

        self.lags, self.lags_names, self.max_lag = initialize_lags(
            type(self).__name__, lags
        )
        self.window_size = max(
            [
                ws
                for ws in [self.max_lag, self.max_size_window_features]
                if ws is not None
            ]
        )
        if self.differentiation is not None:
            self.window_size += self.differentiation
            self.differentiator.set_params(window_size=self.window_size)

    def set_window_features(
        self, window_features: object | list[object] | None = None
    ) -> None:
        """
        Set new value to the attribute `window_features`.

        Attributes `max_size_window_features`, `window_features_names`,
        `window_features_class_names` and `window_size` are also updated.

        Args:
            window_features: Instance or list of instances used to create window features.
                Window features are created from the original time series and are
                included as predictors.

        Returns:
            None

        Examples:
            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from spotforecast2_safe.preprocessing import RollingFeatures
            >>> import pandas as pd
            >>> import numpy as np
            >>> forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
            >>> rolling = RollingFeatures(stats=['mean', 'std'], window_sizes=[3, 5])
            >>> forecaster.set_window_features(window_features=rolling)
            >>> forecaster.window_features_names
            ['roll_mean_3', 'roll_std_3', 'roll_mean_5', 'roll_std_5']
            >>> forecaster.window_size
            5
        """

        if window_features is None and self.lags is None:
            raise ValueError(
                "At least one of the arguments `lags` or `window_features` "
                "must be different from None. This is required to create the "
                "predictors used in training the forecaster."
            )

        (
            self.window_features,
            self.window_features_names,
            self.max_size_window_features,
        ) = initialize_window_features(window_features)
        self.window_features_class_names = None
        if window_features is not None:
            self.window_features_class_names = [
                type(wf).__name__ for wf in self.window_features
            ]
        self.window_size = max(
            [
                ws
                for ws in [self.max_lag, self.max_size_window_features]
                if ws is not None
            ]
        )
        if self.differentiation is not None:
            self.window_size += self.differentiation
            self.differentiator.set_params(window_size=self.window_size)

    def get_feature_importances(self, sort_importance: bool = True) -> pd.DataFrame:
        """
        Return feature importances of the estimator stored in the forecaster.
        Only valid when estimator stores internally the feature importances in the
        attribute `feature_importances_` or `coef_`. Otherwise, returns `None`.

        Args:
            sort_importance: If `True`, sorts the feature importances in descending order.

        Returns:
            pd.DataFrame: Feature importances associated with each predictor.

        Raises:
            NotFittedError: If the forecaster is not fitted.

        Examples:
            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> import pandas as pd
            >>> import numpy as np
            >>> forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
            >>> forecaster.fit(y=pd.Series(np.arange(20)))
            >>> forecaster.get_feature_importances()
              feature  importance
            0   lag_1         1.0
            1   lag_2         0.0
            2   lag_3         0.0
        """

        if not self.is_fitted:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `get_feature_importances()`."
            )

        if isinstance(self.estimator, Pipeline):
            estimator = self.estimator[-1]
        else:
            estimator = self.estimator

        if hasattr(estimator, "feature_importances_"):
            feature_importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            feature_importances = estimator.coef_
        else:
            warnings.warn(
                f"Impossible to access feature importances for estimator of type "
                f"{type(estimator)}. This method is only valid when the "
                f"estimator stores internally the feature importances in the "
                f"attribute `feature_importances_` or `coef_`.",
                UserWarning,
            )
            feature_importances = None

        if feature_importances is not None:
            feature_importances = pd.DataFrame(
                {
                    "feature": self.X_train_features_names_out_,
                    "importance": feature_importances,
                }
            )
            if sort_importance:
                feature_importances = feature_importances.sort_values(
                    by="importance", ascending=False
                )

        return feature_importances

    def set_in_sample_residuals(
        self,
        y: pd.Series,
        exog: pd.Series | pd.DataFrame | None = None,
        random_state: int = 123,
    ) -> None:
        """
        Set in-sample residuals in case they were not calculated during the
        training process.

        In-sample residuals are calculated as the difference between the true
        values and the predictions made by the forecaster using the training
        data. The following internal attributes are updated:

        + `in_sample_residuals_`: residuals stored in a numpy ndarray.
        + `binner_intervals_`: intervals used to bin the residuals are calculated
        using the quantiles of the predicted values.
        + `in_sample_residuals_by_bin_`: residuals are binned according to the
        predicted value they are associated with and stored in a dictionary, where
        the keys are the intervals of the predicted values and the values are
        the residuals associated with that range.

        A total of 10_000 residuals are stored in the attribute `in_sample_residuals_`.
        If the number of residuals is greater than 10_000, a random sample of
        10_000 residuals is stored. The number of residuals stored per bin is
        limited to `10_000 // self.binner.n_bins_`.

        Args:
                y: Target time series.
            exog: Exogenous variables.
            random_state: Random state for reproducibility.

        Returns:
            None

        Raises:
            NotFittedError: If the forecaster is not fitted.
            IndexError: If the index range of `y` does not match the range
                used during training.
            ValueError: If the features generated from the provided data do not
                match those used during the training process.

        Examples:
            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
            >>> forecaster.fit(y=pd.Series(np.arange(20)), store_in_sample_residuals=False)
            >>> forecaster.set_in_sample_residuals(y=pd.Series(np.arange(20)))
            >>> forecaster.in_sample_residuals_
            array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        """
        if not self.is_fitted:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `set_in_sample_residuals()`."
            )

        check_y(y=y)
        y_index_range = check_extract_values_and_index(
            data=y, data_label="`y`", return_values=False
        )[1][[0, -1]]

        if not y_index_range.equals(self.training_range_):
            raise IndexError(
                f"The index range of `y` does not match the range "
                f"used during training. Please ensure the index is aligned "
                f"with the training data.\n"
                f"    Expected : {self.training_range_}\n"
                f"    Received : {y_index_range}"
            )

        (
            X_train,
            y_train,
            _,
            _,
            _,
            X_train_features_names_out_,
            *_,
        ) = self._create_train_X_y(y=y, exog=exog)

        if not X_train_features_names_out_ == self.X_train_features_names_out_:
            raise ValueError(
                f"Feature mismatch detected after matrix creation. The features "
                f"generated from the provided data do not match those used during "
                f"the training process. To correctly set in-sample residuals, "
                f"ensure that the same data and preprocessing steps are applied.\n"
                f"    Expected output : {self.X_train_features_names_out_}\n"
                f"    Current output  : {X_train_features_names_out_}"
            )

        self._binning_in_sample_residuals(
            y_true=y_train.to_numpy(),
            y_pred=self.estimator.predict(X_train).ravel(),
            store_in_sample_residuals=True,
            random_state=random_state,
        )

    def set_out_sample_residuals(
        self,
        y_true: np.ndarray | pd.Series,
        y_pred: np.ndarray | pd.Series,
        append: bool = False,
        random_state: int = 123,
    ) -> None:
        """
        Set new values to the attribute `out_sample_residuals_`.

        Out of sample residuals are meant to be calculated using observations that
        did not participate in the training process. `y_true` and `y_pred` are
        expected to be in the original scale of the time series. Residuals are
        calculated as `y_true` - `y_pred`, after applying the necessary
        transformations and differentiations if the forecaster includes them
        (`self.transformer_y` and `self.differentiation`). Two internal attributes
        are updated:

        + `out_sample_residuals_`: residuals stored in a numpy ndarray.
        + `out_sample_residuals_by_bin_`: residuals are binned according to the
        predicted value they are associated with and stored in a dictionary, where
        the keys are the intervals of the predicted values and the values are
        the residuals associated with that range. If a bin is empty, it is filled
        with a random sample of residuals from other bins. This is done to ensure
        that all bins have at least one residual and can be used in the prediction
        process.

        A total of 10_000 residuals are stored in the attribute `out_sample_residuals_`.
        If the number of residuals is greater than 10_000, a random sample of
        10_000 residuals is stored. The number of residuals stored per bin is
        limited to `10_000 // self.binner.n_bins_`.

        Args:
            y_true: True values of the time series in the original scale.
            y_pred: Predicted values of the time series in the original scale.
            append: If `True`, new residuals are added to the once already stored
                in the forecaster. If after appending the new residuals, the limit
                of `10_000 // self.binner.n_bins_` values per bin is reached, a
                random sample of residuals is stored.
            random_state: Random state for reproducibility.

        Returns:
            None

        Raises:
            NotFittedError: If the forecaster is not fitted.
            TypeError: If `y_true` or `y_pred` are not `numpy ndarray` or `pandas Series`.
            ValueError: If `y_true` and `y_pred` have different length or index (if Series).

        Examples:
            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> import pandas as pd
            >>> import numpy as np
            >>> forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
            >>> forecaster.fit(y=pd.Series(np.arange(20)), store_in_sample_residuals=False)
            >>> y_true = np.array([20, 21, 22, 23, 24])
            >>> y_pred = np.array([20.1, 20.9, 22.2, 22.8, 24.0])
            >>> forecaster.set_out_sample_residuals(y_true=y_true, y_pred=y_pred)
            >>> forecaster.out_sample_residuals_
            array([-0.1,  0.1, -0.2,  0.2,  0. ])
        """
        if not self.is_fitted:
            raise NotFittedError(
                "This forecaster is not fitted yet. Call `fit` with appropriate "
                "arguments before using `set_out_sample_residuals()`."
            )

        if not isinstance(y_true, (np.ndarray, pd.Series)):
            raise TypeError(
                f"`y_true` argument must be `numpy ndarray` or `pandas Series`. "
                f"Got {type(y_true)}."
            )

        if not isinstance(y_pred, (np.ndarray, pd.Series)):
            raise TypeError(
                f"`y_pred` argument must be `numpy ndarray` or `pandas Series`. "
                f"Got {type(y_pred)}."
            )

        if len(y_true) != len(y_pred):
            raise ValueError(
                f"`y_true` and `y_pred` must have the same length. "
                f"Got {len(y_true)} and {len(y_pred)}."
            )

        if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
            if not y_true.index.equals(y_pred.index):
                raise ValueError("`y_true` and `y_pred` must have the same index.")

        if not isinstance(y_pred, np.ndarray):
            y_pred = y_pred.to_numpy()
        if not isinstance(y_true, np.ndarray):
            y_true = y_true.to_numpy()

        if self.transformer_y:
            y_true = transform_numpy(
                array=y_true,
                transformer=self.transformer_y,
                fit=False,
                inverse_transform=False,
            )
            y_pred = transform_numpy(
                array=y_pred,
                transformer=self.transformer_y,
                fit=False,
                inverse_transform=False,
            )

        if self.differentiation is not None:
            differentiator = copy(self.differentiator)
            differentiator.set_params(window_size=None)
            y_true = differentiator.fit_transform(y_true)[self.differentiation :]
            y_pred = differentiator.fit_transform(y_pred)[self.differentiation :]

        data = pd.DataFrame(
            {"prediction": y_pred, "residuals": y_true - y_pred}
        ).dropna()
        y_pred = data["prediction"].to_numpy()
        residuals = data["residuals"].to_numpy()

        if self.binner is not None:
            data["bin"] = self.binner.transform(y_pred).astype(int)
            residuals_by_bin = (
                data.groupby("bin")["residuals"].apply(np.array).to_dict()
            )
        else:
            residuals_by_bin = {}

        out_sample_residuals = (
            np.array([])
            if self.out_sample_residuals_ is None
            else self.out_sample_residuals_
        )
        out_sample_residuals_by_bin = (
            {}
            if self.out_sample_residuals_by_bin_ is None
            else self.out_sample_residuals_by_bin_
        )
        if append:
            out_sample_residuals = np.concatenate([out_sample_residuals, residuals])
            for k, v in residuals_by_bin.items():
                if k in out_sample_residuals_by_bin:
                    out_sample_residuals_by_bin[k] = np.concatenate(
                        (out_sample_residuals_by_bin[k], v)
                    )
                else:
                    out_sample_residuals_by_bin[k] = v
        else:
            out_sample_residuals = residuals
            out_sample_residuals_by_bin = residuals_by_bin

        if self.binner is not None:
            max_samples = 10_000 // self.binner.n_bins
            rng = np.random.default_rng(seed=random_state)

            for k, v in out_sample_residuals_by_bin.items():
                if len(v) > max_samples:
                    out_sample_residuals_by_bin[k] = rng.choice(
                        v, size=max_samples, replace=False
                    )

            bin_keys = (
                [] if self.binner_intervals_ is None else self.binner_intervals_.keys()
            )
            empty_bins = [
                k
                for k in bin_keys
                if k not in out_sample_residuals_by_bin
                or len(out_sample_residuals_by_bin[k]) == 0
            ]

            if empty_bins:
                warnings.warn(
                    f"The following bins have no out of sample residuals: {empty_bins}. "
                    f"No predicted values fall in the interval "
                    f"{[self.binner_intervals_[bin] for bin in empty_bins]}. "
                    f"Empty bins will be filled with a random sample of residuals.",
                    ResidualsUsageWarning,
                )
                empty_bin_size = min(max_samples, len(out_sample_residuals))
                for k in empty_bins:
                    out_sample_residuals_by_bin[k] = rng.choice(
                        a=out_sample_residuals, size=empty_bin_size, replace=False
                    )

        self.out_sample_residuals_ = out_sample_residuals
        self.out_sample_residuals_by_bin_ = out_sample_residuals_by_bin
