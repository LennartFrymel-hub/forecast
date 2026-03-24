# SPDX-FileCopyrightText: skforecast team
# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later AND BSD-3-Clause


"""ForecasterBase class.

This module contains the base class for all forecasters in spotforecast2_safe and spotforecast.
All forecasters should specify all the parameters that can be set at the
class level in their __init__.

Examples:
    Create a custom forecaster inheriting from ForecasterBase:

    >>> from spotforecast2_safe.forecaster.base import ForecasterBase
    >>> import pandas as pd
    >>> import numpy as np
    >>> class MyForecaster(ForecasterBase):
    ...     def __init__(self, estimator):
    ...         self.estimator = estimator
    ...         self.__spotforecast_tags__ = {'hide_lags': True}
    ...     def create_train_X_y(self, y, exog=None):
    ...         return pd.DataFrame(), pd.Series(dtype=float)
    ...     def fit(self, y, exog=None):
    ...         pass
    ...     def predict(self, steps, last_window=None, exog=None):
    ...         return pd.Series(np.zeros(steps))
    ...     def set_params(self, params):
    ...         pass
    >>> from sklearn.linear_model import Ridge
    >>> forecaster = MyForecaster(estimator=Ridge())
    >>> forecaster
    MyForecaster(estimator=Ridge())
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any
import textwrap
import warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class ForecasterBase(ABC):
    """Base class for all forecasters in spotforecast2.

    All forecasters should specify all the parameters that can be set at
    the class level in their __init__.

    Attributes:
        __spotforecast_tags__: Dictionary with forecaster tags that characterize
            the behavior of the forecaster.

    Examples:
        To see all abstract methods that need to be implemented:

        >>> import inspect
        >>> from spotforecast2_safe.forecaster.base import ForecasterBase
        >>> [m[0] for m in inspect.getmembers(ForecasterBase, predicate=inspect.isabstract)]
        ['create_train_X_y', 'fit', 'predict', 'set_params']
    """

    def _preprocess_repr(
        self,
        estimator: object | None = None,
        training_range_: dict[str, str] | None = None,
        series_names_in_: list[str] | None = None,
        exog_names_in_: list[str] | None = None,
        transformer_series: object | dict[str, object] | None = None,
    ) -> tuple[str, str | None, str | None, str | None, str | None]:
        """Prepare the information to be displayed when a Forecaster object is printed.

        Args:
            estimator: Estimator object. Default is None.
            training_range_: Training range. Only used for ForecasterRecursiveMultiSeries.
                Default is None.
            series_names_in_: Names of the series used in the forecaster.
                Only used for ForecasterRecursiveMultiSeries. Default is None.
            exog_names_in_: Names of the exogenous variables used in the forecaster.
                Default is None.
            transformer_series: Transformer used in the series.
                Only used for ForecasterRecursiveMultiSeries. Default is None.

        Returns:
            Tuple containing params (estimator parameters string), training_range_
            (training range string representation), series_names_in_ (series names
            string representation), exog_names_in_ (exogenous variable names string
            representation), and transformer_series (transformer string representation).

        Examples:
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from sklearn.linear_model import Ridge
            >>> estimator = Ridge(alpha=0.5)
            >>> forecaster = ForecasterRecursive(estimator=estimator, lags=3)
            >>> params, tr, sn, en, ts = forecaster._preprocess_repr(estimator=estimator)
            >>> params
            "{'alpha': 0.5, 'copy_X': True, 'fit_intercept': True, 'max_iter': None, 'positive': False, 'random_state': None, 'solver': 'auto', 'tol': 0.0001}"
        """

        if estimator is not None:
            if isinstance(estimator, Pipeline):
                name_pipe_steps = tuple(
                    name + "__" for name in estimator.named_steps.keys()
                )
                params = {
                    key: value
                    for key, value in estimator.get_params().items()
                    if key.startswith(name_pipe_steps)
                }
            else:
                params = estimator.get_params()
            params = str(params)
        else:
            params = None

        if training_range_ is not None:
            training_range_ = [
                f"'{k}': {v.astype(str).to_list()}" for k, v in training_range_.items()
            ]
            if len(training_range_) > 10:
                training_range_ = training_range_[:5] + ["..."] + training_range_[-5:]
            training_range_ = ", ".join(training_range_)

        if series_names_in_ is not None:
            if len(series_names_in_) > 50:
                series_names_in_ = (
                    series_names_in_[:25] + ["..."] + series_names_in_[-25:]
                )
            series_names_in_ = ", ".join(series_names_in_)

        if exog_names_in_ is not None:
            if len(exog_names_in_) > 50:
                exog_names_in_ = exog_names_in_[:25] + ["..."] + exog_names_in_[-25:]
            exog_names_in_ = ", ".join(exog_names_in_)

        if transformer_series is not None:
            if isinstance(transformer_series, dict):
                transformer_series = [
                    f"'{k}': {v}" for k, v in transformer_series.items()
                ]
                if len(transformer_series) > 10:
                    transformer_series = (
                        transformer_series[:5] + ["..."] + transformer_series[-5:]
                    )
                transformer_series = ", ".join(transformer_series)
            else:
                transformer_series = str(transformer_series)

        return (
            params,
            training_range_,
            series_names_in_,
            exog_names_in_,
            transformer_series,
        )

    def _format_text_repr(
        self,
        text: str,
        max_text_length: int = 58,
        width: int = 80,
        indent: str = "    ",
    ) -> str:
        """Format text for __repr__ method.

        Args:
            text: Text to format.
            max_text_length: Maximum length of the text before wrapping. Default is 58.
            width: Maximum width of the text. Default is 80.
            indent: Indentation of the text. Default is four spaces.

        Returns:
            Formatted text string with proper wrapping and indentation.

        Examples:
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from sklearn.linear_model import Ridge
            >>> forecaster = ForecasterRecursive(estimator=Ridge(), lags=3)
            >>> forecaster._format_text_repr("Short text")
            'Short text'
        """

        if text is not None and len(text) > max_text_length:
            text = "\n    " + textwrap.fill(
                str(text), width=width, subsequent_indent=indent
            )

        return text

    @abstractmethod
    def create_train_X_y(
        self, y: pd.Series, exog: pd.Series | pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Create training matrices from univariate time series and exogenous variables.

        Args:
            y: Training time series.
            exog: Exogenous variable(s) included as predictor(s). Must have the same
                number of observations as y and their indexes must be aligned.
                Default is None.

        Returns:
            Tuple containing X_train (training values/predictors with shape
            (len(y) - max_lag, len(lags))) and y_train (target values of the
            time series related to each row of X_train with shape (len(y) - max_lag,)).

        Examples:
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from sklearn.linear_model import Ridge
            >>> import pandas as pd
            >>> import numpy as np
            >>> forecaster = ForecasterRecursive(estimator=Ridge(), lags=3)
            >>> y = pd.Series(np.arange(10), name='y')
            >>> X_train, y_train = forecaster.create_train_X_y(y)
            >>> X_train.head(2)
               lag_1  lag_2  lag_3
            3    2.0    1.0    0.0
            4    3.0    2.0    1.0
            >>> y_train.head(2)
            3    3
            4    4
            Name: y, dtype: int64
        """

        pass

    @abstractmethod
    def fit(self, y: pd.Series, exog: pd.Series | pd.DataFrame | None = None) -> None:
        """Training Forecaster.

        Args:
            y: Training time series.
            exog: Exogenous variable(s) included as predictor(s). Must have the same
                number of observations as y and their indexes must be aligned so
                that y[i] is regressed on exog[i]. Default is None.

        Returns:
            None

        Examples:
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from sklearn.linear_model import Ridge
            >>> import pandas as pd
            >>> import numpy as np
            >>> forecaster = ForecasterRecursive(estimator=Ridge(), lags=3)
            >>> y = pd.Series(np.arange(10), name='y')
            >>> forecaster.fit(y)
            >>> forecaster.is_fitted
            True
        """

        pass

    @abstractmethod
    def predict(
        self,
        steps: int,
        last_window: pd.Series | pd.DataFrame | None = None,
        exog: pd.Series | pd.DataFrame | None = None,
    ) -> pd.Series:
        """Predict n steps ahead.

        Args:
            steps: Number of steps to predict.
            last_window: Series values used to create the predictors (lags) needed in the
                first iteration of the prediction (t + 1). If None, the values stored in
                last_window are used to calculate the initial predictors, and the
                predictions start right after training data. Default is None.
            exog: Exogenous variable(s) included as predictor(s). Default is None.

        Returns:
            Predicted values as a pandas Series.

        Examples:
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from sklearn.linear_model import Ridge
            >>> import pandas as pd
            >>> import numpy as np
            >>> forecaster = ForecasterRecursive(estimator=Ridge(), lags=3)
            >>> y = pd.Series(np.arange(10), name='y')
            >>> forecaster.fit(y)
            >>> forecaster.predict(steps=3)
            10    9.5
            11    9.0
            12    8.5
            Name: pred, dtype: float64
        """

        pass

    @abstractmethod
    def set_params(self, params: dict[str, object]) -> None:
        """Set new values to the parameters of the scikit-learn model stored in the forecaster.

        Args:
            params: Parameters values dictionary.

        Returns:
            None

        Examples:
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from sklearn.linear_model import Ridge
            >>> forecaster = ForecasterRecursive(estimator=Ridge(alpha=1.0), lags=3)
            >>> forecaster.set_params({'estimator__alpha': 0.5})
            >>> forecaster.estimator.alpha
            0.5
        """

        pass

    def set_lags(
        self, lags: int | list[int] | np.ndarray[int] | range[int] | None = None
    ) -> None:
        """Set new value to the attribute lags.

        Attributes max_lag and window_size are also updated.

        Args:
            lags: Lags used as predictors. Index starts at 1, so lag 1 is equal to t-1.
                If int: include lags from 1 to lags (included). If list, 1d numpy ndarray,
                or range: include only lags present in lags, all elements must be int.
                If None: no lags are included as predictors. Default is None.

        Returns:
            None

        Examples:
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from sklearn.linear_model import Ridge
            >>> forecaster = ForecasterRecursive(estimator=Ridge(), lags=3)
            >>> forecaster.set_lags(lags=5)
            >>> forecaster.lags
            array([1, 2, 3, 4, 5])
        """

        pass

    def set_window_features(
        self, window_features: object | list[object] | None = None
    ) -> None:
        """Set new value to the attribute window_features.

        Attributes max_size_window_features, window_features_names,
        window_features_class_names and window_size are also updated.

        Args:
            window_features: Instance or list of instances used to create window features.
                Window features are created from the original time series and are
                included as predictors. Default is None.

        Returns:
            None

        Examples:
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from spotforecast2_safe.forecaster.preprocessing import RollingFeatures
            >>> from sklearn.linear_model import Ridge
            >>> forecaster = ForecasterRecursive(estimator=Ridge(), lags=3)
            >>> window_feat = RollingFeatures(stats='mean', window_sizes=3)
            >>> forecaster.set_window_features(window_features=window_feat)
            >>> forecaster.window_features
            [RollingFeatures(stats=['mean'], window_sizes=[3])]
        """

        pass

    def get_tags(self) -> dict[str, Any]:
        """Return the tags that characterize the behavior of the forecaster.

        Returns:
            Dictionary with forecaster tags describing behavior and capabilities.

        Examples:
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from sklearn.linear_model import Ridge
            >>> forecaster = ForecasterRecursive(estimator=Ridge(), lags=3)
            >>> tags = forecaster.get_tags()
            >>> tags['forecaster_task']
            'regression'
        """

        return self.__spotforecast_tags__

    def summary(self) -> None:
        """Show forecaster information.

        Returns:
            None

        Examples:
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from sklearn.linear_model import Ridge
            >>> forecaster = ForecasterRecursive(estimator=Ridge(), lags=3)
            >>> forecaster.summary()
            ForecasterRecursive
            ===================
            Estimator: Ridge()
            Lags: [1 2 3]
            ...
        """

        print(self.__repr__())

    def __setstate__(self, state: dict) -> None:
        """Custom __setstate__ to ensure backward compatibility when unpickling.

        This method is called when an object is unpickled (deserialized).
        It handles the migration of deprecated attributes to their new names.

        Args:
            state: The state dictionary from the pickled object.

        Returns:
            None

        Examples:
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from sklearn.linear_model import Ridge
            >>> import pickle
            >>> forecaster = ForecasterRecursive(estimator=Ridge(), lags=3)
            >>> pickled_forecaster = pickle.dumps(forecaster)
            >>> unpickled_forecaster = pickle.loads(pickled_forecaster)
        """

        if "regressor" in state and "estimator" not in state:
            state["estimator"] = state.pop("regressor")

        self.__dict__.update(state)

    @property
    def regressor(self) -> Any:
        """Deprecated property. Use estimator instead.

        Returns:
            The estimator object.

        Examples:
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from sklearn.linear_model import Ridge
            >>> forecaster = ForecasterRecursive(estimator=Ridge(), lags=3)
            >>> forecaster.regressor  # Raises FutureWarning
            Ridge()
        """
        warnings.warn(
            "The `regressor` attribute is deprecated and will be removed in future "
            "versions. Use `estimator` instead.",
            FutureWarning,
        )
        return self.estimator
