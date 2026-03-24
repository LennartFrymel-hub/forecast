# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Recursive forecaster model wrappers for different estimators."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from spotforecast2_safe.data.data import Period
from spotforecast2_safe.data.fetch_data import (
    load_timeseries,
    load_timeseries_forecast,
)
from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.model_selection import TimeSeriesFold, backtesting_forecaster
from spotforecast2_safe.preprocessing import LinearlyInterpolateTS
from spotforecast2_safe.preprocessing.exog_builder import ExogBuilder

logger = logging.getLogger(__name__)


class ForecasterRecursiveModel:
    """
    Base wrapper around ForecasterRecursive to match application logic.

    This class manages the lifecycle of a recursive forecaster, including
    feature building, tuning (simulated), and packaging predictions for UI.

    Attributes:
        iteration (int): The current training iteration.
        end_dev (pd.Timestamp): The end date of the development/training period.
        train_size (Optional[pd.Timedelta]): Lookback window for training data.
        preprocessor (ExogBuilder): Builder for exogenous features.
        name (str): Label for the model type.
        forecaster (Optional[ForecasterRecursive]): The underlying forecaster instance.
        is_tuned (bool): Flag indicating if hyperparameter tuning has been performed.
        predict_size (int): Prediction horizon in hours.
        refit_size (int): Refit interval in days.
        random_state (int): Seed for reproducibility.

    Examples:
        >>> import pandas as pd
        >>> from spotforecast2_safe.manager.models.forecaster_recursive_model import ForecasterRecursiveModel
        >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        >>> from sklearn.linear_model import LinearRegression
        >>>
        >>> model = ForecasterRecursiveModel(iteration=0)
        >>> model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=1)
        >>> model.name = "linear"
        >>> model.tune()
        >>> model.is_tuned
        True
    """

    def __init__(
        self,
        iteration: int,
        end_dev: Union[str, pd.Timestamp] = None,
        train_size: Optional[pd.Timedelta] = None,
        periods: Optional[List[Period]] = None,
        country_code: str = "DE",
        random_state: int = 123456789,
        predict_size: int = 24,
        refit_size: int = 7,
        name: str = "base",
        **kwargs: Any,
    ):
        """
        Initialize the Recursive Forecaster Model.

        Args:
            iteration:
                Current iteration index.
            end_dev:
                Cutoff date for training, e.g., "2025-12-31 00:00+00:00".
                If None, it is calculated automatically to be one day
                before the latest available index in the data.
                Defaults to None.
            train_size:
                Time window for training data lookback.
            periods:
                List of Period objects for cyclical encoding.
            country_code:
                ISO country code for holidays. Defaults to "DE".
            random_state:
                Random seed. Defaults to 123456789.
            predict_size:
                Forecast horizon in hours. Defaults to 24.
            refit_size:
                Retraining frequency in days. Defaults to 7.
            name:
                Model name identifier. Defaults to "base".
            **kwargs:
                Additional parameters for forward compatibility.

        Returns:
            None

        Raises:
            ValueError: If the forecaster has not been initialized.

        Examples:
            >>> import pandas as pd
            >>> from spotforecast2_safe.manager.models.forecaster_recursive_model import ForecasterRecursiveModel
            >>> model = ForecasterRecursiveModel(iteration=0)
            >>> model.name
            'base'
            >>> model.end_dev
            NaT
            >>> model.train_size
            >>> model.random_state
            123456789
            >>> model.predict_size
            24
            >>> model.refit_size
            7
            >>> model.preprocessor # doctest: +ELLIPSIS
            <spotforecast2_safe.preprocessing.exog_builder.ExogBuilder object at 0x...>
            >>> model.is_tuned
            False
        """
        self.iteration = iteration
        self.end_dev = pd.to_datetime(end_dev, utc=True)
        self.train_size = train_size
        self.random_state = random_state
        self.predict_size = predict_size
        self.refit_size = refit_size

        self.preprocessor = ExogBuilder(periods=periods, country_code=country_code)
        self.name = name
        self.forecaster: Optional[ForecasterRecursive] = None
        self.is_tuned = False
        self.save_model_to_file = kwargs.pop("save_model_to_file", True)
        self.results_tuning: Optional[pd.DataFrame] = None
        self.best_params: Optional[dict] = None
        self.best_lags: Optional[List[int]] = None
        self.metrics = ["mean_absolute_error", "mean_absolute_percentage_error"]

    # Maps config attribute name → model __init__ kwarg name.
    # Only entries where the names differ need to be listed; identical
    # names are forwarded as-is by from_config().
    _CONFIG_ATTR_MAP: Dict[str, str] = {
        "API_COUNTRY_CODE": "country_code",
        "end_train_default": "end_dev",
    }

    @classmethod
    def from_config(
        cls, iteration: int, config: Any, **overrides: Any
    ) -> "ForecasterRecursiveModel":
        """Create a model instance with defaults drawn from a config object.

        Extracts every ``__init__`` parameter that exists as a config
        attribute, translating the two known name mismatches
        (``API_COUNTRY_CODE`` → ``country_code``,
        ``end_train_default`` → ``end_dev``).  Caller-supplied
        *overrides* take precedence over config values.

        Args:
            iteration: Current iteration index.
            config: A config object (e.g. ``ConfigMulti`` or ``ConfigEntsoe``)
                whose attributes overlap with the model's ``__init__`` kwargs.
            **overrides: Explicit keyword arguments that override config values.

        Returns:
            An instance of *cls* (or the calling subclass).

        Examples:
            >>> from spotforecast2_safe.manager.configurator.config_multi import ConfigMulti
            >>> from spotforecast2_safe.manager.models.forecaster_recursive_model import (
            ...     ForecasterRecursiveModel,
            ... )
            >>> cfg = ConfigMulti(country_code="FR", predict_size=48)
            >>> model = ForecasterRecursiveModel.from_config(iteration=1, config=cfg)
            >>> model.predict_size
            48
        """
        import inspect

        # Discover which kwargs __init__ accepts across the full MRO.
        # Subclass __init__ often uses **kwargs to forward to super(),
        # so we walk every class in the chain to collect all explicit params.
        init_params: set = set()
        for klass in cls.__mro__:
            if klass is object:
                continue
            init_fn = klass.__dict__.get("__init__")
            if init_fn is None:
                continue
            sig = inspect.signature(init_fn)
            for name, p in sig.parameters.items():
                if name == "self":
                    continue
                if p.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ):
                    init_params.add(name)
        init_params.discard("iteration")  # handled as a positional arg

        kwargs: Dict[str, Any] = {}

        # 1. Translated names (config attr → model kwarg)
        for cfg_attr, kwarg_name in cls._CONFIG_ATTR_MAP.items():
            if hasattr(config, cfg_attr) and kwarg_name in init_params:
                kwargs[kwarg_name] = getattr(config, cfg_attr)

        # 2. Identically named attributes
        for param_name in init_params:
            if param_name not in kwargs and hasattr(config, param_name):
                kwargs[param_name] = getattr(config, param_name)

        # 3. Caller overrides win
        kwargs.update(overrides)

        return cls(iteration=iteration, **kwargs)

    def get_params(self, deep: bool = True) -> Dict[str, object]:
        """
        Get parameters for this forecaster model.

        Collects wrapper-level parameters (``iteration``, ``end_dev``,
        ``train_size``, ``random_state``, ``predict_size``, ``refit_size``,
        ``name``) and, when a forecaster is attached, delegates to
        :meth:`ForecasterRecursive.get_params` for forecaster-level
        parameters (``estimator``, ``lags``, ``window_features``, etc.).

        Args:
            deep: If True, will return the parameters for this forecaster model and
                contained sub-objects that are estimators.

        Returns:
            params: Dictionary of parameter names mapped to their values.

        Examples:
            >>> from spotforecast2_safe.manager.models.forecaster_recursive_model import (
            ...     ForecasterRecursiveModel,
            ... )
            >>> model = ForecasterRecursiveModel(iteration=0)
            >>> p = model.get_params(deep=False)
            >>> p["iteration"]
            0
            >>> p["name"]
            'base'
            >>> p["predict_size"]
            24
            >>> "forecaster" not in p  # forecaster is None
            True

            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> model2 = ForecasterRecursiveModel(iteration=1)
            >>> model2.forecaster = ForecasterRecursive(
            ...     estimator=LinearRegression(), lags=3
            ... )
            >>> p2 = model2.get_params(deep=False)
            >>> len(p2["forecaster__lags"])
            3
            >>> isinstance(p2["forecaster__estimator"], LinearRegression)
            True

            >>> p3 = model2.get_params(deep=True)
            >>> "forecaster__estimator__fit_intercept" in p3
            True
        """
        # Wrapper-level parameters
        params: Dict[str, object] = {}
        for key in [
            "iteration",
            "end_dev",
            "train_size",
            "random_state",
            "predict_size",
            "refit_size",
            "name",
        ]:
            if hasattr(self, key):
                params[key] = getattr(self, key)

        # Delegate to ForecasterRecursive.get_params when available
        if self.forecaster is not None:
            forecaster_params = self.forecaster.get_params(deep=deep)
            for key, value in forecaster_params.items():
                params[f"forecaster__{key}"] = value

        return params

    def set_params(
        self, params: Dict[str, object] = None, **kwargs: object
    ) -> "ForecasterRecursiveModel":
        """
        Set the parameters of this forecaster model.

        Wrapper-level keys (``iteration``, ``name``, ``predict_size``, …)
        are set directly on the model.  Keys prefixed with ``forecaster__``
        are forwarded to :meth:`ForecasterRecursive.set_params`.

        Args:
            params: Optional dictionary of parameter names mapped to their
                new values.  If provided, these parameters are set first.
            **kwargs: Additional parameter names mapped to their new values.
                Parameters can target the wrapper (e.g. ``name="new"``),
                the forecaster (e.g. ``forecaster__lags=5``), or the
                estimator inside the forecaster
                (e.g. ``forecaster__estimator__fit_intercept=False``).

        Returns:
            ForecasterRecursiveModel: The model instance with updated
                parameters (supports method chaining).

        Examples:
            Setting wrapper-level parameters:

            >>> from spotforecast2_safe.manager.models.forecaster_recursive_model import (
            ...     ForecasterRecursiveModel,
            ... )
            >>> model = ForecasterRecursiveModel(iteration=0)
            >>> _ = model.set_params(name="updated", predict_size=48)
            >>> model.name
            'updated'
            >>> model.predict_size
            48

            Setting forecaster-level parameters on an attached forecaster:

            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> model2 = ForecasterRecursiveModel(iteration=1)
            >>> model2.forecaster = ForecasterRecursive(
            ...     estimator=LinearRegression(), lags=3
            ... )
            >>> _ = model2.set_params(
            ...     params={"forecaster__estimator__fit_intercept": False}
            ... )
            >>> model2.forecaster.estimator.fit_intercept
            False
        """
        # Merge params dict and kwargs
        all_params: Dict[str, object] = {}
        if params is not None:
            all_params.update(params)
        all_params.update(kwargs)

        if not all_params:
            return self

        # Separate forecaster-level from wrapper-level params
        forecaster_params: Dict[str, object] = {}
        for key, value in all_params.items():
            if key.startswith("forecaster__"):
                # Strip the 'forecaster__' prefix before forwarding
                sub_key = key[len("forecaster__") :]
                forecaster_params[sub_key] = value
            else:
                setattr(self, key, value)

        # Delegate forecaster-level params
        if forecaster_params and self.forecaster is not None:
            if hasattr(self.forecaster, "set_params"):
                self.forecaster.set_params(**forecaster_params)
            else:
                for param_name, value in forecaster_params.items():
                    setattr(self.forecaster, param_name, value)

        return self

    def tune(self) -> None:
        """Simulate hyperparameter tuning.

        In ``spotforecast2-safe`` this is a simulated stub that marks the
        model as tuned without performing an actual Bayesian search.  A
        full implementation using ``bayesian_search_forecaster`` will be
        provided in the ``spotforecast2`` package.

        #TODO: Implement real Bayesian search in spotforecast2
        """
        logger.info("Tuning %s model (simulated)...", self.name)
        self.is_tuned = True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_to_file(
        self,
        model_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Serialize the model to disk via :func:`joblib.dump`.

        Args:
            model_dir: Directory for the model file.  If *None*,
                defaults to :func:`get_cache_home`.

        Examples:
            >>> import tempfile
            >>> from spotforecast2_safe.manager.models.forecaster_recursive_model import (
            ...     ForecasterRecursiveModel,
            ... )
            >>> model = ForecasterRecursiveModel(iteration=0, name="test")
            >>> with tempfile.TemporaryDirectory() as tmpdir:
            ...     model.save_to_file(model_dir=tmpdir)
            ...     import os; any("test_forecaster_0" in f for f in os.listdir(tmpdir))
            True
        """
        from spotforecast2_safe.manager.trainer import get_path_model

        path_to_save = get_path_model(self.name, self.iteration, model_dir=model_dir)
        path_to_save.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Saving %s Forecaster %d to %s.",
            self.name.upper(),
            self.iteration,
            path_to_save,
        )
        dump(self, path_to_save, compress=3)

    # ------------------------------------------------------------------
    # Cross-validation helpers
    # ------------------------------------------------------------------

    def _build_cv(
        self,
        train_size: int,
        fixed_train_size: bool = False,
        refit: Union[int, bool] = False,
    ) -> TimeSeriesFold:
        """Build cross-validation time folds for tuning and backtesting.

        Args:
            train_size: Number of observations in the initial training set.
            fixed_train_size: Whether to keep the training window fixed.
            refit: Refit frequency (``False`` = no refit).

        Returns:
            TimeSeriesFold: Configured fold object.
        """
        return TimeSeriesFold(
            steps=self.predict_size,
            refit=refit,
            initial_train_size=train_size,
            fixed_train_size=fixed_train_size,
            gap=0,
            skip_folds=None,
            allow_incomplete_fold=True,
        )

    def _get_init_train(
        self, min_val: pd.Timestamp, end_val: pd.Timestamp
    ) -> pd.Timestamp:
        """Return the start of the training period.

        If ``train_size`` is *None*, uses the earliest available timestamp.
        Otherwise computes ``end_val - train_size`` and caps at ``min_val``.

        Args:
            min_val: Earliest timestamp in the dataset.
            end_val: End of the training/dev period.

        Returns:
            pd.Timestamp: Start of the training window.

        Examples:
            >>> import pandas as pd
            >>> from spotforecast2_safe.manager.models.forecaster_recursive_model import (
            ...     ForecasterRecursiveModel,
            ... )
            >>> model = ForecasterRecursiveModel(iteration=0)
            >>> start = pd.Timestamp("2020-01-01", tz="UTC")
            >>> end = pd.Timestamp("2025-12-31", tz="UTC")
            >>> model._get_init_train(start, end) == start
            True
            >>> model.train_size = pd.Timedelta(days=365)
            >>> model._get_init_train(start, end)
            Timestamp('2024-12-31 00:00:00+0000', tz='UTC')
        """
        if self.train_size is None:
            return min_val
        init_train = end_val - self.train_size
        return max(min_val, init_train)

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def fit(self, y: pd.Series, exog: Optional[pd.DataFrame] = None) -> None:
        """
        Fit the underlying forecaster.

        Args:
            y: Target time series.
            exog: Optional exogenous features.

        Raises:
            ValueError: If the forecaster has not been initialized.

        Examples:
            >>> import pandas as pd
            >>> import numpy as np
            >>> from spotforecast2_safe.manager.models.forecaster_recursive_model import ForecasterRecursiveModel
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from sklearn.linear_model import LinearRegression
            >>>
            >>> # Example 1: Basic usage with pd.Series
            >>> model = ForecasterRecursiveModel(iteration=0)
            >>> model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
            >>> y = pd.Series(
            ...     np.random.rand(10),
            ...     index=pd.date_range("2023-01-01", periods=10, freq="h")
            ... )
            >>> model.fit(y=y)
            >>> model.forecaster.is_fitted
            True
            >>>
            >>> # Example 2: Usage with exogenous variables
            >>> model_exog = ForecasterRecursiveModel(iteration=0)
            >>> model_exog.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
            >>> exog = pd.DataFrame(
            ...     np.random.rand(10, 2),
            ...     index=y.index,
            ...     columns=["exog_1", "exog_2"]
            ... )
            >>> model_exog.fit(y=y, exog=exog)
            >>> model_exog.forecaster.is_fitted
            True
        """
        if self.forecaster is None:
            raise ValueError("Forecaster not initialized")

        self.forecaster.fit(y=y, exog=exog)

    def fit_with_best(self) -> None:
        """Fit the forecaster using the recorded best hyperparameters.

        After tuning (or manually setting ``best_params`` and ``best_lags``),
        this method loads the data, sets the optimal parameters/lags, and
        fits the forecaster on the full training + dev set up to
        ``end_dev``.

        Raises:
            ValueError: If the forecaster has not been initialized.
        """
        logger.info(
            "Fitting %s Forecaster %d for predictions",
            self.name.upper(),
            self.iteration,
        )

        # Check if model is tuned
        if self.best_params is None or self.best_lags is None:
            logger.warning("Model is not tuned! Starting tuning first...")
            self.tune()

        if self.forecaster is None:
            raise ValueError("Forecaster not initialized")

        # Load data
        y = load_timeseries()
        y = LinearlyInterpolateTS().fit_transform(y)
        X = self.preprocessor.build(start_date=y.index.min(), end_date=self.end_dev)

        # Apply best params and lags
        logger.info("Setting parameters...")
        if self.best_params is not None:
            self.forecaster.set_params(**self.best_params)
        logger.info("Setting lags...")
        if self.best_lags is not None and hasattr(self.forecaster, "set_lags"):
            self.forecaster.set_lags(self.best_lags)

        # Determine training window
        start_train = self._get_init_train(y.index.min(), self.end_dev)

        # Fit
        logger.info(
            "Fitting over %s to %s ...",
            start_train,
            self.end_dev,
        )
        self.forecaster.fit(
            y.loc[start_train : self.end_dev],
            exog=X.loc[start_train : self.end_dev],
        )
        logger.info("Training done!")

    # ------------------------------------------------------------------
    # Training data access
    # ------------------------------------------------------------------

    def _get_training_data(
        self,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Create lag/window training matrices via the forecaster.

        Returns:
            Tuple[pd.DataFrame, pd.Series]: ``(X_train, y_train)``.

        Raises:
            ValueError: If the forecaster has not been initialized.
        """
        if self.forecaster is None:
            raise ValueError("Forecaster not initialized")

        y = load_timeseries()
        y = LinearlyInterpolateTS().fit_transform(y)
        X = self.preprocessor.build(start_date=y.index.min(), end_date=self.end_dev)

        start_train = self._get_init_train(y.index.min(), self.end_dev)

        X_train, y_train = self.forecaster.create_train_X_y(
            y=y.loc[start_train : self.end_dev],
            exog=X.loc[start_train : self.end_dev],
        )
        return X_train, y_train

    # ------------------------------------------------------------------
    # Backtesting
    # ------------------------------------------------------------------

    def backtest(self) -> pd.DataFrame:
        """Back-test the forecaster on the test data.

        Returns:
            pd.DataFrame: Backtesting metric values.

        Raises:
            ValueError: If the forecaster has not been initialized.
        """
        logger.info(
            "Backtesting %s Forecaster %d",
            self.name.upper(),
            self.iteration,
        )

        if self.forecaster is None:
            raise ValueError("Forecaster not initialized")

        y = load_timeseries()
        y = LinearlyInterpolateTS().fit_transform(y)
        X = self.preprocessor.build(start_date=y.index.min(), end_date=y.index.max())

        self.fit_with_best()

        start_train = self._get_init_train(y.index.min(), self.end_dev)
        fixed_train_size = self.train_size is not None
        end_train = self.end_dev - pd.Timedelta(
            hours=self.predict_size * self.refit_size
        )
        length_training = len(y.loc[start_train:end_train])

        metrics, _ = backtesting_forecaster(
            self.forecaster,
            y,
            cv=self._build_cv(
                train_size=length_training,
                fixed_train_size=fixed_train_size,
                refit=False,
            ),
            metric=self.metrics,
            exog=X,
        )
        logger.info("Backtesting results: %s.", metrics.to_dict())
        return metrics

    # ------------------------------------------------------------------
    # Prediction & error methods
    # ------------------------------------------------------------------

    def predict(
        self,
        delta_predict: Optional[pd.Timedelta] = None,
    ) -> Tuple[dict, Tuple[pd.Series, pd.Series]]:
        """Generate predictions and compute error metrics.

        Args:
            delta_predict: Optional time horizon to predict.  If *None*
                or if it exceeds the available data, predicts to the end
                of the dataset.

        Returns:
            Tuple[dict, Tuple[pd.Series, pd.Series]]:
                ``(metrics, (y_actual, y_predicted))``.

        Raises:
            ValueError: If the forecaster has not been initialized.
        """
        logger.info(
            "Making predictions with %s Forecaster %d",
            self.name.upper(),
            self.iteration,
        )

        if not self.is_tuned:
            self.tune()

        if self.forecaster is None:
            raise ValueError("Forecaster not initialized")

        y = load_timeseries()
        y = LinearlyInterpolateTS().fit_transform(y)
        X = self.preprocessor.build(start_date=y.index.min(), end_date=y.index.max())

        start_future = self.end_dev + pd.Timedelta(hours=1)

        if delta_predict is None or delta_predict > y.index.max() - start_future:
            end_predict = y.index.max()
        else:
            end_predict = start_future + delta_predict

        idx_future = pd.date_range(start=start_future, end=end_predict, freq="h")

        n_steps = len(idx_future)
        if n_steps > self.refit_size * self.predict_size:
            logger.info(
                "Predicting %d hours (about %d days), retraining might be necessary!",
                n_steps,
                n_steps // 24,
            )

        y_predicted = self.forecaster.predict(steps=n_steps, exog=X.loc[idx_future])

        metrics_out: dict = {}
        metrics_out["mape"] = mean_absolute_percentage_error(
            y.loc[idx_future], y_predicted
        )
        metrics_out["mae"] = mean_absolute_error(y.loc[idx_future], y_predicted)
        logger.info(
            "MAPE: %.2f, MAE: %.2f",
            metrics_out["mape"],
            metrics_out["mae"],
        )
        return metrics_out, (y.loc[idx_future], y_predicted)

    def get_error_training(
        self,
    ) -> Tuple[dict, Tuple[pd.Series, pd.Series]]:
        """Compute in-sample error on the training data.

        Returns:
            Tuple[dict, Tuple[pd.Series, pd.Series]]:
                ``(metrics, (y_train, y_train_pred))``.
        """
        logger.info(
            "Obtaining training estimation with Forecaster %d",
            self.iteration,
        )

        if not self.is_tuned:
            self.tune()

        if self.forecaster is None:
            raise ValueError("Forecaster not initialized")

        X_train, y_train = self._get_training_data()
        y_train_pred = pd.Series(
            self.forecaster.estimator.predict(X_train),
            index=y_train.index,
        )

        metrics_out: dict = {}
        metrics_out["mape"] = mean_absolute_percentage_error(y_train, y_train_pred)
        metrics_out["mae"] = mean_absolute_error(y_train, y_train_pred)
        logger.info(
            "Train MAPE: %.2f, MAE: %.2f",
            metrics_out["mape"],
            metrics_out["mae"],
        )
        return metrics_out, (y_train, y_train_pred)

    def get_error_forecast(
        self,
        delta_predict: Optional[pd.Timedelta] = None,
    ) -> Tuple[dict, Tuple[pd.Series, pd.Series]]:
        """Compute the error of the ENTSO-E benchmark forecast.

        Args:
            delta_predict: Optional prediction horizon.

        Returns:
            Tuple[dict, Tuple[pd.Series, pd.Series]]:
                ``(metrics, (y_actual, y_forecast))``.
        """
        y = load_timeseries()
        y = LinearlyInterpolateTS().fit_transform(y)

        y_forecast = load_timeseries_forecast()
        y_forecast = LinearlyInterpolateTS().fit_transform(y_forecast)

        start_future = self.end_dev + pd.Timedelta(hours=1)

        if delta_predict is None or delta_predict > y.index.max() - start_future:
            end_predict = y.index.max()
        else:
            end_predict = start_future + delta_predict

        idx_future = pd.date_range(start=start_future, end=end_predict, freq="h")

        metrics_out: dict = {}
        metrics_out["mape"] = mean_absolute_percentage_error(
            y.loc[idx_future], y_forecast.loc[idx_future]
        )
        metrics_out["mae"] = mean_absolute_error(
            y.loc[idx_future], y_forecast.loc[idx_future]
        )
        return metrics_out, (y.loc[idx_future], y_forecast.loc[idx_future])

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def get_feature_importance(self) -> Optional[pd.DataFrame]:
        """Return feature importances from the underlying estimator.

        Only supported for tree-based models (``xgb``, ``lgbm``).

        Returns:
            pd.DataFrame or None: Feature importances, or *None* if the
                model does not support this operation.
        """
        if self.name not in ["xgb", "lgbm"]:
            logger.error("Regressor does not support feature importance!")
            return None
        if self.forecaster is None:
            raise ValueError("Forecaster not initialized")
        return self.forecaster.get_feature_importances()

    def get_global_shap_feature_importance(self, frac: float = 0.1) -> pd.Series:
        """Return global SHAP-based feature importances.

        .. note::

            This is a stub.  The full implementation using
            ``shap.TreeExplainer`` will be provided in the
            ``spotforecast2`` package.

        #TODO: Implement shap feature importance in spotforecast2

        Args:
            frac: Fraction of training data to use for SHAP values.

        Returns:
            pd.Series: Empty series (stub).
        """
        logger.warning(
            "get_global_shap_feature_importance is a stub in "
            "spotforecast2-safe. Use spotforecast2 for the full "
            "implementation."
        )
        return pd.Series(dtype=float)

    # ------------------------------------------------------------------
    # Prediction packaging (delegates to predict / get_error_*)
    # ------------------------------------------------------------------

    def package_prediction(self, predict_size: Optional[int] = None) -> Dict[str, Any]:
        """Package predictions, training errors, and benchmarks for the UI.

        This is the main entry-point used by the application layer.
        It delegates to :meth:`predict`, :meth:`get_error_training`,
        and :meth:`get_error_forecast`.

        Args:
            predict_size: Optional override for the prediction horizon.

        Returns:
            Dict[str, Any]: A result package containing actual values,
                predictions, and calculated metrics (MAE, MAPE).

        Examples:
            >>> import os
            >>> import tempfile
            >>> import pandas as pd
            >>> from pathlib import Path
            >>> from spotforecast2_safe.manager.models.forecaster_recursive_lgbm import ForecasterRecursiveLGBM
            >>> from spotforecast2_safe.data.fetch_data import get_package_data_home
            >>>
            >>> # Setup temporary data environment
            >>> tmp_dir = tempfile.mkdtemp()
            >>> os.environ["SPOTFORECAST2_DATA"] = tmp_dir
            >>> data_path = Path(tmp_dir) / "interim"
            >>> data_path.mkdir(parents=True)
            >>>
            >>> # Load demo data and rename columns to match expectations
            >>> demo_path = get_package_data_home() / "demo01.csv"
            >>> df = pd.read_csv(demo_path)
            >>> df = df.rename(columns={
            ...     "Time": "Time (UTC)",
            ...     "Actual": "Actual Load",
            ...     "Forecast": "Forecasted Load"
            ... })
            >>> df.to_csv(data_path / "energy_load.csv", index=False)
            >>>
            >>> # Initialize model — override forecaster for small demo data
            >>> from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
            >>> from lightgbm import LGBMRegressor
            >>> model = ForecasterRecursiveLGBM(iteration=0, end_dev="2022-01-05 00:00+00:00")
            >>> model.forecaster = ForecasterRecursive(
            ...     estimator=LGBMRegressor(n_jobs=-1, verbose=-1, random_state=123456789),
            ...     lags=12,
            ... )
            >>> result = model.package_prediction(predict_size=24)
            >>>
            >>> # Validate output
            >>> "train_actual" in result and "future_pred" in result
            True
            >>>
            >>> # Cleanup
            >>> import shutil
            >>> shutil.rmtree(tmp_dir)
            >>> del os.environ["SPOTFORECAST2_DATA"]
        """
        if self.forecaster is None:
            logger.error("Forecaster not initialized")
            return {}

        try:
            y = load_timeseries()
            y = LinearlyInterpolateTS().fit_transform(y)

            # Benchmark (back-test mode only; ignored in genuine-future mode)
            try:
                future_forecast_series = load_timeseries_forecast()
                future_forecast_series = LinearlyInterpolateTS().fit_transform(
                    future_forecast_series
                )
            except (FileNotFoundError, KeyError):
                future_forecast_series = None

            # Train / test split
            y_train = y.loc[: self.end_dev]
            y_test = y.loc[self.end_dev :]

            if predict_size is None:
                predict_size = self.predict_size

            predict_hours = predict_size * self.refit_size
            logger.info(
                "Prediction window: predict_size=%d, refit_size=%d, "
                "predict_hours=%d, available_test_hours=%d",
                predict_size,
                self.refit_size,
                predict_hours,
                len(y_test),
            )

            # Fit on training data (shared by both modes)
            self.forecaster.fit(y=y_train)

            # In-sample predictions (shared by both modes)
            train_pred = self.forecaster.predict(steps=len(y_train))
            train_pred.index = y_train.index[-len(train_pred) :]
            y_train_aligned = y_train.loc[train_pred.index]

            metrics_train = {
                "mae": mean_absolute_error(y_train_aligned, train_pred),
                "mape": mean_absolute_percentage_error(y_train_aligned, train_pred),
            }

            # --- Mode selection ---
            # Genuine-future mode: end_dev is at or beyond the last observed timestamp,
            # so there is no held-out test data. Predict `predict_hours` steps into
            # the true future; skip future metrics (no ground truth).
            # Back-test mode: a held-out test period exists; compute future metrics.
            genuine_future_mode = len(y_test) < predict_hours

            if genuine_future_mode:
                logger.info(
                    "Genuine-future mode: %d test hours available (< %d required). "
                    "Predicting %d steps beyond end_dev with no ground truth.",
                    len(y_test),
                    predict_hours,
                    predict_hours,
                )
                future_pred = self.forecaster.predict(steps=predict_hours)

                result: Dict[str, Any] = {
                    "train_actual": y_train_aligned,
                    "future_actual": pd.Series(dtype="float64"),  # no ground truth
                    "train_pred": train_pred,
                    "future_pred": future_pred,
                    "metrics_train": metrics_train,
                }

            else:
                # Back-test mode: existing behaviour preserved exactly.
                if len(y_test) > predict_hours:
                    y_test = y_test.iloc[:predict_hours]
                    logger.info("Limited test data to %d hours", predict_hours)
                else:
                    logger.info("Using all %d available test hours", len(y_test))

                future_pred = self.forecaster.predict(steps=len(y_test))
                future_pred.index = y_test.index[: len(future_pred)]
                y_test_aligned = y_test.loc[future_pred.index]

                metrics_future = {
                    "mae": mean_absolute_error(y_test_aligned, future_pred),
                    "mape": mean_absolute_percentage_error(y_test_aligned, future_pred),
                }

                # 24 h window metrics
                f_24h_pred = future_pred.iloc[: min(24, len(future_pred))]
                f_24h_actual = y_test_aligned.iloc[: min(24, len(y_test_aligned))]
                metrics_future_24h = {
                    "mae": mean_absolute_error(f_24h_actual, f_24h_pred),
                    "mape": mean_absolute_percentage_error(f_24h_actual, f_24h_pred),
                }

                logger.info(
                    "Metrics computed: Full horizon (MAE=%.2f, MAPE=%.4f), "
                    "24h window (MAE=%.2f, MAPE=%.4f)",
                    metrics_future["mae"],
                    metrics_future["mape"],
                    metrics_future_24h["mae"],
                    metrics_future_24h["mape"],
                )

                result = {
                    "train_actual": y_train_aligned,
                    "future_actual": y_test_aligned,
                    "train_pred": train_pred,
                    "future_pred": future_pred,
                    "metrics_train": metrics_train,
                    "metrics_future": metrics_future,
                    "metrics_future_one_day": metrics_future_24h,
                }

                # Add benchmark if available
                if future_forecast_series is not None:
                    forecast_test = future_forecast_series.loc[y_test_aligned.index]
                    result["future_forecast"] = forecast_test
                    result["metrics_forecast"] = {
                        "mae": mean_absolute_error(y_test_aligned, forecast_test),
                        "mape": mean_absolute_percentage_error(
                            y_test_aligned, forecast_test
                        ),
                    }

            return result

        except Exception as e:
            logger.error("Error generating prediction package: %s", e, exc_info=True)
            return {}
