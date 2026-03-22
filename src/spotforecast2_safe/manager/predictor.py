# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Module for managing model predictions.

Public helpers
--------------
- :func:`build_prediction_package` — assemble a standardized prediction dict
  from a fitted forecaster (in-sample fit + future forecast + metrics).
- :func:`get_model_prediction` — load the latest persisted model and call its
  ``package_prediction()`` method.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

from spotforecast2_safe.manager.trainer import get_last_model

logger = logging.getLogger(__name__)


def build_prediction_package(
    forecaster: Any,
    target: str,
    y_train: pd.Series,
    predict_size: int,
    exog_train: Optional[pd.DataFrame] = None,
    exog_future: Optional[pd.DataFrame] = None,
    df_test: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Build a prediction package compatible with PredictionFigure.

    Computes true in-sample predictions via the fitted regressor and generates
    a genuine future forecast of ``predict_size`` steps. If ``df_test`` is
    supplied, ground-truth test values are injected and future metrics
    (MAE, MAPE) are computed.

    Args:
        forecaster: A fitted skforecast ``ForecasterRecursive`` (or compatible)
            instance with ``create_train_X_y``, ``regressor``, and ``predict``
            methods.
        target: Column name of the target series. Used to look up ground-truth
            values inside ``df_test`` when provided.
        y_train: Training time series indexed by a timezone-aware
            ``DatetimeIndex``.
        predict_size: Number of future steps to forecast.
        exog_train: Exogenous feature DataFrame aligned with ``y_train``.
            Pass ``None`` when no exogenous features are used.
        exog_future: Exogenous feature DataFrame covering the forecast horizon.
            Pass ``None`` when no exogenous features are used.
        df_test: Optional test DataFrame that must contain a ``"DateTime"``
            column and a column named ``target``. When supplied, the matching
            ground-truth slice is injected into the returned package and future
            metrics are computed.

    Returns:
        A dictionary with the following keys:

        - **train_actual** (``pd.Series``) — observed training values aligned
            to the in-sample prediction index (lags consumed from the start).
        - **train_pred** (``pd.Series``) — in-sample fitted values from the
            underlying regressor.
        - **future_actual** (``pd.Series``) — always an empty ``float64``
            Series; the field exists for interface compatibility with
            ``PredictionFigure``.
        - **future_pred** (``pd.Series``) — ``predict_size``-step-ahead
            forecast.
        - **metrics_train** (``dict``) — ``{"mae": float, "mape": float}``
            computed on the aligned in-sample window.
        - **metrics_future** (``dict``) — ``{"mae": float, "mape": float}``
            computed against test ground truth, or ``{}`` when unavailable.
        - **metrics_future_one_day** (``dict``) — reserved for downstream
            one-day metrics; always ``{}``.
        - **validation_passed** (``bool``) — always ``True``; field reserved
            for downstream safety checks.
        - **test_actual** (``pd.Series``, optional) — present only when
            ``df_test`` contains matching rows for the forecast horizon.

    Examples:

        ```{python}
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import Ridge
        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        from spotforecast2_safe.manager.predictor import build_prediction_package

        rng = np.random.default_rng(42)
        idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
        y_train = pd.Series(rng.normal(100, 10, 200), index=idx, name="load")

        forecaster = ForecasterRecursive(estimator=Ridge(), lags=24)
        forecaster.fit(y=y_train)

        pkg = build_prediction_package(
            forecaster=forecaster,
            target="load",
            y_train=y_train,
            predict_size=24,
        )
        print(f"Keys: {sorted(pkg.keys())}")
        print(f"Future predictions: {len(pkg['future_pred'])} steps")
        print(f"Validation passed: {pkg['validation_passed']}")
        ```

        ```{python}
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import Ridge
        from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
        from spotforecast2_safe.manager.predictor import build_prediction_package

        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=200, freq="h", tz="UTC")
        y_train = pd.Series(rng.normal(50, 5, 200), index=idx, name="power")

        forecaster = ForecasterRecursive(estimator=Ridge(), lags=24)
        forecaster.fit(y=y_train)

        # Test DataFrame covering the 24-hour forecast horizon
        df_test = pd.DataFrame({
            "DateTime": pd.date_range("2024-01-09 08:00", periods=24, freq="h"),
            "power": rng.normal(50, 5, 24),
        })

        pkg = build_prediction_package(
            forecaster=forecaster,
            target="power",
            y_train=y_train,
            predict_size=24,
            df_test=df_test,
        )
        print(f"test_actual present: {'test_actual' in pkg}")
        print(f"metrics_future keys: {list(pkg['metrics_future'].keys())}")
        print(f"MAE on future: {pkg['metrics_future']['mae']:.4f}")
        ```
    """
    # --- In-sample predictions (true fit, not forward predictions) ---
    X_train_matrix, y_train_internal = forecaster.create_train_X_y(
        y=y_train,
        exog=exog_train,
    )
    train_pred_values = forecaster.regressor.predict(X_train_matrix)
    train_pred = pd.Series(train_pred_values, index=y_train_internal.index)
    y_train_aligned = y_train.loc[train_pred.index]

    metrics_train = {
        "mae": float(mean_absolute_error(y_train_aligned, train_pred)),
        "mape": float(mean_absolute_percentage_error(y_train_aligned, train_pred)),
    }

    # --- Future predictions ---
    future_pred = forecaster.predict(steps=predict_size, exog=exog_future)

    pred_pkg: Dict[str, Any] = {
        "train_actual": y_train_aligned,
        "train_pred": train_pred,
        "future_actual": pd.Series(dtype="float64"),
        "future_pred": future_pred,
        "metrics_train": metrics_train,
        "metrics_future": {},
        "metrics_future_one_day": {},
        "validation_passed": True,
    }

    # Inject test ground truth if available
    if df_test is not None and target in df_test.columns:
        ts = df_test.set_index("DateTime")[target]
        if ts.index.tzinfo is None:
            ts.index = ts.index.tz_localize("UTC")
        test_actual = ts.reindex(future_pred.index).dropna()
        pred_pkg["test_actual"] = test_actual
        if len(test_actual) > 0:
            pred_pkg["metrics_future"] = {
                "mae": float(
                    mean_absolute_error(test_actual, future_pred.loc[test_actual.index])
                ),
                "mape": float(
                    mean_absolute_percentage_error(
                        test_actual, future_pred.loc[test_actual.index]
                    )
                ),
            }

    return pred_pkg


def get_model_prediction(
    model_name: str,
    model_dir: Optional[Union[str, Path]] = None,
    predict_size: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Get the prediction package from the latest trained model.
    This function retrieves the latest iteration of a specified model from the
    cache and calls its `package_prediction` method to obtain a comprehensive
    set of predictions and metrics.

    Args:
        model_name: Name of the model to use (e.g., 'lgbm', 'xgb').
        model_dir: Directory where models are stored. If None, defaults to
            the library's cache home.
        predict_size: Optional override for the prediction horizon.

    Returns:
        A dictionary containing predictions and metrics if a model is found and
        successfully executes `package_prediction`. Returns None otherwise.

    Notes:
         `predict_size` is accepted by `get_model_prediction()` but only has effect if the concrete model's `package_prediction()` accepts it.
         The original `ForecasterRecursiveModel.package_prediction()` does not — so this parameter is currently forward-looking API design, not yet wired end-to-end.

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from spotforecast2_safe.manager.predictor import get_model_prediction
        >>> from joblib import dump
        >>>
        >>> # Example 1: No model found scenario
        >>> with tempfile.TemporaryDirectory() as tmpdir:
        ...     result = get_model_prediction('lgbm', model_dir=tmpdir)
        ...     print(f"Result when no model exists: {result}")
        Result when no model exists: None
        >>>
        >>> # Example 2: Model found but no package_prediction method
        >>> class SimpleModel:  # doctest: +SKIP
        ...     '''Simple model without package_prediction method'''
        ...     def __init__(self):
        ...         self.name = 'simple'
        >>>
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     model_dir = Path(tmpdir)
        ...     simple_model = SimpleModel()
        ...     dump(simple_model, model_dir / "test_forecaster_1.joblib")
        ...     result = get_model_prediction('test', model_dir=model_dir)
        ...     print(f"Result without package_prediction: {result}")
        Result without package_prediction: None
        >>>
        >>> # Example 3: Successful prediction package retrieval
        >>> class ForecastModel:  # doctest: +SKIP
        ...     '''Model with package_prediction method'''
        ...     def __init__(self):
        ...         self.name = 'xgb'
        ...     def package_prediction(self):
        ...         return {
        ...             'predictions': [1.0, 2.0, 3.0],
        ...             'metrics': {'mse': 0.05, 'mae': 0.02}
        ...         }
        >>>
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     model_dir = Path(tmpdir)
        ...     forecast_model = ForecastModel()
        ...     dump(forecast_model, model_dir / "xgb_forecaster_1.joblib")
        ...     result = get_model_prediction('xgb', model_dir=model_dir)
        ...     print(f"Predictions available: {'predictions' in result}")
        ...     print(f"Metrics available: {'metrics' in result}")
        Predictions available: True
        Metrics available: True
        >>>
        >>> # Example 4: Safety-critical - verify prediction integrity
        >>> class SafetyModel:  # doctest: +SKIP
        ...     '''Safety model with validation'''
        ...     def __init__(self):
        ...         self.name = 'safety_forecaster'
        ...     def package_prediction(self):
        ...         return {
        ...             'predictions': [10.5, 11.2],
        ...             'confidence_intervals': [(10.0, 11.0), (10.8, 11.6)],
        ...             'validation_passed': True
        ...         }
        >>>
        >>> with tempfile.TemporaryDirectory() as tmpdir:  # doctest: +SKIP
        ...     model_dir = Path(tmpdir)
        ...     safety_model = SafetyModel()
        ...     dump(safety_model, model_dir / "safety_forecaster_forecaster_2.joblib")
        ...     pkg = get_model_prediction('safety_forecaster', model_dir=model_dir)
        ...     if pkg:
        ...         print(f"Validation status: {pkg['validation_passed']}")
        Validation status: True
    """
    n_iteration, model = get_last_model(model_name, model_dir)

    if n_iteration < 0 or model is None:
        logger.error(
            "No trained model found for '%s'. Please train a model first.", model_name
        )
        return None

    logger.info(
        "Making predictions using %s model (iteration %d)...",
        model_name,
        n_iteration,
    )

    if not hasattr(model, "package_prediction"):
        logger.error(
            "Model '%s' (iteration %d) does not implement 'package_prediction' method.",
            model_name,
            n_iteration,
        )
        return None

    try:
        prediction_package = model.package_prediction(predict_size=predict_size)
        return prediction_package
    except Exception as e:
        logger.error(
            "Error occurred while generating prediction package for '%s': %s",
            model_name,
            e,
            exc_info=True,
        )
        return None
