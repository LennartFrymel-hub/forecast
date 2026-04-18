# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Recursive forecaster model wrappers for different estimators."""

import logging
from typing import Any

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive

from .forecaster_recursive_model import ForecasterRecursiveModel

# Try to import XGBoost
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

logger = logging.getLogger(__name__)


class ForecasterRecursiveXGB(ForecasterRecursiveModel):
    """
    ForecasterRecursive specialization using XGBoost.

    Attributes:
        forecaster: The XGBoost forecaster.
        name: The name of the forecaster.
    """

    def __init__(self, iteration: int, lags: int = 12, **kwargs: Any):
        """
        Initialize the XGBoost Recursive Forecaster.

        Args:
            iteration: Current iteration index.
            lags: Number of autoregressive lags.
            **kwargs: Passed to ForecasterRecursiveModel.

        Returns:
            None

        Raises:
            ImportError: If xgboost is not installed.

        Examples:
            >>> import pandas as pd
            >>> from sklearn.linear_model import LinearRegression
            >>> from spotforecast2_safe.manager.models.forecaster_recursive_model import ForecasterRecursiveModel
            >>>
            >>> # Initialization
            >>> model = ForecasterRecursiveXGB(iteration=0)
            >>> model.name
            'xgb'
            >>> # If XGBoost is not available, we can still test the interface
            >>> # using a mock forecaster or checking attributes
            >>> isinstance(model, ForecasterRecursiveModel)
            True
        """
        super().__init__(iteration, name="xgb", **kwargs)
        if XGBRegressor is not None:
            self.forecaster = ForecasterRecursive(
                estimator=XGBRegressor(n_jobs=-1, random_state=self.random_state),
                lags=lags,
            )
        else:
            logger.warning(
                "XGBoost not installed. This model will fail during fit/predict."
            )
