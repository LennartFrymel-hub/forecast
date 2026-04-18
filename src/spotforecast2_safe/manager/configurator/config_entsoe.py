# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Configuration for ENTSO-E task pipeline."""

from dataclasses import replace
from typing import Dict, List, Optional

import pandas as pd

from spotforecast2_safe.data import Period


class ConfigEntsoe:
    """Configuration for the ENTSO-E forecasting pipeline.

    This class manages all configuration parameters for the ENTSO-E task,
    including API settings, training/prediction intervals, and feature
    engineering specifications. All parameters can be customized during
    initialization or used with sensible defaults.

    Args:
        api_country_code (str): ISO country code for ENTSO-E API queries.
        periods (Optional[List[Period]]): List of Period objects defining cyclical feature encodings.
        lags_consider (Optional[List[int]]): List of lag values to consider for feature selection.
        train_size (Optional[pd.Timedelta]): Time window for training data.
        end_train_default (str): Default end date for training period (ISO format with timezone).
        delta_val (Optional[pd.Timedelta]): Validation window size.
        predict_size (int): Number of hours to predict ahead.
        refit_size (int): Number of days between model refits.
        random_state (int): Random seed for reproducibility.
        n_hyperparameters_trials (int): Number of trials for hyperparameter optimization.
        data_filename (str): Path to the interim merged data file.

    Attributes:
        API_COUNTRY_CODE (str): ISO country code for API queries.
        periods (List[Period]): Cyclical feature encoding specifications.
        lags_consider (List[int]): Lag values for autoregressive features.
        train_size (pd.Timedelta): Training data window.
        end_train_default (str): Default training end date.
        delta_val (pd.Timedelta): Validation window.
        predict_size (int): Prediction horizon in hours.
        refit_size (int): Refit interval in days.
        random_state (int): Random seed.
        n_hyperparameters_trials (int): Hyperparameter tuning trials.

    Notes:
        The default period configurations use specific `n_periods` to balance resolution and smoothing:
        - **Daily**: `n_periods=12` (24h) provides ~2h resolution, smoothing hourly noise and halving dimensionality.
        - **Weekly**: `n_periods` typically matches range (1:1) to distinguish day-of-week patterns.
        - **Yearly**: `n_periods=12` (365d) provides ~1 month resolution, capturing broad seasonal trends without overfitting.

        See `docs/PERIOD_CONFIGURATION_RATIONALE.md` for a detailed analysis.

    Examples:
        >>> from spotforecast2_safe import Config
        >>> import pandas as pd
        >>>
        >>> # Use default configuration
        >>> config = Config()
        >>> config.API_COUNTRY_CODE
        'DE'
        >>> config.predict_size
        24
        >>> config.random_state
        314159
        >>>
        >>> # Create custom configuration
        >>> custom_config = Config(
        ...     api_country_code='FR',
        ...     predict_size=48,
        ...     random_state=42
        ... )
        >>> custom_config.API_COUNTRY_CODE
        'FR'
        >>> custom_config.predict_size
        48
        >>>
        >>> # Verify training window
        >>> config.train_size == pd.Timedelta(days=3 * 365)
        True
        >>>
        >>> # Check default periods
        >>> len(config.periods)
        5
        >>> config.periods[0].name
        'daily'
    """

    def __init__(
        self,
        api_country_code: str = "DE",
        periods: Optional[List[Period]] = None,
        lags_consider: Optional[List[int]] = None,
        train_size: Optional[pd.Timedelta] = None,
        end_train_default: str = "2025-12-31 00:00+00:00",
        delta_val: Optional[pd.Timedelta] = None,
        predict_size: int = 24,
        refit_size: int = 7,
        random_state: int = 314159,
        n_hyperparameters_trials: int = 20,
        data_filename: str = "interim/energy_load.csv",
    ):
        """Initialize ConfigEntsoe with specified or default parameters."""
        self.API_COUNTRY_CODE = api_country_code

        # Default periods use deliberate n_periods choices:
        # - daily: n_periods=12 for 24 hours (2:1 ratio) provides 2-hour resolution,
        #   balancing detail vs overfitting while reducing dimensionality by 50%
        # - weekly/monthly/quarterly: n_periods matches range_size (1:1 ratio)
        # - yearly: n_periods=12 for 365 days (30:1 ratio) provides strong smoothing
        # See docs/PERIOD_CONFIGURATION_RATIONALE.md for detailed analysis
        self.periods = (
            periods
            if periods is not None
            else [
                Period(name="daily", n_periods=12, column="hour", input_range=(1, 24)),
                Period(
                    name="weekly", n_periods=7, column="dayofweek", input_range=(0, 6)
                ),
                Period(
                    name="monthly", n_periods=12, column="month", input_range=(1, 12)
                ),
                Period(
                    name="quarterly", n_periods=4, column="quarter", input_range=(1, 4)
                ),
                Period(
                    name="yearly",
                    n_periods=12,
                    column="dayofyear",
                    input_range=(1, 365),
                ),
            ]
        )
        self.lags_consider = (
            lags_consider if lags_consider is not None else list(range(1, 24))
        )
        self.train_size = (
            train_size if train_size is not None else pd.Timedelta(days=3 * 365)
        )
        self.end_train_default = end_train_default
        self.delta_val = (
            delta_val if delta_val is not None else pd.Timedelta(hours=24 * 7 * 10)
        )
        self.predict_size = predict_size
        self.refit_size = refit_size
        self.random_state = random_state
        self.n_hyperparameters_trials = n_hyperparameters_trials
        self.data_filename = data_filename

    def get_params(self, deep: bool = True) -> Dict[str, object]:
        """
        Get parameters for this configuration object.

        Args:
            deep: If True, will return the parameters for this configuration and
                contained sub-objects that are estimators.

        Returns:
            params: Dictionary of parameter names mapped to their values.

        Examples:
            >>> from spotforecast2_safe.manager.configurator.config_entsoe import ConfigEntsoe
            >>> config = ConfigEntsoe(api_country_code="FR")
            >>> p = config.get_params()
            >>> p["api_country_code"]
            'FR'
            >>> p["predict_size"]
            24
        """
        params = {
            "api_country_code": self.API_COUNTRY_CODE,
            "periods": self.periods,
            "lags_consider": self.lags_consider,
            "train_size": self.train_size,
            "end_train_default": self.end_train_default,
            "delta_val": self.delta_val,
            "predict_size": self.predict_size,
            "refit_size": self.refit_size,
            "random_state": self.random_state,
            "n_hyperparameters_trials": self.n_hyperparameters_trials,
            "data_filename": self.data_filename,
        }

        # Expose period sub-objects via the '__' notation if deep=True
        if deep and self.periods is not None:
            for period in self.periods:
                prefix = f"periods__{period.name}"
                params[f"{prefix}__n_periods"] = period.n_periods
                params[f"{prefix}__column"] = period.column
                params[f"{prefix}__input_range"] = period.input_range

        return params

    def set_params(
        self, params: Dict[str, object] = None, **kwargs: object
    ) -> "ConfigEntsoe":
        """
        Set the parameters of this configuration object.

        Args:
            params: Optional dictionary of parameter names mapped to their
                new values.
            **kwargs: Additional parameter names mapped to their new values.
                It supports configuring nested 'Period' objects using the
                `periods__<name>__<param>` notation.

        Returns:
            ConfigEntsoe: The configuration instance with updated
                parameters (supports method chaining).

        Examples:
            >>> from spotforecast2_safe.manager.configurator.config_entsoe import ConfigEntsoe
            >>> config = ConfigEntsoe()
            >>> _ = config.set_params(api_country_code="FR", predict_size=48)
            >>> config.API_COUNTRY_CODE
            'FR'
            >>> config.predict_size
            48

            >>> # Deep parameter setting
            >>> _ = config.set_params(periods__daily__n_periods=24)
            >>> next(p.n_periods for p in config.periods if p.name == "daily")
            24
        """
        # Merge params dict and kwargs
        all_params: Dict[str, object] = {}
        if params is not None:
            all_params.update(params)
        all_params.update(kwargs)

        if not all_params:
            return self

        nested_period_params = {}
        flat_params = {}

        for key, value in all_params.items():
            if key.startswith("periods__"):
                parts = key.split("__")
                if len(parts) == 3:
                    _, p_name, p_param = parts
                    if p_name not in nested_period_params:
                        nested_period_params[p_name] = {}
                    nested_period_params[p_name][p_param] = value
                else:
                    raise ValueError(
                        f"Invalid deep parameter format: {key}. "
                        "Expected format: periods__<name>__<param>"
                    )
            else:
                flat_params[key] = value

        # Set standard parameters first
        for key, value in flat_params.items():
            if key == "api_country_code":
                self.API_COUNTRY_CODE = value
            elif hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(
                    f"Invalid parameter {key} for {self.__class__.__name__}. "
                    "Check the list of available parameters with `get_params()`."
                )

        # Apply nested parameters to frozen Period dataclasses
        if nested_period_params and self.periods is not None:
            existing_names = {p.name for p in self.periods}
            for p_name in nested_period_params:
                if p_name not in existing_names:
                    raise ValueError(
                        f"Period with name '{p_name}' not found in configuration."
                    )

            new_periods = []
            for period in self.periods:
                if period.name in nested_period_params:
                    # Period is a frozen dataclass, so we utilize replace() to replicate
                    # an updated version.
                    updated_period = replace(
                        period, **nested_period_params[period.name]
                    )
                    new_periods.append(updated_period)
                else:
                    new_periods.append(period)
            self.periods = new_periods

        return self
