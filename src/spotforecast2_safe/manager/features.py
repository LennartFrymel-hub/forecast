# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Feature engineering helpers for exogenous variable pipelines.

This module provides five public helper functions used to transform raw
exogenous inputs into model-ready feature matrices:

- :func:`apply_cyclical_encoding` — convert periodic integer features (hour,
  month, …) to sine/cosine pairs via
  :class:`~feature_engine.creation.CyclicalFeatures`.
- :func:`create_interaction_features` — append bilinear interaction terms
  between calendar, weather-window, and holiday columns using
  :class:`~sklearn.preprocessing.PolynomialFeatures`.
- :func:`select_exogenous_features` — filter and deduplicate the column list
  that should be passed as ``exog`` to a recursive forecaster.
- :func:`merge_data_and_covariates` — inner-join target data with exogenous
  features and produce separate train and prediction covariate slices.
- :func:`get_target_data` — extract the training series and exogenous
  feature slices for a single target column from the shared pipeline
  state held in a :class:`~spotforecast2_safe.manager.configurator.config_multi.ConfigMulti`
  object.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.preprocessing import PolynomialFeatures

if TYPE_CHECKING:
    from spotforecast2_safe.manager.configurator.config_multi import ConfigMulti

try:
    from feature_engine.creation import CyclicalFeatures
except ImportError:  # pragma: no cover
    raise ImportError(
        "feature_engine is required. Install with: pip install feature-engine"
    )


# =============================================================================
# apply_cyclical_encoding
# =============================================================================


def apply_cyclical_encoding(
    data: pd.DataFrame,
    features_to_encode: Optional[List[str]] = None,
    max_values: Optional[Dict[str, int]] = None,
    drop_original: bool = False,
) -> pd.DataFrame:
    """Apply cyclical (sine/cosine) encoding to periodic integer features.

    Converts periodic calendar and solar features into sine/cosine pairs so
    that a distance-based model can recognise that hour 23 is close to hour 0.
    Columns that appear in *features_to_encode* but are absent from *data* are
    silently skipped.

    Uses :class:`~feature_engine.creation.CyclicalFeatures` internally, so
    the new columns follow the ``<feature>_sin`` / ``<feature>_cos`` naming
    convention.

    Args:
        data: DataFrame whose columns contain the periodic features to encode.
            The index is preserved unchanged.
        features_to_encode: Column names subject to cyclical encoding.
            Defaults to ``["month", "week", "day_of_week", "hour",
            "sunrise_hour", "sunset_hour"]``.
        max_values: Mapping from feature name to its natural period (inclusive
            maximum).  Defaults to the standard calendar / solar periods:
            ``month→12``, ``week→52``, ``day_of_week→6``, ``hour→24``,
            ``sunrise_hour→24``, ``sunset_hour→24``.
        drop_original: If ``True``, the original integer columns are removed
            from the output DataFrame.  Defaults to ``False``.

    Returns:
        pd.DataFrame: Copy of *data* with sine and cosine columns appended (or
        replaced when *drop_original* is ``True``).

    Examples:
        Encode hour and month from a small hourly time series:

        ```{python}
        import pandas as pd
        from spotforecast2_safe.manager.features import apply_cyclical_encoding

        idx = pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")
        df = pd.DataFrame(
            {"month": idx.month, "hour": idx.hour},
            index=idx,
        )

        result = apply_cyclical_encoding(
            df,
            features_to_encode=["month", "hour"],
            max_values={"month": 12, "hour": 24},
        )
        print("columns:", result.columns.tolist())
        print("shape:  ", result.shape)
        ```
    """
    if features_to_encode is None:
        features_to_encode = [
            "month",
            "week",
            "day_of_week",
            "hour",
            "sunrise_hour",
            "sunset_hour",
        ]

    if max_values is None:
        max_values = {
            "month": 12,
            "week": 52,
            "day_of_week": 6,
            "hour": 24,
            "sunrise_hour": 24,
            "sunset_hour": 24,
        }

    available_features = [f for f in features_to_encode if f in data.columns]
    if not available_features:
        return data
    available_max_values = {
        k: v for k, v in max_values.items() if k in available_features
    }

    cyclical_encoder = CyclicalFeatures(
        variables=available_features,
        max_values=available_max_values,
        drop_original=drop_original,
    )
    return cyclical_encoder.fit_transform(data)


# =============================================================================
# create_interaction_features
# =============================================================================


def create_interaction_features(
    exogenous_features: pd.DataFrame,
    weather_aligned: pd.DataFrame,
    base_cols: Optional[List[str]] = None,
    weather_window_pattern: str = "_window_",
    include_weather_funcs: Optional[List[str]] = None,
    holiday_col: str = "is_holiday",
    degree: int = 1,
) -> pd.DataFrame:
    """Append bilinear interaction terms to an exogenous feature matrix.

    Constructs interaction features between calendar cyclical columns, weather
    rolling-window statistics, raw weather variables, and (optionally) a
    holiday indicator.  Only pairwise products are generated
    (``interaction_only=True``) without a bias term.  The new columns are
    named ``poly_<col_a>__<col_b>`` and are appended to the right of
    *exogenous_features*.

    Args:
        exogenous_features: DataFrame containing all current exogenous features.
            Must already include the columns listed in *base_cols*.
        weather_aligned: DataFrame containing only the raw (un-transformed)
            weather variables.  Used to identify which columns in
            *exogenous_features* are raw weather columns.
        base_cols: Calendar cyclical columns that form the left side of each
            interaction.  Defaults to
            ``["day_of_week_sin", "day_of_week_cos", "hour_sin", "hour_cos"]``.
        weather_window_pattern: Substring used to detect rolling-window weather
            columns in *exogenous_features*.  Defaults to ``"_window_"``.
        include_weather_funcs: Rolling-aggregation suffixes to include.
            Defaults to ``["_mean", "_min", "_max"]``.
        holiday_col: Name of the binary holiday indicator column.  If present
            in *exogenous_features* it is added to the interaction pool.
            Defaults to ``"is_holiday"``.
        degree: Polynomial degree passed to
            :class:`~sklearn.preprocessing.PolynomialFeatures`.  At degree 1
            only pairwise products are produced (no higher-order terms).
            Defaults to ``1``.

    Returns:
        pd.DataFrame: *exogenous_features* with ``poly_*`` interaction columns
        appended.  The original columns are unchanged.

    Examples:
        Add interaction terms between cyclical hour features and a weather
        variable:

        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2_safe.manager.features import create_interaction_features

        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")

        weather = pd.DataFrame(
            {"temperature": rng.normal(10, 3, 72)},
            index=idx,
        )
        exog = pd.DataFrame(
            {
                "day_of_week_sin": np.sin(2 * np.pi * idx.dayofweek / 7),
                "day_of_week_cos": np.cos(2 * np.pi * idx.dayofweek / 7),
                "hour_sin": np.sin(2 * np.pi * idx.hour / 24),
                "hour_cos": np.cos(2 * np.pi * idx.hour / 24),
                "temperature": weather["temperature"],
            },
            index=idx,
        )

        result = create_interaction_features(
            exogenous_features=exog,
            weather_aligned=weather,
        )
        poly_cols = [c for c in result.columns if c.startswith("poly_")]
        print("poly columns:", poly_cols[:4])
        print("total columns:", result.shape[1])
        ```
    """
    if base_cols is None:
        base_cols = [
            "day_of_week_sin",
            "day_of_week_cos",
            "hour_sin",
            "hour_cos",
        ]

    if include_weather_funcs is None:
        include_weather_funcs = ["_mean", "_min", "_max"]

    transformer_poly = PolynomialFeatures(
        degree=degree, interaction_only=True, include_bias=False
    )
    transformer_poly = transformer_poly.set_output(transform="pandas")

    weather_window_cols = [
        col
        for col in exogenous_features.columns
        if weather_window_pattern in col
        and any(func in col for func in include_weather_funcs)
    ]

    raw_weather_cols = [
        col
        for col in exogenous_features.columns
        if col in weather_aligned.columns and col not in weather_window_cols
    ]

    poly_cols = list(base_cols)
    poly_cols.extend(weather_window_cols)
    poly_cols.extend(raw_weather_cols)
    if holiday_col in exogenous_features.columns:
        poly_cols.append(holiday_col)

    poly_features = transformer_poly.fit_transform(exogenous_features[poly_cols])
    poly_features = poly_features.drop(columns=poly_cols)
    poly_features.columns = [f"poly_{col}" for col in poly_features.columns]
    poly_features.columns = poly_features.columns.str.replace(" ", "__")

    return pd.concat([exogenous_features, poly_features], axis=1)


# =============================================================================
# select_exogenous_features
# =============================================================================


def select_exogenous_features(
    exogenous_features: pd.DataFrame,
    weather_aligned: pd.DataFrame,
    cyclical_regex: str = "_sin$|_cos$",
    include_weather_windows: bool = False,
    include_holiday_features: bool = False,
    include_poly_features: bool = False,
    additional_exog_columns: Optional[List[str]] = None,
) -> List[str]:

    """Select and deduplicate exogenous feature columns for model training.

    Builds a prioritised, deduplicated list of column names from
    *exogenous_features* suitable for passing as ``exog`` to a recursive
    forecaster.  The selection order is:

    1. Cyclical sine/cosine columns (always included).
    2. Weather rolling-window columns (optional, ``include_weather_windows``).
    3. Raw weather columns shared with *weather_aligned*.
    4. Holiday-related columns starting with ``"holiday"`` (optional).
    5. Polynomial interaction columns starting with ``"poly_"`` (optional).

    Duplicates are removed while preserving insertion order.

    Args:
        exogenous_features: DataFrame containing the full set of candidate
            feature columns.
        weather_aligned: DataFrame whose column names identify the raw (
            non-window, non-polynomial) weather variables.
        cyclical_regex: Regular expression matched against column names to
            detect cyclical sine/cosine features.  Defaults to
            ``"_sin$|_cos$"``.
        include_weather_windows: If ``True``, include rolling-window weather
            columns (those containing ``"_window_"`` plus ``"_mean"``,
            ``"_min"``, or ``"_max"``).  Defaults to ``False``.
        include_holiday_features: If ``True``, include columns whose names
            start with ``"holiday"``.  Defaults to ``False``.
        include_poly_features: If ``True``, include polynomial interaction
            columns whose names start with ``"poly_"``.  Defaults to
            ``False``.

    Returns:
        List[str]: Deduplicated list of selected column names in priority
        order.

    Examples:
        Select cyclical and raw weather columns from a feature matrix:

        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2_safe.manager.features import select_exogenous_features

        rng = np.random.default_rng(1)
        idx = pd.date_range("2024-01-01", periods=24, freq="h", tz="UTC")

        weather = pd.DataFrame({"wind_speed": rng.uniform(0, 10, 24)}, index=idx)
        exog = pd.DataFrame(
            {
                "hour_sin": np.sin(2 * np.pi * idx.hour / 24),
                "hour_cos": np.cos(2 * np.pi * idx.hour / 24),
                "wind_speed": weather["wind_speed"],
                "holiday_flag": 0,
            },
            index=idx,
        )

        selected = select_exogenous_features(
            exogenous_features=exog,
            weather_aligned=weather,
            include_holiday_features=False,
        )
        print("selected:", selected)
        ```
    """
    exog_list: List[str] = []

    exog_list.extend(exogenous_features.filter(regex=cyclical_regex).columns.tolist())

    if include_weather_windows:
        weather_window_features = [
            col
            for col in exogenous_features.columns
            if "_window_" in col and ("_mean" in col or "_min" in col or "_max" in col)
        ]
        exog_list.extend(weather_window_features)

    raw_weather_features = [
        col for col in exogenous_features.columns if col in weather_aligned.columns
    ]
    exog_list.extend(raw_weather_features)

    if include_holiday_features:
        holiday_related = [
            col
            for col in exogenous_features.columns
            if (
                col.startswith("holiday")
                or col == "is_holiday"
                or col.startswith("is_holiday")
            )
        ]
        exog_list.extend(holiday_related)

    if additional_exog_columns:
        missing_additional = [
            col
            for col in additional_exog_columns
            if col not in exogenous_features.columns
        ]

        if missing_additional:
            raise ValueError(
                "Additional exogenous feature columns are missing from "
                f"exogenous_features: {missing_additional}"
            )

        exog_list.extend(additional_exog_columns)

    if include_poly_features:
        poly_features_list = [
            col for col in exogenous_features.columns if col.startswith("poly_")
        ]
        exog_list.extend(poly_features_list)

    return list(dict.fromkeys(exog_list))


# =============================================================================
# merge_data_and_covariates
# =============================================================================


def merge_data_and_covariates(
    data: pd.DataFrame,
    exogenous_features: pd.DataFrame,
    target_columns: List[str],
    exog_features: List[str],
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    cov_end: Union[str, pd.Timestamp],
    forecast_horizon: int,
    cast_dtype: Optional[str] = "float32",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Merge target data with exogenous features and split into train/predict slices.

    Performs an inner join of the selected *target_columns* from *data* with
    the selected *exog_features* from *exogenous_features* over the training
    window ``[start, end]``.  A separate prediction covariate slice
    ``(end+1h, cov_end]`` is also returned for use during inference.

    String timestamps are converted to UTC-aware
    :class:`~pandas.Timestamp` objects automatically.

    Args:
        data: DataFrame containing one or more target time series with a
            tz-aware :class:`~pandas.DatetimeIndex`.
        exogenous_features: DataFrame with all exogenous feature columns,
            covering at least the window ``[start, cov_end]``.
        target_columns: Column names of the target variables to keep from
            *data*.
        exog_features: Column names of the exogenous features to include in
            the merged output and the prediction slice.
        start: Inclusive start of the training window.  String values are
            parsed with ``utc=True``.
        end: Inclusive end of the training window.  String values are parsed
            with ``utc=True``.
        cov_end: Inclusive end of the covariate (forecast) window.  String
            values are parsed with ``utc=True``.
        forecast_horizon: Number of forecast steps ahead (informational; used
            by calling code to validate slice length).
        cast_dtype: NumPy dtype string applied to the merged training
            DataFrame via :meth:`~pandas.DataFrame.astype`.  Pass ``None`` to
            skip casting.  Defaults to ``"float32"``.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A three-tuple
        ``(data_with_exog, exo_tmp, exo_pred)`` where:

        - **data_with_exog** — training-window DataFrame with target and
            exogenous columns merged (inner join on index).
        - **exo_tmp** — full exogenous slice over ``[start, end]`` (all
            columns, not just *exog_features*).
        - **exo_pred** — forecast-window exogenous slice over
            ``(end+1h, cov_end]`` (all columns).

    Examples:
        Merge a toy target series with calendar features over a 3-day window:

        ```{python}
        import numpy as np
        import pandas as pd
        from spotforecast2_safe.manager.features import merge_data_and_covariates

        idx = pd.date_range("2024-01-01", periods=120, freq="h", tz="UTC")
        data = pd.DataFrame({"load": np.random.default_rng(42).normal(100, 10, 120)}, index=idx)
        exog = pd.DataFrame(
            {"hour_sin": np.sin(2 * np.pi * idx.hour / 24),
             "hour_cos": np.cos(2 * np.pi * idx.hour / 24)},
            index=idx,
        )

        start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        end   = pd.Timestamp("2024-01-04 23:00", tz="UTC")  # 96 h training
        cov_end = pd.Timestamp("2024-01-05 23:00", tz="UTC")  # 24 h forecast

        merged, exo_train, exo_pred = merge_data_and_covariates(
            data=data,
            exogenous_features=exog,
            target_columns=["load"],
            exog_features=["hour_sin", "hour_cos"],
            start=start,
            end=end,
            cov_end=cov_end,
            forecast_horizon=24,
        )
        print("merged shape:   ", merged.shape)
        print("exo_train shape:", exo_train.shape)
        print("exo_pred shape: ", exo_pred.shape)
        ```
    """
    if isinstance(start, str):
        start = pd.to_datetime(start, utc=True)
    if isinstance(end, str):
        end = pd.to_datetime(end, utc=True)
    if isinstance(cov_end, str):
        cov_end = pd.to_datetime(cov_end, utc=True)

    exo_tmp = exogenous_features.loc[start:end].copy()
    exo_pred = exogenous_features.loc[end + pd.Timedelta(hours=1) : cov_end].copy()

    data_with_exog = data[target_columns].merge(
        exo_tmp[exog_features],
        left_index=True,
        right_index=True,
        how="inner",
    )

    if cast_dtype is not None:
        data_with_exog = data_with_exog.astype(cast_dtype)

    return data_with_exog, exo_tmp, exo_pred


# =============================================================================
# get_target_data
# =============================================================================


def get_target_data(
    target: str,
    df_pipeline: pd.DataFrame,
    config: "ConfigMulti",
    data_with_exog: Optional[pd.DataFrame] = None,
    exog_feature_names: Optional[List[str]] = None,
    exo_pred: Optional[pd.DataFrame] = None,
) -> Tuple[pd.Series, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """Extract the training series and exogenous slices for one target column.

    Clips the target column of *df_pipeline* to the training window defined by
    ``config.start_train_ts`` and ``config.end_train_ts``.  When exogenous
    features are enabled (``config.use_exogenous_features is True``) and
    *data_with_exog* is provided, the matching exogenous training slice and
    forecast-horizon slice are also returned; otherwise both are ``None``.

    This function is the canonical way to extract per-target data from the
    shared pipeline state so that outlier removal, imputation, and feature
    engineering are applied consistently across all forecasting tasks.

    Args:
        target: Name of the target column to extract from *df_pipeline*.
        df_pipeline: DataFrame with a tz-aware :class:`~pandas.DatetimeIndex`
            containing all target columns produced by the preprocessing
            pipeline.
        config: Pipeline configuration object.  Must have the following
            attributes set before calling this function:

            - ``start_train_ts`` — inclusive start of the training window
              (:class:`~pandas.Timestamp`, tz-aware).
            - ``end_train_ts`` — inclusive end of the training window
              (:class:`~pandas.Timestamp`, tz-aware).
            - ``use_exogenous_features`` — ``bool`` flag controlling whether
              exogenous features are used.
        data_with_exog: Merged DataFrame of target and exogenous columns
            covering at least ``[config.start_train_ts, config.end_train_ts]``.
            Required when ``config.use_exogenous_features`` is ``True``.
            Pass ``None`` (default) to skip exogenous slicing.
        exog_feature_names: Column names to select from *data_with_exog* and
            *exo_pred*.  Required when *data_with_exog* is not ``None``.
            Pass ``None`` (default) when exogenous features are disabled.
        exo_pred: Exogenous feature DataFrame covering the forecast horizon
            ``(config.end_train_ts, config.cov_end]``.  Required when
            *data_with_exog* is not ``None``.  Pass ``None`` (default) when
            exogenous features are disabled.

    Returns:
        Tuple[pd.Series, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        A three-tuple ``(y_train, exog_train, exog_future)`` where:

        - **y_train** — 1-D Series with the target values over the training
            window ``[config.start_train_ts, config.end_train_ts]``, squeezed
            to a plain :class:`~pandas.Series`.
        - **exog_train** — DataFrame of selected exogenous features over the
            training window, cast to ``float32``.  ``None`` when exogenous
            features are disabled or *data_with_exog* is ``None``.
        - **exog_future** — DataFrame of selected exogenous features covering
            the forecast horizon, cast to ``float32``.  ``None`` when exogenous
            features are disabled or *exo_pred* is ``None``.

    Examples:
        Extract training data for a single target without exogenous features:

        ```{python}
        import pandas as pd
        import numpy as np
        from spotforecast2_safe.manager.features import get_target_data
        from spotforecast2_safe.manager.configurator.config_multi import ConfigMulti

        idx = pd.date_range("2024-01-01", periods=168, freq="h", tz="UTC")
        df_pipeline = pd.DataFrame({"load": np.random.default_rng(0).normal(100, 10, 168)}, index=idx)

        config = ConfigMulti(
            targets=["load"],
            use_exogenous_features=False,
        )
        config.start_train_ts = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        config.end_train_ts   = pd.Timestamp("2024-01-07 23:00", tz="UTC")

        y_train, exog_train, exog_future = get_target_data(
            target="load",
            df_pipeline=df_pipeline,
            config=config,
        )
        print(f"y_train length: {len(y_train)}")
        print(f"exog_train:     {exog_train}")
        print(f"exog_future:    {exog_future}")
        ```

        Extract training data with exogenous features enabled:

        ```{python}
        import pandas as pd
        import numpy as np
        from spotforecast2_safe.manager.features import get_target_data
        from spotforecast2_safe.manager.configurator.config_multi import ConfigMulti

        rng = np.random.default_rng(1)
        idx_train = pd.date_range("2024-01-01", periods=168, freq="h", tz="UTC")
        idx_future = pd.date_range("2024-01-08", periods=24, freq="h", tz="UTC")

        df_pipeline = pd.DataFrame({"load": rng.normal(100, 10, 168)}, index=idx_train)

        data_with_exog = pd.DataFrame(
            {
                "load": df_pipeline["load"],
                "hour_sin": np.sin(2 * np.pi * idx_train.hour / 24),
                "hour_cos": np.cos(2 * np.pi * idx_train.hour / 24),
            },
            index=idx_train,
        )
        exo_pred = pd.DataFrame(
            {
                "hour_sin": np.sin(2 * np.pi * idx_future.hour / 24),
                "hour_cos": np.cos(2 * np.pi * idx_future.hour / 24),
            },
            index=idx_future,
        )

        config = ConfigMulti(targets=["load"], use_exogenous_features=True)
        config.start_train_ts = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        config.end_train_ts   = pd.Timestamp("2024-01-07 23:00", tz="UTC")

        y_train, exog_train, exog_future = get_target_data(
            target="load",
            df_pipeline=df_pipeline,
            config=config,
            data_with_exog=data_with_exog,
            exog_feature_names=["hour_sin", "hour_cos"],
            exo_pred=exo_pred,
        )
        print(f"y_train length:     {len(y_train)}")
        print(f"exog_train shape:   {exog_train.shape}")
        print(f"exog_future shape:  {exog_future.shape}")
        print(f"exog_train dtype:   {exog_train.dtypes.iloc[0]}")
        ```
    """
    y_train = (
        df_pipeline[target].loc[config.start_train_ts : config.end_train_ts].squeeze()
    )

    exog_train: Optional[pd.DataFrame] = None
    exog_future: Optional[pd.DataFrame] = None

    if (
        config.use_exogenous_features
        and data_with_exog is not None
        and exog_feature_names is not None
    ):
        exog_train = (
            data_with_exog[exog_feature_names]
            .loc[config.start_train_ts : config.end_train_ts]
            .astype("float32")
        )

        if exo_pred is not None:
            if isinstance(exo_pred, pd.DataFrame):
                exog_future = exo_pred[exog_feature_names].astype("float32")
            else:
                raise TypeError(
                    "exo_pred must be a pandas DataFrame when using exogenous features."
                )

    return y_train, exog_train, exog_future
