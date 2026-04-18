# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for spotforecast2_safe.manager.features.

Covers:
- apply_cyclical_encoding: output shape, new columns, drop_original, missing columns
- create_interaction_features: poly column naming, no-weather case, holiday column
- select_exogenous_features: default selection, optional switches, deduplication
- merge_data_and_covariates: train/pred split, inner join, cast_dtype, string timestamps
- get_target_data: y_train slice, exog disabled/enabled, dtype, missing target
- Package-level imports from spotforecast2_safe.manager
- Backward-compatible private aliases in n2n_predict_with_covariates
"""

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.manager.configurator.config_multi import ConfigMulti
from spotforecast2_safe.manager.features import (
    apply_cyclical_encoding,
    create_interaction_features,
    get_target_data,
    merge_data_and_covariates,
    select_exogenous_features,
)

# =============================================================================
# Shared fixtures
# =============================================================================


@pytest.fixture
def hourly_idx():
    """48-hour DatetimeIndex starting 2024-01-01."""
    return pd.date_range("2024-01-01", periods=48, freq="h", tz="UTC")


@pytest.fixture
def calendar_df(hourly_idx):
    """Simple calendar-feature DataFrame (month, week, day_of_week, hour)."""
    return pd.DataFrame(
        {
            "month": hourly_idx.month,
            "week": hourly_idx.isocalendar().week.astype(int).values,
            "day_of_week": hourly_idx.dayofweek,
            "hour": hourly_idx.hour,
        },
        index=hourly_idx,
        dtype=float,
    )


@pytest.fixture
def weather_df(hourly_idx):
    """Single-column weather DataFrame."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({"temperature": rng.normal(10, 3, 48)}, index=hourly_idx)


@pytest.fixture
def cyclical_exog(hourly_idx):
    """Exogenous DataFrame with only cyclical (sin/cos) columns."""
    return pd.DataFrame(
        {
            "day_of_week_sin": np.sin(2 * np.pi * hourly_idx.dayofweek / 7),
            "day_of_week_cos": np.cos(2 * np.pi * hourly_idx.dayofweek / 7),
            "hour_sin": np.sin(2 * np.pi * hourly_idx.hour / 24),
            "hour_cos": np.cos(2 * np.pi * hourly_idx.hour / 24),
        },
        index=hourly_idx,
    )


@pytest.fixture
def full_exog(hourly_idx, weather_df):
    """Exogenous DataFrame with cyclical + weather + holiday columns."""
    return pd.DataFrame(
        {
            "day_of_week_sin": np.sin(2 * np.pi * hourly_idx.dayofweek / 7),
            "day_of_week_cos": np.cos(2 * np.pi * hourly_idx.dayofweek / 7),
            "hour_sin": np.sin(2 * np.pi * hourly_idx.hour / 24),
            "hour_cos": np.cos(2 * np.pi * hourly_idx.hour / 24),
            "temperature": weather_df["temperature"].values,
            "holiday_xmas": 0,
        },
        index=hourly_idx,
    )


# =============================================================================
# apply_cyclical_encoding
# =============================================================================


class TestApplyCyclicalEncoding:
    def test_new_sin_cos_columns_added(self, calendar_df):
        result = apply_cyclical_encoding(
            calendar_df,
            features_to_encode=["month", "hour"],
            max_values={"month": 12, "hour": 24},
        )
        assert "month_sin" in result.columns
        assert "month_cos" in result.columns
        assert "hour_sin" in result.columns
        assert "hour_cos" in result.columns

    def test_original_columns_kept_by_default(self, calendar_df):
        result = apply_cyclical_encoding(
            calendar_df,
            features_to_encode=["hour"],
            max_values={"hour": 24},
        )
        assert "hour" in result.columns

    def test_drop_original_removes_source_columns(self, calendar_df):
        result = apply_cyclical_encoding(
            calendar_df,
            features_to_encode=["hour"],
            max_values={"hour": 24},
            drop_original=True,
        )
        assert "hour" not in result.columns

    def test_output_row_count_unchanged(self, calendar_df):
        result = apply_cyclical_encoding(calendar_df)
        assert len(result) == len(calendar_df)

    def test_missing_column_silently_skipped(self, calendar_df):
        """Columns in features_to_encode absent from data should not raise."""
        result = apply_cyclical_encoding(
            calendar_df,
            features_to_encode=["hour", "nonexistent_col"],
            max_values={"hour": 24, "nonexistent_col": 10},
        )
        assert "hour_sin" in result.columns
        assert "nonexistent_col_sin" not in result.columns

    def test_default_features_encode_all_standard_columns(self, calendar_df):
        # calendar_df has month, week, day_of_week, hour — all in default list
        result = apply_cyclical_encoding(calendar_df)
        for feat in ["month", "week", "day_of_week", "hour"]:
            assert f"{feat}_sin" in result.columns
            assert f"{feat}_cos" in result.columns

    def test_sin_cos_values_in_range(self, calendar_df):
        result = apply_cyclical_encoding(
            calendar_df,
            features_to_encode=["hour"],
            max_values={"hour": 24},
        )
        assert result["hour_sin"].between(-1.0, 1.0).all()
        assert result["hour_cos"].between(-1.0, 1.0).all()

    def test_empty_features_to_encode_returns_original(self, calendar_df):
        result = apply_cyclical_encoding(calendar_df, features_to_encode=[])
        pd.testing.assert_frame_equal(result, calendar_df)

    def test_index_preserved(self, calendar_df):
        result = apply_cyclical_encoding(calendar_df)
        pd.testing.assert_index_equal(result.index, calendar_df.index)


# =============================================================================
# create_interaction_features
# =============================================================================


class TestCreateInteractionFeatures:
    def test_poly_columns_appended(self, cyclical_exog, weather_df):
        exog = pd.concat([cyclical_exog, weather_df], axis=1)
        result = create_interaction_features(
            exogenous_features=exog,
            weather_aligned=weather_df,
            degree=2,
        )
        poly_cols = [c for c in result.columns if c.startswith("poly_")]
        assert len(poly_cols) > 0

    def test_original_columns_preserved(self, cyclical_exog, weather_df):
        exog = pd.concat([cyclical_exog, weather_df], axis=1)
        result = create_interaction_features(
            exogenous_features=exog,
            weather_aligned=weather_df,
            degree=2,
        )
        for col in cyclical_exog.columns:
            assert col in result.columns
        assert "temperature" in result.columns

    def test_row_count_unchanged(self, cyclical_exog, weather_df):
        exog = pd.concat([cyclical_exog, weather_df], axis=1)
        result = create_interaction_features(
            exogenous_features=exog,
            weather_aligned=weather_df,
        )
        assert len(result) == len(exog)

    def test_no_weather_columns_still_works(self, cyclical_exog):
        """Empty weather_aligned → no weather cols in poly pool, but base_cols still interact."""
        empty_weather = pd.DataFrame(index=cyclical_exog.index)
        result = create_interaction_features(
            exogenous_features=cyclical_exog,
            weather_aligned=empty_weather,
            degree=2,
        )
        poly_cols = [c for c in result.columns if c.startswith("poly_")]
        assert len(poly_cols) > 0

    def test_holiday_col_included_when_present(
        self, cyclical_exog, weather_df, hourly_idx
    ):
        exog = pd.concat([cyclical_exog, weather_df], axis=1)
        exog["is_holiday"] = 0.0
        result = create_interaction_features(
            exogenous_features=exog,
            weather_aligned=weather_df,
            holiday_col="is_holiday",
            degree=2,
        )
        poly_cols = [c for c in result.columns if c.startswith("poly_")]
        # At least one poly col should involve is_holiday
        is_holiday_involved = any("is_holiday" in c for c in poly_cols)
        assert is_holiday_involved

    def test_poly_col_names_contain_no_spaces(self, cyclical_exog, weather_df):
        exog = pd.concat([cyclical_exog, weather_df], axis=1)
        result = create_interaction_features(
            exogenous_features=exog,
            weather_aligned=weather_df,
        )
        for col in result.columns:
            assert " " not in col


# =============================================================================
# select_exogenous_features
# =============================================================================


class TestSelectExogenousFeatures:
    def test_cyclical_always_selected(self, full_exog, weather_df):
        selected = select_exogenous_features(
            exogenous_features=full_exog,
            weather_aligned=weather_df,
        )
        assert "hour_sin" in selected
        assert "hour_cos" in selected
        assert "day_of_week_sin" in selected

    def test_raw_weather_always_selected(self, full_exog, weather_df):
        selected = select_exogenous_features(
            exogenous_features=full_exog,
            weather_aligned=weather_df,
        )
        assert "temperature" in selected

    def test_holiday_excluded_by_default(self, full_exog, weather_df):
        selected = select_exogenous_features(
            exogenous_features=full_exog,
            weather_aligned=weather_df,
            include_holiday_features=False,
        )
        assert "holiday_xmas" not in selected

    def test_holiday_included_when_flag_set(self, full_exog, weather_df):
        selected = select_exogenous_features(
            exogenous_features=full_exog,
            weather_aligned=weather_df,
            include_holiday_features=True,
        )
        assert "holiday_xmas" in selected

    def test_no_duplicates(self, full_exog, weather_df):
        selected = select_exogenous_features(
            exogenous_features=full_exog,
            weather_aligned=weather_df,
            include_weather_windows=True,
            include_holiday_features=True,
            include_poly_features=True,
        )
        assert len(selected) == len(set(selected))

    def test_returns_list(self, full_exog, weather_df):
        selected = select_exogenous_features(
            exogenous_features=full_exog,
            weather_aligned=weather_df,
        )
        assert isinstance(selected, list)

    def test_window_features_selected_when_flag_set(self, hourly_idx, weather_df):
        exog = pd.DataFrame(
            {
                "hour_sin": np.sin(2 * np.pi * hourly_idx.hour / 24),
                "temperature": weather_df["temperature"].values,
                "temperature_window_mean": weather_df["temperature"]
                .rolling(3)
                .mean()
                .values,
                "temperature_window_min": weather_df["temperature"]
                .rolling(3)
                .min()
                .values,
            },
            index=hourly_idx,
        )
        exog = exog.bfill()
        selected = select_exogenous_features(
            exogenous_features=exog,
            weather_aligned=weather_df,
            include_weather_windows=True,
        )
        assert "temperature_window_mean" in selected
        assert "temperature_window_min" in selected

    def test_poly_features_selected_when_flag_set(self, hourly_idx, weather_df):
        exog = pd.DataFrame(
            {
                "hour_sin": np.sin(2 * np.pi * hourly_idx.hour / 24),
                "temperature": weather_df["temperature"].values,
                "poly_hour_sin__temperature": np.ones(48),
            },
            index=hourly_idx,
        )
        selected = select_exogenous_features(
            exogenous_features=exog,
            weather_aligned=weather_df,
            include_poly_features=True,
        )
        assert "poly_hour_sin__temperature" in selected


# =============================================================================
# merge_data_and_covariates
# =============================================================================


class TestMergeDataAndCovariates:
    @pytest.fixture
    def base_data(self, hourly_idx):
        rng = np.random.default_rng(42)
        return pd.DataFrame({"load": rng.normal(100, 10, 48)}, index=hourly_idx)

    @pytest.fixture
    def base_exog(self, hourly_idx):
        return pd.DataFrame(
            {
                "hour_sin": np.sin(2 * np.pi * hourly_idx.hour / 24),
                "hour_cos": np.cos(2 * np.pi * hourly_idx.hour / 24),
            },
            index=hourly_idx,
        )

    def test_train_shape(self, base_data, base_exog):
        start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        end = pd.Timestamp("2024-01-01 23:00", tz="UTC")  # 24 rows
        cov_end = pd.Timestamp("2024-01-02 23:00", tz="UTC")
        merged, _, _ = merge_data_and_covariates(
            data=base_data,
            exogenous_features=base_exog,
            target_columns=["load"],
            exog_features=["hour_sin", "hour_cos"],
            start=start,
            end=end,
            cov_end=cov_end,
            forecast_horizon=24,
        )
        assert merged.shape == (24, 3)  # load + 2 exog

    def test_pred_slice_length(self, base_data, base_exog):
        start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        end = pd.Timestamp("2024-01-01 23:00", tz="UTC")
        cov_end = pd.Timestamp("2024-01-02 23:00", tz="UTC")
        _, _, exo_pred = merge_data_and_covariates(
            data=base_data,
            exogenous_features=base_exog,
            target_columns=["load"],
            exog_features=["hour_sin", "hour_cos"],
            start=start,
            end=end,
            cov_end=cov_end,
            forecast_horizon=24,
        )
        assert len(exo_pred) == 24

    def test_cast_dtype_float32(self, base_data, base_exog):
        start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        end = pd.Timestamp("2024-01-01 23:00", tz="UTC")
        cov_end = pd.Timestamp("2024-01-02 23:00", tz="UTC")
        merged, _, _ = merge_data_and_covariates(
            data=base_data,
            exogenous_features=base_exog,
            target_columns=["load"],
            exog_features=["hour_sin", "hour_cos"],
            start=start,
            end=end,
            cov_end=cov_end,
            forecast_horizon=24,
            cast_dtype="float32",
        )
        assert all(merged[c].dtype == np.float32 for c in merged.columns)

    def test_cast_dtype_none_keeps_original(self, base_data, base_exog):
        start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        end = pd.Timestamp("2024-01-01 23:00", tz="UTC")
        cov_end = pd.Timestamp("2024-01-02 23:00", tz="UTC")
        merged, _, _ = merge_data_and_covariates(
            data=base_data,
            exogenous_features=base_exog,
            target_columns=["load"],
            exog_features=["hour_sin", "hour_cos"],
            start=start,
            end=end,
            cov_end=cov_end,
            forecast_horizon=24,
            cast_dtype=None,
        )
        # Should not raise; dtype is float64 from fixture
        assert merged["load"].dtype == np.float64

    def test_string_timestamps_accepted(self, base_data, base_exog):
        merged, _, _ = merge_data_and_covariates(
            data=base_data,
            exogenous_features=base_exog,
            target_columns=["load"],
            exog_features=["hour_sin", "hour_cos"],
            start="2024-01-01 00:00",
            end="2024-01-01 23:00",
            cov_end="2024-01-02 23:00",
            forecast_horizon=24,
        )
        assert len(merged) == 24

    def test_exo_tmp_contains_all_exog_columns(self, base_data, base_exog):
        start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        end = pd.Timestamp("2024-01-01 23:00", tz="UTC")
        cov_end = pd.Timestamp("2024-01-02 23:00", tz="UTC")
        _, exo_tmp, _ = merge_data_and_covariates(
            data=base_data,
            exogenous_features=base_exog,
            target_columns=["load"],
            exog_features=["hour_sin", "hour_cos"],
            start=start,
            end=end,
            cov_end=cov_end,
            forecast_horizon=24,
        )
        assert list(exo_tmp.columns) == ["hour_sin", "hour_cos"]

    def test_merged_columns_include_target_and_exog(self, base_data, base_exog):
        start = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        end = pd.Timestamp("2024-01-01 23:00", tz="UTC")
        cov_end = pd.Timestamp("2024-01-02 23:00", tz="UTC")
        merged, _, _ = merge_data_and_covariates(
            data=base_data,
            exogenous_features=base_exog,
            target_columns=["load"],
            exog_features=["hour_sin", "hour_cos"],
            start=start,
            end=end,
            cov_end=cov_end,
            forecast_horizon=24,
        )
        assert "load" in merged.columns
        assert "hour_sin" in merged.columns
        assert "hour_cos" in merged.columns


# =============================================================================
# Package-level imports
# =============================================================================


class TestPackageLevelImports:
    def test_import_from_manager(self):
        from spotforecast2_safe.manager import (
            apply_cyclical_encoding,
            create_interaction_features,
            merge_data_and_covariates,
            select_exogenous_features,
        )

        assert callable(apply_cyclical_encoding)
        assert callable(create_interaction_features)
        assert callable(select_exogenous_features)
        assert callable(merge_data_and_covariates)

    def test_import_from_manager_features(self):
        from spotforecast2_safe.manager.features import (
            apply_cyclical_encoding,
            create_interaction_features,
            merge_data_and_covariates,
            select_exogenous_features,
        )

        assert callable(apply_cyclical_encoding)
        assert callable(create_interaction_features)
        assert callable(select_exogenous_features)
        assert callable(merge_data_and_covariates)


# =============================================================================
# Backward-compatible private aliases in n2n_predict_with_covariates
# =============================================================================


class TestBackwardCompatibleAliases:
    @pytest.fixture(autouse=True)
    def _n2n_module(self):
        """Return the actual n2n_predict_with_covariates module (not the function)."""
        import sys

        # Ensure the module is imported so it appears in sys.modules.
        import spotforecast2_safe.processing.n2n_predict_with_covariates  # noqa: F401

        self.mod = sys.modules[
            "spotforecast2_safe.processing.n2n_predict_with_covariates"
        ]

    def test_private_aliases_exist(self):
        assert hasattr(self.mod, "_apply_cyclical_encoding")
        assert hasattr(self.mod, "_create_interaction_features")
        assert hasattr(self.mod, "_select_exogenous_features")
        assert hasattr(self.mod, "_merge_data_and_covariates")

    def test_aliases_point_to_public_functions(self):
        from spotforecast2_safe.manager.features import (
            apply_cyclical_encoding,
            create_interaction_features,
            merge_data_and_covariates,
            select_exogenous_features,
        )

        assert self.mod._apply_cyclical_encoding is apply_cyclical_encoding
        assert self.mod._create_interaction_features is create_interaction_features
        assert self.mod._select_exogenous_features is select_exogenous_features
        assert self.mod._merge_data_and_covariates is merge_data_and_covariates


# =============================================================================
# Shared fixtures for get_target_data
# =============================================================================


@pytest.fixture
def pipeline_idx():
    """168-hour (7-day) UTC DatetimeIndex starting 2024-01-01."""
    return pd.date_range("2024-01-01", periods=168, freq="h", tz="UTC")


@pytest.fixture
def df_pipeline(pipeline_idx):
    """Two-column target DataFrame over 168 hours."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {"load": rng.normal(100, 10, 168), "temp_load": rng.normal(50, 5, 168)},
        index=pipeline_idx,
    )


@pytest.fixture
def base_config(pipeline_idx):
    """ConfigMulti with training window covering the full 168-hour index."""
    cfg = ConfigMulti(targets=["load"], use_exogenous_features=False)
    cfg.start_train_ts = pipeline_idx[0]
    cfg.end_train_ts = pipeline_idx[-1]
    return cfg


@pytest.fixture
def exog_idx(pipeline_idx):
    """Future 24-hour index starting right after pipeline_idx."""
    return pd.date_range("2024-01-08", periods=24, freq="h", tz="UTC")


@pytest.fixture
def data_with_exog_df(pipeline_idx, df_pipeline):
    """Merged DataFrame that includes target and two exogenous columns."""
    return pd.DataFrame(
        {
            "load": df_pipeline["load"],
            "hour_sin": np.sin(2 * np.pi * pipeline_idx.hour / 24),
            "hour_cos": np.cos(2 * np.pi * pipeline_idx.hour / 24),
        },
        index=pipeline_idx,
    )


@pytest.fixture
def exo_pred_df(exog_idx):
    """Future exogenous DataFrame for the 24-hour forecast horizon."""
    return pd.DataFrame(
        {
            "hour_sin": np.sin(2 * np.pi * exog_idx.hour / 24),
            "hour_cos": np.cos(2 * np.pi * exog_idx.hour / 24),
        },
        index=exog_idx,
    )


# =============================================================================
# TestGetTargetDataNoExog
# =============================================================================


class TestGetTargetDataNoExog:
    """get_target_data with use_exogenous_features=False."""

    def test_returns_tuple_of_three(self, df_pipeline, base_config):
        result = get_target_data("load", df_pipeline, base_config)
        assert isinstance(result, tuple) and len(result) == 3

    def test_y_train_is_series(self, df_pipeline, base_config):
        y_train, _, _ = get_target_data("load", df_pipeline, base_config)
        assert isinstance(y_train, pd.Series)

    def test_y_train_length(self, df_pipeline, base_config):
        y_train, _, _ = get_target_data("load", df_pipeline, base_config)
        assert len(y_train) == 168

    def test_y_train_values_match_pipeline(self, df_pipeline, base_config):
        y_train, _, _ = get_target_data("load", df_pipeline, base_config)
        expected = df_pipeline["load"].loc[
            base_config.start_train_ts : base_config.end_train_ts
        ]
        pd.testing.assert_series_equal(y_train, expected)

    def test_exog_train_is_none(self, df_pipeline, base_config):
        _, exog_train, _ = get_target_data("load", df_pipeline, base_config)
        assert exog_train is None

    def test_exog_future_is_none(self, df_pipeline, base_config):
        _, _, exog_future = get_target_data("load", df_pipeline, base_config)
        assert exog_future is None

    def test_partial_training_window(self, df_pipeline, pipeline_idx):
        """Training window shorter than the full index."""
        cfg = ConfigMulti(targets=["load"], use_exogenous_features=False)
        cfg.start_train_ts = pipeline_idx[10]
        cfg.end_train_ts = pipeline_idx[72]
        y_train, _, _ = get_target_data("load", df_pipeline, cfg)
        assert len(y_train) == 63  # 72 - 10 + 1

    def test_second_target_column(self, df_pipeline, pipeline_idx):
        """Works for a target column other than the first."""
        cfg = ConfigMulti(targets=["temp_load"], use_exogenous_features=False)
        cfg.start_train_ts = pipeline_idx[0]
        cfg.end_train_ts = pipeline_idx[-1]
        y_train, _, _ = get_target_data("temp_load", df_pipeline, cfg)
        pd.testing.assert_series_equal(
            y_train,
            df_pipeline["temp_load"].loc[cfg.start_train_ts : cfg.end_train_ts],
        )

    def test_y_train_squeezed(self, df_pipeline, base_config):
        """Result of squeeze() must be a Series, not a DataFrame."""
        y_train, _, _ = get_target_data("load", df_pipeline, base_config)
        assert y_train.ndim == 1

    def test_missing_target_raises(self, df_pipeline, base_config):
        """Requesting a column not in df_pipeline should raise KeyError."""
        with pytest.raises(KeyError):
            get_target_data("nonexistent", df_pipeline, base_config)


# =============================================================================
# TestGetTargetDataWithExog
# =============================================================================


class TestGetTargetDataWithExog:
    """get_target_data with use_exogenous_features=True and full exog inputs."""

    @pytest.fixture(autouse=True)
    def _setup(self, df_pipeline, pipeline_idx, data_with_exog_df, exo_pred_df):
        self.df_pipeline = df_pipeline
        self.pipeline_idx = pipeline_idx
        self.data_with_exog = data_with_exog_df
        self.exo_pred = exo_pred_df
        self.exog_names = ["hour_sin", "hour_cos"]

        self.cfg = ConfigMulti(targets=["load"], use_exogenous_features=True)
        self.cfg.start_train_ts = pipeline_idx[0]
        self.cfg.end_train_ts = pipeline_idx[-1]

    def _call(self):
        return get_target_data(
            target="load",
            df_pipeline=self.df_pipeline,
            config=self.cfg,
            data_with_exog=self.data_with_exog,
            exog_feature_names=self.exog_names,
            exo_pred=self.exo_pred,
        )

    def test_exog_train_is_dataframe(self):
        _, exog_train, _ = self._call()
        assert isinstance(exog_train, pd.DataFrame)

    def test_exog_future_is_dataframe(self):
        _, _, exog_future = self._call()
        assert isinstance(exog_future, pd.DataFrame)

    def test_exog_train_columns(self):
        _, exog_train, _ = self._call()
        assert list(exog_train.columns) == self.exog_names

    def test_exog_future_columns(self):
        _, _, exog_future = self._call()
        assert list(exog_future.columns) == self.exog_names

    def test_exog_train_length(self):
        _, exog_train, _ = self._call()
        assert len(exog_train) == 168

    def test_exog_future_length(self):
        _, _, exog_future = self._call()
        assert len(exog_future) == 24

    def test_exog_train_dtype_float32(self):
        _, exog_train, _ = self._call()
        assert (exog_train.dtypes == "float32").all()

    def test_exog_future_dtype_float32(self):
        _, _, exog_future = self._call()
        assert (exog_future.dtypes == "float32").all()

    def test_exog_train_window_matches_config(self):
        _, exog_train, _ = self._call()
        assert exog_train.index[0] == self.cfg.start_train_ts
        assert exog_train.index[-1] == self.cfg.end_train_ts

    def test_exog_future_index_after_train_end(self):
        _, _, exog_future = self._call()
        assert (exog_future.index > self.cfg.end_train_ts).all()

    def test_y_train_still_correct_with_exog(self):
        y_train, _, _ = self._call()
        expected = self.df_pipeline["load"].loc[
            self.cfg.start_train_ts : self.cfg.end_train_ts
        ]
        pd.testing.assert_series_equal(y_train, expected)


# =============================================================================
# TestGetTargetDataExogDisabledButProvided
# =============================================================================


class TestGetTargetDataExogDisabledButProvided:
    """When use_exogenous_features=False, exog outputs must be None even if
    data_with_exog and exo_pred are supplied."""

    def test_exog_train_none_when_flag_false(
        self, df_pipeline, pipeline_idx, data_with_exog_df, exo_pred_df
    ):
        cfg = ConfigMulti(targets=["load"], use_exogenous_features=False)
        cfg.start_train_ts = pipeline_idx[0]
        cfg.end_train_ts = pipeline_idx[-1]
        _, exog_train, _ = get_target_data(
            "load",
            df_pipeline,
            cfg,
            data_with_exog=data_with_exog_df,
            exog_feature_names=["hour_sin", "hour_cos"],
            exo_pred=exo_pred_df,
        )
        assert exog_train is None

    def test_exog_future_none_when_flag_false(
        self, df_pipeline, pipeline_idx, data_with_exog_df, exo_pred_df
    ):
        cfg = ConfigMulti(targets=["load"], use_exogenous_features=False)
        cfg.start_train_ts = pipeline_idx[0]
        cfg.end_train_ts = pipeline_idx[-1]
        _, _, exog_future = get_target_data(
            "load",
            df_pipeline,
            cfg,
            data_with_exog=data_with_exog_df,
            exog_feature_names=["hour_sin", "hour_cos"],
            exo_pred=exo_pred_df,
        )
        assert exog_future is None


# =============================================================================
# TestGetTargetDataExogEnabledButNone
# =============================================================================


class TestGetTargetDataExogEnabledButNone:
    """When use_exogenous_features=True but data_with_exog is None, outputs
    should gracefully return None (pipeline ran without covariates)."""

    def test_exog_train_none_when_data_with_exog_is_none(
        self, df_pipeline, pipeline_idx
    ):
        cfg = ConfigMulti(targets=["load"], use_exogenous_features=True)
        cfg.start_train_ts = pipeline_idx[0]
        cfg.end_train_ts = pipeline_idx[-1]
        _, exog_train, _ = get_target_data("load", df_pipeline, cfg)
        assert exog_train is None

    def test_exog_future_none_when_data_with_exog_is_none(
        self, df_pipeline, pipeline_idx
    ):
        cfg = ConfigMulti(targets=["load"], use_exogenous_features=True)
        cfg.start_train_ts = pipeline_idx[0]
        cfg.end_train_ts = pipeline_idx[-1]
        _, _, exog_future = get_target_data("load", df_pipeline, cfg)
        assert exog_future is None


# =============================================================================
# TestGetTargetDataPackageImport
# =============================================================================


class TestGetTargetDataPackageImport:
    """get_target_data must be importable from the package-level __init__."""

    def test_importable_from_manager(self):
        from spotforecast2_safe.manager import get_target_data as gtd

        assert callable(gtd)

    def test_same_object_as_features_module(self):
        from spotforecast2_safe.manager import get_target_data as gtd_manager

        assert gtd_manager is get_target_data


# =============================================================================
# TestGetTargetDataDocstringExamples
# =============================================================================


class TestGetTargetDataDocstringExamples:
    """Verify that the docstring living examples produce the expected output."""

    def test_example_no_exog(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2024-01-01", periods=168, freq="h", tz="UTC")
        df_pipeline = pd.DataFrame({"load": rng.normal(100, 10, 168)}, index=idx)

        config = ConfigMulti(targets=["load"], use_exogenous_features=False)
        config.start_train_ts = pd.Timestamp("2024-01-01 00:00", tz="UTC")
        config.end_train_ts = pd.Timestamp("2024-01-07 23:00", tz="UTC")

        y_train, exog_train, exog_future = get_target_data(
            target="load",
            df_pipeline=df_pipeline,
            config=config,
        )
        assert len(y_train) == 168
        assert exog_train is None
        assert exog_future is None

    def test_example_with_exog(self):
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
        config.end_train_ts = pd.Timestamp("2024-01-07 23:00", tz="UTC")

        y_train, exog_train, exog_future = get_target_data(
            target="load",
            df_pipeline=df_pipeline,
            config=config,
            data_with_exog=data_with_exog,
            exog_feature_names=["hour_sin", "hour_cos"],
            exo_pred=exo_pred,
        )
        assert len(y_train) == 168
        assert exog_train.shape == (168, 2)
        assert exog_future.shape == (24, 2)
        assert (exog_train.dtypes == "float32").all()
