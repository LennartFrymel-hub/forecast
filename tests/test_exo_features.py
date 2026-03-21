# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for spotforecast2_safe.manager.exo (calendar, holiday, weather features).

Covers:
- get_calendar_features: shape, columns, dtypes, string inputs
- get_day_night_features: shape, columns, values, string inputs
- get_holiday_features: shape, column, holiday marker, string inputs
- Package-level imports from spotforecast2_safe.manager.exo
- Backward-compatible private aliases in n2n_predict_with_covariates
"""

import numpy as np
import pandas as pd
import pytest
from astral import LocationInfo

from spotforecast2_safe.manager.exo import (
    get_calendar_features,
    get_day_night_features,
    get_holiday_features,
)
from spotforecast2_safe.manager.exo.calendar import (
    get_calendar_features as _cal,
    get_day_night_features as _dn,
    get_holiday_features as _hol,
)
from spotforecast2_safe.manager.exo.weather import get_weather_features


# =============================================================================
# Shared fixtures
# =============================================================================


@pytest.fixture
def week_start():
    return pd.Timestamp("2024-01-01", tz="UTC")


@pytest.fixture
def week_end():
    return pd.Timestamp("2024-01-07 23:00", tz="UTC")


@pytest.fixture
def dortmund():
    """LocationInfo for Dortmund, Germany (default in the codebase)."""
    return LocationInfo(latitude=51.5136, longitude=7.4653, timezone="UTC")


@pytest.fixture
def reference_data():
    """Minimal time-series DataFrame for holiday/weather validation helpers."""
    idx = pd.date_range("2023-01-01", periods=500, freq="h", tz="UTC")
    return pd.DataFrame({"power": np.ones(500)}, index=idx)


# =============================================================================
# get_calendar_features
# =============================================================================


class TestGetCalendarFeatures:
    def test_shape(self, week_start, week_end):
        result = get_calendar_features(start=week_start, cov_end=week_end)
        # 7 days * 24 hours = 168 rows
        assert result.shape == (168, 4)

    def test_default_columns(self, week_start, week_end):
        result = get_calendar_features(start=week_start, cov_end=week_end)
        assert list(result.columns) == ["month", "week", "day_of_week", "hour"]

    def test_custom_features(self, week_start, week_end):
        result = get_calendar_features(
            start=week_start,
            cov_end=week_end,
            features_to_extract=["month", "hour"],
        )
        assert list(result.columns) == ["month", "hour"]
        assert result.shape[1] == 2

    def test_index_is_datetime_with_tz(self, week_start, week_end):
        result = get_calendar_features(start=week_start, cov_end=week_end)
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.tz is not None

    def test_month_values(self, week_start, week_end):
        result = get_calendar_features(start=week_start, cov_end=week_end)
        # All in January
        assert result["month"].unique().tolist() == [1]

    def test_hour_range(self, week_start, week_end):
        result = get_calendar_features(start=week_start, cov_end=week_end)
        assert result["hour"].min() == 0
        assert result["hour"].max() == 23

    def test_string_inputs(self):
        result = get_calendar_features(
            start="2024-03-01",
            cov_end="2024-03-07 23:00",
        )
        assert result.shape[0] == 168

    def test_no_nan(self, week_start, week_end):
        result = get_calendar_features(start=week_start, cov_end=week_end)
        assert result.isnull().sum().sum() == 0

    def test_returns_dataframe(self, week_start, week_end):
        result = get_calendar_features(start=week_start, cov_end=week_end)
        assert isinstance(result, pd.DataFrame)

    def test_single_day(self):
        result = get_calendar_features(
            start=pd.Timestamp("2024-06-15", tz="UTC"),
            cov_end=pd.Timestamp("2024-06-15 23:00", tz="UTC"),
        )
        assert result.shape[0] == 24

    def test_module_level_import_same_function(self, week_start, week_end):
        """Public alias from __init__ must be the same object as direct import."""
        assert get_calendar_features is _cal


# =============================================================================
# get_day_night_features
# =============================================================================


class TestGetDayNightFeatures:
    def test_shape(self, week_start, week_end, dortmund):
        result = get_day_night_features(
            start=week_start, cov_end=week_end, location=dortmund
        )
        assert result.shape == (168, 4)

    def test_columns(self, week_start, week_end, dortmund):
        result = get_day_night_features(
            start=week_start, cov_end=week_end, location=dortmund
        )
        assert list(result.columns) == [
            "sunrise_hour",
            "sunset_hour",
            "daylight_hours",
            "is_daylight",
        ]

    def test_is_daylight_binary(self, week_start, week_end, dortmund):
        result = get_day_night_features(
            start=week_start, cov_end=week_end, location=dortmund
        )
        unique_vals = set(result["is_daylight"].unique())
        assert unique_vals.issubset({0, 1})

    def test_sunrise_hour_range(self, week_start, week_end, dortmund):
        result = get_day_night_features(
            start=week_start, cov_end=week_end, location=dortmund
        )
        assert result["sunrise_hour"].between(0, 23).all()

    def test_sunset_hour_range(self, week_start, week_end, dortmund):
        result = get_day_night_features(
            start=week_start, cov_end=week_end, location=dortmund
        )
        assert result["sunset_hour"].between(0, 23).all()

    def test_daylight_hours_consistent(self, week_start, week_end, dortmund):
        result = get_day_night_features(
            start=week_start, cov_end=week_end, location=dortmund
        )
        expected = result["sunset_hour"] - result["sunrise_hour"]
        pd.testing.assert_series_equal(
            result["daylight_hours"], expected, check_names=False
        )

    def test_no_nan(self, week_start, week_end, dortmund):
        result = get_day_night_features(
            start=week_start, cov_end=week_end, location=dortmund
        )
        assert result.isnull().sum().sum() == 0

    def test_string_inputs(self, dortmund):
        result = get_day_night_features(
            start="2024-06-01",
            cov_end="2024-06-07 23:00",
            location=dortmund,
        )
        assert result.shape[0] == 168

    def test_index_tz_aware(self, week_start, week_end, dortmund):
        result = get_day_night_features(
            start=week_start, cov_end=week_end, location=dortmund
        )
        assert result.index.tz is not None

    def test_summer_longer_daylight(self, dortmund):
        """Days in summer should be longer than in winter (Dortmund)."""
        summer = get_day_night_features(
            start=pd.Timestamp("2024-06-21", tz="UTC"),
            cov_end=pd.Timestamp("2024-06-21 23:00", tz="UTC"),
            location=dortmund,
        )
        winter = get_day_night_features(
            start=pd.Timestamp("2024-12-21", tz="UTC"),
            cov_end=pd.Timestamp("2024-12-21 23:00", tz="UTC"),
            location=dortmund,
        )
        assert summer["daylight_hours"].iloc[0] > winter["daylight_hours"].iloc[0]

    def test_module_level_import_same_function(self, week_start, week_end, dortmund):
        assert get_day_night_features is _dn


# =============================================================================
# get_holiday_features
# =============================================================================


class TestGetHolidayFeatures:
    def test_shape(self, week_start, week_end, reference_data):
        result = get_holiday_features(
            data=reference_data,
            start=week_start,
            cov_end=week_end,
            forecast_horizon=24,
        )
        assert result.shape == (168, 1)

    def test_column_name(self, week_start, week_end, reference_data):
        result = get_holiday_features(
            data=reference_data,
            start=week_start,
            cov_end=week_end,
            forecast_horizon=24,
        )
        assert list(result.columns) == ["is_holiday"]

    def test_new_years_day_is_holiday(self, reference_data):
        """Jan 1 is a public holiday in Germany."""
        result = get_holiday_features(
            data=reference_data,
            start=pd.Timestamp("2024-01-01", tz="UTC"),
            cov_end=pd.Timestamp("2024-01-01 23:00", tz="UTC"),
            forecast_horizon=24,
            country_code="DE",
            state="NW",
        )
        assert (result["is_holiday"] == 1).all()

    def test_non_holiday_is_zero(self, reference_data):
        """A regular working day should have is_holiday == 0."""
        result = get_holiday_features(
            data=reference_data,
            start=pd.Timestamp("2024-01-15", tz="UTC"),
            cov_end=pd.Timestamp("2024-01-15 23:00", tz="UTC"),
            forecast_horizon=24,
            country_code="DE",
            state="NW",
        )
        assert (result["is_holiday"] == 0).all()

    def test_no_nan(self, week_start, week_end, reference_data):
        result = get_holiday_features(
            data=reference_data,
            start=week_start,
            cov_end=week_end,
            forecast_horizon=24,
        )
        assert result.isnull().sum().sum() == 0

    def test_binary_values(self, week_start, week_end, reference_data):
        result = get_holiday_features(
            data=reference_data,
            start=week_start,
            cov_end=week_end,
            forecast_horizon=24,
        )
        assert set(result["is_holiday"].unique()).issubset({0, 1})

    def test_integer_dtype(self, week_start, week_end, reference_data):
        result = get_holiday_features(
            data=reference_data,
            start=week_start,
            cov_end=week_end,
            forecast_horizon=24,
        )
        assert result["is_holiday"].dtype in (np.int64, np.int32, int)

    def test_string_inputs(self, reference_data):
        result = get_holiday_features(
            data=reference_data,
            start="2024-01-01",
            cov_end="2024-01-07 23:00",
            forecast_horizon=24,
        )
        assert result.shape[0] == 168

    def test_index_tz_aware(self, week_start, week_end, reference_data):
        result = get_holiday_features(
            data=reference_data,
            start=week_start,
            cov_end=week_end,
            forecast_horizon=24,
        )
        assert result.index.tz is not None

    def test_module_level_import_same_function(
        self, week_start, week_end, reference_data
    ):
        assert get_holiday_features is _hol


# =============================================================================
# Package-level __init__ imports
# =============================================================================


class TestExoPackageImports:
    def test_all_symbols_importable(self):
        from spotforecast2_safe.manager.exo import (  # noqa: F401
            get_calendar_features,
            get_day_night_features,
            get_holiday_features,
            get_weather_features,
        )

    def test_all_declared_in___all__(self):
        import spotforecast2_safe.manager.exo as exo_module

        assert "get_calendar_features" in exo_module.__all__
        assert "get_day_night_features" in exo_module.__all__
        assert "get_holiday_features" in exo_module.__all__
        assert "get_weather_features" in exo_module.__all__

    def test_manager_exports_exo_functions(self):
        from spotforecast2_safe.manager import (  # noqa: F401
            get_calendar_features,
            get_day_night_features,
            get_holiday_features,
            get_weather_features,
        )


# =============================================================================
# Backward-compatible private aliases in n2n_predict_with_covariates
# =============================================================================


class TestBackwardCompatibleAliases:
    def test_private_calendar_alias(self):
        from spotforecast2_safe.processing.n2n_predict_with_covariates import (
            _get_calendar_features,
        )

        assert _get_calendar_features is get_calendar_features

    def test_private_day_night_alias(self):
        from spotforecast2_safe.processing.n2n_predict_with_covariates import (
            _get_day_night_features,
        )

        assert _get_day_night_features is get_day_night_features

    def test_private_holiday_alias(self):
        from spotforecast2_safe.processing.n2n_predict_with_covariates import (
            _get_holiday_features,
        )

        assert _get_holiday_features is get_holiday_features

    def test_private_weather_alias(self):
        from spotforecast2_safe.processing.n2n_predict_with_covariates import (
            _get_weather_features,
        )

        assert _get_weather_features is get_weather_features
