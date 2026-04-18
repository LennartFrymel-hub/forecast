# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for unified weather caching via cache_home parameter.

Verifies that:
- fetch_weather_data enables caching when cache_home is provided.
- fetch_weather_data disables caching when cache_home is None.
- get_weather_features passes cache_home through to fetch_weather_data.
- No 'cached' boolean parameter exists in the public API.
"""

import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from spotforecast2_safe.data.fetch_data import fetch_weather_data
from spotforecast2_safe.manager.exo.weather import get_weather_features

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HOURLY_PARAMS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "snowfall",
    "weather_code",
    "pressure_msl",
    "surface_pressure",
    "cloud_cover",
    "cloud_cover_low",
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_gusts_10m",
]


def _make_api_response(n_hours: int = 24, start: str = "2023-01-01") -> MagicMock:
    """Return a mock requests.Response for Open-Meteo."""
    times = (
        pd.date_range(start, periods=n_hours, freq="h")
        .strftime("%Y-%m-%dT%H:%M")
        .tolist()
    )
    hourly: dict = {"time": times}
    for param in HOURLY_PARAMS:
        hourly[param] = [1.0] * n_hours

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"hourly": hourly}
    return mock_resp


# ---------------------------------------------------------------------------
# Signature sanity — no 'cached' parameter
# ---------------------------------------------------------------------------


class TestNoCachedParameter:
    """The old boolean 'cached' parameter must not exist."""

    def test_fetch_weather_data_no_cached_param(self):
        """fetch_weather_data must not accept a 'cached' keyword argument."""
        sig = inspect.signature(fetch_weather_data)
        assert "cached" not in sig.parameters

    def test_get_weather_features_no_cached_param(self):
        """get_weather_features must not accept a 'cached' keyword argument."""
        sig = inspect.signature(get_weather_features)
        assert "cached" not in sig.parameters

    def test_fetch_weather_data_has_cache_home(self):
        """fetch_weather_data must accept cache_home."""
        sig = inspect.signature(fetch_weather_data)
        assert "cache_home" in sig.parameters

    def test_get_weather_features_has_cache_home(self):
        """get_weather_features must accept cache_home."""
        sig = inspect.signature(get_weather_features)
        assert "cache_home" in sig.parameters


# ---------------------------------------------------------------------------
# fetch_weather_data caching behaviour
# ---------------------------------------------------------------------------


class TestFetchWeatherDataCaching:
    """fetch_weather_data caching is controlled solely by cache_home."""

    def test_cache_home_none_means_no_cache(self):
        """When cache_home is None, WeatherService gets cache_path=None."""
        with patch("spotforecast2_safe.data.fetch_data.WeatherService") as mock_cls:
            mock_svc = MagicMock()
            mock_svc.get_dataframe.return_value = pd.DataFrame()
            mock_cls.return_value = mock_svc

            fetch_weather_data(cov_start="2023-01-01", cov_end="2023-01-02")

            init_kwargs = mock_cls.call_args[1]
            assert init_kwargs["cache_path"] is None

    def test_cache_home_provided_means_caching(self):
        """When cache_home is provided, WeatherService gets a cache path."""
        with patch("spotforecast2_safe.data.fetch_data.WeatherService") as mock_cls:
            with patch(
                "spotforecast2_safe.data.fetch_data.get_cache_home"
            ) as mock_home:
                mock_home.return_value = Path("/my/cache")
                mock_svc = MagicMock()
                mock_svc.get_dataframe.return_value = pd.DataFrame()
                mock_cls.return_value = mock_svc

                fetch_weather_data(
                    cov_start="2023-01-01",
                    cov_end="2023-01-02",
                    cache_home="/my/cache",
                )

                init_kwargs = mock_cls.call_args[1]
                assert init_kwargs["cache_path"] == Path(
                    "/my/cache/weather_cache.parquet"
                )

    @patch("requests.Session.get")
    def test_cache_file_written(self, mock_get, tmp_path):
        """A parquet cache file is created when cache_home is provided."""
        mock_get.return_value = _make_api_response(48)
        cache_file = tmp_path / "weather_cache.parquet"

        fetch_weather_data(
            cov_start="2023-01-01",
            cov_end="2023-01-02",
            cache_home=tmp_path,
        )

        assert cache_file.exists()

    @patch("requests.Session.get")
    def test_no_cache_file_without_cache_home(self, mock_get, tmp_path):
        """No cache file is created when cache_home is None."""
        mock_get.return_value = _make_api_response(24)

        fetch_weather_data(
            cov_start="2023-01-01",
            cov_end="2023-01-01",
        )

        # Nothing should appear in tmp_path (not used as cache)
        assert not list(tmp_path.iterdir())

    @patch("requests.Session.get")
    def test_cache_reused_on_second_call(self, mock_get, tmp_path):
        """Second call with same range does not hit the API again."""
        mock_get.return_value = _make_api_response(72, start="2023-01-01")

        fetch_weather_data(
            cov_start="2023-01-01",
            cov_end="2023-01-02",
            cache_home=tmp_path,
        )
        first_count = mock_get.call_count

        fetch_weather_data(
            cov_start="2023-01-01",
            cov_end="2023-01-02",
            cache_home=tmp_path,
        )

        assert mock_get.call_count == first_count


# ---------------------------------------------------------------------------
# get_weather_features passes cache_home through
# ---------------------------------------------------------------------------


class TestGetWeatherFeaturesCaching:
    """get_weather_features forwards cache_home to fetch_weather_data."""

    def test_cache_home_forwarded(self):
        """cache_home is passed through to fetch_weather_data."""
        idx = pd.date_range("2023-01-01", periods=48, freq="h", tz="UTC")
        data = pd.DataFrame({"value": range(48)}, index=idx)

        with patch(
            "spotforecast2_safe.manager.exo.weather.fetch_weather_data"
        ) as mock_fetch:
            weather_df = pd.DataFrame(
                {p: [1.0] * 48 for p in HOURLY_PARAMS},
                index=idx,
            )
            mock_fetch.return_value = weather_df

            get_weather_features(
                data=data,
                start="2023-01-01",
                cov_end="2023-01-02 23:00",
                forecast_horizon=24,
                cache_home="/some/path",
            )

            call_kwargs = mock_fetch.call_args[1]
            assert call_kwargs["cache_home"] == "/some/path"

    def test_cache_home_none_by_default(self):
        """cache_home defaults to None in get_weather_features."""
        idx = pd.date_range("2023-01-01", periods=48, freq="h", tz="UTC")
        data = pd.DataFrame({"value": range(48)}, index=idx)

        with patch(
            "spotforecast2_safe.manager.exo.weather.fetch_weather_data"
        ) as mock_fetch:
            weather_df = pd.DataFrame(
                {p: [1.0] * 48 for p in HOURLY_PARAMS},
                index=idx,
            )
            mock_fetch.return_value = weather_df

            get_weather_features(
                data=data,
                start="2023-01-01",
                cov_end="2023-01-02 23:00",
                forecast_horizon=24,
            )

            call_kwargs = mock_fetch.call_args[1]
            assert call_kwargs["cache_home"] is None
