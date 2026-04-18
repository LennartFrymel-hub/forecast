# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for WeatherClient and WeatherService in spotforecast2_safe.

Verifies that:
- WeatherClient and WeatherService are importable from spotforecast2_safe.weather.
- WeatherClient instantiation and core API methods work correctly.
- WeatherService caching, hybrid-fetch, and fallback logic work correctly.
"""

from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import pandas as pd
import pytest

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
    """Return a mock requests.Response for Open-Meteo with n_hours rows."""
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
# Import tests
# ---------------------------------------------------------------------------


class TestImports:
    """WeatherClient and WeatherService are importable from spotforecast2_safe.weather."""

    def test_import_weather_client(self):
        """WeatherClient is importable from spotforecast2_safe.weather."""
        from spotforecast2_safe.weather import WeatherClient  # noqa: F401

    def test_import_weather_service(self):
        """WeatherService is importable from spotforecast2_safe.weather."""
        from spotforecast2_safe.weather import WeatherService  # noqa: F401

    def test_import_both(self):
        """WeatherClient and WeatherService are importable together."""

    def test_dunder_all(self):
        """spotforecast2_safe.weather.__all__ exposes both classes."""
        from spotforecast2_safe.weather import __all__ as weather_all

        assert "WeatherClient" in weather_all
        assert "WeatherService" in weather_all

    def test_weather_service_is_subclass(self):
        """WeatherService is a subclass of WeatherClient."""
        from spotforecast2_safe.weather import WeatherClient, WeatherService

        assert issubclass(WeatherService, WeatherClient)


# ---------------------------------------------------------------------------
# WeatherClient unit tests
# ---------------------------------------------------------------------------


class TestWeatherClientInstantiation:
    """WeatherClient can be instantiated with valid lat/lon."""

    def test_basic_instantiation(self):
        """WeatherClient stores latitude and longitude on init."""
        from spotforecast2_safe.weather import WeatherClient

        client = WeatherClient(latitude=52.52, longitude=13.405)
        assert client.latitude == 52.52
        assert client.longitude == 13.405

    def test_session_created(self):
        """WeatherClient creates a requests Session with retry logic."""
        import requests

        from spotforecast2_safe.weather import WeatherClient

        client = WeatherClient(latitude=51.0, longitude=7.0)
        assert isinstance(client._session, requests.Session)

    def test_constants(self):
        """ARCHIVE_BASE_URL and FORECAST_BASE_URL are non-empty strings."""
        from spotforecast2_safe.weather import WeatherClient

        assert WeatherClient.ARCHIVE_BASE_URL.startswith("https://")
        assert WeatherClient.FORECAST_BASE_URL.startswith("https://")

    def test_hourly_params_non_empty(self):
        """HOURLY_PARAMS contains at least the standard 15 parameters."""
        from spotforecast2_safe.weather import WeatherClient

        assert len(WeatherClient.HOURLY_PARAMS) >= 15
        assert "temperature_2m" in WeatherClient.HOURLY_PARAMS
        assert "wind_speed_10m" in WeatherClient.HOURLY_PARAMS


class TestWeatherClientFetchArchive:
    """WeatherClient.fetch_archive parses API responses into DataFrames."""

    @patch("requests.Session.get")
    def test_fetch_archive_returns_dataframe(self, mock_get):
        """fetch_archive returns a DataFrame with expected columns."""
        from spotforecast2_safe.weather import WeatherClient

        mock_get.return_value = _make_api_response(48)
        client = WeatherClient(latitude=52.52, longitude=13.405)
        start = pd.Timestamp("2023-01-01", tz="UTC")
        end = pd.Timestamp("2023-01-02 23:00", tz="UTC")

        df = client.fetch_archive(start, end, timezone="UTC")

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 48
        assert "temperature_2m" in df.columns
        assert df.index.name == "datetime"

    @patch("requests.Session.get")
    def test_fetch_archive_all_params_present(self, mock_get):
        """fetch_archive returns all 15 standard hourly parameters."""
        from spotforecast2_safe.weather import WeatherClient

        mock_get.return_value = _make_api_response(24)
        client = WeatherClient(latitude=51.0, longitude=7.0)
        start = pd.Timestamp("2023-06-01", tz="UTC")
        end = pd.Timestamp("2023-06-01 23:00", tz="UTC")

        df = client.fetch_archive(start, end)

        for param in HOURLY_PARAMS:
            assert param in df.columns, f"Missing column: {param}"

    @patch("requests.Session.get")
    def test_fetch_archive_http_error_raises(self, mock_get):
        """fetch_archive propagates HTTP errors as exceptions."""
        import requests as req

        from spotforecast2_safe.weather import WeatherClient

        mock_get.side_effect = req.exceptions.ConnectionError("unreachable")
        client = WeatherClient(latitude=51.0, longitude=7.0)

        with pytest.raises(req.exceptions.RequestException):
            client.fetch_archive(
                pd.Timestamp("2023-01-01", tz="UTC"),
                pd.Timestamp("2023-01-02", tz="UTC"),
            )

    @patch("requests.Session.get")
    def test_fetch_archive_api_error_key_raises(self, mock_get):
        """fetch_archive raises ValueError when API returns an error payload."""
        from spotforecast2_safe.weather import WeatherClient

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"error": True, "reason": "Bad request"}
        mock_get.return_value = mock_resp

        client = WeatherClient(latitude=51.0, longitude=7.0)
        with pytest.raises(ValueError, match="Open-Meteo API error"):
            client.fetch_archive(
                pd.Timestamp("2023-01-01", tz="UTC"),
                pd.Timestamp("2023-01-02", tz="UTC"),
            )

    @patch("requests.Session.get")
    def test_fetch_archive_empty_hourly_raises(self, mock_get):
        """fetch_archive raises ValueError when hourly data is absent."""
        from spotforecast2_safe.weather import WeatherClient

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {}
        mock_get.return_value = mock_resp

        client = WeatherClient(latitude=51.0, longitude=7.0)
        with pytest.raises(ValueError, match="No hourly data"):
            client.fetch_archive(
                pd.Timestamp("2023-01-01", tz="UTC"),
                pd.Timestamp("2023-01-02", tz="UTC"),
            )


class TestWeatherClientFetchForecast:
    """WeatherClient.fetch_forecast fetches future data from the Forecast API."""

    @patch("requests.Session.get")
    def test_fetch_forecast_returns_dataframe(self, mock_get):
        """fetch_forecast returns a non-empty DataFrame."""
        from spotforecast2_safe.weather import WeatherClient

        mock_get.return_value = _make_api_response(48)
        client = WeatherClient(latitude=51.5, longitude=7.5)

        df = client.fetch_forecast(days_ahead=2)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 48

    @patch("requests.Session.get")
    def test_fetch_forecast_uses_forecast_url(self, mock_get):
        """fetch_forecast calls the Forecast API (not the Archive API)."""
        from spotforecast2_safe.weather import WeatherClient

        mock_get.return_value = _make_api_response(24)
        client = WeatherClient(latitude=51.5, longitude=7.5)
        client.fetch_forecast(days_ahead=1)

        call_url = mock_get.call_args[0][0]
        parsed = urlparse(call_url)
        assert parsed.hostname == "api.open-meteo.com"
        assert "archive" not in parsed.path


# ---------------------------------------------------------------------------
# WeatherService unit tests
# ---------------------------------------------------------------------------


class TestWeatherServiceInstantiation:
    """WeatherService inherits from WeatherClient and adds caching."""

    def test_is_subclass_of_weather_client(self):
        """WeatherService is a subclass of WeatherClient."""
        from spotforecast2_safe.weather import WeatherClient, WeatherService

        assert issubclass(WeatherService, WeatherClient)

    def test_default_cache_path_is_none(self):
        """WeatherService.cache_path defaults to None."""
        from spotforecast2_safe.weather import WeatherService

        svc = WeatherService(latitude=51.0, longitude=7.0)
        assert svc.cache_path is None

    def test_custom_cache_path_stored(self, tmp_path):
        """WeatherService stores the provided cache_path."""
        from spotforecast2_safe.weather import WeatherService

        cp = tmp_path / "test_cache.parquet"
        svc = WeatherService(latitude=51.0, longitude=7.0, cache_path=cp)
        assert svc.cache_path == cp

    def test_use_forecast_default_true(self):
        """WeatherService.use_forecast defaults to True."""
        from spotforecast2_safe.weather import WeatherService

        svc = WeatherService(latitude=51.0, longitude=7.0)
        assert svc.use_forecast is True


class TestWeatherServiceGetDataframe:
    """WeatherService.get_dataframe orchestrates fetch, cache, and finalize."""

    @patch("requests.Session.get")
    def test_returns_dataframe_in_range(self, mock_get):
        """get_dataframe returns a DataFrame covering the requested range."""
        from spotforecast2_safe.weather import WeatherService

        mock_get.return_value = _make_api_response(72, start="2023-01-01")
        svc = WeatherService(latitude=51.0, longitude=7.0)

        df = svc.get_dataframe(
            start="2023-01-01",
            end="2023-01-03",
            timezone="UTC",
            fallback_on_failure=False,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1

    @patch("requests.Session.get")
    def test_no_nulls_in_output(self, mock_get):
        """get_dataframe forward-fills so the output has no NaN values."""
        from spotforecast2_safe.weather import WeatherService

        mock_get.return_value = _make_api_response(48, start="2023-06-01")
        svc = WeatherService(latitude=51.0, longitude=7.0)

        df = svc.get_dataframe(
            start="2023-06-01",
            end="2023-06-02",
            timezone="UTC",
            fallback_on_failure=False,
        )

        assert not df.isnull().any().any()

    @patch("requests.Session.get")
    def test_cache_is_written(self, mock_get, tmp_path):
        """get_dataframe writes a parquet cache file when cache_path is set."""
        from spotforecast2_safe.weather import WeatherService

        mock_get.return_value = _make_api_response(48, start="2023-01-01")
        cp = tmp_path / "weather_cache.parquet"
        svc = WeatherService(latitude=51.0, longitude=7.0, cache_path=cp)

        svc.get_dataframe(
            start="2023-01-01",
            end="2023-01-02",
            timezone="UTC",
            fallback_on_failure=False,
        )

        assert cp.exists(), "Cache file was not created."

    @patch("requests.Session.get")
    def test_cache_is_read_on_second_call(self, mock_get, tmp_path):
        """get_dataframe uses cached data on the second call (no extra API hit)."""
        from spotforecast2_safe.weather import WeatherService

        mock_get.return_value = _make_api_response(72, start="2023-01-01")
        cp = tmp_path / "weather_cache.parquet"
        svc = WeatherService(latitude=51.0, longitude=7.0, cache_path=cp)

        svc.get_dataframe(
            start="2023-01-01",
            end="2023-01-02",
            timezone="UTC",
            fallback_on_failure=False,
        )
        first_call_count = mock_get.call_count

        svc.get_dataframe(
            start="2023-01-01",
            end="2023-01-02",
            timezone="UTC",
            fallback_on_failure=False,
        )

        assert (
            mock_get.call_count == first_call_count
        ), "API was called again even though cache covers the requested range."

    @patch("requests.Session.get")
    def test_fallback_on_api_failure(self, mock_get, tmp_path):
        """get_dataframe falls back to cached data when the API call fails."""
        from spotforecast2_safe.weather import WeatherService

        cp = tmp_path / "fallback_cache.parquet"
        svc = WeatherService(latitude=51.0, longitude=7.0, cache_path=cp)

        mock_get.return_value = _make_api_response(48, start="2023-01-01")
        svc.get_dataframe(
            start="2023-01-01",
            end="2023-01-02",
            timezone="UTC",
            fallback_on_failure=False,
        )

        import requests as req

        mock_get.side_effect = req.exceptions.ConnectionError("down")

        df = svc.get_dataframe(
            start="2023-01-03",
            end="2023-01-04",
            timezone="UTC",
            fallback_on_failure=True,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) >= 1

    @patch("requests.Session.get")
    def test_api_failure_no_cache_raises(self, mock_get):
        """get_dataframe raises when fallback is False and API fails."""
        import requests as req

        from spotforecast2_safe.weather import WeatherService

        mock_get.side_effect = req.exceptions.ConnectionError("unreachable")
        svc = WeatherService(latitude=51.0, longitude=7.0)

        with pytest.raises(Exception):
            svc.get_dataframe(
                start="2023-01-01",
                end="2023-01-02",
                timezone="UTC",
                fallback_on_failure=False,
            )


class TestWeatherServiceFinalize:
    """WeatherService._finalize_df resamples, and raises or fills gaps."""

    def test_finalize_default_raises_on_nan(self):
        """Default ``fill_missing=False`` refuses to return NaN rows."""
        from spotforecast2_safe.weather import WeatherService

        svc = WeatherService(latitude=51.0, longitude=7.0)
        idx = pd.date_range("2023-01-01", periods=5, freq="h", tz="UTC")
        df = pd.DataFrame({"temperature_2m": [1.0, None, None, 4.0, 5.0]}, index=idx)

        with pytest.raises(ValueError, match="missing row"):
            svc._finalize_df(df, freq="h", timezone="UTC")

    def test_finalize_fill_missing_true_restores_legacy(self):
        """``fill_missing=True`` opts into forward/back-fill behavior."""
        from spotforecast2_safe.weather import WeatherService

        svc = WeatherService(latitude=51.0, longitude=7.0)
        idx = pd.date_range("2023-01-01", periods=5, freq="h", tz="UTC")
        df = pd.DataFrame({"temperature_2m": [1.0, None, None, 4.0, 5.0]}, index=idx)

        result = svc._finalize_df(df, freq="h", timezone="UTC", fill_missing=True)
        assert not result.isnull().any().any()

    def test_finalize_hourly_does_not_resample(self):
        """_finalize_df leaves hourly data unchanged when freq='h'."""
        from spotforecast2_safe.weather import WeatherService

        svc = WeatherService(latitude=51.0, longitude=7.0)
        idx = pd.date_range("2023-01-01", periods=6, freq="h", tz="UTC")
        df = pd.DataFrame({"temperature_2m": list(range(6))}, index=idx)

        result = svc._finalize_df(df, freq="h", timezone="UTC")
        assert len(result) == 6
