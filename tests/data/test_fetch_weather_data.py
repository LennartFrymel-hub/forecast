# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from spotforecast2_safe.data.fetch_data import fetch_weather_data


@pytest.fixture
def mock_weather_response():
    """Create a mock successful response from Open-Meteo."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200

    # Generate 24 hours of dummy data
    times = (
        pd.date_range("2023-01-01", periods=24, freq="h")
        .strftime("%Y-%m-%dT%H:00")
        .tolist()
    )
    dummy_data = {
        "hourly": {
            "time": times,
            "temperature_2m": [20.0] * 24,
            "relative_humidity_2m": [50.0] * 24,
            "precipitation": [0.0] * 24,
            "rain": [0.0] * 24,
            "snowfall": [0.0] * 24,
            "weather_code": [0] * 24,
            "pressure_msl": [1013.0] * 24,
            "surface_pressure": [1013.0] * 24,
            "cloud_cover": [0] * 24,
            "cloud_cover_low": [0] * 24,
            "cloud_cover_mid": [0] * 24,
            "cloud_cover_high": [0] * 24,
            "wind_speed_10m": [5.0] * 24,
            "wind_direction_10m": [180.0] * 24,
            "wind_gusts_10m": [10.0] * 24,
        }
    }
    mock_resp.json.return_value = dummy_data
    return mock_resp


@patch("requests.Session.get")
def test_fetch_weather_data_success(mock_get, mock_weather_response):
    """Test successful weather data fetch with API mocking."""
    mock_get.return_value = mock_weather_response

    start = "2023-01-01T00:00"
    end = "2023-01-01T23:00"

    df = fetch_weather_data(cov_start=start, cov_end=end)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 24
    assert "temperature_2m" in df.columns
    assert not df.isnull().any().any()
    assert df.index.tz is not None


@patch("requests.Session.get")
def test_fetch_weather_data_api_failure_no_fallback(mock_get):
    """Test API failure when fallback is disabled."""
    mock_get.side_effect = Exception("Connection Error")

    with pytest.raises(Exception):
        fetch_weather_data(
            cov_start="2023-01-01",
            cov_end="2023-01-01",
            fallback_on_failure=False,
        )


@patch("requests.Session.get")
def test_fetch_weather_data_fallback_logic(mock_get, tmp_path):
    """Test fallback logic: repeat last 24h of cached data if API fails."""
    # 1. Setup a cache file with some data
    cache_file = tmp_path / "weather_cache.parquet"
    idx = pd.date_range("2023-01-01", periods=24, freq="h", tz="UTC")
    cached_df = pd.DataFrame({"temperature_2m": [10.0] * 24}, index=idx)
    cached_df.to_parquet(cache_file)

    # 2. Mock API failure
    mock_get.side_effect = Exception("API Offline")

    # 3. Fetch data for a NEW range, expecting fallback to repeat cache
    with patch(
        "spotforecast2_safe.data.fetch_data.get_cache_home", return_value=tmp_path
    ):
        df = fetch_weather_data(
            cov_start="2023-01-02T00:00",
            cov_end="2023-01-02T05:00",
            fallback_on_failure=True,
            cache_home=tmp_path,
        )

    # Should have 6 points (0 to 5)
    assert len(df) == 6
    # Should repeat the value from cache (10.0)
    assert (df["temperature_2m"] == 10.0).all()


def test_fetch_weather_data_cache_integrity(tmp_path, mock_weather_response):
    """Test that data is correctly written to and read from cache."""
    cache_file = tmp_path / "weather_cache.parquet"

    with patch("requests.Session.get", return_value=mock_weather_response):
        with patch(
            "spotforecast2_safe.data.fetch_data.get_cache_home", return_value=tmp_path
        ):
            # Fetch and populate cache
            fetch_weather_data(
                cov_start="2023-01-01T00:00", cov_end="2023-01-01T23:00", cache_home=tmp_path
            )

            assert cache_file.exists()

            # Fetch again without API call (implicitly via cache logic if date range matches)
            # Actually WeatherService.get_dataframe checks cache first.
            with patch("requests.Session.get") as mock_get_fail:
                mock_get_fail.side_effect = Exception(
                    "Should not call API if fully cached"
                )
                df = fetch_weather_data(
                    cov_start="2023-01-01T00:00",
                    cov_end="2023-01-01T23:00",
                    cache_home=tmp_path,
                )
                assert len(df) == 24
                assert not mock_get_fail.called


def test_fetch_weather_data_alignment():
    """Verify that the resulting index exactly matches requested range and frequency."""
    # We'll mock the internal _fetch_hybrid to avoid networking but keep the alignment logic
    with patch(
        "spotforecast2_safe.weather.weather_client.WeatherService._fetch_hybrid"
    ) as mock_hybrid:
        start = "2023-01-01T00:00:00Z"
        end = "2023-01-01T10:00:00Z"
        freq = "2h"

        # API returns hourly
        idx_hourly = pd.date_range("2023-01-01", periods=11, freq="h", tz="UTC")
        mock_df = pd.DataFrame({"temp": np.arange(11)}, index=idx_hourly)
        mock_hybrid.return_value = mock_df

        df = fetch_weather_data(cov_start=start, cov_end=end, freq=freq)

        # Expected index: 0, 2, 4, 6, 8, 10 -> 6 points
        assert len(df) == 6
        assert df.index.freqstr == "2h"
        assert df.index[0] == pd.Timestamp(start)
        assert df.index[-1] == pd.Timestamp(end)
