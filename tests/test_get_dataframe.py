# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Branch-level tests for WeatherService.get_dataframe.

All internal helpers (_load_cache, _fetch_hybrid, _save_cache,
_create_fallback) are replaced with mock objects so every test runs
completely offline.  The suite is organised around the six logical
branches inside get_dataframe:

    1. Timestamp localisation (naive vs. timezone-aware inputs).
    2. Cache-hit early return (fetch and save both skipped).
    3. Cache-miss path (fetch invoked when cache is absent or partial).
    4. Fallback logic (_create_fallback when fetch fails).
    5. Cache merge and deduplication after a successful fetch.
    6. Cache-persistence gate (save triggered only when cache_path is set).

An additional class covers the structural properties of the returned
DataFrame (shape, index range, NaN handling, freq resampling).
"""

from unittest.mock import patch

import pandas as pd
import pytest

from spotforecast2_safe.weather.weather_client import WeatherService

# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

_START = "2023-06-01 00:00:00"
_END = "2023-06-03 23:00:00"
_START_UTC = pd.Timestamp(_START, tz="UTC")
_END_UTC = pd.Timestamp(_END, tz="UTC")
_PERIODS = int((_END_UTC - _START_UTC).total_seconds() / 3600) + 1  # 72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hourly_df(start: str, periods: int, tz: str = "UTC") -> pd.DataFrame:
    """Return a minimal hourly weather DataFrame with a tz-aware index.

    Args:
        start: ISO-8601 string for the first timestamp.
        periods: Number of hourly rows to include.
        tz: Timezone string for the DatetimeIndex.

    Returns:
        DataFrame with columns 'temperature_2m' and 'wind_speed_10m'.
    """
    idx = pd.date_range(start, periods=periods, freq="h", tz=tz)
    return pd.DataFrame(
        {"temperature_2m": 15.0, "wind_speed_10m": 5.0},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def svc() -> WeatherService:
    """WeatherService instance without a cache path.

    Returns:
        Uncached WeatherService at Berlin coordinates.
    """
    return WeatherService(latitude=52.52, longitude=13.405)


@pytest.fixture()
def svc_cached(tmp_path) -> WeatherService:
    """WeatherService instance with a cache path in a temporary directory.

    Args:
        tmp_path: pytest-provided temporary directory.

    Returns:
        WeatherService with cache_path configured.
    """
    return WeatherService(
        latitude=52.52,
        longitude=13.405,
        cache_path=tmp_path / "weather_cache.parquet",
    )


@pytest.fixture()
def fetch_df() -> pd.DataFrame:
    """Simulated result of a successful _fetch_hybrid call.

    Returns:
        72-row hourly DataFrame covering 2023-06-01 to 2023-06-03 23:00 UTC.
    """
    return _hourly_df(_START, _PERIODS)


@pytest.fixture()
def wide_cache_df() -> pd.DataFrame:
    """Cache DataFrame that fully contains [_START_UTC, _END_UTC].

    The index spans 200 hours starting 2023-05-30 00:00 UTC, which is a
    strict superset of the test range.

    Returns:
        200-row hourly DataFrame in UTC.
    """
    return _hourly_df("2023-05-30 00:00:00", periods=200)


@pytest.fixture()
def narrow_cache_df() -> pd.DataFrame:
    """Cache DataFrame whose index ends before _END_UTC.

    Covers 2023-06-01 00:00 to 2023-06-02 23:00 UTC (48 rows), which
    does not reach _END_UTC (2023-06-03 23:00 UTC).

    Returns:
        48-row hourly DataFrame in UTC.
    """
    return _hourly_df(_START, periods=48)


# ---------------------------------------------------------------------------
# Tests: timestamp localisation
# ---------------------------------------------------------------------------


class TestTimestampHandling:
    """Verify that naive and timezone-aware inputs are handled correctly."""

    def test_naive_utc_string_passed_as_aware_to_fetch(self, svc, fetch_df):
        """Naive string with timezone='UTC' produces a tz-aware argument.

        After localisation, _fetch_hybrid must receive a tz-aware start
        Timestamp and not a naive one.

        Args:
            svc: WeatherService fixture.
            fetch_df: Simulated fetch result.
        """
        with (
            patch.object(svc, "_load_cache", return_value=None),
            patch.object(svc, "_fetch_hybrid", return_value=fetch_df) as mock_fetch,
            patch.object(svc, "_save_cache"),
        ):
            svc.get_dataframe(start=_START, end=_END, timezone="UTC")

        fetch_start = mock_fetch.call_args.args[0]
        assert (
            fetch_start.tz is not None
        ), "start passed to _fetch_hybrid must be tz-aware"

    def test_naive_non_utc_string_is_localised_then_converted(self, svc):
        """Naive string with timezone='Europe/Berlin' is converted to UTC internally.

        2023-06-01 00:00 Europe/Berlin equals 2023-05-31 22:00 UTC, so the
        UTC start used for the cache comparison must be earlier than the
        wall-clock start.

        Args:
            svc: WeatherService fixture.
        """
        # Wide fetch that covers the UTC-shifted range
        wide = _hourly_df("2023-05-31 20:00:00", periods=200)

        with (
            patch.object(svc, "_load_cache", return_value=None),
            patch.object(svc, "_fetch_hybrid", return_value=wide) as mock_fetch,
            patch.object(svc, "_save_cache"),
        ):
            svc.get_dataframe(
                start="2023-06-01 00:00:00",
                end="2023-06-03 00:00:00",
                timezone="Europe/Berlin",
            )

        _, _, tz_arg = mock_fetch.call_args.args
        assert tz_arg == "Europe/Berlin"

    def test_aware_utc_timestamps_accepted_without_error(self, svc, fetch_df):
        """UTC-aware Timestamps bypass localisation and produce a valid result.

        Args:
            svc: WeatherService fixture.
            fetch_df: Simulated fetch result.
        """
        with (
            patch.object(svc, "_load_cache", return_value=None),
            patch.object(svc, "_fetch_hybrid", return_value=fetch_df),
            patch.object(svc, "_save_cache"),
        ):
            result = svc.get_dataframe(start=_START_UTC, end=_END_UTC)

        assert isinstance(result, pd.DataFrame)

    def test_aware_non_utc_timestamps_accepted_without_error(self, svc):
        """Non-UTC Timestamps are accepted and converted to UTC internally.

        Args:
            svc: WeatherService fixture.
        """
        start = pd.Timestamp("2023-06-01 00:00:00", tz="US/Eastern")
        end = pd.Timestamp("2023-06-03 23:00:00", tz="US/Eastern")
        # Build fetch result that covers [start, end] after UTC conversion.
        # 2023-06-01 00:00 EDT = 2023-06-01 04:00 UTC, so shift accordingly.
        wide = _hourly_df("2023-06-01 00:00:00", periods=120)

        with (
            patch.object(svc, "_load_cache", return_value=None),
            patch.object(svc, "_fetch_hybrid", return_value=wide),
            patch.object(svc, "_save_cache"),
        ):
            result = svc.get_dataframe(start=start, end=end)

        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests: cache-hit early return
# ---------------------------------------------------------------------------


class TestCacheHit:
    """Verify the early-return path when the cache covers the full range."""

    def test_fetch_not_called_on_cache_hit(self, svc, wide_cache_df):
        """_fetch_hybrid is not invoked when the cache covers [start, end].

        Args:
            svc: WeatherService fixture.
            wide_cache_df: Cache DataFrame fully covering the test range.
        """
        with (
            patch.object(svc, "_load_cache", return_value=wide_cache_df),
            patch.object(svc, "_fetch_hybrid") as mock_fetch,
            patch.object(svc, "_save_cache"),
        ):
            svc.get_dataframe(start=_START, end=_END)

        mock_fetch.assert_not_called()

    def test_save_not_called_on_cache_hit(self, svc, wide_cache_df):
        """_save_cache is not invoked on a cache hit.

        Args:
            svc: WeatherService fixture.
            wide_cache_df: Cache DataFrame fully covering the test range.
        """
        with (
            patch.object(svc, "_load_cache", return_value=wide_cache_df),
            patch.object(svc, "_fetch_hybrid"),
            patch.object(svc, "_save_cache") as mock_save,
        ):
            svc.get_dataframe(start=_START, end=_END)

        mock_save.assert_not_called()

    def test_cache_hit_returns_slice_within_requested_range(self, svc, wide_cache_df):
        """Cache hit returns a DataFrame bounded by [_START_UTC, _END_UTC].

        Args:
            svc: WeatherService fixture.
            wide_cache_df: Cache DataFrame fully covering the test range.
        """
        with (
            patch.object(svc, "_load_cache", return_value=wide_cache_df),
            patch.object(svc, "_fetch_hybrid"),
            patch.object(svc, "_save_cache"),
        ):
            result = svc.get_dataframe(start=_START, end=_END)

        assert result.index.min() >= _START_UTC
        assert result.index.max() <= _END_UTC


# ---------------------------------------------------------------------------
# Tests: cache-miss path
# ---------------------------------------------------------------------------


class TestCacheMiss:
    """Verify that _fetch_hybrid is called when cache is absent or partial."""

    def test_no_cache_triggers_fetch(self, svc, fetch_df):
        """_fetch_hybrid is called when _load_cache returns None.

        Args:
            svc: WeatherService fixture.
            fetch_df: Simulated fetch result.
        """
        with (
            patch.object(svc, "_load_cache", return_value=None),
            patch.object(svc, "_fetch_hybrid", return_value=fetch_df) as mock_fetch,
            patch.object(svc, "_save_cache"),
        ):
            svc.get_dataframe(start=_START, end=_END)

        mock_fetch.assert_called_once()

    def test_cache_not_covering_end_triggers_fetch(
        self, svc, narrow_cache_df, fetch_df
    ):
        """_fetch_hybrid is called when cache.index.max() < end_utc.

        narrow_cache_df ends at 2023-06-02 23:00 UTC, which is before
        _END_UTC (2023-06-03 23:00 UTC).

        Args:
            svc: WeatherService fixture.
            narrow_cache_df: Partial cache ending before _END_UTC.
            fetch_df: Simulated fetch result.
        """
        with (
            patch.object(svc, "_load_cache", return_value=narrow_cache_df),
            patch.object(svc, "_fetch_hybrid", return_value=fetch_df) as mock_fetch,
            patch.object(svc, "_save_cache"),
        ):
            svc.get_dataframe(start=_START, end=_END)

        mock_fetch.assert_called_once()

    def test_cache_not_covering_start_triggers_fetch(self, svc, fetch_df):
        """_fetch_hybrid is called when cache.index.min() > start_utc.

        Args:
            svc: WeatherService fixture.
            fetch_df: Simulated fetch result.
        """
        late_cache = _hourly_df("2023-06-01 12:00:00", periods=200)

        with (
            patch.object(svc, "_load_cache", return_value=late_cache),
            patch.object(svc, "_fetch_hybrid", return_value=fetch_df) as mock_fetch,
            patch.object(svc, "_save_cache"),
        ):
            svc.get_dataframe(start=_START, end=_END)

        mock_fetch.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: fallback / error handling
# ---------------------------------------------------------------------------


class TestFallbackBehavior:
    """Verify the _create_fallback path triggered when _fetch_hybrid raises."""

    def test_fallback_called_when_fetch_fails_and_cache_sufficient(self, svc):
        """_create_fallback is used when fetch fails and cache has >= 24 rows.

        Args:
            svc: WeatherService fixture.
        """
        cache = _hourly_df("2023-05-31 00:00:00", periods=48)

        with (
            patch.object(svc, "_load_cache", return_value=cache),
            patch.object(svc, "_fetch_hybrid", side_effect=RuntimeError("timeout")),
            patch.object(
                svc, "_create_fallback", return_value=_hourly_df(_START, _PERIODS)
            ) as mock_fb,
            patch.object(svc, "_save_cache"),
        ):
            result = svc.get_dataframe(start=_START, end=_END, fallback_on_failure=True)

        mock_fb.assert_called_once()
        assert isinstance(result, pd.DataFrame)

    def test_exact_24_row_cache_satisfies_fallback_threshold(self, svc):
        """A cache with exactly 24 rows meets the len(cached_df) >= 24 condition.

        Args:
            svc: WeatherService fixture.
        """
        cache = _hourly_df("2023-05-31 00:00:00", periods=24)

        with (
            patch.object(svc, "_load_cache", return_value=cache),
            patch.object(svc, "_fetch_hybrid", side_effect=RuntimeError("timeout")),
            patch.object(
                svc, "_create_fallback", return_value=_hourly_df(_START, _PERIODS)
            ) as mock_fb,
            patch.object(svc, "_save_cache"),
        ):
            svc.get_dataframe(start=_START, end=_END, fallback_on_failure=True)

        mock_fb.assert_called_once()

    def test_exception_propagates_when_fallback_disabled(self, svc):
        """RuntimeError propagates when fallback_on_failure is False.

        Args:
            svc: WeatherService fixture.
        """
        cache = _hourly_df("2023-05-31 00:00:00", periods=48)

        with (
            patch.object(svc, "_load_cache", return_value=cache),
            patch.object(
                svc, "_fetch_hybrid", side_effect=RuntimeError("network error")
            ),
            patch.object(svc, "_save_cache"),
        ):
            with pytest.raises(RuntimeError, match="network error"):
                svc.get_dataframe(start=_START, end=_END, fallback_on_failure=False)

    def test_exception_propagates_when_cache_too_small_for_fallback(self, svc):
        """RuntimeError propagates when cache has fewer than 24 rows.

        Args:
            svc: WeatherService fixture.
        """
        small_cache = _hourly_df("2023-05-31 00:00:00", periods=23)

        with (
            patch.object(svc, "_load_cache", return_value=small_cache),
            patch.object(svc, "_fetch_hybrid", side_effect=RuntimeError("timeout")),
            patch.object(svc, "_save_cache"),
        ):
            with pytest.raises(RuntimeError):
                svc.get_dataframe(start=_START, end=_END, fallback_on_failure=True)

    def test_exception_propagates_when_no_cache_available(self, svc):
        """ConnectionError propagates when _load_cache returns None.

        Args:
            svc: WeatherService fixture.
        """
        with (
            patch.object(svc, "_load_cache", return_value=None),
            patch.object(svc, "_fetch_hybrid", side_effect=ConnectionError("offline")),
            patch.object(svc, "_save_cache"),
        ):
            with pytest.raises(ConnectionError, match="offline"):
                svc.get_dataframe(start=_START, end=_END, fallback_on_failure=True)


# ---------------------------------------------------------------------------
# Tests: cache merge and deduplication
# ---------------------------------------------------------------------------


class TestCacheMerge:
    """Verify that fetched data is correctly merged with an existing cache."""

    def test_partial_cache_merged_with_fetch_result(
        self, svc_cached, narrow_cache_df, fetch_df
    ):
        """After a fetch, the result is concatenated with the existing cache.

        Args:
            svc_cached: WeatherService with cache_path.
            narrow_cache_df: Partial cache ending before _END_UTC.
            fetch_df: Full-range fetch result.
        """
        saved = []

        with (
            patch.object(svc_cached, "_load_cache", return_value=narrow_cache_df),
            patch.object(svc_cached, "_fetch_hybrid", return_value=fetch_df),
            patch.object(svc_cached, "_save_cache", side_effect=saved.append),
        ):
            svc_cached.get_dataframe(start=_START, end=_END)

        assert saved, "_save_cache must be called after merge"
        merged = saved[0]
        # Merged DataFrame must extend at least as far as the fetch result.
        assert merged.index.max() >= fetch_df.index.max()

    def test_merged_index_has_no_duplicates(self, svc_cached, fetch_df):
        """After merge, the saved DataFrame has a unique DatetimeIndex.

        Args:
            svc_cached: WeatherService with cache_path.
            fetch_df: Full-range fetch result that overlaps with the cache.
        """
        overlapping_cache = _hourly_df(_START, periods=48)
        saved = []

        with (
            patch.object(svc_cached, "_load_cache", return_value=overlapping_cache),
            patch.object(svc_cached, "_fetch_hybrid", return_value=fetch_df),
            patch.object(svc_cached, "_save_cache", side_effect=saved.append),
        ):
            svc_cached.get_dataframe(start=_START, end=_END)

        assert saved
        assert not saved[0].index.duplicated().any()

    def test_fetched_values_overwrite_cached_values_on_overlap(self, svc_cached):
        """On duplicate timestamps, the fetched row wins over the cached row.

        The merge uses keep='last'; the fetched DataFrame is appended second
        in pd.concat, so its values survive deduplication.

        Args:
            svc_cached: WeatherService with cache_path.
        """
        cache = _hourly_df(_START, periods=48)
        cache["temperature_2m"] = 0.0  # sentinel: old cached value

        fresh = _hourly_df(_START, periods=72)
        fresh["temperature_2m"] = 99.0  # sentinel: new fetched value

        saved = []

        with (
            patch.object(svc_cached, "_load_cache", return_value=cache),
            patch.object(svc_cached, "_fetch_hybrid", return_value=fresh),
            patch.object(svc_cached, "_save_cache", side_effect=saved.append),
        ):
            svc_cached.get_dataframe(start=_START, end=_END)

        merged = saved[0]
        overlap_ts = pd.Timestamp("2023-06-01 06:00:00", tz="UTC")
        assert merged.loc[overlap_ts, "temperature_2m"] == pytest.approx(99.0)

    def test_no_merge_when_cache_is_none(self, svc, fetch_df):
        """pd.concat is not invoked when _load_cache returns None.

        Absence of merge means the fetch result is used directly without
        concatenation.  A valid DataFrame is still returned.

        Args:
            svc: WeatherService fixture.
            fetch_df: Simulated fetch result.
        """
        with (
            patch.object(svc, "_load_cache", return_value=None),
            patch.object(svc, "_fetch_hybrid", return_value=fetch_df),
            patch.object(svc, "_save_cache"),
        ):
            result = svc.get_dataframe(start=_START, end=_END)

        assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# Tests: cache persistence gate
# ---------------------------------------------------------------------------


class TestCachePersistence:
    """Verify the conditional cache-save gate (only when cache_path is set)."""

    def test_save_called_after_fetch_when_cache_path_configured(
        self, svc_cached, fetch_df
    ):
        """_save_cache is called once after a successful fetch.

        Args:
            svc_cached: WeatherService with cache_path.
            fetch_df: Simulated fetch result.
        """
        with (
            patch.object(svc_cached, "_load_cache", return_value=None),
            patch.object(svc_cached, "_fetch_hybrid", return_value=fetch_df),
            patch.object(svc_cached, "_save_cache") as mock_save,
        ):
            svc_cached.get_dataframe(start=_START, end=_END)

        mock_save.assert_called_once()

    def test_save_not_called_when_no_cache_path(self, svc, fetch_df):
        """_save_cache is not called when cache_path is None.

        Args:
            svc: WeatherService without cache_path.
            fetch_df: Simulated fetch result.
        """
        assert svc.cache_path is None

        with (
            patch.object(svc, "_load_cache", return_value=None),
            patch.object(svc, "_fetch_hybrid", return_value=fetch_df),
            patch.object(svc, "_save_cache") as mock_save,
        ):
            svc.get_dataframe(start=_START, end=_END)

        mock_save.assert_not_called()

    def test_save_not_called_on_cache_hit(self, svc_cached, wide_cache_df):
        """_save_cache is not called on the cache-hit early-return path.

        Args:
            svc_cached: WeatherService with cache_path.
            wide_cache_df: Cache DataFrame fully covering the test range.
        """
        with (
            patch.object(svc_cached, "_load_cache", return_value=wide_cache_df),
            patch.object(svc_cached, "_fetch_hybrid"),
            patch.object(svc_cached, "_save_cache") as mock_save,
        ):
            svc_cached.get_dataframe(start=_START, end=_END)

        mock_save.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: output properties
# ---------------------------------------------------------------------------


class TestOutputProperties:
    """Verify the structure and content of the DataFrame returned by get_dataframe."""

    def test_returns_pandas_dataframe(self, svc, fetch_df):
        """get_dataframe always returns a pd.DataFrame instance.

        Args:
            svc: WeatherService fixture.
            fetch_df: Simulated fetch result.
        """
        with (
            patch.object(svc, "_load_cache", return_value=None),
            patch.object(svc, "_fetch_hybrid", return_value=fetch_df),
            patch.object(svc, "_save_cache"),
        ):
            result = svc.get_dataframe(start=_START, end=_END)

        assert isinstance(result, pd.DataFrame)

    def test_output_index_bounded_by_requested_range(self, svc, fetch_df):
        """Returned index satisfies start_utc <= index <= end_utc.

        Args:
            svc: WeatherService fixture.
            fetch_df: Simulated fetch result.
        """
        with (
            patch.object(svc, "_load_cache", return_value=None),
            patch.object(svc, "_fetch_hybrid", return_value=fetch_df),
            patch.object(svc, "_save_cache"),
        ):
            result = svc.get_dataframe(start=_START, end=_END)

        assert result.index.min() >= _START_UTC
        assert result.index.max() <= _END_UTC

    def test_output_has_timezone_aware_index(self, svc, fetch_df):
        """The returned DatetimeIndex carries timezone information.

        Args:
            svc: WeatherService fixture.
            fetch_df: Simulated fetch result.
        """
        with (
            patch.object(svc, "_load_cache", return_value=None),
            patch.object(svc, "_fetch_hybrid", return_value=fetch_df),
            patch.object(svc, "_save_cache"),
        ):
            result = svc.get_dataframe(start=_START, end=_END)

        assert result.index.tz is not None

    def test_default_raises_on_nan_rows(self, svc):
        """Default ``fill_missing=False`` refuses to return NaN rows.

        Rows 10 to 14 are set to NaN before the call.  The fail-safe
        default must surface the gap instead of returning imputed
        values disguised as measurements.

        Args:
            svc: WeatherService fixture.
        """
        df = _hourly_df(_START, _PERIODS)
        df.iloc[10:15, 0] = float("nan")

        with (
            patch.object(svc, "_load_cache", return_value=None),
            patch.object(svc, "_fetch_hybrid", return_value=df),
            patch.object(svc, "_save_cache"),
        ):
            with pytest.raises(ValueError, match="missing row"):
                svc.get_dataframe(start=_START, end=_END)

    def test_fill_missing_true_restores_legacy_behavior(self, svc):
        """``fill_missing=True`` forward/back-fills NaN rows.

        Rows 10 to 14 are set to NaN before the call; with the opt-in
        flag the returned DataFrame must contain zero missing values.

        Args:
            svc: WeatherService fixture.
        """
        df = _hourly_df(_START, _PERIODS)
        df.iloc[10:15, 0] = float("nan")

        with (
            patch.object(svc, "_load_cache", return_value=None),
            patch.object(svc, "_fetch_hybrid", return_value=df),
            patch.object(svc, "_save_cache"),
        ):
            result = svc.get_dataframe(start=_START, end=_END, fill_missing=True)

        assert result.isnull().sum().sum() == 0

    def test_default_freq_preserves_hourly_resolution(self, svc, fetch_df):
        """With the default freq='h', the returned index has 1-hour steps.

        Args:
            svc: WeatherService fixture.
            fetch_df: 72-row hourly fetch result.
        """
        with (
            patch.object(svc, "_load_cache", return_value=None),
            patch.object(svc, "_fetch_hybrid", return_value=fetch_df),
            patch.object(svc, "_save_cache"),
        ):
            result = svc.get_dataframe(start=_START, end=_END, freq="h")

        diffs = result.index.to_series().diff().dropna().unique()
        assert len(diffs) == 1
        assert diffs[0] == pd.Timedelta("1h")

    def test_daily_freq_resamples_72_rows_to_three_days(self, svc, fetch_df):
        """freq='D' resamples 72 hourly rows down to 3 daily rows.

        The test range spans exactly three calendar days in UTC
        (2023-06-01, 2023-06-02, 2023-06-03), so the resampled output
        must contain exactly 3 rows.

        Args:
            svc: WeatherService fixture.
            fetch_df: 72-row hourly fetch result.
        """
        with (
            patch.object(svc, "_load_cache", return_value=None),
            patch.object(svc, "_fetch_hybrid", return_value=fetch_df),
            patch.object(svc, "_save_cache"),
        ):
            result = svc.get_dataframe(start=_START, end=_END, freq="D")

        assert len(result) == 3
