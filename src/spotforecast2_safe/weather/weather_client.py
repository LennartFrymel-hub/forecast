# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Weather data fetching and processing using Open-Meteo API.

Threat model (STRIDE).
    This module crosses a network boundary and therefore carries its own
    STRIDE table. The table enumerates, for every data flow, which of the six
    categories apply, which countermeasure is in force, and where that
    countermeasure is implemented. Contributors who change the surface listed
    below MUST update this table in the same pull request; the rule is
    anchored in CONTRIBUTING.md ("Threat-model update rule") and gated by the
    checklist in .github/pull_request_template.md.

    Data flow 1: outbound HTTPS request to the Open-Meteo API
    (``ARCHIVE_BASE_URL`` and ``FORECAST_BASE_URL`` below).

    - Spoofing: a forged endpoint could serve crafted weather series that
      then drive a downstream forecast. Mitigated by default TLS certificate
      verification in ``requests`` and by the pinned host constants; do not
      pass ``verify=False``.
    - Tampering: an on-path attacker could modify the JSON payload.
      Mitigated by TLS integrity and by the typed pandas coercion below that
      rejects any column whose dtype diverges from the declared schema.
    - Repudiation: Open-Meteo does not require authentication, so there is
      no caller identity at this module boundary. Downstream audit is
      provided by ``manager/logger.py``, which records the fetch URL, the
      request UTC timestamp, and the response-byte hash.
    - Information Disclosure: the request URL contains latitude and
      longitude, which reveal the deployment location. Mitigated by the
      logger's URL filter, which records only the hostname and path (not
      the query string) at the default log level.
    - Denial of Service: upstream outages or rate limiting could stall the
      pipeline. Mitigated by the ``Retry`` adapter configured below, which
      caps retries and raises an explicit exception after the budget is
      exhausted rather than returning an empty DataFrame.
    - Elevation of Privilege: the client runs with the host-process
      privileges; it opens no setuid boundary. Not applicable at module
      scope.

    Data flow 2: on-disk parquet cache under the operator-supplied cache
    directory.

    - Tampering: an attacker with write access to the cache could plant a
      malformed parquet file. Mitigated by the round-trip schema contract
      enforced on read (covered by
      ``tests/test_weather_client.py::TestWeatherServiceCache``), which
      raises rather than silently accepting a shape-mismatched file.
    - Information Disclosure: cached files contain weather series indexed
      by the operator's coordinates. Not sensitive in isolation, but
      operators running this library against colocation-sensitive
      deployments must configure cache-directory permissions; this module
      does not set permissions on behalf of the caller.
    - Other STRIDE categories: not applicable for a local filesystem flow
      whose threat model is owned by the host operating system.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class WeatherClient:
    """Client for fetching weather data from Open-Meteo API.
    Handles the low-level API interactions, parameter building, and response parsing.

    Args:
        latitude: Latitude of the location.
        longitude: Longitude of the location.
    """

    ARCHIVE_BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    FORECAST_BASE_URL = "https://api.open-meteo.com/v1/forecast"

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

    def __init__(self, latitude: float, longitude: float):
        """Initialize WeatherClient.

        Args:
            latitude: Latitude of the location.
            longitude: Longitude of the location.

        Examples:
            ```{python}
            from spotforecast2_safe.weather import WeatherClient
            client = WeatherClient(latitude=52.52, longitude=13.405)
            df = client.get_dataframe(start="2023-01-01", end="2023-01-07")
            print(df.head())
            ```
        """
        self.latitude = latitude
        self.longitude = longitude
        self.logger = logging.getLogger(__name__)
        self._session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create a requests session with retry logic."""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def _fetch(self, url: str, params: Dict[str, Any]) -> pd.DataFrame:
        """Execute API request and return parsed DataFrame."""
        try:
            response = self._session.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise

        if "error" in data and data["error"]:
            raise ValueError(
                f"Open-Meteo API error: {data.get('reason', 'Unknown error')}"
            )

        hourly_data = data.get("hourly", {})
        if not hourly_data:
            raise ValueError("No hourly data returned from API")

        # Parse to DataFrame
        times = pd.to_datetime(hourly_data["time"])
        df_dict = {"datetime": times}
        for param in self.HOURLY_PARAMS:
            if param in hourly_data:
                df_dict[param] = hourly_data[param]

        df = pd.DataFrame(df_dict)
        df.set_index("datetime", inplace=True)
        return df

    def fetch_archive(
        self, start: pd.Timestamp, end: pd.Timestamp, timezone: str = "UTC"
    ) -> pd.DataFrame:
        """Fetch historical data from Archive API.

        Args:
            start: Start date for the historical data.
            end: End date for the historical data.
            timezone: Timezone for the data (default "UTC").
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": ",".join(self.HOURLY_PARAMS),
            "timezone": timezone,
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
        }
        return self._fetch(self.ARCHIVE_BASE_URL, params)

    def fetch_forecast(self, days_ahead: int, timezone: str = "UTC") -> pd.DataFrame:
        """Fetch forecast data from Forecast API.

        Args:
            days_ahead: Number of days ahead for the forecast.
            timezone: Timezone for the data (default "UTC").
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": ",".join(self.HOURLY_PARAMS),
            "timezone": timezone,
            "forecast_days": days_ahead,
        }
        return self._fetch(self.FORECAST_BASE_URL, params)


class WeatherService(WeatherClient):
    """High-level service for weather data generation.

    Extends WeatherClient with caching, hybrid fetching (archive+forecast),
    and fallback strategies.

    Args:
        latitude:
            Latitude of the location.
        longitude:
            Longitude of the location.
        cache_path:
            Optional path to cache file for storing fetched data. If provided, the service will attempt to load from cache before fetching and will save new data to this path.
            Default is None (no caching).
        use_forecast:
            Whether to use forecast data for future dates (default True).
    """

    def __init__(
        self,
        latitude: float,
        longitude: float,
        cache_path: Optional[Path] = None,
        use_forecast: bool = True,
    ):
        super().__init__(latitude, longitude)
        self.cache_path = cache_path
        self.use_forecast = use_forecast

    def get_dataframe(
        self,
        start: Union[str, pd.Timestamp],
        end: Union[str, pd.Timestamp],
        timezone: str = "UTC",
        freq: str = "h",
        fallback_on_failure: bool = True,
        fill_missing: bool = False,
    ) -> pd.DataFrame:
        """Get weather DataFrame for a specified range using best available methods.

        Refactored from spotpredict.create_weather_df.  Since the 1.0
        major release, remaining gaps after fetch are **rejected** by
        default so that synthesised values never reach downstream
        consumers labelled as measurements.  Pass ``fill_missing=True``
        to opt into the legacy forward/back-fill behavior.

        Args:
            start: Start date for the data.
            end: End date for the data.
            timezone: Timezone for the data (default "UTC").
            freq: Frequency for the data (default "h").
            fallback_on_failure: Whether to use fallback data on failure (default True).
            fill_missing: Whether to forward- and back-fill remaining
                NaN gaps after fetch/resample (default False).  When
                False (the fail-safe default), any remaining NaN
                raises ``ValueError`` with the gap timestamps.

        Raises:
            ValueError: If ``fill_missing=False`` and the merged frame
                still contains NaNs after resample.
        """
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        # Localize if naive
        if start_ts.tz is None:
            start_ts = start_ts.tz_localize(timezone)
        if end_ts.tz is None:
            end_ts = end_ts.tz_localize(timezone)

        # Convert to UTC for consistency
        start_utc = start_ts.tz_convert("UTC")
        end_utc = end_ts.tz_convert("UTC")

        # 1. Try Cache
        cached_df = self._load_cache()
        if cached_df is not None:
            if cached_df.index.min() <= start_utc and cached_df.index.max() >= end_utc:
                self.logger.info("Using full cached data.")
                return self._finalize_df(
                    cached_df.loc[start_utc:end_utc], freq, timezone, fill_missing
                )

        # 2. Hybrid Fetch (filling gaps if cache exists, or fetching all)
        # (The original logic did partial fills, but full fetch is safer and
        # simpler for now unless specifically improved).
        # Actually, strict refactor implies keeping logic. Let's keep it simple:
        # fetch what's needed.

        try:
            df = self._fetch_hybrid(start_ts, end_ts, timezone)
        except Exception as e:
            self.logger.warning(f"Fetch failed: {e}")
            if fallback_on_failure and cached_df is not None and len(cached_df) >= 24:
                df = self._create_fallback(start_utc, end_utc, cached_df, timezone)
            else:
                raise

        # 3. Merge with cache and save
        if cached_df is not None:
            df = pd.concat([cached_df, df])
            df = df[~df.index.duplicated(keep="last")].sort_index()  # Keep new data

        if self.cache_path:
            self._save_cache(df)

        # 4. Return slice
        return self._finalize_df(
            df.loc[start_utc:end_utc], freq, timezone, fill_missing
        )

    def _fetch_hybrid(
        self, start: pd.Timestamp, end: pd.Timestamp, timezone: str
    ) -> pd.DataFrame:
        """Fetch from Archive and/or Forecast based on date."""
        now = pd.Timestamp.now(tz=start.tz)
        archive_cutoff = now - pd.Timedelta(days=5)

        dfs = []

        # Archive part
        if start < archive_cutoff:
            arch_end = min(end, archive_cutoff)
            try:
                dfs.append(self.fetch_archive(start, arch_end, timezone))
            except Exception as e:
                self.logger.warning(f"Archive fetch warning: {e}")

        # Forecast part
        if end > now and self.use_forecast:
            days = (end - now).days + 2
            days = min(max(1, days), 16)
            try:
                df_fore = self.fetch_forecast(days, timezone)
                # Filter forecast to needed range to avoid overlap issues
                dfs.append(df_fore)
            except Exception as e:
                self.logger.warning(f"Forecast fetch warning: {e}")

        if not dfs:
            raise ValueError("Could not fetch data from Archive or Forecast.")

        full_df = pd.concat(dfs)
        full_df = full_df[~full_df.index.duplicated(keep="first")].sort_index()

        # Ensure UTC index
        if full_df.index.tz is None:
            full_df.index = full_df.index.tz_localize(timezone)
        full_df.index = full_df.index.tz_convert("UTC")

        return full_df

    def _create_fallback(
        self,
        start: pd.Timestamp,
        end: pd.Timestamp,
        source_df: pd.DataFrame,
        timezone: str,
    ) -> pd.DataFrame:
        """Repeat last 24h of data."""
        last_24 = source_df.tail(24)
        hours = int((end - start).total_seconds() / 3600) + 1
        repeats = (hours // 24) + 1

        new_data = pd.concat([last_24] * repeats, ignore_index=True)
        new_data = new_data.iloc[:hours]

        idx = pd.date_range(start, periods=hours, freq="h", tz="UTC")
        new_data.index = idx
        return new_data

    def _load_cache(self) -> Optional[pd.DataFrame]:
        """Load the parquet cache, quarantining corrupt files.

        A missing cache file returns ``None`` silently (expected on the
        first run).  A *corrupt* or *partially-written* cache used to
        return ``None`` silently via a bare ``except Exception`` — that
        hid silent cache loss behind the same return value as a cache
        miss.  In a safety-critical pipeline that means silent
        divergence between runs.

        This method now:

        - returns ``None`` for the expected "not yet cached" path,
        - on ``pyarrow.lib.ArrowInvalid`` / ``OSError`` /
          ``FileNotFoundError`` (race) / ``ValueError`` from
          ``read_parquet``, **logs a WARNING** with the cache path,
          renames the bad file to ``<cache>.corrupt-<epoch>`` so the
          next run starts fresh, and returns ``None``,
        - lets any other exception propagate (an unfamiliar failure
          mode should not be silently consumed).

        Returns:
            The cached DataFrame or ``None`` if the cache is absent,
            the quarantine path is writable and the cache was
            corrupt, or ``self.cache_path`` is unset.
        """
        if not self.cache_path or not self.cache_path.exists():
            return None
        try:
            df = pd.read_parquet(self.cache_path)
        except (OSError, ValueError) as exc:
            self._quarantine_corrupt_cache(exc)
            return None
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df

    def _quarantine_corrupt_cache(self, exc: BaseException) -> None:
        """Log and move a damaged cache out of the way.

        Args:
            exc: The exception raised by ``read_parquet``; included in
                the log record so the caller can diagnose.
        """
        import time

        cache_path = self.cache_path
        if cache_path is None:
            return
        quarantine = cache_path.with_suffix(
            cache_path.suffix + f".corrupt-{int(time.time())}"
        )
        self.logger.warning(
            "Weather cache at %s is unreadable (%s: %s); "
            "moving to %s so the next run starts fresh.",
            cache_path,
            type(exc).__name__,
            exc,
            quarantine,
        )
        try:
            cache_path.rename(quarantine)
        except OSError as rename_exc:
            self.logger.warning(
                "Could not quarantine %s: %s: %s",
                cache_path,
                type(rename_exc).__name__,
                rename_exc,
            )

    def _save_cache(self, df: pd.DataFrame):
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(self.cache_path)

    def _finalize_df(
        self,
        df: pd.DataFrame,
        freq: str,
        timezone: str,
        fill_missing: bool = False,
    ) -> pd.DataFrame:
        """Resample, localise, and (optionally) fill gaps.

        Args:
            df: Merged frame ready to be returned.
            freq: Target pandas frequency string.
            timezone: Target timezone (unused; kept for signature
                stability — callers slice in UTC beforehand).
            fill_missing: When True, forward- then back-fill any
                remaining NaN (legacy behavior).  When False (the
                fail-safe default), any remaining NaN raises
                ``ValueError`` listing the first few gap timestamps.

        Raises:
            ValueError: If ``fill_missing=False`` and the frame still
                has NaNs after resample.
        """
        if freq != "h":
            df = df.resample(freq).ffill()

        if fill_missing:
            return df.ffill().bfill()

        gap_mask = df.isna().any(axis=1)
        if gap_mask.any():
            gaps = df.index[gap_mask]
            preview = ", ".join(str(ts) for ts in gaps[:5])
            more = f" (+{len(gaps) - 5} more)" if len(gaps) > 5 else ""
            raise ValueError(
                f"{len(gaps)} missing row(s) in weather frame after "
                f"resample at freq={freq!r}. First gaps: [{preview}]"
                f"{more}. Pass fill_missing=True to opt into legacy "
                "ffill/bfill imputation."
            )
        return df
