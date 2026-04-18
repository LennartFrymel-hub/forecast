# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
ENTSO-E Transparency Platform data downloader.

This module provides utilities to download electricity load and forecast data
directly from the ENTSO-E API and merge local raw data files into a consistent
dataset for forecasting.
"""

import logging
import time
from typing import Optional

import numpy as np
import pandas as pd

from spotforecast2_safe.data.fetch_data import fetch_data, get_data_home

logger = logging.getLogger(__name__)


def merge_build_manual(output_file: str = "energy_load.csv") -> None:
    """
    Merge all raw CSV files from the 'raw' directory into a single interim file.

    This function looks for all `.csv` files in `get_data_home() / "raw"`,
    sorts them by time index, and saves the unique combined data to
    `get_data_home() / "interim" / output_file`.

    Args:
        output_file: The name of the combined output file.
            Defaults to "energy_load.csv".

    Raises:
        FileNotFoundError: If the raw directory does not exist.
        ValueError: If no valid CSV files are found for merging.

    Notes:
        Logging information can be selected by setting the log level for the
        `spotforecast2_safe.downloader.entsoe` logger. Common levels are
        `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`.

        Examples:
            # Show only warnings and errors (default)
            >>> import logging
            >>> logging.getLogger("spotforecast2_safe.downloader.entsoe").setLevel(logging.WARNING)

            # Show informative messages about the merging process
            >>> logging.getLogger("spotforecast2_safe.downloader.entsoe").setLevel(logging.INFO)

            # Show detailed debug information
            >>> logging.getLogger("spotforecast2_safe.downloader.entsoe").setLevel(logging.DEBUG)

    Examples:
        # Example 1: Merge with default output file (if raw data exists)
        >>> from spotforecast2_safe.downloader.entsoe import merge_build_manual
        >>> try:
        ...     merge_build_manual()
        ... except Exception:
        ...     pass  # Ignore errors if no raw data exists

        # Example 2: Merge with a custom output file name
        >>> try:
        ...     merge_build_manual(output_file="custom_energy_load.csv")
        ... except Exception:
        ...     pass
    """
    data_home = get_data_home()
    raw_dir = data_home / "raw"
    interim_dir = data_home / "interim"

    if not raw_dir.exists():
        logger.warning(
            "Raw data directory %s does not exist. Nothing to merge.", raw_dir
        )
        return

    logger.info("Merging raw files from %s...", raw_dir)

    list_dfs = []
    for csv_file in raw_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            # Assuming 'Time (UTC)' is the index name as per spotprivate config
            # We'll try common index names or the first column
            index_col = "Time (UTC)" if "Time (UTC)" in df.columns else 0
            df.rename(
                columns={
                    (
                        df.columns[0] if isinstance(index_col, int) else index_col
                    ): "Time (UTC)"
                },
                inplace=True,
            )

            df["Time (UTC)"] = pd.to_datetime(df["Time (UTC)"], utc=True)
            df.set_index("Time (UTC)", inplace=True)

            # Clean data: handle placeholders like '-'
            for col in df.columns:
                df[col] = df[col].apply(lambda x: np.nan if x == "-" else x)

            list_dfs.append(df)
        except Exception as e:
            logger.error("Failed to process raw file %s: %s", csv_file, e)

    if not list_dfs:
        logger.info("No valid raw data files found for merging.")
        return

    merged_df = pd.concat(list_dfs)
    merged_df = merged_df[~merged_df.index.duplicated(keep="last")].sort_index()

    # Filter out future data points if any (only keep what's theoretically "actual" up to now)
    merged_df = merged_df.loc[merged_df.index <= pd.Timestamp.now(tz="UTC")]

    interim_dir.mkdir(parents=True, exist_ok=True)
    output_path = interim_dir / output_file
    merged_df.to_csv(output_path)
    logger.info("Successfully merged data saved to %s", output_path)


def download_new_data(
    api_key: str,
    country_code: str = "FR",
    start: Optional[str] = None,
    end: Optional[str] = None,
    force: bool = False,
) -> None:
    """
    Download new load and forecast data from ENTSO-E.

    This function queries the ENTSO-E Transparency Platform for a given period.
    If no start date is provided, it automatically resumes from the last
    available data point.

    Args:
        api_key: The ENTSO-E API key.
        country_code: The country code to query (e.g., 'FR', 'DE').
            Defaults to "FR".
        start: Start date in 'YYYYMMDDHH00' format.
        end: End date in 'YYYYMMDDHH00' format.
        force: If True, bypass the 24h cooldown check.

    Raises:
        ImportError:
            If the Python package 'entsoe-py' is not installed.
        ValueError:
            If data fetching fails after retries.

    Notes:
        Logging information can be selected by setting the log level for the
        `spotforecast2_safe.downloader.entsoe` logger. Common levels are
        `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`. For example, to
        show only warnings and errors (default):
        import logging;
        logging.getLogger("spotforecast2_safe.downloader.entsoe").setLevel(logging.WARNING)
        to show informative messages about the merging process
        import logging;
        logging.getLogger("spotforecast2_safe.downloader.entsoe").setLevel(logging.INFO)
        to show detailed debug information:
        import logging;
        logging.getLogger("spotforecast2_safe.downloader.entsoe").setLevel(logging.DEBUG)

    Examples:
        # Example 1: Basic download for Germany with specific start/end dates
        >>> from spotforecast2_safe.downloader.entsoe import download_new_data
        >>> try:
        ...     download_new_data(
        ...         api_key="YOUR_API_KEY",
        ...         country_code="DE",
        ...         start="202301010000",
        ...         end="202301020000",
        ...         force=True
        ...     )
        ... except (ImportError, ValueError, Exception):
        ...     # In a real scenario, handle errors appropriately
        ...     pass

        # Example 2: Incremental download (automatically resumes from last data point)
        >>> try:
        ...     download_new_data(api_key="YOUR_API_KEY", country_code="FR")
        ... except (ImportError, Exception):
        ...     pass

        # Example 3: Forced download bypassing the 24h cooldown check
        >>> try:
        ...     download_new_data(
        ...         api_key="YOUR_API_KEY",
        ...         country_code="DE",
        ...         force=True
        ...     )
        ... except (ImportError, Exception):
        ...     pass
    """

    try:
        from entsoe import EntsoePandasClient
    except ImportError as e:
        raise ImportError(
            "The 'entsoe-py' library is required for this functionality. "
            "Install it with: pip install entsoe-py"
        ) from e

    # First merge existing files to get the latest index
    merge_build_manual()

    logger.info("Initiating data download from ENTSO-E...")

    # Determine start date
    if start is None:
        try:
            current_data = fetch_data()  # This might look at interim or a specific file
            start_date = current_data.index[-1] + pd.Timedelta(hours=1)
            logger.info(
                "No start date provided. Resuming from last data point: %s", start_date
            )
        except (FileNotFoundError, ValueError, IndexError) as exc:
            # Narrow fallback: only the three signals that mean "no prior
            # data on disk yet". Anything else (ImportError, OSError
            # beyond FileNotFoundError, KeyError from a schema change,
            # ...) is a real bug and must propagate so the caller can see
            # it instead of silently starting from 7 days ago.
            start_date = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)
            logger.warning(
                "No previous data found (%s: %s). Starting from default date: %s",
                type(exc).__name__,
                exc,
                start_date,
            )
    else:
        start_date = pd.to_datetime(start, utc=True)
        logger.info("Using provided start date: %s", start_date)

    # Determine end date
    if end is None:
        end_date = pd.Timestamp.now(tz="UTC").floor("D")
        logger.info("No end date provided. Using current date: %s", end_date)
    else:
        end_date = pd.to_datetime(end, utc=True)
        logger.info("Using provided end date: %s", end_date)

    # Safety check: avoid redundant small downloads
    hours_diff = (end_date - start_date).total_seconds() / 3600
    if hours_diff < 24 and not force:
        logger.info(
            "Last download was too recent (%.1f hours ago). Skipping.", 24 - hours_diff
        )
        return

    client = EntsoePandasClient(api_key=api_key)

    # Retry loop
    retry_counter = 0
    success = False
    downloaded_df = None

    while retry_counter < 5:
        try:
            logger.info(
                "Downloading data from %s to %s (attempt %d/5)...",
                start_date,
                end_date,
                retry_counter + 1,
            )
            downloaded_df = client.query_load_and_forecast(
                country_code=country_code, start=start_date, end=end_date
            )
            success = True
            break
        except Exception as e:
            logger.warning("Download failed: %s. Retrying in 5s...", e)
            retry_counter += 1
            time.sleep(5)

    if not success or downloaded_df is None:
        logger.error("Failed to download data from ENTSO-E after 5 attempts.")
        return

    # Save to raw
    data_home = get_data_home()
    raw_dir = data_home / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    date_format = "%Y%m%d%H00"
    file_name = f"entsoe_load_{start_date.strftime(date_format)}_{end_date.strftime(date_format)}.csv"
    output_path = raw_dir / file_name

    downloaded_df.index.name = "Time (UTC)"
    downloaded_df.to_csv(output_path)
    logger.info("Downloaded data saved to %s", output_path)

    # Final merge to integrate new data
    merge_build_manual()
