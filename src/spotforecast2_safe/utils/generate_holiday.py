# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Utilities for generating holiday dataframe as covariate."""

from typing import Union

import holidays
import pandas as pd


def create_holiday_df(
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    tz: str = "UTC",
    freq: str = "h",
    country_code: str = "DE",
    state: str = "NW",
) -> pd.DataFrame:
    """Create a DataFrame with datetime index and a binary holiday indicator column.

    Expands daily holidays to all timestamps in the desired frequency.

    Args:
        start: Start date/datetime.
        end: End date/datetime.
        tz: Timezone to use if not inferred from start/end.
        freq: Frequency of the resulting DataFrame.
        country_code: Country code for holidays (e.g. "DE", "US").
        state: State code for holidays (e.g. "NW", "CA").

    Returns:
        pd.DataFrame: DataFrame with index covering [start, end] at `freq`,
                      and a 'holiday' column (1 if holiday, 0 otherwise).

    Examples:
        >>> df = create_holiday_df("2023-12-24", "2023-12-26", freq="D")
        >>> df["holiday"].tolist()
        [0, 1, 1]
    """
    # If start/end are Timestamps with timezones, use that timezone instead of
    # the default. This avoids conflicts when timezone-aware Timestamps are
    # passed with a different tz parameter
    inferred_tz = None
    if isinstance(start, pd.Timestamp) and start.tz is not None:
        inferred_tz = str(start.tz)
    elif isinstance(end, pd.Timestamp) and end.tz is not None:
        inferred_tz = str(end.tz)

    # Use inferred timezone if available, otherwise use the provided tz parameter
    effective_tz = inferred_tz if inferred_tz is not None else tz

    # When creating date_range with timezone-aware Timestamps, don't pass tz parameter
    # to avoid conflicts - pandas will infer it from the Timestamps
    if inferred_tz is not None:
        full_index = pd.date_range(start=start, end=end, freq=freq)
    else:
        full_index = pd.date_range(start=start, end=end, freq=freq, tz=effective_tz)

    # Generate the range of unique days covered by start/end to check for holidays
    # We normalize to midnight to ensure we cover every day involved
    unique_dates = pd.Series(full_index.normalize().unique())

    # Get holidays for the country/state
    country_holidays = holidays.country_holidays(country_code, subdiv=state)

    # Check each day if it is a holiday
    is_holiday_series = pd.Series(
        [1 if d.date() in country_holidays else 0 for d in unique_dates],
        index=unique_dates,
    )

    # Broadcast holiday status from unique dates to the full index
    # We use mapping to avoid reindex issues with different timezone objects
    df_full = pd.DataFrame(index=full_index)
    df_full["is_holiday"] = (
        full_index.normalize().map(is_holiday_series).fillna(0).astype(int)
    )

    return df_full
