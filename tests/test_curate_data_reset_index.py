# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for reset_index() in preprocessing.curate_data.

Covers:
- Default index_name ("DateTime")
- Custom index_name
- Resulting column name matches index_name
- Resulting index is a RangeIndex
- Data values are preserved
- Column order (index column first, then data columns)
- Multi-column DataFrames
- UTC timezone-aware DatetimeIndex
- Empty DataFrame
- Single-row DataFrame
- Pre-named index (same name as requested)
- Input DataFrame index.name mutation does not affect caller's original name
"""

import pandas as pd

from spotforecast2_safe.preprocessing.curate_data import reset_index

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n: int = 5, tz: str | None = None) -> pd.DataFrame:
    """Return a simple DataFrame with a DatetimeIndex."""
    freq = "h"
    idx = pd.date_range("2023-01-01", periods=n, freq=freq, tz=tz)
    return pd.DataFrame({"value": range(n)}, index=idx)


# ---------------------------------------------------------------------------
# Default behaviour
# ---------------------------------------------------------------------------


class TestResetIndexDefault:
    """Verify reset_index() with the default index_name."""

    def test_returns_dataframe(self):
        result = reset_index(_make_df())
        assert isinstance(result, pd.DataFrame)

    def test_default_index_name_column_exists(self):
        result = reset_index(_make_df())
        assert "DateTime" in result.columns

    def test_default_index_is_range_index(self):
        result = reset_index(_make_df())
        assert isinstance(result.index, pd.RangeIndex)

    def test_default_range_index_starts_at_zero(self):
        result = reset_index(_make_df(n=3))
        assert list(result.index) == [0, 1, 2]

    def test_data_values_preserved(self):
        df = _make_df(n=5)
        result = reset_index(df)
        assert list(result["value"]) == list(range(5))

    def test_datetime_column_first(self):
        result = reset_index(_make_df())
        assert result.columns[0] == "DateTime"

    def test_row_count_unchanged(self):
        df = _make_df(n=7)
        result = reset_index(df)
        assert len(result) == 7


# ---------------------------------------------------------------------------
# Custom index_name
# ---------------------------------------------------------------------------


class TestResetIndexCustomName:
    """Verify reset_index() respects a user-supplied index_name."""

    def test_custom_name_column_exists(self):
        result = reset_index(_make_df(), index_name="Timestamp")
        assert "Timestamp" in result.columns

    def test_custom_name_column_is_first(self):
        result = reset_index(_make_df(), index_name="ts")
        assert result.columns[0] == "ts"

    def test_default_name_absent_when_custom_supplied(self):
        result = reset_index(_make_df(), index_name="ts")
        assert "DateTime" not in result.columns

    def test_custom_name_matches_datetime_values(self):
        df = _make_df(n=3)  # tz-naive; reset_index localizes to UTC
        expected = df.index.tz_localize("UTC").tolist()
        result = reset_index(df, index_name="MyTime")
        assert list(result["MyTime"]) == expected

    def test_custom_name_same_as_existing_index_name(self):
        df = _make_df(n=3)
        df.index.name = "DateTime"
        result = reset_index(df, index_name="DateTime")
        assert "DateTime" in result.columns
        assert isinstance(result.index, pd.RangeIndex)


# ---------------------------------------------------------------------------
# Multi-column DataFrame
# ---------------------------------------------------------------------------


class TestResetIndexMultiColumn:
    """Verify reset_index() handles DataFrames with multiple data columns."""

    def test_all_data_columns_preserved(self):
        idx = pd.date_range("2023-01-01", periods=4, freq="h")
        df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}, index=idx)
        result = reset_index(df)
        assert "a" in result.columns
        assert "b" in result.columns

    def test_column_count(self):
        idx = pd.date_range("2023-01-01", periods=4, freq="h")
        df = pd.DataFrame({"a": range(4), "b": range(4), "c": range(4)}, index=idx)
        result = reset_index(df)
        # 3 data columns + 1 index column
        assert len(result.columns) == 4

    def test_data_values_intact_multi_col(self):
        idx = pd.date_range("2023-01-01", periods=3, freq="h")
        df = pd.DataFrame({"x": [10, 20, 30]}, index=idx)
        result = reset_index(df)
        assert list(result["x"]) == [10, 20, 30]


# ---------------------------------------------------------------------------
# Timezone-aware index
# ---------------------------------------------------------------------------


class TestResetIndexTimezone:
    """Verify reset_index() works with timezone-aware DatetimeIndex."""

    def test_utc_index_preserved_in_column(self):
        df = _make_df(n=3, tz="UTC")
        expected = df.index.tolist()
        result = reset_index(df)
        assert list(result["DateTime"]) == expected

    def test_timezone_aware_result_is_range_index(self):
        df = _make_df(n=3, tz="UTC")
        result = reset_index(df)
        assert isinstance(result.index, pd.RangeIndex)

    def test_non_utc_timezone(self):
        df = _make_df(n=3, tz="Europe/Berlin")
        result = reset_index(df, index_name="ts")
        assert "ts" in result.columns
        assert isinstance(result.index, pd.RangeIndex)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestResetIndexEdgeCases:
    """Verify reset_index() handles boundary conditions."""

    def test_empty_dataframe(self):
        idx = pd.DatetimeIndex([])
        df = pd.DataFrame({"value": []}, index=idx)
        result = reset_index(df)
        assert "DateTime" in result.columns
        assert len(result) == 0

    def test_single_row_dataframe(self):
        df = _make_df(n=1)
        result = reset_index(df)
        assert len(result) == 1
        assert isinstance(result.index, pd.RangeIndex)

    def test_unnamed_index(self):
        df = _make_df(n=3)
        df.index.name = None  # explicitly unnamed
        result = reset_index(df)
        assert "DateTime" in result.columns

    def test_pre_named_index_overwritten(self):
        df = _make_df(n=3)
        df.index.name = "OldName"
        result = reset_index(df, index_name="NewName")
        assert "NewName" in result.columns
        assert "OldName" not in result.columns

    def test_range_index_input_does_not_raise(self):
        """Regression: RangeIndex has no tzinfo — must not raise AttributeError."""
        df = pd.DataFrame({"value": range(3)})  # default RangeIndex
        result = reset_index(df)
        assert isinstance(result.index, pd.RangeIndex)

    def test_integer_index_input_does_not_raise(self):
        """Non-DatetimeIndex with no tzinfo must not raise AttributeError."""
        df = pd.DataFrame({"value": range(3)}, index=[10, 20, 30])
        result = reset_index(df, index_name="idx")
        assert "idx" in result.columns


# ---------------------------------------------------------------------------
# Timezone localization of naive DatetimeIndex
# ---------------------------------------------------------------------------


class TestResetIndexNaiveLocalization:
    """Naive DatetimeIndex must be localized before reset; aware index unchanged."""

    def test_naive_index_localized_to_utc_by_default(self):
        df = _make_df(n=3)  # tz-naive
        result = reset_index(df)
        col = result["DateTime"]
        assert col.dt.tz is not None
        assert str(col.dt.tz) == "UTC"

    def test_naive_index_localized_to_custom_timezone(self):
        df = _make_df(n=3)  # tz-naive
        result = reset_index(df, timezone="Europe/Berlin")
        col = result["DateTime"]
        assert col.dt.tz is not None
        assert "Berlin" in str(col.dt.tz)

    def test_aware_index_not_re_localized(self):
        """Already tz-aware index must pass through unchanged."""
        df = _make_df(n=3, tz="UTC")
        result = reset_index(df)
        col = result["DateTime"]
        assert str(col.dt.tz) == "UTC"

    def test_aware_non_utc_index_not_re_localized(self):
        df = _make_df(n=3, tz="Europe/Berlin")
        result = reset_index(df)
        col = result["DateTime"]
        assert "Berlin" in str(col.dt.tz)

    def test_result_is_always_range_index(self):
        for tz in (None, "UTC", "Europe/Berlin"):
            df = _make_df(n=3, tz=tz)
            result = reset_index(df)
            assert isinstance(result.index, pd.RangeIndex)
