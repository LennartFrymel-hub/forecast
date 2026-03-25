# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for load_timeseries_forecast in spotforecast2_safe.data.fetch_data.

The test suite is organised around five concerns:

    1. Happy-path contracts — return type, index properties, and value
       correctness on well-formed input.
    2. Missing-value handling — NaN rows are filled by ffill/bfill.
    3. Error handling — FileNotFoundError when the CSV is absent;
       KeyError (raised by pandas) when the column is missing.
    4. data_home resolution — explicit argument, environment variable,
       and the default (~/ path) all resolve consistently.
    5. Symmetry with load_timeseries — both functions share CSV parsing
       logic and must produce compatible results from the same file.
"""

import os
import shutil
import pytest
import pandas as pd
from pathlib import Path

from spotforecast2_safe.data.fetch_data import (
    load_timeseries_forecast,
    load_timeseries,
    get_package_data_home,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_ROWS = 24  # rows written by the shared fixture


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_energy_csv(interim_dir: Path, n_rows: int = _N_ROWS) -> Path:
    """Write a minimal ``energy_load.csv`` into interim_dir.

    Args:
        interim_dir: Directory where the file is written.
        n_rows: Number of hourly rows to include.

    Returns:
        Path to the written CSV file.
    """
    idx = pd.date_range("2023-01-01 00:00", periods=n_rows, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "Time (UTC)": idx.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "Forecasted Load": [float(i) * 1.1 for i in range(n_rows)],
            "Actual Load": [float(i) * 1.0 for i in range(n_rows)],
        }
    )
    csv_path = interim_dir / "energy_load.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


def _write_energy_csv_with_nans(interim_dir: Path, nan_rows: list) -> Path:
    """Write an energy_load.csv where selected Forecasted Load rows are NaN.

    Args:
        interim_dir: Directory where the file is written.
        nan_rows: Row indices (0-based) to set as NaN in Forecasted Load.

    Returns:
        Path to the written CSV file.
    """
    idx = pd.date_range("2023-01-01 00:00", periods=_N_ROWS, freq="h", tz="UTC")
    forecasted = [float(i) * 1.1 for i in range(_N_ROWS)]
    for r in nan_rows:
        forecasted[r] = float("nan")
    df = pd.DataFrame(
        {
            "Time (UTC)": idx.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "Forecasted Load": forecasted,
            "Actual Load": [float(i) for i in range(_N_ROWS)],
        }
    )
    csv_path = interim_dir / "energy_load.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def data_dir(tmp_path) -> Path:
    """Temporary data directory with a valid energy_load.csv.

    Args:
        tmp_path: pytest built-in temporary directory fixture.

    Returns:
        Root data directory (the function receives this as data_home).
    """
    interim = tmp_path / "interim"
    interim.mkdir()
    _write_energy_csv(interim)
    return tmp_path


@pytest.fixture()
def data_dir_with_nans(tmp_path) -> Path:
    """Temporary data directory with NaN rows in Forecasted Load.

    Rows 5, 6, and 7 are NaN.

    Args:
        tmp_path: pytest built-in temporary directory fixture.

    Returns:
        Root data directory.
    """
    interim = tmp_path / "interim"
    interim.mkdir()
    _write_energy_csv_with_nans(interim, nan_rows=[5, 6, 7])
    return tmp_path


@pytest.fixture()
def data_dir_no_forecast_col(tmp_path) -> Path:
    """Temporary data directory with a CSV lacking the Forecasted Load column.

    Args:
        tmp_path: pytest built-in temporary directory fixture.

    Returns:
        Root data directory.
    """
    interim = tmp_path / "interim"
    interim.mkdir()
    idx = pd.date_range("2023-01-01", periods=_N_ROWS, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "Time (UTC)": idx.strftime("%Y-%m-%d %H:%M:%S+00:00"),
            "Actual Load": list(range(_N_ROWS)),
        }
    )
    (interim / "energy_load.csv").write_text(df.to_csv(index=False))
    return tmp_path


# ---------------------------------------------------------------------------
# Tests: happy-path contracts
# ---------------------------------------------------------------------------


class TestReturnType:
    """Verify basic structural contracts of the returned object."""

    def test_returns_series(self, data_dir):
        """load_timeseries_forecast returns a pd.Series instance.

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        result = load_timeseries_forecast(data_home=data_dir)
        assert isinstance(result, pd.Series)

    def test_series_name_is_forecasted_load(self, data_dir):
        """The returned Series name is 'Forecasted Load'.

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        result = load_timeseries_forecast(data_home=data_dir)
        assert result.name == "Forecasted Load"

    def test_length_matches_csv_row_count(self, data_dir):
        """Series length equals the number of rows written to the fixture CSV.

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        result = load_timeseries_forecast(data_home=data_dir)
        assert len(result) == _N_ROWS


# ---------------------------------------------------------------------------
# Tests: index properties
# ---------------------------------------------------------------------------


class TestIndexProperties:
    """Verify the DatetimeIndex of the returned Series."""

    def test_index_is_datetimeindex(self, data_dir):
        """The index is a pd.DatetimeIndex.

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        result = load_timeseries_forecast(data_home=data_dir)
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_index_is_utc(self, data_dir):
        """The index carries UTC timezone information.

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        result = load_timeseries_forecast(data_home=data_dir)
        assert result.index.tz is not None
        assert str(result.index.tz) == "UTC"

    def test_index_name_is_datetime(self, data_dir):
        """The index name is 'datetime' after renaming inside the function.

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        result = load_timeseries_forecast(data_home=data_dir)
        assert result.index.name == "datetime"

    def test_index_frequency_is_hourly(self, data_dir):
        """The index has hourly frequency after asfreq('h').

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        result = load_timeseries_forecast(data_home=data_dir)
        diffs = result.index.to_series().diff().dropna().unique()
        assert len(diffs) == 1
        assert diffs[0] == pd.Timedelta("1h")

    def test_index_starts_at_expected_timestamp(self, data_dir):
        """The first timestamp matches the start of the fixture date range.

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        result = load_timeseries_forecast(data_home=data_dir)
        assert result.index[0] == pd.Timestamp("2023-01-01 00:00", tz="UTC")


# ---------------------------------------------------------------------------
# Tests: value correctness
# ---------------------------------------------------------------------------


class TestValueCorrectness:
    """Verify that the correct numeric column is loaded."""

    def test_values_come_from_forecasted_load_column(self, data_dir):
        """Values are from the 'Forecasted Load' column, not 'Actual Load'.

        The fixture sets Forecasted Load = i * 1.1 and Actual Load = i * 1.0,
        so the first value must equal 0.0 and the second 1.1.

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        result = load_timeseries_forecast(data_home=data_dir)
        assert result.iloc[0] == pytest.approx(0.0)
        assert result.iloc[1] == pytest.approx(1.1)

    def test_values_differ_from_actual_load(self, data_dir):
        """Forecasted Load values differ from Actual Load values.

        This guards against accidentally reading the wrong column.

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        forecast = load_timeseries_forecast(data_home=data_dir)
        actual = load_timeseries(data_home=data_dir)
        assert not forecast.equals(actual)

    def test_dtype_is_float(self, data_dir):
        """Series dtype is a floating-point type.

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        result = load_timeseries_forecast(data_home=data_dir)
        assert pd.api.types.is_float_dtype(result)


# ---------------------------------------------------------------------------
# Tests: missing-value handling
# ---------------------------------------------------------------------------


class TestMissingValueHandling:
    """Verify that NaN rows are forward/backward filled."""

    def test_no_nans_in_clean_data(self, data_dir):
        """A clean CSV produces a Series with zero NaN values.

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        result = load_timeseries_forecast(data_home=data_dir)
        assert result.isna().sum() == 0

    def test_nans_are_filled_by_ffill_bfill(self, data_dir_with_nans):
        """NaN rows are eliminated by forward-fill followed by backward-fill.

        Args:
            data_dir_with_nans: Fixture with NaN rows in Forecasted Load.
        """
        result = load_timeseries_forecast(data_home=data_dir_with_nans)
        assert result.isna().sum() == 0

    def test_ffill_uses_preceding_value(self, tmp_path):
        """The filled value at the first NaN position equals the row before it.

        Row 5 is NaN; after ffill its value must equal row 4's value (4 * 1.1).

        Args:
            tmp_path: pytest built-in temporary directory fixture.
        """
        interim = tmp_path / "interim"
        interim.mkdir()
        _write_energy_csv_with_nans(interim, nan_rows=[5])
        result = load_timeseries_forecast(data_home=tmp_path)
        assert result.iloc[5] == pytest.approx(4 * 1.1)

    def test_leading_nan_filled_by_bfill(self, tmp_path):
        """A NaN at row 0 (no preceding value) is filled by backward-fill.

        Row 0 is NaN; after bfill its value must equal row 1's value (1 * 1.1).

        Args:
            tmp_path: pytest built-in temporary directory fixture.
        """
        interim = tmp_path / "interim"
        interim.mkdir()
        _write_energy_csv_with_nans(interim, nan_rows=[0])
        result = load_timeseries_forecast(data_home=tmp_path)
        assert result.iloc[0] == pytest.approx(1 * 1.1)


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Verify that the correct exceptions are raised on bad input."""

    def test_raises_file_not_found_when_csv_absent(self, tmp_path):
        """FileNotFoundError is raised when energy_load.csv does not exist.

        Args:
            tmp_path: pytest built-in temporary directory fixture.
        """
        (tmp_path / "interim").mkdir()
        with pytest.raises(FileNotFoundError, match="energy_load.csv"):
            load_timeseries_forecast(data_home=tmp_path)

    def test_raises_file_not_found_when_interim_directory_absent(self, tmp_path):
        """FileNotFoundError is raised when the interim directory is missing.

        Args:
            tmp_path: pytest built-in temporary directory fixture.
        """
        with pytest.raises(FileNotFoundError):
            load_timeseries_forecast(data_home=tmp_path)

    def test_raises_key_error_when_column_missing(self, data_dir_no_forecast_col):
        """KeyError is raised when 'Forecasted Load' column is absent.

        Args:
            data_dir_no_forecast_col: Fixture CSV without the column.
        """
        with pytest.raises(KeyError):
            load_timeseries_forecast(data_home=data_dir_no_forecast_col)

    def test_error_message_contains_csv_path(self, tmp_path):
        """FileNotFoundError message references the missing file path.

        Args:
            tmp_path: pytest built-in temporary directory fixture.
        """
        (tmp_path / "interim").mkdir()
        with pytest.raises(FileNotFoundError, match=str(tmp_path)):
            load_timeseries_forecast(data_home=tmp_path)


# ---------------------------------------------------------------------------
# Tests: data_home resolution
# ---------------------------------------------------------------------------


class TestDataHomeResolution:
    """Verify that data_home is resolved correctly from different sources."""

    def test_explicit_path_argument_is_used(self, data_dir):
        """Passing data_home explicitly uses that directory.

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        result = load_timeseries_forecast(data_home=data_dir)
        assert isinstance(result, pd.Series)

    def test_string_path_accepted(self, data_dir):
        """data_home accepts a str as well as a Path object.

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        result = load_timeseries_forecast(data_home=str(data_dir))
        assert isinstance(result, pd.Series)

    def test_environment_variable_controls_resolution(self, data_dir):
        """When data_home is None, the SPOTFORECAST2_DATA env var is used.

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        original = os.environ.get("SPOTFORECAST2_DATA")
        try:
            os.environ["SPOTFORECAST2_DATA"] = str(data_dir)
            result = load_timeseries_forecast(data_home=None)
            assert isinstance(result, pd.Series)
        finally:
            if original is None:
                os.environ.pop("SPOTFORECAST2_DATA", None)
            else:
                os.environ["SPOTFORECAST2_DATA"] = original

    def test_tilde_expansion_in_string_path(self, data_dir):
        """A path containing ~ is expanded to the user home directory.

        This test creates a subpath inside the user home, writes the fixture
        there, and confirms load_timeseries_forecast resolves it correctly.

        Args:
            data_dir: Unused; present for fixture isolation.
        """
        home = Path.home()
        target = home / "_sf2_test_tilde_expansion"
        interim = target / "interim"
        interim.mkdir(parents=True, exist_ok=True)
        try:
            _write_energy_csv(interim)
            result = load_timeseries_forecast(data_home="~/_sf2_test_tilde_expansion")
            assert isinstance(result, pd.Series)
        finally:
            shutil.rmtree(target, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests: symmetry with load_timeseries
# ---------------------------------------------------------------------------


class TestSymmetryWithLoadTimeseries:
    """load_timeseries and load_timeseries_forecast share parsing logic."""

    def test_both_return_series(self, data_dir):
        """Both functions return pd.Series from the same CSV.

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        forecast = load_timeseries_forecast(data_home=data_dir)
        actual = load_timeseries(data_home=data_dir)
        assert isinstance(forecast, pd.Series)
        assert isinstance(actual, pd.Series)

    def test_indexes_are_equal(self, data_dir):
        """Both functions produce identical DatetimeIndex values.

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        forecast = load_timeseries_forecast(data_home=data_dir)
        actual = load_timeseries(data_home=data_dir)
        pd.testing.assert_index_equal(forecast.index, actual.index)

    def test_same_length(self, data_dir):
        """Both Series have the same length for the same CSV.

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        forecast = load_timeseries_forecast(data_home=data_dir)
        actual = load_timeseries(data_home=data_dir)
        assert len(forecast) == len(actual)

    def test_forecast_and_actual_values_differ(self, data_dir):
        """Values differ because separate columns are read.

        The fixture sets Forecasted = i * 1.1 and Actual = i * 1.0.

        Args:
            data_dir: Fixture with a valid energy_load.csv.
        """
        forecast = load_timeseries_forecast(data_home=data_dir)
        actual = load_timeseries(data_home=data_dir)
        assert not (forecast.values == actual.values).all()
