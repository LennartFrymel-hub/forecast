# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.preprocessing.linearly_interpolate_ts import (
    LinearlyInterpolateTS,
)


def test_linear_interpolate_ts_series():
    """Trailing NaN bridged via ffill_bfill opt-in."""
    s = pd.Series([1.0, np.nan, 3.0, np.nan])
    interpolator = LinearlyInterpolateTS(on_missing="ffill_bfill")
    s_filled = interpolator.transform(s)

    expected = [1.0, 2.0, 3.0, 3.0]
    assert s_filled.tolist() == expected
    assert s_filled.dtype == "float64"


def test_linear_interpolate_ts_dataframe():
    """passthrough preserves residual leading NaN that linear interp cannot fill."""
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [np.nan, 10.0, 20.0]})
    interpolator = LinearlyInterpolateTS(on_missing="passthrough")
    df_filled = interpolator.transform(df)

    assert df_filled["a"].tolist() == [1.0, 2.0, 3.0]
    # passthrough leaves the residual leading NaN intact
    assert np.isnan(df_filled["b"].iloc[0])
    assert df_filled["b"].iloc[1] == 10.0
    assert df_filled["b"].iloc[2] == 20.0


def test_linear_interpolate_ts_docstring_example():
    """Mirror the canonical fill example used in the class docstring."""
    s = pd.Series([1.0, np.nan, 3.0, np.nan])
    interpolator = LinearlyInterpolateTS(on_missing="ffill_bfill")
    s_filled = interpolator.fit_transform(s)
    assert s_filled.tolist() == [1.0, 2.0, 3.0, 3.0]


@pytest.mark.parametrize("on_missing", ["raise", "ffill_bfill", "passthrough"])
def test_linear_interpolate_ts_no_nan(on_missing):
    """All three modes are no-ops on a NaN-free input."""
    s = pd.Series([1.0, 2.0, 3.0])
    interpolator = LinearlyInterpolateTS(on_missing=on_missing)
    s_filled = interpolator.transform(s)
    assert s_filled.tolist() == [1.0, 2.0, 3.0]


def test_raises_default_when_nan_remains():
    """Default fail-safe contract: residual NaN raises ValueError."""
    idx = pd.date_range("2026-01-01", periods=4, freq="h")
    s = pd.Series([1.0, np.nan, 3.0, np.nan], index=idx)
    interpolator = LinearlyInterpolateTS()  # default on_missing="raise"
    with pytest.raises(ValueError) as excinfo:
        interpolator.fit_transform(s)
    msg = str(excinfo.value)
    assert "missing value" in msg
    # The trailing-NaN timestamp should appear in the gap preview.
    assert str(idx[-1]) in msg
    # Hint must point the caller to the two opt-in alternatives.
    assert "ffill_bfill" in msg
    assert "passthrough" in msg


def test_raise_passes_when_interpolation_fills_all():
    """Interior gap fully bridged by linear interp does not raise."""
    s = pd.Series([1.0, np.nan, 3.0])
    out = LinearlyInterpolateTS().fit_transform(s)
    assert out.tolist() == [1.0, 2.0, 3.0]


def test_ffill_bfill_handles_leading_nan():
    """ffill_bfill back-fills a leading NaN that linear interp cannot bridge."""
    s = pd.Series([np.nan, 2.0, 3.0])
    out = LinearlyInterpolateTS(on_missing="ffill_bfill").fit_transform(s)
    assert out.tolist() == [2.0, 2.0, 3.0]


def test_passthrough_returns_residual_nan():
    """passthrough returns the linearly-interpolated series with NaN intact."""
    s = pd.Series([1.0, np.nan, 3.0, np.nan])
    out = LinearlyInterpolateTS(on_missing="passthrough").fit_transform(s)
    assert out.iloc[:3].tolist() == [1.0, 2.0, 3.0]
    assert np.isnan(out.iloc[-1])


def test_invalid_value_raises():
    """Unrecognized on_missing value is a ValueError."""
    s = pd.Series([1.0, 2.0, 3.0])
    interpolator = LinearlyInterpolateTS(on_missing="garbage")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="on_missing must be"):
        interpolator.fit_transform(s)


def test_raise_dataframe_lists_first_gap_row():
    """For DataFrames the gap preview uses rows that have any residual NaN."""
    idx = pd.date_range("2026-02-01", periods=3, freq="h")
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [np.nan, 10.0, 20.0]}, index=idx)
    with pytest.raises(ValueError) as excinfo:
        LinearlyInterpolateTS().fit_transform(df)
    assert str(idx[0]) in str(excinfo.value)
