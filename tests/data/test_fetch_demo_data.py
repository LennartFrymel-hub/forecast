# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import pandas as pd
from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home


def test_fetch_demo01_data():
    """Test that demo01.csv is loaded correctly using get_package_data_home()."""
    demo_path = get_package_data_home() / "demo01.csv"
    df = fetch_data(filename=demo_path)

    # Check if it's a DataFrame
    assert isinstance(df, pd.DataFrame)

    # Check for expected columns
    assert "Forecasted Load" in df.columns
    assert "Actual Load" in df.columns

    # Check index is datetime with timezone
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None

    # Check that it's not empty
    assert not df.empty

    print("\nVerified demo01.csv loading:")
    print(df.head())


def test_fetch_demo02():
    """Test that demo02.csv is loaded correctly using get_package_data_home()."""
    demo_path = get_package_data_home() / "demo02.csv"
    assert demo_path.exists(), f"Demo file not found at {demo_path}"

    df = fetch_data(filename=str(demo_path))

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert df.index.name == "DateTime"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None

    print("\nVerified demo02.csv loading:")
    print(df.head())
