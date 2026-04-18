# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from spotforecast2_safe.downloader.entsoe import download_new_data


@pytest.fixture
def mock_entsoe_data():
    """Load a subset of demo01.csv and format it like ENTSO-E response."""
    demo_path = (
        Path(__file__).parents[1] / "src/spotforecast2_safe/datasets/csv/demo01.csv"
    )
    df = pd.read_csv(demo_path)
    # The new demo01.csv already has "Time (UTC)"
    if "Time (UTC)" in df.columns:
        df["Time (UTC)"] = pd.to_datetime(df["Time (UTC)"], utc=True)
        df.set_index("Time (UTC)", inplace=True)
    elif "Time" in df.columns:
        df["Time"] = pd.to_datetime(df["Time"], utc=True)
        df.set_index("Time", inplace=True)
        df.index.name = "Time (UTC)"

    # Standard ENTSO-E load/forecast columns are usually 'Actual Load' and 'Day-ahead Forecast'
    # or similar. For our mock, we just need it to be a valid DataFrame.
    return df.head(100)


@patch("spotforecast2_safe.downloader.entsoe.merge_build_manual")
@patch("spotforecast2_safe.downloader.entsoe.get_data_home")
@patch("spotforecast2_safe.downloader.entsoe.fetch_data")
def test_download_new_data_example_logic(
    mock_fetch, mock_data_home, mock_merge, mock_entsoe_data, tmp_path
):
    """
    Validate the logic shown in the docstring examples by mocking the ENTSO-E client.
    Uses demo01.csv as a source for the mocked data.
    """
    # Setup tmp directories
    mock_data_home.return_value = tmp_path
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    # Mock the ENTSO-E library in sys.modules so patch works even if not installed
    mock_entsoe = MagicMock()
    with patch.dict("sys.modules", {"entsoe": mock_entsoe}):
        # Patch the class through the module reference
        with patch("entsoe.EntsoePandasClient") as MockClient:
            instance = MockClient.return_value
            # Use our demo data as the response
            instance.query_load_and_forecast.return_value = mock_entsoe_data

            # Example 1 Logic: Specific dates
            download_new_data(
                api_key="dummy_key",
                country_code="DE",
                start="202301010000",
                end="202301020000",
                force=True,
            )

            # Verify call parameters
            instance.query_load_and_forecast.assert_called_once()
            args, kwargs = instance.query_load_and_forecast.call_args
            assert kwargs["country_code"] == "DE"
            assert kwargs["start"] == pd.to_datetime("202301010000", utc=True)

            # Verify file creation
            csv_files = list(raw_dir.glob("*.csv"))
            assert len(csv_files) == 1
            saved_df = pd.read_csv(csv_files[0], index_col=0)
            assert len(saved_df) == 100
            assert "Actual Load" in saved_df.columns


def test_download_new_data_no_entsoe_py():
    """Verify that a helpful error is raised if entsoe-py is missing."""
    # Specifically ensure 'entsoe' import fails
    with patch.dict("sys.modules", {"entsoe": None}):
        with pytest.raises(ImportError, match="entsoe-py"):
            download_new_data(api_key="key")
