# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

from unittest.mock import MagicMock, patch

import pandas as pd

from spotforecast2_safe.downloader.entsoe import download_new_data, merge_build_manual


def test_merge_build_manual_creates_merged_file(tmp_path):
    # Setup: create raw directory and CSVs
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    df1 = pd.DataFrame(
        {"Time (UTC)": ["2026-01-01 00:00", "2026-01-01 01:00"], "Actual Load": [1, 2]}
    )
    df2 = pd.DataFrame(
        {"Time (UTC)": ["2026-01-01 01:00", "2026-01-01 02:00"], "Actual Load": [2, 3]}
    )
    df1.to_csv(raw_dir / "a.csv", index=False)
    df2.to_csv(raw_dir / "b.csv", index=False)
    with patch(
        "spotforecast2_safe.downloader.entsoe.get_data_home", return_value=tmp_path
    ):
        merge_build_manual(output_file="merged.csv")
    merged = pd.read_csv(interim_dir / "merged.csv", index_col=0, parse_dates=True)
    assert len(merged) == 3
    assert "Actual Load" in merged.columns


def test_merge_build_manual_no_raw_dir(tmp_path, caplog):
    # No raw dir should log a warning and do nothing
    with patch(
        "spotforecast2_safe.downloader.entsoe.get_data_home", return_value=tmp_path
    ):
        merge_build_manual(output_file="should_not_exist.csv")
    assert not (tmp_path / "interim" / "should_not_exist.csv").exists()


def test_download_new_data_success(tmp_path):
    # Patch get_data_home, fetch_data, and sys.modules["entsoe"]
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    with (
        patch(
            "spotforecast2_safe.downloader.entsoe.get_data_home", return_value=tmp_path
        ),
        patch("spotforecast2_safe.downloader.entsoe.fetch_data") as mock_fetch,
    ):
        # Setup fetch_data to return a DataFrame with a recent index
        dates = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        mock_fetch.return_value = pd.DataFrame(index=dates)
        # Patch sys.modules to inject a mock entsoe module
        import sys

        mock_entsoe_mod = MagicMock()
        mock_client_class = mock_entsoe_mod.EntsoePandasClient
        mock_client = mock_client_class.return_value
        mock_df = pd.DataFrame(
            {"Actual Load": [123]}, index=[dates[-1] + pd.Timedelta(hours=1)]
        )
        mock_client.query_load_and_forecast.return_value = mock_df
        sys.modules["entsoe"] = mock_entsoe_mod
        download_new_data(api_key="fake_key", force=True)
        # Check that a file was created in raw
        files = list(raw_dir.glob("entsoe_load_*.csv"))
        assert len(files) == 1
        # Check merged file exists
        merged = pd.read_csv(interim_dir / "energy_load.csv")
        assert "Actual Load" in merged.columns


def test_download_new_data_cooldown_skips(tmp_path):
    # Patch get_data_home, fetch_data, and sys.modules["entsoe"]
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir()
    with (
        patch(
            "spotforecast2_safe.downloader.entsoe.get_data_home", return_value=tmp_path
        ),
        patch("spotforecast2_safe.downloader.entsoe.fetch_data") as mock_fetch,
    ):
        now = pd.Timestamp.now(tz="UTC")
        mock_fetch.return_value = pd.DataFrame(index=[now - pd.Timedelta(hours=2)])
        import sys

        mock_entsoe_mod = MagicMock()
        mock_client_class = mock_entsoe_mod.EntsoePandasClient
        sys.modules["entsoe"] = mock_entsoe_mod
        download_new_data(api_key="fake_key", force=False)
        # Should not call the client
        mock_client_class.assert_not_called()
