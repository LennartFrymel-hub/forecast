# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from spotforecast2_safe.downloader.entsoe import download_new_data, merge_build_manual

# Mock entsoe before importing our module
mock_entsoe = MagicMock()
sys.modules["entsoe"] = mock_entsoe


class TestEntsoeDownloader(unittest.TestCase):
    """Tests for the ENTSO-E downloader."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.raw_dir = self.test_dir / "raw"
        self.raw_dir.mkdir()
        self.interim_dir = self.test_dir / "interim"
        self.interim_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    def test_merge_build_manual(self, mock_get_home):
        """Test merging raw CSV files."""
        mock_get_home.return_value = self.test_dir

        # Create some dummy raw files
        df1 = pd.DataFrame(
            {
                "Time (UTC)": ["2026-01-01 00:00", "2026-01-01 01:00"],
                "Actual": [100, 110],
            }
        )
        df2 = pd.DataFrame(
            {
                "Time (UTC)": ["2026-01-01 01:00", "2026-01-01 02:00"],
                "Actual": [110, 120],
            }
        )

        df1.to_csv(self.raw_dir / "file1.csv", index=False)
        df2.to_csv(self.raw_dir / "file2.csv", index=False)

        merge_build_manual(output_file="test_merged.csv")

        # Verify output
        output_path = self.interim_dir / "test_merged.csv"
        self.assertTrue(output_path.exists())

        merged_df = pd.read_csv(output_path, index_col=0, parse_dates=True)
        self.assertEqual(len(merged_df), 3)  # Overlap handled
        self.assertEqual(merged_df.index[0], pd.Timestamp("2026-01-01 00:00:00+0000"))

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    @patch("spotforecast2_safe.downloader.entsoe.fetch_data")
    def test_download_new_data_success(self, mock_fetch, mock_get_home):
        """Test successful data download."""
        mock_get_home.return_value = self.test_dir

        # Setup mock fetch_data for start date calculation
        dates = pd.date_range("2026-01-01", periods=5, freq="h", tz="UTC")
        mock_fetch.return_value = pd.DataFrame(index=dates)

        # Patch sys.modules to inject a mock entsoe module
        import sys
        from unittest.mock import MagicMock

        mock_entsoe_mod = MagicMock()
        mock_client_class = mock_entsoe_mod.EntsoePandasClient
        mock_client = mock_client_class.return_value
        mock_df = pd.DataFrame(
            {"Actual": [123]}, index=[dates[-1] + pd.Timedelta(hours=1)]
        )
        mock_client.query_load_and_forecast.return_value = mock_df
        sys.modules["entsoe"] = mock_entsoe_mod

        download_new_data(api_key="fake_key", force=True)

        # Verify client call
        mock_client_class.assert_called_once_with(api_key="fake_key")

        # Verify file creation in raw
        raw_files = list(self.raw_dir.glob("entsoe_load_*.csv"))
        self.assertEqual(len(raw_files), 1)

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    @patch("spotforecast2_safe.downloader.entsoe.fetch_data")
    def test_download_new_data_cooldown(self, mock_fetch, mock_get_home):
        """Test that download is skipped if too recent."""
        mock_get_home.return_value = self.test_dir

        # Last index is very recent
        now = pd.Timestamp.now(tz="UTC")
        mock_fetch.return_value = pd.DataFrame(index=[now - pd.Timedelta(hours=2)])

        import sys
        from unittest.mock import MagicMock

        mock_entsoe_mod = MagicMock()
        mock_client_class = mock_entsoe_mod.EntsoePandasClient
        sys.modules["entsoe"] = mock_entsoe_mod

        download_new_data(api_key="fake_key", force=False)

        mock_client_class.assert_not_called()


if __name__ == "__main__":
    unittest.main()
