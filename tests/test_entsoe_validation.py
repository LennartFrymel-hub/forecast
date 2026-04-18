# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from spotforecast2_safe.downloader.entsoe import merge_build_manual


class TestEntsoeValidation(unittest.TestCase):
    """Validation tests for the ENTSO-E data merging process."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.raw_dir = self.test_dir / "raw"
        self.raw_dir.mkdir()
        self.interim_dir = self.test_dir / "interim"
        self.interim_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("spotforecast2_safe.downloader.entsoe.get_data_home")
    def test_merge_preserves_actual_load(self, mock_get_home):
        """Test that 'Actual Load' column is preserved correctly."""
        mock_get_home.return_value = self.test_dir

        # File 1: Standard format with "Actual Load"
        df1 = pd.DataFrame(
            {
                "Time (UTC)": ["2026-01-01 00:00", "2026-01-01 01:00"],
                "Actual Load": [100.0, 110.0],
                "Forecasted Load": [105.0, 115.0],
            }
        )

        # File 2: Standard format with "Actual Load"
        df2 = pd.DataFrame(
            {
                "Time (UTC)": ["2026-01-01 02:00", "2026-01-01 03:00"],
                "Actual Load": [120.0, 130.0],
                "Forecasted Load": [125.0, 135.0],
            }
        )

        df1.to_csv(self.raw_dir / "old_format.csv", index=False)
        df2.to_csv(self.raw_dir / "new_format.csv", index=False)

        merge_build_manual(output_file="validation_data.csv")

        output_path = self.interim_dir / "validation_data.csv"
        self.assertTrue(output_path.exists())

        merged_df = pd.read_csv(output_path, index_col=0, parse_dates=True)

        # 1. Validate Columns
        self.assertIn("Actual Load", merged_df.columns)
        self.assertNotIn("Actual", merged_df.columns)
        self.assertIn("Forecasted Load", merged_df.columns)

        # 2. Validate Shape (4 unique hours)
        self.assertEqual(len(merged_df), 4)

        # 3. Validate Data Integrity (No NaNs in target columns)
        self.assertFalse(
            merged_df["Actual Load"].isna().any(),
            "Found NaNs in Actual Load - merging failed",
        )

        # Check specific values to ensure alignment
        # 00:00 -> 100.0 (from Actual)
        self.assertEqual(
            merged_df.loc[pd.Timestamp("2026-01-01 00:00:00+00:00"), "Actual Load"],
            100.0,
        )
        # 02:00 -> 120.0 (from Actual Load)
        self.assertEqual(
            merged_df.loc[pd.Timestamp("2026-01-01 02:00:00+00:00"), "Actual Load"],
            120.0,
        )


if __name__ == "__main__":
    unittest.main()
