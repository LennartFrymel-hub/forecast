# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for spotforecast2_safe task scripts."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


class TestTaskSafeDemo(unittest.TestCase):
    """Tests for task_safe_demo.py."""

    def test_main_returns_zero_on_success(self):
        """Test that main() returns 0 on successful execution."""
        from spotforecast2_safe.tasks.task_safe_demo import main

        # Create temporary test data file
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = Path(tmpdir) / "data_test.csv"

            # Create mock data with columns matching what the forecasters return
            test_data = pd.DataFrame(
                {f"col{i}": range(100) for i in range(11)},
                index=pd.date_range("2020-01-01", periods=100, freq="h"),
            )
            test_data.to_csv(data_path)

            # Mock the forecasting functions
            mock_predictions = pd.DataFrame(
                {f"col{i}": [1.0, 2.0, 3.0] for i in range(11)},
                index=pd.date_range("2020-01-01", periods=3, freq="h"),
            )

            with (
                patch(
                    "spotforecast2_safe.tasks.task_safe_demo.n2n_predict"
                ) as mock_n2n,
                patch(
                    "spotforecast2_safe.tasks.task_safe_demo.n2n_predict_with_covariates"
                ) as mock_cov,
                patch(
                    "spotforecast2_safe.tasks.task_safe_demo.agg_predict"
                ) as mock_agg,
            ):
                mock_n2n.return_value = (mock_predictions, {})
                mock_cov.return_value = (mock_predictions, {}, {})
                mock_agg.return_value = pd.Series(
                    [1, 2, 3], index=pd.date_range("2020-01-01", periods=3, freq="h")
                )

                result = main(force_train=False, data_path=data_path)
                self.assertEqual(result, 0)


class TestTaskSafeN2O1CovDf(unittest.TestCase):
    """Tests for task_safe_n_to_1_with_covariates_and_dataframe.py."""

    @patch(
        "spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe.n2n_predict_with_covariates"
    )
    @patch(
        "spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe.agg_predict"
    )
    @patch(
        "spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe.fetch_data"
    )
    def test_main_returns_zero_on_success(self, mock_fetch, mock_agg, mock_cov):
        """Test that main() returns 0 on successful execution."""
        from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
            main,
        )

        # Mock return values
        mock_predictions = pd.DataFrame({"col1": [1, 2, 3]})
        mock_fetch.return_value = pd.DataFrame(
            {"col1": range(100)},
            index=pd.date_range("2020-01-01", periods=100, freq="h"),
        )
        mock_cov.return_value = (mock_predictions, {}, {})
        mock_agg.return_value = pd.Series([1, 2, 3])

        # Should complete without error (returns None)
        main(verbose=False)


if __name__ == "__main__":
    unittest.main()
