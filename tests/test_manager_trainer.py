# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from spotforecast2_safe.manager.trainer import get_last_model


class TestTrainer(unittest.TestCase):
    """Tests for the trainer manager."""

    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch("spotforecast2_safe.manager.trainer.load")
    def test_get_last_model(self, mock_load):
        """Test get_last_model retrieval logic."""
        model_name = "testmodel"
        # Create dummy files
        (self.test_dir / f"{model_name}_forecaster_0.joblib").touch()
        (self.test_dir / f"{model_name}_forecaster_5.joblib").touch()
        (self.test_dir / f"{model_name}_forecaster_2.joblib").touch()

        mock_model = MagicMock()
        mock_load.return_value = mock_model

        iteration, model = get_last_model(model_name, self.test_dir)

        self.assertEqual(iteration, 5)
        self.assertEqual(model, mock_model)
        mock_load.assert_called_once_with(
            self.test_dir / f"{model_name}_forecaster_5.joblib"
        )

    def test_get_last_model_not_found(self):
        """Test get_last_model when no files exist."""
        iteration, model = get_last_model("nonexistent", self.test_dir)
        self.assertEqual(iteration, -1)
        self.assertIsNone(model)


if __name__ == "__main__":
    unittest.main()
