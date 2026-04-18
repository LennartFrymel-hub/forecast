# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for standalone functions ported from chag25a models.py."""

import tempfile
from pathlib import Path

from joblib import dump

from spotforecast2_safe.manager.trainer import get_path_model, load_iteration

# ------------------------------------------------------------------ #
# get_path_model
# ------------------------------------------------------------------ #


class TestGetPathModel:
    def test_returns_correct_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = get_path_model("lgbm", 3, model_dir=tmpdir)
            assert p.name == "lgbm_forecaster_3.joblib"

    def test_uses_cache_home_if_no_model_dir(self):
        p = get_path_model("xgb", 0)
        assert "xgb_forecaster_0.joblib" in str(p)

    def test_returns_path_object(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            p = get_path_model("lgbm", 1, model_dir=tmpdir)
            assert isinstance(p, Path)


# ------------------------------------------------------------------ #
# load_iteration
# ------------------------------------------------------------------ #


class TestLoadIteration:
    def test_returns_none_for_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = load_iteration("lgbm", 99, model_dir=tmpdir)
            assert result is None

    def test_loads_saved_model(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save a dummy object
            model_path = Path(tmpdir) / "lgbm_forecaster_1.joblib"
            dummy = {"name": "lgbm", "iteration": 1}
            dump(dummy, model_path)

            result = load_iteration("lgbm", 1, model_dir=tmpdir)
            assert result is not None
            assert result["name"] == "lgbm"
            assert result["iteration"] == 1
