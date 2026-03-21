# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for ForecasterRecursiveModel.from_config and subclass inheritance."""

import pandas as pd

from spotforecast2_safe.manager.configurator.config_entsoe import ConfigEntsoe
from spotforecast2_safe.manager.configurator.config_multi import ConfigMulti
from spotforecast2_safe.manager.models.forecaster_recursive_lgbm import (
    ForecasterRecursiveLGBM,
)
from spotforecast2_safe.manager.models.forecaster_recursive_model import (
    ForecasterRecursiveModel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cfg(**overrides):
    """Return a ConfigMulti with optional overrides applied."""
    cfg = ConfigMulti()
    if overrides:
        cfg.set_params(**overrides)
    return cfg


# ---------------------------------------------------------------------------
# Basic from_config behaviour
# ---------------------------------------------------------------------------


class TestFromConfigBasic:
    """Core from_config functionality on the base model class."""

    def test_returns_model_instance(self):
        model = ForecasterRecursiveModel.from_config(iteration=0, config=_cfg())
        assert isinstance(model, ForecasterRecursiveModel)

    def test_iteration_is_set(self):
        model = ForecasterRecursiveModel.from_config(iteration=5, config=_cfg())
        assert model.iteration == 5

    def test_predict_size_from_config(self):
        cfg = _cfg(predict_size=48)
        model = ForecasterRecursiveModel.from_config(iteration=0, config=cfg)
        assert model.predict_size == 48

    def test_refit_size_from_config(self):
        cfg = _cfg(refit_size=14)
        model = ForecasterRecursiveModel.from_config(iteration=0, config=cfg)
        assert model.refit_size == 14

    def test_random_state_from_config(self):
        cfg = _cfg(random_state=42)
        model = ForecasterRecursiveModel.from_config(iteration=0, config=cfg)
        assert model.random_state == 42

    def test_train_size_from_config(self):
        ts = pd.Timedelta(days=100)
        cfg = _cfg(train_size=ts)
        model = ForecasterRecursiveModel.from_config(iteration=0, config=cfg)
        assert model.train_size == ts


# ---------------------------------------------------------------------------
# Name mismatch translation
# ---------------------------------------------------------------------------


class TestFromConfigNameTranslation:
    """Verify the two known name mismatches are correctly translated."""

    def test_api_country_code_maps_to_country_code(self):
        # API_COUNTRY_CODE is a read-only property alias for country_code;
        # set country_code directly to change the ISO code.
        cfg = _cfg(country_code="FR")
        model = ForecasterRecursiveModel.from_config(iteration=0, config=cfg)
        # country_code is stored inside the preprocessor
        assert model.preprocessor.country_code == "FR"

    def test_end_train_default_maps_to_end_dev(self):
        cfg = _cfg(end_train_default="2025-06-15 00:00+00:00")
        model = ForecasterRecursiveModel.from_config(iteration=0, config=cfg)
        assert model.end_dev == pd.Timestamp("2025-06-15 00:00+00:00")


# ---------------------------------------------------------------------------
# Overrides take precedence
# ---------------------------------------------------------------------------


class TestFromConfigOverrides:
    """Caller-supplied overrides must beat config values."""

    def test_override_predict_size(self):
        cfg = _cfg(predict_size=24)
        model = ForecasterRecursiveModel.from_config(
            iteration=0, config=cfg, predict_size=72
        )
        assert model.predict_size == 72

    def test_override_random_state(self):
        cfg = _cfg(random_state=1)
        model = ForecasterRecursiveModel.from_config(
            iteration=0, config=cfg, random_state=999
        )
        assert model.random_state == 999

    def test_override_end_dev_beats_config(self):
        cfg = _cfg(end_train_default="2025-06-15 00:00+00:00")
        model = ForecasterRecursiveModel.from_config(
            iteration=0, config=cfg, end_dev="2025-01-01 00:00+00:00"
        )
        assert model.end_dev == pd.Timestamp("2025-01-01 00:00+00:00")

    def test_override_name(self):
        model = ForecasterRecursiveModel.from_config(
            iteration=0, config=_cfg(), name="custom_name"
        )
        assert model.name == "custom_name"


# ---------------------------------------------------------------------------
# Subclass inheritance (ForecasterRecursiveLGBM)
# ---------------------------------------------------------------------------


class TestFromConfigSubclass:
    """from_config must work correctly on subclasses."""

    def test_returns_subclass_instance(self):
        model = ForecasterRecursiveLGBM.from_config(iteration=1, config=_cfg())
        assert isinstance(model, ForecasterRecursiveLGBM)

    def test_subclass_has_forecaster(self):
        model = ForecasterRecursiveLGBM.from_config(iteration=1, config=_cfg())
        assert model.forecaster is not None

    def test_subclass_inherits_config_values(self):
        cfg = _cfg(predict_size=48, random_state=7)
        model = ForecasterRecursiveLGBM.from_config(iteration=1, config=cfg)
        assert model.predict_size == 48
        assert model.random_state == 7

    def test_subclass_lags_override(self):
        model = ForecasterRecursiveLGBM.from_config(iteration=1, config=_cfg(), lags=24)
        assert len(model.forecaster.lags) == 24


# ---------------------------------------------------------------------------
# Works with both ConfigEntsoe and ConfigMulti
# ---------------------------------------------------------------------------


class TestFromConfigBothConfigTypes:
    """Both config classes must work identically."""

    def test_config_entsoe(self):
        cfg = ConfigEntsoe(api_country_code="ES", predict_size=12)
        model = ForecasterRecursiveModel.from_config(iteration=0, config=cfg)
        assert model.predict_size == 12
        assert model.preprocessor.country_code == "ES"

    def test_config_multi(self):
        cfg = ConfigMulti(country_code="PL", predict_size=36)
        model = ForecasterRecursiveModel.from_config(iteration=0, config=cfg)
        assert model.predict_size == 36
        assert model.preprocessor.country_code == "PL"


# ---------------------------------------------------------------------------
# Config attributes not relevant to the model are ignored
# ---------------------------------------------------------------------------


class TestFromConfigIgnoresIrrelevant:
    """Config-only attributes (delta_val, data_filename, etc.) must not leak."""

    def test_delta_val_not_set_on_model(self):
        model = ForecasterRecursiveModel.from_config(iteration=0, config=_cfg())
        # delta_val is not a model parameter — it must not appear
        assert not hasattr(model, "delta_val")

    def test_data_filename_not_set_on_model(self):
        model = ForecasterRecursiveModel.from_config(iteration=0, config=_cfg())
        assert not hasattr(model, "data_filename")

    def test_n_hyperparameters_trials_not_set_on_model(self):
        model = ForecasterRecursiveModel.from_config(iteration=0, config=_cfg())
        assert not hasattr(model, "n_hyperparameters_trials")

    def test_lags_consider_not_set_on_model(self):
        model = ForecasterRecursiveModel.from_config(iteration=0, config=_cfg())
        assert not hasattr(model, "lags_consider")
