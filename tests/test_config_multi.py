# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Pytest tests for ConfigMulti in manager.configurator.config_multi.

Covers:
- Default parameter values
- Custom parameter initialization
- get_params() shallow and deep
- set_params() flat parameters and deep periods__ notation
- set_params() method chaining
- set_params() error handling (invalid param, invalid period name, bad format)
- Immutability: original Period dataclasses are not mutated
"""

import pandas as pd
import pytest

from spotforecast2_safe.data import Period
from spotforecast2_safe.manager.configurator.config_multi import ConfigMulti


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default() -> ConfigMulti:
    """Return a ConfigMulti with default parameters."""
    return ConfigMulti()


# ---------------------------------------------------------------------------
# Default parameter values
# ---------------------------------------------------------------------------


class TestConfigMultiDefaults:
    """Verify that all default values match the documented defaults."""

    def test_api_country_code_default(self):
        assert _default().API_COUNTRY_CODE == "DE"

    def test_predict_size_default(self):
        assert _default().predict_size == 24

    def test_refit_size_default(self):
        assert _default().refit_size == 7

    def test_random_state_default(self):
        assert _default().random_state == 314159

    def test_n_hyperparameters_trials_default(self):
        assert _default().n_hyperparameters_trials == 20

    def test_end_train_default(self):
        assert _default().end_train_default == "2025-12-31 00:00+00:00"

    def test_data_filename_default(self):
        assert _default().data_filename == "interim/energy_load.csv"

    def test_train_size_default(self):
        assert _default().train_size == pd.Timedelta(days=3 * 365)

    def test_delta_val_default(self):
        assert _default().delta_val == pd.Timedelta(hours=24 * 7 * 10)

    def test_lags_consider_default(self):
        assert _default().lags_consider == list(range(1, 24))

    def test_periods_default_count(self):
        assert len(_default().periods) == 5

    def test_periods_default_names(self):
        names = [p.name for p in _default().periods]
        assert names == ["daily", "weekly", "monthly", "quarterly", "yearly"]

    def test_periods_daily_n_periods(self):
        daily = next(p for p in _default().periods if p.name == "daily")
        assert daily.n_periods == 12
        assert daily.column == "hour"
        assert daily.input_range == (1, 24)

    def test_periods_weekly_n_periods(self):
        weekly = next(p for p in _default().periods if p.name == "weekly")
        assert weekly.n_periods == 7
        assert weekly.column == "dayofweek"
        assert weekly.input_range == (0, 6)

    def test_periods_yearly_n_periods(self):
        yearly = next(p for p in _default().periods if p.name == "yearly")
        assert yearly.n_periods == 12
        assert yearly.column == "dayofyear"
        assert yearly.input_range == (1, 365)


# ---------------------------------------------------------------------------
# Custom initialization
# ---------------------------------------------------------------------------


class TestConfigMultiCustomInit:
    """Verify that constructor arguments override defaults."""

    def test_custom_api_country_code(self):
        cfg = ConfigMulti(api_country_code="FR")
        assert cfg.API_COUNTRY_CODE == "FR"

    def test_custom_predict_size(self):
        cfg = ConfigMulti(predict_size=48)
        assert cfg.predict_size == 48

    def test_custom_random_state(self):
        cfg = ConfigMulti(random_state=42)
        assert cfg.random_state == 42

    def test_custom_train_size(self):
        t = pd.Timedelta(days=365)
        cfg = ConfigMulti(train_size=t)
        assert cfg.train_size == t

    def test_custom_lags_consider(self):
        lags = [1, 2, 3]
        cfg = ConfigMulti(lags_consider=lags)
        assert cfg.lags_consider == lags

    def test_custom_periods(self):
        custom = [
            Period(name="daily", n_periods=24, column="hour", input_range=(1, 24))
        ]
        cfg = ConfigMulti(periods=custom)
        assert len(cfg.periods) == 1
        assert cfg.periods[0].n_periods == 24

    def test_custom_data_filename(self):
        cfg = ConfigMulti(data_filename="interim/custom.csv")
        assert cfg.data_filename == "interim/custom.csv"

    def test_custom_n_hyperparameters_trials(self):
        cfg = ConfigMulti(n_hyperparameters_trials=50)
        assert cfg.n_hyperparameters_trials == 50

    def test_custom_refit_size(self):
        cfg = ConfigMulti(refit_size=14)
        assert cfg.refit_size == 14

    def test_custom_delta_val(self):
        d = pd.Timedelta(days=30)
        cfg = ConfigMulti(delta_val=d)
        assert cfg.delta_val == d


# ---------------------------------------------------------------------------
# get_params()
# ---------------------------------------------------------------------------


class TestConfigMultiGetParams:
    """Verify get_params() returns correct shallow and deep parameter dicts."""

    def test_get_params_returns_dict(self):
        assert isinstance(_default().get_params(), dict)

    def test_get_params_contains_all_flat_keys(self):
        p = _default().get_params(deep=False)
        expected_keys = {
            "api_country_code",
            "periods",
            "lags_consider",
            "train_size",
            "end_train_default",
            "delta_val",
            "predict_size",
            "refit_size",
            "random_state",
            "n_hyperparameters_trials",
            "data_filename",
            "targets",
        }
        assert expected_keys.issubset(p.keys())

    def test_get_params_api_country_code_value(self):
        cfg = ConfigMulti(api_country_code="ES")
        assert cfg.get_params()["api_country_code"] == "ES"

    def test_get_params_predict_size_value(self):
        cfg = ConfigMulti(predict_size=48)
        assert cfg.get_params()["predict_size"] == 48

    def test_get_params_shallow_no_deep_keys(self):
        p = _default().get_params(deep=False)
        deep_keys = [k for k in p if k.startswith("periods__")]
        assert deep_keys == []

    def test_get_params_deep_contains_period_keys(self):
        p = _default().get_params(deep=True)
        assert "periods__daily__n_periods" in p
        assert "periods__weekly__n_periods" in p
        assert "periods__yearly__n_periods" in p

    def test_get_params_deep_daily_n_periods_value(self):
        p = _default().get_params(deep=True)
        assert p["periods__daily__n_periods"] == 12

    def test_get_params_deep_contains_column_keys(self):
        p = _default().get_params(deep=True)
        assert "periods__daily__column" in p
        assert p["periods__daily__column"] == "hour"

    def test_get_params_deep_contains_input_range_keys(self):
        p = _default().get_params(deep=True)
        assert "periods__daily__input_range" in p
        assert p["periods__daily__input_range"] == (1, 24)


# ---------------------------------------------------------------------------
# set_params() — flat parameters
# ---------------------------------------------------------------------------


class TestConfigMultiSetParamsFlat:
    """Verify set_params() correctly updates flat (top-level) attributes."""

    def test_set_api_country_code_via_kwargs(self):
        cfg = _default()
        cfg.set_params(api_country_code="PL")
        assert cfg.API_COUNTRY_CODE == "PL"

    def test_set_predict_size_via_kwargs(self):
        cfg = _default()
        cfg.set_params(predict_size=48)
        assert cfg.predict_size == 48

    def test_set_random_state_via_dict(self):
        cfg = _default()
        cfg.set_params(params={"random_state": 99})
        assert cfg.random_state == 99

    def test_set_params_dict_and_kwargs_merged(self):
        cfg = _default()
        cfg.set_params(params={"predict_size": 12}, random_state=7)
        assert cfg.predict_size == 12
        assert cfg.random_state == 7

    def test_set_params_returns_self(self):
        cfg = _default()
        result = cfg.set_params(predict_size=12)
        assert result is cfg

    def test_set_params_method_chaining(self):
        cfg = _default()
        cfg.set_params(predict_size=12).set_params(random_state=1)
        assert cfg.predict_size == 12
        assert cfg.random_state == 1

    def test_set_params_empty_call_returns_self(self):
        cfg = _default()
        result = cfg.set_params()
        assert result is cfg

    def test_set_params_data_filename(self):
        cfg = _default()
        cfg.set_params(data_filename="interim/other.csv")
        assert cfg.data_filename == "interim/other.csv"

    def test_set_params_refit_size(self):
        cfg = _default()
        cfg.set_params(refit_size=3)
        assert cfg.refit_size == 3

    def test_set_params_n_hyperparameters_trials(self):
        cfg = _default()
        cfg.set_params(n_hyperparameters_trials=100)
        assert cfg.n_hyperparameters_trials == 100


# ---------------------------------------------------------------------------
# set_params() — deep Period parameters
# ---------------------------------------------------------------------------


class TestConfigMultiSetParamsDeep:
    """Verify set_params() with periods__<name>__<param> notation."""

    def test_set_daily_n_periods(self):
        cfg = _default()
        cfg.set_params(periods__daily__n_periods=24)
        daily = next(p for p in cfg.periods if p.name == "daily")
        assert daily.n_periods == 24

    def test_other_periods_unchanged_after_deep_set(self):
        cfg = _default()
        original_weekly = next(p for p in cfg.periods if p.name == "weekly")
        cfg.set_params(periods__daily__n_periods=24)
        weekly = next(p for p in cfg.periods if p.name == "weekly")
        assert weekly.n_periods == original_weekly.n_periods

    def test_set_yearly_n_periods(self):
        cfg = _default()
        cfg.set_params(periods__yearly__n_periods=365)
        yearly = next(p for p in cfg.periods if p.name == "yearly")
        assert yearly.n_periods == 365

    def test_set_weekly_column(self):
        cfg = _default()
        cfg.set_params(periods__weekly__column="dayofweek")
        weekly = next(p for p in cfg.periods if p.name == "weekly")
        assert weekly.column == "dayofweek"

    def test_set_multiple_deep_params(self):
        cfg = _default()
        cfg.set_params(periods__daily__n_periods=6, periods__yearly__n_periods=52)
        daily = next(p for p in cfg.periods if p.name == "daily")
        yearly = next(p for p in cfg.periods if p.name == "yearly")
        assert daily.n_periods == 6
        assert yearly.n_periods == 52

    def test_original_period_objects_not_mutated(self):
        """Period is a frozen dataclass; set_params must replace, not mutate."""
        cfg = _default()
        original_periods = list(cfg.periods)
        original_daily_n = next(
            p for p in original_periods if p.name == "daily"
        ).n_periods
        cfg.set_params(periods__daily__n_periods=99)
        # original_periods list still holds the old Period instances
        old_daily = next(p for p in original_periods if p.name == "daily")
        assert old_daily.n_periods == original_daily_n  # unchanged


# ---------------------------------------------------------------------------
# set_params() — error handling
# ---------------------------------------------------------------------------


class TestConfigMultiSetParamsErrors:
    """Verify set_params() raises ValueError for invalid inputs."""

    def test_invalid_flat_param_raises(self):
        cfg = _default()
        with pytest.raises(ValueError, match="Invalid parameter"):
            cfg.set_params(nonexistent_param=42)

    def test_invalid_period_name_raises(self):
        cfg = _default()
        with pytest.raises(
            ValueError, match="Period with name 'nonexistent' not found"
        ):
            cfg.set_params(periods__nonexistent__n_periods=10)

    def test_invalid_deep_param_format_raises(self):
        """Only 3-part keys (periods__name__param) are allowed."""
        cfg = _default()
        with pytest.raises(ValueError, match="Invalid deep parameter format"):
            cfg.set_params(**{"periods__daily": 10})

    def test_too_many_parts_in_deep_key_raises(self):
        cfg = _default()
        with pytest.raises(ValueError, match="Invalid deep parameter format"):
            cfg.set_params(**{"periods__daily__n_periods__extra": 10})


# ---------------------------------------------------------------------------
# targets attribute
# ---------------------------------------------------------------------------


class TestConfigMultiTargets:
    """Verify the targets attribute behaves correctly."""

    def test_targets_default_is_none(self):
        assert _default().targets is None

    def test_targets_set_in_constructor(self):
        cfg = ConfigMulti(targets=["A", "B", "C"])
        assert cfg.targets == ["A", "B", "C"]

    def test_targets_empty_list(self):
        cfg = ConfigMulti(targets=[])
        assert cfg.targets == []

    def test_targets_single_element(self):
        cfg = ConfigMulti(targets=["A"])
        assert cfg.targets == ["A"]

    def test_targets_direct_assignment(self):
        cfg = _default()
        cfg.targets = ["X", "Y"]
        assert cfg.targets == ["X", "Y"]

    def test_targets_direct_assignment_to_none(self):
        cfg = ConfigMulti(targets=["A", "B"])
        cfg.targets = None
        assert cfg.targets is None

    def test_targets_in_get_params(self):
        cfg = ConfigMulti(targets=["A", "B"])
        p = cfg.get_params()
        assert "targets" in p
        assert p["targets"] == ["A", "B"]

    def test_targets_none_in_get_params(self):
        p = _default().get_params()
        assert "targets" in p
        assert p["targets"] is None

    def test_targets_in_get_params_shallow(self):
        cfg = ConfigMulti(targets=["A"])
        p = cfg.get_params(deep=False)
        assert "targets" in p
        assert p["targets"] == ["A"]

    def test_targets_in_get_params_deep(self):
        cfg = ConfigMulti(targets=["A"])
        p = cfg.get_params(deep=True)
        assert "targets" in p
        assert p["targets"] == ["A"]

    def test_set_params_updates_targets(self):
        cfg = _default()
        cfg.set_params(targets=["A", "B"])
        assert cfg.targets == ["A", "B"]

    def test_set_params_targets_via_dict(self):
        cfg = _default()
        cfg.set_params(params={"targets": ["C", "D"]})
        assert cfg.targets == ["C", "D"]

    def test_set_params_targets_to_none(self):
        cfg = ConfigMulti(targets=["A"])
        cfg.set_params(targets=None)
        assert cfg.targets is None

    def test_set_params_targets_method_chaining(self):
        cfg = _default()
        result = cfg.set_params(targets=["A"]).set_params(predict_size=48)
        assert result.targets == ["A"]
        assert result.predict_size == 48

    def test_targets_contains_flat_key_in_get_params(self):
        """Ensure targets key present even when deep=True (no nesting for targets)."""
        cfg = ConfigMulti(targets=["A", "B"])
        p = cfg.get_params(deep=True)
        # targets is a flat key — no periods__-style expansion expected
        assert p["targets"] == ["A", "B"]
        assert not any(k.startswith("targets__") for k in p)

    def test_targets_list_not_shared_between_instances(self):
        """Mutating one config's targets must not affect another."""
        cfg1 = ConfigMulti(targets=["A", "B"])
        cfg2 = ConfigMulti(targets=["A", "B"])
        cfg1.targets.append("C")
        assert "C" not in cfg2.targets

    def test_targets_preserved_alongside_other_params(self):
        cfg = ConfigMulti(targets=["A"], predict_size=48, random_state=7)
        assert cfg.targets == ["A"]
        assert cfg.predict_size == 48
        assert cfg.random_state == 7
