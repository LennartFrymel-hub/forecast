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

    def test_country_code_default(self):
        assert _default().country_code == "DE"

    def test_api_country_code_property_returns_country_code(self):
        """API_COUNTRY_CODE is a read-only property aliasing country_code."""
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

    def test_custom_country_code(self):
        cfg = ConfigMulti(country_code="FR")
        assert cfg.country_code == "FR"

    def test_api_country_code_reflects_country_code(self):
        cfg = ConfigMulti(country_code="FR")
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
            "country_code",
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
            "index_name",
            "data_source",
            "data_test",
            "start_download",
            "end_download",
        }
        assert expected_keys.issubset(p.keys())

    def test_get_params_country_code_value(self):
        cfg = ConfigMulti(country_code="ES")
        assert cfg.get_params()["country_code"] == "ES"

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

    def test_set_country_code_via_kwargs(self):
        cfg = _default()
        cfg.set_params(country_code="PL")
        assert cfg.country_code == "PL"
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


# ---------------------------------------------------------------------------
# country_code / API_COUNTRY_CODE
# ---------------------------------------------------------------------------


class TestConfigMultiCountryCode:
    """Verify country_code is the single source of truth; API_COUNTRY_CODE is a property alias."""

    def test_country_code_default(self):
        assert _default().country_code == "DE"

    def test_api_country_code_property_matches_country_code(self):
        assert _default().API_COUNTRY_CODE == _default().country_code

    def test_custom_country_code_reflected_in_property(self):
        cfg = ConfigMulti(country_code="FR")
        assert cfg.API_COUNTRY_CODE == "FR"

    def test_api_country_code_is_property(self):
        """API_COUNTRY_CODE must be a property, not a mutable instance attribute."""
        assert isinstance(type(_default()).__dict__["API_COUNTRY_CODE"], property)

    def test_api_country_code_cannot_be_set_directly(self):
        """Setting API_COUNTRY_CODE directly must raise AttributeError (read-only property)."""
        cfg = _default()
        with pytest.raises(AttributeError):
            cfg.API_COUNTRY_CODE = "ES"

    def test_set_params_country_code_updates_property(self):
        cfg = _default()
        cfg.set_params(country_code="IT")
        assert cfg.API_COUNTRY_CODE == "IT"

    def test_country_code_in_get_params(self):
        cfg = ConfigMulti(country_code="ES")
        assert cfg.get_params()["country_code"] == "ES"

    def test_api_country_code_not_in_get_params(self):
        """api_country_code was removed; only country_code should appear."""
        p = _default().get_params()
        assert "api_country_code" not in p


# ---------------------------------------------------------------------------
# New attributes: index_name, data_source, data_test, start_download, end_download
# ---------------------------------------------------------------------------


class TestConfigMultiNewAttributes:
    """Verify defaults and custom values for the new data-source attributes."""

    def test_index_name_default(self):
        assert _default().index_name == "DateTime"

    def test_data_source_default(self):
        assert _default().data_source == "data_in.csv"

    def test_data_test_default(self):
        assert _default().data_test == "data_test.csv"

    def test_start_download_default_is_none(self):
        assert _default().start_download is None

    def test_end_download_default_is_none(self):
        assert _default().end_download is None

    def test_custom_index_name(self):
        cfg = ConfigMulti(index_name="Timestamp")
        assert cfg.index_name == "Timestamp"

    def test_custom_data_source(self):
        cfg = ConfigMulti(data_source="demo10.csv")
        assert cfg.data_source == "demo10.csv"

    def test_custom_data_test(self):
        cfg = ConfigMulti(data_test="demo11.csv")
        assert cfg.data_test == "demo11.csv"

    def test_custom_start_download(self):
        cfg = ConfigMulti(start_download="202401010000")
        assert cfg.start_download == "202401010000"

    def test_custom_end_download(self):
        cfg = ConfigMulti(end_download="202412312300")
        assert cfg.end_download == "202412312300"

    def test_new_attrs_in_get_params(self):
        p = _default().get_params()
        assert "index_name" in p
        assert "data_source" in p
        assert "data_test" in p
        assert "start_download" in p
        assert "end_download" in p

    def test_new_attrs_values_in_get_params(self):
        cfg = ConfigMulti(
            index_name="ts",
            data_source="data_in.csv",
            data_test="data_test.csv",
            start_download="202401010000",
            end_download="202412312300",
        )
        p = cfg.get_params()
        assert p["index_name"] == "ts"
        assert p["data_source"] == "data_in.csv"
        assert p["data_test"] == "data_test.csv"
        assert p["start_download"] == "202401010000"
        assert p["end_download"] == "202412312300"

    def test_set_params_index_name(self):
        cfg = _default()
        cfg.set_params(index_name="ts")
        assert cfg.index_name == "ts"

    def test_set_params_data_source(self):
        cfg = _default()
        cfg.set_params(data_source="demo10.csv")
        assert cfg.data_source == "demo10.csv"

    def test_set_params_data_test(self):
        cfg = _default()
        cfg.set_params(data_test="demo11.csv")
        assert cfg.data_test == "demo11.csv"

    def test_set_params_start_download(self):
        cfg = _default()
        cfg.set_params(start_download="202401010000")
        assert cfg.start_download == "202401010000"

    def test_set_params_end_download(self):
        cfg = _default()
        cfg.set_params(end_download="202412312300")
        assert cfg.end_download == "202412312300"

    def test_direct_assignment_start_download(self):
        cfg = _default()
        cfg.start_download = "202401010000"
        assert cfg.start_download == "202401010000"

    def test_direct_assignment_end_download(self):
        cfg = _default()
        cfg.end_download = "202412312300"
        assert cfg.end_download == "202412312300"

    def test_new_attrs_preserved_alongside_targets(self):
        cfg = ConfigMulti(
            targets=["A"],
            index_name="DateTime",
            data_source="data_in.csv",
            start_download="202401010000",
        )
        assert cfg.targets == ["A"]
        assert cfg.index_name == "DateTime"
        assert cfg.data_source == "data_in.csv"
        assert cfg.start_download == "202401010000"


# ---------------------------------------------------------------------------
# Derived date ranges and bounds attributes
# ---------------------------------------------------------------------------


class TestConfigMultiDerivedAttributes:
    """Tests for data_start, data_end, cov_start, cov_end, bounds."""

    # --- Defaults ---

    def test_data_start_default_is_none(self):
        assert ConfigMulti().data_start is None

    def test_data_end_default_is_none(self):
        assert ConfigMulti().data_end is None

    def test_cov_start_default_is_none(self):
        assert ConfigMulti().cov_start is None

    def test_cov_end_default_is_none(self):
        assert ConfigMulti().cov_end is None

    def test_bounds_default_is_none(self):
        assert ConfigMulti().bounds is None

    # --- Custom init values ---

    def test_custom_data_start(self):
        ts = pd.Timestamp("2022-01-01", tz="UTC")
        cfg = ConfigMulti(data_start=ts)
        assert cfg.data_start == ts

    def test_custom_data_end(self):
        ts = pd.Timestamp("2024-12-31", tz="UTC")
        cfg = ConfigMulti(data_end=ts)
        assert cfg.data_end == ts

    def test_custom_cov_start(self):
        ts = pd.Timestamp("2022-01-01", tz="UTC")
        cfg = ConfigMulti(cov_start=ts)
        assert cfg.cov_start == ts

    def test_custom_cov_end(self):
        ts = pd.Timestamp("2025-01-01", tz="UTC")
        cfg = ConfigMulti(cov_end=ts)
        assert cfg.cov_end == ts

    def test_custom_bounds(self):
        b = [(-100, 100), (0, 500)]
        cfg = ConfigMulti(bounds=b)
        assert cfg.bounds == b

    # --- In get_params ---

    def test_derived_attrs_in_get_params(self):
        p = ConfigMulti().get_params()
        for key in ("data_start", "data_end", "cov_start", "cov_end", "bounds"):
            assert key in p

    def test_derived_attrs_default_values_in_get_params(self):
        p = ConfigMulti().get_params()
        assert p["data_start"] is None
        assert p["data_end"] is None
        assert p["cov_start"] is None
        assert p["cov_end"] is None
        assert p["bounds"] is None

    def test_derived_attrs_custom_values_in_get_params(self):
        ts = pd.Timestamp("2023-06-15", tz="UTC")
        b = [(0, 1000)]
        cfg = ConfigMulti(
            data_start=ts, data_end=ts, cov_start=ts, cov_end=ts, bounds=b
        )
        p = cfg.get_params()
        assert p["data_start"] == ts
        assert p["data_end"] == ts
        assert p["cov_start"] == ts
        assert p["cov_end"] == ts
        assert p["bounds"] == b

    # --- set_params ---

    def test_set_params_data_start(self):
        ts = pd.Timestamp("2022-03-01", tz="UTC")
        cfg = ConfigMulti()
        cfg.set_params(data_start=ts)
        assert cfg.data_start == ts

    def test_set_params_data_end(self):
        ts = pd.Timestamp("2024-06-30", tz="UTC")
        cfg = ConfigMulti()
        cfg.set_params(data_end=ts)
        assert cfg.data_end == ts

    def test_set_params_cov_start(self):
        ts = pd.Timestamp("2022-03-01", tz="UTC")
        cfg = ConfigMulti()
        cfg.set_params(cov_start=ts)
        assert cfg.cov_start == ts

    def test_set_params_cov_end(self):
        ts = pd.Timestamp("2025-01-01", tz="UTC")
        cfg = ConfigMulti()
        cfg.set_params(cov_end=ts)
        assert cfg.cov_end == ts

    def test_set_params_bounds(self):
        b = [(-500, 500), (0, 200)]
        cfg = ConfigMulti()
        cfg.set_params(bounds=b)
        assert cfg.bounds == b

    def test_set_params_all_derived_at_once(self):
        ts_s = pd.Timestamp("2022-01-01", tz="UTC")
        ts_e = pd.Timestamp("2024-12-31", tz="UTC")
        ts_ce = pd.Timestamp("2025-01-01", tz="UTC")
        b = [(0, 100)]
        cfg = ConfigMulti()
        cfg.set_params(
            data_start=ts_s, data_end=ts_e, cov_start=ts_s, cov_end=ts_ce, bounds=b
        )
        assert cfg.data_start == ts_s
        assert cfg.data_end == ts_e
        assert cfg.cov_start == ts_s
        assert cfg.cov_end == ts_ce
        assert cfg.bounds == b

    def test_set_params_method_chaining(self):
        ts = pd.Timestamp("2023-01-01", tz="UTC")
        cfg = ConfigMulti()
        result = cfg.set_params(data_start=ts)
        assert result is cfg

    # --- Direct assignment ---

    def test_direct_assignment_data_start(self):
        ts = pd.Timestamp("2021-07-01", tz="UTC")
        cfg = ConfigMulti()
        cfg.data_start = ts
        assert cfg.data_start == ts

    def test_direct_assignment_data_end(self):
        ts = pd.Timestamp("2023-12-31", tz="UTC")
        cfg = ConfigMulti()
        cfg.data_end = ts
        assert cfg.data_end == ts

    def test_direct_assignment_cov_start(self):
        ts = pd.Timestamp("2021-07-01", tz="UTC")
        cfg = ConfigMulti()
        cfg.cov_start = ts
        assert cfg.cov_start == ts

    def test_direct_assignment_cov_end(self):
        ts = pd.Timestamp("2024-01-25", tz="UTC")
        cfg = ConfigMulti()
        cfg.cov_end = ts
        assert cfg.cov_end == ts

    def test_direct_assignment_bounds(self):
        b = [(-2500, 4500), (-10, 3000)]
        cfg = ConfigMulti()
        cfg.bounds = b
        assert cfg.bounds == b

    # --- Isolation between instances ---

    def test_bounds_not_shared_between_instances(self):
        b = [(0, 100)]
        cfg1 = ConfigMulti(bounds=b)
        cfg2 = ConfigMulti()
        assert cfg2.bounds is None
        cfg1.bounds.append((200, 300))
        assert cfg2.bounds is None

    # --- Coexistence with other params ---

    def test_derived_attrs_preserved_alongside_targets(self):
        ts = pd.Timestamp("2023-01-01", tz="UTC")
        b = [(0, 500)]
        cfg = ConfigMulti(targets=["X"], data_start=ts, bounds=b)
        assert cfg.targets == ["X"]
        assert cfg.data_start == ts
        assert cfg.bounds == b


# ---------------------------------------------------------------------------
# Pipeline control attributes: verbose, cache_data, cache_home, data_home,
# end_train_ts, start_train_ts, n_trials_optuna, n_trials_spotoptim,
# n_initial_spotoptim, task
# ---------------------------------------------------------------------------


class TestConfigMultiPipelineAttributes:
    """Tests for the 10 new pipeline-control and tuning attributes."""

    # --- Defaults ---

    def test_verbose_default_is_false(self):
        assert ConfigMulti().verbose is False

    def test_cache_data_default_is_false(self):
        assert ConfigMulti().cache_data is False

    def test_cache_home_default_is_none(self):
        assert ConfigMulti().cache_home is None

    def test_data_home_default_is_none(self):
        assert ConfigMulti().data_home is None

    def test_end_train_ts_default_is_none(self):
        assert ConfigMulti().end_train_ts is None

    def test_start_train_ts_default_is_none(self):
        assert ConfigMulti().start_train_ts is None

    def test_n_trials_optuna_default(self):
        assert ConfigMulti().n_trials_optuna == 15

    def test_n_trials_spotoptim_default(self):
        assert ConfigMulti().n_trials_spotoptim == 10

    def test_n_initial_spotoptim_default(self):
        assert ConfigMulti().n_initial_spotoptim == 5

    def test_task_default_is_lazy(self):
        assert ConfigMulti().task == "lazy"

    # --- Custom init values ---

    def test_custom_verbose(self):
        assert ConfigMulti(verbose=True).verbose is True

    def test_custom_cache_data(self):
        assert ConfigMulti(cache_data=True).cache_data is True

    def test_custom_cache_home(self):
        cfg = ConfigMulti(cache_home="/tmp/cache")
        assert cfg.cache_home == "/tmp/cache"

    def test_custom_data_home(self):
        cfg = ConfigMulti(data_home="/tmp/data")
        assert cfg.data_home == "/tmp/data"

    def test_custom_end_train_ts(self):
        ts = pd.Timestamp("2024-12-31", tz="UTC")
        assert ConfigMulti(end_train_ts=ts).end_train_ts == ts

    def test_custom_start_train_ts(self):
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        assert ConfigMulti(start_train_ts=ts).start_train_ts == ts

    def test_custom_n_trials_optuna(self):
        assert ConfigMulti(n_trials_optuna=50).n_trials_optuna == 50

    def test_custom_n_trials_spotoptim(self):
        assert ConfigMulti(n_trials_spotoptim=20).n_trials_spotoptim == 20

    def test_custom_n_initial_spotoptim(self):
        assert ConfigMulti(n_initial_spotoptim=8).n_initial_spotoptim == 8

    def test_custom_task(self):
        for task in ("lazy", "training", "optuna", "spotoptim"):
            assert ConfigMulti(task=task).task == task

    # --- In get_params ---

    def test_all_new_attrs_in_get_params(self):
        p = ConfigMulti().get_params()
        for key in (
            "verbose",
            "cache_data",
            "cache_home",
            "data_home",
            "end_train_ts",
            "start_train_ts",
            "n_trials_optuna",
            "n_trials_spotoptim",
            "n_initial_spotoptim",
            "task",
        ):
            assert key in p, f"'{key}' missing from get_params()"

    def test_new_attrs_default_values_in_get_params(self):
        p = ConfigMulti().get_params()
        assert p["verbose"] is False
        assert p["cache_data"] is False
        assert p["cache_home"] is None
        assert p["data_home"] is None
        assert p["end_train_ts"] is None
        assert p["start_train_ts"] is None
        assert p["n_trials_optuna"] == 15
        assert p["n_trials_spotoptim"] == 10
        assert p["n_initial_spotoptim"] == 5
        assert p["task"] == "lazy"

    def test_custom_values_reflected_in_get_params(self):
        ts = pd.Timestamp("2023-06-01", tz="UTC")
        cfg = ConfigMulti(
            verbose=True,
            cache_data=True,
            cache_home="/c",
            data_home="/d",
            end_train_ts=ts,
            start_train_ts=ts,
            n_trials_optuna=30,
            n_trials_spotoptim=25,
            n_initial_spotoptim=10,
            task="optuna",
        )
        p = cfg.get_params()
        assert p["verbose"] is True
        assert p["cache_data"] is True
        assert p["cache_home"] == "/c"
        assert p["data_home"] == "/d"
        assert p["end_train_ts"] == ts
        assert p["start_train_ts"] == ts
        assert p["n_trials_optuna"] == 30
        assert p["n_trials_spotoptim"] == 25
        assert p["n_initial_spotoptim"] == 10
        assert p["task"] == "optuna"

    # --- set_params ---

    def test_set_params_verbose(self):
        cfg = ConfigMulti()
        cfg.set_params(verbose=True)
        assert cfg.verbose is True

    def test_set_params_cache_data(self):
        cfg = ConfigMulti()
        cfg.set_params(cache_data=True)
        assert cfg.cache_data is True

    def test_set_params_cache_home(self):
        cfg = ConfigMulti()
        cfg.set_params(cache_home="/cache")
        assert cfg.cache_home == "/cache"

    def test_set_params_data_home(self):
        cfg = ConfigMulti()
        cfg.set_params(data_home="/data")
        assert cfg.data_home == "/data"

    def test_set_params_end_train_ts(self):
        ts = pd.Timestamp("2024-06-30", tz="UTC")
        cfg = ConfigMulti()
        cfg.set_params(end_train_ts=ts)
        assert cfg.end_train_ts == ts

    def test_set_params_start_train_ts(self):
        ts = pd.Timestamp("2023-07-01", tz="UTC")
        cfg = ConfigMulti()
        cfg.set_params(start_train_ts=ts)
        assert cfg.start_train_ts == ts

    def test_set_params_n_trials_optuna(self):
        cfg = ConfigMulti()
        cfg.set_params(n_trials_optuna=100)
        assert cfg.n_trials_optuna == 100

    def test_set_params_n_trials_spotoptim(self):
        cfg = ConfigMulti()
        cfg.set_params(n_trials_spotoptim=50)
        assert cfg.n_trials_spotoptim == 50

    def test_set_params_n_initial_spotoptim(self):
        cfg = ConfigMulti()
        cfg.set_params(n_initial_spotoptim=20)
        assert cfg.n_initial_spotoptim == 20

    def test_set_params_task(self):
        cfg = ConfigMulti()
        cfg.set_params(task="spotoptim")
        assert cfg.task == "spotoptim"

    def test_set_params_method_chaining(self):
        cfg = ConfigMulti()
        result = cfg.set_params(verbose=True, task="training")
        assert result is cfg

    # --- Direct assignment ---

    def test_direct_assignment_verbose(self):
        cfg = ConfigMulti()
        cfg.verbose = True
        assert cfg.verbose is True

    def test_direct_assignment_end_train_ts(self):
        ts = pd.Timestamp("2025-01-01", tz="UTC")
        cfg = ConfigMulti()
        cfg.end_train_ts = ts
        assert cfg.end_train_ts == ts

    def test_direct_assignment_start_train_ts(self):
        ts = pd.Timestamp("2024-01-01", tz="UTC")
        cfg = ConfigMulti()
        cfg.start_train_ts = ts
        assert cfg.start_train_ts == ts

    def test_direct_assignment_task(self):
        cfg = ConfigMulti()
        cfg.task = "training"
        assert cfg.task == "training"

    def test_direct_assignment_n_trials_optuna(self):
        cfg = ConfigMulti()
        cfg.n_trials_optuna = 200
        assert cfg.n_trials_optuna == 200

    # --- Coexistence with existing params ---

    def test_pipeline_attrs_alongside_country_code(self):
        cfg = ConfigMulti(
            country_code="FR", verbose=True, task="optuna", n_trials_optuna=20
        )
        assert cfg.country_code == "FR"
        assert cfg.verbose is True
        assert cfg.task == "optuna"
        assert cfg.n_trials_optuna == 20


# ---------------------------------------------------------------------------
# agg_weights attribute
# ---------------------------------------------------------------------------


_WEIGHTS = [1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, 1.0]


class TestConfigMultiAggWeights:
    """Tests for the agg_weights attribute."""

    # --- Defaults ---

    def test_agg_weights_default_is_none(self):
        assert ConfigMulti().agg_weights is None

    # --- Constructor ---

    def test_agg_weights_set_in_constructor(self):
        cfg = ConfigMulti(agg_weights=_WEIGHTS)
        assert cfg.agg_weights == _WEIGHTS

    def test_agg_weights_empty_list(self):
        cfg = ConfigMulti(agg_weights=[])
        assert cfg.agg_weights == []

    def test_agg_weights_single_element(self):
        cfg = ConfigMulti(agg_weights=[1.0])
        assert cfg.agg_weights == [1.0]

    def test_agg_weights_all_positive(self):
        weights = [1.0, 2.0, 0.5]
        cfg = ConfigMulti(agg_weights=weights)
        assert cfg.agg_weights == weights

    def test_agg_weights_mixed_signs(self):
        weights = [1.0, -1.0, 0.5, -0.5]
        cfg = ConfigMulti(agg_weights=weights)
        assert cfg.agg_weights == weights

    # --- Direct assignment ---

    def test_agg_weights_direct_assignment(self):
        cfg = ConfigMulti()
        cfg.agg_weights = _WEIGHTS
        assert cfg.agg_weights == _WEIGHTS

    def test_agg_weights_direct_assignment_to_none(self):
        cfg = ConfigMulti(agg_weights=_WEIGHTS)
        cfg.agg_weights = None
        assert cfg.agg_weights is None

    # --- Slicing (pipeline usage) ---

    def test_agg_weights_slicing_for_active_targets(self):
        """Slicing to match len(config.targets) is the intended usage pattern."""
        cfg = ConfigMulti(agg_weights=_WEIGHTS, targets=["A", "B", "C"])
        active = cfg.agg_weights[: len(cfg.targets)]
        assert active == _WEIGHTS[:3]
        assert len(active) == 3

    def test_agg_weights_slice_single_target(self):
        cfg = ConfigMulti(agg_weights=_WEIGHTS, targets=["A"])
        active = cfg.agg_weights[: len(cfg.targets)]
        assert active == [1.0]

    # --- get_params() ---

    def test_agg_weights_in_get_params(self):
        cfg = ConfigMulti(agg_weights=_WEIGHTS)
        p = cfg.get_params()
        assert "agg_weights" in p
        assert p["agg_weights"] == _WEIGHTS

    def test_agg_weights_none_in_get_params(self):
        p = ConfigMulti().get_params()
        assert "agg_weights" in p
        assert p["agg_weights"] is None

    def test_agg_weights_in_get_params_shallow(self):
        cfg = ConfigMulti(agg_weights=[1.0, -1.0])
        p = cfg.get_params(deep=False)
        assert "agg_weights" in p
        assert p["agg_weights"] == [1.0, -1.0]

    def test_agg_weights_in_get_params_deep(self):
        cfg = ConfigMulti(agg_weights=[1.0, -1.0])
        p = cfg.get_params(deep=True)
        assert "agg_weights" in p
        assert p["agg_weights"] == [1.0, -1.0]

    # --- set_params() ---

    def test_set_params_agg_weights_via_kwargs(self):
        cfg = ConfigMulti()
        cfg.set_params(agg_weights=_WEIGHTS)
        assert cfg.agg_weights == _WEIGHTS

    def test_set_params_agg_weights_via_dict(self):
        cfg = ConfigMulti()
        cfg.set_params(params={"agg_weights": [1.0, -1.0]})
        assert cfg.agg_weights == [1.0, -1.0]

    def test_set_params_agg_weights_to_none(self):
        cfg = ConfigMulti(agg_weights=_WEIGHTS)
        cfg.set_params(agg_weights=None)
        assert cfg.agg_weights is None

    def test_set_params_agg_weights_method_chaining(self):
        cfg = ConfigMulti()
        result = cfg.set_params(agg_weights=[1.0]).set_params(predict_size=48)
        assert result.agg_weights == [1.0]
        assert result.predict_size == 48

    # --- Coexistence ---

    def test_agg_weights_preserved_alongside_targets(self):
        cfg = ConfigMulti(targets=["A", "B"], agg_weights=[1.0, -1.0])
        assert cfg.targets == ["A", "B"]
        assert cfg.agg_weights == [1.0, -1.0]

    def test_agg_weights_alongside_other_params(self):
        cfg = ConfigMulti(
            country_code="FR",
            predict_size=48,
            agg_weights=_WEIGHTS,
        )
        assert cfg.country_code == "FR"
        assert cfg.predict_size == 48
        assert cfg.agg_weights == _WEIGHTS
