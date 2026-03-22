# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""
Tests for apply_imputation() in spotforecast2_safe.preprocessing.imputation.

Coverage:
- Importability from both the submodule and the preprocessing package
- Linear method: fills NaN values, returns weight_func=None, preserves shape
- Weighted method: fills NaN values, returns WeightFunction, correct weights
- Only listed targets are interpolated (linear)
- No-gap data: weighted WeightFunction returns all-ones array (not None)
- Logging: NaN count before/after is emitted at INFO level
- Warning emitted when NaN values remain after imputation
- ValueError raised for unknown imputation_method
"""

import logging
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.preprocessing.imputation import (
    WeightFunction,
    apply_imputation,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_df(
    n: int = 100, gap_slice: slice | None = None, cols: list[str] | None = None
) -> pd.DataFrame:
    if cols is None:
        cols = ["A", "B"]
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(7)
    data = {c: rng.uniform(1, 10, n) for c in cols}
    df = pd.DataFrame(data, index=idx)
    if gap_slice is not None:
        for c in cols:
            df.iloc[gap_slice, df.columns.get_loc(c)] = np.nan
    return df


def _cfg(
    method: str, window_size: int = 10, cols: list[str] | None = None
) -> SimpleNamespace:
    if cols is None:
        cols = ["A", "B"]
    return SimpleNamespace(
        imputation_method=method, targets=cols, window_size=window_size
    )


@pytest.fixture
def df_gap():
    return _make_df(n=100, gap_slice=slice(20, 25))


@pytest.fixture
def df_no_gap():
    return _make_df(n=100)


@pytest.fixture
def log():
    return logging.getLogger("test_sf2safe_apply_imputation")


# ---------------------------------------------------------------------------
# Import paths
# ---------------------------------------------------------------------------


class TestApplyImputationImportPaths:
    """apply_imputation must be importable from the submodule and the package."""

    def test_importable_from_submodule(self):
        from spotforecast2_safe.preprocessing.imputation import apply_imputation as fn

        assert callable(fn)

    def test_importable_from_package(self):
        from spotforecast2_safe.preprocessing import apply_imputation as fn

        assert callable(fn)

    def test_submodule_and_package_are_same_object(self):
        from spotforecast2_safe.preprocessing.imputation import apply_imputation as fn1
        from spotforecast2_safe.preprocessing import apply_imputation as fn2

        assert fn1 is fn2


# ---------------------------------------------------------------------------
# Linear imputation
# ---------------------------------------------------------------------------


class TestApplyImputationLinear:
    """Linear strategy: fill NaN, return weight_func=None."""

    def test_fills_all_nans(self, df_gap, log):
        result, _ = apply_imputation(df_gap.copy(), _cfg("linear"), log)
        assert result.isnull().sum().sum() == 0

    def test_returns_none_weight_func(self, df_gap, log):
        _, wf = apply_imputation(df_gap.copy(), _cfg("linear"), log)
        assert wf is None

    def test_returns_dataframe(self, df_gap, log):
        result, _ = apply_imputation(df_gap.copy(), _cfg("linear"), log)
        assert isinstance(result, pd.DataFrame)

    def test_preserves_shape(self, df_gap, log):
        result, _ = apply_imputation(df_gap.copy(), _cfg("linear"), log)
        assert result.shape == df_gap.shape

    def test_no_gap_values_unchanged(self, df_no_gap, log):
        original_sum = df_no_gap.sum().sum()
        result, _ = apply_imputation(df_no_gap.copy(), _cfg("linear"), log)
        assert pytest.approx(result.sum().sum(), rel=1e-9) == original_sum

    def test_only_listed_targets_filled(self, log):
        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame(
            {
                "A": [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "B": [1.0, None, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            },
            index=idx,
        )
        result, _ = apply_imputation(df, _cfg("linear", cols=["A"]), log)
        assert result["A"].isnull().sum() == 0
        assert result["B"].isnull().sum() == 1  # B was not listed


# ---------------------------------------------------------------------------
# Weighted imputation
# ---------------------------------------------------------------------------


class TestApplyImputationWeighted:
    """Weighted strategy: fill NaN, return WeightFunction with correct weights."""

    def test_fills_all_nans(self, df_gap, log):
        result, _ = apply_imputation(
            df_gap.copy(), _cfg("weighted", window_size=5), log
        )
        assert result.isnull().sum().sum() == 0

    def test_returns_weight_function(self, df_gap, log):
        _, wf = apply_imputation(df_gap.copy(), _cfg("weighted", window_size=5), log)
        assert isinstance(wf, WeightFunction)

    def test_returns_dataframe(self, df_gap, log):
        result, _ = apply_imputation(
            df_gap.copy(), _cfg("weighted", window_size=5), log
        )
        assert isinstance(result, pd.DataFrame)

    def test_preserves_shape(self, df_gap, log):
        result, _ = apply_imputation(
            df_gap.copy(), _cfg("weighted", window_size=5), log
        )
        assert result.shape == df_gap.shape

    def test_weight_func_has_zero_zones_near_gap(self, df_gap, log):
        _, wf = apply_imputation(df_gap.copy(), _cfg("weighted", window_size=5), log)
        assert (wf.weights_series == 0.0).any()

    def test_no_gap_weight_func_returns_ones(self, df_no_gap, log):
        """For gap-free data, WeightFunction must return an all-ones array (not None)."""
        _, wf = apply_imputation(df_no_gap.copy(), _cfg("weighted", window_size=5), log)
        result = wf(df_no_gap.index)
        assert result is not None
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, np.ones(len(df_no_gap)))

    def test_weight_func_callable(self, df_gap, log):
        _, wf = apply_imputation(df_gap.copy(), _cfg("weighted", window_size=5), log)
        assert callable(wf)


# ---------------------------------------------------------------------------
# Unknown method
# ---------------------------------------------------------------------------


class TestApplyImputationUnknownMethod:
    def test_raises_value_error(self, df_gap, log):
        with pytest.raises(ValueError, match="Unknown imputation_method"):
            apply_imputation(df_gap.copy(), _cfg("spline"), log)

    def test_error_contains_method_name(self, df_gap, log):
        with pytest.raises(ValueError, match="kriging"):
            apply_imputation(df_gap.copy(), _cfg("kriging"), log)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TestApplyImputationLogging:
    def _msgs(self, df, config, caplog, level=logging.INFO) -> list[str]:
        logger_name = "test_sf2safe_apply_imputation"
        with caplog.at_level(level, logger=logger_name):
            apply_imputation(df.copy(), config, logging.getLogger(logger_name))
        return [r.message for r in caplog.records if r.levelno == level]

    def test_logs_nan_before(self, df_gap, caplog):
        msgs = self._msgs(df_gap, _cfg("linear"), caplog)
        assert any("before" in m.lower() for m in msgs)

    def test_logs_nan_after(self, df_gap, caplog):
        msgs = self._msgs(df_gap, _cfg("linear"), caplog)
        assert any("after" in m.lower() for m in msgs)

    def test_nan_before_is_positive(self, df_gap, caplog):
        msgs = self._msgs(df_gap, _cfg("linear"), caplog)
        before_msgs = [m for m in msgs if "before" in m.lower()]
        assert before_msgs
        numbers = re.findall(r"\d+", before_msgs[0])
        assert any(int(n) > 0 for n in numbers)

    def test_nan_after_zero_for_linear(self, df_gap, caplog):
        msgs = self._msgs(df_gap, _cfg("linear"), caplog)
        after_msgs = [m for m in msgs if "after" in m.lower()]
        assert after_msgs
        numbers = re.findall(r"\d+", after_msgs[-1])
        assert numbers[-1] == "0"

    def test_warning_when_nans_remain(self, caplog):
        """WARNING must be emitted when imputation leaves NaN cells."""
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame({"A": [None, 2.0, 3.0, 4.0, None]}, index=idx)
        logger_name = "test_sf2safe_apply_imputation"
        from spotforecast2_safe.preprocessing.linearly_interpolate_ts import (
            LinearlyInterpolateTS,
        )

        interp = LinearlyInterpolateTS()
        trial = df.copy()
        trial["A"] = interp.fit_transform(trial["A"])
        if trial["A"].isnull().sum() > 0:
            with caplog.at_level(logging.WARNING, logger=logger_name):
                apply_imputation(
                    df.copy(),
                    _cfg("linear", cols=["A"]),
                    logging.getLogger(logger_name),
                )
            warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
            assert warnings

    def test_no_warning_when_all_filled(self, df_gap, caplog):
        logger_name = "test_sf2safe_apply_imputation"
        with caplog.at_level(logging.WARNING, logger=logger_name):
            apply_imputation(
                df_gap.copy(), _cfg("linear"), logging.getLogger(logger_name)
            )
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert not warnings
