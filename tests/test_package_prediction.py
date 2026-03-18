# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Comprehensive pytests for package_prediction method in ForecasterRecursiveModel.

Covers both operating modes:
- Back-test mode:        end_dev well before last timestamp; y_test non-empty;
                         future metrics are computed.
- Genuine-future mode:   end_dev at/beyond last timestamp; y_test too short;
                         predict_hours steps forecast into the true future;
                         future metrics are absent.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.manager.models.forecaster_recursive_model import (
    ForecasterRecursiveModel,
)


@pytest.fixture
def tmp_data_extra(monkeypatch):
    """Setup a temporary data environment for package_prediction tests."""
    tmp_dir = tempfile.mkdtemp()
    monkeypatch.setenv("SPOTFORECAST2_DATA", tmp_dir)

    interim_dir = Path(tmp_dir) / "interim"
    interim_dir.mkdir(parents=True)

    yield interim_dir

    shutil.rmtree(tmp_dir)


def create_mock_data(path: Path, columns=None):
    """Create a mock energy_load.csv file.

    Uses column names expected by ``load_timeseries`` / ``load_timeseries_forecast``.
    """
    if columns is None:
        columns = ["Actual Load", "Forecasted Load"]

    dates = pd.date_range("2022-01-01", periods=100, freq="h")
    df = pd.DataFrame(np.random.rand(100, len(columns)), index=dates, columns=columns)
    df.index.name = "Time (UTC)"
    df.to_csv(path / "energy_load.csv")
    return df


def test_package_prediction_success(tmp_data_extra):
    """Verify package_prediction returns expected structure on success."""
    create_mock_data(tmp_data_extra)

    model = ForecasterRecursiveModel(iteration=0, end_dev="2022-01-03 00:00+00:00")
    model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    result = model.package_prediction(predict_size=2)

    assert isinstance(result, dict)
    assert "train_actual" in result
    assert "future_actual" in result
    assert "train_pred" in result
    assert "future_pred" in result
    assert "metrics_train" in result
    assert "metrics_future" in result
    assert "metrics_future_one_day" in result
    assert "future_forecast" in result
    assert "metrics_forecast" in result


def test_package_prediction_no_forecast_column(tmp_data_extra):
    """Verify package_prediction works even if benchmark forecast is missing."""
    create_mock_data(tmp_data_extra, columns=["Actual Load"])

    model = ForecasterRecursiveModel(iteration=0, end_dev="2022-01-03 00:00+00:00")
    model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    result = model.package_prediction(predict_size=2)

    assert "train_actual" in result
    assert "future_forecast" not in result
    assert "metrics_forecast" not in result


def test_package_prediction_missing_actual_load(tmp_data_extra, caplog):
    """Verify package_prediction returns {} and logs error if Actual Load is missing."""
    create_mock_data(tmp_data_extra, columns=["Wrong Column"])

    model = ForecasterRecursiveModel(iteration=0)
    model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    with caplog.at_level("ERROR"):
        result = model.package_prediction()

    assert result == {}
    # load_timeseries raises KeyError which is caught by package_prediction's except
    assert "Error generating prediction package" in caplog.text


def test_package_prediction_custom_predict_size(tmp_data_extra):
    """Verify that predict_size override is respected."""
    create_mock_data(tmp_data_extra)

    # Default predict_size is 24, refit_size is 7 -> 168 hours
    model = ForecasterRecursiveModel(
        iteration=0, end_dev="2022-01-03 00:00+00:00", predict_size=24, refit_size=7
    )
    model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    # Override with predict_size=1 -> 7 hours
    result = model.package_prediction(predict_size=1)

    assert "future_actual" in result
    assert len(result["future_actual"]) == 7
    assert len(result["future_pred"]) == 7


def test_package_prediction_exception_handling(tmp_data_extra, monkeypatch, caplog):
    """Verify that exceptions during processing return {} and log the error."""
    create_mock_data(tmp_data_extra)

    model = ForecasterRecursiveModel(iteration=0)
    model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)

    # Force an exception during load_timeseries by removing the data file
    import os

    os.remove(tmp_data_extra / "energy_load.csv")

    with caplog.at_level("ERROR"):
        result = model.package_prediction()

    assert result == {}
    assert "Error generating prediction package" in caplog.text


# ---------------------------------------------------------------------------
# Genuine-future mode tests
# End_dev is set to the *last* timestamp in the dataset so y_test is empty.
# Expected behaviour: predict_hours steps into the true future, no metrics_future.
# ---------------------------------------------------------------------------

PREDICT_HOURS = 24  # predict_size=24, refit_size=1


def _make_genuine_future_model(tmp_data_extra, predict_size=PREDICT_HOURS):
    """Return a model whose end_dev equals the last timestamp in the mock data."""
    df = create_mock_data(tmp_data_extra, columns=["Actual Load"])
    last_ts = df.index[-1]  # UTC-aware because create_mock_data uses date_range
    model = ForecasterRecursiveModel(
        iteration=0,
        end_dev=last_ts,
        predict_size=predict_size,
        refit_size=1,
        name="test_gf",
    )
    model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    return model


def test_genuine_future_required_keys(tmp_data_extra):
    """Genuine-future mode must return the five core keys."""
    model = _make_genuine_future_model(tmp_data_extra)
    result = model.package_prediction(predict_size=PREDICT_HOURS)
    for key in (
        "train_actual",
        "train_pred",
        "future_pred",
        "future_actual",
        "metrics_train",
    ):
        assert key in result, f"Missing key: '{key}'"


def test_genuine_future_no_future_metric_keys(tmp_data_extra):
    """metrics_future and metrics_future_one_day must be absent in genuine-future mode."""
    model = _make_genuine_future_model(tmp_data_extra)
    result = model.package_prediction(predict_size=PREDICT_HOURS)
    assert "metrics_future" not in result
    assert "metrics_future_one_day" not in result


def test_genuine_future_pred_length(tmp_data_extra):
    """future_pred must contain exactly predict_hours steps."""
    model = _make_genuine_future_model(tmp_data_extra)
    result = model.package_prediction(predict_size=PREDICT_HOURS)
    assert len(result["future_pred"]) == PREDICT_HOURS


def test_genuine_future_actual_is_empty_series(tmp_data_extra):
    """future_actual must be an empty Series (no ground truth available)."""
    model = _make_genuine_future_model(tmp_data_extra)
    result = model.package_prediction(predict_size=PREDICT_HOURS)
    fa = result["future_actual"]
    assert isinstance(fa, pd.Series)
    assert len(fa) == 0


def test_genuine_future_pred_index_after_end_dev(tmp_data_extra):
    """All future_pred timestamps must be strictly after end_dev."""
    df = create_mock_data(tmp_data_extra, columns=["Actual Load"])
    last_ts = df.index[-1]
    model = ForecasterRecursiveModel(
        iteration=0, end_dev=last_ts, predict_size=PREDICT_HOURS, refit_size=1
    )
    model.forecaster = ForecasterRecursive(estimator=LinearRegression(), lags=3)
    result = model.package_prediction(predict_size=PREDICT_HOURS)
    # load_timeseries() produces UTC-aware timestamps; align last_ts for comparison
    last_ts_utc = last_ts if last_ts.tzinfo is not None else last_ts.tz_localize("UTC")
    assert result["future_pred"].index.min() > last_ts_utc


def test_genuine_future_metrics_train_present(tmp_data_extra):
    """metrics_train (in-sample) must still be computed in genuine-future mode."""
    model = _make_genuine_future_model(tmp_data_extra)
    result = model.package_prediction(predict_size=PREDICT_HOURS)
    mt = result["metrics_train"]
    assert "mae" in mt and "mape" in mt
    assert mt["mae"] >= 0 and mt["mape"] >= 0


def test_genuine_future_custom_predict_size(tmp_data_extra):
    """predict_size argument must control future_pred length in genuine-future mode."""
    model = _make_genuine_future_model(tmp_data_extra, predict_size=48)
    result = model.package_prediction(predict_size=48)
    assert len(result["future_pred"]) == 48


def test_genuine_future_no_benchmark_key(tmp_data_extra):
    """future_forecast benchmark key must be absent (no y_test to align against)."""
    model = _make_genuine_future_model(tmp_data_extra)
    result = model.package_prediction(predict_size=PREDICT_HOURS)
    assert "future_forecast" not in result
    assert "metrics_forecast" not in result
