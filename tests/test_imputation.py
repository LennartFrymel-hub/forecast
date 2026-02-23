import numpy as np
import pandas as pd
import pytest

from spotforecast2_safe.preprocessing import imputation as imputation_module


def _sample_data():
    index = pd.date_range("2024-01-01", periods=5, freq="h")
    return pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0]}, index=index)


def test_get_missing_weights_empty_raises():
    empty = pd.DataFrame({"a": []})
    with pytest.raises(ValueError, match="Input data is empty"):
        imputation_module.get_missing_weights(empty)


def test_get_missing_weights_invalid_window_size():
    data = _sample_data()
    with pytest.raises(ValueError, match="window_size must be a positive integer"):
        imputation_module.get_missing_weights(data, window_size=0)
    with pytest.raises(ValueError, match="window_size must be smaller"):
        imputation_module.get_missing_weights(data, window_size=len(data))


def test_get_missing_weights_fills_and_returns_series(capsys):
    data = _sample_data()

    filled, missing_weights = imputation_module.get_missing_weights(
        data, window_size=2, verbose=True
    )

    captured = capsys.readouterr().out

    assert "Number of rows with missing values" in captured
    assert filled.isna().sum().sum() == 0
    assert missing_weights.index.equals(data.index)
    assert np.issubdtype(missing_weights.dtype, np.floating)


def test_custom_weights_returns_values_for_index():
    index = pd.date_range("2024-01-01", periods=3, freq="h")
    weights = pd.Series([0.0, 1.0, 1.0], index=index)

    result = imputation_module.custom_weights(pd.Index([index[0]]), weights)

    assert isinstance(result, np.ndarray)
    assert result.tolist() == [0.0]


def test_custom_weights_raises_for_missing_index():
    index = pd.date_range("2024-01-01", periods=2, freq="h")
    weights = pd.Series([1.0, 1.0], index=index)

    with pytest.raises(ValueError, match="Index not found in weights_series"):
        imputation_module.custom_weights(
            pd.Index([pd.Timestamp("2024-01-10")]), weights
        )
