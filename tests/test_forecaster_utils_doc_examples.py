import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from spotforecast2_safe.forecaster.recursive import ForecasterRecursive
from spotforecast2_safe.forecaster.utils import (
    check_extract_values_and_index,
    check_preprocess_series,
    check_residuals_input,
    date_to_index_position,
    exog_to_direct_numpy,
    get_style_repr_html,
    initialize_estimator,
    initialize_transformer_series,
    initialize_window_features,
    predict_multivariate,
    prepare_steps_direct,
    transform_numpy,
)


def test_check_preprocess_series_examples():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")

    # Wide-format DataFrame
    df_wide = pd.DataFrame(
        {
            "series_1": [1, 2, 3, 4, 5],
            "series_2": [5, 4, 3, 2, 1],
        },
        index=dates,
    )
    with pytest.warns(UserWarning):
        series_dict, series_indexes = check_preprocess_series(df_wide)
    assert "series_1" in series_dict
    assert len(series_dict["series_1"]) == 5
    assert series_indexes["series_1"].freq == "D"

    # Long-format DataFrame
    df_long = pd.DataFrame(
        {
            "series_id": ["series_1"] * 5 + ["series_2"] * 5,
            "value": [1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
        },
        index=pd.MultiIndex.from_product(
            [["series_1", "series_2"], dates], names=["series_id", "date"]
        ),
    )
    with pytest.warns(UserWarning):
        series_dict, series_indexes = check_preprocess_series(df_long)
    assert "series_1" in series_dict
    assert len(series_dict["series_1"]) == 5

    # Dictionary of Series
    series_dict_input = {
        "series_1": pd.Series([1, 2, 3, 4, 5], index=dates),
        "series_2": pd.Series([5, 4, 3, 2, 1], index=dates),
    }
    series_dict, series_indexes = check_preprocess_series(series_dict_input)
    assert "series_1" in series_dict
    assert "series_2" in series_dict

    # Dictionary of DataFrames
    df_series_1 = pd.DataFrame({"value": [1, 2, 3, 4, 5]}, index=dates)
    df_series_2 = pd.DataFrame({"value": [5, 4, 3, 2, 1]}, index=dates)
    series_dict_input = {
        "series_1": df_series_1,
        "series_2": df_series_2,
    }
    series_dict, series_indexes = check_preprocess_series(series_dict_input)
    assert "series_1" in series_dict
    assert isinstance(series_dict["series_1"], pd.Series)


def test_exog_to_direct_numpy_example():
    exog = np.array([10, 20, 30, 40, 50])
    steps = 3
    exog_direct, exog_direct_names = exog_to_direct_numpy(exog, steps)
    expected = np.array([[10, 20, 30], [20, 30, 40], [30, 40, 50]])
    assert np.array_equal(exog_direct, expected)
    assert exog_direct_names is None


def test_prepare_steps_direct_examples():
    max_step = 5
    # Case 1: int steps
    assert prepare_steps_direct(max_step, 3) == [1, 2, 3]
    # Case 2: list steps
    assert prepare_steps_direct(max_step, [1, 3, 5]) == [1, 3, 5]
    # Case 3: None steps
    assert prepare_steps_direct(max_step, None) == [1, 2, 3, 4, 5]


def test_transform_numpy_examples():
    array = np.array([1, 2, 3, 4, 5])
    transformer = StandardScaler()
    array_transformed = transform_numpy(array, transformer, fit=True)
    assert len(array_transformed) == 5

    array_inversed = transform_numpy(
        array_transformed, transformer, inverse_transform=True
    )
    assert np.allclose(array_inversed, array)


def test_initialize_window_features_example():
    class MockWF:
        def __init__(self, names, sizes):
            self.features_names = names
            self.window_sizes = sizes

        def transform_batch(self, X):
            pass

        def transform(self, X):
            pass

    wf1 = MockWF(["f1"], 7)
    wf2 = MockWF(["f2", "f3"], 3)

    wf_list, names, max_size = initialize_window_features([wf1, wf2])
    assert len(wf_list) == 2
    assert names == ["f1", "f2", "f3"]
    assert max_size == 7


def test_check_extract_values_and_index_example():
    y = pd.Series([1, 2, 3], index=pd.date_range("2020-01-01", periods=3, freq="D"))
    values, index = check_extract_values_and_index(y)
    assert np.array_equal(values, np.array([1, 2, 3]))
    assert isinstance(index, pd.DatetimeIndex)


def test_get_style_repr_html_example():
    style, uid = get_style_repr_html(is_fitted=True)
    assert isinstance(style, str)
    assert isinstance(uid, str)
    html = f"{style}<div class='container-{uid}'>Forecaster Info</div>"
    assert "background-color" in html


def test_check_residuals_input_example():
    forecaster_name = "ForecasterRecursiveMultiSeries"
    use_in_sample_residuals = True
    in_sample_residuals_ = {
        "series_1": np.array([0.1, -0.2]),
        "series_2": np.array([0.3, -0.1]),
    }
    out_sample_residuals_ = None
    use_binned_residuals = False
    # Should not raise exception
    check_residuals_input(
        forecaster_name,
        use_in_sample_residuals,
        in_sample_residuals_,
        out_sample_residuals_,
        use_binned_residuals,
        in_sample_residuals_by_bin_=None,
        out_sample_residuals_by_bin_=None,
        levels=["series_1", "series_2"],
        encoding="onehot",
    )


def test_date_to_index_position_examples():
    index = pd.date_range(start="2020-01-01", periods=10, freq="D")
    # Integer input
    assert date_to_index_position(index, 5) == 5
    # Date input for prediction
    assert date_to_index_position(index, "2020-01-15", method="prediction") == 5
    # Date input for validation
    assert date_to_index_position(index, "2020-01-05", method="validation") == 5


def test_initialize_estimator_examples():
    estimator = LinearRegression()
    # Using estimator
    assert initialize_estimator(estimator=estimator) == estimator
    # Using deprecated regressor
    with pytest.warns(FutureWarning):
        assert initialize_estimator(regressor=estimator) == estimator


def test_predict_multivariate_example():
    y1 = pd.Series([1, 2, 3, 4, 5])
    y2 = pd.Series([2, 4, 6, 8, 10])
    f1 = ForecasterRecursive(estimator=LinearRegression(), lags=2)
    f2 = ForecasterRecursive(estimator=LinearRegression(), lags=2)
    f1.fit(y=y1)
    f2.fit(y=y2)
    forecasters = {"target1": f1, "target2": f2}
    predictions = predict_multivariate(forecasters, steps_ahead=2)
    assert isinstance(predictions, pd.DataFrame)
    assert list(predictions.columns) == ["target1", "target2"]
    assert len(predictions) == 2


def test_initialize_transformer_series_examples():
    series = ["series1", "series2", "series3"]
    # No transformation
    result = initialize_transformer_series(
        forecaster_name="ForecasterDirectMultiVariate",
        series_names_in_=series,
        transformer_series=None,
    )
    assert result == {"series1": None, "series2": None, "series3": None}

    # Same transformer
    scaler = StandardScaler()
    result = initialize_transformer_series(
        forecaster_name="ForecasterDirectMultiVariate",
        series_names_in_=["series1", "series2"],
        transformer_series=scaler,
    )
    assert len(result) == 2
    assert isinstance(result["series1"], StandardScaler)
    assert result["series1"] is not result["series2"]

    # Different transformers
    transformers = {"series1": StandardScaler(), "series2": MinMaxScaler()}
    result = initialize_transformer_series(
        forecaster_name="ForecasterDirectMultiVariate",
        series_names_in_=["series1", "series2"],
        transformer_series=transformers,
    )
    assert isinstance(result["series1"], StandardScaler)
    assert isinstance(result["series2"], MinMaxScaler)
