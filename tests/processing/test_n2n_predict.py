from unittest.mock import MagicMock, patch

import pandas as pd

from spotforecast2_safe.processing.n2n_predict import n2n_predict


@patch(
    "spotforecast2_safe.processing.n2n_predict._model_directory_exists",
    return_value=False,
)
@patch("spotforecast2_safe.manager.persistence.dump")
@patch("spotforecast2_safe.manager.persistence.load")
@patch("spotforecast2_safe.processing.n2n_predict.fetch_data")
@patch("spotforecast2_safe.processing.n2n_predict.get_start_end")
@patch("spotforecast2_safe.processing.n2n_predict.basic_ts_checks")
@patch("spotforecast2_safe.processing.n2n_predict.agg_and_resample_data")
@patch("spotforecast2_safe.processing.n2n_predict.mark_outliers")
@patch("spotforecast2_safe.processing.n2n_predict.split_rel_train_val_test")
@patch("spotforecast2_safe.processing.n2n_predict.ForecasterEquivalentDate")
@patch("spotforecast2_safe.processing.n2n_predict.predict_multivariate")
def test_n2n_predict_flow(
    mock_predict_multivariate,
    mock_forecaster_class,
    mock_split,
    mock_mark_outliers,
    mock_agg,
    mock_checks,
    mock_start_end,
    mock_fetch_data,
    mock_load,
    mock_dump,
    mock_dir_exists,
):
    # --- Setup Mocks ---

    # Mock Data
    mock_df = pd.DataFrame(
        {"col_1": [1, 2, 3], "col_2": [4, 5, 6]},
        index=pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
    )

    mock_fetch_data.return_value = mock_df
    mock_start_end.return_value = ("2023-01-01", "2023-01-03", None, None)
    mock_agg.return_value = mock_df
    mock_mark_outliers.return_value = (mock_df, None)

    # Mock Split
    mock_split.return_value = (mock_df, mock_df, mock_df)

    # Mock Forecaster
    mock_forecaster_instance = MagicMock()
    mock_forecaster_class.return_value = mock_forecaster_instance

    # Mock Prediction
    mock_predictions = pd.DataFrame(
        {"col_1": [10, 11], "col_2": [12, 13]},
        index=pd.to_datetime(["2023-01-04", "2023-01-05"]),
    )
    mock_predict_multivariate.return_value = mock_predictions

    # --- Run Function ---
    columns = ["col_1", "col_2"]
    predictions, forecasters = n2n_predict(
        columns=columns, forecast_horizon=2, verbose=False
    )

    # --- Assertions ---

    # Verify fetch_data called with correct columns
    mock_fetch_data.assert_called_once_with(filename="data_in.csv", columns=columns)

    # Verify preprocessing steps called
    mock_start_end.assert_called_once()
    mock_checks.assert_called_once()
    mock_agg.assert_called_once()
    mock_mark_outliers.assert_called_once()

    # Verify split called
    mock_split.assert_called_once()

    # Verify forecaster fit called for each column
    assert mock_forecaster_class.call_count == 2
    assert mock_forecaster_instance.fit.call_count == 2

    # Verify predict called
    mock_predict_multivariate.assert_called_once()

    # Verify output
    pd.testing.assert_frame_equal(predictions, mock_predictions)
    assert len(forecasters) == 2  # Should have forecasters for both columns


@patch(
    "spotforecast2_safe.processing.n2n_predict._model_directory_exists",
    return_value=False,
)
@patch("spotforecast2_safe.manager.persistence.dump")
@patch("spotforecast2_safe.manager.persistence.load")
@patch("spotforecast2_safe.processing.n2n_predict.fetch_data")
@patch("spotforecast2_safe.processing.n2n_predict.get_start_end")
@patch("spotforecast2_safe.processing.n2n_predict.basic_ts_checks")
@patch("spotforecast2_safe.processing.n2n_predict.agg_and_resample_data")
@patch("spotforecast2_safe.processing.n2n_predict.mark_outliers")
@patch("spotforecast2_safe.processing.n2n_predict.split_rel_train_val_test")
@patch("spotforecast2_safe.processing.n2n_predict.ForecasterEquivalentDate")
@patch("spotforecast2_safe.processing.n2n_predict.predict_multivariate")
def test_n2n_predict_combined_calculation(
    mock_predict_multivariate,
    mock_forecaster_class,
    mock_split,
    mock_mark_outliers,
    mock_agg,
    mock_checks,
    mock_start_end,
    mock_fetch_data,
    mock_load,
    mock_dump,
    mock_dir_exists,
):
    # Test that combined prediction is calculated if all required columns are present in output

    # Mock Data (doesn't matter much for this test as we mock output of predict_multivariate)
    mock_df = pd.DataFrame({"col": [1]}, index=pd.to_datetime(["2023-01-01"]))
    mock_fetch_data.return_value = mock_df
    mock_start_end.return_value = ("2023-01-01", "2023-01-01", None, None)
    mock_agg.return_value = mock_df
    mock_mark_outliers.return_value = (mock_df, None)
    mock_split.return_value = (mock_df, mock_df, mock_df)

    # Mock Predictions with ALL required columns
    required_cols = [f"col_{i}" for i in range(5)]

    mock_pred_data = {col: [10.0] for col in required_cols}
    mock_predictions = pd.DataFrame(
        mock_pred_data, index=pd.to_datetime(["2023-01-02"])
    )
    mock_predict_multivariate.return_value = mock_predictions

    # Run
    predictions, forecasters = n2n_predict(columns=["dummy"], verbose=False)

    # Check that predictions are returned correctly (combined not calculated here anymore)
    assert predictions.equals(mock_predictions)
    assert isinstance(forecasters, dict)  # Should return forecasters dictionary
