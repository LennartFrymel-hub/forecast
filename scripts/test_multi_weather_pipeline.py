from pathlib import Path
from pprint import pprint

from spotforecast2_safe.data.fetch_data import fetch_data, get_package_data_home
from spotforecast2_safe.processing.n2n_predict_with_covariates import (
    n2n_predict_with_covariates,
)


DE_WEATHER_LOCATIONS = [
    {"name": "hamburg", "latitude": 53.5511, "longitude": 9.9937, "weight": 1.0},
    {"name": "berlin", "latitude": 52.5200, "longitude": 13.4050, "weight": 1.0},
    {"name": "koeln", "latitude": 50.9375, "longitude": 6.9603, "weight": 1.0},
    {"name": "frankfurt", "latitude": 50.1109, "longitude": 8.6821, "weight": 1.0},
    {"name": "muenchen", "latitude": 48.1372, "longitude": 11.5756, "weight": 1.0},
]


def main() -> None:
    data = fetch_data(filename=get_package_data_home() / "demo10.csv")

    predictions, forecasters, feature_info = n2n_predict_with_covariates(
        data=data,
        forecast_horizon=3,
        contamination=0.01,
        window_size=72,
        lags=24,
        train_ratio=0.8,

        latitude=51.1657,
        longitude=10.4515,
        timezone="UTC",
        country_code="DE",
        state="NW",

        weather_locations=DE_WEATHER_LOCATIONS,
        add_weighted_weather_average=True,
        keep_regional_weather_features=False,
        weather_cache_home=Path("weather_cache") / "test_multi",

        include_weather_windows=True,
        include_holiday_features=False,
        include_poly_features=False,

        force_train=True,
        model_dir=Path("models_cache") / "test_multi_weather",

        verbose=True,
        show_progress=True,
    )

    print()
    print("Predictions:")
    print(predictions.head())

    print()
    print("Prediction shape:")
    print(predictions.shape)

    print()
    print("Forecaster targets:")
    print(list(forecasters.keys()))

    print()
    print("Feature info:")
    pprint(feature_info)


if __name__ == "__main__":
    main()