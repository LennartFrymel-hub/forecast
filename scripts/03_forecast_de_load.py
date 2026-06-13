import os
from pathlib import Path

from spotforecast2_safe.data.fetch_data import load_timeseries
from spotforecast2_safe.tasks.task_safe_n_to_1_with_covariates_and_dataframe import (
    n_to_1_with_covariates,
)

DE_WEATHER_LOCATIONS = [
    {
        "name": "hamburg",
        "latitude": 53.5511,
        "longitude": 9.9937,
        "weight": 1.0,
    },
    {
        "name": "berlin",
        "latitude": 52.5200,
        "longitude": 13.4050,
        "weight": 1.0,
    },
    {
        "name": "koeln",
        "latitude": 50.9375,
        "longitude": 6.9603,
        "weight": 1.0,
    },
    {
        "name": "frankfurt",
        "latitude": 50.1109,
        "longitude": 8.6821,
        "weight": 1.0,
    },
    {
        "name": "stuttgart",
        "latitude": 48.7758,
        "longitude": 9.1829,
        "weight": 1.0,
    },
    {
        "name": "muenchen",
        "latitude": 48.1372,
        "longitude": 11.5756,
        "weight": 1.0,
    },
    {
        "name": "leipzig",
        "latitude": 51.3397,
        "longitude": 12.3731,
        "weight": 1.0,
    },
    {
        "name": "hannover",
        "latitude": 52.3759,
        "longitude": 9.7320,
        "weight": 1.0,
    },
]

def main() -> None:
    data_home = Path(os.environ.get("SPOTFORECAST2_DATA", Path.cwd() / "data_entsoe"))
    model_dir = Path(os.environ.get("SPOTFORECAST2_CACHE", Path.cwd() / "models_cache"))

    print(f"Datenordner: {data_home}")
    print(f"Modellordner: {model_dir}")

    # 1. Deutsche Stromverbrauchsdaten laden
    actual_load = load_timeseries(
        data_home=data_home,
        on_missing="ffill_bfill",
    )

    # 2. In DataFrame umwandeln
    #    Die Forecasting-Pipeline erwartet eine oder mehrere Zielspalten.
    data = actual_load.rename("load_de").to_frame()

    print()
    print("Input-Daten:")
    print(data.head())
    print(data.tail())
    print(data.shape)

    # 3. Forecasting starten
    predictions, combined_prediction, model_metrics, feature_info = n_to_1_with_covariates(
        data=data,

        # Naechste 24 Stunden prognostizieren
        forecast_horizon=24,

        # Stromverbrauch hat starke Tages- und Wochenmuster.
        # 168 Stunden = 7 Tage Historie
        lags=168,

        # Rolling-Fenster groesser als lags
        window_size=336,

        # 85 % Training, Rest Validierung/Test innerhalb der Pipeline
        train_ratio=0.85,

        # Deutschland-Mitte als einfache Wetter-Naeherung
        latitude=51.1657,
        longitude=10.4515,


         # HIER EINFueGEN: Multi-Wetter-Konfiguration
         weather_locations=DE_WEATHER_LOCATIONS,
         add_weighted_weather_average=True,
         keep_regional_weather_features=False,
         weather_cache_home=Path("weather_cache") / "de_multi",


        # Das Projekt arbeitet intern sauber mit UTC
        timezone="UTC",

        # Feiertage Deutschland
        country_code="DE",

        # Vereinfachung: NRW.
        # Fuer Gesamtdeutschland spaeter besser differenzierter modellieren.
        state="NW",

        # Exogene Features
        include_weather_windows=True,
        include_holiday_features=False,
        include_poly_features=False,

        # Nur eine Zielspalte, daher Gewicht 1.0
        weights={"load_de": 1.0},

        # Modelle neu trainieren
        force_train=True,
        model_dir=model_dir / "de_load_multi_weather",

        verbose=True,
        show_progress=True,
    )

    # 4. Ergebnisse speichern
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    predictions_file = output_dir / "de_load_predictions.csv"
    combined_file = output_dir / "de_load_combined_prediction.csv"

    predictions.to_csv(predictions_file)
    combined_prediction.rename("forecast_load_de").to_csv(combined_file)

    print()
    print("Einzelprognose:")
    print(predictions.head())

    print()
    print("Kombinierte Prognose:")
    print(combined_prediction.head())

    print()
    print("Gespeichert:")
    print(predictions_file)
    print(combined_file)


if __name__ == "__main__":
    main()