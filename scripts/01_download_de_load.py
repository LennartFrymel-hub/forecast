import os
from pathlib import Path

import pandas as pd

from spotforecast2_safe.downloader.entsoe import download_new_data


def main() -> None:
    api_key = os.environ.get("ENTSOE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ENTSOE_API_KEY fehlt. Setze ihn vorher in PowerShell mit:\n"
            '$env:ENTSOE_API_KEY="DEIN_TOKEN"'
        )

    data_home = Path(os.environ.get("SPOTFORECAST2_DATA", Path.cwd() / "data_entsoe"))
    os.environ["SPOTFORECAST2_DATA"] = str(data_home)

    (data_home / "raw").mkdir(parents=True, exist_ok=True)
    (data_home / "interim").mkdir(parents=True, exist_ok=True)

    start = "202201010000"
    end = pd.Timestamp.now(tz="UTC").floor("h").strftime("%Y%m%d%H00")

    print("Lade ENTSO-E Lastdaten fuer Deutschland")
    print(f"Von: {start}")
    print(f"Bis: {end}")
    print(f"Datenordner: {data_home}")

    download_new_data(
        api_key=api_key,
        country_code="DE",
        start=start,
        end=end,
        force=True,
    )

    expected_file = data_home / "interim" / "energy_load.csv"

    if not expected_file.exists():
        raise FileNotFoundError(
            f"Die erwartete Datei wurde nicht erzeugt: {expected_file}"
        )

    print()
    print("Download abgeschlossen.")
    print(f"Datei erzeugt: {expected_file}")


if __name__ == "__main__":
    main()