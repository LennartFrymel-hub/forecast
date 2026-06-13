import os

import pandas as pd
from entsoe import EntsoePandasClient


def summarize(df: pd.DataFrame | pd.Series, code: str) -> None:
    print()
    print("=" * 80)
    print(f"Country code: {code}")
    print("=" * 80)

    if isinstance(df, pd.Series):
        print("Type: Series")
        print(df.head())
        print(df.tail())
        print("Rows:", len(df))
        print("Index:", df.index.min(), "to", df.index.max())
        print("Missing:", int(df.isna().sum()))
        return

    print("Type: DataFrame")
    print(df.head())
    print(df.tail())
    print("Rows:", len(df))
    print("Index:", df.index.min(), "to", df.index.max())
    print("Columns:", list(df.columns))
    print("Missing:")
    print(df.isna().sum())


def main() -> None:
    api_key = os.environ.get("ENTSOE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "ENTSOE_API_KEY is missing. Set it first."
        )

    client = EntsoePandasClient(api_key=api_key)

    start = pd.Timestamp("2025-01-01 00:00", tz="Europe/Berlin")
    end = pd.Timestamp("2025-01-08 00:00", tz="Europe/Berlin")

    print("Compare DE vs DE_LU")
    print(f"Start: {start}")
    print(f"End:   {end}")

    for code in ["DE", "DE_LU"]:
        try:
            result = client.query_load_and_forecast(
                country_code=code,
                start=start,
                end=end,
            )
            summarize(result, code)

        except Exception as exc:
            print()
            print("=" * 80)
            print(f"Country code: {code}")
            print("=" * 80)
            print(f"FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()