#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Lennart Frymel
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import importlib.util
import io
import os
from pathlib import Path
import sys

import pandas as pd

try:
    import atheris
except ModuleNotFoundError:  # pragma: no cover - local smoke fallback.
    atheris = None


def _load_ensure_utc_datetime_index():
    source_root = Path(os.environ.get("SRC", "/src")) / "forecast"
    module_path = (
        source_root
        / "src"
        / "spotforecast2_safe"
        / "manager"
        / "postprocessing"
        / "forecast_corrections.py"
    )

    if not module_path.exists():
        module_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "spotforecast2_safe"
            / "manager"
            / "postprocessing"
            / "forecast_corrections.py"
        )

    spec = importlib.util.spec_from_file_location("forecast_corrections", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load forecast corrections from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ensure_utc_datetime_index


ensure_utc_datetime_index = _load_ensure_utc_datetime_index()


EXPECTED_PARSE_EXCEPTIONS = (
    KeyError,
    TypeError,
    ValueError,
    OverflowError,
    pd.errors.ParserError,
)


def TestOneInput(data: bytes) -> None:
    text = data[:4096].decode("utf-8", errors="ignore")
    if not text.strip():
        return

    try:
        df = pd.read_csv(io.StringIO(text), nrows=48)
    except EXPECTED_PARSE_EXCEPTIONS:
        return

    if df.empty:
        return

    try:
        ensure_utc_datetime_index(df)
    except EXPECTED_PARSE_EXCEPTIONS:
        pass

    if {"timestamp_utc", "forecast_mw"}.issubset(df.columns):
        timestamps = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        forecasts = pd.to_numeric(df["forecast_mw"], errors="coerce")
        valid = timestamps.notna() & forecasts.notna()
        if valid.any():
            series = pd.Series(forecasts[valid].to_numpy(), index=timestamps[valid])
            _ = series.sort_index().asfreq("h")


def main() -> None:
    if atheris is None:
        TestOneInput(sys.stdin.buffer.read())
        return

    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
