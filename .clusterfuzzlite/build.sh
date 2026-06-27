#!/bin/bash
# SPDX-FileCopyrightText: 2026 Lennart Frymel
# SPDX-License-Identifier: AGPL-3.0-or-later

set -euo pipefail

cd "${SRC}/forecast"

python -m pip install --upgrade pip
python -m pip install pandas atheris==2.3.0

compile_python_fuzzer fuzz/fuzz_forecast_parsing.py