# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Exogenous feature engineering sub-package.

Provides public helpers for building weather, calendar, day/night, and
holiday features that can be passed as exogenous variables to
:class:`~spotforecast2_safe.forecaster.recursive.ForecasterRecursive`
and related forecasters.

Modules:
    weather: :func:`get_weather_features`
    calendar: :func:`get_calendar_features`, :func:`get_day_night_features`,
              :func:`get_holiday_features`
"""

from spotforecast2_safe.manager.exo.calendar import (
    get_calendar_features,
    get_day_night_features,
    get_holiday_features,
)
from spotforecast2_safe.manager.exo.weather import get_weather_features

__all__ = [
    "get_calendar_features",
    "get_day_night_features",
    "get_holiday_features",
    "get_weather_features",
]
