# SPDX-FileCopyrightText: 2026 bartzbeielstein
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Linear interpolation transformer for time series data."""

from dataclasses import dataclass
from typing import Any, Literal, Union

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

OnMissing = Literal["raise", "ffill_bfill", "passthrough"]


@dataclass
class LinearlyInterpolateTS(BaseEstimator, TransformerMixin):
    """Transformer that applies linear interpolation to time series data.

    The transformer always runs ``y.interpolate(method="linear")`` first.
    The ``on_missing`` keyword then chooses how to handle any NaN that
    linear interpolation cannot bridge (typically the leading and
    trailing endpoints of the series).

    Args:
        on_missing: Contract for residual NaN after linear interpolation.

            - ``"raise"`` (default, fail-safe): raise ``ValueError`` if
              any NaN remains. Refuses to silently embed imputed values
              disguised as measurements.
            - ``"ffill_bfill"``: explicit opt-in to endpoint
              forward-then-backward fill, so that leading and trailing
              NaNs are both bridged.
            - ``"passthrough"``: return the linearly interpolated
              series unchanged. The caller promises to handle the
              residual NaN downstream.

    Raises:
        ValueError: If ``on_missing`` is not one of the three accepted
            values, or if ``on_missing="raise"`` (the default) and any
            NaN remains after linear interpolation.

    Examples:
        >>> import pandas as pd
        >>> import numpy as np
        >>> from spotforecast2_safe.preprocessing.linearly_interpolate_ts import LinearlyInterpolateTS

        Interior gaps are bridged by linear interpolation under every
        mode; the default ``"raise"`` then succeeds because nothing
        remains:

        >>> s = pd.Series([1.0, np.nan, 3.0])
        >>> LinearlyInterpolateTS().fit_transform(s).tolist()
        [1.0, 2.0, 3.0]

        Endpoint NaNs are bridged only when the caller opts in
        explicitly:

        >>> s = pd.Series([1.0, np.nan, 3.0, np.nan])
        >>> LinearlyInterpolateTS(on_missing="ffill_bfill").fit_transform(s).tolist()
        [1.0, 2.0, 3.0, 3.0]

        ``"passthrough"`` lets the residual NaN survive for the caller
        to handle:

        >>> out = LinearlyInterpolateTS(on_missing="passthrough").fit_transform(s)
        >>> bool(out.isna().iloc[-1])
        True
    """

    on_missing: OnMissing = "raise"

    def fit(self, X: Any, y: Any = None) -> "LinearlyInterpolateTS":
        """Fitted transformer (no-op).

        Args:
            X: Input data.
            y: Ignored.

        Returns:
            self: The fitted transformer.
        """
        return self

    def transform(
        self, X: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Transform the input data by applying linear interpolation.

        Args:
            X: Input Series or DataFrame to interpolate.

        Returns:
            Union[pd.Series, pd.DataFrame]: Interpolated data, with
            residual NaN handled according to ``self.on_missing``.
        """
        return self.apply(X)

    def apply(
        self, y: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Apply linear interpolation and dispatch on ``on_missing``.

        Args:
            y: Input Series or DataFrame.

        Returns:
            Union[pd.Series, pd.DataFrame]: The transformed data. For
            ``on_missing="ffill_bfill"`` any residual endpoint NaN has
            been forward- and back-filled; for ``"passthrough"`` it
            survives; for ``"raise"`` (the default) the method raises
            instead of returning a NaN-bearing result.

        Raises:
            ValueError: If ``self.on_missing`` is not a recognized
                value, or if ``self.on_missing == "raise"`` and any NaN
                remains after linear interpolation.
        """
        if self.on_missing not in ("raise", "ffill_bfill", "passthrough"):
            raise ValueError(
                f"on_missing must be 'raise', 'ffill_bfill', or "
                f"'passthrough'; got {self.on_missing!r}."
            )

        # limit_area="inside" prevents pandas from extrapolating
        # trailing NaN by carrying the last value forward, so endpoint
        # NaNs survive into the on_missing dispatch below.
        y_filled = y.interpolate(method="linear", limit_area="inside").astype("float")

        if self.on_missing == "passthrough":
            return y_filled

        if self.on_missing == "ffill_bfill":
            return y_filled.ffill().bfill()

        # on_missing == "raise"
        residual_mask = y_filled.isna()
        has_residual = (
            residual_mask.any().any()
            if isinstance(residual_mask, pd.DataFrame)
            else residual_mask.any()
        )
        if has_residual:
            if isinstance(y_filled, pd.DataFrame):
                gap_index = y_filled.index[residual_mask.any(axis=1)]
            else:
                gap_index = y_filled.index[residual_mask]
            preview = ", ".join(str(ts) for ts in gap_index[:5])
            more = f" (+{len(gap_index) - 5} more)" if len(gap_index) > 5 else ""
            raise ValueError(
                f"{len(gap_index)} missing value(s) remain after linear "
                f"interpolation. First gaps: [{preview}]{more}. "
                "Pass on_missing='ffill_bfill' to opt into endpoint "
                "forward/backward fill or on_missing='passthrough' to "
                "return the interpolated series with residual NaN."
            )
        return y_filled
