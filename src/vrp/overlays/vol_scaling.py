"""Overlay 2 — realized-vol position scaling.

At each bar, scale position size by target_vol / realized_vol_trailing_N.
Cap leverage at 1.0 (no upsizing above baseline). Realized vol is
computed from the strategy's own daily returns (not the underlying).

Matches the project-spec overlay: target_vol = 10% annualized,
window = 20 trading days, leverage cap 1.0.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vrp.util.annualize import TRADING_DAYS


def target_vol_scale(daily_return: pd.Series,
                     target_vol: float = 0.10,
                     window: int = 20,
                     leverage_cap: float = 1.0) -> pd.Series:
    """Return target-vol-scaled daily return series.

    Scale factor at time t = min(leverage_cap, target_vol / rv_{t-1}),
    using the previous bar's realized vol (no lookahead).
    """
    r = daily_return.fillna(0.0)
    rv = r.rolling(window=window, min_periods=window).std(ddof=0) * np.sqrt(TRADING_DAYS)
    rv = rv.shift(1)
    raw_scale = target_vol / rv.replace(0.0, np.nan)
    scale = raw_scale.clip(upper=leverage_cap).fillna(leverage_cap)
    return (r * scale).rename(daily_return.name)
