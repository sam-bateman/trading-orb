"""SPX daily price loader via yfinance.

Returns a DataFrame indexed by trading date with columns
['open', 'high', 'low', 'close', 'volume']. We use the total-return index
when appropriate (downstream code uses 'close' for realized-vol, which is
correct for price — do NOT use a TR index here).
"""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from . import cache

_KEY_PREFIX = "spx_daily"


def load_spx(start: str = "2005-01-01", end: str | None = None,
             use_cache: bool = True) -> pd.DataFrame:
    key = f"{_KEY_PREFIX}__{start}__{end or 'live'}"
    if use_cache:
        cached = cache.load(key)
        if cached is not None:
            return cached

    raw = yf.download("^GSPC", start=start, end=end, auto_adjust=False,
                      progress=False, group_by="column")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw[["Open", "High", "Low", "Close", "Volume"]].rename(
        columns=str.lower
    )
    df.index.name = "date"
    df = df.dropna()
    cache.save(key, df)
    return df
