"""Annualization helpers.

All functions assume daily return series. Annualization uses 252 trading
days (US equities convention). See: Israelov & Nielsen (2015) for why this
matters when comparing to CBOE index benchmarks (they use 252 as well).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252


def ann_return(daily_returns: pd.Series) -> float:
    """Compound daily returns to an annualized figure.

    Returns (1 + r).prod() ** (252 / n) - 1 so that short series annualize
    consistently with long series.
    """
    r = daily_returns.dropna()
    if len(r) == 0:
        return float("nan")
    total = (1.0 + r).prod()
    years = len(r) / TRADING_DAYS
    return float(total ** (1.0 / years) - 1.0) if years > 0 else float("nan")


def ann_vol(daily_returns: pd.Series) -> float:
    """Annualized volatility = std(daily) * sqrt(252)."""
    r = daily_returns.dropna()
    if len(r) < 2:
        return float("nan")
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))


def sharpe_ratio(daily_returns: pd.Series, rf: float = 0.0) -> float:
    """Annualized Sharpe ratio.

    rf is interpreted as an annual rate. Excess return = ann_return - rf.
    """
    mu = ann_return(daily_returns)
    sig = ann_vol(daily_returns)
    if sig is None or sig == 0 or np.isnan(sig):
        return float("inf") if mu - rf > 0 else float("nan")
    return (mu - rf) / sig
