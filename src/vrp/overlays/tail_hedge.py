"""Overlay 3 — tail-hedge spend.

For each monthly option cycle in the underlying put-writing strategy,
spend `hedge_spend_pct` of the premium collected on a long 5-delta
1-month SPX put. The hedge is marked daily via Black-Scholes (same IV
proxy as the strategy).

Net daily return = strategy_daily_return + hedge_daily_return, where
both are expressed on the same capital base (K_short of the strategy's
short put). The hedge's day-0 debit shows up as a small negative
contribution on the cycle's opening bar.

Spec parameters:
    hedge_delta      = -0.05  (5-delta)
    hedge_spend_pct  = 0.15   (15% of collected premium)
    maturity_days    = 30
"""
from __future__ import annotations

from typing import Dict

import pandas as pd

from vrp.util.bs import bs_price, strike_from_delta


def add_tail_hedge(strategy_result: Dict[str, object],
                   spx: pd.Series, vix: pd.Series,
                   hedge_delta: float = -0.05,
                   hedge_spend_pct: float = 0.15,
                   maturity_days: int = 30,
                   r: float = 0.0) -> Dict[str, object]:
    """Return a dict with net_daily_return, hedge_daily_return, hedge_legs."""
    if not (-1.0 < hedge_delta < 0.0):
        raise ValueError(f"hedge_delta must be in (-1, 0), got {hedge_delta}")

    positions = strategy_result["positions"]
    daily = strategy_result["daily_return"]

    aligned = pd.concat([spx.rename("S"), vix.rename("vix")],
                         axis=1).dropna()
    hedge_daily = pd.Series(0.0, index=aligned.index)
    hedge_legs = []

    for _, row in positions.iterrows():
        open_date = row["open_date"]
        close_date = row["close_date"]
        S0 = float(row["S0"])
        sigma0 = float(row["sigma0"])
        K_short = float(row["K_short"])
        premium_collected = float(row["premium_collected"])
        T0 = maturity_days / 365.0

        hedge_spend = max(0.0, hedge_spend_pct * premium_collected)
        if hedge_spend <= 0:
            continue

        K_hedge = strike_from_delta(S0, T0, sigma0, r, "put", hedge_delta)
        p_hedge_open = bs_price(S0, K_hedge, T0, sigma0, r, "put")
        if p_hedge_open <= 0:
            continue
        hedge_qty = hedge_spend / p_hedge_open

        cycle_idx = aligned.index[(aligned.index >= open_date)
                                    & (aligned.index < close_date)]
        if len(cycle_idx) < 2:
            continue

        marks = []
        for d in cycle_idx:
            dte = max(maturity_days - (d - open_date).days, 0)
            T = dte / 365.0
            sigma = float(aligned.loc[d, "vix"]) / 100.0
            S = float(aligned.loc[d, "S"])
            if T <= 0 or sigma <= 0:
                marks.append(max(K_hedge - S, 0.0))
            else:
                marks.append(bs_price(S, K_hedge, T, sigma, r, "put"))
        marks_s = pd.Series(marks, index=cycle_idx)

        pos_value = hedge_qty * (marks_s - p_hedge_open)
        daily_hedge_return = pos_value.diff().fillna(pos_value.iloc[0]) / K_short
        daily_hedge_return.iloc[0] -= hedge_spend / K_short

        hedge_daily.loc[cycle_idx] = (hedge_daily.loc[cycle_idx].values
                                         + daily_hedge_return.values)

        hedge_legs.append({
            "open_date": open_date,
            "K_hedge": K_hedge,
            "p_hedge_open": p_hedge_open,
            "hedge_qty": hedge_qty,
            "hedge_spend": hedge_spend,
        })

    net = daily.add(hedge_daily, fill_value=0.0).rename("net_daily_return")

    return {
        "net_daily_return": net,
        "hedge_daily_return": hedge_daily.rename("hedge_daily_return"),
        "hedge_legs": pd.DataFrame(hedge_legs),
    }
