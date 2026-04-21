"""Overlay 1 — VIX regime filter.

Go to cash when VIX spot > 30 OR VIX term structure inverts
(front-month VX > second-month VX). Re-enter when VIX < 25 AND term
structure is back in contango, confirmed for 7 consecutive trading days
(anti-whipsaw).

Rule is hand-picked from the project spec; not tuned to data.
"""
from __future__ import annotations

import pandas as pd


HIGH_VIX = 30.0
LOW_VIX = 25.0
CONFIRMATION_DAYS = 7


def vix_regime_mask(vix_spot: pd.Series,
                    vx_front: pd.Series,
                    vx_second: pd.Series,
                    high_vix: float = HIGH_VIX,
                    low_vix: float = LOW_VIX,
                    confirm_days: int = CONFIRMATION_DAYS) -> pd.Series:
    """Daily boolean mask. True = allowed to trade, False = go to cash."""
    aligned = pd.concat([vix_spot.rename("vix"),
                          vx_front.rename("front"),
                          vx_second.rename("second")], axis=1).dropna()

    stress = (aligned["vix"] > high_vix) | (aligned["front"] > aligned["second"])
    calm = (aligned["vix"] < low_vix) & (aligned["front"] < aligned["second"])

    calm_streak = calm.astype(int).groupby(
        (~calm).cumsum()
    ).cumsum()

    mask = pd.Series(False, index=aligned.index)
    active = True
    for i, (is_stress, streak) in enumerate(zip(stress.values, calm_streak.values)):
        if active:
            if is_stress:
                active = False
        else:
            if streak >= confirm_days:
                active = True
        mask.iloc[i] = active
    return mask


def apply_mask(daily_return: pd.Series, mask: pd.Series) -> pd.Series:
    """Zero out daily_return on bars where mask is False."""
    aligned = pd.concat([daily_return.rename("r"),
                          mask.rename("m")], axis=1)
    aligned["m"] = aligned["m"].fillna(False).astype(bool)
    out = aligned["r"].where(aligned["m"], 0.0)
    return out.rename(daily_return.name)
