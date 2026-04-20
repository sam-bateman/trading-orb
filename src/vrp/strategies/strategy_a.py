"""Strategy A — VIX futures term-structure calendar (dollar-neutral).

A dollar-neutral VIX-futures calendar spread with a 5-day pre-expiry roll.
Two directions:

- ``short_front`` (the spec's direction): short front-month VX, long
  second-month VX. Captures ``r_second − r_front`` daily. Losing in
  practice — see the Phase 1 README for the analysis.
- ``long_front``: long front-month VX, short second-month VX. Captures
  ``r_front − r_second`` daily. Profitable in the Phase 1 backtest.

See:
- Whaley (2013) "Trading Volatility: At What Cost?"
- Alexander, Korovilas (2013) for a critique of naive VX calendars.

This implementation targets gross notional = $1 per side (so $2 gross).
Interpret ``daily_pnl`` as dollars on that unit capital, and
``daily_return`` as the daily return on $2 of gross capital.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal

import pandas as pd


ROLL_DAYS_BEFORE_EXPIRY = 5

Direction = Literal["short_front", "long_front"]


@dataclass
class StrategyAConfig:
    roll_days_before_expiry: int = ROLL_DAYS_BEFORE_EXPIRY
    tc_bps_per_roll: float = 1.0
    gross_notional_per_leg: float = 1.0
    direction: Direction = "short_front"


def _is_roll_day(days_to_front_expiry: pd.Series, roll_days: int) -> pd.Series:
    return days_to_front_expiry <= roll_days


def _leg_signs(direction: Direction) -> tuple[float, float]:
    """Return (front_sign, second_sign) for a given calendar direction."""
    if direction == "short_front":
        return -1.0, +1.0
    if direction == "long_front":
        return +1.0, -1.0
    raise ValueError(
        f"direction must be 'short_front' or 'long_front', got {direction!r}"
    )


def run_strategy_a(vx: pd.DataFrame,
                   roll_days_before_expiry: int = ROLL_DAYS_BEFORE_EXPIRY,
                   tc_bps_per_roll: float = 1.0,
                   gross_notional_per_leg: float = 1.0,
                   direction: Direction = "short_front") -> Dict[str, pd.Series]:
    """Run Strategy A on a continuous VX front/second DataFrame.

    Parameters
    ----------
    vx:
        Index = date. Columns include ``front_settle``, ``second_settle``,
        ``front_expiry``, ``days_to_front_expiry``.
    roll_days_before_expiry:
        Flag bars within this many days of front expiry as roll days;
        transaction costs are applied on those bars.
    tc_bps_per_roll:
        Basis points per leg on roll days. 1 bp is the baseline estimate.
    gross_notional_per_leg:
        Dollar notional per leg. Gross capital is ``2 * this`` by
        construction (dollar-neutral).
    direction:
        ``"short_front"`` (default, spec's direction) or ``"long_front"``.

    Returns
    -------
    Dict with keys ``daily_pnl``, ``daily_return``, ``positions``.
    """
    cfg = StrategyAConfig(roll_days_before_expiry=roll_days_before_expiry,
                          tc_bps_per_roll=tc_bps_per_roll,
                          gross_notional_per_leg=gross_notional_per_leg,
                          direction=direction)

    df = vx.copy().sort_index()
    front_qty = cfg.gross_notional_per_leg / df["front_settle"]
    second_qty = cfg.gross_notional_per_leg / df["second_settle"]

    front_sign, second_sign = _leg_signs(cfg.direction)

    # Position taken at close of day t earns PnL from t -> t+1. diff().shift(-1)
    # aligns (settle_{t+1} - settle_t) to t; after PnL is computed at t it is
    # shifted by +1 to land on the date the PnL is recognized (t+1).
    d_front = df["front_settle"].diff().shift(-1)
    d_second = df["second_settle"].diff().shift(-1)
    daily_pnl = (front_sign * front_qty * d_front) + (second_sign * second_qty * d_second)
    daily_pnl = daily_pnl.shift(1).fillna(0.0)

    roll_mask = _is_roll_day(df["days_to_front_expiry"], cfg.roll_days_before_expiry)
    tc = roll_mask.astype(float) * (cfg.tc_bps_per_roll * 1e-4) * (
        2 * cfg.gross_notional_per_leg
    )
    daily_pnl = daily_pnl - tc

    gross = 2.0 * cfg.gross_notional_per_leg
    daily_return = daily_pnl / gross

    positions = pd.DataFrame({
        "front_qty": front_sign * front_qty,
        "second_qty": second_sign * second_qty,
        "is_roll_day": roll_mask.astype(bool),
    }, index=df.index)

    return {
        "daily_pnl": daily_pnl,
        "daily_return": daily_return,
        "positions": positions,
    }
