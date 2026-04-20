"""Sanity check: Strategy A cumulative PnL should correlate positively with
the negative of VXX cumulative return in contango-heavy periods.

Strategy A is dollar-neutral short-front / long-second, while VXX holds a
~30-day rolling blend — so the relationship is directional (both profit
when the term structure stays in contango and vol is quiet) but not
identical. We expect a daily-return correlation between 0.3 and 0.7. If it
is negative or below 0.2, the roll logic or PnL sign in Strategy A is
probably wrong; investigate before moving on.

Because VXX has reverse splits and a 2019 issuance change, pre-2018
divergence is expected. Shape-level agreement across both windows is the
real signal.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from vrp.data.vx_futures import load_vx_continuous
from vrp.strategies.strategy_a import run_strategy_a


def main() -> None:
    vx = load_vx_continuous(start="2013-01-01", end="2024-12-31")
    result = run_strategy_a(vx)
    strat_equity = (1 + result["daily_return"]).cumprod()

    vxx = yf.download("VXX", start="2013-01-01", end="2024-12-31",
                      auto_adjust=True, progress=False)["Close"].dropna()
    if hasattr(vxx, "columns"):
        vxx = vxx.squeeze()
    neg_vxx_equity = (vxx.iloc[0] / vxx)

    common = strat_equity.index.intersection(neg_vxx_equity.index)
    df = pd.DataFrame({
        "strategy_a": strat_equity.loc[common],
        "neg_vxx": neg_vxx_equity.loc[common],
    })
    corr = df.pct_change().dropna().corr().iloc[0, 1]

    out_dir = Path(__file__).resolve().parent.parent / "reports" / "strategy_a"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    (df / df.iloc[0]).plot(ax=ax)
    ax.set_title(
        f"Strategy A vs 1/VXX (normalized). Daily-return corr = {corr:.2f}"
    )
    fig.savefig(out_dir / "sanity_vxx.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Daily return correlation Strategy A vs -VXX: {corr:.3f}")
    print("Expected: 0.3-0.7. If |corr| < 0.2 or negative, investigate.")


if __name__ == "__main__":
    main()
