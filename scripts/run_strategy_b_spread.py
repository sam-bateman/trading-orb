"""Strategy B — put-spread variant comparison.

Runs the synthetic engine twice:
1. Naked put (short -0.30Δ).
2. Put spread (short -0.30Δ, long -0.10Δ).

Spread truncates the left tail at the bought-put strike, at the cost of
a smaller net premium.

Outputs:
    reports/strategy_b_spread/comparison.json
    reports/strategy_b_spread/equity.png
    reports/strategy_b_spread/drawdown.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vrp.data.spx import load_spx
from vrp.data.vix import load_vix
from vrp.report.metrics import summary, drawdown_series
from vrp.strategies.strategy_b import run_strategy_b

TRAIN_START, TRAIN_END = "2013-01-01", "2018-12-31"
TEST_START,  TEST_END  = "2019-01-01", "2024-12-31"


def _windowed(ret: pd.Series) -> dict:
    return {
        "train": summary(ret.loc[TRAIN_START:TRAIN_END]),
        "test":  summary(ret.loc[TEST_START:TEST_END]),
    }


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "strategy_b_spread"
    out_dir.mkdir(parents=True, exist_ok=True)

    spx = load_spx(start=TRAIN_START, end=TEST_END)["close"]
    vix = load_vix(start=TRAIN_START, end=TEST_END)

    naked = run_strategy_b(spx, vix, target_delta=-0.30,
                           long_put_delta=None, tc_pct_of_premium=0.05)
    spread = run_strategy_b(spx, vix, target_delta=-0.30,
                            long_put_delta=-0.10, tc_pct_of_premium=0.05)

    results = {
        "naked_put": _windowed(naked["daily_return"]),
        "put_spread": _windowed(spread["daily_return"]),
    }
    (out_dir / "comparison.json").write_text(json.dumps(results, indent=2))

    naked_eq = (1 + naked["daily_return"]).cumprod()
    spread_eq = (1 + spread["daily_return"]).cumprod()

    fig, ax = plt.subplots(figsize=(11, 4))
    pd.DataFrame({"naked -0.30\u0394": naked_eq,
                  "spread -0.30\u0394 / -0.10\u0394": spread_eq}).plot(ax=ax)
    ax.set_title("Strategy B — naked put vs put spread")
    ax.set_ylabel("Equity (1 = starting capital)")
    ax.axvspan(TEST_START, TEST_END, alpha=0.08, color="red", label="out-of-sample")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out_dir / "equity.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 3))
    pd.DataFrame({
        "naked -0.30\u0394":           drawdown_series(naked["daily_return"]),
        "spread -0.30\u0394 / -0.10\u0394": drawdown_series(spread["daily_return"]),
    }).plot(ax=ax)
    ax.set_title("Strategy B — drawdown comparison")
    ax.set_ylabel("Drawdown")
    fig.savefig(out_dir / "drawdown.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Strategy B spread comparison outputs written to {out_dir}")

    def _brief(bundle):
        return {k: {"sharpe": v["sharpe"],
                     "ann_return": v["ann_return"],
                     "max_drawdown": v["max_drawdown"]} for k, v in bundle.items()}

    print("Naked:",  json.dumps(_brief(results["naked_put"]), indent=2))
    print("Spread:", json.dumps(_brief(results["put_spread"]), indent=2))


if __name__ == "__main__":
    main()
