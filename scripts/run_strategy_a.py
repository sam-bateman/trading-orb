"""End-to-end run of Strategy A.

Produces:
    reports/strategy_a/metrics_train.json
    reports/strategy_a/metrics_test.json
    reports/strategy_a/regime_metrics_test.json
    reports/strategy_a/equity_curve.png
    reports/strategy_a/drawdown.png

Train/test split: the original plan specified 2006-2016 train / 2017-2024
test. CBOE VX per-contract historical data is only available from 2013
onward (pre-2013 returns HTTP 403). We therefore use:

    Train: 2013-01-01 to 2018-12-31  (6 years, includes Feb 2018 Volmageddon)
    Test:  2019-01-01 to 2024-12-31  (6 years, includes COVID 2020 and 2022 bear)

Both windows contain at least one significant short-vol blow-up event.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vrp.data.vx_futures import load_vx_continuous
from vrp.report.metrics import summary, drawdown_series
from vrp.report.regimes import regime_metrics
from vrp.strategies.strategy_a import run_strategy_a

TRAIN_START, TRAIN_END = "2013-01-01", "2018-12-31"
TEST_START,  TEST_END  = "2019-01-01", "2024-12-31"


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "strategy_a"
    out_dir.mkdir(parents=True, exist_ok=True)

    vx = load_vx_continuous(start=TRAIN_START, end=TEST_END)
    result = run_strategy_a(vx, roll_days_before_expiry=5, tc_bps_per_roll=1.0)
    ret = result["daily_return"]

    train_ret = ret.loc[TRAIN_START:TRAIN_END]
    test_ret  = ret.loc[TEST_START:TEST_END]

    (out_dir / "metrics_train.json").write_text(json.dumps(summary(train_ret), indent=2))
    (out_dir / "metrics_test.json").write_text(json.dumps(summary(test_ret), indent=2))
    (out_dir / "regime_metrics_test.json").write_text(
        json.dumps(regime_metrics(test_ret), indent=2, default=str)
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    (1 + ret).cumprod().plot(ax=ax)
    ax.set_title("Strategy A — Cumulative Return (full sample)")
    ax.set_ylabel("Equity (1 = starting capital)")
    ax.axvspan(TEST_START, TEST_END, alpha=0.1, color="red",
               label="out-of-sample")
    ax.legend()
    fig.savefig(out_dir / "equity_curve.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 3))
    drawdown_series(ret).plot(ax=ax, color="red")
    ax.set_title("Strategy A — Drawdown")
    ax.set_ylabel("Drawdown")
    fig.savefig(out_dir / "drawdown.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Strategy A metrics written to {out_dir}")
    print("Train summary:", json.dumps(summary(train_ret), indent=2))
    print("Test  summary:", json.dumps(summary(test_ret),  indent=2))


if __name__ == "__main__":
    main()
