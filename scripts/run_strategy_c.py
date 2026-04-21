"""Strategy C baseline — conditional put-writer at threshold = 2 vol points.

Runs two variants at the spec-default threshold:
1. Naked put (short -0.30Δ), gated.
2. Spread (short -0.30Δ, long -0.10Δ), gated.

Baseline threshold of 2.0 is a judgment call, not tuned on the data.
The sensitivity script (run_strategy_c_sensitivity.py) picks the train-
optimal threshold and evaluates held-out test performance separately.

Outputs:
    reports/strategy_c/metrics_train.json
    reports/strategy_c/metrics_test.json
    reports/strategy_c/active_months.json
    reports/strategy_c/equity.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from vrp.data.spx import load_spx
from vrp.data.vix import load_vix
from vrp.report.metrics import summary
from vrp.strategies.strategy_b import run_strategy_b
from vrp.strategies.strategy_c import run_strategy_c
from vrp.util.vol import close_to_close_rv
from vrp.util.vrp_signal import compute_vrp

TRAIN_START, TRAIN_END = "2013-01-01", "2018-12-31"
TEST_START,  TEST_END  = "2019-01-01", "2024-12-31"
THRESHOLD = 2.0


def _windowed(ret: pd.Series) -> dict:
    return {
        "train": summary(ret.loc[TRAIN_START:TRAIN_END]),
        "test":  summary(ret.loc[TEST_START:TEST_END]),
    }


def _brief(r: dict) -> dict:
    return {"sharpe": r["sharpe"], "ann_return": r["ann_return"],
             "max_drawdown": r["max_drawdown"]}


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent / "reports" / "strategy_c"
    out_dir.mkdir(parents=True, exist_ok=True)

    spx = load_spx(start=TRAIN_START, end=TEST_END)["close"]
    vix = load_vix(start=TRAIN_START, end=TEST_END)
    rv = close_to_close_rv(spx, window=20)
    vrp = compute_vrp(vix, rv).dropna()

    naked_b = run_strategy_b(spx, vix, target_delta=-0.30,
                              tc_pct_of_premium=0.05)
    spread_b = run_strategy_b(spx, vix, target_delta=-0.30,
                               long_put_delta=-0.10, tc_pct_of_premium=0.05)

    naked_c = run_strategy_c(spx, vix, vrp, threshold=THRESHOLD,
                              target_delta=-0.30, tc_pct_of_premium=0.05)
    spread_c = run_strategy_c(spx, vix, vrp, threshold=THRESHOLD,
                               target_delta=-0.30, long_put_delta=-0.10,
                               tc_pct_of_premium=0.05)

    metrics = {
        "threshold_vol_points": THRESHOLD,
        "naked_b":  _windowed(naked_b["daily_return"]),
        "naked_c":  _windowed(naked_c["daily_return"]),
        "spread_b": _windowed(spread_b["daily_return"]),
        "spread_c": _windowed(spread_c["daily_return"]),
    }
    (out_dir / "metrics_train.json").write_text(
        json.dumps({k: (v["train"] if isinstance(v, dict) and "train" in v else v)
                     for k, v in metrics.items()}, indent=2)
    )
    (out_dir / "metrics_test.json").write_text(
        json.dumps({k: (v["test"] if isinstance(v, dict) and "test" in v else v)
                     for k, v in metrics.items()}, indent=2)
    )
    (out_dir / "active_months.json").write_text(json.dumps({
        "naked": naked_c["active_months_fraction"],
        "spread": spread_c["active_months_fraction"],
        "threshold_vol_points": THRESHOLD,
    }, indent=2))

    fig, ax = plt.subplots(figsize=(11, 4))
    pd.DataFrame({
        "B naked (unconditional)": (1 + naked_b["daily_return"]).cumprod(),
        "C naked (gated)":         (1 + naked_c["daily_return"]).cumprod(),
        "B spread (unconditional)":(1 + spread_b["daily_return"]).cumprod(),
        "C spread (gated)":        (1 + spread_c["daily_return"]).cumprod(),
    }).plot(ax=ax)
    ax.set_title(f"Strategy C vs B — threshold = {THRESHOLD} vol points")
    ax.set_ylabel("Equity (1 = starting capital)")
    ax.axvspan(TEST_START, TEST_END, alpha=0.08, color="red", label="out-of-sample")
    ax.legend(loc="upper left", fontsize=8)
    fig.savefig(out_dir / "equity.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    print(f"Strategy C baseline outputs written to {out_dir}")
    print(f"Naked active months fraction:  {naked_c['active_months_fraction']:.2%}")
    print(f"Spread active months fraction: {spread_c['active_months_fraction']:.2%}")
    print("\nComparison:")
    for name, block in metrics.items():
        if not isinstance(block, dict) or "train" not in block:
            continue
        print(f"  {name}: train={_brief(block['train'])} test={_brief(block['test'])}")


if __name__ == "__main__":
    main()
