# Volatility Risk Premium Harvesting — Design Spec

**Date:** 2026-04-19
**Type:** Resume/portfolio project for quant research roles
**Framing:** Replication + comparative study. **Not** a novel-alpha claim.

## Context

A research project systematically harvesting the volatility risk premium
(VRP) on SPX, comparing three implementation methods with explicit tail-risk
management. The deliverable is a GitHub repo framed as a replication and
comparative study.

Resume/portfolio audience: the interviewer looks for honest treatment of tail
risk, methodological rigor, clean attribution of returns to the documented
risk premium rather than overfitting, and acknowledgment of limitations.
Inflated Sharpe numbers are red flags, not selling points.

## Core Thesis

The VRP exists because investors pay an insurance premium for crash protection.
Implied volatility on SPX options has historically exceeded subsequent realized
volatility by ~3-4 vol points. This premium is harvestable but comes with
severe tail risk (2008, Feb 2018, March 2020). The research question is NOT
"does VRP exist" (it does, extensively documented) but **"what is the most
robust way to harvest it after accounting for tail events, and how much of
the gross premium survives honest tail-risk accounting?"**

## Core Principles (non-negotiable)

1. **No lookahead bias.** Use only information available at the decision point.
   If realized vol is computed over `[t-20, t]`, trade at `t+1`.
2. **Strict train/test split:** train 2006-2016, test 2017-2024. The test set
   **must** contain Feb 2018 and March 2020 — these are the defining events
   for this strategy and excluding them is academically dishonest.
3. **No parameter tuning on the test set.** Pick thresholds, lookback windows,
   and overlays on the training set only. Report test set results once.
4. **Transaction costs modeled explicitly.** VIX futures: 1-2 bps slippage
   per roll. Options: bid-ask is substantial, use 5-10% of premium as the
   cost estimate for ATM, wider for OTM.
5. **Every design choice cites a paper or is explicitly flagged as a
   judgment call** that needs sensitivity analysis.

## Strategies

### Strategy A — VIX Futures Term Structure Carry

- Short front-month VIX future, long second-month VIX future. Dollar-neutral.
- Roll 5 trading days before front-month expiration (document the choice;
  test sensitivity).
- Position sizing: constant notional to start; scale later.
- Expected: steady gains in contango, sharp losses in backwardation.

### Strategy B — Systematic Put-Writing

- Sell 1-month SPX puts at delta ≈ −0.30 (Black-Scholes, given ATM IV).
- Cash-secured (no leverage in baseline).
- Roll monthly at expiration.
- Benchmark against **CBOE PUT index** (published version of this strategy;
  primary backtest series).
- Also implement a put-spread version (sell −0.30 delta, buy −0.10 delta)
  and compare.

### Strategy C — Conditional VRP Harvester (the "research contribution")

- At each month-end, compute current VRP estimate:
  `VRP_t = IV_t - RV_t`
  where `IV_t` is 30-day ATM SPX implied vol (use VIX as proxy; flag) and
  `RV_t` is 20-day realized vol.
- Sell vol (via Strategy B mechanism) only when `VRP_t > threshold`
  (start at 2 vol points; run sensitivity analysis).
- When `VRP_t` below threshold, stay in cash.
- Hypothesis: VRP is time-varying; conditional exposure improves risk-adjusted
  returns even at the cost of lower total premium.
- Cite Dew-Becker et al. on term structure of variance risk premia.

## Tail-Risk Overlays (all three applied to Strategy C)

### Overlay 1 — VIX Regime Filter

- Go to cash when `VIX > 30` OR when VIX term structure inverts
  (front-month VX > second-month VX).
- Re-enter when `VIX < 25` and contango restored for 7 days of confirmation
  (anti-whipsaw).

### Overlay 2 — Realized Vol Position Scaling

- Scale position by `target_vol / realized_vol_trailing_20d`.
- Target: 10% annualized portfolio vol.
- Leverage cap: 1.0× (no upsizing above baseline).

### Overlay 3 — Tail Hedge Spend

- Use 15% of premium collected to buy 1-month 5-delta SPX puts.
- Quantify: does this reduce max drawdown more than it costs in returns?

## Data

- **SPX prices:** `yfinance` (`^GSPC` / `^SPX`).
- **VIX spot:** `yfinance` (`^VIX`).
- **VIX futures:** CBOE historical settlements (free CSVs from cboe.com).
- **CBOE indices (PUT, BXM, BXMD):** CBOE Indices historical (free).
- **SPX option chains (live, for validation only):** `yfinance`; acknowledge
  that backtesting option-level strategies requires tick/EOD option data we
  do not have. Strategy B's historical backtest uses the published PUT index.
- **SPX realized vol:** computed from daily returns.
- **SPX implied vol:** VIX as proxy (approximation; flagged).

## Required Reporting

For each of A, B, C, and C + each overlay:

- Annualized return, vol, Sharpe, Sortino
- Max drawdown, drawdown duration, time to recovery
- Worst single day / week / month
- Regime performance: 2008 GFC, 2015 vol spike, Feb 2018, March 2020, 2022
- Return distribution: skewness, kurtosis, 1%/5%/95%/99% percentiles
- Correlation to SPX (expect strongly negative in tails)
- Alpha vs SPX and vs CBOE PUT index
- Turnover and transaction cost sensitivity (sweep 5-30 bps futures; 5-15%
  of premium for options)

## Tail-Risk Analysis (dedicated section)

1. **Block bootstrap** of returns preserving vol clustering (block size
   20-60 days) to estimate tail distributions. Report bootstrapped 1%-VaR
   and expected shortfall.
2. **Stress test:** what would Strategy A have done in October 1987 if we
   extrapolate? (VIX-equivalent spiked 20+ points in a day.)
3. **Explicit disclosure:** "This strategy is short volatility. It will lose
   money catastrophically in a vol spike. Backtests underestimate tail risk
   because the sample contains only ~2-3 major vol events."

## Deliverables

1. Clean Python package: `src/vrp/`, `tests/`, `notebooks/`, `pyproject.toml`.
2. Reproducible: `pip install -e .` then `python -m vrp.run_all` regenerates
   all tables and figures.
3. Tests for core functions: VRP computation, roll logic, Black-Scholes
   helpers, return calculation.
4. Results notebook generating paper-style figures and tables.
5. README structured as a paper: Abstract, Literature Review, Data,
   Methodology, Results, Tail-Risk Analysis, Limitations, References.

## References

- Bakshi & Kapadia (2003). Delta-Hedged Gains and the Negative Market
  Volatility Risk Premium.
- Carr & Wu (2009). Variance Risk Premiums.
- Dew-Becker, Giglio, Le, Rodriguez (2017). The Price of Variance Risk.
- Israelov & Nielsen (2015). Covered Calls Uncovered (AQR) — implementation
  realism critique.
- Bondarenko (2014). Why Are Put Options So Expensive?
- CBOE White Paper on PUT Index methodology.

## Phased Build Plan

**Phase 1 (this plan):** scaffolding, data layer, Strategy A end-to-end,
sanity check vs published literature.

**Phase 2:** Strategy B (PUT index backtest + synthetic put-writer for recent
data), comparison to Strategy A.

**Phase 3:** Strategy C conditional harvester; threshold sensitivity analysis
on training window.

**Phase 4:** Overlays 1-3, each applied independently to C and reported.

**Phase 5:** Bootstrap tail analysis, 1987 stress test, paper-style writeup.

Each phase gets its own plan, written only after the prior phase is validated.

## Tooling Constraints

- Default: `pandas`, `numpy`, `scipy`, `statsmodels`. `arch` for GARCH if
  needed. No ML frameworks.
- Real docstrings citing source papers.
- Commit after each sub-step.
- If a backtest produces Sharpe > 2.0, treat as a bug. Investigate.
- Annualization conventions (252 days, sqrt scaling) are explicit everywhere.

## Red-Flag Rules

- If tuning a threshold on the test set: STOP, flag it.
- If correlation to SPX is not strongly negative in tails: suspect a bug.
- If Strategy A doesn't roughly match VXX (inverse) history: suspect a bug.
