# Volatility Risk Premium Harvesting — A Comparative Study

A research project replicating and comparing three implementations of VRP
harvesting on SPX, with explicit tail-risk accounting. Framed as a
comparative study of known strategies, not a novel-alpha claim. Primary
audience: readers evaluating quantitative research work.

## Abstract (Phase 1)

Phase 1 implements the naive baseline: a dollar-neutral short-front /
long-second VIX futures calendar spread with a 5-day pre-expiry roll, 1 bp
per-leg transaction cost on roll days, trained on 2013–2018 and evaluated
out-of-sample on 2019–2024. Both windows contain a major short-vol
blow-up event (Feb 2018 Volmageddon in train; COVID 2020 and the 2022
bear in test). The naive baseline **loses 12–15% annualized with ~60%
maximum drawdown** in both windows. The cross-correlation of daily
Strategy A returns with negative-VXX returns is 0.61 (as expected for a
VIX-shape trade), confirming the engine is directionally sound — the
negative cumulative PnL is a structural property of the naive
construction, not an implementation bug. This result motivates
Strategies B and C in subsequent phases.

## Why the naive baseline fails

Dollar-neutral calendar P&L per day reduces algebraically to
`r_second − r_front`, where `r` is the daily return of each leg. This is
a *shape* trade, not a *carry* trade: it profits when the term structure
steepens and loses when it flattens or inverts. The positive roll drift
that makes outright short-VIX trades (short VXX, short-front calendar,
etc.) profitable over time is cancelled out by the dollar-neutral
construction. What remains is:

- ≈zero expected daily return during quiet contango (both legs decay at
  similar rates per $ notional)
- large negative realizations during vol spikes (front responds more
  than second → `r_second − r_front` strongly negative)

Accumulated over 11 years including 2015, 2018, 2020, and 2022, the
tails dominate and the strategy bleeds.

This is the same result academics have documented for naive calendar
short-vol trades; see Alexander & Korovilas (2013) for a direct
treatment. The finding motivates two subsequent phases: (i) Strategy B,
a cash-secured put-writing variant that captures the VRP directly rather
than via a calendar shape, and (ii) Strategy C, a conditional variant
that gates exposure on a VRP-threshold signal.

## Strategies

- **Strategy A — VIX Term-Structure Carry (naive baseline).**
  Dollar-neutral short front-month VX, long second-month VX. Rolled 5
  trading days before expiry. **(Implemented in Phase 1.)**
- **Strategy B — Systematic Put-Writing.** Monthly −0.30Δ SPX puts,
  cash-secured; benchmarked against CBOE PUT index. *(Phase 2.)*
- **Strategy C — Conditional VRP Harvester.** Strategy B, gated on
  `VRP_t = IV_t − RV_t > threshold`. *(Phase 3.)*

Overlays (applied to C in Phase 4): VIX regime filter, realized-vol
position scaling, tail-hedge spend.

## Literature

- Bakshi & Kapadia (2003). Delta-Hedged Gains and the Negative Market
  Volatility Risk Premium.
- Carr & Wu (2009). Variance Risk Premiums.
- Alexander, Korovilas (2013). Understanding ETNs on VIX Futures.
- Dew-Becker, Giglio, Le, Rodriguez (2017). The Price of Variance Risk.
- Israelov & Nielsen (2015). Covered Calls Uncovered (AQR).
- Bondarenko (2014). Why Are Put Options So Expensive?

## Data

- SPX, VIX spot: Yahoo Finance (`^GSPC`, `^VIX`).
- VIX futures (VX) settlements: CBOE CDN per-contract historical CSVs
  at `cdn.cboe.com/data/us/futures/market_statistics/historical_data/`.
  Coverage is 2013-present; pre-2013 contracts return HTTP 403.
- CBOE benchmark indices (PUT, BXM): CBOE daily CSVs.

## Methodology notes

- Train 2013-01-01 → 2018-12-31. Test 2019-01-01 → 2024-12-31.
- The original spec targeted 2006 onward. CBOE per-contract VX history
  begins in 2013, which forced the reduction. The 2013+ window still
  contains sufficient regime variety (2015 flash crash, 2018
  Volmageddon, 2020 COVID, 2022 bear) for a meaningful test.
- No test-window parameter tuning.
- Transaction costs: 1 bp per leg per VX roll (baseline). Sensitivity
  sweep 5-30 bps is deferred to the Phase 1 robustness pass.
- Annualization: 252 trading days throughout.

## Reproduce Strategy A

```bash
pip install -e '.[dev]'
python scripts/run_strategy_a.py
python scripts/sanity_vxx.py
```

Outputs land in `reports/strategy_a/` (gitignored — regenerate from
source). `notebooks/01_strategy_a.ipynb` regenerates the same figures
interactively.

## Sanity-check gate

The daily-return correlation between Strategy A and `−VXX` is required
to fall in `[0.3, 0.7]`. Correlation outside this band indicates either
a sign bug in the VX PnL computation or a data-ingestion issue. At
Phase 1 completion: **0.609** (in-band).

## Phase 1 results

| window | Sharpe | ann. return | ann. vol | max DD | DD duration |
|---|---|---|---|---|---|
| train (2013-2018) | −0.74 | −14.7% | 19.9% | −59.6% | 1415 days |
| test (2019-2024)  | −0.60 | −11.7% | 19.5% | −58.2% | 1438 days |

Both Sharpe numbers are negative; both reflect the structural property
described above, not a bug. The VXX correlation gate (0.609) confirms
the engine.

## Limitations (Phase 1)

- Dollar-neutral continuous rolling is an approximation of how a real
  fund would trade this exposure; intraday slippage, contract-size
  rounding, and margin dynamics make the live path worse than the
  backtest.
- VIX is used as an IV proxy in later phases (variance-swap construct,
  not strictly ATM IV). Flagged in those phases' code.
- Sample contains ≤4 major vol events in 11 years. Backtests of
  short-vol strategies systematically underestimate tail risk; the
  bootstrap and stress-test analyses in Phase 5 are designed to correct
  for this.
- CBOE 2013-start truncation removes GFC 2008 from the sample. 2008
  is the canonical short-vol catastrophe and its absence is a known
  limitation of this replication.
