# Intraday Opening Range Breakout (ORB) Strategy

A rigorously tested intraday trading strategy that trades breakouts from the first 20 minutes of the trading day. Walk-forward validated on 10 years of data (2016-2026) across 20 liquid US stocks.

## Strategy Summary

| Parameter | Value |
|---|---|
| Opening Range | First 20 minutes (9:30-9:50 AM ET) |
| Entry Window | 10:00 - 11:30 AM ET |
| Direction | Both (long breakouts + short breakdowns) |
| Volume Filter | 1.2x relative volume required |
| Target | 0.75x OR range |
| Stop Loss | 0.50x OR range (1.5:1 reward/risk) |
| Trailing Stop | Breakeven after 1x OR profit |
| Time Exit | Flatten all by 3:50 PM |
| Max Positions | 3 simultaneous |

## 10-Year Backtest Results (2016-2026)

| Metric | Value |
|---|---|
| Total Trades | 4,292 |
| Win Rate | 50.3% |
| Net PnL | +$30,989 ($400 risk/trade) |
| Annualized Return | +4.8% |
| Profit Factor | 1.31 |
| Sharpe Ratio | 2.47 |
| Max Drawdown | $1,271 |
| Calmar Ratio | 24.39 |
| Beta to SPY | -0.004 (market neutral) |
| Profitable Years | 11/11 (100%) |
| Profitable Months | 91/122 (75%) |
| Profitable Tickers | 19/20 (95%) |
| Monte Carlo (10k trials) | 100% profitable |
| Survives 3x slippage | Yes |

## How It Was Built

This strategy was developed using a systematic, multi-phase process designed to prevent overfitting:

### Phase 1: Universe Selection
Screened 150+ stocks for intraday tradability (volume, volatility, spread). Selected 20 names. Analyzed intraday volume profiles, range by hour, and autocorrelation structure to determine strategy direction (trend-following vs mean-reversion).

### Phase 2: Data Pipeline
Built a clean data pipeline with timezone handling, gap detection, validation, and derived columns (VWAP, opening range, relative volume, previous day levels).

### Phase 3: Strategy Hypothesis
Defined the Opening Range Breakout hypothesis with precise entry/exit logic. Generated signals and visually inspected example trades before backtesting.

### Phase 4: Realistic Backtesting
Event-driven backtester with $0.01/share slippage, $0.005/share commission, next-bar fills, position sizing based on fixed dollar risk, daily loss limits, and no overnight holds.

### Phase 5: Statistical Validation
Full validation suite: PnL distribution, robustness checks (half-split, remove top 5 trades, increase slippage, timing sensitivity), Monte Carlo simulation (10,000 trials).

### Optimization Iterations (V1-V5)
Ran ~8,000 parameter combinations across 5 rounds on 41 days of Yahoo data:
- **V1**: Found morning-only entries outperform (Sharpe 4.02)
- **V2**: Found short-only dominates on 41 days (Sharpe 7.19)
- **V3**: Ultra-fine-tuned stops (Sharpe 8.27)
- **V4**: Added both directions with asymmetric params
- **V5**: Identified potential overfitting to 41-day window

### The Critical Test
Recognized that 41 days was insufficient. Pulled 12 months of data from Alpaca — strategy degraded significantly. Then added volatility filter (1.2x relative volume) and re-optimized with walk-forward validation (train on months 1-6, test on months 7-12). Test period outperformed training period.

### Maximum History Validation
Pulled 10 years of 5-minute data from Alpaca (3.47M bars). Strategy passed all 11 robustness checks with no modifications. Every single year from 2016-2026 was profitable.

## Key Insight

The volume filter is the edge. Without it, breakouts are a coin flip. Requiring 1.2x above-average volume on the breakout bar filters out weak/fake breakouts and only enters when there's institutional participation behind the move.

## Project Structure

```
src/
  # Data Pipeline
  fetch_alpaca_data.py      # Fetch intraday data from Alpaca (12 months)
  fetch_max_data.py         # Fetch max history (10 years)
  intraday_data.py          # Clean, validate, enrich intraday data
  phase1_screener.py        # Stock universe screening

  # Strategy
  orb_strategy.py           # Signal generation + feature engineering

  # Backtesting
  backtester_v2.py          # Event-driven backtester with realistic execution
  run_12m_backtest.py       # 12-month backtest
  run_max_backtest.py       # 10-year backtest
  run_12m_optimization.py   # Walk-forward optimization

  # Optimization
  run_strategy_sims.py      # V1: 6 strategy variants
  run_deep_sims.py          # V1: 3,000+ parameter sweep
  run_deep_sims_v2.py       # V2: Short-only discovery
  run_deep_sims_v3.py       # V3: Fine-tuning
  run_deep_sims_v4.py       # V4: Direction-neutral
  run_deep_sims_v5.py       # V5: Chart-driven refinements

  # Live Trading
  paper_trade_orb.py        # Alpaca paper trading bot

  # Reports
  generate_report.py        # V1 report
  generate_report_v4.py     # V4 report
  generate_final_report.py  # Final 12-month report
  phase5_validation.py      # Full statistical validation
```

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Set your Alpaca API keys in `src/paper_trade_orb.py` or as environment variables.

## Usage

### Fetch Data
```bash
python src/fetch_max_data.py
```

### Run 10-Year Backtest
```bash
python src/run_max_backtest.py
```

### Paper Trade
```bash
python src/paper_trade_orb.py
```

### Monitor
```bash
tail -f paper_trade_orb.log
cat paper_trade_logs/trade_log.json
```

## Disclaimer

This is not financial advice. This project is for educational and research purposes only. Trading involves substantial risk of loss. Past performance, including backtested performance, does not guarantee future results. Never trade money you can't afford to lose.

The strategy was developed and tested using historical data. Real-world execution may differ due to slippage, latency, partial fills, and changing market conditions. Always paper trade extensively before risking real capital.
