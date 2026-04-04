"""
Test the V4 asymmetric ORB config on the full 12-month Alpaca dataset.
The 41-day initial backtest looked good — this is the sanity check on 6x more data.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from fetch_alpaca_data import load_12m_dataset
from intraday_data import add_opening_range
from orb_strategy import compute_orb_features
from run_deep_sims_v4 import generate_signals_v4
from backtester_v2 import Backtester

OUTPUT_DIR = Path(__file__).parent.parent / "backtest_12m_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def run_backtest_12m():
    """Load 12-month data, run V4 signals, backtest, and print the full results with robustness checks."""
    print("=" * 70)
    print("12-MONTH BACKTEST — V4 ASYMMETRIC ORB")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load 12-month data
    data_raw = load_12m_dataset()
    print(f"Loaded {len(data_raw)} symbols")

    total_bars = sum(len(df) for df in data_raw.values())
    days = max(len(df['trading_day'].unique()) for df in data_raw.values())
    print(f"Total bars: {total_bars:,}")
    print(f"Trading days: {days}")

    # Add features
    print("Adding features...")
    for s in data_raw:
        data_raw[s] = compute_orb_features(data_raw[s])

    # V4 best config (honest — no ticker/day exclusions)
    params = {
        'or_minutes': 12, 'long_target': 0.5, 'long_stop': 0.375,
        'short_target': 1.125, 'short_stop': 0.375,
        'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': 11.5,
        'direction': 'both', 'risk_per_trade': 400, 'max_positions': 3,
        'min_or_range_pct': 0, 'gap_adaptive': False, 'gap_contra': False,
    }

    # Generate signals
    print("Generating signals...")
    data = {}
    total_signals = 0
    for symbol, df in data_raw.items():
        data[symbol] = generate_signals_v4(df.copy(), params)
        sigs = (data[symbol]['signal'] != 0).sum()
        total_signals += sigs
        print(f"  {symbol}: {sigs} signals")

    print(f"\nTotal signals: {total_signals}")

    # Run backtest
    print("\nRunning backtest...")
    bt = Backtester(risk_per_trade=400, max_positions=3)
    trade_log = bt.run(data)
    trade_log.to_csv(OUTPUT_DIR / 'trades_12m.csv', index=False)

    if len(trade_log) == 0:
        print("NO TRADES")
        return

    # Full stats
    n = len(trade_log)
    winners = trade_log[trade_log['pnl_net'] > 0]
    losers = trade_log[trade_log['pnl_net'] <= 0]
    longs = trade_log[trade_log['direction'] == 'LONG']
    shorts = trade_log[trade_log['direction'] == 'SHORT']

    total_pnl = trade_log['pnl_net'].sum()
    win_rate = len(winners) / n * 100
    avg_trade = total_pnl / n
    avg_win = winners['pnl_net'].mean() if len(winners) > 0 else 0
    avg_loss = losers['pnl_net'].mean() if len(losers) > 0 else 0
    pf = winners['pnl_net'].sum() / abs(losers['pnl_net'].sum()) if losers['pnl_net'].sum() != 0 else 0
    costs = trade_log['slippage_cost'].sum() + trade_log['commission_cost'].sum()

    cum = trade_log['pnl_net'].cumsum()
    max_dd = (cum - cum.cummax()).min()

    trade_log['trade_date'] = pd.to_datetime(trade_log['entry_time']).dt.date
    daily_pnl = trade_log.groupby('trade_date')['pnl_net'].sum()
    trading_days = len(daily_pnl)
    winning_days = (daily_pnl > 0).sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0

    target_hits = (trade_log['exit_reason'] == 'target_hit').sum()

    # Robustness
    half = n // 2
    h1 = trade_log.iloc[:half]['pnl_net'].sum()
    h2 = trade_log.iloc[half:]['pnl_net'].sum()

    # Quarters
    q_size = n // 4
    q1 = trade_log.iloc[:q_size]['pnl_net'].sum()
    q2 = trade_log.iloc[q_size:q_size*2]['pnl_net'].sum()
    q3 = trade_log.iloc[q_size*2:q_size*3]['pnl_net'].sum()
    q4 = trade_log.iloc[q_size*3:]['pnl_net'].sum()

    top5 = trade_log.nlargest(5, 'pnl_net')['pnl_net'].sum()
    pnl_no_top5 = total_pnl - top5

    extra_slip = trade_log['shares'] * 0.01 * 2
    pnl_2x_slip = total_pnl - extra_slip.sum()
    pnl_3x_slip = total_pnl - extra_slip.sum() * 2

    # Monte Carlo
    np.random.seed(42)
    mc = np.array([np.random.choice(trade_log['pnl_net'].values, size=n, replace=True).sum()
                   for _ in range(10000)])
    mc_profitable = (mc > 0).mean() * 100

    # Direction breakdown
    long_pnl = longs['pnl_net'].sum() if len(longs) > 0 else 0
    short_pnl = shorts['pnl_net'].sum() if len(shorts) > 0 else 0

    # Monthly breakdown
    trade_log['trade_month'] = pd.to_datetime(trade_log['entry_time']).dt.to_period('M')
    monthly = trade_log.groupby('trade_month')['pnl_net'].agg(['sum', 'count'])

    # Ticker breakdown
    ticker_pnl = trade_log.groupby('symbol')['pnl_net'].agg(['sum', 'count', 'mean'])
    profitable_tickers = (ticker_pnl['sum'] > 0).sum()

    ann_return = (total_pnl / 100_000) * (252 / trading_days) * 100

    # Print report
    print(f"\n{'='*70}")
    print(f"12-MONTH BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"  Period:               {trading_days} trading days (~{trading_days/21:.0f} months)")
    print(f"  Total Trades:         {n}")
    print(f"  Winners:              {len(winners)} ({win_rate:.1f}%)")
    print(f"  Losers:               {len(losers)} ({100-win_rate:.1f}%)")
    print(f"  NET PnL:              ${total_pnl:+,.2f}")
    print(f"  Ann. Return:          {ann_return:+.1f}%")
    print(f"  Avg Trade:            ${avg_trade:+,.2f}")
    print(f"  Avg Winner:           ${avg_win:+,.2f}")
    print(f"  Avg Loser:            ${avg_loss:+,.2f}")
    print(f"  Win/Loss Ratio:       {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0 else "")
    print(f"  Profit Factor:        {pf:.2f}")
    print(f"  Max Drawdown:         ${max_dd:,.2f}")
    print(f"  Sharpe Ratio:         {sharpe:.2f}")
    print(f"  Winning Days:         {winning_days}/{trading_days} ({winning_days/trading_days*100:.0f}%)")
    print(f"  Target Hit Rate:      {target_hits/n*100:.1f}%")
    print(f"  Total Costs:          ${costs:,.2f}")

    print(f"\n--- DIRECTION ---")
    print(f"  Long:  {len(longs)} trades, PnL ${long_pnl:+,.2f}")
    print(f"  Short: {len(shorts)} trades, PnL ${short_pnl:+,.2f}")

    print(f"\n--- ROBUSTNESS ---")
    print(f"  First half:           ${h1:+,.2f}")
    print(f"  Second half:          ${h2:+,.2f}")
    print(f"  Both halves profit:   {'YES' if h1 > 0 and h2 > 0 else 'NO'}")
    print(f"  Q1: ${q1:+,.2f} | Q2: ${q2:+,.2f} | Q3: ${q3:+,.2f} | Q4: ${q4:+,.2f}")
    print(f"  Profitable quarters:  {sum(1 for q in [q1,q2,q3,q4] if q > 0)}/4")
    print(f"  PnL without top 5:    ${pnl_no_top5:+,.2f}")
    print(f"  Outlier dependent:    {'YES' if total_pnl > 0 and pnl_no_top5 < 0 else 'NO'}")
    print(f"  With 2x slippage:     ${pnl_2x_slip:+,.2f}")
    print(f"  With 3x slippage:     ${pnl_3x_slip:+,.2f}")
    print(f"  Monte Carlo profit:   {mc_profitable:.1f}%")

    print(f"\n--- MONTHLY ---")
    for month, row in monthly.iterrows():
        marker = "+" if row['sum'] > 0 else "-"
        print(f"  {month}: ${row['sum']:+8,.2f} ({int(row['count'])} trades) {marker}")

    profitable_months = (monthly['sum'] > 0).sum()
    print(f"  Profitable months:    {profitable_months}/{len(monthly)}")

    print(f"\n--- TICKERS ---")
    for sym in ticker_pnl.sort_values('sum', ascending=False).index:
        r = ticker_pnl.loc[sym]
        print(f"  {sym:6s}: ${r['sum']:+8,.2f} ({int(r['count'])} trades)")
    print(f"  Profitable: {profitable_tickers}/{len(ticker_pnl)}")

    # VERDICT
    print(f"\n{'='*70}")
    print("VERDICT (12-MONTH)")
    print(f"{'='*70}")

    checks = {
        'Net profitable': total_pnl > 0,
        'PF >= 1.2': pf >= 1.2,
        'Sharpe >= 1.0': sharpe >= 1.0,
        'Both halves profitable': h1 > 0 and h2 > 0,
        '3+ quarters profitable': sum(1 for q in [q1,q2,q3,q4] if q > 0) >= 3,
        'MC > 70%': mc_profitable > 70,
        'Survives 2x slippage': pnl_2x_slip > 0,
        'Not outlier dependent': not (total_pnl > 0 and pnl_no_top5 < 0),
        '50%+ tickers profitable': profitable_tickers >= len(ticker_pnl) * 0.5,
        '50%+ months profitable': profitable_months >= len(monthly) * 0.5,
    }

    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    passed = sum(1 for v in checks.values() if v)
    failed = sum(1 for v in checks.values() if not v)
    print(f"\n  {passed} PASS / {failed} FAIL")

    if failed == 0:
        print("\n  VERDICT: STRATEGY IS VALIDATED ON 12 MONTHS. Proceed to paper trading.")
    elif failed <= 2:
        print("\n  VERDICT: MARGINAL. Some concerns but worth paper trading with caution.")
    else:
        print("\n  VERDICT: STRATEGY DOES NOT HOLD UP. The 41-day results were likely overfit.")

    print(f"\n{'='*70}")
    print(f"  41-day vs 12-month comparison:")
    print(f"  {'Metric':<25} {'41 days':>12} {'234 days':>12}")
    print(f"  {'Net PnL':<25} {'$+1,738':>12} ${total_pnl:>+11,.2f}")
    print(f"  {'Sharpe':<25} {'6.53':>12} {sharpe:>12.2f}")
    print(f"  {'PF':<25} {'1.41':>12} {pf:>12.2f}")
    print(f"  {'Win Rate':<25} {'41.8%':>12} {win_rate:>11.1f}%")
    print(f"  {'Max DD':<25} {'$-640':>12} ${max_dd:>11,.2f}")
    print(f"  {'MC Profitable':<25} {'99%':>12} {mc_profitable:>11.0f}%")
    print(f"{'='*70}")

    return trade_log


if __name__ == "__main__":
    trade_log = run_backtest_12m()
