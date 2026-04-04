"""
Full report on the best strategy found in the deep sims.
Loads the trade log, runs all the Phase 5 validation checks, and generates charts — all in one go.
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

OUTPUT_DIR = Path(__file__).parent.parent / "report_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Load trade log
tl = pd.read_csv(Path(__file__).parent.parent / "phase4_output" / "best_strategy_trades.csv")
n = len(tl)
winners = tl[tl['pnl_net'] > 0]
losers = tl[tl['pnl_net'] <= 0]

# ================================================================
# CORE METRICS
# ================================================================
total_pnl = tl['pnl_net'].sum()
total_gross = tl['pnl_gross'].sum()
total_costs = tl['slippage_cost'].sum() + tl['commission_cost'].sum()
win_rate = len(winners) / n * 100
avg_win = winners['pnl_net'].mean() if len(winners) > 0 else 0
avg_loss = losers['pnl_net'].mean() if len(losers) > 0 else 0
avg_trade = tl['pnl_net'].mean()
pf = winners['pnl_net'].sum() / abs(losers['pnl_net'].sum()) if losers['pnl_net'].sum() != 0 else 0
median_trade = tl['pnl_net'].median()

# Max drawdown
cum = tl['pnl_net'].cumsum()
peak = cum.cummax()
dd = cum - peak
max_dd = dd.min()

# Consecutive losses
is_loss = (tl['pnl_net'] <= 0).astype(int)
consec = is_loss.groupby((is_loss != is_loss.shift()).cumsum()).cumsum()
max_consec = consec.max()

# Daily
tl['trade_date'] = pd.to_datetime(tl['entry_time']).dt.date
daily_pnl = tl.groupby('trade_date')['pnl_net'].sum()
trading_days = len(daily_pnl)
winning_days = (daily_pnl > 0).sum()
sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0
calmar = (total_pnl / abs(max_dd)) if max_dd != 0 else 0

# Annualized
ann_return_pct = (total_pnl / 100_000) * (252 / trading_days) * 100

# By ticker
ticker_pnl = tl.groupby('symbol')['pnl_net'].agg(['sum', 'count', 'mean'])
profitable_tickers = (ticker_pnl['sum'] > 0).sum()

# By exit reason
reason_stats = tl.groupby('exit_reason').agg(
    count=('pnl_net', 'count'),
    total_pnl=('pnl_net', 'sum'),
    avg_pnl=('pnl_net', 'mean'),
    win_rate=('pnl_net', lambda x: (x > 0).mean() * 100),
)

# By direction
long_trades = tl[tl['direction'] == 'LONG']
short_trades = tl[tl['direction'] == 'SHORT']

# Half split
half = n // 2
h1 = tl.iloc[:half]
h2 = tl.iloc[half:]
h1_pnl = h1['pnl_net'].sum()
h2_pnl = h2['pnl_net'].sum()

# Remove top 5
top5_pnl = tl.nlargest(5, 'pnl_net')['pnl_net'].sum()
pnl_without_top5 = total_pnl - top5_pnl

# Slippage sensitivity
extra_slip = tl['shares'] * 0.01 * 2
pnl_2x_slip = total_pnl - extra_slip.sum()
pnl_3x_slip = total_pnl - extra_slip.sum() * 2

# Monte Carlo
np.random.seed(42)
mc_results = [np.random.choice(tl['pnl_net'].values, size=n, replace=True).sum() for _ in range(10000)]
mc = np.array(mc_results)
mc_profitable = (mc > 0).mean() * 100

# ================================================================
# GENERATE REPORT
# ================================================================

report = []
report.append("=" * 72)
report.append("INTRADAY TRADING STRATEGY — FULL VALIDATION REPORT")
report.append("=" * 72)
report.append("")
report.append("STRATEGY: Opening Range Breakout (Morning Session)")
report.append("  Opening Range:    30-minute (9:30 - 10:00 AM ET)")
report.append("  Entry Window:     10:00 - 11:30 AM ET only")
report.append("  Entry Trigger:    Close above/below OR with 1.2x relative volume")
report.append("  Target:           0.75x OR range from entry")
report.append("  Stop:             0.50x OR range from entry (1.5:1 R:R)")
report.append("  Trailing:         Move stop to breakeven after 1x OR profit")
report.append("  Time Exit:        Flatten all by 3:50 PM ET")
report.append("  Max Positions:    3 simultaneous")
report.append("  Risk Per Trade:   $400")
report.append("  Slippage:         $0.01/share")
report.append("  Commission:       $0.005/share")
report.append("")
report.append(f"  Backtest Period:  {trading_days} trading days, 20 stocks")
report.append(f"  Starting Capital: $100,000")
report.append("")

report.append("-" * 72)
report.append("1. PERFORMANCE SUMMARY")
report.append("-" * 72)
report.append(f"  Total Trades:          {n}")
report.append(f"  Winners:               {len(winners)} ({win_rate:.1f}%)")
report.append(f"  Losers:                {len(losers)} ({100-win_rate:.1f}%)")
report.append(f"  Gross PnL:             ${total_gross:+,.2f}")
report.append(f"  Total Costs:           ${total_costs:,.2f} ({total_costs/total_gross*100:.1f}% of gross)" if total_gross > 0 else f"  Total Costs:           ${total_costs:,.2f}")
report.append(f"  NET PnL:               ${total_pnl:+,.2f}")
report.append(f"  Annualized Return:     {ann_return_pct:+.1f}%")
report.append(f"  Avg Trade (net):       ${avg_trade:+,.2f}")
report.append(f"  Median Trade:          ${median_trade:+,.2f}")
report.append(f"  Avg Winner:            ${avg_win:+,.2f}")
report.append(f"  Avg Loser:             ${avg_loss:+,.2f}")
report.append(f"  Win/Loss Ratio:        {abs(avg_win/avg_loss):.2f}x" if avg_loss != 0 else "")
report.append(f"  Profit Factor:         {pf:.2f}")
report.append(f"  Max Drawdown:          ${max_dd:,.2f}")
report.append(f"  Max Consec. Losses:    {max_consec}")
report.append(f"  Sharpe Ratio (ann.):   {sharpe:.2f}")
report.append(f"  Calmar Ratio:          {calmar:.2f}")
report.append(f"  Winning Days:          {winning_days}/{trading_days} ({winning_days/trading_days*100:.0f}%)")
report.append("")

report.append("-" * 72)
report.append("2. BREAKDOWN BY EXIT REASON")
report.append("-" * 72)
for reason, row in reason_stats.iterrows():
    report.append(f"  {reason:20s}  {int(row['count']):>4} trades  "
                  f"PnL ${row['total_pnl']:>+8,.2f}  "
                  f"Avg ${row['avg_pnl']:>+7,.2f}  "
                  f"WinR {row['win_rate']:.0f}%")
report.append("")

report.append("-" * 72)
report.append("3. BREAKDOWN BY DIRECTION")
report.append("-" * 72)
for label, subset in [("LONG", long_trades), ("SHORT", short_trades)]:
    if len(subset) == 0: continue
    wr = (subset['pnl_net'] > 0).mean() * 100
    report.append(f"  {label:6s}  {len(subset):>4} trades  "
                  f"PnL ${subset['pnl_net'].sum():>+8,.2f}  "
                  f"Avg ${subset['pnl_net'].mean():>+7,.2f}  "
                  f"WinR {wr:.0f}%")
report.append("")

report.append("-" * 72)
report.append("4. BREAKDOWN BY TICKER")
report.append("-" * 72)
for symbol in ticker_pnl.sort_values('sum', ascending=False).index:
    row = ticker_pnl.loc[symbol]
    marker = "  " if row['sum'] > 0 else "  "
    report.append(f"  {symbol:6s}  {int(row['count']):>3} trades  "
                  f"PnL ${row['sum']:>+8,.2f}  "
                  f"Avg ${row['mean']:>+7,.2f}")
report.append(f"\n  Profitable tickers: {profitable_tickers}/{len(ticker_pnl)}")
report.append("")

report.append("-" * 72)
report.append("5. ROBUSTNESS CHECKS")
report.append("-" * 72)
report.append(f"  First half PnL:        ${h1_pnl:+,.2f} ({len(h1)} trades)")
report.append(f"  Second half PnL:       ${h2_pnl:+,.2f} ({len(h2)} trades)")
report.append(f"  Both halves profit:    {'YES' if h1_pnl > 0 and h2_pnl > 0 else 'NO'}")
report.append(f"")
report.append(f"  Top 5 trades total:    ${top5_pnl:+,.2f}")
report.append(f"  PnL without top 5:     ${pnl_without_top5:+,.2f}")
report.append(f"  Outlier dependent:     {'YES' if total_pnl > 0 and pnl_without_top5 < 0 else 'NO'}")
report.append(f"")
report.append(f"  With 2x slippage:      ${pnl_2x_slip:+,.2f}")
report.append(f"  With 3x slippage:      ${pnl_3x_slip:+,.2f}")
report.append(f"  Survives 2x slip:      {'YES' if pnl_2x_slip > 0 else 'NO'}")
report.append("")

report.append("-" * 72)
report.append("6. MONTE CARLO SIMULATION (10,000 trials)")
report.append("-" * 72)
report.append(f"  Probability of profit: {mc_profitable:.1f}%")
report.append(f"  Mean final PnL:        ${np.mean(mc):+,.2f}")
report.append(f"  Median final PnL:      ${np.median(mc):+,.2f}")
report.append(f"  5th percentile:        ${np.percentile(mc, 5):+,.2f}")
report.append(f"  95th percentile:       ${np.percentile(mc, 95):+,.2f}")
report.append(f"  Worst case (1st pct):  ${np.percentile(mc, 1):+,.2f}")
report.append("")

# VERDICT
report.append("=" * 72)
report.append("7. VERDICT")
report.append("=" * 72)

issues = []
passes = []

if total_pnl > 0: passes.append("Net profitable")
else: issues.append("Net negative")

if pf >= 1.2: passes.append(f"Profit factor {pf:.2f} >= 1.2")
elif pf >= 1.0: issues.append(f"Profit factor {pf:.2f} marginal (want >= 1.2)")
else: issues.append(f"Profit factor {pf:.2f} < 1.0")

if sharpe >= 1.5: passes.append(f"Sharpe {sharpe:.2f} >= 1.5")
elif sharpe >= 0.5: passes.append(f"Sharpe {sharpe:.2f} acceptable")
else: issues.append(f"Sharpe {sharpe:.2f} too low")

if h1_pnl > 0 and h2_pnl > 0: passes.append("Both halves profitable")
else: issues.append("NOT profitable in both halves")

if mc_profitable >= 70: passes.append(f"Monte Carlo {mc_profitable:.0f}% profitable")
elif mc_profitable >= 55: passes.append(f"Monte Carlo {mc_profitable:.0f}% (marginal)")
else: issues.append(f"Monte Carlo {mc_profitable:.0f}% — not reliable")

if pnl_2x_slip > 0: passes.append("Survives 2x slippage")
else: issues.append("Dies with 2x slippage")

if not (total_pnl > 0 and pnl_without_top5 < 0): passes.append("Not outlier dependent")
else: issues.append("Depends on top 5 trades")

if profitable_tickers >= len(ticker_pnl) * 0.4: passes.append(f"{profitable_tickers}/{len(ticker_pnl)} tickers profitable")
else: issues.append(f"Only {profitable_tickers}/{len(ticker_pnl)} tickers profitable")

report.append("")
report.append(f"  PASSES ({len(passes)}):")
for p in passes:
    report.append(f"    [PASS] {p}")
report.append(f"")
report.append(f"  ISSUES ({len(issues)}):")
for i in issues:
    report.append(f"    [FAIL] {i}")

report.append("")
critical = [i for i in issues if any(w in i.lower() for w in ['net negative', 'not profitable in both', 'dies with', 'depends on'])]

if len(critical) == 0 and len(passes) >= 5:
    report.append("  RECOMMENDATION: PROCEED TO PAPER TRADING (Phase 6)")
    report.append("  Run for minimum 10 trading days. Compare live results to backtest.")
elif len(critical) == 0:
    report.append("  RECOMMENDATION: MARGINAL — consider refinements before paper trading")
else:
    report.append("  RECOMMENDATION: DO NOT TRADE LIVE — address critical issues first")

report.append("")
report.append("=" * 72)
report.append("  DISCLAIMER: This is not financial advice. Past performance")
report.append("  does not guarantee future results. Trade at your own risk.")
report.append("=" * 72)

# Print and save
text = "\n".join(report)
print(text)

with open(OUTPUT_DIR / 'strategy_report.txt', 'w') as f:
    f.write(text)

# ================================================================
# CHARTS
# ================================================================

fig, axes = plt.subplots(3, 3, figsize=(20, 16))

# 1. Equity curve
ax = axes[0, 0]
ax.plot(cum.values, 'b-', linewidth=1.2)
ax.fill_between(range(len(cum)), cum.values, alpha=0.15, color='blue')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_title('Cumulative Net PnL', fontweight='bold')
ax.set_xlabel('Trade #')
ax.set_ylabel('PnL ($)')
ax.grid(True, alpha=0.3)

# 2. PnL distribution
ax = axes[0, 1]
ax.hist(tl['pnl_net'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='red', linewidth=1)
ax.axvline(x=avg_trade, color='orange', linewidth=1.5, linestyle='--', label=f'Avg: ${avg_trade:+.2f}')
ax.set_title('Trade PnL Distribution', fontweight='bold')
ax.legend()

# 3. Daily PnL
ax = axes[0, 2]
colors = ['green' if v > 0 else 'red' for v in daily_pnl.values]
ax.bar(range(len(daily_pnl)), daily_pnl.values, color=colors, alpha=0.7)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_title(f'Daily PnL ({winning_days}/{trading_days} winning days)', fontweight='bold')
ax.set_ylabel('PnL ($)')

# 4. By ticker
ax = axes[1, 0]
sorted_tickers = ticker_pnl.sort_values('sum')
colors = ['green' if v > 0 else 'red' for v in sorted_tickers['sum'].values]
ax.barh(sorted_tickers.index, sorted_tickers['sum'].values, color=colors, alpha=0.7)
ax.set_title('PnL by Ticker', fontweight='bold')
ax.set_xlabel('Total PnL ($)')

# 5. By exit reason
ax = axes[1, 1]
colors = ['green' if v > 0 else 'red' for v in reason_stats['total_pnl'].values]
ax.bar(reason_stats.index, reason_stats['total_pnl'].values, color=colors, alpha=0.7)
ax.set_title('PnL by Exit Reason', fontweight='bold')
ax.set_ylabel('Total PnL ($)')

# 6. By hour
ax = axes[1, 2]
tl['entry_hour'] = pd.to_datetime(tl['entry_time']).dt.hour
hourly = tl.groupby('entry_hour')['pnl_net'].sum()
colors = ['green' if v > 0 else 'red' for v in hourly.values]
ax.bar(hourly.index, hourly.values, color=colors, alpha=0.7)
ax.set_title('PnL by Entry Hour', fontweight='bold')
ax.set_xlabel('Hour (ET)')

# 7. Monte Carlo
ax = axes[2, 0]
ax.hist(mc, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='red', linewidth=1.5)
ax.axvline(x=total_pnl, color='green', linewidth=1.5, linestyle='--', label=f'Actual: ${total_pnl:+,.0f}')
ax.set_title(f'Monte Carlo: {mc_profitable:.0f}% Profitable', fontweight='bold')
ax.set_xlabel('Final PnL ($)')
ax.legend()

# 8. Win rate by ticker
ax = axes[2, 1]
ticker_wr = tl.groupby('symbol')['pnl_net'].apply(lambda x: (x > 0).mean() * 100)
ticker_wr = ticker_wr.sort_values()
colors = ['green' if v >= 50 else 'red' for v in ticker_wr.values]
ax.barh(ticker_wr.index, ticker_wr.values, color=colors, alpha=0.7)
ax.axvline(x=50, color='black', linewidth=0.5, linestyle='--')
ax.set_title('Win Rate by Ticker', fontweight='bold')
ax.set_xlabel('Win Rate %')

# 9. Direction comparison
ax = axes[2, 2]
dir_data = {'LONG': long_trades['pnl_net'].sum(), 'SHORT': short_trades['pnl_net'].sum()}
colors = ['green' if v > 0 else 'red' for v in dir_data.values()]
ax.bar(dir_data.keys(), dir_data.values(), color=colors, alpha=0.7, width=0.4)
ax.set_title('PnL by Direction', fontweight='bold')
ax.set_ylabel('Total PnL ($)')

plt.suptitle('ORB Strategy — Morning Session (10:00-11:30 AM)\n'
             f'Net PnL: ${total_pnl:+,.2f} | Sharpe: {sharpe:.2f} | '
             f'Win Rate: {win_rate:.1f}% | PF: {pf:.2f}',
             fontsize=14, fontweight='bold', y=1.01)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'strategy_report.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\nCharts saved to {OUTPUT_DIR / 'strategy_report.png'}")
print(f"Report saved to {OUTPUT_DIR / 'strategy_report.txt'}")
