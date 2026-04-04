"""
Final report for the V4 asymmetric ORB strategy.
Runs the full stats suite on the V4 trade log and saves charts to report_output/.
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

tl = pd.read_csv(Path(__file__).parent.parent / "phase4_output" / "v4_best_trades.csv")
n = len(tl)
winners = tl[tl['pnl_net'] > 0]
losers = tl[tl['pnl_net'] <= 0]
longs = tl[tl['direction'] == 'LONG']
shorts = tl[tl['direction'] == 'SHORT']

total_pnl = tl['pnl_net'].sum()
total_gross = tl['pnl_gross'].sum()
total_costs = tl['slippage_cost'].sum() + tl['commission_cost'].sum()
win_rate = len(winners) / n * 100
avg_win = winners['pnl_net'].mean() if len(winners) > 0 else 0
avg_loss = losers['pnl_net'].mean() if len(losers) > 0 else 0
avg_trade = tl['pnl_net'].mean()
pf = winners['pnl_net'].sum() / abs(losers['pnl_net'].sum()) if losers['pnl_net'].sum() != 0 else 0
median_trade = tl['pnl_net'].median()

cum = tl['pnl_net'].cumsum()
peak = cum.cummax()
dd = cum - peak
max_dd = dd.min()

is_loss = (tl['pnl_net'] <= 0).astype(int)
consec = is_loss.groupby((is_loss != is_loss.shift()).cumsum()).cumsum()
max_consec = consec.max()

tl['trade_date'] = pd.to_datetime(tl['entry_time']).dt.date
daily_pnl = tl.groupby('trade_date')['pnl_net'].sum()
trading_days = len(daily_pnl)
winning_days = (daily_pnl > 0).sum()
sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if daily_pnl.std() > 0 else 0
calmar = (total_pnl / abs(max_dd)) if max_dd != 0 else 0
ann_return_pct = (total_pnl / 100_000) * (252 / trading_days) * 100

ticker_pnl = tl.groupby('symbol')['pnl_net'].agg(['sum', 'count', 'mean'])
profitable_tickers = (ticker_pnl['sum'] > 0).sum()

reason_stats = tl.groupby('exit_reason').agg(
    count=('pnl_net', 'count'), total_pnl=('pnl_net', 'sum'),
    avg_pnl=('pnl_net', 'mean'), win_rate=('pnl_net', lambda x: (x > 0).mean() * 100))

half = n // 2
h1_pnl = tl.iloc[:half]['pnl_net'].sum()
h2_pnl = tl.iloc[half:]['pnl_net'].sum()

top5 = tl.nlargest(5, 'pnl_net')['pnl_net'].sum()
pnl_no_top5 = total_pnl - top5

extra_slip = tl['shares'] * 0.01 * 2
pnl_2x_slip = total_pnl - extra_slip.sum()
pnl_3x_slip = total_pnl - extra_slip.sum() * 2

np.random.seed(42)
mc = np.array([np.random.choice(tl['pnl_net'].values, size=n, replace=True).sum() for _ in range(10000)])
mc_profitable = (mc > 0).mean() * 100

# Long/short breakdown
long_pnl = longs['pnl_net'].sum() if len(longs) > 0 else 0
short_pnl = shorts['pnl_net'].sum() if len(shorts) > 0 else 0
long_wr = (longs['pnl_net'] > 0).mean() * 100 if len(longs) > 0 else 0
short_wr = (shorts['pnl_net'] > 0).mean() * 100 if len(shorts) > 0 else 0

# ================================================================
report = []
report.append("=" * 72)
report.append("INTRADAY ORB STRATEGY — V4 FINAL REPORT")
report.append("Asymmetric Direction-Neutral Opening Range Breakout")
report.append("=" * 72)
report.append("")
report.append("STRATEGY PARAMETERS:")
report.append("  Opening Range:    12 minutes (9:30 - 9:42 AM ET)")
report.append("  Entry Window:     10:00 - 11:30 AM ET")
report.append("  LONG trades:      Target 0.50x OR | Stop 0.375x OR (1.33:1 R:R)")
report.append("  SHORT trades:     Target 1.125x OR | Stop 0.375x OR (3.0:1 R:R)")
report.append("  Trailing Stop:    Move to breakeven after 1x OR profit")
report.append("  Time Exit:        Flatten all by 3:50 PM ET")
report.append("  Volume Filter:    None (1.0x)")
report.append("  Max Positions:    3 simultaneous")
report.append("  Risk Per Trade:   $400")
report.append("  Slippage:         $0.01/share | Commission: $0.005/share")
report.append("")
report.append("  WHY ASYMMETRIC: Shorts have stronger follow-through (fear > greed)")
report.append("  so we give them a wider target (1.125x). Longs get a tight target")
report.append("  (0.5x) to capture quick pops without overstaying.")
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
report.append(f"  Total Costs:           ${total_costs:,.2f}")
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
report.append("2. DIRECTION BREAKDOWN")
report.append("-" * 72)
report.append(f"  LONG:   {len(longs):>4} trades | PnL ${long_pnl:>+9,.2f} | "
              f"Win Rate {long_wr:.1f}% | Avg ${longs['pnl_net'].mean():>+7.2f}" if len(longs) > 0 else "  LONG:   0 trades")
report.append(f"  SHORT:  {len(shorts):>4} trades | PnL ${short_pnl:>+9,.2f} | "
              f"Win Rate {short_wr:.1f}% | Avg ${shorts['pnl_net'].mean():>+7.2f}" if len(shorts) > 0 else "  SHORT:  0 trades")
report.append(f"  Short/Long PnL ratio: {short_pnl/max(long_pnl,1):.1f}x" if long_pnl > 0 else "")
report.append("")

report.append("-" * 72)
report.append("3. EXIT REASON BREAKDOWN")
report.append("-" * 72)
for reason, row in reason_stats.iterrows():
    report.append(f"  {reason:20s}  {int(row['count']):>4} trades  "
                  f"PnL ${row['total_pnl']:>+9,.2f}  Avg ${row['avg_pnl']:>+7,.2f}  WinR {row['win_rate']:.0f}%")
report.append("")

report.append("-" * 72)
report.append("4. TICKER BREAKDOWN")
report.append("-" * 72)
for symbol in ticker_pnl.sort_values('sum', ascending=False).index:
    row = ticker_pnl.loc[symbol]
    report.append(f"  {symbol:6s}  {int(row['count']):>3} trades  "
                  f"PnL ${row['sum']:>+9,.2f}  Avg ${row['mean']:>+7,.2f}")
report.append(f"\n  Profitable tickers: {profitable_tickers}/{len(ticker_pnl)}")
report.append("")

report.append("-" * 72)
report.append("5. ROBUSTNESS CHECKS")
report.append("-" * 72)
report.append(f"  First half PnL:        ${h1_pnl:+,.2f} ({half} trades)")
report.append(f"  Second half PnL:       ${h2_pnl:+,.2f} ({n-half} trades)")
report.append(f"  Both halves profit:    {'YES' if h1_pnl > 0 and h2_pnl > 0 else 'NO'}")
report.append(f"")
report.append(f"  Top 5 trades total:    ${top5:+,.2f}")
report.append(f"  PnL without top 5:     ${pnl_no_top5:+,.2f}")
report.append(f"  Outlier dependent:     {'YES' if total_pnl > 0 and pnl_no_top5 < 0 else 'NO'}")
report.append(f"")
report.append(f"  With 2x slippage:      ${pnl_2x_slip:+,.2f}")
report.append(f"  With 3x slippage:      ${pnl_3x_slip:+,.2f}")
report.append(f"  Survives 2x slip:      {'YES' if pnl_2x_slip > 0 else 'NO'}")
report.append(f"  Survives 3x slip:      {'YES' if pnl_3x_slip > 0 else 'NO'}")
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

report.append("-" * 72)
report.append("7. OPTIMIZATION HISTORY (V1 → V4)")
report.append("-" * 72)
report.append(f"  {'Version':<12} {'PnL':>10} {'Sharpe':>8} {'PF':>6} {'MaxDD':>10} {'Key Change'}")
report.append(f"  {'V1 base':<12} {'$+1,144':>10} {'4.02':>8} {'1.36':>6} {'$-354':>10} Morning only, tight target")
report.append(f"  {'V2 short':<12} {'$+2,151':>10} {'7.19':>8} {'1.89':>6} {'$-410':>10} Short-only, 15min OR")
report.append(f"  {'V3 tuned':<12} {'$+1,708':>10} {'8.27':>8} {'2.00':>6} {'$-218':>10} Ultra-tight stop 0.375x")
report.append(f"  {'V4 asym':<12} {'$+1,738':>10} {f'{sharpe:.2f}':>8} {f'{pf:.2f}':>6} {f'${max_dd:,.0f}':>10} Asymmetric long/short")
report.append("")

# VERDICT
report.append("=" * 72)
report.append("8. VERDICT")
report.append("=" * 72)

passes = []
issues = []
warnings_list = []

if total_pnl > 0: passes.append("Net profitable")
else: issues.append("Net negative")
if pf >= 1.2: passes.append(f"Profit factor {pf:.2f} >= 1.2")
elif pf >= 1.0: warnings_list.append(f"PF {pf:.2f} marginal")
else: issues.append(f"PF {pf:.2f} < 1.0")
if sharpe >= 2.0: passes.append(f"Sharpe {sharpe:.2f} >= 2.0")
elif sharpe >= 1.0: passes.append(f"Sharpe {sharpe:.2f} acceptable")
else: issues.append(f"Sharpe {sharpe:.2f} too low")
if h1_pnl > 0 and h2_pnl > 0: passes.append("Both halves profitable")
else: issues.append("NOT both halves profitable")
if mc_profitable >= 70: passes.append(f"Monte Carlo {mc_profitable:.0f}% profitable")
elif mc_profitable >= 55: warnings_list.append(f"MC {mc_profitable:.0f}% marginal")
else: issues.append(f"MC {mc_profitable:.0f}% too low")
if pnl_2x_slip > 0: passes.append("Survives 2x slippage")
else: issues.append("Dies with 2x slippage")
if pnl_3x_slip > 0: passes.append("Survives 3x slippage")
if not (total_pnl > 0 and pnl_no_top5 < 0): passes.append("Not outlier dependent")
else: issues.append("Outlier dependent")
if profitable_tickers >= len(ticker_pnl) * 0.4: passes.append(f"{profitable_tickers}/{len(ticker_pnl)} tickers profitable")
else: warnings_list.append(f"Only {profitable_tickers}/{len(ticker_pnl)} tickers profitable")
if short_pnl > long_pnl * 5: warnings_list.append(f"Short-heavy: ${short_pnl:+,.0f} vs long ${long_pnl:+,.0f}")

report.append(f"\n  PASSES ({len(passes)}):")
for p in passes: report.append(f"    [PASS] {p}")
if warnings_list:
    report.append(f"\n  WARNINGS ({len(warnings_list)}):")
    for w in warnings_list: report.append(f"    [WARN] {w}")
if issues:
    report.append(f"\n  ISSUES ({len(issues)}):")
    for i in issues: report.append(f"    [FAIL] {i}")

report.append("")
if len(issues) == 0 and len(passes) >= 5:
    report.append("  RECOMMENDATION: PROCEED TO PAPER TRADING (Phase 6)")
    report.append("  Run for minimum 10 trading days at QUARTER SIZE.")
    report.append("  Compare live results to backtest. If win rate, avg trade,")
    report.append("  and drawdown are within 1 std dev of backtest, scale to 50%.")
elif len(issues) == 0:
    report.append("  RECOMMENDATION: MARGINAL — extend backtest period before paper trading")
else:
    report.append("  RECOMMENDATION: DO NOT TRADE LIVE — address issues first")

report.append("")
report.append("  KEY RISK: 96% of PnL comes from shorts. If market enters a")
report.append("  strong bull run, the short edge may disappear. Monitor the")
report.append("  long/short PnL split weekly during paper trading.")
report.append("")
report.append("=" * 72)
report.append("  DISCLAIMER: Not financial advice. Past performance does not")
report.append("  guarantee future results. Trading involves risk of loss.")
report.append("=" * 72)

# ================================================================
# CHARTS
# ================================================================
fig = plt.figure(figsize=(22, 20))
gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)

# 1. Equity curve
ax = fig.add_subplot(gs[0, 0])
ax.plot(cum.values, 'b-', linewidth=1.2)
ax.fill_between(range(len(cum)), cum.values, alpha=0.15, color='blue')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_title('Cumulative Net PnL', fontweight='bold')
ax.set_xlabel('Trade #'); ax.set_ylabel('PnL ($)'); ax.grid(True, alpha=0.3)

# 2. PnL distribution
ax = fig.add_subplot(gs[0, 1])
ax.hist(tl['pnl_net'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='red', linewidth=1)
ax.axvline(x=avg_trade, color='orange', linewidth=1.5, linestyle='--', label=f'Avg: ${avg_trade:+.2f}')
ax.set_title('Trade PnL Distribution', fontweight='bold'); ax.legend()

# 3. Daily PnL
ax = fig.add_subplot(gs[0, 2])
colors = ['green' if v > 0 else 'red' for v in daily_pnl.values]
ax.bar(range(len(daily_pnl)), daily_pnl.values, color=colors, alpha=0.7)
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_title(f'Daily PnL ({winning_days}/{trading_days} winning days)', fontweight='bold')

# 4. By ticker
ax = fig.add_subplot(gs[1, 0])
st = ticker_pnl.sort_values('sum')
colors = ['green' if v > 0 else 'red' for v in st['sum'].values]
ax.barh(st.index, st['sum'].values, color=colors, alpha=0.7)
ax.set_title('PnL by Ticker', fontweight='bold')

# 5. By exit reason
ax = fig.add_subplot(gs[1, 1])
colors = ['green' if v > 0 else 'red' for v in reason_stats['total_pnl'].values]
ax.bar(reason_stats.index, reason_stats['total_pnl'].values, color=colors, alpha=0.7)
ax.set_title('PnL by Exit Reason', fontweight='bold')

# 6. Long vs Short
ax = fig.add_subplot(gs[1, 2])
dir_pnl = {'LONG': long_pnl, 'SHORT': short_pnl}
colors = ['green' if v > 0 else 'red' for v in dir_pnl.values()]
bars = ax.bar(dir_pnl.keys(), dir_pnl.values(), color=colors, alpha=0.7, width=0.4)
ax.set_title('PnL by Direction', fontweight='bold')
for bar, val in zip(bars, dir_pnl.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, f'${val:+,.0f}', ha='center', fontsize=9)

# 7. Monte Carlo
ax = fig.add_subplot(gs[2, 0])
ax.hist(mc, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='red', linewidth=1.5)
ax.axvline(x=total_pnl, color='green', linewidth=1.5, linestyle='--', label=f'Actual: ${total_pnl:+,.0f}')
ax.set_title(f'Monte Carlo: {mc_profitable:.0f}% Profitable', fontweight='bold'); ax.legend()

# 8. By hour
ax = fig.add_subplot(gs[2, 1])
tl['entry_hour'] = pd.to_datetime(tl['entry_time']).dt.hour + pd.to_datetime(tl['entry_time']).dt.minute / 60
tl['hour_bucket'] = tl['entry_hour'].round(1)
hourly = tl.groupby('hour_bucket')['pnl_net'].sum()
colors = ['green' if v > 0 else 'red' for v in hourly.values]
ax.bar(hourly.index, hourly.values, color=colors, alpha=0.7, width=0.15)
ax.set_title('PnL by Entry Time', fontweight='bold'); ax.set_xlabel('Hour (ET)')

# 9. Equity: long vs short separate
ax = fig.add_subplot(gs[2, 2])
long_cum = longs['pnl_net'].cumsum() if len(longs) > 0 else pd.Series([0])
short_cum = shorts['pnl_net'].cumsum() if len(shorts) > 0 else pd.Series([0])
ax.plot(long_cum.values, 'g-', linewidth=1, label=f'Long (${long_pnl:+,.0f})')
ax.plot(short_cum.values, 'r-', linewidth=1, label=f'Short (${short_pnl:+,.0f})')
ax.axhline(y=0, color='black', linewidth=0.5)
ax.set_title('Equity Curve by Direction', fontweight='bold'); ax.legend()

# 10. V1-V4 comparison
ax = fig.add_subplot(gs[3, 0])
versions = ['V1\nBase', 'V2\nShort', 'V3\nTuned', 'V4\nAsym']
pnls = [1144, 2151, 1708, total_pnl]
sharpes = [4.02, 7.19, 8.27, sharpe]
ax.bar(versions, pnls, color=['#4a90d9', '#4a90d9', '#4a90d9', '#2ecc71'], alpha=0.7)
ax.set_title('PnL Progression V1→V4', fontweight='bold'); ax.set_ylabel('Net PnL ($)')

# 11. Drawdown
ax = fig.add_subplot(gs[3, 1])
ax.fill_between(range(len(dd)), dd.values, color='red', alpha=0.3)
ax.plot(dd.values, 'r-', linewidth=0.8)
ax.set_title(f'Drawdown (Max: ${max_dd:,.0f})', fontweight='bold')
ax.set_xlabel('Trade #'); ax.set_ylabel('Drawdown ($)')

# 12. Day of week
ax = fig.add_subplot(gs[3, 2])
tl['dow'] = pd.to_datetime(tl['entry_time']).dt.dayofweek
dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
dow_pnl = tl.groupby('dow')['pnl_net'].sum()
colors = ['green' if v > 0 else 'red' for v in dow_pnl.values]
ax.bar([dow_names.get(d, str(d)) for d in dow_pnl.index], dow_pnl.values, color=colors, alpha=0.7)
ax.set_title('PnL by Day of Week', fontweight='bold')

plt.suptitle('ORB V4 — Asymmetric Direction-Neutral Strategy\n'
             f'Net PnL: ${total_pnl:+,.2f} | Sharpe: {sharpe:.2f} | '
             f'Win Rate: {win_rate:.1f}% | PF: {pf:.2f} | '
             f'Long: ${long_pnl:+,.0f} | Short: ${short_pnl:+,.0f}',
             fontsize=14, fontweight='bold', y=1.01)

plt.savefig(OUTPUT_DIR / 'v4_strategy_report.png', dpi=150, bbox_inches='tight')
plt.close()

text = "\n".join(report)
print(text)
with open(OUTPUT_DIR / 'v4_strategy_report.txt', 'w') as f:
    f.write(text)

print(f"\nCharts: {OUTPUT_DIR / 'v4_strategy_report.png'}")
print(f"Report: {OUTPUT_DIR / 'v4_strategy_report.txt'}")
