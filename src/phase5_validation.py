"""
Phase 5: Statistical Validation

Determines whether the backtest results are real or noise.
Includes: core metrics, distribution analysis, robustness checks,
Monte Carlo simulation, and an honest assessment.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

OUTPUT_DIR = Path(__file__).parent.parent / "phase5_output"


def run_phase5(trade_log: pd.DataFrame, equity_curve: pd.DataFrame = None):
    """Full statistical validation of backtest results."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("PHASE 5: STATISTICAL VALIDATION")
    print("=" * 70)

    tl = trade_log.copy()
    n = len(tl)

    if n == 0:
        print("No trades to analyze.")
        return

    winners = tl[tl['pnl_net'] > 0]
    losers = tl[tl['pnl_net'] <= 0]

    # ================================================================
    # 1. CORE METRICS
    # ================================================================
    print("\n--- 1. CORE METRICS ---\n")

    total_pnl = tl['pnl_net'].sum()
    win_rate = len(winners) / n * 100
    avg_win = winners['pnl_net'].mean() if len(winners) > 0 else 0
    avg_loss = losers['pnl_net'].mean() if len(losers) > 0 else 0
    profit_factor = winners['pnl_net'].sum() / abs(losers['pnl_net'].sum()) if len(losers) > 0 and losers['pnl_net'].sum() != 0 else 0
    avg_trade = tl['pnl_net'].mean()
    total_costs = tl['slippage_cost'].sum() + tl['commission_cost'].sum()

    # Max drawdown from trade-by-trade equity
    cumulative = tl['pnl_net'].cumsum()
    peak = cumulative.cummax()
    drawdown = cumulative - peak
    max_dd = drawdown.min()
    max_dd_idx = drawdown.idxmin()

    # Max consecutive losses
    is_loss = (tl['pnl_net'] <= 0).astype(int)
    consec = is_loss.groupby((is_loss != is_loss.shift()).cumsum()).cumsum()
    max_consec_loss = consec.max()

    # Daily returns for Sharpe/Calmar
    tl['trade_date'] = pd.to_datetime(tl['entry_time']).dt.date
    daily_pnl = tl.groupby('trade_date')['pnl_net'].sum()
    trading_days = len(daily_pnl)

    if trading_days > 1 and daily_pnl.std() > 0:
        sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
    else:
        sharpe = 0

    annualized_return = daily_pnl.sum() / 100_000 * (252 / max(trading_days, 1))
    calmar = annualized_return / (abs(max_dd) / 100_000) if max_dd != 0 else 0

    metrics = {
        'Total trades': n,
        'Winners': f"{len(winners)} ({win_rate:.1f}%)",
        'Losers': f"{len(losers)} ({100-win_rate:.1f}%)",
        'Net PnL': f"${total_pnl:+,.2f}",
        'Total costs': f"${total_costs:,.2f}",
        'Avg trade (net)': f"${avg_trade:+,.2f}",
        'Avg winner': f"${avg_win:+,.2f}",
        'Avg loser': f"${avg_loss:+,.2f}",
        'Profit factor': f"{profit_factor:.2f}",
        'Max drawdown': f"${max_dd:,.2f}",
        'Max consecutive losses': int(max_consec_loss),
        'Sharpe ratio (ann.)': f"{sharpe:.2f}",
        'Calmar ratio': f"{calmar:.2f}",
    }

    for k, v in metrics.items():
        print(f"  {k:30s} {v}")

    # ================================================================
    # 2. DISTRIBUTION ANALYSIS
    # ================================================================
    print("\n--- 2. DISTRIBUTION ANALYSIS ---\n")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 2a. PnL histogram
    ax = axes[0, 0]
    ax.hist(tl['pnl_net'], bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linewidth=1)
    ax.axvline(x=avg_trade, color='orange', linewidth=1.5, linestyle='--', label=f'Avg: ${avg_trade:+.2f}')
    ax.set_title('Trade PnL Distribution')
    ax.set_xlabel('Net PnL ($)')
    ax.legend()

    skewness = tl['pnl_net'].skew()
    kurtosis = tl['pnl_net'].kurtosis()
    print(f"  PnL skewness: {skewness:.2f} ({'right-skewed' if skewness > 0 else 'left-skewed'})")
    print(f"  PnL kurtosis: {kurtosis:.2f} ({'fat tails' if kurtosis > 3 else 'normal tails'})")

    # 2b. PnL by time of day
    ax = axes[0, 1]
    tl['entry_hour'] = pd.to_datetime(tl['entry_time']).dt.hour
    hourly_pnl = tl.groupby('entry_hour')['pnl_net'].sum()
    colors = ['green' if v > 0 else 'red' for v in hourly_pnl.values]
    ax.bar(hourly_pnl.index, hourly_pnl.values, color=colors, alpha=0.7)
    ax.set_title('PnL by Entry Hour')
    ax.set_xlabel('Hour (ET)')
    ax.set_ylabel('Total PnL ($)')

    print(f"  PnL by hour: {dict(hourly_pnl.round(2))}")

    # 2c. PnL by day of week
    ax = axes[0, 2]
    tl['day_of_week'] = pd.to_datetime(tl['entry_time']).dt.dayofweek
    dow_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
    dow_pnl = tl.groupby('day_of_week')['pnl_net'].sum()
    colors = ['green' if v > 0 else 'red' for v in dow_pnl.values]
    ax.bar([dow_names.get(d, str(d)) for d in dow_pnl.index], dow_pnl.values, color=colors, alpha=0.7)
    ax.set_title('PnL by Day of Week')
    ax.set_ylabel('Total PnL ($)')

    # 2d. PnL by ticker
    ax = axes[1, 0]
    ticker_pnl = tl.groupby('symbol')['pnl_net'].sum().sort_values()
    colors = ['green' if v > 0 else 'red' for v in ticker_pnl.values]
    ax.barh(ticker_pnl.index, ticker_pnl.values, color=colors, alpha=0.7)
    ax.set_title('PnL by Ticker')
    ax.set_xlabel('Total PnL ($)')

    # Check concentration
    top_ticker_pnl = ticker_pnl.max()
    bottom_ticker_pnl = ticker_pnl.min()
    profitable_tickers = (ticker_pnl > 0).sum()
    print(f"  Profitable tickers: {profitable_tickers}/{len(ticker_pnl)}")
    print(f"  Best ticker: {ticker_pnl.idxmax()} (${top_ticker_pnl:+,.2f})")
    print(f"  Worst ticker: {ticker_pnl.idxmin()} (${bottom_ticker_pnl:+,.2f})")

    # 2e. PnL by exit reason
    ax = axes[1, 1]
    reason_pnl = tl.groupby('exit_reason')['pnl_net'].agg(['sum', 'count', 'mean'])
    colors = ['green' if v > 0 else 'red' for v in reason_pnl['sum'].values]
    ax.bar(reason_pnl.index, reason_pnl['sum'].values, color=colors, alpha=0.7)
    ax.set_title('PnL by Exit Reason')
    ax.set_ylabel('Total PnL ($)')

    # 2f. Equity curve
    ax = axes[1, 2]
    cum_pnl = tl['pnl_net'].cumsum()
    ax.plot(cum_pnl.values, 'b-', linewidth=1)
    ax.fill_between(range(len(cum_pnl)), cum_pnl.values, alpha=0.1, color='blue')
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_title('Cumulative PnL (Trade-by-Trade)')
    ax.set_xlabel('Trade #')
    ax.set_ylabel('Cumulative Net PnL ($)')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'distribution_analysis.png', dpi=150)
    plt.close()
    print(f"\n  Saved distribution_analysis.png")

    # ================================================================
    # 3. ROBUSTNESS CHECKS
    # ================================================================
    print("\n--- 3. ROBUSTNESS CHECKS ---\n")

    # 3a. Split in half
    half = n // 2
    first_half = tl.iloc[:half]
    second_half = tl.iloc[half:]
    pnl_h1 = first_half['pnl_net'].sum()
    pnl_h2 = second_half['pnl_net'].sum()
    wr_h1 = (first_half['pnl_net'] > 0).mean() * 100
    wr_h2 = (second_half['pnl_net'] > 0).mean() * 100
    print(f"  First half:  PnL ${pnl_h1:+,.2f}, Win rate {wr_h1:.1f}% ({len(first_half)} trades)")
    print(f"  Second half: PnL ${pnl_h2:+,.2f}, Win rate {wr_h2:.1f}% ({len(second_half)} trades)")
    both_profitable = pnl_h1 > 0 and pnl_h2 > 0
    print(f"  Both halves profitable: {'YES' if both_profitable else 'NO'}")

    # 3b. Remove best 5 trades
    without_best5 = tl.nsmallest(n - 5, 'pnl_net')  # Everything except top 5
    # Actually: remove top 5
    top5_pnl = tl.nlargest(5, 'pnl_net')['pnl_net'].sum()
    pnl_without_top5 = total_pnl - top5_pnl
    print(f"\n  Top 5 trades contribute: ${top5_pnl:+,.2f}")
    print(f"  PnL without top 5: ${pnl_without_top5:+,.2f}")
    print(f"  Depends on outliers: {'YES (fragile!)' if pnl_without_top5 < 0 and total_pnl > 0 else 'NO'}")

    # 3c. Increase slippage to $0.02
    tl_2x_slip = tl.copy()
    extra_slip = tl_2x_slip['shares'] * 0.01 * 2  # Additional $0.01/share entry+exit
    pnl_2x_slip = total_pnl - extra_slip.sum()
    print(f"\n  With 2x slippage ($0.02/share): ${pnl_2x_slip:+,.2f}")

    # 3d. Increase commission by 50%
    extra_comm = tl['commission_cost'] * 0.5
    pnl_extra_comm = total_pnl - extra_comm.sum()
    print(f"  With 1.5x commission: ${pnl_extra_comm:+,.2f}")

    # 3e. Entry timing sensitivity
    # Simulate +/- random noise on entry price
    np.random.seed(42)
    timing_results = []
    for trial in range(100):
        noise = np.random.uniform(-0.02, 0.02, n)  # +/- 2 cents
        adjusted_pnl = tl['pnl_net'] + (noise * tl['shares'] * np.where(tl['direction'] == 'LONG', -1, 1))
        timing_results.append(adjusted_pnl.sum())

    timing_std = np.std(timing_results)
    timing_mean = np.mean(timing_results)
    print(f"\n  Entry timing sensitivity (100 trials, +/- $0.02):")
    print(f"    Mean PnL: ${timing_mean:+,.2f}")
    print(f"    Std Dev:  ${timing_std:,.2f}")
    print(f"    Range:    ${min(timing_results):+,.2f} to ${max(timing_results):+,.2f}")
    print(f"    Fragile:  {'YES' if timing_std > abs(total_pnl) else 'NO'}")

    # ================================================================
    # 4. MONTE CARLO SIMULATION
    # ================================================================
    print("\n--- 4. MONTE CARLO SIMULATION ---\n")

    trade_pnls = tl['pnl_net'].values
    n_simulations = 10_000

    mc_final_equity = []
    mc_max_drawdown = []

    for _ in range(n_simulations):
        shuffled = np.random.choice(trade_pnls, size=len(trade_pnls), replace=True)
        cumsum = np.cumsum(shuffled)
        mc_final_equity.append(cumsum[-1])

        peak = np.maximum.accumulate(cumsum)
        dd = cumsum - peak
        mc_max_drawdown.append(dd.min())

    mc_final = np.array(mc_final_equity)
    mc_dd = np.array(mc_max_drawdown)

    print(f"  Final equity distribution (10,000 simulations):")
    print(f"    Mean:   ${np.mean(mc_final):+,.2f}")
    print(f"    Median: ${np.median(mc_final):+,.2f}")
    print(f"    5th %%:  ${np.percentile(mc_final, 5):+,.2f}")
    print(f"    95th %%: ${np.percentile(mc_final, 95):+,.2f}")
    print(f"    %% profitable: {(mc_final > 0).mean()*100:.1f}%")

    print(f"\n  Max drawdown distribution:")
    print(f"    Mean:   ${np.mean(mc_dd):,.2f}")
    print(f"    Median: ${np.median(mc_dd):,.2f}")
    print(f"    95th %%: ${np.percentile(mc_dd, 5):,.2f} (plan for this)")

    # Plot Monte Carlo
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(mc_final, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=0, color='red', linewidth=1.5)
    ax.axvline(x=total_pnl, color='green', linewidth=1.5, linestyle='--', label=f'Actual: ${total_pnl:+,.0f}')
    ax.set_title(f'Monte Carlo: Final PnL Distribution\n{(mc_final > 0).mean()*100:.0f}% profitable')
    ax.set_xlabel('Final PnL ($)')
    ax.legend()

    ax = axes[1]
    ax.hist(mc_dd, bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax.axvline(x=max_dd, color='green', linewidth=1.5, linestyle='--', label=f'Actual: ${max_dd:,.0f}')
    ax.axvline(x=np.percentile(mc_dd, 5), color='red', linewidth=1.5, linestyle='--',
               label=f'95th pct: ${np.percentile(mc_dd, 5):,.0f}')
    ax.set_title('Monte Carlo: Max Drawdown Distribution')
    ax.set_xlabel('Max Drawdown ($)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'monte_carlo.png', dpi=150)
    plt.close()
    print(f"\n  Saved monte_carlo.png")

    # ================================================================
    # 5. HONEST ASSESSMENT
    # ================================================================
    print("\n" + "=" * 70)
    print("5. HONEST ASSESSMENT")
    print("=" * 70)

    issues = []
    positives = []

    # Check profitability
    if total_pnl <= 0:
        issues.append(f"Strategy is NET NEGATIVE (${total_pnl:+,.2f}). It does not make money after costs.")
    else:
        positives.append(f"Net profitable: ${total_pnl:+,.2f}")

    # Check profit factor
    if profit_factor < 1.0:
        issues.append(f"Profit factor {profit_factor:.2f} < 1.0. Losers outweigh winners.")
    elif profit_factor < 1.2:
        issues.append(f"Profit factor {profit_factor:.2f} is marginal. Needs > 1.3 for real-world viability.")
    else:
        positives.append(f"Profit factor {profit_factor:.2f} is solid.")

    # Check win rate vs R:R
    if win_rate < 45 and avg_win < abs(avg_loss) * 1.5:
        issues.append(f"Win rate {win_rate:.1f}% is low and average win (${avg_win:+.2f}) "
                      f"doesn't compensate vs avg loss (${avg_loss:+.2f}).")

    # Check half-split
    if not both_profitable:
        issues.append("Strategy is NOT profitable in both halves of the data. Unreliable.")

    # Check outlier dependence
    if total_pnl > 0 and pnl_without_top5 < 0:
        issues.append("Profitability depends on top 5 trades. Very fragile.")

    # Check cost sensitivity
    if total_pnl > 0 and pnl_2x_slip < 0:
        issues.append("Strategy dies with 2x slippage. Execution-sensitive.")

    # Check timing sensitivity
    if timing_std > abs(total_pnl) * 0.5:
        issues.append("High sensitivity to entry timing. Fragile execution dependency.")

    # Check ticker concentration
    if profitable_tickers < len(ticker_pnl) * 0.4:
        issues.append(f"Only {profitable_tickers}/{len(ticker_pnl)} tickers profitable. Not diversified.")

    # Check target hit rate
    target_hits = (tl['exit_reason'] == 'target_hit').sum()
    target_rate = target_hits / n * 100
    if target_rate < 15:
        issues.append(f"Only {target_rate:.0f}% of trades hit target. "
                      f"Target may be too ambitious ({target_hits}/{n} trades).")

    # Check time exits
    time_exits = (tl['exit_reason'] == 'time_exit').sum()
    time_exit_rate = time_exits / n * 100
    if time_exit_rate > 50:
        issues.append(f"{time_exit_rate:.0f}% of trades exit on time ({time_exits}/{n}). "
                      f"Strategy often doesn't reach target or stop within the day.")

    # Monte Carlo
    mc_profitable_pct = (mc_final > 0).mean() * 100
    if mc_profitable_pct < 55:
        issues.append(f"Monte Carlo shows only {mc_profitable_pct:.0f}% chance of profit. "
                      f"Not statistically distinguishable from chance.")
    else:
        positives.append(f"Monte Carlo: {mc_profitable_pct:.0f}% chance of profit.")

    # Check Sharpe
    if sharpe < 0.5:
        issues.append(f"Sharpe ratio {sharpe:.2f} is too low for a daytrading strategy.")

    # Print assessment
    print()
    if positives:
        print("POSITIVES:")
        for p in positives:
            print(f"  + {p}")
    print()
    if issues:
        print("ISSUES:")
        for i in issues:
            print(f"  - {i}")

    print()
    # Overall verdict
    critical_issues = [i for i in issues if any(w in i.lower() for w in
                       ['net negative', 'does not make money', 'not profitable in both',
                        'depends on top 5', 'dies with'])]

    if critical_issues:
        print("VERDICT: DO NOT TRADE THIS STRATEGY LIVE.")
        print("The strategy has fundamental problems that cannot be fixed with parameter tuning.")
        print("Go back to Phase 3 and reconsider the hypothesis.")
    elif len(issues) > 3:
        print("VERDICT: STRATEGY IS MARGINAL.")
        print("It might work but has too many yellow flags. Needs refinement before paper trading.")
    else:
        print("VERDICT: CAUTIOUSLY PROCEED TO PAPER TRADING.")
        print("Results are reasonable but need live validation. Run Phase 6 for 2+ weeks.")

    print("\n" + "=" * 70)

    # Save all results
    results = {
        'metrics': metrics,
        'issues': issues,
        'positives': positives,
        'monte_carlo_profitable_pct': mc_profitable_pct,
    }

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    # Load trade log from Phase 4
    trade_log_path = Path(__file__).parent.parent / "phase4_output" / "trade_log.csv"
    if not trade_log_path.exists():
        print("Run Phase 4 first (backtester_v2.py)")
        sys.exit(1)

    trade_log = pd.read_csv(trade_log_path)
    results = run_phase5(trade_log)
