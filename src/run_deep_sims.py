"""
Deep Strategy Simulations — Sweep across many more variables.

Building on Variant C (morning only) as the base since it had the best Sharpe.
Now testing:
  - OR period: 15min, 30min, 45min
  - Target multipliers: 0.5x, 0.75x, 1.0x, 1.25x, 1.5x, 2.0x
  - Stop multipliers: 0.25x, 0.5x, 0.75x, 1.0x
  - Volume threshold: 1.0x, 1.2x, 1.5x, 2.0x
  - Entry window: 10:00-11:30, 10:00-12:00, 10:00-14:00, 10:30-11:30
  - Risk per trade: $100, $200, $300, $500
  - Position cap: 3, 5, 8 simultaneous
  - With/without trailing stop to breakeven
"""

import sys
import itertools
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from intraday_data import load_dataset, DEFAULT_UNIVERSE, add_opening_range
from orb_strategy import compute_orb_features
from backtester_v2 import Backtester

OUTPUT_DIR = Path(__file__).parent.parent / "deep_sim_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_signals_parametric(df, target_mult, stop_mult, vol_thresh,
                                 entry_start, entry_end, or_minutes=30):
    """Fully parameterized signal generation."""
    df = df.copy()

    # Recompute OR if different period
    if or_minutes != 30:
        # Drop existing OR columns before recomputing
        or_cols = [c for c in df.columns if c.startswith('or_') or c in ('above_or', 'below_or', 'dist_from_or_high', 'dist_from_or_low')]
        df = df.drop(columns=or_cols, errors='ignore')
        df = add_opening_range(df, minutes=or_minutes)

    df['signal'] = 0
    df['signal_type'] = None
    df['target_price'] = np.nan
    df['stop_price'] = np.nan
    df['entry_reason'] = None

    for day in df['trading_day'].unique():
        day_mask = df['trading_day'] == day
        day_df = df[day_mask]

        if len(day_df) < 10:
            continue

        or_high = day_df['or_high'].iloc[0]
        or_low = day_df['or_low'].iloc[0]
        or_range = day_df['or_range'].iloc[0]

        if pd.isna(or_high) or pd.isna(or_low) or or_range <= 0:
            continue

        or_range_pct = or_range / ((or_high + or_low) / 2) * 100
        if or_range_pct < 0.3:
            continue

        long_triggered = False
        short_triggered = False

        for idx in day_df.index:
            row = df.loc[idx]
            time = row['time_decimal']

            if time < entry_start or time > entry_end:
                continue

            rel_vol = row.get('rel_volume', 1.0)
            if pd.isna(rel_vol):
                rel_vol = 1.0

            if not long_triggered and row['Close'] > or_high and rel_vol >= vol_thresh:
                entry_price = row['Close']
                df.loc[idx, 'signal'] = 1
                df.loc[idx, 'signal_type'] = 'long_breakout'
                df.loc[idx, 'target_price'] = entry_price + (or_range * target_mult)
                df.loc[idx, 'stop_price'] = entry_price - (or_range * stop_mult)
                df.loc[idx, 'entry_reason'] = 'long'
                long_triggered = True

            if not short_triggered and row['Close'] < or_low and rel_vol >= vol_thresh:
                entry_price = row['Close']
                df.loc[idx, 'signal'] = -1
                df.loc[idx, 'signal_type'] = 'short_breakout'
                df.loc[idx, 'target_price'] = entry_price - (or_range * target_mult)
                df.loc[idx, 'stop_price'] = entry_price + (or_range * stop_mult)
                df.loc[idx, 'entry_reason'] = 'short'
                short_triggered = True

    return df


def run_single_sim(data_raw, params):
    """Run one parameter combination. Returns stats dict."""
    data = {}
    total_signals = 0

    for symbol, df in data_raw.items():
        result = generate_signals_parametric(
            df.copy(),
            target_mult=params['target_mult'],
            stop_mult=params['stop_mult'],
            vol_thresh=params['vol_thresh'],
            entry_start=params['entry_start'],
            entry_end=params['entry_end'],
            or_minutes=params['or_minutes'],
        )
        data[symbol] = result
        total_signals += (result['signal'] != 0).sum()

    if total_signals == 0:
        return None

    bt = Backtester(
        risk_per_trade=params['risk_per_trade'],
        max_positions=params['max_positions'],
    )
    trade_log = bt.run(data)

    if len(trade_log) == 0:
        return None

    winners = trade_log[trade_log['pnl_net'] > 0]
    losers = trade_log[trade_log['pnl_net'] <= 0]
    total_pnl = trade_log['pnl_net'].sum()
    win_rate = len(winners) / len(trade_log) * 100
    costs = trade_log['slippage_cost'].sum() + trade_log['commission_cost'].sum()

    pf = winners['pnl_net'].sum() / abs(losers['pnl_net'].sum()) if len(losers) > 0 and losers['pnl_net'].sum() != 0 else 0

    cum = trade_log['pnl_net'].cumsum()
    max_dd = (cum - cum.cummax()).min()

    trade_log['trade_date'] = pd.to_datetime(trade_log['entry_time']).dt.date
    daily_pnl = trade_log.groupby('trade_date')['pnl_net'].sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if len(daily_pnl) > 1 and daily_pnl.std() > 0 else 0

    target_hits = (trade_log['exit_reason'] == 'target_hit').sum()

    # Quick Monte Carlo (1000 trials for speed)
    mc_results = [np.random.choice(trade_log['pnl_net'].values, size=len(trade_log), replace=True).sum()
                  for _ in range(1000)]
    mc_profitable = (np.array(mc_results) > 0).mean() * 100

    # Half-split check
    half = len(trade_log) // 2
    h1_pnl = trade_log.iloc[:half]['pnl_net'].sum()
    h2_pnl = trade_log.iloc[half:]['pnl_net'].sum()

    return {
        **params,
        'signals': total_signals,
        'trades': len(trade_log),
        'win_rate': round(win_rate, 1),
        'net_pnl': round(total_pnl, 2),
        'avg_trade': round(total_pnl / len(trade_log), 2),
        'profit_factor': round(pf, 2),
        'max_drawdown': round(max_dd, 2),
        'sharpe': round(sharpe, 2),
        'target_hit_pct': round(target_hits / len(trade_log) * 100, 1),
        'costs': round(costs, 2),
        'mc_profitable': round(mc_profitable, 1),
        'h1_pnl': round(h1_pnl, 2),
        'h2_pnl': round(h2_pnl, 2),
        'both_halves_profitable': h1_pnl > 0 and h2_pnl > 0,
        'calmar': round(total_pnl / abs(max_dd), 2) if max_dd != 0 else 0,
    }


def main():
    print("=" * 70)
    print("DEEP STRATEGY SIMULATIONS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Load and prep data
    data_raw = load_dataset(DEFAULT_UNIVERSE)
    print(f"Loaded {len(data_raw)} symbols")

    for symbol in data_raw:
        data_raw[symbol] = compute_orb_features(data_raw[symbol])

    # Parameter grid
    param_grid = {
        'or_minutes': [15, 30, 45],
        'target_mult': [0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
        'stop_mult': [0.25, 0.5, 0.75, 1.0],
        'vol_thresh': [1.0, 1.2, 1.5, 2.0],
        'entry_start': [10.0],  # Fixed: after OR forms
        'entry_end': [11.5, 12.0, 14.0],
        'risk_per_trade': [200, 400],
        'max_positions': [3, 5],
    }

    # Generate all combinations
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    all_combos = list(itertools.product(*values))
    total = len(all_combos)

    print(f"Total parameter combinations: {total}")
    print(f"Estimated time: ~{total * 0.5 / 60:.0f} minutes\n")

    results = []
    best_pnl = -999999
    best_sharpe = -999999

    for i, combo in enumerate(all_combos):
        params = dict(zip(keys, combo))

        # Skip invalid combos (stop > target makes no sense for R:R)
        if params['stop_mult'] > params['target_mult']:
            continue

        result = run_single_sim(data_raw, params)

        if result is not None:
            results.append(result)

            # Track best
            if result['net_pnl'] > best_pnl:
                best_pnl = result['net_pnl']
            if result['sharpe'] > best_sharpe and result['net_pnl'] > 0:
                best_sharpe = result['sharpe']

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}] Tested, {len(results)} valid, "
                  f"best PnL ${best_pnl:+,.0f}, best Sharpe {best_sharpe:.2f}")

    print(f"\n{'='*70}")
    print(f"COMPLETED: {len(results)} valid configurations tested")
    print(f"{'='*70}")

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('sharpe', ascending=False)
    df.to_csv(OUTPUT_DIR / 'all_sim_results.csv', index=False)

    # Filter: profitable + MC > 55%
    viable = df[(df['net_pnl'] > 0) & (df['mc_profitable'] > 55)]
    print(f"\nViable strategies (profitable + MC>55%): {len(viable)}/{len(df)}")

    # Filter: also both halves profitable
    robust = viable[viable['both_halves_profitable'] == True]
    print(f"Robust strategies (also both halves profitable): {len(robust)}/{len(viable)}")

    if len(robust) > 0:
        top20 = robust.head(20)
    elif len(viable) > 0:
        top20 = viable.head(20)
    else:
        top20 = df.head(20)

    print(f"\n{'='*130}")
    print(f"TOP 20 BY SHARPE")
    print(f"{'='*130}")
    print(f"{'OR':>4} {'Tgt':>5} {'Stop':>5} {'Vol':>5} {'Window':>12} {'Risk':>5} {'Pos':>4} "
          f"{'Trades':>6} {'Win%':>6} {'NetPnL':>10} {'PF':>5} {'Sharpe':>7} {'MC%':>5} "
          f"{'MaxDD':>9} {'BothH':>6} {'AvgTrd':>7}")
    print("-" * 130)

    for _, row in top20.iterrows():
        window = f"{row['entry_start']:.0f}-{row['entry_end']:.1f}"
        both = "YES" if row['both_halves_profitable'] else "no"
        print(f"{row['or_minutes']:>3}m {row['target_mult']:>5.2f} {row['stop_mult']:>5.2f} "
              f"{row['vol_thresh']:>4.1f}x {window:>12} ${row['risk_per_trade']:>4.0f} "
              f"{row['max_positions']:>4} {row['trades']:>6} {row['win_rate']:>5.1f}% "
              f"${row['net_pnl']:>+9,.0f} {row['profit_factor']:>5.2f} {row['sharpe']:>7.2f} "
              f"{row['mc_profitable']:>4.0f}% ${row['max_drawdown']:>8,.0f} {both:>6} "
              f"${row['avg_trade']:>+6.2f}")

    print(f"{'='*130}")

    # Save top results
    if len(robust) > 0:
        robust.to_csv(OUTPUT_DIR / 'robust_strategies.csv', index=False)
        print(f"\nRobust strategies saved to {OUTPUT_DIR / 'robust_strategies.csv'}")

    # Best overall
    if len(robust) > 0:
        best = robust.iloc[0]
        label = "ROBUST BEST"
    elif len(viable) > 0:
        best = viable.iloc[0]
        label = "VIABLE BEST (not both-halves-profitable)"
    else:
        best = df.iloc[0]
        label = "BEST (not profitable overall)"

    print(f"\n{label}:")
    print(f"  OR: {best['or_minutes']}min | Target: {best['target_mult']}x | Stop: {best['stop_mult']}x | "
          f"Vol: {best['vol_thresh']}x")
    print(f"  Window: {best['entry_start']}-{best['entry_end']} | Risk: ${best['risk_per_trade']} | "
          f"Max Positions: {best['max_positions']}")
    print(f"  Trades: {best['trades']} | Win Rate: {best['win_rate']}% | Net PnL: ${best['net_pnl']:+,.2f}")
    print(f"  PF: {best['profit_factor']} | Sharpe: {best['sharpe']} | MC: {best['mc_profitable']}% | "
          f"Max DD: ${best['max_drawdown']:,.2f}")
    print(f"  Both halves profitable: {best['both_halves_profitable']}")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return df


if __name__ == "__main__":
    df = main()
