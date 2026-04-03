"""
Deep Strategy Simulations V2 — Methodical optimization based on V1 findings.

V1 LEARNINGS:
  - Morning only (10:00-11:30) is clearly best
  - 30-min and 15-min OR both work, 45-min is worse
  - Tight targets (0.5-1.0x) beat wide targets
  - 0.5x stop is the sweet spot
  - Shorts outperform longs (+$731 vs +$413)
  - 10:00 and 10:30 AM are best entry times
  - TSLA, PLTR, AMAT are consistent losers
  - AMD, LRCX, WMT, HOOD are top performers

V2 NEW VARIABLES TO TEST:
  1. Finer OR periods: 15, 20, 25, 30 min
  2. Finer target/stop: 0.25x increments
  3. Gap filter: only trade in direction of gap, or skip large gaps
  4. Min OR range filter: skip narrow ORs (low vol days)
  5. Trend filter: only trade if price is on same side of prev close
  6. Ticker filter: exclude worst 5 tickers vs include all
  7. Short-only mode: given shorts outperform
  8. Entry delay: wait 1-2 bars after breakout for confirmation
  9. Wider entry window: 10:00-12:00 vs 10:00-11:30 vs 10:15-11:15
  10. Max trades per day: 1, 2, 3
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

OUTPUT_DIR = Path(__file__).parent.parent / "deep_sim_v2_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Tickers that lost money in V1
LOSER_TICKERS = {"TSLA", "PLTR", "AMAT", "CRM", "GOOGL", "MSFT", "NVDA"}


def generate_signals_v2(df, params):
    """V2 signal generation with all new filters."""
    df = df.copy()

    or_minutes = params.get('or_minutes', 30)
    target_mult = params.get('target_mult', 0.75)
    stop_mult = params.get('stop_mult', 0.5)
    vol_thresh = params.get('vol_thresh', 1.2)
    entry_start = params.get('entry_start', 10.0)
    entry_end = params.get('entry_end', 11.5)
    min_or_range_pct = params.get('min_or_range_pct', 0.0)
    max_gap_pct = params.get('max_gap_pct', 999)
    gap_direction_filter = params.get('gap_direction_filter', False)
    trend_filter = params.get('trend_filter', False)
    direction_filter = params.get('direction_filter', 'both')  # 'both', 'long', 'short'
    entry_delay_bars = params.get('entry_delay_bars', 0)

    # Recompute OR if needed
    if or_minutes != 30:
        or_cols = [c for c in df.columns if c.startswith('or_') or c in
                   ('above_or', 'below_or', 'dist_from_or_high', 'dist_from_or_low')]
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

        mid_price = (or_high + or_low) / 2
        or_range_pct = or_range / mid_price * 100

        # FILTER: Min OR range
        if or_range_pct < min_or_range_pct:
            continue

        # FILTER: Gap size
        prev_close = day_df['prev_close'].iloc[0] if 'prev_close' in day_df.columns else None
        if prev_close and not pd.isna(prev_close) and prev_close > 0:
            gap_pct = abs((day_df['Open'].iloc[0] - prev_close) / prev_close * 100)
            gap_direction = 1 if day_df['Open'].iloc[0] > prev_close else -1
            if gap_pct > max_gap_pct:
                continue
        else:
            gap_direction = 0

        long_triggered = False
        short_triggered = False
        breakout_bar_long = None
        breakout_bar_short = None
        bars_since_long = 0
        bars_since_short = 0

        for idx in day_df.index:
            row = df.loc[idx]
            time = row['time_decimal']

            if time < entry_start or time > entry_end:
                continue

            rel_vol = row.get('rel_volume', 1.0)
            if pd.isna(rel_vol):
                rel_vol = 1.0

            # Track bars since breakout for entry delay
            if breakout_bar_long is not None:
                bars_since_long += 1
            if breakout_bar_short is not None:
                bars_since_short += 1

            # LONG BREAKOUT
            if not long_triggered and row['Close'] > or_high and rel_vol >= vol_thresh:
                if direction_filter == 'short':
                    long_triggered = True  # Skip longs
                    continue

                # Gap direction filter
                if gap_direction_filter and gap_direction == -1:
                    long_triggered = True  # Don't go long on gap-down days
                    continue

                # Trend filter: price should be above prev close
                if trend_filter and prev_close and not pd.isna(prev_close):
                    if row['Close'] < prev_close:
                        continue

                if entry_delay_bars == 0:
                    entry_price = row['Close']
                    df.loc[idx, 'signal'] = 1
                    df.loc[idx, 'signal_type'] = 'long_breakout'
                    df.loc[idx, 'target_price'] = entry_price + (or_range * target_mult)
                    df.loc[idx, 'stop_price'] = entry_price - (or_range * stop_mult)
                    df.loc[idx, 'entry_reason'] = 'long'
                    long_triggered = True
                else:
                    breakout_bar_long = idx
                    bars_since_long = 0

            # Delayed long entry
            if breakout_bar_long is not None and bars_since_long == entry_delay_bars and not long_triggered:
                entry_price = row['Close']
                if entry_price > or_high:  # Still above OR after delay
                    df.loc[idx, 'signal'] = 1
                    df.loc[idx, 'signal_type'] = 'long_breakout'
                    df.loc[idx, 'target_price'] = entry_price + (or_range * target_mult)
                    df.loc[idx, 'stop_price'] = entry_price - (or_range * stop_mult)
                    df.loc[idx, 'entry_reason'] = 'long_delayed'
                long_triggered = True
                breakout_bar_long = None

            # SHORT BREAKOUT
            if not short_triggered and row['Close'] < or_low and rel_vol >= vol_thresh:
                if direction_filter == 'long':
                    short_triggered = True
                    continue

                if gap_direction_filter and gap_direction == 1:
                    short_triggered = True  # Don't short on gap-up days
                    continue

                if trend_filter and prev_close and not pd.isna(prev_close):
                    if row['Close'] > prev_close:
                        continue

                if entry_delay_bars == 0:
                    entry_price = row['Close']
                    df.loc[idx, 'signal'] = -1
                    df.loc[idx, 'signal_type'] = 'short_breakout'
                    df.loc[idx, 'target_price'] = entry_price - (or_range * target_mult)
                    df.loc[idx, 'stop_price'] = entry_price + (or_range * stop_mult)
                    df.loc[idx, 'entry_reason'] = 'short'
                    short_triggered = True
                else:
                    breakout_bar_short = idx
                    bars_since_short = 0

            # Delayed short entry
            if breakout_bar_short is not None and bars_since_short == entry_delay_bars and not short_triggered:
                entry_price = row['Close']
                if entry_price < or_low:
                    df.loc[idx, 'signal'] = -1
                    df.loc[idx, 'signal_type'] = 'short_breakout'
                    df.loc[idx, 'target_price'] = entry_price - (or_range * target_mult)
                    df.loc[idx, 'stop_price'] = entry_price + (or_range * stop_mult)
                    df.loc[idx, 'entry_reason'] = 'short_delayed'
                short_triggered = True
                breakout_bar_short = None

    return df


def run_single_sim(data_raw, params, universe=None):
    """Run one parameter combination."""
    data = {}
    total_signals = 0

    symbols = universe if universe else list(data_raw.keys())

    for symbol in symbols:
        if symbol not in data_raw:
            continue
        result = generate_signals_v2(data_raw[symbol].copy(), params)
        data[symbol] = result
        total_signals += (result['signal'] != 0).sum()

    if total_signals == 0:
        return None

    bt = Backtester(
        risk_per_trade=params.get('risk_per_trade', 400),
        max_positions=params.get('max_positions', 3),
    )
    trade_log = bt.run(data)

    if len(trade_log) < 10:  # Need minimum trades for statistical validity
        return None

    winners = trade_log[trade_log['pnl_net'] > 0]
    losers = trade_log[trade_log['pnl_net'] <= 0]
    total_pnl = trade_log['pnl_net'].sum()
    n = len(trade_log)
    win_rate = len(winners) / n * 100
    costs = trade_log['slippage_cost'].sum() + trade_log['commission_cost'].sum()
    pf = winners['pnl_net'].sum() / abs(losers['pnl_net'].sum()) if len(losers) > 0 and losers['pnl_net'].sum() != 0 else 0

    cum = trade_log['pnl_net'].cumsum()
    max_dd = (cum - cum.cummax()).min()

    trade_log['trade_date'] = pd.to_datetime(trade_log['entry_time']).dt.date
    daily_pnl = trade_log.groupby('trade_date')['pnl_net'].sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if len(daily_pnl) > 1 and daily_pnl.std() > 0 else 0

    target_hits = (trade_log['exit_reason'] == 'target_hit').sum()

    mc_results = [np.random.choice(trade_log['pnl_net'].values, size=n, replace=True).sum()
                  for _ in range(1000)]
    mc_profitable = (np.array(mc_results) > 0).mean() * 100

    half = n // 2
    h1_pnl = trade_log.iloc[:half]['pnl_net'].sum()
    h2_pnl = trade_log.iloc[half:]['pnl_net'].sum()

    # Avg trade must be positive and meaningful
    avg_trade = total_pnl / n

    return {
        **{k: v for k, v in params.items() if not callable(v)},
        'signals': total_signals,
        'trades': n,
        'win_rate': round(win_rate, 1),
        'net_pnl': round(total_pnl, 2),
        'avg_trade': round(avg_trade, 2),
        'profit_factor': round(pf, 2),
        'max_drawdown': round(max_dd, 2),
        'sharpe': round(sharpe, 2),
        'target_hit_pct': round(target_hits / n * 100, 1),
        'costs': round(costs, 2),
        'mc_profitable': round(mc_profitable, 1),
        'h1_pnl': round(h1_pnl, 2),
        'h2_pnl': round(h2_pnl, 2),
        'both_halves': h1_pnl > 0 and h2_pnl > 0,
        'calmar': round(total_pnl / abs(max_dd), 2) if max_dd != 0 else 0,
        'n_symbols': len([s for s in data if (data[s]['signal'] != 0).any()]),
    }


def main():
    print("=" * 70)
    print("DEEP STRATEGY SIMULATIONS V2")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    data_raw = load_dataset(DEFAULT_UNIVERSE)
    print(f"Loaded {len(data_raw)} symbols")
    for s in data_raw:
        data_raw[s] = compute_orb_features(data_raw[s])

    # Filtered universe (exclude worst tickers from V1)
    good_tickers = [s for s in DEFAULT_UNIVERSE if s not in LOSER_TICKERS]

    # ============================================================
    # BUILD PARAMETER GRID — focused around V1 winners
    # ============================================================

    configs = []

    # GROUP 1: Fine-tune the V1 winner with all 20 tickers
    for or_min in [15, 20, 25, 30]:
        for tgt in [0.5, 0.625, 0.75, 0.875, 1.0, 1.25]:
            for stp in [0.375, 0.5, 0.625, 0.75]:
                if stp > tgt:
                    continue
                for vol in [1.0, 1.2, 1.5]:
                    for window_end in [11.25, 11.5, 12.0]:
                        configs.append({
                            'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                            'vol_thresh': vol, 'entry_start': 10.0, 'entry_end': window_end,
                            'risk_per_trade': 400, 'max_positions': 3,
                            'direction_filter': 'both', 'gap_direction_filter': False,
                            'trend_filter': False, 'min_or_range_pct': 0,
                            'entry_delay_bars': 0, 'max_gap_pct': 999,
                            'universe': 'all',
                        })

    # GROUP 2: Same grid but with filtered tickers (remove losers)
    for or_min in [15, 25, 30]:
        for tgt in [0.5, 0.75, 1.0]:
            for stp in [0.375, 0.5, 0.75]:
                if stp > tgt:
                    continue
                for vol in [1.0, 1.2]:
                    configs.append({
                        'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                        'vol_thresh': vol, 'entry_start': 10.0, 'entry_end': 11.5,
                        'risk_per_trade': 400, 'max_positions': 3,
                        'direction_filter': 'both', 'gap_direction_filter': False,
                        'trend_filter': False, 'min_or_range_pct': 0,
                        'entry_delay_bars': 0, 'max_gap_pct': 999,
                        'universe': 'filtered',
                    })

    # GROUP 3: Short-only (since shorts outperformed)
    for or_min in [15, 25, 30]:
        for tgt in [0.5, 0.75, 1.0, 1.25]:
            for stp in [0.375, 0.5, 0.75]:
                if stp > tgt:
                    continue
                for vol in [1.0, 1.2, 1.5]:
                    configs.append({
                        'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                        'vol_thresh': vol, 'entry_start': 10.0, 'entry_end': 11.5,
                        'risk_per_trade': 400, 'max_positions': 3,
                        'direction_filter': 'short', 'gap_direction_filter': False,
                        'trend_filter': False, 'min_or_range_pct': 0,
                        'entry_delay_bars': 0, 'max_gap_pct': 999,
                        'universe': 'all',
                    })

    # GROUP 4: With gap filter + trend filter
    for or_min in [15, 25, 30]:
        for tgt in [0.5, 0.75, 1.0]:
            for stp in [0.375, 0.5]:
                for gap_filter in [True, False]:
                    for trend_filter in [True, False]:
                        if not gap_filter and not trend_filter:
                            continue  # Already covered above
                        for vol in [1.0, 1.2]:
                            configs.append({
                                'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                                'vol_thresh': vol, 'entry_start': 10.0, 'entry_end': 11.5,
                                'risk_per_trade': 400, 'max_positions': 3,
                                'direction_filter': 'both',
                                'gap_direction_filter': gap_filter,
                                'trend_filter': trend_filter,
                                'min_or_range_pct': 0,
                                'entry_delay_bars': 0, 'max_gap_pct': 999,
                                'universe': 'all',
                            })

    # GROUP 5: Min OR range filter (skip low-vol days)
    for or_min in [15, 25, 30]:
        for tgt in [0.5, 0.75, 1.0]:
            for stp in [0.375, 0.5]:
                for min_or in [0.3, 0.5, 0.75, 1.0]:
                    configs.append({
                        'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                        'vol_thresh': 1.2, 'entry_start': 10.0, 'entry_end': 11.5,
                        'risk_per_trade': 400, 'max_positions': 3,
                        'direction_filter': 'both', 'gap_direction_filter': False,
                        'trend_filter': False, 'min_or_range_pct': min_or,
                        'entry_delay_bars': 0, 'max_gap_pct': 999,
                        'universe': 'all',
                    })

    # GROUP 6: Entry delay (confirmation bar)
    for or_min in [15, 25, 30]:
        for tgt in [0.75, 1.0, 1.25]:
            for stp in [0.5, 0.75]:
                for delay in [1, 2]:
                    configs.append({
                        'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                        'vol_thresh': 1.2, 'entry_start': 10.0, 'entry_end': 11.5,
                        'risk_per_trade': 400, 'max_positions': 3,
                        'direction_filter': 'both', 'gap_direction_filter': False,
                        'trend_filter': False, 'min_or_range_pct': 0,
                        'entry_delay_bars': delay, 'max_gap_pct': 999,
                        'universe': 'all',
                    })

    # GROUP 7: Max gap filter (skip large gap days)
    for or_min in [15, 30]:
        for tgt in [0.75, 1.0]:
            for stp in [0.5]:
                for max_gap in [0.5, 1.0, 1.5, 2.0]:
                    configs.append({
                        'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                        'vol_thresh': 1.2, 'entry_start': 10.0, 'entry_end': 11.5,
                        'risk_per_trade': 400, 'max_positions': 3,
                        'direction_filter': 'both', 'gap_direction_filter': False,
                        'trend_filter': False, 'min_or_range_pct': 0,
                        'entry_delay_bars': 0, 'max_gap_pct': max_gap,
                        'universe': 'all',
                    })

    # GROUP 8: Position sizing variants on best setups
    for risk in [200, 300, 500, 600]:
        for max_pos in [2, 3, 5]:
            configs.append({
                'or_minutes': 30, 'target_mult': 0.75, 'stop_mult': 0.5,
                'vol_thresh': 1.2, 'entry_start': 10.0, 'entry_end': 11.5,
                'risk_per_trade': risk, 'max_positions': max_pos,
                'direction_filter': 'both', 'gap_direction_filter': False,
                'trend_filter': False, 'min_or_range_pct': 0,
                'entry_delay_bars': 0, 'max_gap_pct': 999,
                'universe': 'all',
            })

    total = len(configs)
    print(f"Total configurations: {total}")
    print(f"  Group 1 (fine-tune all tickers): ~{sum(1 for c in configs if c['universe']=='all' and c['direction_filter']=='both' and not c['gap_direction_filter'] and not c['trend_filter'] and c['min_or_range_pct']==0 and c['entry_delay_bars']==0 and c['max_gap_pct']==999)}")
    print(f"  Group 2 (filtered tickers): ~{sum(1 for c in configs if c['universe']=='filtered')}")
    print(f"  Group 3 (short-only): ~{sum(1 for c in configs if c['direction_filter']=='short')}")
    print(f"  Group 4 (gap+trend filters): ~{sum(1 for c in configs if c['gap_direction_filter'] or c['trend_filter'])}")
    print(f"  Group 5 (min OR range): ~{sum(1 for c in configs if c['min_or_range_pct'] > 0)}")
    print(f"  Group 6 (entry delay): ~{sum(1 for c in configs if c['entry_delay_bars'] > 0)}")
    print(f"  Group 7 (max gap): ~{sum(1 for c in configs if c['max_gap_pct'] < 999)}")
    print(f"\n")

    results = []
    best_pnl = -999999
    best_sharpe = -999999

    for i, params in enumerate(configs):
        universe = good_tickers if params.get('universe') == 'filtered' else None
        result = run_single_sim(data_raw, params, universe=universe)

        if result is not None:
            results.append(result)
            if result['net_pnl'] > best_pnl:
                best_pnl = result['net_pnl']
            if result['sharpe'] > best_sharpe and result['net_pnl'] > 0:
                best_sharpe = result['sharpe']

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}] {len(results)} valid, "
                  f"best PnL ${best_pnl:+,.0f}, best Sharpe {best_sharpe:.2f}")

    print(f"\n{'='*70}")
    print(f"COMPLETED: {len(results)} valid configurations")
    print(f"{'='*70}")

    df = pd.DataFrame(results)
    df = df.sort_values('sharpe', ascending=False)
    df.to_csv(OUTPUT_DIR / 'all_results_v2.csv', index=False)

    viable = df[(df['net_pnl'] > 0) & (df['mc_profitable'] > 55)]
    robust = viable[viable['both_halves'] == True]

    print(f"\nViable (profitable + MC>55%): {len(viable)}/{len(df)}")
    print(f"Robust (+ both halves): {len(robust)}/{len(df)}")

    # Show top 25
    top = robust.head(25) if len(robust) >= 25 else (viable.head(25) if len(viable) >= 25 else df.head(25))

    print(f"\n{'='*150}")
    print(f"TOP 25 BY SHARPE (robust)")
    print(f"{'='*150}")
    print(f"{'OR':>4} {'Tgt':>5} {'Stp':>5} {'Vol':>5} {'Win':>10} {'Dir':>6} {'Gap':>4} {'Trnd':>5} "
          f"{'MinOR':>5} {'Dly':>4} {'Univ':>6} {'Trd':>5} {'W%':>5} {'PnL':>10} {'PF':>5} "
          f"{'Shrp':>6} {'MC%':>5} {'DD':>9} {'BH':>4} {'Avg':>7}")
    print("-" * 150)

    for _, row in top.iterrows():
        win = f"{row.get('entry_start',10):.0f}-{row.get('entry_end',11.5):.1f}"
        d = str(row.get('direction_filter', 'both'))[:5]
        gap = "Y" if row.get('gap_direction_filter', False) else "N"
        trnd = "Y" if row.get('trend_filter', False) else "N"
        mor = f"{row.get('min_or_range_pct', 0):.1f}"
        dly = str(int(row.get('entry_delay_bars', 0)))
        univ = str(row.get('universe', 'all'))[:5]
        bh = "YES" if row.get('both_halves', False) else "no"

        print(f"{int(row['or_minutes']):>3}m {row['target_mult']:>5.3f} {row['stop_mult']:>5.3f} "
              f"{row['vol_thresh']:>4.1f}x {win:>10} {d:>6} {gap:>4} {trnd:>5} "
              f"{mor:>5} {dly:>4} {univ:>6} {row['trades']:>5} {row['win_rate']:>4.1f}% "
              f"${row['net_pnl']:>+9,.0f} {row['profit_factor']:>5.2f} "
              f"{row['sharpe']:>6.2f} {row['mc_profitable']:>4.0f}% ${row['max_drawdown']:>8,.0f} "
              f"{bh:>4} ${row['avg_trade']:>+6.2f}")

    print(f"{'='*150}")

    # Save robust
    if len(robust) > 0:
        robust.to_csv(OUTPUT_DIR / 'robust_v2.csv', index=False)

    # Best overall
    if len(robust) > 0:
        best = robust.iloc[0]
        label = "ROBUST BEST"
    elif len(viable) > 0:
        best = viable.iloc[0]
        label = "VIABLE BEST"
    else:
        best = df.iloc[0]
        label = "BEST (not profitable)"

    print(f"\n{label}:")
    print(f"  OR: {best['or_minutes']}min | Target: {best['target_mult']}x | Stop: {best['stop_mult']}x | "
          f"Vol: {best['vol_thresh']}x")
    print(f"  Window: {best['entry_start']}-{best['entry_end']} | Dir: {best.get('direction_filter','both')} | "
          f"Universe: {best.get('universe','all')}")
    print(f"  Gap filter: {best.get('gap_direction_filter',False)} | Trend filter: {best.get('trend_filter',False)} | "
          f"Min OR: {best.get('min_or_range_pct',0)}% | Delay: {best.get('entry_delay_bars',0)} bars")
    print(f"  Trades: {best['trades']} | Win%: {best['win_rate']}% | PnL: ${best['net_pnl']:+,.2f}")
    print(f"  PF: {best['profit_factor']} | Sharpe: {best['sharpe']} | MC: {best['mc_profitable']}%")
    print(f"  Max DD: ${best['max_drawdown']:,.2f} | Calmar: {best['calmar']}")
    print(f"  Both halves: {best['both_halves']}")

    # Compare V1 best vs V2 best
    print(f"\n{'='*70}")
    print(f"V1 BEST vs V2 BEST COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {'V1 Best':>15} {'V2 Best':>15}")
    print(f"  {'Net PnL':<25} {'$+1,144':>15} ${best['net_pnl']:>+14,.2f}")
    print(f"  {'Sharpe':<25} {'4.02':>15} {best['sharpe']:>15.2f}")
    print(f"  {'Win Rate':<25} {'50.7%':>15} {best['win_rate']:>14.1f}%")
    print(f"  {'Profit Factor':<25} {'1.36':>15} {best['profit_factor']:>15.2f}")
    print(f"  {'Max Drawdown':<25} {'$-354':>15} ${best['max_drawdown']:>14,.2f}")
    print(f"  {'MC Profitable':<25} {'95%':>15} {best['mc_profitable']:>14.0f}%")

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to {OUTPUT_DIR}")

    return df


if __name__ == "__main__":
    df = main()
