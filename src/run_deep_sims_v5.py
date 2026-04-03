"""
Deep Strategy Simulations V5 — Refinements based on V4 chart patterns.

NEW VARIABLES:
  1. Day-of-week filter (Wed/Thu only, or exclude Mon/Fri)
  2. Tighter entry window (10:10-10:50 vs 10:00-11:30)
  3. Ticker exclusion (remove CRM, UNH, GOOGL)
  4. Longs only on gap-up days, shorts only on gap-down days
  5. Momentum confirmation (last N bars in breakout direction)
  6. Previous day range filter (narrow prev day = better breakouts)
  7. OR width relative to prev day range (coiled spring detection)
  8. Higher risk on high-conviction setups
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from intraday_data import load_dataset, DEFAULT_UNIVERSE, add_opening_range
from orb_strategy import compute_orb_features
from backtester_v2 import Backtester

OUTPUT_DIR = Path(__file__).parent.parent / "deep_sim_v5_results"
OUTPUT_DIR.mkdir(exist_ok=True)

EXCLUDE_TICKERS = {"CRM", "UNH", "GOOGL"}


def generate_signals_v5(df, params):
    """V5 signal generation with all refinements."""
    df = df.copy()

    or_minutes = params.get('or_minutes', 12)
    entry_start = params.get('entry_start', 10.0)
    entry_end = params.get('entry_end', 11.5)
    long_target = params.get('long_target', 0.5)
    long_stop = params.get('long_stop', 0.375)
    short_target = params.get('short_target', 1.125)
    short_stop = params.get('short_stop', 0.375)
    vol_thresh = params.get('vol_thresh', 1.0)
    direction = params.get('direction', 'both')

    # New V5 filters
    allowed_days = params.get('allowed_days', None)  # e.g. [1,2,3] for Tue/Wed/Thu
    gap_direction_match = params.get('gap_direction_match', False)  # Long only gap-up, short only gap-down
    momentum_bars = params.get('momentum_bars', 0)  # Require N bars in breakout direction
    min_or_range_pct = params.get('min_or_range_pct', 0.0)
    max_or_vs_prev_range = params.get('max_or_vs_prev_range', 999)  # OR < X% of prev day range = coiled
    min_or_vs_prev_range = params.get('min_or_vs_prev_range', 0)

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

        # Day-of-week filter
        if allowed_days is not None:
            dow = pd.to_datetime(day_df['Date'].iloc[0]).dayofweek
            if dow not in allowed_days:
                continue

        or_high = day_df['or_high'].iloc[0]
        or_low = day_df['or_low'].iloc[0]
        or_range = day_df['or_range'].iloc[0]

        if pd.isna(or_high) or pd.isna(or_low) or or_range <= 0:
            continue

        or_range_pct = or_range / ((or_high + or_low) / 2) * 100
        if or_range_pct < min_or_range_pct:
            continue

        # Previous day range comparison (coiled spring)
        prev_high = day_df['prev_high'].iloc[0] if 'prev_high' in day_df.columns else None
        prev_low = day_df['prev_low'].iloc[0] if 'prev_low' in day_df.columns else None
        if prev_high and prev_low and not pd.isna(prev_high) and not pd.isna(prev_low):
            prev_range = prev_high - prev_low
            if prev_range > 0:
                or_vs_prev = or_range / prev_range
                if or_vs_prev > max_or_vs_prev_range or or_vs_prev < min_or_vs_prev_range:
                    continue

        # Gap direction
        prev_close = day_df['prev_close'].iloc[0] if 'prev_close' in day_df.columns else None
        if prev_close and not pd.isna(prev_close) and prev_close > 0:
            gap_dir = 1 if day_df['Open'].iloc[0] > prev_close else -1
        else:
            gap_dir = 0

        long_triggered = False
        short_triggered = False

        day_indices = list(day_df.index)

        for pos, idx in enumerate(day_indices):
            row = df.loc[idx]
            time = row['time_decimal']

            if time < entry_start or time > entry_end:
                continue

            rel_vol = row.get('rel_volume', 1.0)
            if pd.isna(rel_vol):
                rel_vol = 1.0

            # LONG
            if direction in ('both', 'long') and not long_triggered:
                if row['Close'] > or_high and rel_vol >= vol_thresh:
                    # Gap direction match
                    if gap_direction_match and gap_dir == -1:
                        long_triggered = True
                        continue

                    # Momentum confirmation
                    if momentum_bars > 0 and pos >= momentum_bars:
                        recent = [df.loc[day_indices[pos - j], 'Close'] for j in range(momentum_bars + 1)]
                        if not all(recent[j] > recent[j + 1] for j in range(len(recent) - 1)):
                            continue  # Not all bars moving up

                    entry_price = row['Close']
                    df.loc[idx, 'signal'] = 1
                    df.loc[idx, 'signal_type'] = 'long_breakout'
                    df.loc[idx, 'target_price'] = entry_price + (or_range * long_target)
                    df.loc[idx, 'stop_price'] = entry_price - (or_range * long_stop)
                    df.loc[idx, 'entry_reason'] = 'long'
                    long_triggered = True

            # SHORT
            if direction in ('both', 'short') and not short_triggered:
                if row['Close'] < or_low and rel_vol >= vol_thresh:
                    if gap_direction_match and gap_dir == 1:
                        short_triggered = True
                        continue

                    if momentum_bars > 0 and pos >= momentum_bars:
                        recent = [df.loc[day_indices[pos - j], 'Close'] for j in range(momentum_bars + 1)]
                        if not all(recent[j] < recent[j + 1] for j in range(len(recent) - 1)):
                            continue

                    entry_price = row['Close']
                    df.loc[idx, 'signal'] = -1
                    df.loc[idx, 'signal_type'] = 'short_breakout'
                    df.loc[idx, 'target_price'] = entry_price - (or_range * short_target)
                    df.loc[idx, 'stop_price'] = entry_price + (or_range * short_stop)
                    df.loc[idx, 'entry_reason'] = 'short'
                    short_triggered = True

    return df


def run_sim(data_raw, params, exclude_tickers=None):
    """Run one config."""
    data = {}
    total_signals = 0

    for symbol, df_raw in data_raw.items():
        if exclude_tickers and symbol in exclude_tickers:
            continue
        result = generate_signals_v5(df_raw.copy(), params)
        data[symbol] = result
        total_signals += (result['signal'] != 0).sum()

    if total_signals == 0:
        return None

    bt = Backtester(
        risk_per_trade=params.get('risk_per_trade', 400),
        max_positions=params.get('max_positions', 3),
    )
    trade_log = bt.run(data)

    if len(trade_log) < 10:
        return None

    winners = trade_log[trade_log['pnl_net'] > 0]
    losers = trade_log[trade_log['pnl_net'] <= 0]
    n = len(trade_log)
    total_pnl = trade_log['pnl_net'].sum()
    win_rate = len(winners) / n * 100
    pf = winners['pnl_net'].sum() / abs(losers['pnl_net'].sum()) if len(losers) > 0 and losers['pnl_net'].sum() != 0 else 0

    cum = trade_log['pnl_net'].cumsum()
    max_dd = (cum - cum.cummax()).min()

    trade_log['trade_date'] = pd.to_datetime(trade_log['entry_time']).dt.date
    daily_pnl = trade_log.groupby('trade_date')['pnl_net'].sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if len(daily_pnl) > 1 and daily_pnl.std() > 0 else 0

    target_hits = (trade_log['exit_reason'] == 'target_hit').sum()
    mc = [np.random.choice(trade_log['pnl_net'].values, size=n, replace=True).sum() for _ in range(1000)]
    mc_profitable = (np.array(mc) > 0).mean() * 100

    half = n // 2
    h1 = trade_log.iloc[:half]['pnl_net'].sum()
    h2 = trade_log.iloc[half:]['pnl_net'].sum()

    extra_slip = trade_log['shares'] * 0.01 * 2
    pnl_2x_slip = total_pnl - extra_slip.sum()
    top5 = trade_log.nlargest(5, 'pnl_net')['pnl_net'].sum()

    longs = trade_log[trade_log['direction'] == 'LONG']
    shorts = trade_log[trade_log['direction'] == 'SHORT']

    return {
        **{k: v for k, v in params.items() if not callable(v) and k != 'allowed_days'},
        'allowed_days': str(params.get('allowed_days', 'all')),
        'exclude_tickers': bool(exclude_tickers),
        'trades': n, 'long_trades': len(longs), 'short_trades': len(shorts),
        'long_pnl': round(longs['pnl_net'].sum() if len(longs) > 0 else 0, 2),
        'short_pnl': round(shorts['pnl_net'].sum() if len(shorts) > 0 else 0, 2),
        'win_rate': round(win_rate, 1), 'net_pnl': round(total_pnl, 2),
        'avg_trade': round(total_pnl / n, 2), 'profit_factor': round(pf, 2),
        'max_drawdown': round(max_dd, 2), 'sharpe': round(sharpe, 2),
        'target_hit_pct': round(target_hits / n * 100, 1),
        'mc_profitable': round(mc_profitable, 1),
        'h1_pnl': round(h1, 2), 'h2_pnl': round(h2, 2),
        'both_halves': h1 > 0 and h2 > 0,
        'calmar': round(total_pnl / abs(max_dd), 2) if max_dd != 0 else 0,
        'pnl_2x_slip': round(pnl_2x_slip, 2),
        'survives_2x_slip': pnl_2x_slip > 0,
        'not_outlier_dep': not (total_pnl > 0 and (total_pnl - top5) < 0),
    }


def main():
    print("=" * 70)
    print("DEEP STRATEGY SIMULATIONS V5 — CHART-DRIVEN REFINEMENTS")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    data_raw = load_dataset(DEFAULT_UNIVERSE)
    print(f"Loaded {len(data_raw)} symbols")
    for s in data_raw:
        data_raw[s] = compute_orb_features(data_raw[s])

    configs = []

    # V4 best base params
    BASE = {
        'or_minutes': 12, 'long_target': 0.5, 'long_stop': 0.375,
        'short_target': 1.125, 'short_stop': 0.375, 'vol_thresh': 1.0,
        'entry_start': 10.0, 'entry_end': 11.5, 'direction': 'both',
        'risk_per_trade': 400, 'max_positions': 3,
    }

    # GROUP 1: Day-of-week filters
    day_combos = [
        None,               # All days (baseline)
        [1, 2, 3],          # Tue/Wed/Thu
        [2, 3],             # Wed/Thu only
        [0, 1, 2, 3],       # Mon-Thu (skip Fri)
        [1, 2, 3, 4],       # Tue-Fri (skip Mon)
        [0, 2, 3, 4],       # Skip Tue
        [0, 1, 3, 4],       # Skip Wed
    ]
    for days in day_combos:
        for or_min in [12, 15]:
            for s_tgt in [1.0, 1.125, 1.25]:
                for s_stp in [0.35, 0.375]:
                    configs.append({**BASE, 'or_minutes': or_min,
                                    'short_target': s_tgt, 'short_stop': s_stp,
                                    'allowed_days': days, 'momentum_bars': 0,
                                    'gap_direction_match': False,
                                    'min_or_range_pct': 0, 'max_or_vs_prev_range': 999,
                                    'min_or_vs_prev_range': 0, 'group': '1_dow'})

    # GROUP 2: Tighter entry windows
    for start in [10.0, 10.083, 10.167, 10.25]:  # 10:00, 10:05, 10:10, 10:15
        for end in [10.667, 10.833, 11.0, 11.25, 11.5]:  # 10:40, 10:50, 11:00, 11:15, 11:30
            if end <= start:
                continue
            for or_min in [12, 15]:
                configs.append({**BASE, 'or_minutes': or_min,
                                'entry_start': start, 'entry_end': end,
                                'allowed_days': None, 'momentum_bars': 0,
                                'gap_direction_match': False,
                                'min_or_range_pct': 0, 'max_or_vs_prev_range': 999,
                                'min_or_vs_prev_range': 0, 'group': '2_window'})

    # GROUP 3: Ticker exclusion
    for or_min in [12, 15]:
        for s_tgt in [1.0, 1.125]:
            for s_stp in [0.35, 0.375]:
                configs.append({**BASE, 'or_minutes': or_min,
                                'short_target': s_tgt, 'short_stop': s_stp,
                                'allowed_days': None, 'momentum_bars': 0,
                                'gap_direction_match': False,
                                'min_or_range_pct': 0, 'max_or_vs_prev_range': 999,
                                'min_or_vs_prev_range': 0,
                                'group': '3_exclude_tickers', '_exclude': True})

    # GROUP 4: Gap direction match (long only on gap-up, short only on gap-down)
    for or_min in [12, 15]:
        for s_tgt in [0.875, 1.0, 1.125, 1.25]:
            for s_stp in [0.35, 0.375, 0.5]:
                for l_tgt in [0.5, 0.75, 1.0]:
                    for l_stp in [0.375, 0.5]:
                        if s_stp > s_tgt or l_stp > l_tgt:
                            continue
                        configs.append({**BASE, 'or_minutes': or_min,
                                        'long_target': l_tgt, 'long_stop': l_stp,
                                        'short_target': s_tgt, 'short_stop': s_stp,
                                        'allowed_days': None, 'momentum_bars': 0,
                                        'gap_direction_match': True,
                                        'min_or_range_pct': 0, 'max_or_vs_prev_range': 999,
                                        'min_or_vs_prev_range': 0, 'group': '4_gap_match'})

    # GROUP 5: Momentum confirmation (2 or 3 bars)
    for or_min in [12, 15]:
        for mom in [2, 3]:
            for s_tgt in [1.0, 1.125, 1.25]:
                for s_stp in [0.35, 0.375]:
                    configs.append({**BASE, 'or_minutes': or_min,
                                    'short_target': s_tgt, 'short_stop': s_stp,
                                    'allowed_days': None, 'momentum_bars': mom,
                                    'gap_direction_match': False,
                                    'min_or_range_pct': 0, 'max_or_vs_prev_range': 999,
                                    'min_or_vs_prev_range': 0, 'group': '5_momentum'})

    # GROUP 6: Coiled spring (OR < X% of prev day range)
    for or_min in [12, 15]:
        for max_ratio in [0.3, 0.4, 0.5, 0.6, 0.75]:
            for s_tgt in [1.0, 1.125, 1.25]:
                configs.append({**BASE, 'or_minutes': or_min,
                                'short_target': s_tgt, 'short_stop': 0.375,
                                'allowed_days': None, 'momentum_bars': 0,
                                'gap_direction_match': False,
                                'min_or_range_pct': 0,
                                'max_or_vs_prev_range': max_ratio, 'min_or_vs_prev_range': 0,
                                'group': '6_coiled'})

    # GROUP 7: Best combos — stack multiple filters
    for days in [None, [1, 2, 3]]:
        for or_min in [12, 15]:
            for gap_match in [True, False]:
                for mom in [0, 2]:
                    for max_ratio in [999, 0.5]:
                        for s_tgt in [1.0, 1.125]:
                            configs.append({**BASE, 'or_minutes': or_min,
                                            'short_target': s_tgt, 'short_stop': 0.375,
                                            'entry_start': 10.0, 'entry_end': 11.25,
                                            'allowed_days': days, 'momentum_bars': mom,
                                            'gap_direction_match': gap_match,
                                            'min_or_range_pct': 0,
                                            'max_or_vs_prev_range': max_ratio,
                                            'min_or_vs_prev_range': 0,
                                            'group': '7_combo'})

    # GROUP 8: Position sizing on promising combos
    for risk in [300, 400, 500, 600]:
        for max_pos in [2, 3, 4]:
            configs.append({**BASE, 'risk_per_trade': risk, 'max_positions': max_pos,
                            'allowed_days': [1, 2, 3],  # Best DOW from visual
                            'momentum_bars': 0, 'gap_direction_match': False,
                            'min_or_range_pct': 0, 'max_or_vs_prev_range': 999,
                            'min_or_vs_prev_range': 0, 'group': '8_sizing'})

    total = len(configs)
    print(f"Total configurations: {total}")

    from collections import Counter
    groups = Counter(c.get('group', '?') for c in configs)
    for g, cnt in sorted(groups.items()):
        print(f"  {g}: {cnt}")
    print()

    results = []
    best_pnl = -999999
    best_sharpe = -999999

    for i, params in enumerate(configs):
        exclude = EXCLUDE_TICKERS if params.pop('_exclude', False) else None
        result = run_sim(data_raw, params, exclude_tickers=exclude)

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
    df.to_csv(OUTPUT_DIR / 'all_results_v5.csv', index=False)

    viable = df[(df['net_pnl'] > 0) & (df['mc_profitable'] > 55)]
    robust = viable[viable['both_halves'] == True]
    bulletproof = robust[(robust['survives_2x_slip'] == True) & (robust['not_outlier_dep'] == True)]

    print(f"\nViable: {len(viable)}/{len(df)}")
    print(f"Robust: {len(robust)}/{len(df)}")
    print(f"Bulletproof: {len(bulletproof)}/{len(df)}")

    # Best by group
    print(f"\n{'='*70}")
    print("BEST BY GROUP")
    print(f"{'='*70}")
    for group in sorted(groups.keys()):
        grp_bp = bulletproof[bulletproof['group'] == group] if 'group' in bulletproof.columns else pd.DataFrame()
        if len(grp_bp) > 0:
            best = grp_bp.iloc[0]
            print(f"  {group:20s}: PnL ${best['net_pnl']:>+8,.0f} | Sharpe {best['sharpe']:>6.2f} | "
                  f"PF {best['profit_factor']:>5.2f} | Trades {best['trades']:>4} | "
                  f"DD ${best['max_drawdown']:>7,.0f} | MC {best['mc_profitable']:>3.0f}%")
        else:
            grp_v = viable[viable['group'] == group] if 'group' in viable.columns else pd.DataFrame()
            if len(grp_v) > 0:
                best = grp_v.iloc[0]
                print(f"  {group:20s}: PnL ${best['net_pnl']:>+8,.0f} | Sharpe {best['sharpe']:>6.2f} | "
                      f"(viable only)")
            else:
                print(f"  {group:20s}: No profitable configs")

    # Top 20
    top = bulletproof.head(20) if len(bulletproof) >= 10 else robust.head(20)

    print(f"\n{'='*140}")
    print("TOP 20 BULLETPROOF")
    print(f"{'='*140}")
    print(f"{'OR':>4} {'Days':>12} {'Win':>10} {'Gap':>4} {'Mom':>4} {'Coil':>5} {'ExTk':>5} "
          f"{'Grp':>12} {'Trd':>5} {'W%':>5} {'PnL':>10} {'PF':>5} {'Shrp':>6} {'MC%':>5} "
          f"{'DD':>9} {'BH':>4} {'LPnL':>9} {'SPnL':>9}")
    print("-" * 140)

    for _, row in top.iterrows():
        win = f"{row.get('entry_start',10):.1f}-{row.get('entry_end',11.5):.1f}"
        days = str(row.get('allowed_days', 'all'))[:12]
        gap = "Y" if row.get('gap_direction_match', False) else "N"
        mom = str(int(row.get('momentum_bars', 0)))
        coil = f"{row.get('max_or_vs_prev_range', 999):.1f}" if row.get('max_or_vs_prev_range', 999) < 999 else "N"
        extk = "Y" if row.get('exclude_tickers', False) else "N"
        grp = str(row.get('group', ''))[:12]
        bh = "YES" if row.get('both_halves', False) else "no"

        print(f"{int(row['or_minutes']):>3}m {days:>12} {win:>10} {gap:>4} {mom:>4} {coil:>5} {extk:>5} "
              f"{grp:>12} {row['trades']:>5} {row['win_rate']:>4.1f}% "
              f"${row['net_pnl']:>+9,.0f} {row['profit_factor']:>5.2f} "
              f"{row['sharpe']:>6.2f} {row['mc_profitable']:>4.0f}% ${row['max_drawdown']:>8,.0f} "
              f"{bh:>4} ${row['long_pnl']:>+8,.0f} ${row['short_pnl']:>+8,.0f}")

    print(f"{'='*140}")

    # Overall best
    pool = bulletproof if len(bulletproof) > 0 else robust
    best = pool.iloc[0]

    print(f"\nV5 BEST:")
    for k, v in best.items():
        if k not in ('group',) and v != 0 and not (isinstance(v, float) and abs(v) < 0.001):
            print(f"  {k}: {v}")

    # Comparison
    print(f"\n{'='*70}")
    print(f"PROGRESSION V1 → V5")
    print(f"{'='*70}")
    print(f"  {'Ver':<6} {'PnL':>10} {'Sharpe':>8} {'PF':>6} {'DD':>10} {'MC':>6}")
    print(f"  {'V1':<6} {'$+1,144':>10} {'4.02':>8} {'1.36':>6} {'$-354':>10} {'95%':>6}")
    print(f"  {'V2':<6} {'$+2,151':>10} {'7.19':>8} {'1.89':>6} {'$-410':>10} {'100%':>6}")
    print(f"  {'V3':<6} {'$+1,708':>10} {'8.27':>8} {'2.00':>6} {'$-218':>10} {'100%':>6}")
    print(f"  {'V4':<6} {'$+1,738':>10} {'6.53':>8} {'1.41':>6} {'$-640':>10} {'99%':>6}")
    print(f"  {'V5':<6} ${best['net_pnl']:>+9,.0f} {best['sharpe']:>8.2f} {best['profit_factor']:>6.2f} "
          f"${best['max_drawdown']:>9,.0f} {best['mc_profitable']:>5.0f}%")

    if len(bulletproof) > 0:
        bulletproof.to_csv(OUTPUT_DIR / 'bulletproof_v5.csv', index=False)

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return df


if __name__ == "__main__":
    df = main()
