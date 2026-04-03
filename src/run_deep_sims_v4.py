"""
Tests whether the V3 short bias holds up or if both directions can be made to work.
V3's 41-day short-only results are likely biased — V4 runs the same params on both sides,
adds asymmetric target/stop for longs vs shorts, and introduces gap-adaptive direction filtering.
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

OUTPUT_DIR = Path(__file__).parent.parent / "deep_sim_v4_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_signals_v4(df, params):
    """ORB signals with separate target/stop for longs and shorts, plus optional gap-adaptive direction gating."""
    df = df.copy()

    or_minutes = params.get('or_minutes', 15)
    entry_start = params.get('entry_start', 10.0)
    entry_end = params.get('entry_end', 11.5)
    vol_thresh = params.get('vol_thresh', 1.0)
    direction = params.get('direction', 'both')

    # Asymmetric params
    long_target = params.get('long_target', params.get('target_mult', 1.0))
    long_stop = params.get('long_stop', params.get('stop_mult', 0.375))
    short_target = params.get('short_target', params.get('target_mult', 1.0))
    short_stop = params.get('short_stop', params.get('stop_mult', 0.375))

    min_or_range_pct = params.get('min_or_range_pct', 0.0)
    gap_adaptive = params.get('gap_adaptive', False)  # Only trade in direction of gap
    gap_contra = params.get('gap_contra', False)  # Only trade AGAINST gap (fade)

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

        or_range_pct = or_range / ((or_high + or_low) / 2) * 100
        if or_range_pct < min_or_range_pct:
            continue

        # Gap direction
        prev_close = day_df['prev_close'].iloc[0] if 'prev_close' in day_df.columns else None
        if prev_close and not pd.isna(prev_close) and prev_close > 0:
            gap_dir = 1 if day_df['Open'].iloc[0] > prev_close else -1
        else:
            gap_dir = 0

        # Determine allowed directions for today
        allow_long = direction in ('both', 'long')
        allow_short = direction in ('both', 'short')

        if gap_adaptive and gap_dir != 0:
            allow_long = allow_long and gap_dir == 1
            allow_short = allow_short and gap_dir == -1
        if gap_contra and gap_dir != 0:
            allow_long = allow_long and gap_dir == -1
            allow_short = allow_short and gap_dir == 1

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

            # LONG
            if allow_long and not long_triggered and row['Close'] > or_high and rel_vol >= vol_thresh:
                entry_price = row['Close']
                df.loc[idx, 'signal'] = 1
                df.loc[idx, 'signal_type'] = 'long_breakout'
                df.loc[idx, 'target_price'] = entry_price + (or_range * long_target)
                df.loc[idx, 'stop_price'] = entry_price - (or_range * long_stop)
                df.loc[idx, 'entry_reason'] = 'long'
                long_triggered = True

            # SHORT
            if allow_short and not short_triggered and row['Close'] < or_low and rel_vol >= vol_thresh:
                entry_price = row['Close']
                df.loc[idx, 'signal'] = -1
                df.loc[idx, 'signal_type'] = 'short_breakout'
                df.loc[idx, 'target_price'] = entry_price - (or_range * short_target)
                df.loc[idx, 'stop_price'] = entry_price + (or_range * short_stop)
                df.loc[idx, 'entry_reason'] = 'short'
                short_triggered = True

    return df


def run_sim(data_raw, params):
    """Run one config and return a stats dict with long/short split, slippage sensitivity, and outlier check."""
    data = {}
    total_signals = 0

    for symbol, df_raw in data_raw.items():
        result = generate_signals_v4(df_raw.copy(), params)
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

    costs = trade_log['slippage_cost'].sum() + trade_log['commission_cost'].sum()
    extra_slip = trade_log['shares'] * 0.01 * 2
    pnl_2x_slip = total_pnl - extra_slip.sum()
    top5 = trade_log.nlargest(5, 'pnl_net')['pnl_net'].sum()

    # Direction breakdown
    longs = trade_log[trade_log['direction'] == 'LONG']
    shorts = trade_log[trade_log['direction'] == 'SHORT']
    long_pnl = longs['pnl_net'].sum() if len(longs) > 0 else 0
    short_pnl = shorts['pnl_net'].sum() if len(shorts) > 0 else 0

    return {
        **{k: v for k, v in params.items() if not callable(v)},
        'signals': total_signals,
        'trades': n,
        'long_trades': len(longs),
        'short_trades': len(shorts),
        'long_pnl': round(long_pnl, 2),
        'short_pnl': round(short_pnl, 2),
        'win_rate': round(win_rate, 1),
        'net_pnl': round(total_pnl, 2),
        'avg_trade': round(total_pnl / n, 2),
        'profit_factor': round(pf, 2),
        'max_drawdown': round(max_dd, 2),
        'sharpe': round(sharpe, 2),
        'target_hit_pct': round(target_hits / n * 100, 1),
        'mc_profitable': round(mc_profitable, 1),
        'h1_pnl': round(h1, 2),
        'h2_pnl': round(h2, 2),
        'both_halves': h1 > 0 and h2 > 0,
        'calmar': round(total_pnl / abs(max_dd), 2) if max_dd != 0 else 0,
        'pnl_2x_slip': round(pnl_2x_slip, 2),
        'survives_2x_slip': pnl_2x_slip > 0,
        'not_outlier_dep': not (total_pnl > 0 and (total_pnl - top5) < 0),
    }


def main():
    """Run the V4 grid, print best by direction type and group, and output the overall bulletproof winner."""
    print("=" * 70)
    print("DEEP STRATEGY SIMULATIONS V4 — DIRECTION-NEUTRAL")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    data_raw = load_dataset(DEFAULT_UNIVERSE)
    print(f"Loaded {len(data_raw)} symbols")
    for s in data_raw:
        data_raw[s] = compute_orb_features(data_raw[s])

    configs = []

    # ============================================================
    # GROUP 1: Both directions with V3 best params (baseline)
    # ============================================================
    for or_min in [12, 15, 18]:
        for tgt in [0.75, 0.875, 1.0, 1.125, 1.25]:
            for stp in [0.3, 0.35, 0.375, 0.4, 0.5]:
                if stp > tgt:
                    continue
                for direction in ['both', 'short', 'long']:
                    for entry_end in [11.25, 11.5]:
                        configs.append({
                            'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                            'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': entry_end,
                            'direction': direction, 'risk_per_trade': 400, 'max_positions': 3,
                            'min_or_range_pct': 0, 'gap_adaptive': False, 'gap_contra': False,
                            'group': '1_baseline',
                        })

    # ============================================================
    # GROUP 2: Asymmetric params (tight stop on shorts, wider on longs)
    # ============================================================
    for or_min in [12, 15]:
        for s_tgt in [0.875, 1.0, 1.125]:
            for s_stp in [0.3, 0.375]:
                for l_tgt in [0.5, 0.75, 1.0]:
                    for l_stp in [0.375, 0.5, 0.625]:
                        if l_stp > l_tgt or s_stp > s_tgt:
                            continue
                        configs.append({
                            'or_minutes': or_min,
                            'long_target': l_tgt, 'long_stop': l_stp,
                            'short_target': s_tgt, 'short_stop': s_stp,
                            'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': 11.5,
                            'direction': 'both', 'risk_per_trade': 400, 'max_positions': 3,
                            'min_or_range_pct': 0, 'gap_adaptive': False, 'gap_contra': False,
                            'group': '2_asymmetric',
                        })

    # ============================================================
    # GROUP 3: Gap-adaptive (trade in direction of overnight gap)
    # ============================================================
    for or_min in [12, 15, 18]:
        for tgt in [0.75, 1.0, 1.25]:
            for stp in [0.35, 0.375, 0.5]:
                if stp > tgt:
                    continue
                configs.append({
                    'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                    'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': 11.5,
                    'direction': 'both', 'risk_per_trade': 400, 'max_positions': 3,
                    'min_or_range_pct': 0, 'gap_adaptive': True, 'gap_contra': False,
                    'group': '3_gap_follow',
                })

    # ============================================================
    # GROUP 4: Gap-contra (fade the gap — trade against overnight direction)
    # ============================================================
    for or_min in [12, 15, 18]:
        for tgt in [0.5, 0.75, 1.0]:
            for stp in [0.35, 0.375, 0.5]:
                if stp > tgt:
                    continue
                configs.append({
                    'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                    'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': 11.5,
                    'direction': 'both', 'risk_per_trade': 400, 'max_positions': 3,
                    'min_or_range_pct': 0, 'gap_adaptive': False, 'gap_contra': True,
                    'group': '4_gap_fade',
                })

    # ============================================================
    # GROUP 5: Both directions + OR range filter
    # ============================================================
    for or_min in [12, 15]:
        for tgt in [0.875, 1.0, 1.125]:
            for stp in [0.35, 0.375]:
                for min_or in [0.3, 0.5, 0.75]:
                    configs.append({
                        'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                        'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': 11.5,
                        'direction': 'both', 'risk_per_trade': 400, 'max_positions': 3,
                        'min_or_range_pct': min_or, 'gap_adaptive': False, 'gap_contra': False,
                        'group': '5_or_filter',
                    })

    # ============================================================
    # GROUP 6: Position sizing on best both-direction setups
    # ============================================================
    for risk in [200, 300, 400, 500, 600]:
        for max_pos in [2, 3, 4, 5]:
            configs.append({
                'or_minutes': 15, 'target_mult': 1.0, 'stop_mult': 0.375,
                'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': 11.5,
                'direction': 'both', 'risk_per_trade': risk, 'max_positions': max_pos,
                'min_or_range_pct': 0, 'gap_adaptive': False, 'gap_contra': False,
                'group': '6_sizing',
            })

    # ============================================================
    # GROUP 7: Asymmetric + gap adaptive combo
    # ============================================================
    for or_min in [12, 15]:
        for s_tgt in [1.0, 1.125]:
            for s_stp in [0.35, 0.375]:
                for l_tgt in [0.75, 1.0]:
                    for l_stp in [0.5, 0.625]:
                        if l_stp > l_tgt or s_stp > s_tgt:
                            continue
                        for gap in [True, False]:
                            configs.append({
                                'or_minutes': or_min,
                                'long_target': l_tgt, 'long_stop': l_stp,
                                'short_target': s_tgt, 'short_stop': s_stp,
                                'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': 11.5,
                                'direction': 'both', 'risk_per_trade': 400, 'max_positions': 3,
                                'min_or_range_pct': 0, 'gap_adaptive': gap, 'gap_contra': False,
                                'group': '7_asym_gap',
                            })

    total = len(configs)
    print(f"Total configurations: {total}")

    # Count by group
    from collections import Counter
    groups = Counter(c.get('group', '?') for c in configs)
    for g, cnt in sorted(groups.items()):
        print(f"  {g}: {cnt}")
    print()

    results = []
    best_pnl = -999999
    best_sharpe = -999999

    for i, params in enumerate(configs):
        result = run_sim(data_raw, params)

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
    df.to_csv(OUTPUT_DIR / 'all_results_v4.csv', index=False)

    viable = df[(df['net_pnl'] > 0) & (df['mc_profitable'] > 55)]
    robust = viable[viable['both_halves'] == True]
    bulletproof = robust[(robust['survives_2x_slip'] == True) & (robust['not_outlier_dep'] == True)]

    print(f"\nViable: {len(viable)}/{len(df)}")
    print(f"Robust: {len(robust)}/{len(df)}")
    print(f"Bulletproof: {len(bulletproof)}/{len(df)}")

    # Show best by direction type
    print(f"\n{'='*70}")
    print("BEST BY DIRECTION TYPE")
    print(f"{'='*70}")

    for dir_type in ['both', 'short', 'long']:
        dir_df = bulletproof[bulletproof['direction'] == dir_type] if len(bulletproof) > 0 else robust[robust['direction'] == dir_type]
        if len(dir_df) == 0:
            dir_df = viable[viable['direction'] == dir_type]
        if len(dir_df) == 0:
            print(f"\n  {dir_type.upper()}: No profitable configs found")
            continue
        best = dir_df.iloc[0]
        print(f"\n  {dir_type.upper()} BEST:")
        print(f"    PnL: ${best['net_pnl']:+,.2f} | Sharpe: {best['sharpe']:.2f} | PF: {best['profit_factor']:.2f}")
        print(f"    Trades: {best['trades']} (L:{best['long_trades']}, S:{best['short_trades']}) | "
              f"Win%: {best['win_rate']:.1f}%")
        print(f"    Long PnL: ${best['long_pnl']:+,.2f} | Short PnL: ${best['short_pnl']:+,.2f}")
        print(f"    Max DD: ${best['max_drawdown']:,.2f} | MC: {best['mc_profitable']:.0f}% | "
              f"Both halves: {best['both_halves']}")

    # Show best by group
    print(f"\n{'='*70}")
    print("BEST BY STRATEGY GROUP")
    print(f"{'='*70}")

    for group in sorted(groups.keys()):
        grp_df = df[df['group'] == group]
        grp_profitable = grp_df[grp_df['net_pnl'] > 0]
        if len(grp_profitable) == 0:
            print(f"  {group}: No profitable configs")
            continue
        best = grp_profitable.sort_values('sharpe', ascending=False).iloc[0]
        print(f"  {group}: PnL ${best['net_pnl']:+,.0f} | Sharpe {best['sharpe']:.2f} | "
              f"PF {best['profit_factor']:.2f} | Dir: {best['direction']} | "
              f"Trades: {best['trades']}")

    # Top 25 overall
    top = bulletproof.head(25) if len(bulletproof) >= 10 else robust.head(25)

    print(f"\n{'='*160}")
    print("TOP 25 BULLETPROOF BY SHARPE")
    print(f"{'='*160}")
    print(f"{'OR':>4} {'Dir':>6} {'LTgt':>5} {'LStp':>5} {'STgt':>5} {'SStp':>5} {'Win':>10} "
          f"{'Gap':>5} {'Grp':>12} {'Trd':>5} {'L/S':>7} {'W%':>5} {'PnL':>10} {'PF':>5} "
          f"{'Shrp':>6} {'MC%':>5} {'DD':>9} {'BH':>4} {'LPnL':>9} {'SPnL':>9}")
    print("-" * 160)

    for _, row in top.iterrows():
        win = f"{row.get('entry_start',10):.0f}-{row.get('entry_end',11.5):.1f}"
        d = str(row.get('direction', 'both'))[:5]
        lt = row.get('long_target', row.get('target_mult', 0))
        ls = row.get('long_stop', row.get('stop_mult', 0))
        st = row.get('short_target', row.get('target_mult', 0))
        ss = row.get('short_stop', row.get('stop_mult', 0))
        gap = "adpt" if row.get('gap_adaptive', False) else ("fade" if row.get('gap_contra', False) else "N")
        grp = str(row.get('group', ''))[:12]
        bh = "YES" if row.get('both_halves', False) else "no"
        ratio = f"{int(row['long_trades'])}/{int(row['short_trades'])}"

        print(f"{int(row['or_minutes']):>3}m {d:>6} {lt:>5.3f} {ls:>5.3f} {st:>5.3f} {ss:>5.3f} "
              f"{win:>10} {gap:>5} {grp:>12} {row['trades']:>5} {ratio:>7} {row['win_rate']:>4.1f}% "
              f"${row['net_pnl']:>+9,.0f} {row['profit_factor']:>5.2f} "
              f"{row['sharpe']:>6.2f} {row['mc_profitable']:>4.0f}% ${row['max_drawdown']:>8,.0f} "
              f"{bh:>4} ${row['long_pnl']:>+8,.0f} ${row['short_pnl']:>+8,.0f}")

    print(f"{'='*160}")

    # Overall best
    pool = bulletproof if len(bulletproof) > 0 else robust
    best = pool.iloc[0]

    print(f"\nOVERALL BEST:")
    print(f"  OR: {best['or_minutes']}min | Direction: {best['direction']}")
    print(f"  Long: target {best.get('long_target', best.get('target_mult',0))}x, stop {best.get('long_stop', best.get('stop_mult',0))}x")
    print(f"  Short: target {best.get('short_target', best.get('target_mult',0))}x, stop {best.get('short_stop', best.get('stop_mult',0))}x")
    print(f"  Gap: {'adaptive' if best.get('gap_adaptive') else 'contra' if best.get('gap_contra') else 'none'}")
    print(f"  Trades: {best['trades']} (L:{best['long_trades']}, S:{best['short_trades']})")
    print(f"  PnL: ${best['net_pnl']:+,.2f} (Long: ${best['long_pnl']:+,.2f}, Short: ${best['short_pnl']:+,.2f})")
    print(f"  Sharpe: {best['sharpe']} | PF: {best['profit_factor']} | MC: {best['mc_profitable']}%")
    print(f"  Max DD: ${best['max_drawdown']:,.2f} | Both halves: {best['both_halves']}")

    # V3 vs V4
    print(f"\n{'='*70}")
    print(f"V3 BEST (short-only) vs V4 BEST")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {'V3 (short)':>15} {'V4 Best':>15}")
    print(f"  {'Net PnL':<25} {'$+1,708':>15} ${best['net_pnl']:>+14,.2f}")
    print(f"  {'Sharpe':<25} {'8.27':>15} {best['sharpe']:>15.2f}")
    print(f"  {'Profit Factor':<25} {'2.00':>15} {best['profit_factor']:>15.2f}")
    print(f"  {'Max Drawdown':<25} {'$-218':>15} ${best['max_drawdown']:>14,.2f}")
    print(f"  {'Direction':<25} {'short-only':>15} {best['direction']:>15}")
    print(f"  {'Trades':<25} {'110':>15} {best['trades']:>15}")

    # Save
    if len(bulletproof) > 0:
        bulletproof.to_csv(OUTPUT_DIR / 'bulletproof_v4.csv', index=False)
    if len(robust) > 0:
        robust.to_csv(OUTPUT_DIR / 'robust_v4.csv', index=False)

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return df


if __name__ == "__main__":
    df = main()
