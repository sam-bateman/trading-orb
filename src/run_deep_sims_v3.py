"""
Ultra-fine tuning around V2's short-only 15-min OR discovery.
V2 established the core setup. V3 tightens the knobs: sub-0.1x stop increments,
OR period from 10-20 min, breakout strength filter, VWAP confirmation,
volume spike on the entry bar, and OR range bounds to skip extreme days.
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

OUTPUT_DIR = Path(__file__).parent.parent / "deep_sim_v3_results"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_signals_v3(df, params):
    """ORB signal generation with breakout strength filter, VWAP confirmation, OR range bounds, and volume spike gate."""
    df = df.copy()

    or_minutes = params.get('or_minutes', 15)
    target_mult = params.get('target_mult', 1.0)
    stop_mult = params.get('stop_mult', 0.375)
    vol_thresh = params.get('vol_thresh', 1.0)
    entry_start = params.get('entry_start', 10.0)
    entry_end = params.get('entry_end', 11.5)
    direction = params.get('direction', 'short')
    min_breakout_pct = params.get('min_breakout_pct', 0.0)  # Min % beyond OR level
    min_or_range_pct = params.get('min_or_range_pct', 0.0)
    max_or_range_pct = params.get('max_or_range_pct', 999.0)
    require_below_vwap = params.get('require_below_vwap', False)  # For shorts
    vol_spike_mult = params.get('vol_spike_mult', 0.0)  # 0 = disabled, else bar vol > X * 5-bar avg
    allow_reentry = params.get('allow_reentry', False)

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

    # Precompute 5-bar volume average for spike detection
    if vol_spike_mult > 0:
        df['vol_5bar_avg'] = df['Volume'].rolling(5).mean()

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

        mid = (or_high + or_low) / 2
        or_range_pct = or_range / mid * 100

        # OR range filters
        if or_range_pct < min_or_range_pct or or_range_pct > max_or_range_pct:
            continue

        short_triggered = False
        long_triggered = False

        for idx in day_df.index:
            row = df.loc[idx]
            time = row['time_decimal']

            if time < entry_start or time > entry_end:
                continue

            rel_vol = row.get('rel_volume', 1.0)
            if pd.isna(rel_vol):
                rel_vol = 1.0

            # Volume spike check
            if vol_spike_mult > 0:
                vol_avg = row.get('vol_5bar_avg', 0)
                if pd.isna(vol_avg) or vol_avg <= 0:
                    continue
                if row['Volume'] < vol_avg * vol_spike_mult:
                    continue

            # SHORT BREAKOUT
            if direction in ('short', 'both') and (not short_triggered or allow_reentry):
                if row['Close'] < or_low and rel_vol >= vol_thresh:
                    # Breakout strength filter
                    breakout_dist = (or_low - row['Close']) / or_low * 100
                    if breakout_dist < min_breakout_pct:
                        continue

                    # VWAP confirmation
                    if require_below_vwap and 'vwap' in df.columns:
                        vwap = row.get('vwap', 0)
                        if not pd.isna(vwap) and row['Close'] >= vwap:
                            continue

                    entry_price = row['Close']
                    df.loc[idx, 'signal'] = -1
                    df.loc[idx, 'signal_type'] = 'short_breakout'
                    df.loc[idx, 'target_price'] = entry_price - (or_range * target_mult)
                    df.loc[idx, 'stop_price'] = entry_price + (or_range * stop_mult)
                    df.loc[idx, 'entry_reason'] = 'short'
                    short_triggered = True

            # LONG BREAKOUT
            if direction in ('long', 'both') and (not long_triggered or allow_reentry):
                if row['Close'] > or_high and rel_vol >= vol_thresh:
                    breakout_dist = (row['Close'] - or_high) / or_high * 100
                    if breakout_dist < min_breakout_pct:
                        continue

                    if require_below_vwap and 'vwap' in df.columns:
                        # For longs, require above VWAP
                        vwap = row.get('vwap', 0)
                        if not pd.isna(vwap) and row['Close'] <= vwap:
                            continue

                    entry_price = row['Close']
                    df.loc[idx, 'signal'] = 1
                    df.loc[idx, 'signal_type'] = 'long_breakout'
                    df.loc[idx, 'target_price'] = entry_price + (or_range * target_mult)
                    df.loc[idx, 'stop_price'] = entry_price - (or_range * stop_mult)
                    df.loc[idx, 'entry_reason'] = 'long'
                    long_triggered = True

    return df


def run_sim(data_raw, params, universe=None):
    """Run one config and return a stats dict with slippage sensitivity and outlier dependency fields."""
    data = {}
    total_signals = 0
    symbols = universe or list(data_raw.keys())

    for symbol in symbols:
        if symbol not in data_raw:
            continue
        result = generate_signals_v3(data_raw[symbol].copy(), params)
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

    # Extra: cost drag, slippage test
    costs = trade_log['slippage_cost'].sum() + trade_log['commission_cost'].sum()
    extra_slip = trade_log['shares'] * 0.01 * 2
    pnl_2x_slip = total_pnl - extra_slip.sum()

    # Top 5 dependency
    top5 = trade_log.nlargest(5, 'pnl_net')['pnl_net'].sum()
    pnl_no_top5 = total_pnl - top5

    return {
        **{k: v for k, v in params.items() if not callable(v)},
        'signals': total_signals,
        'trades': n,
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
        'costs': round(costs, 2),
        'pnl_2x_slip': round(pnl_2x_slip, 2),
        'pnl_no_top5': round(pnl_no_top5, 2),
        'survives_2x_slip': pnl_2x_slip > 0,
        'not_outlier_dep': not (total_pnl > 0 and pnl_no_top5 < 0),
    }


def main():
    """Run the V3 grid and rank by bulletproof criteria: profitable, both halves, 2x slippage, no outlier dependency."""
    print("=" * 70)
    print("DEEP STRATEGY SIMULATIONS V3")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    data_raw = load_dataset(DEFAULT_UNIVERSE)
    print(f"Loaded {len(data_raw)} symbols")
    for s in data_raw:
        data_raw[s] = compute_orb_features(data_raw[s])

    # Tickers that were good for shorts in V2
    short_good = [s for s in DEFAULT_UNIVERSE if s not in {"GOOGL", "MSFT", "NVDA", "JPM", "XOM"}]

    configs = []

    # GROUP 1: Ultra-fine tune the V2 winner (short-only, 15-min OR)
    for or_min in [10, 12, 15, 18, 20]:
        for tgt in [0.75, 0.875, 1.0, 1.125, 1.25, 1.375, 1.5]:
            for stp in [0.25, 0.3, 0.35, 0.375, 0.4, 0.45]:
                if stp > tgt:
                    continue
                for entry_end in [11.0, 11.25, 11.5]:
                    configs.append({
                        'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                        'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': entry_end,
                        'direction': 'short', 'risk_per_trade': 400, 'max_positions': 3,
                        'min_breakout_pct': 0, 'min_or_range_pct': 0, 'max_or_range_pct': 999,
                        'require_below_vwap': False, 'vol_spike_mult': 0, 'allow_reentry': False,
                        'universe': 'all',
                    })

    # GROUP 2: Breakout strength filter (require > 0.05% or 0.1% beyond OR)
    for or_min in [12, 15, 18]:
        for tgt in [0.875, 1.0, 1.125]:
            for stp in [0.3, 0.375]:
                for bp in [0.05, 0.1, 0.15, 0.2]:
                    configs.append({
                        'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                        'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': 11.5,
                        'direction': 'short', 'risk_per_trade': 400, 'max_positions': 3,
                        'min_breakout_pct': bp, 'min_or_range_pct': 0, 'max_or_range_pct': 999,
                        'require_below_vwap': False, 'vol_spike_mult': 0, 'allow_reentry': False,
                        'universe': 'all',
                    })

    # GROUP 3: VWAP confirmation (price must be below VWAP for shorts)
    for or_min in [12, 15, 18]:
        for tgt in [0.875, 1.0, 1.125]:
            for stp in [0.3, 0.375]:
                configs.append({
                    'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                    'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': 11.5,
                    'direction': 'short', 'risk_per_trade': 400, 'max_positions': 3,
                    'min_breakout_pct': 0, 'min_or_range_pct': 0, 'max_or_range_pct': 999,
                    'require_below_vwap': True, 'vol_spike_mult': 0, 'allow_reentry': False,
                    'universe': 'all',
                })

    # GROUP 4: Volume spike on breakout bar
    for or_min in [12, 15, 18]:
        for tgt in [0.875, 1.0, 1.125]:
            for stp in [0.3, 0.375]:
                for vs in [1.5, 2.0, 2.5]:
                    configs.append({
                        'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                        'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': 11.5,
                        'direction': 'short', 'risk_per_trade': 400, 'max_positions': 3,
                        'min_breakout_pct': 0, 'min_or_range_pct': 0, 'max_or_range_pct': 999,
                        'require_below_vwap': False, 'vol_spike_mult': vs, 'allow_reentry': False,
                        'universe': 'all',
                    })

    # GROUP 5: OR range filters (skip very narrow or very wide ORs)
    for or_min in [12, 15, 18]:
        for tgt in [0.875, 1.0, 1.125]:
            for stp in [0.3, 0.375]:
                for min_or in [0.3, 0.5, 0.75]:
                    for max_or in [2.0, 3.0, 5.0]:
                        configs.append({
                            'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                            'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': 11.5,
                            'direction': 'short', 'risk_per_trade': 400, 'max_positions': 3,
                            'min_breakout_pct': 0, 'min_or_range_pct': min_or, 'max_or_range_pct': max_or,
                            'require_below_vwap': False, 'vol_spike_mult': 0, 'allow_reentry': False,
                            'universe': 'all',
                        })

    # GROUP 6: Filtered tickers for shorts
    for or_min in [12, 15, 18]:
        for tgt in [0.875, 1.0, 1.125, 1.25]:
            for stp in [0.3, 0.375, 0.4]:
                configs.append({
                    'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                    'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': 11.5,
                    'direction': 'short', 'risk_per_trade': 400, 'max_positions': 3,
                    'min_breakout_pct': 0, 'min_or_range_pct': 0, 'max_or_range_pct': 999,
                    'require_below_vwap': False, 'vol_spike_mult': 0, 'allow_reentry': False,
                    'universe': 'short_good',
                })

    # GROUP 7: Entry window fine-tune
    for entry_start in [9.833, 10.0, 10.083, 10.25]:  # 9:50, 10:00, 10:05, 10:15
        for entry_end in [11.0, 11.25, 11.5, 11.75]:
            for tgt in [0.875, 1.0, 1.125]:
                configs.append({
                    'or_minutes': 15, 'target_mult': tgt, 'stop_mult': 0.375,
                    'vol_thresh': 1.0, 'entry_start': entry_start, 'entry_end': entry_end,
                    'direction': 'short', 'risk_per_trade': 400, 'max_positions': 3,
                    'min_breakout_pct': 0, 'min_or_range_pct': 0, 'max_or_range_pct': 999,
                    'require_below_vwap': False, 'vol_spike_mult': 0, 'allow_reentry': False,
                    'universe': 'all',
                })

    # GROUP 8: Position sizing + max positions on best base
    for risk in [300, 400, 500, 600, 800]:
        for max_pos in [2, 3, 4, 5]:
            configs.append({
                'or_minutes': 15, 'target_mult': 1.0, 'stop_mult': 0.375,
                'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': 11.5,
                'direction': 'short', 'risk_per_trade': risk, 'max_positions': max_pos,
                'min_breakout_pct': 0, 'min_or_range_pct': 0, 'max_or_range_pct': 999,
                'require_below_vwap': False, 'vol_spike_mult': 0, 'allow_reentry': False,
                'universe': 'all',
            })

    # GROUP 9: Best combos — combine multiple filters
    for or_min in [12, 15]:
        for tgt in [1.0, 1.125]:
            for stp in [0.35, 0.375]:
                for vwap in [True, False]:
                    for bp in [0, 0.1]:
                        for min_or in [0, 0.5]:
                            configs.append({
                                'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                                'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': 11.5,
                                'direction': 'short', 'risk_per_trade': 400, 'max_positions': 3,
                                'min_breakout_pct': bp, 'min_or_range_pct': min_or, 'max_or_range_pct': 999,
                                'require_below_vwap': vwap, 'vol_spike_mult': 0, 'allow_reentry': False,
                                'universe': 'all',
                            })

    total = len(configs)
    print(f"Total configurations: {total}\n")

    results = []
    best_pnl = -999999
    best_sharpe = -999999

    for i, params in enumerate(configs):
        universe = short_good if params.get('universe') == 'short_good' else None
        result = run_sim(data_raw, params, universe=universe)

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
    df.to_csv(OUTPUT_DIR / 'all_results_v3.csv', index=False)

    viable = df[(df['net_pnl'] > 0) & (df['mc_profitable'] > 55)]
    robust = viable[viable['both_halves'] == True]
    bulletproof = robust[(robust['survives_2x_slip'] == True) & (robust['not_outlier_dep'] == True)]

    print(f"\nViable (profit + MC>55%): {len(viable)}/{len(df)}")
    print(f"Robust (+ both halves): {len(robust)}/{len(df)}")
    print(f"Bulletproof (+ 2x slip + no outlier dep): {len(bulletproof)}/{len(df)}")

    top = bulletproof.head(25) if len(bulletproof) >= 10 else robust.head(25) if len(robust) >= 10 else df.head(25)
    top_label = "BULLETPROOF" if len(bulletproof) >= 10 else "ROBUST"

    print(f"\n{'='*160}")
    print(f"TOP 25 {top_label} BY SHARPE")
    print(f"{'='*160}")
    print(f"{'OR':>4} {'Tgt':>6} {'Stp':>6} {'Win':>10} {'BkPct':>5} {'VWAP':>5} {'VSp':>4} "
          f"{'MinOR':>5} {'MaxOR':>5} {'Univ':>5} {'Trd':>5} {'W%':>5} {'PnL':>10} {'PF':>5} "
          f"{'Shrp':>6} {'MC%':>5} {'DD':>9} {'BH':>4} {'2xS':>4} {'!OL':>4} {'Avg':>7}")
    print("-" * 160)

    for _, row in top.iterrows():
        win = f"{row.get('entry_start',10):.1f}-{row.get('entry_end',11.5):.1f}"
        vw = "Y" if row.get('require_below_vwap', False) else "N"
        vs = f"{row.get('vol_spike_mult',0):.0f}" if row.get('vol_spike_mult',0) > 0 else "N"
        bp = f"{row.get('min_breakout_pct',0):.2f}" if row.get('min_breakout_pct',0) > 0 else "0"
        mor = f"{row.get('min_or_range_pct',0):.1f}"
        mxor = f"{row.get('max_or_range_pct',999):.0f}" if row.get('max_or_range_pct',999) < 999 else "any"
        univ = str(row.get('universe', 'all'))[:5]
        bh = "YES" if row.get('both_halves', False) else "no"
        s2 = "YES" if row.get('survives_2x_slip', False) else "no"
        ol = "YES" if row.get('not_outlier_dep', False) else "no"

        print(f"{int(row['or_minutes']):>3}m {row['target_mult']:>6.3f} {row['stop_mult']:>6.3f} "
              f"{win:>10} {bp:>5} {vw:>5} {vs:>4} "
              f"{mor:>5} {mxor:>5} {univ:>5} {row['trades']:>5} {row['win_rate']:>4.1f}% "
              f"${row['net_pnl']:>+9,.0f} {row['profit_factor']:>5.2f} "
              f"{row['sharpe']:>6.2f} {row['mc_profitable']:>4.0f}% ${row['max_drawdown']:>8,.0f} "
              f"{bh:>4} {s2:>4} {ol:>4} ${row['avg_trade']:>+6.2f}")

    print(f"{'='*160}")

    # Best
    pool = bulletproof if len(bulletproof) > 0 else robust if len(robust) > 0 else df
    best = pool.iloc[0]
    label = "BULLETPROOF" if len(bulletproof) > 0 else "ROBUST" if len(robust) > 0 else "BEST"

    print(f"\n{label} BEST:")
    print(f"  OR: {best['or_minutes']}min | Target: {best['target_mult']}x | Stop: {best['stop_mult']}x")
    print(f"  Window: {best['entry_start']}-{best['entry_end']} | Direction: {best.get('direction','short')}")
    print(f"  VWAP req: {best.get('require_below_vwap',False)} | Breakout %: {best.get('min_breakout_pct',0)}")
    print(f"  Min OR: {best.get('min_or_range_pct',0)}% | Vol spike: {best.get('vol_spike_mult',0)}")
    print(f"  Trades: {best['trades']} | Win%: {best['win_rate']}% | PnL: ${best['net_pnl']:+,.2f}")
    print(f"  PF: {best['profit_factor']} | Sharpe: {best['sharpe']} | MC: {best['mc_profitable']}%")
    print(f"  Max DD: ${best['max_drawdown']:,.2f} | Calmar: {best['calmar']}")
    print(f"  Both halves: {best['both_halves']} | 2x slip: {best.get('survives_2x_slip',False)} | "
          f"No outlier: {best.get('not_outlier_dep',False)}")

    # V2 vs V3
    print(f"\n{'='*70}")
    print(f"V2 BEST vs V3 BEST")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {'V2 Best':>15} {'V3 Best':>15}")
    print(f"  {'Net PnL':<25} {'$+2,151':>15} ${best['net_pnl']:>+14,.2f}")
    print(f"  {'Sharpe':<25} {'7.19':>15} {best['sharpe']:>15.2f}")
    print(f"  {'Profit Factor':<25} {'1.89':>15} {best['profit_factor']:>15.2f}")
    print(f"  {'Max Drawdown':<25} {'$-410':>15} ${best['max_drawdown']:>14,.2f}")
    print(f"  {'MC Profitable':<25} {'100%':>15} {best['mc_profitable']:>14.0f}%")
    print(f"  {'Avg Trade':<25} {'$+14.54':>15} ${best['avg_trade']:>+14.2f}")

    if len(bulletproof) > 0:
        bulletproof.to_csv(OUTPUT_DIR / 'bulletproof_v3.csv', index=False)
    if len(robust) > 0:
        robust.to_csv(OUTPUT_DIR / 'robust_v3.csv', index=False)

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return df


if __name__ == "__main__":
    df = main()
