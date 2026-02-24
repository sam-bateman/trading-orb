"""
12-Month Optimization — Find what actually works over a full year.

Key changes from V1-V5:
  - Optimize ON 12-month data (not 41 days)
  - Walk-forward: optimize on months 1-6, validate on months 7-12
  - Add volatility regime filter (only trade when ATR is elevated)
  - Add previous day range filter (breakouts work better after tight days)
  - Test short-only seriously (longs lost money over 12 months)
  - Focus on reducing outlier dependency
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
from backtester_v2 import Backtester

OUTPUT_DIR = Path(__file__).parent.parent / "optimization_12m_output"
OUTPUT_DIR.mkdir(exist_ok=True)


def add_volatility_features(df):
    """Add vol regime features for filtering."""
    df = df.copy()

    # 5-day ATR as % of price
    if 'prev_high' in df.columns and 'prev_low' in df.columns and 'prev_close' in df.columns:
        # Use daily high-low from intraday data
        daily_range = df.groupby('trading_day').agg(
            day_high=('High', 'max'), day_low=('Low', 'min'), day_close=('Close', 'last')
        ).reset_index()
        daily_range['day_range_pct'] = (daily_range['day_high'] - daily_range['day_low']) / daily_range['day_close'] * 100
        daily_range['avg_range_5d'] = daily_range['day_range_pct'].rolling(5).mean()
        daily_range['avg_range_20d'] = daily_range['day_range_pct'].rolling(20).mean()
        daily_range['vol_ratio'] = daily_range['avg_range_5d'] / (daily_range['avg_range_20d'] + 1e-10)
        # Prev day range for "coiled spring"
        daily_range['prev_day_range_pct'] = daily_range['day_range_pct'].shift(1)

        df = df.merge(daily_range[['trading_day', 'avg_range_5d', 'avg_range_20d',
                                    'vol_ratio', 'prev_day_range_pct']],
                      on='trading_day', how='left')
    else:
        df['avg_range_5d'] = 0
        df['avg_range_20d'] = 0
        df['vol_ratio'] = 1.0
        df['prev_day_range_pct'] = 0

    return df


def generate_signals_12m(df, params):
    """Signal generation with volatility and quality filters."""
    df = df.copy()

    or_minutes = params.get('or_minutes', 12)
    target_mult = params.get('target_mult', 1.0)
    stop_mult = params.get('stop_mult', 0.375)
    vol_thresh = params.get('vol_thresh', 1.0)
    entry_start = params.get('entry_start', 10.0)
    entry_end = params.get('entry_end', 11.5)
    direction = params.get('direction', 'short')

    # Volatility filters
    min_vol_ratio = params.get('min_vol_ratio', 0)  # Recent vol > X * 20d avg
    max_vol_ratio = params.get('max_vol_ratio', 999)
    min_prev_day_range = params.get('min_prev_day_range', 0)  # Prev day range > X%
    max_prev_day_range = params.get('max_prev_day_range', 999)  # Skip if prev day was too wild
    min_or_range_pct = params.get('min_or_range_pct', 0)

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

        # Volatility regime filter
        vr = day_df['vol_ratio'].iloc[0] if 'vol_ratio' in day_df.columns else 1.0
        if pd.isna(vr):
            vr = 1.0
        if vr < min_vol_ratio or vr > max_vol_ratio:
            continue

        # Previous day range filter
        pdr = day_df['prev_day_range_pct'].iloc[0] if 'prev_day_range_pct' in day_df.columns else 0
        if pd.isna(pdr):
            pdr = 0
        if pdr < min_prev_day_range or pdr > max_prev_day_range:
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

            # LONG
            if direction in ('both', 'long') and not long_triggered:
                if row['Close'] > or_high and rel_vol >= vol_thresh:
                    entry_price = row['Close']
                    df.loc[idx, 'signal'] = 1
                    df.loc[idx, 'signal_type'] = 'long'
                    df.loc[idx, 'target_price'] = entry_price + (or_range * target_mult)
                    df.loc[idx, 'stop_price'] = entry_price - (or_range * stop_mult)
                    df.loc[idx, 'entry_reason'] = 'long'
                    long_triggered = True

            # SHORT
            if direction in ('both', 'short') and not short_triggered:
                if row['Close'] < or_low and rel_vol >= vol_thresh:
                    entry_price = row['Close']
                    df.loc[idx, 'signal'] = -1
                    df.loc[idx, 'signal_type'] = 'short'
                    df.loc[idx, 'target_price'] = entry_price - (or_range * target_mult)
                    df.loc[idx, 'stop_price'] = entry_price + (or_range * stop_mult)
                    df.loc[idx, 'entry_reason'] = 'short'
                    short_triggered = True

    return df


def run_sim(data, params, start_day=None, end_day=None):
    """Run one config, optionally on a date subset."""
    sim_data = {}
    total_signals = 0

    for symbol, df in data.items():
        df_slice = df.copy()
        if start_day is not None:
            df_slice = df_slice[df_slice['trading_day'] >= start_day]
        if end_day is not None:
            df_slice = df_slice[df_slice['trading_day'] <= end_day]

        if len(df_slice) < 100:
            continue

        result = generate_signals_12m(df_slice, params)
        sim_data[symbol] = result
        total_signals += (result['signal'] != 0).sum()

    if total_signals == 0:
        return None

    bt = Backtester(risk_per_trade=params.get('risk_per_trade', 400),
                    max_positions=params.get('max_positions', 3))
    tl = bt.run(sim_data)

    if len(tl) < 15:
        return None

    n = len(tl)
    winners = tl[tl['pnl_net'] > 0]
    losers = tl[tl['pnl_net'] <= 0]
    total_pnl = tl['pnl_net'].sum()
    pf = winners['pnl_net'].sum() / abs(losers['pnl_net'].sum()) if losers['pnl_net'].sum() != 0 else 0

    cum = tl['pnl_net'].cumsum()
    max_dd = (cum - cum.cummax()).min()

    tl['trade_date'] = pd.to_datetime(tl['entry_time']).dt.date
    daily_pnl = tl.groupby('trade_date')['pnl_net'].sum()
    sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252) if len(daily_pnl) > 1 and daily_pnl.std() > 0 else 0

    mc = [np.random.choice(tl['pnl_net'].values, size=n, replace=True).sum() for _ in range(1000)]
    mc_profitable = (np.array(mc) > 0).mean() * 100

    half = n // 2
    h1 = tl.iloc[:half]['pnl_net'].sum()
    h2 = tl.iloc[half:]['pnl_net'].sum()

    top5 = tl.nlargest(5, 'pnl_net')['pnl_net'].sum()

    extra_slip = tl['shares'] * 0.01 * 2
    pnl_2x_slip = total_pnl - extra_slip.sum()

    longs = tl[tl['direction'] == 'LONG']
    shorts = tl[tl['direction'] == 'SHORT']

    tl['trade_month'] = pd.to_datetime(tl['entry_time']).dt.to_period('M')
    monthly = tl.groupby('trade_month')['pnl_net'].sum()
    profitable_months = (monthly > 0).sum()

    return {
        **{k: v for k, v in params.items() if not callable(v)},
        'trades': n, 'win_rate': round(len(winners) / n * 100, 1),
        'net_pnl': round(total_pnl, 2), 'avg_trade': round(total_pnl / n, 2),
        'profit_factor': round(pf, 2), 'max_drawdown': round(max_dd, 2),
        'sharpe': round(sharpe, 2), 'mc_profitable': round(mc_profitable, 1),
        'h1_pnl': round(h1, 2), 'h2_pnl': round(h2, 2),
        'both_halves': h1 > 0 and h2 > 0,
        'outlier_dep': total_pnl > 0 and (total_pnl - top5) < 0,
        'pnl_2x_slip': round(pnl_2x_slip, 2), 'survives_2x_slip': pnl_2x_slip > 0,
        'long_pnl': round(longs['pnl_net'].sum() if len(longs) > 0 else 0, 2),
        'short_pnl': round(shorts['pnl_net'].sum() if len(shorts) > 0 else 0, 2),
        'profitable_months': profitable_months, 'total_months': len(monthly),
        'calmar': round(total_pnl / abs(max_dd), 2) if max_dd != 0 else 0,
    }


def main():
    print("=" * 70)
    print("12-MONTH STRATEGY OPTIMIZATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    data_raw = load_12m_dataset()
    print(f"Loaded {len(data_raw)} symbols")

    # Add features
    for s in data_raw:
        data_raw[s] = compute_orb_features(data_raw[s])
        data_raw[s] = add_volatility_features(data_raw[s])

    # Get date range for walk-forward split
    all_days = sorted(set().union(*[set(df['trading_day'].unique()) for df in data_raw.values()]))
    mid_point = all_days[len(all_days) // 2]
    print(f"Date range: {all_days[0]} to {all_days[-1]}")
    print(f"Walk-forward split: train {all_days[0]}-{mid_point} | test {mid_point}-{all_days[-1]}")

    configs = []

    # GROUP 1: Direction test (both vs short-only) with vol filter
    for direction in ['short', 'both']:
        for or_min in [10, 12, 15, 20]:
            for tgt in [0.5, 0.75, 1.0, 1.25]:
                for stp in [0.3, 0.375, 0.5]:
                    if stp > tgt:
                        continue
                    for min_vr in [0, 0.8, 1.0, 1.2]:
                        configs.append({
                            'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                            'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': 11.5,
                            'direction': direction, 'risk_per_trade': 400, 'max_positions': 3,
                            'min_vol_ratio': min_vr, 'max_vol_ratio': 999,
                            'min_prev_day_range': 0, 'max_prev_day_range': 999,
                            'min_or_range_pct': 0, 'group': '1_vol_filter',
                        })

    # GROUP 2: Previous day range filter (coiled spring)
    for direction in ['short', 'both']:
        for or_min in [12, 15]:
            for tgt in [0.75, 1.0, 1.25]:
                for stp in [0.375, 0.5]:
                    if stp > tgt:
                        continue
                    for min_pdr in [0, 1.0, 1.5, 2.0]:
                        for max_pdr in [3.0, 5.0, 999]:
                            configs.append({
                                'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                                'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': 11.5,
                                'direction': direction, 'risk_per_trade': 400, 'max_positions': 3,
                                'min_vol_ratio': 0, 'max_vol_ratio': 999,
                                'min_prev_day_range': min_pdr, 'max_prev_day_range': max_pdr,
                                'min_or_range_pct': 0, 'group': '2_prev_range',
                            })

    # GROUP 3: Combined vol + prev day filters
    for direction in ['short', 'both']:
        for or_min in [12, 15]:
            for tgt in [0.75, 1.0]:
                for stp in [0.375, 0.5]:
                    if stp > tgt:
                        continue
                    for min_vr in [0.8, 1.0, 1.2]:
                        for min_pdr in [1.0, 1.5]:
                            configs.append({
                                'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                                'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': 11.5,
                                'direction': direction, 'risk_per_trade': 400, 'max_positions': 3,
                                'min_vol_ratio': min_vr, 'max_vol_ratio': 999,
                                'min_prev_day_range': min_pdr, 'max_prev_day_range': 999,
                                'min_or_range_pct': 0, 'group': '3_combined',
                            })

    # GROUP 4: OR range minimum (skip low-vol days)
    for direction in ['short', 'both']:
        for or_min in [12, 15]:
            for tgt in [0.75, 1.0, 1.25]:
                for stp in [0.375, 0.5]:
                    if stp > tgt:
                        continue
                    for min_or in [0.3, 0.5, 0.75, 1.0]:
                        configs.append({
                            'or_minutes': or_min, 'target_mult': tgt, 'stop_mult': stp,
                            'vol_thresh': 1.0, 'entry_start': 10.0, 'entry_end': 11.5,
                            'direction': direction, 'risk_per_trade': 400, 'max_positions': 3,
                            'min_vol_ratio': 0, 'max_vol_ratio': 999,
                            'min_prev_day_range': 0, 'max_prev_day_range': 999,
                            'min_or_range_pct': min_or, 'group': '4_min_or',
                        })

    total = len(configs)
    print(f"\nTotal configs: {total}")

    # ============================================================
    # PHASE A: Optimize on FIRST HALF (months 1-6)
    # ============================================================
    print(f"\n{'='*70}")
    print(f"PHASE A: OPTIMIZING ON FIRST HALF ({all_days[0]} to {mid_point})")
    print(f"{'='*70}")

    results_train = []
    best_pnl = -999999
    best_sharpe = -999999

    for i, params in enumerate(configs):
        result = run_sim(data_raw, params, end_day=mid_point)
        if result is not None:
            result['phase'] = 'train'
            results_train.append(result)
            if result['net_pnl'] > best_pnl:
                best_pnl = result['net_pnl']
            if result['sharpe'] > best_sharpe and result['net_pnl'] > 0:
                best_sharpe = result['sharpe']

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{total}] {len(results_train)} valid, "
                  f"best PnL ${best_pnl:+,.0f}, best Sharpe {best_sharpe:.2f}")

    train_df = pd.DataFrame(results_train)
    train_df = train_df.sort_values('sharpe', ascending=False)

    # Get top configs from training
    train_viable = train_df[(train_df['net_pnl'] > 0) & (train_df['mc_profitable'] > 55) &
                            (train_df['both_halves'] == True) & (train_df['outlier_dep'] == False)]

    print(f"\nTrain results: {len(train_viable)} viable out of {len(train_df)}")

    if len(train_viable) == 0:
        print("No viable configs on training set. Relaxing filters...")
        train_viable = train_df[train_df['net_pnl'] > 0].head(20)

    # ============================================================
    # PHASE B: VALIDATE TOP 50 on SECOND HALF (months 7-12)
    # ============================================================
    top_n = min(50, len(train_viable))
    top_configs = train_viable.head(top_n)

    print(f"\n{'='*70}")
    print(f"PHASE B: VALIDATING TOP {top_n} ON SECOND HALF ({mid_point} to {all_days[-1]})")
    print(f"{'='*70}")

    results_test = []

    for i, (_, row) in enumerate(top_configs.iterrows()):
        # Reconstruct params from row
        params = {
            'or_minutes': int(row['or_minutes']), 'target_mult': row['target_mult'],
            'stop_mult': row['stop_mult'], 'vol_thresh': row['vol_thresh'],
            'entry_start': row['entry_start'], 'entry_end': row['entry_end'],
            'direction': row['direction'], 'risk_per_trade': int(row['risk_per_trade']),
            'max_positions': int(row['max_positions']),
            'min_vol_ratio': row.get('min_vol_ratio', 0),
            'max_vol_ratio': row.get('max_vol_ratio', 999),
            'min_prev_day_range': row.get('min_prev_day_range', 0),
            'max_prev_day_range': row.get('max_prev_day_range', 999),
            'min_or_range_pct': row.get('min_or_range_pct', 0),
        }

        result_test = run_sim(data_raw, params, start_day=mid_point)
        result_full = run_sim(data_raw, params)  # Also run on full period

        if result_test is not None and result_full is not None:
            result_test['phase'] = 'test'
            result_test['train_pnl'] = row['net_pnl']
            result_test['train_sharpe'] = row['sharpe']
            result_test['full_pnl'] = result_full['net_pnl']
            result_test['full_sharpe'] = result_full['sharpe']
            result_test['full_pf'] = result_full['profit_factor']
            result_test['full_dd'] = result_full['max_drawdown']
            result_test['full_mc'] = result_full['mc_profitable']
            result_test['full_both_halves'] = result_full['both_halves']
            result_test['full_outlier_dep'] = result_full['outlier_dep']
            results_test.append(result_test)

    test_df = pd.DataFrame(results_test)
    if len(test_df) > 0:
        test_df = test_df.sort_values('full_sharpe', ascending=False)

    # ============================================================
    # RESULTS
    # ============================================================
    print(f"\n{'='*70}")
    print("WALK-FORWARD RESULTS")
    print(f"{'='*70}")

    if len(test_df) == 0:
        print("No configs survived validation.")
        return

    # Configs that are profitable in BOTH train and test
    validated = test_df[(test_df['net_pnl'] > 0) & (test_df['train_pnl'] > 0)]
    print(f"Profitable in both halves: {len(validated)}/{len(test_df)}")

    # Full-period robust
    full_robust = test_df[(test_df['full_pnl'] > 0) & (test_df['full_both_halves'] == True) &
                          (test_df['full_outlier_dep'] == False) & (test_df['full_mc'] > 70)]
    print(f"Full-period robust: {len(full_robust)}/{len(test_df)}")

    top = full_robust.head(20) if len(full_robust) >= 5 else validated.head(20) if len(validated) >= 5 else test_df.head(20)

    print(f"\n{'='*140}")
    print("TOP 20 WALK-FORWARD VALIDATED")
    print(f"{'='*140}")
    print(f"{'OR':>4} {'Dir':>6} {'Tgt':>5} {'Stp':>5} {'MinVR':>6} {'MinPDR':>7} {'MinOR':>6} "
          f"{'Grp':>12} {'TrnPnL':>9} {'TstPnL':>9} {'FullPnL':>9} {'Shrp':>6} {'PF':>5} "
          f"{'MC%':>5} {'DD':>9} {'BH':>4} {'!OL':>4}")
    print("-" * 140)

    for _, row in top.iterrows():
        bh = "YES" if row.get('full_both_halves', False) else "no"
        ol = "yes" if row.get('full_outlier_dep', True) else "NO"
        print(f"{int(row['or_minutes']):>3}m {str(row['direction'])[:5]:>6} "
              f"{row['target_mult']:>5.2f} {row['stop_mult']:>5.3f} "
              f"{row.get('min_vol_ratio',0):>6.1f} {row.get('min_prev_day_range',0):>7.1f} "
              f"{row.get('min_or_range_pct',0):>6.2f} "
              f"{str(row.get('group',''))[:12]:>12} "
              f"${row['train_pnl']:>+8,.0f} ${row['net_pnl']:>+8,.0f} "
              f"${row['full_pnl']:>+8,.0f} {row['full_sharpe']:>6.2f} "
              f"{row['full_pf']:>5.2f} {row['full_mc']:>4.0f}% "
              f"${row['full_dd']:>8,.0f} {bh:>4} {ol:>4}")

    print(f"{'='*140}")

    # Best
    if len(full_robust) > 0:
        best = full_robust.iloc[0]
        label = "WALK-FORWARD VALIDATED BEST"
    elif len(validated) > 0:
        best = validated.iloc[0]
        label = "VALIDATED (relaxed)"
    else:
        best = test_df.iloc[0]
        label = "BEST (not fully validated)"

    print(f"\n{label}:")
    print(f"  OR: {int(best['or_minutes'])}min | Dir: {best['direction']} | "
          f"Target: {best['target_mult']}x | Stop: {best['stop_mult']}x")
    print(f"  Vol filter: {best.get('min_vol_ratio', 0)} | Prev day range: {best.get('min_prev_day_range', 0)}")
    print(f"  TRAIN PnL: ${best['train_pnl']:+,.2f}")
    print(f"  TEST PnL:  ${best['net_pnl']:+,.2f}")
    print(f"  FULL PnL:  ${best['full_pnl']:+,.2f} | Sharpe: {best['full_sharpe']} | PF: {best['full_pf']}")
    print(f"  Full DD:   ${best['full_dd']:,.2f} | MC: {best['full_mc']}%")
    print(f"  Both halves: {best['full_both_halves']} | Outlier dep: {best['full_outlier_dep']}")

    # Save
    test_df.to_csv(OUTPUT_DIR / 'walkforward_results.csv', index=False)
    train_df.to_csv(OUTPUT_DIR / 'train_results.csv', index=False)

    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    return test_df


if __name__ == "__main__":
    df = main()
