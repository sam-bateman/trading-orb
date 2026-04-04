"""
Six ORB variants side by side, based on what Phase 5 flagged as worth exploring.
A is the original baseline. B through D tighten the parameters. E is a completely
different hypothesis (VWAP mean reversion). F is the morning-only combo that looked best.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from copy import deepcopy

sys.path.insert(0, str(Path(__file__).parent))

from intraday_data import load_dataset, DEFAULT_UNIVERSE
from orb_strategy import compute_orb_features
from backtester_v2 import Backtester

OUTPUT_DIR = Path(__file__).parent.parent / "sim_results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# STRATEGY VARIANTS
# ============================================================

def generate_signals_variant(df, variant="A"):
    """Generate ORB signals for the given variant letter. Dispatches to VWAP reversion for variant E."""
    df = df.copy()
    df['signal'] = 0
    df['signal_type'] = None
    df['target_price'] = np.nan
    df['stop_price'] = np.nan
    df['entry_reason'] = None

    # Variant-specific parameters
    configs = {
        "A": {"target_mult": 1.5, "stop_mult": 0.75, "vol_thresh": 1.2,
               "entry_start": 10.0, "entry_end": 14.5},
        "B": {"target_mult": 0.75, "stop_mult": 0.5, "vol_thresh": 1.2,
               "entry_start": 10.0, "entry_end": 14.5},
        "C": {"target_mult": 1.0, "stop_mult": 0.6, "vol_thresh": 1.2,
               "entry_start": 10.0, "entry_end": 11.5},
        "D": {"target_mult": 0.75, "stop_mult": 0.5, "vol_thresh": 1.2,
               "entry_start": 10.0, "entry_end": 14.5, "skip_start": 12.0, "skip_end": 13.0},
        "E": "vwap_reversion",  # Completely different strategy
        "F": {"target_mult": 0.75, "stop_mult": 0.5, "vol_thresh": 1.5,
               "entry_start": 10.0, "entry_end": 11.5},
    }

    cfg = configs[variant]

    if cfg == "vwap_reversion":
        return _generate_vwap_reversion(df)

    target_mult = cfg["target_mult"]
    stop_mult = cfg["stop_mult"]
    vol_thresh = cfg["vol_thresh"]
    entry_start = cfg["entry_start"]
    entry_end = cfg["entry_end"]
    skip_start = cfg.get("skip_start", None)
    skip_end = cfg.get("skip_end", None)

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

            # Skip midday if configured
            if skip_start and skip_end and skip_start <= time <= skip_end:
                continue

            rel_vol = row.get('rel_volume', 1.0)
            if pd.isna(rel_vol):
                rel_vol = 1.0

            # LONG
            if not long_triggered and row['Close'] > or_high and rel_vol >= vol_thresh:
                entry_price = row['Close']
                df.loc[idx, 'signal'] = 1
                df.loc[idx, 'signal_type'] = 'long_breakout'
                df.loc[idx, 'target_price'] = entry_price + (or_range * target_mult)
                df.loc[idx, 'stop_price'] = entry_price - (or_range * stop_mult)
                df.loc[idx, 'entry_reason'] = f"V{variant} Long"
                long_triggered = True

            # SHORT
            if not short_triggered and row['Close'] < or_low and rel_vol >= vol_thresh:
                entry_price = row['Close']
                df.loc[idx, 'signal'] = -1
                df.loc[idx, 'signal_type'] = 'short_breakout'
                df.loc[idx, 'target_price'] = entry_price - (or_range * target_mult)
                df.loc[idx, 'stop_price'] = entry_price + (or_range * stop_mult)
                df.loc[idx, 'entry_reason'] = f"V{variant} Short"
                short_triggered = True

    return df


def _generate_vwap_reversion(df):
    """Variant E: Short when price is >1% above VWAP with declining volume, long when >1% below.
    Target is return to VWAP. Stop is 0.5% beyond entry."""
    df = df.copy()
    df['signal'] = 0
    df['signal_type'] = None
    df['target_price'] = np.nan
    df['stop_price'] = np.nan
    df['entry_reason'] = None

    VWAP_THRESHOLD = 1.0  # % away from VWAP to trigger
    STOP_PCT = 0.5         # % stop loss
    ENTRY_START = 10.5
    ENTRY_END = 14.5
    MAX_REL_VOL = 1.0      # Declining volume = exhaustion

    if 'vwap' not in df.columns or 'dist_from_vwap_pct' not in df.columns:
        return df

    for day in df['trading_day'].unique():
        day_mask = df['trading_day'] == day
        day_df = df[day_mask]

        long_triggered = False
        short_triggered = False

        for idx in day_df.index:
            row = df.loc[idx]
            time = row['time_decimal']

            if time < ENTRY_START or time > ENTRY_END:
                continue

            dist = row.get('dist_from_vwap_pct', 0)
            rel_vol = row.get('rel_volume', 1.0)
            if pd.isna(dist) or pd.isna(rel_vol):
                continue

            vwap = row.get('vwap', 0)
            if vwap <= 0:
                continue

            # SHORT: Price too far above VWAP, volume declining
            if not short_triggered and dist > VWAP_THRESHOLD and rel_vol < MAX_REL_VOL:
                entry_price = row['Close']
                df.loc[idx, 'signal'] = -1
                df.loc[idx, 'signal_type'] = 'vwap_short'
                df.loc[idx, 'target_price'] = vwap  # Target = return to VWAP
                df.loc[idx, 'stop_price'] = entry_price * (1 + STOP_PCT / 100)
                df.loc[idx, 'entry_reason'] = f"VE Short: {dist:.1f}% above VWAP, RelVol {rel_vol:.1f}"
                short_triggered = True

            # LONG: Price too far below VWAP, volume declining
            if not long_triggered and dist < -VWAP_THRESHOLD and rel_vol < MAX_REL_VOL:
                entry_price = row['Close']
                df.loc[idx, 'signal'] = 1
                df.loc[idx, 'signal_type'] = 'vwap_long'
                df.loc[idx, 'target_price'] = vwap  # Target = return to VWAP
                df.loc[idx, 'stop_price'] = entry_price * (1 - STOP_PCT / 100)
                df.loc[idx, 'entry_reason'] = f"VE Long: {dist:.1f}% below VWAP, RelVol {rel_vol:.1f}"
                long_triggered = True

    return df


# ============================================================
# RUN ALL VARIANTS
# ============================================================

def run_all_sims():
    """Run all six variants, print a comparison table, and flag the winner."""
    print("=" * 70)
    print("STRATEGY SIMULATIONS — 6 VARIANTS")
    print("=" * 70)

    # Load data
    data_raw = load_dataset(DEFAULT_UNIVERSE)
    print(f"Loaded {len(data_raw)} symbols\n")

    # Add features once
    for symbol in data_raw:
        data_raw[symbol] = compute_orb_features(data_raw[symbol])

    variants = {
        "A": "Original ORB (1.5x target, 0.75x stop)",
        "B": "Tight target (0.75x target, 0.5x stop)",
        "C": "Morning only (10:00-11:30 AM)",
        "D": "No midday + tight target",
        "E": "VWAP mean reversion",
        "F": "Morning + tight target + vol 1.5x",
    }

    results = {}

    for variant, description in variants.items():
        print(f"\n{'='*60}")
        print(f"VARIANT {variant}: {description}")
        print(f"{'='*60}")

        # Generate signals for this variant
        data = {}
        total_signals = 0
        for symbol, df in data_raw.items():
            data[symbol] = generate_signals_variant(df.copy(), variant)
            total_signals += (data[symbol]['signal'] != 0).sum()

        print(f"  Signals: {total_signals}")

        if total_signals == 0:
            print(f"  NO SIGNALS — skipping")
            results[variant] = {"description": description, "trades": 0, "net_pnl": 0}
            continue

        # Run backtest
        bt = Backtester()
        trade_log = bt.run(data)

        if len(trade_log) == 0:
            print(f"  NO TRADES EXECUTED")
            results[variant] = {"description": description, "trades": 0, "net_pnl": 0}
            continue

        # Quick stats
        winners = trade_log[trade_log['pnl_net'] > 0]
        losers = trade_log[trade_log['pnl_net'] <= 0]
        total_pnl = trade_log['pnl_net'].sum()
        win_rate = len(winners) / len(trade_log) * 100
        avg_win = winners['pnl_net'].mean() if len(winners) > 0 else 0
        avg_loss = losers['pnl_net'].mean() if len(losers) > 0 else 0
        pf = winners['pnl_net'].sum() / abs(losers['pnl_net'].sum()) if len(losers) > 0 and losers['pnl_net'].sum() != 0 else 0
        total_costs = trade_log['slippage_cost'].sum() + trade_log['commission_cost'].sum()

        # Max drawdown
        cum = trade_log['pnl_net'].cumsum()
        max_dd = (cum - cum.cummax()).min()

        # Sharpe
        trade_log['trade_date'] = pd.to_datetime(trade_log['entry_time']).dt.date
        daily_pnl = trade_log.groupby('trade_date')['pnl_net'].sum()
        if len(daily_pnl) > 1 and daily_pnl.std() > 0:
            sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
        else:
            sharpe = 0

        # Target hit rate
        target_hits = (trade_log['exit_reason'] == 'target_hit').sum()
        target_rate = target_hits / len(trade_log) * 100

        # Monte Carlo (quick version)
        mc_results = []
        for _ in range(5000):
            shuffled = np.random.choice(trade_log['pnl_net'].values, size=len(trade_log), replace=True)
            mc_results.append(shuffled.sum())
        mc_profitable = (np.array(mc_results) > 0).mean() * 100

        results[variant] = {
            "description": description,
            "trades": len(trade_log),
            "signals": total_signals,
            "winners": len(winners),
            "win_rate": round(win_rate, 1),
            "net_pnl": round(total_pnl, 2),
            "avg_trade": round(total_pnl / len(trade_log), 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(pf, 2),
            "max_drawdown": round(max_dd, 2),
            "sharpe": round(sharpe, 2),
            "target_hit_rate": round(target_rate, 1),
            "costs": round(total_costs, 2),
            "mc_profitable_pct": round(mc_profitable, 1),
        }

        # Save trade log
        trade_log.to_csv(OUTPUT_DIR / f'trades_variant_{variant}.csv', index=False)

        print(f"  Trades: {len(trade_log)}")
        print(f"  Win rate: {win_rate:.1f}%")
        print(f"  Net PnL: ${total_pnl:+,.2f}")
        print(f"  Profit factor: {pf:.2f}")
        print(f"  Target hits: {target_rate:.0f}%")
        print(f"  Max DD: ${max_dd:,.2f}")
        print(f"  Sharpe: {sharpe:.2f}")
        print(f"  MC profitable: {mc_profitable:.0f}%")

    # ============================================================
    # COMPARISON TABLE
    # ============================================================
    print("\n\n" + "=" * 110)
    print("COMPARISON TABLE")
    print("=" * 110)
    print(f"{'Var':<4} {'Description':<38} {'Trades':>6} {'WinR%':>6} {'NetPnL':>10} {'AvgTrd':>8} "
          f"{'PF':>5} {'Sharpe':>7} {'TgtHit%':>7} {'MC%':>5} {'MaxDD':>10}")
    print("-" * 110)

    for v, r in sorted(results.items()):
        if r['trades'] == 0:
            print(f"{v:<4} {r['description']:<38} {'NO TRADES':>6}")
            continue

        pnl_str = f"${r['net_pnl']:+,.0f}"
        dd_str = f"${r['max_drawdown']:,.0f}"
        marker = " ***" if r['net_pnl'] > 0 and r['mc_profitable_pct'] > 55 else ""

        print(f"{v:<4} {r['description']:<38} {r['trades']:>6} {r['win_rate']:>5.1f}% "
              f"{pnl_str:>10} {r['avg_trade']:>+7.2f} {r['profit_factor']:>5.2f} "
              f"{r['sharpe']:>7.2f} {r['target_hit_rate']:>6.1f}% {r['mc_profitable_pct']:>4.0f}% "
              f"{dd_str:>10}{marker}")

    print("=" * 110)
    print("*** = Positive PnL AND >55% Monte Carlo probability of profit")

    # Find best variant
    profitable = {k: v for k, v in results.items() if v['trades'] > 0 and v['net_pnl'] > 0}
    if profitable:
        best = max(profitable, key=lambda k: profitable[k]['sharpe'])
        print(f"\nBEST VARIANT: {best} — {results[best]['description']}")
        print(f"  Net PnL: ${results[best]['net_pnl']:+,.2f}, Sharpe: {results[best]['sharpe']:.2f}, "
              f"MC: {results[best]['mc_profitable_pct']:.0f}%")
    else:
        print(f"\nNO PROFITABLE VARIANTS FOUND.")
        # Find least bad
        least_bad = min(results, key=lambda k: abs(results[k].get('net_pnl', -99999)))
        print(f"Least bad: {least_bad} — ${results[least_bad]['net_pnl']:+,.2f}")

    # Save comparison
    comp_df = pd.DataFrame(results).T
    comp_df.to_csv(OUTPUT_DIR / 'variant_comparison.csv')
    print(f"\nAll results saved to {OUTPUT_DIR}")

    return results


if __name__ == "__main__":
    results = run_all_sims()
