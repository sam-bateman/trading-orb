"""
Pulls the full 10-year history of 1-min bars from Alpaca and stores as 5-min parquet.
Uses 2-week chunks to stay under the API's SIP restriction cutoff (March 2026).
Splits the universe by listing date — most go back to 2016, a few only to 2019.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import time as time_module

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

DATA_DIR = Path(__file__).parent.parent / "data" / "intraday_max"
DATA_DIR.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("ALPACA_API_KEY", "")
SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")

# Stocks with long histories (2016+)
UNIVERSE_LONG = [
    "NVDA", "TSLA", "MSFT", "AAPL", "AMZN", "GOOGL",
    "AVGO", "AMD", "NFLX", "ORCL",
    "WMT", "JPM", "XOM", "UNH",
    "LRCX", "AMAT", "CRM",
]

# Stocks with shorter histories (2019+)
UNIVERSE_SHORT = ["PLTR", "HOOD", "TSM"]


def fetch_symbol_chunked(client, symbol, start, end):
    """Request 1-min bars in 2-week slices. Silent on failed chunks — some near the SIP cutoff just error."""
    all_dfs = []
    current = start
    chunk_days = 14  # 2 weeks at a time

    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)

        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Minute,
                start=current,
                end=chunk_end,
            )
            bars = client.get_stock_bars(request)
            df = bars.df.reset_index()
            if len(df) > 0:
                all_dfs.append(df)
        except Exception as e:
            # Skip silently — some chunks fail near the end due to SIP restriction
            pass

        current = chunk_end + timedelta(days=1)
        time_module.sleep(0.15)  # Rate limit

    if not all_dfs:
        return pd.DataFrame()

    return pd.concat(all_dfs, ignore_index=True)


def clean_and_resample(df, symbol):
    """Resample 1-min bars to 5-min, convert to ET, filter to regular hours, drop junk rows."""
    df = df.copy()
    df = df.rename(columns={
        'timestamp': 'Date', 'open': 'Open', 'high': 'High',
        'low': 'Low', 'close': 'Close', 'volume': 'Volume',
    })

    df['Date'] = pd.to_datetime(df['Date'])
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_convert('US/Eastern').dt.tz_localize(None)

    # Resample to 5-min
    df = df.set_index('Date')
    df = df.resample('5min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum',
    }).dropna(subset=['Open'])
    df = df.reset_index()

    df['symbol'] = symbol
    df['hour'] = df['Date'].dt.hour
    df['minute'] = df['Date'].dt.minute
    df['time_decimal'] = df['hour'] + df['minute'] / 60
    df['trading_day'] = df['Date'].dt.date

    # Filter to regular hours
    df = df[(df['time_decimal'] >= 9.5) & (df['time_decimal'] < 16.0)]
    df = df[df['Volume'] > 0]
    df = df[(df['High'] >= df['Low']) & (df['Close'] > 0)]
    df = df.drop_duplicates(subset=['Date'], keep='first')
    df = df.sort_values('Date').reset_index(drop=True)

    return df


def add_derived(df, or_minutes=20):
    """Compute VWAP, opening range, prev-day levels, and relative volume. Same logic as fetch_alpaca_data."""
    df = df.copy()

    # VWAP
    df['tp'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['tp_vol'] = df['tp'] * df['Volume']
    df['cum_tp_vol'] = df.groupby('trading_day')['tp_vol'].cumsum()
    df['cum_vol'] = df.groupby('trading_day')['Volume'].cumsum()
    df['vwap'] = df['cum_tp_vol'] / (df['cum_vol'] + 1e-10)
    df = df.drop(columns=['tp', 'tp_vol', 'cum_tp_vol', 'cum_vol'])

    df['cum_volume'] = df.groupby('trading_day')['Volume'].cumsum()
    df['time_bucket'] = df['Date'].dt.floor('15min').dt.strftime('%H:%M')

    # Opening range
    cutoff = 9.5 + (or_minutes / 60)
    or_data = df[df['time_decimal'] < cutoff].groupby('trading_day').agg(
        or_high=('High', 'max'), or_low=('Low', 'min'))
    df = df.merge(or_data, on='trading_day', how='left')
    df['or_range'] = df['or_high'] - df['or_low']
    df['or_range_pct'] = df['or_range'] / ((df['or_high'] + df['or_low']) / 2) * 100

    # Previous day levels
    daily = df.groupby('trading_day').agg(
        prev_high=('High', 'max'), prev_low=('Low', 'min'),
        prev_close=('Close', 'last'), prev_volume=('Volume', 'sum')).reset_index()
    daily['prev_high'] = daily['prev_high'].shift(1)
    daily['prev_low'] = daily['prev_low'].shift(1)
    daily['prev_close'] = daily['prev_close'].shift(1)
    daily['prev_volume'] = daily['prev_volume'].shift(1)
    df = df.merge(daily[['trading_day', 'prev_high', 'prev_low', 'prev_close', 'prev_volume']],
                  on='trading_day', how='left')

    # Relative volume
    df['time_key'] = df['Date'].dt.strftime('%H:%M')
    avg_vol = df.groupby('time_key')['Volume'].transform(
        lambda x: x.rolling(20, min_periods=5).mean())
    df['rel_volume'] = df['Volume'] / (avg_vol + 1)
    df = df.drop(columns=['time_key'])

    df['dist_from_vwap_pct'] = (df['Close'] - df['vwap']) / df['vwap'] * 100

    return df


def build_max_dataset():
    """Fetch, clean, and cache the full history for the whole universe. Skips symbols already on disk."""
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    print("=" * 70)
    print("FETCHING MAXIMUM HISTORICAL DATA (2016-2026)")
    print("=" * 70)

    all_symbols = UNIVERSE_LONG + UNIVERSE_SHORT
    data = {}
    start_long = datetime(2016, 1, 4)
    start_short = datetime(2019, 6, 1)
    end = datetime(2026, 3, 12)  # Before SIP cutoff

    for symbol in all_symbols:
        cache_path = DATA_DIR / f"{symbol}.parquet"

        # Check cache
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            if len(cached) > 50000:
                data[symbol] = cached
                days = len(cached['trading_day'].unique())
                print(f"  {symbol}: {len(cached):>8,} bars, {days:>5} days (cached)")
                continue

        start = start_long if symbol in UNIVERSE_LONG else start_short
        print(f"  {symbol}: fetching {start.date()} to {end.date()}...", end=" ", flush=True)

        try:
            raw = fetch_symbol_chunked(client, symbol, start, end)
            if len(raw) == 0:
                print("NO DATA")
                continue

            cleaned = clean_and_resample(raw, symbol)
            enriched = add_derived(cleaned)

            enriched.to_parquet(cache_path, index=False)
            data[symbol] = enriched

            days = len(enriched['trading_day'].unique())
            print(f"{len(enriched):,} bars, {days} days")

        except Exception as e:
            print(f"FAILED: {e}")

    # Summary
    print(f"\n{'='*70}")
    print(f"Dataset: {len(data)} symbols")
    total_bars = sum(len(df) for df in data.values())
    for symbol, df in sorted(data.items(), key=lambda x: len(x[1]), reverse=True):
        days = len(df['trading_day'].unique())
        start_date = df['Date'].min().date()
        end_date = df['Date'].max().date()
        print(f"  {symbol:6s}: {len(df):>8,} bars, {days:>5} days ({start_date} to {end_date})")
    print(f"\nTotal: {total_bars:,} bars")
    print(f"{'='*70}")

    return data


def load_max_dataset():
    """Load whatever's cached in DATA_DIR. Returns only symbols that have a parquet file."""
    data = {}
    for symbol in UNIVERSE_LONG + UNIVERSE_SHORT:
        path = DATA_DIR / f"{symbol}.parquet"
        if path.exists():
            data[symbol] = pd.read_parquet(path)
    return data


if __name__ == "__main__":
    data = build_max_dataset()
