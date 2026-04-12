"""
Pull 12 months of 5-min bars from Alpaca for the full universe.
Cleans and stores as parquet in the same format as intraday_data.py.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import time as time_module
import warnings
warnings.filterwarnings('ignore')

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

DATA_DIR = Path(__file__).parent.parent / "data" / "intraday_12m"
DATA_DIR_MAX = Path(__file__).parent.parent / "data" / "intraday_max"
DATA_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR_MAX.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("ALPACA_ALPHA_API_KEY", "")
SECRET_KEY = os.environ.get("ALPACA_ALPHA_SECRET_KEY", "")

UNIVERSE = [
    "NVDA", "TSLA", "MSFT", "AAPL", "AMZN", "GOOGL",
    "AVGO", "AMD", "PLTR", "TSM", "ORCL", "NFLX",
    "WMT", "JPM", "XOM", "UNH", "LRCX", "AMAT", "CRM", "HOOD",
]


def fetch_symbol(client, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Download minute bars for one symbol in 30-day chunks (avoids API timeouts)."""
    all_dfs = []
    current = start

    while current < end:
        chunk_end = min(current + timedelta(days=30), end)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=current,
            end=chunk_end,
        )

        try:
            bars = client.get_stock_bars(request)
            df = bars.df.reset_index()
            if len(df) > 0:
                all_dfs.append(df)
        except Exception as e:
            print(f"    Chunk {current.date()}-{chunk_end.date()} failed: {e}")

        current = chunk_end + timedelta(days=1)
        time_module.sleep(0.2)  # Rate limit

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


def clean_alpaca_data(df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, dict]:
    """Resample to 5-min, convert to ET, remove bad bars, match intraday_data column layout."""
    report = {"symbol": symbol, "raw_bars": len(df), "issues": []}

    df = df.copy()

    # Rename columns to match our format
    df = df.rename(columns={
        'timestamp': 'Date',
        'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close',
        'volume': 'Volume',
    })

    # Convert to Eastern time
    df['Date'] = pd.to_datetime(df['Date'])
    if df['Date'].dt.tz is not None:
        df['Date'] = df['Date'].dt.tz_convert('US/Eastern').dt.tz_localize(None)

    # Resample 1-min to 5-min bars
    df = df.set_index('Date')
    df = df.resample('5min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last',
        'Volume': 'sum',
    }).dropna(subset=['Open'])
    df = df.reset_index()

    df['symbol'] = symbol
    df['hour'] = df['Date'].dt.hour
    df['minute'] = df['Date'].dt.minute
    df['time_decimal'] = df['hour'] + df['minute'] / 60
    df['trading_day'] = df['Date'].dt.date

    # Filter to regular hours (9:30-16:00)
    pre_count = len(df)
    df = df[(df['time_decimal'] >= 9.5) & (df['time_decimal'] < 16.0)]
    removed = pre_count - len(df)
    if removed > 0:
        report['issues'].append(f"Removed {removed} pre/post-market bars")

    # Remove zero volume
    bad_vol = (df['Volume'] <= 0).sum()
    if bad_vol > 0:
        df = df[df['Volume'] > 0]
        report['issues'].append(f"Removed {bad_vol} zero-volume bars")

    # Remove bad prices
    bad_price = ((df['High'] < df['Low']) | (df['Close'] <= 0)).sum()
    if bad_price > 0:
        df = df[(df['High'] >= df['Low']) & (df['Close'] > 0)]
        report['issues'].append(f"Removed {bad_price} bad-price bars")

    # Remove duplicates
    dupes = df.duplicated(subset=['Date'], keep='first').sum()
    if dupes > 0:
        df = df.drop_duplicates(subset=['Date'], keep='first')
        report['issues'].append(f"Removed {dupes} duplicate bars")

    report['clean_bars'] = len(df)
    report['trading_days'] = len(df['trading_day'].unique())

    df = df.sort_values('Date').reset_index(drop=True)

    # Keep only needed columns
    df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume',
             'symbol', 'hour', 'minute', 'time_decimal', 'trading_day']]

    return df, report


def add_derived_columns(df: pd.DataFrame, or_minutes: int = 12) -> pd.DataFrame:
    """Tack on VWAP, opening range, prev-day levels, and relative volume."""
    df = df.copy()

    # VWAP
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['tp_vol'] = df['typical_price'] * df['Volume']
    df['cum_tp_vol'] = df.groupby('trading_day')['tp_vol'].cumsum()
    df['cum_vol'] = df.groupby('trading_day')['Volume'].cumsum()
    df['vwap'] = df['cum_tp_vol'] / (df['cum_vol'] + 1e-10)
    df = df.drop(columns=['typical_price', 'tp_vol', 'cum_tp_vol', 'cum_vol'])

    # Cumulative volume
    df['cum_volume'] = df.groupby('trading_day')['Volume'].cumsum()

    # Time bucket
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

    # Distance from VWAP
    df['dist_from_vwap_pct'] = (df['Close'] - df['vwap']) / df['vwap'] * 100

    return df


def build_12m_dataset(months: int = 12):
    """Fetch, clean, and cache 12 months of data for the whole universe."""
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)

    end = datetime.now()
    start = end - timedelta(days=months * 30)

    print(f"Fetching {months} months of 5-min data from Alpaca")
    print(f"Period: {start.date()} to {end.date()}")
    print(f"Symbols: {len(UNIVERSE)}\n")

    data = {}
    reports = []

    for symbol in UNIVERSE:
        print(f"  {symbol}...", end=" ", flush=True)

        # Check cache
        cache_path = DATA_DIR / f"{symbol}.parquet"
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            if len(cached) > 10000:
                data[symbol] = cached
                print(f"{len(cached)} bars (cached)")
                reports.append({"symbol": symbol, "cached": True, "bars": len(cached)})
                continue

        try:
            raw = fetch_symbol(client, symbol, start, end)
            if len(raw) == 0:
                print("NO DATA")
                continue

            cleaned, report = clean_alpaca_data(raw, symbol)
            enriched = add_derived_columns(cleaned)

            # Save
            enriched.to_parquet(cache_path, index=False)
            data[symbol] = enriched
            reports.append(report)

            print(f"{report['clean_bars']} bars, {report['trading_days']} days")

        except Exception as e:
            print(f"FAILED: {e}")
            reports.append({"symbol": symbol, "error": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print(f"Dataset: {len(data)} symbols loaded")
    total_bars = sum(len(df) for df in data.values())
    total_days = max(len(df['trading_day'].unique()) for df in data.values()) if data else 0
    print(f"Total bars: {total_bars:,}")
    print(f"Trading days: ~{total_days}")
    print(f"Saved to: {DATA_DIR}")
    print(f"{'='*60}")

    return data, reports


def load_12m_dataset() -> Dict[str, pd.DataFrame]:
    """Load whatever's already been cached in DATA_DIR. No fetching."""
    data = {}
    for symbol in UNIVERSE:
        path = DATA_DIR / f"{symbol}.parquet"
        if path.exists():
            data[symbol] = pd.read_parquet(path)
    return data


if __name__ == "__main__":
    data, reports = build_12m_dataset()
