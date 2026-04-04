"""
Screen a broad list of tickers for intraday tradability, then dig into microstructure
for the ones that pass. Filters on dollar volume, price range, ATR, and spread estimate.
Follow-up analysis covers volume profiles, hourly range, and return autocorrelation
to flag stocks as trending, mean-reverting, or neutral.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path(__file__).parent.parent / "phase1_output"
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# STEP 1: Screen a broad universe
# ============================================================

# Large liquid names across sectors to screen from
SCREEN_UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "TSLA", "NFLX", "CRM",
    "INTC", "AVGO", "ADBE", "ORCL", "CSCO", "QCOM", "MU", "AMAT", "LRCX", "KLAC",
    "MRVL", "SNPS", "CDNS", "PANW", "CRWD", "DDOG", "SNOW", "NET", "ZS", "FTNT",
    # Finance
    "JPM", "BAC", "GS", "MS", "WFC", "C", "SCHW", "BLK", "AXP", "COF",
    "USB", "PNC", "TFC", "FITB", "KEY", "MCO", "ICE", "CME", "COIN", "HOOD",
    # Healthcare
    "UNH", "JNJ", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY",
    "AMGN", "GILD", "ISRG", "REGN", "VRTX", "MRNA", "BIIB", "ZTS", "HUM", "CI",
    # Consumer
    "WMT", "HD", "MCD", "NKE", "SBUX", "TGT", "LOW", "COST", "TJX", "ROST",
    "DG", "DLTR", "CMG", "YUM", "DPZ", "LULU", "DECK", "BURL",
    # Energy
    "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "VLO", "PSX", "OXY", "DVN",
    "HAL", "FANG", "HES",
    # Industrial
    "CAT", "DE", "BA", "GE", "HON", "UNP", "UPS", "FDX", "RTX", "LMT",
    "NOC", "GD", "WM", "RSG",
    # Materials/Mining
    "FCX", "NEM", "GOLD", "CLF", "X", "AA",
    # Retail/E-comm
    "SHOP", "ETSY", "EBAY", "W", "CHWY",
    # Semis
    "TSM", "ASML", "ON", "SWKS", "MPWR", "TER",
    # Other volatile names
    "SQ", "PYPL", "ROKU", "SNAP", "PINS", "UBER", "LYFT", "DASH", "ABNB", "RBLX",
    "PLTR", "SOFI", "RIVN", "LCID", "NIO", "XPEV", "LI",
]


def fetch_daily_stats(symbols: list, days: int = 60) -> pd.DataFrame:
    """Pull daily OHLCV from yfinance and compute ATR, spread estimate, and dollar volume for each ticker."""
    end = datetime.now()
    start = end - timedelta(days=days + 30)  # Extra buffer for ATR calc

    print(f"Fetching daily data for {len(symbols)} symbols...")
    results = []

    for i, symbol in enumerate(symbols):
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(symbols)}...")
        try:
            df = yf.download(symbol, start=start.strftime('%Y-%m-%d'),
                             end=end.strftime('%Y-%m-%d'), progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0] for col in df.columns]
            if len(df) < 30:
                continue

            # Use last 60 trading days
            df = df.tail(days)

            close = df['Close'].values
            high = df['High'].values
            low = df['Low'].values
            volume = df['Volume'].values

            avg_price = np.mean(close)
            avg_volume = np.mean(volume)
            avg_dollar_volume = np.mean(close * volume)

            # ATR (14-day)
            tr = np.maximum(high[1:] - low[1:],
                            np.maximum(abs(high[1:] - close[:-1]),
                                       abs(low[1:] - close[:-1])))
            atr_14 = np.mean(tr[-14:])
            atr_pct = atr_14 / avg_price * 100

            # Spread estimate: use average (high-low) range as a proxy
            # Real spread data requires L2 quotes, but range gives a rough idea
            avg_range = np.mean(high - low)
            spread_est_pct = (avg_range * 0.01) / avg_price * 100  # ~1% of range is spread

            # Get sector info
            try:
                info = yf.Ticker(symbol).info
                sector = info.get('sector', 'Unknown')
            except:
                sector = 'Unknown'

            results.append({
                'ticker': symbol,
                'avg_price': round(avg_price, 2),
                'avg_volume': int(avg_volume),
                'avg_dollar_volume': round(avg_dollar_volume / 1e6, 1),  # In millions
                'atr_14': round(atr_14, 2),
                'atr_pct': round(atr_pct, 2),
                'spread_est_pct': round(spread_est_pct, 4),
                'avg_daily_range_pct': round(np.mean((high - low) / close) * 100, 2),
                'sector': sector,
            })

        except Exception as e:
            pass

    df = pd.DataFrame(results)
    return df


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only stocks that clear all four liquidity and volatility thresholds."""
    filtered = df[
        (df['avg_dollar_volume'] >= 50) &       # $50M+ daily dollar volume
        (df['avg_price'] >= 20) &                # $20+ price
        (df['avg_price'] <= 500) &               # Under $500
        (df['atr_pct'] >= 1.5) &                 # 1.5%+ ATR as % of price
        (df['spread_est_pct'] < 0.05)            # Tight spread estimate
    ].copy()

    filtered = filtered.sort_values('avg_dollar_volume', ascending=False)
    return filtered


# ============================================================
# STEP 2: Intraday analysis for top picks
# ============================================================

def fetch_intraday_data(symbol: str, days: int = 59) -> pd.DataFrame:
    """Pull 5-min bars from yfinance. Yahoo caps intraday history at ~59 days."""
    end = datetime.now()
    start = end - timedelta(days=days)

    df = yf.download(symbol, start=start.strftime('%Y-%m-%d'),
                     end=end.strftime('%Y-%m-%d'), interval='5m', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()
    if 'Datetime' in df.columns:
        df = df.rename(columns={'Datetime': 'Date'})

    df['Date'] = pd.to_datetime(df['Date'])

    # Filter to regular trading hours (9:30 AM - 4:00 PM ET)
    df['hour'] = df['Date'].dt.hour
    df['minute'] = df['Date'].dt.minute
    df['time_decimal'] = df['hour'] + df['minute'] / 60
    df = df[(df['time_decimal'] >= 9.5) & (df['time_decimal'] < 16.0)]

    # 15-minute bucket for volume profile
    df['time_bucket'] = df['Date'].dt.floor('15min').dt.strftime('%H:%M')
    df['hour_bucket'] = df['hour']
    df['trading_day'] = df['Date'].dt.date

    return df


def compute_volume_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean volume per 15-min bucket — shows when the stock is actually active."""
    profile = df.groupby('time_bucket')['Volume'].mean().reset_index()
    profile.columns = ['time_bucket', 'avg_volume']
    profile = profile.sort_values('time_bucket')
    return profile


def compute_hourly_range(df: pd.DataFrame) -> pd.DataFrame:
    """High minus low by hour, averaged across all days. Useful for spotting which hours move."""
    hourly = df.groupby(['trading_day', 'hour_bucket']).agg(
        high=('High', 'max'),
        low=('Low', 'min'),
        close=('Close', 'last'),
    ).reset_index()
    hourly['range_pct'] = (hourly['high'] - hourly['low']) / hourly['close'] * 100
    avg_range = hourly.groupby('hour_bucket')['range_pct'].mean().reset_index()
    avg_range.columns = ['hour', 'avg_range_pct']
    return avg_range


def compute_autocorrelation(df: pd.DataFrame, periods: int = 1) -> float:
    """Lag-N autocorrelation of 15-min returns. Positive = trending, negative = mean-reverting."""
    # Resample to 15-minute bars
    df_15m = df.set_index('Date').resample('15min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
    }).dropna()

    returns = df_15m['Close'].pct_change().dropna()
    if len(returns) < 20:
        return 0.0

    autocorr = returns.autocorr(lag=periods)
    return autocorr if not np.isnan(autocorr) else 0.0


def analyze_stock(symbol: str) -> dict:
    """Run the full microstructure analysis for one ticker and return a results dict."""
    print(f"  Analyzing {symbol}...")
    try:
        df = fetch_intraday_data(symbol)
        if len(df) < 100:
            return None

        vol_profile = compute_volume_profile(df)
        hourly_range = compute_hourly_range(df)
        autocorr_1 = compute_autocorrelation(df, periods=1)
        autocorr_2 = compute_autocorrelation(df, periods=2)
        autocorr_3 = compute_autocorrelation(df, periods=3)

        # Classify behavior
        avg_autocorr = (autocorr_1 + autocorr_2 + autocorr_3) / 3
        if avg_autocorr > 0.03:
            behavior = "TRENDING"
        elif avg_autocorr < -0.03:
            behavior = "MEAN-REVERTING"
        else:
            behavior = "NEUTRAL"

        return {
            'symbol': symbol,
            'volume_profile': vol_profile,
            'hourly_range': hourly_range,
            'autocorr_lag1': round(autocorr_1, 4),
            'autocorr_lag2': round(autocorr_2, 4),
            'autocorr_lag3': round(autocorr_3, 4),
            'avg_autocorr': round(avg_autocorr, 4),
            'behavior': behavior,
            'intraday_data': df,
        }
    except Exception as e:
        print(f"    Error: {e}")
        return None


# ============================================================
# STEP 3: Visualization
# ============================================================

def plot_volume_profiles(analyses: dict, output_dir: Path):
    """Grid of volume profile charts, one subplot per stock."""
    n = len(analyses)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, (symbol, data) in enumerate(analyses.items()):
        ax = axes[i]
        vp = data['volume_profile']
        ax.bar(range(len(vp)), vp['avg_volume'], color='steelblue', alpha=0.7)
        ax.set_title(f'{symbol} - Volume Profile', fontsize=10)
        ax.set_xticks(range(0, len(vp), 4))
        ax.set_xticklabels(vp['time_bucket'].values[::4], rotation=45, fontsize=7)
        ax.set_ylabel('Avg Volume')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'volume_profiles.png', dpi=150)
    plt.close()
    print(f"  Saved volume_profiles.png")


def plot_hourly_ranges(analyses: dict, output_dir: Path):
    """Grid of hourly range charts — same layout as volume profiles."""
    n = len(analyses)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten() if n > 1 else [axes]

    for i, (symbol, data) in enumerate(analyses.items()):
        ax = axes[i]
        hr = data['hourly_range']
        ax.bar(hr['hour'], hr['avg_range_pct'], color='coral', alpha=0.7)
        ax.set_title(f'{symbol} - Range by Hour', fontsize=10)
        ax.set_xlabel('Hour')
        ax.set_ylabel('Avg Range %')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_dir / 'hourly_ranges.png', dpi=150)
    plt.close()
    print(f"  Saved hourly_ranges.png")


def plot_autocorrelation_summary(analyses: dict, output_dir: Path):
    """Single bar chart: green = trending, red = mean-reverting, gray = neutral."""
    symbols = []
    autocorrs = []
    colors = []

    for symbol, data in analyses.items():
        symbols.append(symbol)
        ac = data['avg_autocorr']
        autocorrs.append(ac)
        if ac > 0.03:
            colors.append('green')
        elif ac < -0.03:
            colors.append('red')
        else:
            colors.append('gray')

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(symbols, autocorrs, color=colors, alpha=0.7)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.axhline(y=0.03, color='green', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axhline(y=-0.03, color='red', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_title('Intraday Behavior: Autocorrelation of 15-min Returns\n'
                 'Green = Trending | Red = Mean-Reverting | Gray = Neutral', fontsize=12)
    ax.set_ylabel('Avg Autocorrelation (lags 1-3)')
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'autocorrelation_summary.png', dpi=150)
    plt.close()
    print(f"  Saved autocorrelation_summary.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PHASE 1: UNIVERSE SELECTION & MARKET MICROSTRUCTURE")
    print("=" * 70)

    # Step 1: Screen
    print("\n--- STEP 1: Screening universe ---")
    stats = fetch_daily_stats(SCREEN_UNIVERSE)
    stats.to_csv(OUTPUT_DIR / 'all_screened_stats.csv', index=False)
    print(f"Screened {len(stats)} stocks")

    filtered = apply_filters(stats)
    print(f"Passed filters: {len(filtered)} stocks")

    # Take top 20 by dollar volume
    top20 = filtered.head(20)
    top20.to_csv(OUTPUT_DIR / 'top20_universe.csv', index=False)

    print(f"\n{'='*90}")
    print(f"{'Ticker':<8} {'Price':>8} {'$Vol(M)':>10} {'ATR':>8} {'ATR%':>8} {'Spread%':>10} {'Range%':>8} {'Sector'}")
    print(f"{'='*90}")
    for _, row in top20.iterrows():
        print(f"{row['ticker']:<8} {row['avg_price']:>8.2f} {row['avg_dollar_volume']:>10.1f} "
              f"{row['atr_14']:>8.2f} {row['atr_pct']:>7.2f}% {row['spread_est_pct']:>9.4f}% "
              f"{row['avg_daily_range_pct']:>7.2f}% {row['sector']}")

    # Step 2: Intraday analysis
    print(f"\n--- STEP 2: Intraday microstructure analysis ---")
    tickers = top20['ticker'].tolist()
    analyses = {}

    for ticker in tickers:
        result = analyze_stock(ticker)
        if result:
            analyses[ticker] = result

    # Step 3: Summary table with behavior classification
    print(f"\n{'='*70}")
    print(f"{'Ticker':<8} {'AutoCorr1':>10} {'AutoCorr2':>10} {'AutoCorr3':>10} {'AvgAC':>8} {'Behavior'}")
    print(f"{'='*70}")
    for symbol, data in analyses.items():
        print(f"{symbol:<8} {data['autocorr_lag1']:>10.4f} {data['autocorr_lag2']:>10.4f} "
              f"{data['autocorr_lag3']:>10.4f} {data['avg_autocorr']:>8.4f} {data['behavior']}")

    # Count behaviors
    behaviors = [d['behavior'] for d in analyses.values()]
    print(f"\nTrending: {behaviors.count('TRENDING')}")
    print(f"Mean-Reverting: {behaviors.count('MEAN-REVERTING')}")
    print(f"Neutral: {behaviors.count('NEUTRAL')}")

    # Step 4: Plots
    print(f"\n--- STEP 3: Generating plots ---")
    plot_volume_profiles(analyses, OUTPUT_DIR)
    plot_hourly_ranges(analyses, OUTPUT_DIR)
    plot_autocorrelation_summary(analyses, OUTPUT_DIR)

    # Save full analysis summary
    summary = []
    for symbol, data in analyses.items():
        row = top20[top20['ticker'] == symbol].iloc[0].to_dict()
        row['autocorr_lag1'] = data['autocorr_lag1']
        row['autocorr_lag2'] = data['autocorr_lag2']
        row['autocorr_lag3'] = data['autocorr_lag3']
        row['avg_autocorr'] = data['avg_autocorr']
        row['behavior'] = data['behavior']
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(OUTPUT_DIR / 'phase1_complete_analysis.csv', index=False)

    print(f"\n{'='*70}")
    print(f"Phase 1 complete. All outputs saved to: {OUTPUT_DIR}")
    print(f"{'='*70}")

    return top20, analyses


if __name__ == "__main__":
    top20, analyses = main()
