"""
Live paper trading via Alpaca — same logic as the backtest, real time.

20-min OR, 0.75x target, 0.5x stop, 1.2x vol filter, entries from
10:00-11:30 AM ET, both directions. Running at quarter size ($100/trade)
until I'm confident the live fill quality matches the backtest.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import DataFeed
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# CRITICAL: All strategy times are in US/Eastern. Host clock may be in any TZ.
ET = ZoneInfo("America/New_York")

def now_et():
    """Current wall-clock time in US/Eastern, as a naive datetime (for time-of-day math)."""
    return datetime.now(ET).replace(tzinfo=None)

LOG_DIR = Path(__file__).parent.parent / "paper_trade_logs"
LOG_DIR.mkdir(exist_ok=True)

API_KEY = os.environ.get("ALPACA_ALPHA_API_KEY", "")
SECRET_KEY = os.environ.get("ALPACA_ALPHA_SECRET_KEY", "")

# Connection settings
REQUEST_TIMEOUT = 30  # seconds
MAX_CONSECUTIVE_ERRORS = 5

UNIVERSE = [
    "NVDA", "TSLA", "MSFT", "AAPL", "AMZN", "GOOGL",
    "AVGO", "AMD", "PLTR", "TSM", "ORCL", "NFLX",
    "WMT", "JPM", "XOM", "UNH", "LRCX", "AMAT", "CRM", "HOOD",
]

# Strategy params (validated on 10 years)
OR_MINUTES = 20
TARGET_MULT = 0.75
STOP_MULT = 0.5
VOL_THRESH = 1.2
ENTRY_START = 10.0   # 10:00 AM ET
ENTRY_END = 11.5     # 11:30 AM ET
FLATTEN_TIME = 15.833  # 3:50 PM ET

# Risk params — QUARTER SIZE for paper trading
RISK_PER_TRADE = 100   # $100 per trade (quarter of backtest $400)
MAX_POSITIONS = 3
DAILY_LOSS_LIMIT = 300  # Stop trading if down $300


@dataclass
class Position:
    symbol: str
    direction: int  # 1=long, -1=short
    entry_price: float
    shares: int
    target: float
    stop: float
    entry_time: str
    or_range: float
    breakeven_active: bool = False


def log(msg):
    ts = now_et().strftime('%Y-%m-%d %H:%M:%S ET')
    line = f"[{ts}] {msg}"
    print(line)
    log_file = LOG_DIR / f"paper_trade_{now_et().strftime('%Y%m%d')}.log"
    with open(log_file, 'a') as f:
        f.write(line + "\n")


class ORBPaperTrader:
    """Runs the ORB strategy live against Alpaca's paper trading endpoint."""

    def __init__(self):
        self._init_clients()

        self.positions: List[Position] = []
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.halted = False
        self.traded_symbols_today = set()
        self.consecutive_errors = 0

        # Track OR for each symbol
        self.opening_ranges: Dict[str, dict] = {}
        self.or_computed = False
        self.positions_synced = False

        # For relative volume
        self.vol_baselines: Dict[str, float] = {}

        log("ORB Paper Trader initialized")
        self._show_account()

    def _init_clients(self):
        """Create fresh Alpaca API clients (call on startup and after connection failures)."""
        self.trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
        self.data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        log("API clients initialized")

    def _show_account(self):
        account = self.trading_client.get_account()
        log(f"Account: ${float(account.portfolio_value):,.2f} | "
            f"Buying power: ${float(account.buying_power):,.2f}")

    def _sync_positions_from_alpaca(self):
        """Reconcile internal position list with Alpaca's actual positions.
        Used for crash recovery: if the bot restarts mid-day, pick up open positions
        instead of leaving them orphaned."""
        try:
            alpaca_positions = self.trading_client.get_all_positions()
        except Exception as e:
            log(f"WARN: could not fetch Alpaca positions on sync - {e}")
            return

        if not alpaca_positions:
            return

        for ap in alpaca_positions:
            if ap.symbol in [p.symbol for p in self.positions]:
                continue
            qty = int(float(ap.qty))
            direction = 1 if qty > 0 else -1
            entry = float(ap.avg_entry_price)
            # We don't know the original OR — reconstruct from current OR data if available
            or_data = self.opening_ranges.get(ap.symbol)
            if or_data:
                or_range = or_data['or_range']
            else:
                # Conservative fallback: use 1% of price as a placeholder range
                or_range = entry * 0.01
            if direction == 1:
                target = entry + or_range * TARGET_MULT
                stop = entry - or_range * STOP_MULT
            else:
                target = entry - or_range * TARGET_MULT
                stop = entry + or_range * STOP_MULT
            self.positions.append(Position(
                symbol=ap.symbol, direction=direction, entry_price=entry,
                shares=abs(qty), target=target, stop=stop,
                entry_time=now_et().isoformat(),
                or_range=or_range,
            ))
            self.traded_symbols_today.add(ap.symbol)
            log(f"RECOVERED: {ap.symbol} {abs(qty)} sh @ ${entry:.2f} "
                f"(direction={'LONG' if direction==1 else 'SHORT'}, target=${target:.2f}, stop=${stop:.2f})")

    def _fetch_bars(self, symbol, minutes=60, from_open=False) -> pd.DataFrame:
        """Pull recent 1-min bars from Alpaca and resample to 5-min.

        If from_open=True, fetch from today's 9:30 AM ET regardless of current time.
        """
        # Build ET-aware datetimes so Alpaca gets the right window regardless of host TZ
        end_et = datetime.now(ET)
        if from_open:
            start_et = end_et.replace(hour=9, minute=30, second=0, microsecond=0)
        else:
            start_et = end_et - timedelta(minutes=minutes + 30)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start_et,
            end=end_et,
            feed=DataFeed.IEX,  # Free tier: IEX has no delay; SIP requires paid sub
        )

        bars = self.data_client.get_stock_bars(request)
        df = bars.df.reset_index()

        if len(df) == 0:
            return pd.DataFrame()

        df = df.rename(columns={
            'timestamp': 'Date', 'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'volume': 'Volume',
        })

        df['Date'] = pd.to_datetime(df['Date'])
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_convert('US/Eastern').dt.tz_localize(None)

        # Resample to 5-min
        df = df.set_index('Date').resample('5min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum',
        }).dropna(subset=['Open']).reset_index()

        return df

    def _fetch_history(self, symbol, days=20) -> pd.DataFrame:
        """Pull the last N trading days of 1-min bars and resample to 5-min.
        Used to build the time-of-day volume baseline."""
        end_et = datetime.now(ET)
        # Pull extra calendar days to cover weekends/holidays
        start_et = end_et - timedelta(days=days * 2)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=start_et,
            end=end_et,
            feed=DataFeed.IEX,
        )
        bars = self.data_client.get_stock_bars(request)
        df = bars.df.reset_index()
        if len(df) == 0:
            return pd.DataFrame()

        df = df.rename(columns={
            'timestamp': 'Date', 'open': 'Open', 'high': 'High',
            'low': 'Low', 'close': 'Close', 'volume': 'Volume',
        })
        df['Date'] = pd.to_datetime(df['Date'])
        if df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_convert('US/Eastern').dt.tz_localize(None)

        df = df.set_index('Date').resample('5min').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min',
            'Close': 'last', 'Volume': 'sum',
        }).dropna(subset=['Open']).reset_index()
        return df

    def compute_volume_baselines(self):
        """Build a time-of-day volume baseline using the last 20 trading days.
        Matches the backtest's rel_volume computation (groupby time-of-day)."""
        log("Computing 20-day time-of-day volume baselines...")
        today = now_et().date()
        for symbol in UNIVERSE:
            try:
                df = self._fetch_history(symbol, days=20)
                if len(df) == 0:
                    continue
                # Exclude today's bars (we want a clean historical baseline)
                df = df[df['Date'].dt.date < today]
                # Filter to regular trading hours
                df['time_dec'] = df['Date'].dt.hour + df['Date'].dt.minute / 60
                df = df[(df['time_dec'] >= 9.5) & (df['time_dec'] < 16.0)]
                if len(df) == 0:
                    continue
                # Average volume per 5-min slot of day (e.g., 10:15 → mean across last 20 days)
                df['time_key'] = df['Date'].dt.strftime('%H:%M')
                self.vol_baselines[symbol] = df.groupby('time_key')['Volume'].mean().to_dict()
            except Exception as e:
                log(f"  {symbol}: history fetch failed - {e}")
        log(f"Volume baselines built for {len(self.vol_baselines)} symbols")

    def compute_opening_ranges(self):
        """Calculate today's 20-min opening range for every symbol in the universe."""
        if self.or_computed:
            return

        log("Computing opening ranges...")
        errors = 0
        for symbol in UNIVERSE:
            try:
                df = self._fetch_bars(symbol, from_open=True)
                if len(df) < 4:
                    continue

                # Get bars from 9:30 to 9:50 (20-min OR)
                or_cutoff = 9.0 + (50 / 60)  # 9:50 AM
                df['time_dec'] = df['Date'].dt.hour + df['Date'].dt.minute / 60
                or_bars = df[(df['time_dec'] >= 9.5) & (df['time_dec'] < or_cutoff)]

                if len(or_bars) == 0:
                    continue

                or_high = or_bars['High'].max()
                or_low = or_bars['Low'].min()
                or_range = or_high - or_low

                if or_range <= 0:
                    continue

                self.opening_ranges[symbol] = {
                    'or_high': or_high,
                    'or_low': or_low,
                    'or_range': or_range,
                    'or_range_pct': or_range / ((or_high + or_low) / 2) * 100,
                }

                log(f"  {symbol}: OR High=${or_high:.2f} Low=${or_low:.2f} "
                    f"Range={or_range:.2f} ({self.opening_ranges[symbol]['or_range_pct']:.1f}%)")

            except Exception as e:
                errors += 1
                log(f"  {symbol}: Failed to compute OR - {e}")

        if len(self.opening_ranges) > 0:
            self.or_computed = True
            log(f"Opening ranges computed for {len(self.opening_ranges)} symbols")
        elif errors > len(UNIVERSE) // 2:
            # Mostly network errors — leave or_computed=False so we retry next loop
            log(f"OR computation failed ({errors} errors) — will retry next cycle")
        else:
            # Data was fetched OK but no valid ranges (e.g., market just opened, no bars yet)
            self.or_computed = True
            log(f"Opening ranges computed for 0 symbols (no valid ranges found)")

    def check_signals(self):
        """Scan the universe for live breakout signals and enter if criteria are met."""
        if self.halted:
            return

        now = now_et()
        current_time = now.hour + now.minute / 60

        # Only check during entry window
        if current_time < ENTRY_START or current_time > ENTRY_END:
            return

        # Max positions check
        if len(self.positions) >= MAX_POSITIONS:
            return

        for symbol in UNIVERSE:
            if symbol in self.traded_symbols_today:
                continue
            if symbol not in self.opening_ranges:
                continue
            if len(self.positions) >= MAX_POSITIONS:
                break

            try:
                df = self._fetch_bars(symbol, minutes=30)
                if len(df) == 0:
                    continue

                latest = df.iloc[-1]
                price = latest['Close']
                volume = latest['Volume']
                or_data = self.opening_ranges[symbol]

                # Relative volume vs 20-day avg for this exact time slot (matches backtest)
                time_key = latest['Date'].strftime('%H:%M')
                baseline_dict = self.vol_baselines.get(symbol, {})
                avg_vol = baseline_dict.get(time_key, 0) if isinstance(baseline_dict, dict) else 0
                rel_vol = volume / (avg_vol + 1) if avg_vol > 0 else 0

                # LONG BREAKOUT
                if price > or_data['or_high'] and rel_vol >= VOL_THRESH:
                    self._enter_trade(symbol, 1, price, or_data, rel_vol)

                # SHORT BREAKOUT
                elif price < or_data['or_low'] and rel_vol >= VOL_THRESH:
                    self._enter_trade(symbol, -1, price, or_data, rel_vol)

            except Exception as e:
                pass  # Skip silently on data errors

    def _enter_trade(self, symbol, direction, price, or_data, rel_vol):
        """Size and submit a market order via Alpaca, then track the position internally."""
        or_range = or_data['or_range']

        if direction == 1:
            target = price + (or_range * TARGET_MULT)
            stop = price - (or_range * STOP_MULT)
        else:
            target = price - (or_range * TARGET_MULT)
            stop = price + (or_range * STOP_MULT)

        # Position size
        risk_per_share = abs(price - stop)
        if risk_per_share <= 0:
            return
        shares = int(RISK_PER_TRADE / risk_per_share)
        if shares < 1:
            return

        # Cap notional, then verify $ risk hasn't blown the budget after capping
        max_shares = int(50000 / price)
        shares = min(shares, max_shares)
        actual_risk = shares * risk_per_share
        if actual_risk > RISK_PER_TRADE * 1.1:  # 10% tolerance
            log(f"SKIP: {symbol} risk ${actual_risk:.0f} exceeds ${RISK_PER_TRADE} (price too high)")
            return

        # Execute
        try:
            side = OrderSide.BUY if direction == 1 else OrderSide.SELL
            order = MarketOrderRequest(
                symbol=symbol, qty=shares, side=side, time_in_force=TimeInForce.DAY
            )
            submitted = self.trading_client.submit_order(order)

            # Poll until the order is fully filled — partial fills must NOT break the loop
            # or we'll record the wrong quantity and orphan the rest of the position.
            order_id = submitted.id
            filled_price = price
            filled_qty = 0
            final_status = None
            for _ in range(20):  # ~10 sec total
                time.sleep(0.5)
                try:
                    o = self.trading_client.get_order_by_id(order_id)
                    status_str = str(o.status).lower().split('.')[-1]  # OrderStatus enum → 'filled' etc
                    final_status = status_str
                    if status_str == 'filled':
                        if o.filled_avg_price:
                            filled_price = float(o.filled_avg_price)
                        if o.filled_qty:
                            filled_qty = int(float(o.filled_qty))
                        break
                    if status_str in ('rejected', 'canceled', 'expired'):
                        log(f"ORDER REJECTED: {symbol} status={status_str}")
                        return
                except Exception:
                    continue

            # Loop exhausted — query final state and use whatever filled
            if filled_qty < 1:
                try:
                    o = self.trading_client.get_order_by_id(order_id)
                    if o.filled_qty:
                        filled_qty = int(float(o.filled_qty))
                    if o.filled_avg_price:
                        filled_price = float(o.filled_avg_price)
                    log(f"ORDER PARTIAL/SLOW: {symbol} status={final_status} filled={filled_qty}/{shares}")
                except Exception as e:
                    log(f"ORDER STATUS UNKNOWN: {symbol} - {e}")
                if filled_qty < 1:
                    log(f"ORDER NOT FILLED: {symbol} - skipping")
                    return

            dir_str = "LONG" if direction == 1 else "SHORT"
            log(f"ENTRY: {dir_str} {filled_qty} {symbol} @ ${filled_price:.2f} | "
                f"Target=${target:.2f} Stop=${stop:.2f} | RelVol={rel_vol:.1f}x")

            self.positions.append(Position(
                symbol=symbol, direction=direction, entry_price=filled_price,
                shares=filled_qty, target=target, stop=stop,
                entry_time=now_et().isoformat(),
                or_range=or_range,
            ))

            self.traded_symbols_today.add(symbol)
            self.daily_trades += 1

            # Safety: too many trades
            if self.daily_trades > 10:
                log("ALERT: >10 trades today. Halting.")
                self.halted = True

        except Exception as e:
            log(f"ORDER FAILED: {symbol} - {e}")

    def check_exits(self):
        """Check every open position for stop, target, time exit, or breakeven trail."""
        now = now_et()
        current_time = now.hour + now.minute / 60

        to_close = []

        for i, pos in enumerate(self.positions):
            try:
                df = self._fetch_bars(pos.symbol, minutes=15)
                if len(df) == 0:
                    continue

                latest = df.iloc[-1]
                price = latest['Close']
                high = latest['High']
                low = latest['Low']

                # TIME EXIT
                if current_time >= FLATTEN_TIME:
                    to_close.append((i, price, 'time_exit'))
                    continue

                # STOP LOSS
                if pos.direction == 1 and low <= pos.stop:
                    to_close.append((i, pos.stop, 'stop_loss'))
                    continue
                elif pos.direction == -1 and high >= pos.stop:
                    to_close.append((i, pos.stop, 'stop_loss'))
                    continue

                # TARGET HIT
                if pos.direction == 1 and high >= pos.target:
                    to_close.append((i, pos.target, 'target_hit'))
                    continue
                elif pos.direction == -1 and low <= pos.target:
                    to_close.append((i, pos.target, 'target_hit'))
                    continue

                # TRAILING: Move stop to breakeven after 1x OR profit
                if not pos.breakeven_active:
                    unrealized = (price - pos.entry_price) * pos.direction
                    if unrealized >= pos.or_range:
                        pos.stop = pos.entry_price + (0.01 * pos.direction)
                        pos.breakeven_active = True
                        log(f"TRAIL: {pos.symbol} stop moved to breakeven ${pos.stop:.2f}")

            except Exception as e:
                pass

        # Close positions (reverse order)
        for i, exit_price, reason in sorted(to_close, reverse=True):
            pos = self.positions.pop(i)
            self._close_trade(pos, exit_price, reason)

    def _close_trade(self, pos, exit_price, reason):
        """Submit a closing market order and log the trade result."""
        try:
            # Submit closing order
            side = OrderSide.SELL if pos.direction == 1 else OrderSide.BUY
            order = MarketOrderRequest(
                symbol=pos.symbol, qty=pos.shares, side=side, time_in_force=TimeInForce.DAY
            )
            self.trading_client.submit_order(order)

            pnl = (exit_price - pos.entry_price) * pos.direction * pos.shares
            self.daily_pnl += pnl

            dir_str = "LONG" if pos.direction == 1 else "SHORT"
            log(f"EXIT: {dir_str} {pos.shares} {pos.symbol} @ ~${exit_price:.2f} | "
                f"Reason: {reason} | PnL: ${pnl:+,.2f} | Daily: ${self.daily_pnl:+,.2f}")

            # Save to trade log
            trade = {
                'date': now_et().isoformat(),
                'symbol': pos.symbol,
                'direction': dir_str,
                'entry_price': pos.entry_price,
                'exit_price': exit_price,
                'shares': pos.shares,
                'reason': reason,
                'pnl': round(pnl, 2),
                'daily_pnl': round(self.daily_pnl, 2),
            }
            log_file = LOG_DIR / 'trade_log.json'
            trades = []
            if log_file.exists():
                with open(log_file) as f:
                    trades = json.load(f)
            trades.append(trade)
            with open(log_file, 'w') as f:
                json.dump(trades, f, indent=2)

            # Daily loss limit
            if self.daily_pnl <= -DAILY_LOSS_LIMIT:
                log(f"DAILY LOSS LIMIT HIT: ${self.daily_pnl:+,.2f}. Halting.")
                self.halted = True

        except Exception as e:
            log(f"CLOSE FAILED: {pos.symbol} - {e}")

    def flatten_all(self):
        """Immediately close everything. Used at end of day or on shutdown."""
        for pos in list(self.positions):
            try:
                df = self._fetch_bars(pos.symbol, minutes=10)
                price = df.iloc[-1]['Close'] if len(df) > 0 else pos.entry_price
                self._close_trade(pos, price, 'flatten')
            except:
                pass
        self.positions = []

    def new_day(self):
        """Clear all daily counters and open range data at the start of each session."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.halted = False
        self.traded_symbols_today = set()
        self.opening_ranges = {}
        self.or_computed = False
        self.positions_synced = False
        self.vol_baselines = {}
        log(f"\n{'='*60}")
        log(f"NEW TRADING DAY: {now_et().strftime('%Y-%m-%d')}")
        log(f"{'='*60}")
        self._show_account()
        # Build the 20-day volume baseline for today
        self.compute_volume_baselines()

    def run(self):
        """Main event loop — runs all day, checks signals and exits every 30 seconds."""
        log("\n" + "=" * 60)
        log("ORB PAPER TRADER — PHASE 6")
        log("=" * 60)
        log(f"Strategy: 20-min OR, 0.75x target, 0.5x stop")
        log(f"Entry: 10:00-11:30 AM, vol >= 1.2x, both directions")
        log(f"Risk: ${RISK_PER_TRADE}/trade (QUARTER SIZE)")
        log(f"Max positions: {MAX_POSITIONS}")
        log(f"Symbols: {len(UNIVERSE)}")
        log("=" * 60)

        current_date = None
        last_status_minute = -1

        while True:
            try:
                clock = self.trading_client.get_clock()

                if not clock.is_open:
                    # End of day — flatten and reset
                    if current_date is not None and len(self.positions) > 0:
                        log("Market closed — flattening positions")
                        self.flatten_all()
                        log(f"Day complete. PnL: ${self.daily_pnl:+,.2f}, Trades: {self.daily_trades}")
                        current_date = None
                    time.sleep(60)
                    continue

                today = now_et().date()
                if today != current_date:
                    if current_date is not None and len(self.positions) > 0:
                        self.flatten_all()
                    self.new_day()
                    current_date = today

                now = now_et()
                current_time = now.hour + now.minute / 60

                # Compute OR after 9:50 AM
                if current_time >= 9.833 and not self.or_computed:
                    self.compute_opening_ranges()

                # Sync any pre-existing Alpaca positions AFTER OR is known
                # so reconstructed stops/targets use the real OR, not a placeholder.
                if self.or_computed and not self.positions_synced:
                    self._sync_positions_from_alpaca()
                    self.positions_synced = True

                # Check for entries (10:00-11:30)
                if ENTRY_START <= current_time <= ENTRY_END:
                    self.check_signals()

                # Check exits always during market hours
                if self.positions:
                    self.check_exits()

                # Flatten at 3:50 PM
                if current_time >= FLATTEN_TIME and self.positions:
                    log("Flatten time — closing all positions")
                    self.flatten_all()

                # Status every 15 min (deduped — only log once per quarter-hour)
                if now.minute % 15 == 0 and now.minute != last_status_minute:
                    last_status_minute = now.minute
                    open_pnl = sum(
                        (self._get_price(p.symbol) - p.entry_price) * p.direction * p.shares
                        for p in self.positions
                    )
                    log(f"STATUS: Positions={len(self.positions)} | "
                        f"Realized=${self.daily_pnl:+,.2f} | Open=${open_pnl:+,.2f} | "
                        f"Trades={self.daily_trades}")

                # Iteration completed successfully — clear the error counter
                self.consecutive_errors = 0
                time.sleep(30)  # Check every 30 seconds

            except KeyboardInterrupt:
                log("Shutting down...")
                self.flatten_all()
                break
            except Exception as e:
                self.consecutive_errors += 1
                log(f"ERROR ({self.consecutive_errors}): {e}")

                if self.consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    log("Too many consecutive errors — reinitializing API clients...")
                    try:
                        self._init_clients()
                        self.consecutive_errors = 0
                        log("API clients reinitialized successfully")
                    except Exception as reinit_err:
                        log(f"Client reinit failed: {reinit_err}")

                # Back off: 30s normally, 120s after repeated failures
                backoff = 120 if self.consecutive_errors >= 3 else 30
                time.sleep(backoff)

    def _get_price(self, symbol):
        try:
            df = self._fetch_bars(symbol, minutes=10)
            return df.iloc[-1]['Close'] if len(df) > 0 else 0
        except:
            return 0


if __name__ == "__main__":
    if not API_KEY or not SECRET_KEY:
        print("FATAL: ALPACA_ALPHA_API_KEY and ALPACA_ALPHA_SECRET_KEY must be set in env")
        sys.exit(1)
    bot = ORBPaperTrader()
    bot.run()
