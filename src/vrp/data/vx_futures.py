"""VIX futures (VX) continuous front/second-month series.

Sources CBOE CFE per-contract daily settlements from the CBOE CDN, then
stitches them into a continuous front/second-month series. The roll
schedule follows CBOE's SOQ rule: expiration is the Wednesday 30 days
prior to the 3rd Friday of the month FOLLOWING the contract's designated
expiry month.

Data source (confirmed working as of 2025):
    https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/
    VX/VX_{YYYY-MM-DD}.csv
    where {YYYY-MM-DD} is the expiry date of the specific contract.
    Coverage: 2013-present (pre-2013 contracts return HTTP 403).

Fallback — local override:
    Set the VX_CSV_OVERRIDE env var to a directory containing one CSV per
    contract, named  VX_{YYYY-MM-DD}.csv (same naming as the CBOE CDN)
    with columns:
        Trade Date, Futures, Open, High, Low, Close, Settle, Change,
        Total Volume, EFP, Open Interest
"""
from __future__ import annotations

import os
import re
from io import StringIO
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from . import cache

_KEY_PREFIX = "vx_futures"

_CONTRACT_CODE_MAP = {
    1: "F", 2: "G", 3: "H", 4: "J", 5: "K", 6: "M",
    7: "N", 8: "Q", 9: "U", 10: "V", 11: "X", 12: "Z",
}

_MONTH_NAME_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

_CBOE_BASE = (
    "https://cdn.cboe.com/data/us/futures/market_statistics/"
    "historical_data/VX/VX_{date}.csv"
)


# ---------------------------------------------------------------------------
# Roll calendar — pure logic, no network
# ---------------------------------------------------------------------------

def _third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    offset = (4 - first.weekday()) % 7  # Mon=0 … Fri=4
    first_friday = first + pd.Timedelta(days=offset)
    return first_friday + pd.Timedelta(days=14)


def vx_expiration(contract_year: int, contract_month: int) -> pd.Timestamp:
    """Expiration date for a VX contract with designated expiry (year, month).

    CBOE rule: settle on the Wednesday 30 days prior to the 3rd Friday
    of the MONTH AFTER the contract's designated month.
    """
    next_month = (
        pd.Timestamp(year=contract_year, month=contract_month, day=1)
        + pd.offsets.MonthBegin(1)
    )
    tf = _third_friday(next_month.year, next_month.month)
    return (tf - pd.Timedelta(days=30)).normalize()


def build_roll_calendar(start: str, end: str) -> pd.DataFrame:
    """Per business day in [start, end]: front and second contract expiries.

    Returns a DataFrame indexed by business date with columns
    ``front_expiry`` and ``second_expiry``.
    """
    idx = pd.bdate_range(start, end)
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)

    # Generate enough expiration dates to cover the range.
    expirations = []
    y, m = s.year, s.month
    while True:
        exp = vx_expiration(y, m)
        expirations.append(exp)
        if exp > e + pd.Timedelta(days=90):
            break
        m += 1
        if m > 12:
            m = 1
            y += 1
    expirations = sorted(set(expirations))

    front_exp = []
    second_exp = []
    for d in idx:
        future = [x for x in expirations if x > d][:2]
        if len(future) < 2:
            front_exp.append(pd.NaT)
            second_exp.append(pd.NaT)
        else:
            front_exp.append(future[0])
            second_exp.append(future[1])

    return pd.DataFrame(
        {"front_expiry": front_exp, "second_expiry": second_exp},
        index=idx,
    )


# ---------------------------------------------------------------------------
# Data fetch — CBOE per-contract CSVs keyed by expiry date
# ---------------------------------------------------------------------------

def _fetch_contract(expiry: pd.Timestamp) -> Optional[pd.DataFrame]:
    """Download the per-contract CSV for the given expiry date.

    Returns None if the contract is unavailable (HTTP 403 / network error).
    """
    date_str = expiry.strftime("%Y-%m-%d")

    override = os.environ.get("VX_CSV_OVERRIDE")
    if override:
        path = Path(override) / f"VX_{date_str}.csv"
        if path.exists():
            return pd.read_csv(path)

    url = _CBOE_BASE.format(date=date_str)
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code == 200 and len(r.text) > 50:
            return pd.read_csv(StringIO(r.text))
    except Exception:
        pass
    return None


def _parse_futures_column(futures_str: str) -> tuple[int, int]:
    """Parse the Futures column from a CBOE per-contract CSV.

    CBOE format: 'F (Jan 2020)'  →  (2020, 1)
    Also handles legacy formats:
        'VX/H20', 'VX H20', 'VXH20', 'H20'  →  (2020, 3)
    """
    s = futures_str.strip()

    # Primary CBOE CDN format: 'F (Jan 2020)' or 'M (Jun 2025)'
    m = re.match(r"[A-Z]\s+\((\w{3})\s+(\d{4})\)", s)
    if m:
        month_name, year = m.group(1), int(m.group(2))
        return year, _MONTH_NAME_MAP[month_name]

    # Legacy format: 'VX/H20', 'VX H20', 'VXH20', 'H20'
    inv = {v: k for k, v in _CONTRACT_CODE_MAP.items()}
    code = (
        s.upper()
        .replace("VX/", "")
        .replace("VX ", "")
        .replace("VX", "")
        .strip()
    )
    letter = code[0]
    year_suffix = code[1:]
    if len(year_suffix) == 2:
        year = 2000 + int(year_suffix)
    elif len(year_suffix) == 4:
        year = int(year_suffix)
    else:
        raise ValueError(f"Cannot parse contract year from '{futures_str}'")
    return year, inv[letter]


# ---------------------------------------------------------------------------
# Continuous series builder
# ---------------------------------------------------------------------------

def load_vx_continuous(
    start: str = "2013-01-01",
    end: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Continuous front/second VX settlement series.

    Returns a DataFrame indexed by trade date with columns:
        front_settle, second_settle, front_expiry, second_expiry,
        days_to_front_expiry

    The roll is mechanical: on each trade date the "front" contract is
    whichever listed VX contract has the nearest expiry strictly *after*
    that date, and "second" is the next one.

    Data coverage: 2013-present (CBOE CDN availability). For pre-2013
    data, set VX_CSV_OVERRIDE to a directory of per-contract CSVs.

    Parameters
    ----------
    start:
        First date to include (inclusive).
    end:
        Last date to include (inclusive). Defaults to today.
    use_cache:
        Read/write a local parquet cache keyed by (start, end).
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    key = f"{_KEY_PREFIX}_continuous__{start}__{end}"

    if use_cache:
        cached = cache.load(key)
        if cached is not None:
            return cached

    # Build the list of contracts we need to cover [start, end].
    # We need contracts whose data window overlaps [start, end].
    # Add a two-month buffer on each side to ensure front/second coverage.
    s = pd.Timestamp(start) - pd.DateOffset(months=2)
    e = pd.Timestamp(end) + pd.DateOffset(months=2)

    y, m = s.year, s.month
    contract_expiries: list[pd.Timestamp] = []
    while True:
        exp = vx_expiration(y, m)
        if exp > e:
            break
        contract_expiries.append(exp)
        m += 1
        if m > 12:
            m = 1
            y += 1

    frames = []
    missing = []
    for exp in contract_expiries:
        df = _fetch_contract(exp)
        if df is not None:
            frames.append(df)
        else:
            missing.append(exp.date())

    if missing:
        print(
            f"VX: {len(missing)} contract(s) unavailable "
            f"(first: {missing[0]}, last: {missing[-1]})"
        )

    if not frames:
        raise RuntimeError(
            "No VX contract data fetched. Set VX_CSV_OVERRIDE to a directory "
            "with VX_{YYYY-MM-DD}.csv files and retry."
        )

    raw = pd.concat(frames, ignore_index=True)
    # Normalise column names
    raw.columns = [c.strip().lower().replace(" ", "_") for c in raw.columns]

    date_col = "trade_date" if "trade_date" in raw.columns else "date"
    raw["trade_date"] = pd.to_datetime(raw[date_col])

    # Settle column: 'settle' is standard in CBOE per-contract files
    settle_col = next(
        (c for c in ("settle", "settlement_price", "settle_price", "close")
         if c in raw.columns),
        None,
    )
    if settle_col is None:
        raise RuntimeError(
            f"No settle column found. Available columns: {raw.columns.tolist()}"
        )

    raw = raw[["trade_date", "futures", settle_col]].dropna()
    raw = raw.rename(columns={settle_col: "settle"})
    raw["settle"] = pd.to_numeric(raw["settle"], errors="coerce")
    raw = raw.dropna(subset=["settle"])
    raw = raw[raw["settle"] > 0]

    raw["expiry"] = raw["futures"].map(
        lambda c: vx_expiration(*_parse_futures_column(c))
    )

    raw = raw.sort_values(["trade_date", "expiry"])

    out_rows = []
    for d, grp in raw.groupby("trade_date"):
        future = grp[grp["expiry"] > d].sort_values("expiry")
        if len(future) < 2:
            continue
        front = future.iloc[0]
        second = future.iloc[1]
        out_rows.append({
            "date": d,
            "front_settle": float(front["settle"]),
            "second_settle": float(second["settle"]),
            "front_expiry": front["expiry"],
            "second_expiry": second["expiry"],
            "days_to_front_expiry": (front["expiry"] - d).days,
        })

    out = pd.DataFrame(out_rows).set_index("date").sort_index()
    out = out.loc[start:end]

    cache.save(key, out)
    return out
