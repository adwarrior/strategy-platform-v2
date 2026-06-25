#!/usr/bin/env python3
"""Two-tier trade-by-trade parity harness for NinjaTrader ↔ platform ports.

Tier 1: Python run on platform MySQL data vs the NT trade log (same data).
Tier 2: Python run on NT's own native export vs the NT trade log (pure logic).

Pure helpers are unit-tested with synthetic inputs; parity() orchestrates a
full check and writes a PARITY_REPORT.md.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _money(s: str) -> float:
    """'-$18.98' -> -18.98 ; '$1,126.02' -> 1126.02."""
    s = str(s).strip().replace("$", "").replace(",", "")
    return float(s) if s not in ("", "-") else 0.0


def parse_nt_trade_log(path: str) -> pd.DataFrame:
    """Parse an NT per-trade CSV export into a normalized trades frame.

    Returns columns: entry_time, exit_time, direction, entry_price,
    exit_price, pnl. NT dates are day-first DD/MM/YYYY.
    """
    raw = pd.read_csv(path)
    out = pd.DataFrame({
        "entry_time": pd.to_datetime(raw["Entry time"], dayfirst=True),
        "exit_time": pd.to_datetime(raw["Exit time"], dayfirst=True),
        "direction": raw["Market pos."].astype(str).str.strip(),
        "entry_price": raw["Entry price"].astype(float),
        "exit_price": raw["Exit price"].astype(float),
        "pnl": raw["Profit"].map(_money),
    })
    return out


def _utc_to_et_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Localize a naive UTC index to UTC, convert to ET, drop tz (ET-naive)."""
    return idx.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None)


def parse_nt_ohlc_export(path: str) -> pd.DataFrame:
    """Parse NT 1-min OHLC export 'YYYYMMDD HHMMSS;O;H;L;C;V' (UTC) to ET-naive."""
    df = pd.read_csv(path, sep=";", header=None,
                     names=["ts", "open", "high", "low", "close", "volume"],
                     dtype={"ts": str})
    idx = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S")
    df.index = _utc_to_et_naive(pd.DatetimeIndex(idx))
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def parse_nt_tick_export(path: str) -> pd.DataFrame:
    """Parse NT tick export 'YYYYMMDD HHMMSS<frac>;price;...;volume' (UTC) to ET-naive.

    The timestamp field is 'YYYYMMDD HHMMSS NNNNNNN' (space-separated subsecond
    fraction in 0.1-microsecond units). price is the first numeric after ';';
    volume is the last field.
    """
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            ts_tokens = parts[0].split()              # ['YYYYMMDD', 'HHMMSS', 'FRACTION']
            ymd, hms = ts_tokens[0], ts_tokens[1]
            frac = ts_tokens[2] if len(ts_tokens) > 2 else "0"
            base = pd.to_datetime(ymd + " " + hms, format="%Y%m%d %H%M%S")
            ts = base + pd.to_timedelta(int(frac or 0) * 100, unit="ns")
            price = float(parts[1])
            vol = float(parts[-1]) if parts[-1].strip() else 0.0
            rows.append((ts, price, vol))
    out = pd.DataFrame(rows, columns=["ts", "price", "volume"]).set_index("ts")
    out.index = _utc_to_et_naive(pd.DatetimeIndex(out.index))
    return out[["price", "volume"]]


def ticks_to_bars(ticks: pd.DataFrame, bar_size: int) -> pd.DataFrame:
    """Aggregate a tick frame (price, volume; time index) into N-tick OHLCV bars.

    Each consecutive block of `bar_size` ticks becomes one bar; the bar's index
    is its first tick's timestamp. A trailing partial block is dropped (matches
    NT, which only forms complete bars).
    """
    n = len(ticks) // bar_size
    bars = []
    times = []
    p = ticks["price"].to_numpy()
    v = ticks["volume"].to_numpy()
    t = ticks.index.to_numpy()
    for i in range(n):
        block = p[i * bar_size:(i + 1) * bar_size]
        vblock = v[i * bar_size:(i + 1) * bar_size]
        bars.append((block[0], block.max(), block.min(), block[-1], vblock.sum()))
        times.append(t[i * bar_size])
    return pd.DataFrame(bars, index=pd.DatetimeIndex(times),
                        columns=["open", "high", "low", "close", "volume"])
