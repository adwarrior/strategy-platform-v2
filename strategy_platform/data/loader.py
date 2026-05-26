"""
Data loader for the strategy platform.

Sources:
  1. Parquet cache (.cache/<symbol>_5m.parquet) — fast, no network (time-bar only)
  2. MySQL emini.historical_data                — 5-minute OHLCV (time-bar strategies)
  3. MySQL emini.historical_data_1m             — 1-minute OHLCV (MES/MNQ/MGC from 2020)
  4. MySQL emini.tick_data                      — raw ticks, aggregated to N-tick bars

Usage::

    from strategy_platform.data.loader import load_5m, load_1m, load_tick_bars, load_ticks_raw, is_oos_split

    # Time-based (5M, resampleable to 15M/30M/etc.)
    df = load_5m("GC=F", start="2024-03-01")
    is_df, oos_df, cutoff = is_oos_split(df)

    # 1-minute bars (MES, MNQ, MGC — history from 2020-01-01)
    df = load_1m("MES", start="2022-01-01")
    is_df, oos_df, cutoff = is_oos_split(df)

    # Tick-based (N-tick OHLCV bars)
    bars = load_tick_bars("MNQ", bar_size=1300, start="2024-09-01")
    is_df, oos_df, cutoff = is_oos_split(bars)

Two strategies use different DB hosts (WickTest: 172.23.48.1, GoldBot7: 192.168.1.228).
Override the host per-call via the `host` parameter, or set DB_HOST in .env as the default.
"""

from __future__ import annotations

import os
from datetime import timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '..', '.env'))

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '.cache')

RESAMPLE_MAP: Dict[str, str] = {
    '15T':  '15min',
    '30T':  '30min',
    '60T':  '60min',
    '240T': '240min',
    'D':    'D',
}

# Known available date ranges for tick data in emini.tick_data.
# These are the MAXIMUM ranges available — NT8 download limits mean no more can be obtained.
# Use these as the outer bounds when configuring a patscalp optimization run.
TICK_DATA_COVERAGE: Dict[str, Dict] = {
    'MNQ': {'start': '2024-09-01', 'end': '2026-04-17', 'notes': '~570 days. Gaps: Apr 18, Oct 13-22 (holidays only)'},
    'ES':  {'start': '2024-06-01', 'end': '2025-10-31', 'notes': '443 days. Gaps: Apr 3-8, some Oct dates (likely holidays)'},
    'MES': {'start': '2024-06-01', 'end': '2025-11-30', 'notes': '468 days. Gaps: Dec 25-31, Apr 18, Aug 20-22 (holidays only)'},
    'MCL': {'start': '2024-07-01', 'end': '2025-10-31', 'notes': '425 days. Zero gaps'},
    'MGC': {'start': '2025-01-01', 'end': '2025-10-31', 'notes': '~10 months. Micro Gold futures (1/10 GC).'},
}

# Symbols with meaningful 1-minute history in historical_data_1m.
# All other symbols in that table only have the live feed (~2026-03-29 onwards).
ONE_MINUTE_SYMBOLS: Dict[str, Dict] = {
    'MES': {'start': '2020-01-01', 'end': None, 'notes': '~6 years. Micro E-mini S&P 500.'},
    'MNQ': {'start': '2020-01-01', 'end': None, 'notes': '~6 years. Micro E-mini Nasdaq-100.'},
    'MGC': {'start': '2020-01-01', 'end': None, 'notes': '~6 years. Micro Gold futures.'},
}

INSTRUMENT_META: Dict[str, Dict] = {
    # ── US Equity Futures (full-size, time bars) ────────────────────────────
    'NQ=F':   {'tick_size': 0.25,    'tick_value':  5.00, 'commission': 3.98},
    'ES=F':   {'tick_size': 0.25,    'tick_value': 12.50, 'commission': 3.98},
    'YM=F':   {'tick_size': 1.00,    'tick_value':  5.00, 'commission': 3.98},
    'RTY=F':  {'tick_size': 0.10,    'tick_value':  5.00, 'commission': 3.98},
    # ── US Equity Futures (micro, 1M / tick bars) ───────────────────────────
    'MNQ':    {'tick_size': 0.25,    'tick_value':  0.50, 'commission': 1.02},
    'MES':    {'tick_size': 0.25,    'tick_value':  1.25, 'commission': 1.02},
    # ── Full-size ES in tick_data (stored without =F suffix) ───────────────
    'ES':     {'tick_size': 0.25,    'tick_value': 12.50, 'commission': 3.98},
    'NQ':     {'tick_size': 0.25,    'tick_value':  5.00, 'commission': 3.98},
    # ── Metals / Energy (full-size, time bars) ──────────────────────────────
    'GC=F':   {'tick_size': 0.10,    'tick_value': 10.00, 'commission': 4.62},
    'SI=F':   {'tick_size': 0.005,   'tick_value': 25.00, 'commission': 4.62},
    'CL=F':   {'tick_size': 0.01,    'tick_value': 10.00, 'commission': 4.62},
    # ── Metals / Energy (micro, 1M / tick bars) ─────────────────────────────
    'MGC':    {'tick_size': 0.10,    'tick_value':  1.00, 'commission': 1.52},
    'GC':     {'tick_size': 0.10,    'tick_value': 10.00, 'commission': 4.62},
    'MCL':    {'tick_size': 0.01,    'tick_value':  1.00, 'commission': 0.50},
    # ── FX Futures (time bars) ──────────────────────────────────────────────
    '6E=F':   {'tick_size': 0.00005, 'tick_value':  6.25, 'commission': 4.72},
    '6B=F':   {'tick_size': 0.0001,  'tick_value':  6.25, 'commission': 4.72},
    '6S=F':   {'tick_size': 0.0001,  'tick_value': 12.50, 'commission': 4.72},
    # ── Crypto (time bars) ──────────────────────────────────────────────────
    'BTC=F':  {'tick_size': 5.00,    'tick_value': 25.00, 'commission': 10.00},

    # ── FX Futures (plain, tick_data parity) ────────────────────────────────
    '6A':     {'tick_size': 0.0001,   'tick_value': 10.00, 'commission':  4.72},  # AUD/USD
    '6C':     {'tick_size': 0.0001,   'tick_value': 10.00, 'commission':  4.72},  # CAD/USD
    '6J':     {'tick_size': 0.000001, 'tick_value': 12.50, 'commission':  4.72},  # JPY/USD
    '6N':     {'tick_size': 0.0001,   'tick_value': 10.00, 'commission':  4.72},  # NZD/USD
    # ── Micro FX Futures ────────────────────────────────────────────────────
    'M6A':    {'tick_size': 0.0001,   'tick_value':  1.00, 'commission':  0.84},  # Micro AUD/USD (10k)
    'M6E':    {'tick_size': 0.0001,   'tick_value':  1.25, 'commission':  0.84},  # Micro EUR/USD (12.5k)
    'M6J':    {'tick_size': 0.000001, 'tick_value':  1.25, 'commission':  0.84},  # Micro JPY/USD (1.25M)
    # ── US Equity Futures (plain, tick_data parity) ──────────────────────────
    'YM':     {'tick_size': 1.00,     'tick_value':  5.00, 'commission':  3.98},  # Mini Dow
    'RTY':    {'tick_size': 0.10,     'tick_value':  5.00, 'commission':  3.98},  # E-mini Russell 2000
    'MYM':    {'tick_size': 1.00,     'tick_value':  0.50, 'commission':  1.02},  # Micro Dow
    'M2K':    {'tick_size': 0.10,     'tick_value':  0.50, 'commission':  1.02},  # Micro Russell 2000
    'EMD':    {'tick_size': 0.10,     'tick_value': 10.00, 'commission':  3.98},  # E-mini S&P MidCap 400
    'NKD':    {'tick_size': 5.00,     'tick_value': 25.00, 'commission':  3.98},  # Nikkei 225 USD
    # ── Metals (plain, tick_data parity) ────────────────────────────────────
    'SI':     {'tick_size': 0.005,    'tick_value': 25.00, 'commission':  4.62},  # Silver
    'HG':     {'tick_size': 0.0005,   'tick_value': 12.50, 'commission':  4.62},  # Copper
    'PA':     {'tick_size': 0.10,     'tick_value':  5.00, 'commission':  4.62},  # Palladium
    'PL':     {'tick_size': 0.10,     'tick_value':  5.00, 'commission':  4.62},  # Platinum
    # ── Energy (plain, tick_data parity) ────────────────────────────────────
    'CL':     {'tick_size': 0.01,     'tick_value': 10.00, 'commission':  3.96},  # Crude Oil
    'HO':     {'tick_size': 0.0001,   'tick_value':  4.20, 'commission':  3.96},  # Heating Oil
    'RB':     {'tick_size': 0.0001,   'tick_value':  4.20, 'commission':  3.96},  # RBOB Gasoline
    'NG':     {'tick_size': 0.001,    'tick_value': 10.00, 'commission':  3.96},  # Natural Gas
    'QM':     {'tick_size': 0.025,    'tick_value': 12.50, 'commission':  3.92},  # E-mini Crude Oil
    'QG':     {'tick_size': 0.005,    'tick_value':  2.50, 'commission':  2.52},  # E-mini Natural Gas
    # ── Livestock ────────────────────────────────────────────────────────────
    'GF':     {'tick_size': 0.00025,  'tick_value': 10.00, 'commission':  5.58},  # Feeder Cattle
    'HE':     {'tick_size': 0.00025,  'tick_value': 10.00, 'commission':  5.58},  # Lean Hogs
    'LE':     {'tick_size': 0.00025,  'tick_value': 10.00, 'commission':  5.58},  # Live Cattle
    # ── Crypto Micro ─────────────────────────────────────────────────────────
    'MBT':    {'tick_size': 5.00,     'tick_value': 25.00, 'commission':  5.52},  # Micro Bitcoin
    'MET':    {'tick_size': 2.50,     'tick_value': 12.50, 'commission':  0.92},  # Micro Ether
    # ── E-mini Metals (ETF-linked) ────────────────────────────────────────────
    'QI':     {'tick_size': 0.001,    'tick_value':  2.50, 'commission':  3.02},  # E-mini Silver (2,500 oz)
    'QO':     {'tick_size': 0.10,     'tick_value':  5.00, 'commission':  3.02},  # E-mini Gold (50 oz)
    # ── Treasuries ────────────────────────────────────────────────────────────
    'ZB':     {'tick_size': 0.03125,  'tick_value': 31.25, 'commission':  3.44},  # 30Y T-Bond (1/32)
    'UB':     {'tick_size': 0.03125,  'tick_value': 31.25, 'commission':  3.12},  # Ultra T-Bond (1/32)
    'ZN':     {'tick_size': 0.015625, 'tick_value': 15.625,'commission':  3.02},  # 10Y T-Note (1/64)
    'ZF':     {'tick_size': 0.0078125,'tick_value':  7.8125,'commission': 2.82},  # 5Y T-Note (1/128)
    'ZT':     {'tick_size': 0.0078125,'tick_value':  7.8125,'commission': 2.72},  # 2Y T-Note (1/128)
    # ── Grains / Softs ───────────────────────────────────────────────────────
    'ZC':     {'tick_size': 0.25,     'tick_value': 12.50, 'commission':  5.58},  # Corn (1/4 cent)
    'ZS':     {'tick_size': 0.25,     'tick_value': 12.50, 'commission':  5.58},  # Soybeans (1/4 cent)
    'ZW':     {'tick_size': 0.25,     'tick_value': 12.50, 'commission':  5.58},  # Wheat (1/4 cent)
    'ZL':     {'tick_size': 0.01,     'tick_value':  6.00, 'commission':  5.58},  # Soybean Oil
    'ZM':     {'tick_size': 0.10,     'tick_value': 10.00, 'commission':  5.58},  # Soybean Meal
}

# Master instrument list — all symbols known to exist in any data source.
# bar_types lists which bar types have MEANINGFUL data (not just live-feed slivers).
# When a user selects a bar_type not in bar_types, the dashboard shows a warning.
SYMBOL_COVERAGE: Dict[str, Dict] = {
    # ── US Equity Futures — full size ───────────────────────────────────────
    'NQ=F':   {'bar_types': ['time'],         'notes': '5M bars from 2008'},
    'ES=F':   {'bar_types': ['time'],         'notes': '5M bars from 2008'},
    'YM=F':   {'bar_types': ['time'],         'notes': '5M bars from 2008'},
    'RTY=F':  {'bar_types': ['time'],         'notes': '5M bars from 2008'},
    # ── US Equity Futures — micro ────────────────────────────────────────────
    'MNQ':    {'bar_types': ['time', '1m', 'tick'], 'notes': '1M from 2020 (resampled to 5M); tick Sep 2024 → Oct 2025'},
    'MES':    {'bar_types': ['time', '1m', 'tick'], 'notes': '1M from 2020 (resampled to 5M); tick Jun 2024 → Nov 2025'},
    # ── Full-size ES tick data (stored as "ES" in tick_data, not "ES=F") ───
    'ES':     {'bar_types': ['tick'],         'notes': 'Tick Jun 2024 → Oct 2025'},
    # ── Metals / Energy — full size ─────────────────────────────────────────
    'GC=F':   {'bar_types': ['time'],         'notes': '5M bars from 2008'},
    'SI=F':   {'bar_types': ['time'],         'notes': '5M bars from 2008'},
    'CL=F':   {'bar_types': ['time'],         'notes': '5M bars from 2008'},
    # ── Metals / Energy — micro ──────────────────────────────────────────────
    'MGC':    {'bar_types': ['time', '1m', 'tick'], 'notes': '1M from 2020 (resampled to 5M); tick Jan 2025 → Oct 2025'},
    'MCL':    {'bar_types': ['tick'],         'notes': 'Tick Jul 2024 → Oct 2025'},
    # ── FX Futures ───────────────────────────────────────────────────────────
    '6E=F':   {'bar_types': ['time'],         'notes': '5M bars from 2008'},
    '6B=F':   {'bar_types': ['time'],         'notes': '5M bars from 2008'},
    '6S=F':   {'bar_types': ['time'],         'notes': '5M bars from 2008'},
    # ── Crypto ───────────────────────────────────────────────────────────────
    'BTC=F':  {'bar_types': ['time'],         'notes': '5M bars from May 2021'},
    # ── European Index ────────────────────────────────────────────────────────
    '^GDAXI': {'bar_types': ['time'],         'notes': '5M bars from Dec 1999'},
}

# Ordered list of all known symbols for the dashboard symbol selectbox.
ALL_SYMBOLS: list = list(SYMBOL_COVERAGE.keys())


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _engine(host: Optional[str] = None):
    h    = host or os.getenv('DB_HOST', '127.0.0.1')
    port = os.getenv('DB_PORT', '3306')
    user = os.getenv('DB_USER', 'adam')
    pw   = os.getenv('DB_PASSWORD', '')
    db   = os.getenv('DB_NAME', 'emini')
    url  = f"mysql+pymysql://{user}:{pw}@{h}:{port}/{db}"
    return create_engine(url)


def _cache_path(symbol: str) -> str:
    safe = symbol.replace('=', '_')
    return os.path.join(CACHE_DIR, f'{safe}_5m.parquet')


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_5m(
    symbol:  str,
    start:   Optional[str] = None,
    end:     Optional[str] = None,
    refresh: bool = False,
    host:    Optional[str] = None,
) -> pd.DataFrame:
    """
    Return 5-minute OHLCV data for *symbol* as a DatetimeIndex DataFrame.

    Parameters
    ----------
    symbol  : e.g. "GC=F", "NQ=F", "ES=F"
    start   : ISO date string for earliest bar, e.g. "2024-03-01"
    end     : ISO date string for latest bar
    refresh : force re-query from MySQL even if cache exists
    host    : MySQL host override (default: DB_HOST from .env)

    Notes
    -----
    - The Parquet cache is only used when start/end are both None.
    - Passing start or end always hits MySQL directly.
    """
    path      = _cache_path(symbol)
    use_cache = (start is None and end is None and not refresh)

    if use_cache and os.path.exists(path):
        print(f"  [{symbol}] loading from cache")
        return pd.read_parquet(path)

    # Symbols with only 1m data — resample to 5m on the fly
    if symbol in ONE_MINUTE_SYMBOLS:
        print(f"  [{symbol}] no 5M table — loading 1M and resampling to 5M...")
        df1 = load_1m(symbol, start=start, end=end, host=host)
        if df1.empty:
            return df1
        # historical_data_1m is stored ET-naive (ingestion does UTC→ET→strip-tz before INSERT).
        # No conversion needed — resample directly on ET-naive timestamps.
        df = df1.resample('5min', label='right', closed='right').agg(
            {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
        ).dropna(subset=['open'])
        return df

    print(f"  [{symbol}] querying MySQL ({host or os.getenv('DB_HOST', '127.0.0.1')})...")
    engine = _engine(host)
    where  = [f"symbol = '{symbol}'"]
    if start:
        where.append(f"datetime >= '{start}'")
    if end:
        where.append(f"datetime <= '{end}'")
    sql = (
        f"SELECT datetime, open, high, low, close, volume "
        f"FROM historical_data "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY datetime"
    )
    df = pd.read_sql(sql, engine, parse_dates=['datetime'])
    df = df.set_index('datetime')
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='first')]

    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        df.to_parquet(path)
        print(f"  [{symbol}] cached to {path}")

    return df


def load_1m(
    symbol:  str,
    start:   Optional[str] = None,
    end:     Optional[str] = None,
    host:    Optional[str] = None,
) -> pd.DataFrame:
    """
    Return 1-minute OHLCV data for *symbol* from emini.historical_data_1m.

    Symbols with full history (2020-01-01 → present): MES, MNQ, MGC.
    All other symbols only have the live feed from ~2026-03-29 onwards.
    No Parquet cache — the table is updated continuously by the live feed.

    Parameters
    ----------
    symbol : e.g. "MES", "MNQ", "MGC"
    start  : ISO date string for earliest bar, e.g. "2022-01-01"
    end    : ISO date string for latest bar
    host   : MySQL host override (default: DB_HOST from .env)
    """
    print(f"  [{symbol}] querying MySQL 1M ({host or os.getenv('DB_HOST', '127.0.0.1')})...")
    engine = _engine(host)
    where  = [f"symbol = '{symbol}'"]
    if start:
        where.append(f"datetime >= '{start}'")
    if end:
        where.append(f"datetime <= '{end}'")
    sql = (
        f"SELECT datetime, open, high, low, close, volume "
        f"FROM historical_data_1m "
        f"WHERE {' AND '.join(where)} "
        f"ORDER BY datetime"
    )
    df = pd.read_sql(sql, engine, parse_dates=['datetime'])
    df = df.set_index('datetime')
    df.index = pd.to_datetime(df.index)
    df = df[~df.index.duplicated(keep='first')]
    print(f"  [{symbol}] {len(df):,} 1M bars loaded.")
    return df


def load_all_timeframes(
    symbol:  str,
    start:   Optional[str] = None,
    end:     Optional[str] = None,
    refresh: bool = False,
    host:    Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Return a dict of OHLCV DataFrames keyed by timeframe string:
      '5T', '15T', '30T', '60T', '240T', 'D'

    All higher timeframes are resampled from the 5-minute base data.
    """
    df5    = load_5m(symbol, start=start, end=end, refresh=refresh, host=host)
    result = {'5T': df5}
    for tf_key, tf_rule in RESAMPLE_MAP.items():
        result[tf_key] = (
            df5.resample(tf_rule, label='right', closed='right')
               .agg({'open': 'first', 'high': 'max', 'low': 'min',
                     'close': 'last', 'volume': 'sum'})
               .dropna()
        )
    return result


def is_oos_split(
    df:        pd.DataFrame,
    train_pct: float = 0.70,
):
    """
    Split *df* into (in_sample, out_of_sample) at *train_pct* of the date range.

    Returns
    -------
    is_df, oos_df, cutoff_date
    """
    dates      = df.index.normalize().unique().sort_values()
    cutoff_idx = int(len(dates) * train_pct)
    cutoff     = dates[cutoff_idx]
    return df[df.index < cutoff], df[df.index >= cutoff], cutoff


def load_ticks_raw(
    symbol:     str,
    start:      Optional[str] = None,
    end:        Optional[str] = None,
    host:       Optional[str] = None,
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """
    Load raw ticks from emini.tick_data without aggregation.

    Returns a DataFrame with columns: ts (DatetimeIndex), price, volume.
    Used by on_each_tick / on_price_change calculate modes.
    """
    engine = _engine(host)
    params: dict = {"sym": symbol.upper()}
    where  = ["symbol = :sym"]
    if start:
        where.append("ts >= :start")
        params["start"] = start
    if end:
        end_dt = pd.Timestamp(end) + timedelta(days=1)
        where.append("ts < :end")
        params["end"] = str(end_dt.date())

    sql = text(
        f"SELECT ts, price, volume FROM tick_data "
        f"WHERE {' AND '.join(where)} ORDER BY ts"
    )

    chunks = []
    with _engine(host).connect() as conn:
        conn.execute(text("SET SESSION net_read_timeout  = 3600"))
        conn.execute(text("SET SESSION net_write_timeout = 3600"))
        conn.execute(text("SET SESSION wait_timeout      = 3600"))
        for chunk in pd.read_sql(sql, conn, params=params, chunksize=chunk_size):
            chunk["ts"] = pd.to_datetime(chunk["ts"])
            chunks.append(chunk)

    if not chunks:
        return pd.DataFrame(columns=["price", "volume"])

    result = pd.concat(chunks, ignore_index=True).set_index("ts")
    return result


def load_tick_bars(
    symbol:     str,
    bar_size:   int,
    start:      Optional[str] = None,
    end:        Optional[str] = None,
    host:       Optional[str] = None,
    chunk_size: int = 500_000,
) -> pd.DataFrame:
    """
    Load raw ticks from emini.tick_data and aggregate to N-tick OHLCV bars.

    Ticks are streamed in chunks so only a small window is in RAM at any time.
    Peak memory is O(chunk_size), not O(total ticks).

    Parameters
    ----------
    symbol     : Plain symbol name as stored in tick_data, e.g. "MNQ"
    bar_size   : Number of ticks per bar, e.g. 1300
    start      : ISO date string, e.g. "2024-09-01" (inclusive)
    end        : ISO date string, e.g. "2025-03-01" (inclusive)
    host       : MySQL host override (default: DB_HOST from .env)
    chunk_size : Rows fetched per DB round-trip (default 500k)

    Returns
    -------
    pd.DataFrame with DatetimeIndex (bar open time) and columns:
        open, high, low, close, volume, tick_count
    Sorted ascending.
    """
    engine = _engine(host)
    params = {"sym": symbol.upper()}
    where  = ["symbol = :sym"]
    if start:
        where.append("ts >= :start")
        params["start"] = start
    if end:
        end_dt = pd.Timestamp(end) + timedelta(days=1)
        where.append("ts < :end")
        params["end"] = str(end_dt.date())

    sql = text(
        f"SELECT ts, price, volume FROM tick_data "
        f"WHERE {' AND '.join(where)} ORDER BY ts"
    )

    print(f"  [{symbol}] streaming ticks from tick_data ({start} → {end})...")

    bars: list = []
    # Carry-over: ticks that didn't fill a complete bar in the previous chunk
    carry_prices  = np.empty(0, dtype=np.float64)
    carry_volumes = np.empty(0, dtype=np.int64)
    carry_times   = np.empty(0, dtype="datetime64[ns]")
    total_ticks   = 0

    with engine.connect() as conn:
        # Extend MySQL session timeouts so large tick datasets don't drop mid-transfer.
        conn.execute(text("SET SESSION net_read_timeout  = 3600"))
        conn.execute(text("SET SESSION net_write_timeout = 3600"))
        conn.execute(text("SET SESSION wait_timeout      = 3600"))
        for chunk in pd.read_sql(sql, conn, params=params, chunksize=chunk_size):
            chunk["ts"] = pd.to_datetime(chunk["ts"])

            # Prepend any leftover ticks from the previous chunk
            prices  = np.concatenate([carry_prices,  chunk["price"].to_numpy(dtype=np.float64)])
            volumes = np.concatenate([carry_volumes, chunk["volume"].to_numpy(dtype=np.int64)])
            times   = np.concatenate([carry_times,   chunk["ts"].to_numpy()])
            total_ticks += len(chunk)

            n          = len(prices)
            n_complete = (n // bar_size) * bar_size  # ticks that form whole bars

            for i in range(0, n_complete, bar_size):
                bp = prices[i : i + bar_size]
                bv = volumes[i : i + bar_size]
                bars.append({
                    "ts":         times[i],
                    "open":       float(bp[0]),
                    "high":       float(bp.max()),
                    "low":        float(bp.min()),
                    "close":      float(bp[-1]),
                    "volume":     int(bv.sum()),
                    "tick_count": bar_size,
                })

            # Keep remainder (< bar_size ticks) for next chunk
            carry_prices  = prices[n_complete:]
            carry_volumes = volumes[n_complete:]
            carry_times   = times[n_complete:]

    # Final partial bar at end of data
    if len(carry_prices) > 0:
        bars.append({
            "ts":         carry_times[0],
            "open":       float(carry_prices[0]),
            "high":       float(carry_prices.max()),
            "low":        float(carry_prices.min()),
            "close":      float(carry_prices[-1]),
            "volume":     int(carry_volumes.sum()),
            "tick_count": len(carry_prices),
        })

    if not bars:
        print(f"  [{symbol}] no ticks found.")
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "tick_count"])

    result = pd.DataFrame(bars).set_index("ts")
    result.index = pd.to_datetime(result.index)
    print(f"  [{symbol}] {total_ticks:,} ticks → {len(result):,} bars at {bar_size}-tick resolution.")
    return result


def get_meta(symbol: str) -> dict:
    """Return tick/commission metadata for *symbol*."""
    if symbol not in INSTRUMENT_META:
        raise KeyError(
            f"No metadata for symbol '{symbol}'. "
            f"Known symbols: {list(INSTRUMENT_META.keys())}"
        )
    return dict(INSTRUMENT_META[symbol])


def load_nt_csv(
    path:       str,
    resample:   str = '5min',
    tz_input:   str = 'UTC',
    tz_output:  str = 'America/New_York',
    start:      Optional[str] = None,
    end:        Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a NinjaTrader exported OHLCV text file and return a DataFrame
    compatible with the strategy platform (same format as load_5m output).

    NT exports semicolon-delimited files with no header:
        YYYYMMDD HHMMSS;open;high;low;close;volume

    Parameters
    ----------
    path        : Path to the NT .txt export file.
    resample    : Pandas offset string for bar resampling, e.g. '5min', '1min'.
                  Use '1min' to skip resampling and keep native bars.
    tz_input    : Timezone of the NT export file. Default: UTC.
                  NT data providers export in UTC regardless of the platform
                  display timezone setting.
    tz_output   : Timezone to output. Default: America/New_York (ET with DST).
                  Must match the MySQL DB convention (ET) so that session
                  boundaries, PS windows and trading hours all align correctly.
    start       : Optional ISO date string to filter from (inclusive).
    end         : Optional ISO date string to filter to (inclusive).

    Returns
    -------
    pd.DataFrame with DatetimeIndex (naive, tz_output local time) and columns:
        open, high, low, close, volume
    """
    df = pd.read_csv(
        path,
        sep=';',
        header=None,
        names=['datetime', 'open', 'high', 'low', 'close', 'volume'],
        dtype={'open': float, 'high': float, 'low': float, 'close': float, 'volume': float},
    )

    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d %H%M%S')
    df = df.set_index('datetime')
    df.index = df.index.tz_localize(tz_input).tz_convert(tz_output)

    # Resample to target bar size if needed.
    # NT exports bars labelled by CLOSE time; the MySQL loader & the rest of
    # the platform also use label='right', closed='right'. Match that here.
    if resample and resample != '1min':
        df = (
            df.resample(resample, label='right', closed='right')
              .agg({'open': 'first', 'high': 'max', 'low': 'min',
                    'close': 'last', 'volume': 'sum'})
              .dropna()
        )

    # Strip timezone for consistency with MySQL loader output
    df.index = df.index.tz_localize(None)

    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end) + pd.Timedelta(days=1)]

    df = df[~df.index.duplicated(keep='first')]
    print(f"  [NT CSV] loaded {len(df):,} bars from {path} (resampled to {resample})")
    return df
