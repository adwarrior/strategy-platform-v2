"""
Hourly tick-delta loader for DeltaDiverge.

Aggregates emini.tick_data into ONE ROW PER HOURLY BUCKET (SQL-side
SUM/GROUP BY — raw ticks never land in pandas), classifying each tick as
buy-initiated (price >= ask, +volume) or sell-initiated (price <= bid,
-volume) per the strategy spec. Ticks that fall strictly between the quotes
are ignored (measured at ~0.04% of MNQ ticks).

tick_data.ts is stored in **UTC** (verified via aurora/tick_loader.py — NOT
the CT-naive convention of historical_data_1m). We bucket in UTC in SQL,
then shift the resulting hourly index to ET-naive in pandas, matching the
bar data's db_timezone='ET' convention used elsewhere in the loader.

Performance: a single query spanning the full ~19-month tick history scans
~570M rows and does not return in reasonable time (measured: 1 week ~=
7.25M rows / ~42s; a 3-month span exceeded 120s). Every call is therefore
chunked by CALENDAR MONTH — one aggregate query per month — and each
month's result is cached to Parquet so repeat runs (dashboard re-clicks,
sweeps re-using the same window) are instant.

Coverage gate (see docs/deltadiverge_spec.md): tick_data volume is known to
be ~42-50% of the true traded volume (thinning bug), and coverage is
INTERMITTENT — some hours have near-zero tick counts despite large true
volume (e.g. 2026-03-04 11:00 ET: 21 ticks vs 49,056 contract-volume in
historical_data_1m). A near-empty hour still produces a delta value that
LOOKS legitimate, so it is not enough to check "is there any data" — we
compute an explicit coverage_ratio = tick_volume / bar_volume per hour and
let the caller gate on it.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
from sqlalchemy import text

from strategy_platform.data import loader

CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.cache', 'deltadiverge_tick_delta')


def _cache_path(symbol: str, month_start: pd.Timestamp) -> str:
    safe = symbol.replace('=', '_')
    return os.path.join(CACHE_DIR, f"{safe}_{month_start:%Y-%m}.parquet")


def _month_starts(start: pd.Timestamp, end: pd.Timestamp) -> list:
    """Calendar-month boundaries (UTC-naive) covering [start, end)."""
    cur = start.normalize().replace(day=1)
    out = []
    while cur < end:
        out.append(cur)
        cur = (cur + pd.DateOffset(months=1))
    return out


def _query_month(symbol: str, month_start: pd.Timestamp, month_end: pd.Timestamp,
                  host: Optional[str]) -> pd.DataFrame:
    """SQL-side hourly aggregation for one calendar month. Returns a UTC-naive
    DatetimeIndex frame with columns: tick_volume, delta, tick_count."""
    engine = loader._engine(host)
    sql = text(
        "SELECT FLOOR(UNIX_TIMESTAMP(ts) / 3600) AS hb, "
        "       SUM(volume) AS tick_volume, "
        "       SUM(CASE WHEN price >= ask THEN volume "
        "                WHEN price <= bid THEN -volume "
        "                ELSE 0 END) AS delta, "
        "       COUNT(*) AS tick_count "
        "FROM tick_data "
        "WHERE symbol = :sym AND ts >= :start AND ts < :end "
        "GROUP BY hb ORDER BY hb"
    )
    with engine.connect() as conn:
        conn.execute(text("SET SESSION net_read_timeout  = 3600"))
        conn.execute(text("SET SESSION net_write_timeout = 3600"))
        conn.execute(text("SET SESSION wait_timeout      = 3600"))
        rows = conn.execute(sql, {"sym": symbol.upper(),
                                   "start": str(month_start),
                                   "end": str(month_end)}).fetchall()
    if not rows:
        return pd.DataFrame(columns=['tick_volume', 'delta', 'tick_count'])

    df = pd.DataFrame(rows, columns=['hb', 'tick_volume', 'delta', 'tick_count'])
    # hb = floor(unix_ts / 3600) -> hour-bucket start, UTC, naive.
    # MySQL FLOOR() returns Decimal via pymysql; cast to int64 before arithmetic
    # so `hb * 3600` stays numeric (not object dtype) for to_datetime(unit='s').
    hb_int = df['hb'].astype('int64')
    idx = pd.DatetimeIndex(pd.to_datetime(hb_int * 3600, unit='s', utc=True))
    df.index = idx.tz_convert(None)
    df = df.drop(columns=['hb'])
    df['tick_volume'] = df['tick_volume'].astype(float)
    df['delta']       = df['delta'].astype(float)
    df['tick_count']  = df['tick_count'].astype(int)
    return df


def load_hourly_tick_delta(
    symbol: str,
    start: str,
    end: str,
    host: Optional[str] = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Return hourly tick-derived delta for *symbol* over [start, end), indexed
    by hour bucket **ET-naive, OPEN-time labelled** (bucket 14:00 = ticks from
    14:00:00 to 14:59:59.999 ET). This is deliberately OPEN-labelled (not the
    platform's usual close-right convention) because it is joined against
    pandas .resample(..., label='right') OHLCV bars downstream by shifting;
    see strategy.py's _attach_tick_delta for the exact join logic and the
    close-vs-open bar-labelling discussion.

    Columns: tick_volume, delta, tick_count.

    Chunked by calendar month with a Parquet cache per (symbol, month) —
    see module docstring for why. A month is cached only once fully queried;
    partial/current months are not cached (refresh=True forces re-query).
    """
    start_ts = pd.Timestamp(start)
    end_ts   = pd.Timestamp(end)
    if end_ts <= start_ts:
        return pd.DataFrame(columns=['tick_volume', 'delta', 'tick_count'])

    os.makedirs(CACHE_DIR, exist_ok=True)
    now_month = pd.Timestamp.utcnow().tz_localize(None).normalize().replace(day=1)

    frames = []
    for m_start in _month_starts(start_ts, end_ts):
        m_end = m_start + pd.DateOffset(months=1)
        is_complete_month = m_start < now_month  # don't cache the in-progress month
        cpath = _cache_path(symbol, m_start)

        if (not refresh) and is_complete_month and os.path.exists(cpath):
            frames.append(pd.read_parquet(cpath))
            continue

        print(f"  [deltadiverge] querying tick_data delta for {symbol} {m_start:%Y-%m}...")
        month_df = _query_month(symbol, m_start, m_end, host)

        if is_complete_month and not month_df.empty:
            month_df.to_parquet(cpath)

        frames.append(month_df)

    if not frames:
        return pd.DataFrame(columns=['tick_volume', 'delta', 'tick_count'])

    out = pd.concat(frames).sort_index()
    out = out[~out.index.duplicated(keep='first')]

    # Shift UTC-naive -> ET-naive (UTC index carries real DST via tz_localize
    # in _query_month before the naive strip is NOT done there -- redo the
    # conversion properly here using a tz-aware round trip so DST is honoured,
    # unlike the fixed +1h CT->ET shift used for historical_data_1m).
    out.index = (out.index.tz_localize('UTC').tz_convert('America/New_York').tz_localize(None))

    # Restrict to the requested window (month chunks over-fetch at the edges).
    out = out[(out.index >= start_ts) & (out.index < end_ts)]
    return out[['tick_volume', 'delta', 'tick_count']]
