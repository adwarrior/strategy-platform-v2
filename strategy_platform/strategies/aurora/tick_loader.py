"""Raw bid/ask tick loader for Aurora. tick_data is UTC -> convert to ET-naive.
Mirrors loader.load_tick_bars DB access but keeps bid/ask and does NOT aggregate."""
from typing import Optional
from datetime import timedelta
import numpy as np
import pandas as pd
from sqlalchemy import text
from strategy_platform.data import loader


def load_raw_ticks(symbol: str, start: str, end: str,
                   host: Optional[str] = None) -> pd.DataFrame:
    engine = loader._engine(host)
    params = {"sym": symbol.upper(), "start": start}
    end_dt = pd.Timestamp(end) + timedelta(days=1)
    params["end"] = str(end_dt.date())
    sql = text("SELECT ts, price, bid, ask, volume FROM tick_data "
               "WHERE symbol = :sym AND ts >= :start AND ts < :end ORDER BY ts")
    with engine.connect() as conn:
        conn.execute(text("SET SESSION net_read_timeout  = 3600"))
        conn.execute(text("SET SESSION net_write_timeout = 3600"))
        conn.execute(text("SET SESSION wait_timeout      = 3600"))
        df = pd.read_sql(sql, conn, params=params)
    if df.empty:
        return df
    df["ts"] = (pd.DatetimeIndex(df["ts"]).tz_localize("UTC")
                .tz_convert("America/New_York").tz_localize(None))
    for c in ("price", "bid", "ask"):
        df[c] = df[c].astype(float)
    df["volume"] = df["volume"].astype(np.int64)
    return df.set_index("ts").sort_index()


def classify_delta(df: pd.DataFrame) -> pd.Series:
    """Signed volume per tick per NT OnMarketData (price vs bid/ask, tick-rule fallback)."""
    price = df["price"].to_numpy(float)
    bid   = df["bid"].to_numpy(float)
    ask   = df["ask"].to_numpy(float)
    vol   = df["volume"].to_numpy(np.int64)
    out   = np.zeros(len(df), dtype=np.int64)
    last  = np.nan
    have  = (ask > 0) & (bid > 0) & (ask >= bid)
    for i in range(len(df)):
        if vol[i] <= 0:
            last = price[i]; continue
        if have[i] and price[i] >= ask[i]:        out[i] =  vol[i]
        elif have[i] and price[i] <= bid[i]:      out[i] = -vol[i]
        elif (not have[i]) and (not np.isnan(last)) and price[i] > last: out[i] =  vol[i]
        elif (not have[i]) and (not np.isnan(last)) and price[i] < last: out[i] = -vol[i]
        else:                                     out[i] = 0
        last = price[i]
    return pd.Series(out, index=df.index, name="delta")
