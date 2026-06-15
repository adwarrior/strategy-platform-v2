"""Daily price loader for the regime check.

The regime labels work on daily bars, so the granularity of the source doesn't
matter — we squash to daily either way. `load_5m` already auto-routes the three
micros MES/MNQ/MGC to the 1-minute table internally (the 5m table lacks them)
and serves everything else from emini.historical_data, so we just call it and
resample to daily. No symbol special-casing needed here.
"""

from __future__ import annotations

import pandas as pd

from strategy_platform.data.loader import load_5m


def load_daily(symbol: str, start: str | None = None, end: str | None = None) -> pd.Series:
    """Return a daily close Series for `symbol` over [start, end].

    Returns an empty Series for "no data" so the caller can message cleanly
    rather than handling exceptions.
    """
    df = load_5m(symbol, start=start, end=end)
    if df is None or df.empty or "close" not in df.columns:
        return pd.Series(dtype=float)
    return df["close"].resample("1D").last().dropna()
