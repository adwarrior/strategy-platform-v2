"""Daily price loader for the regime check.

The regime labels work on daily bars, so granularity of the source doesn't
matter — we squash to daily either way. What matters is which table covers the
symbol: the three micros MES/MNQ/MGC live in the 1-minute table; everything
else lives in the broad 5-minute table (emini.historical_data). Auto-pick so
the user never has to choose.
"""

from __future__ import annotations

import pandas as pd

from strategy_platform.data.loader import load_1m, load_5m

# Symbols whose history lives in historical_data_1m (the 5m table lacks them).
_ONE_MINUTE_SYMBOLS = {"MES", "MNQ", "MGC"}


def load_daily(symbol: str, start: str | None = None, end: str | None = None) -> pd.Series:
    """Return a daily close Series for `symbol` over [start, end].

    Auto-picks 1m vs 5m by symbol, then resamples to daily. Raises nothing for
    "no data" — returns an empty Series so the caller can message cleanly.
    """
    if symbol in _ONE_MINUTE_SYMBOLS:
        df = load_1m(symbol, start=start, end=end)
    else:
        df = load_5m(symbol, start=start, end=end)

    if df is None or df.empty or "close" not in df.columns:
        return pd.Series(dtype=float)

    # Daily close = last traded price of each calendar day. (label='right' to
    # match the platform's bar convention; for a daily close it's just the EOD.)
    daily = df["close"].resample("1D").last().dropna()
    return daily
