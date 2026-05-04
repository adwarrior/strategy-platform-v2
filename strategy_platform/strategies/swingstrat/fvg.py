"""
Fair Value Gap (FVG) detection for SwingStrat.

Scans 3-candle patterns for gaps and validates proximity to Fibonacci levels.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd


def detect_fvgs(
    ltf_df: pd.DataFrame,
    direction: str,
    fib_71: float,
    fib_75: float,
    fvg_lookback_bars: int = 20,
    fvg_proximity_ticks: int = 10,
    tick_size: float = 0.10,
) -> Optional[Dict[str, Any]]:
    """
    Scan recent LTF bars for FVG patterns and validate against Fibonacci proximity.

    3-candle pattern: [i-3, i-2, i-1] where i-1 is most recent closed.

    BEARISH FVG (for SHORT):
      low[i-3] > high[i-1] (gap down from below)
      fvg_bottom = high[i-1], fvg_top = low[i-3]
      entry = fvg_bottom (SHORT limit)

    BULLISH FVG (for LONG):
      high[i-3] < low[i-1] (gap up from above)
      fvg_bottom = high[i-3], fvg_top = low[i-1]
      entry = fvg_top (LONG limit)

    Accept FVG if:
      abs(entry - fib_71) <= fvg_proximity_ticks * tick_size
      OR fib_71 is inside [fvg_bottom, fvg_top]

    Among qualifying FVGs, pick the one closest to fib_71.

    Parameters
    ----------
    ltf_df : pd.DataFrame
        LTF OHLCV data.
    direction : str
        'short' or 'long'.
    fib_71 : float
        Fibonacci 0.71 level.
    fib_75 : float
        Fibonacci 0.75 level.
    fvg_lookback_bars : int
        How many recent bars to scan (default 20).
    fvg_proximity_ticks : int
        Max distance in ticks from fib_71 to qualify (default 10).
    tick_size : float
        Instrument tick size (default 0.10).

    Returns
    -------
    dict or None
        Best FVG dict with keys: 'entry', 'fvg_bottom', 'fvg_top', 'distance_to_fib_71'
        or None if no FVG found.
    """
    if len(ltf_df) < 3:
        return None

    # Scan last fvg_lookback_bars bars (or fewer if data is short)
    scan_start = max(0, len(ltf_df) - fvg_lookback_bars)
    proximity_ticks_dollars = fvg_proximity_ticks * tick_size

    highs = ltf_df['high'].values
    lows = ltf_df['low'].values

    valid_fvgs: List[Dict[str, Any]] = []

    # Scan for 3-candle FVG patterns in the closed bar history.
    # Pattern: bars [a, b, c] where a is oldest, c is most recent closed.
    # We need at least 3 bars: indices (i-2, i-1, i) where i <= len-1.
    # The gap is between bar a's extreme and bar c's extreme (bar b is the displacement).
    for i in range(scan_start + 2, len(ltf_df)):
        a = i - 2  # oldest of the 3 bars
        c = i      # most recent closed bar

        h_a, l_a = highs[a], lows[a]
        h_c, l_c = highs[c], lows[c]

        # BEARISH FVG (for SHORT trades):
        # Bar A is above bar C — gap between low[A] and high[C]
        # Price will retrace UP into the gap: entry at the bottom of the gap (high[C])
        if direction == 'short' and l_a > h_c:
            fvg_bottom = h_c   # bottom of gap = high of most recent bar
            fvg_top    = l_a   # top of gap = low of oldest bar
            entry      = fvg_bottom  # short limit: enter when price retraces up to gap bottom
            dist       = abs(entry - fib_71)
            inside_fvg = fvg_bottom <= fib_71 <= fvg_top

            if dist <= proximity_ticks_dollars or inside_fvg:
                valid_fvgs.append({
                    'entry':             entry,
                    'fvg_bottom':        fvg_bottom,
                    'fvg_top':           fvg_top,
                    'distance_to_fib_71': dist,
                    'fvg_type':          'bearish',
                })

        # BULLISH FVG (for LONG trades):
        # Bar A is below bar C — gap between high[A] and low[C]
        # Price will retrace DOWN into the gap: entry at the top of the gap (low[C])
        elif direction == 'long' and h_a < l_c:
            fvg_bottom = h_a   # bottom of gap = high of oldest bar
            fvg_top    = l_c   # top of gap = low of most recent bar
            entry      = fvg_top  # long limit: enter when price retraces down to gap top
            dist       = abs(entry - fib_71)
            inside_fvg = fvg_bottom <= fib_71 <= fvg_top

            if dist <= proximity_ticks_dollars or inside_fvg:
                valid_fvgs.append({
                    'entry':             entry,
                    'fvg_bottom':        fvg_bottom,
                    'fvg_top':           fvg_top,
                    'distance_to_fib_71': dist,
                    'fvg_type':          'bullish',
                })

    if not valid_fvgs:
        return None

    # Return FVG closest to fib_71
    best = min(valid_fvgs, key=lambda x: x['distance_to_fib_71'])
    return best


def is_fvg_invalidated(
    close: float,
    fvg: Dict[str, Any],
    direction: str,
) -> bool:
    """
    Check if FVG is invalidated by price action.

    SHORT: invalidated if close > fvg_top
    LONG: invalidated if close < fvg_bottom

    Parameters
    ----------
    close : float
        Current bar close.
    fvg : dict
        FVG dict.
    direction : str
        'short' or 'long'.

    Returns
    -------
    bool
        True if FVG is invalidated.
    """
    if direction == 'short':
        return close > fvg['fvg_top']
    else:  # long
        return close < fvg['fvg_bottom']
