"""
Swing leg detection for SwingStrat using fractal analysis.

Detects swing highs/lows and generates BOS (Break of Structure) signals.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def detect_fractals(
    htf_df: pd.DataFrame,
    swing_period: int = 10,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    """
    Detect fractal swing highs and lows on HTF data.

    A fractal HIGH at bar i: high[i] >= all bars in [i-swing_period, i+swing_period]
    A fractal LOW at bar i:  low[i]  <= all bars in [i-swing_period, i+swing_period]

    Fractals confirmed at bar i+swing_period (no lookahead bias).

    Parameters
    ----------
    htf_df : pd.DataFrame
        HTF OHLCV data with DatetimeIndex.
    swing_period : int
        Bars left+right to check for extremes (default 10).

    Returns
    -------
    tuple
        (fractals_list, confirmed_indices)
        - fractals_list: list of dicts with keys 'time', 'type' (high/low), 'price', 'index'
        - confirmed_indices: list of indices where a fractal was confirmed
    """
    if len(htf_df) < 2 * swing_period + 1:
        return [], []

    highs = htf_df['high'].values
    lows = htf_df['low'].values
    times = htf_df.index

    fractals: List[Dict[str, Any]] = []
    confirmed_indices: List[int] = []

    # Scan from swing_period to len-swing_period-1
    # A fractal at i is confirmed when we reach i+swing_period
    for i in range(swing_period, len(htf_df) - swing_period):
        # Check if bar i is a local high:
        # high[i] must be >= all neighbours, excluding itself from the comparison window
        window_highs = np.concatenate([highs[i - swing_period : i], highs[i + 1 : i + swing_period + 1]])
        if len(window_highs) > 0 and highs[i] >= window_highs.max():
            # Confirmed at i+swing_period (no lookahead)
            confirm_idx = i + swing_period
            if confirm_idx < len(htf_df):
                fractals.append({
                    'time': times[i],
                    'type': 'high',
                    'price': float(highs[i]),
                    'index': i,
                })
                confirmed_indices.append(confirm_idx)

        # Check if bar i is a local low
        window_lows = np.concatenate([lows[i - swing_period : i], lows[i + 1 : i + swing_period + 1]])
        if len(window_lows) > 0 and lows[i] <= window_lows.min():
            confirm_idx = i + swing_period
            if confirm_idx < len(htf_df):
                fractals.append({
                    'time': times[i],
                    'type': 'low',
                    'price': float(lows[i]),
                    'index': i,
                })
                confirmed_indices.append(confirm_idx)

    return fractals, confirmed_indices


def _has_liquidity_sweep(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    bos_index: int,
    swept_level: float,
    direction: str,
    sweep_lookback: int,
) -> bool:
    """
    Detect a liquidity sweep of `swept_level` in the HTF bars ending at the BOS bar.

    A sweep is a wick beyond the level with a body close back inside it (a grab, not a
    break). See swingstrat_spec.md ("Liquidity-sweep gate").

    SHORT leg (swept_level = leg origin swing HIGH = buy-side liquidity):
        some bar k in [bos_index - sweep_lookback, bos_index] with
        high[k] > swept_level AND close[k] <= swept_level.
    LONG leg (swept_level = leg origin swing LOW = sell-side liquidity):
        low[k] < swept_level AND close[k] >= swept_level.
    """
    start = max(0, bos_index - sweep_lookback)
    for k in range(start, bos_index + 1):
        if direction == 'short':
            if highs[k] > swept_level and closes[k] <= swept_level:
                return True
        else:  # long
            if lows[k] < swept_level and closes[k] >= swept_level:
                return True
    return False


def compute_htf_legs(
    htf_df: pd.DataFrame,
    swing_period: int = 10,
    sweep_lookback: int = 10,
) -> pd.DataFrame:
    """
    Compute HTF swing legs from fractal detection.

    Detects fractals, then generates BOS (Break of Structure) signals:
    - BOS DOWN: close below last confirmed swing LOW -> new SHORT leg
    - BOS UP: close above last confirmed swing HIGH -> new LONG leg

    Returns a DataFrame with columns:
    - 'leg_id': unique ID for this leg
    - 'direction': 'long' or 'short'
    - 'origin': start price (last swing high for SHORT, last swing low for LONG)
    - 'destination': end price (last swing low for SHORT, last swing high for LONG)
    - 'confirmed_time': time when BOS bar closed
    - 'confirmed_index': index of BOS bar
    - 'swept': bool, whether the leg origin liquidity was swept before the BOS
              (see swingstrat_spec.md). Always populated; gating is opt-in in the caller.
    """
    fractals, confirmed_indices = detect_fractals(htf_df, swing_period)

    if not fractals:
        return pd.DataFrame(columns=['leg_id', 'direction', 'origin', 'destination',
                                     'confirmed_time', 'confirmed_index', 'swept'])

    # Track last confirmed swing high and low
    last_swing_high: Optional[Dict[str, Any]] = None
    last_swing_low: Optional[Dict[str, Any]] = None

    legs: List[Dict[str, Any]] = []
    leg_id = 0

    highs = htf_df['high'].values
    lows = htf_df['low'].values
    closes = htf_df['close'].values
    times = htf_df.index

    for i in range(len(htf_df)):
        close = float(closes[i])

        # Update swing extremes when confirmed
        for frac in fractals:
            if frac['index'] + swing_period == i:
                if frac['type'] == 'high':
                    last_swing_high = frac.copy()
                elif frac['type'] == 'low':
                    last_swing_low = frac.copy()

        # BOS DOWN: close below last swing low
        if (last_swing_low is not None and
            last_swing_high is not None and
            close < last_swing_low['price']):
            leg_id += 1
            origin = float(last_swing_high['price'])
            legs.append({
                'leg_id': leg_id,
                'direction': 'short',
                'origin': origin,
                'destination': float(last_swing_low['price']),
                'confirmed_time': times[i],
                'confirmed_index': i,
                'swept': _has_liquidity_sweep(highs, lows, closes, i, origin,
                                              'short', sweep_lookback),
            })
            last_swing_low = None  # Prevent re-triggering

        # BOS UP: close above last swing high
        elif (last_swing_high is not None and
              last_swing_low is not None and
              close > last_swing_high['price']):
            leg_id += 1
            origin = float(last_swing_low['price'])
            legs.append({
                'leg_id': leg_id,
                'direction': 'long',
                'origin': origin,
                'destination': float(last_swing_high['price']),
                'confirmed_time': times[i],
                'confirmed_index': i,
                'swept': _has_liquidity_sweep(highs, lows, closes, i, origin,
                                              'long', sweep_lookback),
            })
            last_swing_high = None  # Prevent re-triggering

    result_df = pd.DataFrame(legs)
    if not result_df.empty:
        result_df = result_df.set_index('confirmed_time')
    return result_df


def compute_fib_levels(
    leg: Dict[str, Any],
    fib_min: float = 0.71,
    fib_max: float = 0.75,
) -> Dict[str, float]:
    """
    Compute Fibonacci retracement levels on an active leg.

    For SHORT leg (down):
      origin = swing_high, destination = swing_low, range = origin - destination
      fib_50 = destination + 0.50 * range
      fib_71 = destination + fib_min * range
      fib_75 = destination + fib_max * range
      tp = destination, sl = origin

    For LONG leg (up):
      origin = swing_low, destination = swing_high, range = destination - origin
      fib_50 = destination - 0.50 * range
      fib_71 = destination - fib_min * range
      fib_75 = destination - fib_max * range
      tp = destination, sl = origin

    Parameters
    ----------
    leg : dict
        Leg dict with 'direction', 'origin', 'destination' keys.
    fib_min : float
        Fib level for fib_71 (default 0.71).
    fib_max : float
        Fib level for fib_75 (default 0.75).

    Returns
    -------
    dict with keys: fib_50, fib_71, fib_75, tp, sl
    """
    direction = leg['direction']
    origin = float(leg['origin'])
    destination = float(leg['destination'])

    if direction == 'short':
        # origin = high, destination = low
        range_val = origin - destination
        fib_50 = destination + 0.50 * range_val
        fib_71 = destination + fib_min * range_val
        fib_75 = destination + fib_max * range_val
        tp = destination
        sl = origin
    else:  # long
        # origin = low, destination = high
        range_val = destination - origin
        fib_50 = destination - 0.50 * range_val
        fib_71 = destination - fib_min * range_val
        fib_75 = destination - fib_max * range_val
        tp = destination
        sl = origin

    return {
        'fib_50': fib_50,
        'fib_71': fib_71,
        'fib_75': fib_75,
        'tp': tp,
        'sl': sl,
    }
