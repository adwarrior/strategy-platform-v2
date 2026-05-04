"""
MoboBandsPro — Python port of the MoboBandsPro NinjaTrader 8 indicator/strategy.

Logic:
  Computes a Detrended Price Oscillator (DPO) and a Bollinger Band on that DPO
  (the "Mobo" band). A zone state (bull/bear) latches when DPO breaks outside the
  band and carries forward until the next breakout. A "hook" signal fires when the
  DPO re-enters the zone (hooks back toward center) without the pivot having
  touched the center line (optional filter).

  LONG signal:  bull zone, DPO > center, DPO < upper band, hook shape up, slope up
  SHORT signal: bear zone, DPO < center, DPO > lower band, hook shape down, slope dn

  Filters:
    - Band-width filter: only signal when bandwidth > rolling-avg * multiplier
    - Middle-band hook: optionally suppress signals where hook pivot touched center
    - Slope threshold: minimum center-line slope magnitude required

  Entry:  market open of bar following the signal bar (same timing as NT8 strategy)
  Exit:   fixed profit_ticks / stop_ticks bracket; EOD exit at 15:59

Source:   /home/ad/Scripts/indicators/MoboBandsPro.cs  (NT8 C#)
          /home/ad/Scripts/strategies/MoboBandsProStrategy.cs (NT8 C#)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register


# ---------------------------------------------------------------------------
# JMA computation
# ---------------------------------------------------------------------------

def _compute_jma(
    df: pd.DataFrame,
    period: int,
    phase: float,
    fl_period: int,
    up_pct: float = 90.0,
    dn_pct: float = 10.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(df)
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values

    phase_ratio = max(-100.0, min(100.0, phase)) / 100.0 + 1.5
    denom = 0.45 * (period - 1)
    beta  = denom / (denom + 2.0)
    r     = np.sqrt(max(0.5 * (period - 1), 1.0))
    len1  = np.log(r) / np.log(2.0) + 2.0 if r > 1.0 else 2.0
    pow1  = max(len1 - 2.0, 0.5)
    alpha = beta ** pow1

    jma_out   = np.empty(n, dtype=np.float64)
    upper_out = np.empty(n, dtype=np.float64)
    lower_out = np.empty(n, dtype=np.float64)

    ha_open = ha_close = 0.0
    ha_init = False
    e0 = e1 = e2 = jma_val = 0.0
    jma_init = False
    window: list[float] = []

    for i in range(n):
        ohlc4 = (o[i] + h[i] + l[i] + c[i]) / 4.0
        if not ha_init:
            ha_open  = ohlc4
            ha_close = ohlc4
            ha_init  = True
        else:
            ha_open  = (ha_open + ha_close) / 2.0
            ha_close = ohlc4
        price = max(ha_close, h[i]) if ha_close >= ha_open else min(ha_close, l[i])

        if not jma_init:
            e0       = price
            e1       = 0.0
            e2       = 0.0
            jma_val  = price
            jma_init = True
        else:
            e0      = (1.0 - beta) * price + beta * e0
            e1      = (price - e0) * (1.0 - beta) + beta * e1
            ma1     = e0 + phase_ratio * e1
            k       = 1.0 - alpha
            e2      = (ma1 - jma_val) * k * k + alpha * alpha * e2
            jma_val = jma_val + e2

        window.append(jma_val)
        if len(window) > fl_period:
            window.pop(0)

        jma_out[i] = jma_val
        if len(window) >= 2:
            sw = sorted(window)
            m  = len(sw)
            def _pct(p: float) -> float:
                idx_f = p * (m - 1)
                lo    = int(idx_f)
                hi    = lo + 1 if lo + 1 < m else lo
                return sw[lo] + (idx_f - lo) * (sw[hi] - sw[lo])
            upper_out[i] = _pct(up_pct / 100.0)
            lower_out[i] = _pct(dn_pct / 100.0)
        else:
            upper_out[i] = jma_val
            lower_out[i] = jma_val

    return jma_out, upper_out, lower_out


# ---------------------------------------------------------------------------
# Indicator computation
# ---------------------------------------------------------------------------

def _compute_indicators(
    df: pd.DataFrame,
    dpo_period: int,
    mobo_length: int,
    num_dev_up: float,
    num_dev_dn: float,
    enable_wattah_atar: bool = False,
    wa_fast: int = 10,
    wa_slow: int = 30,
    wa_channel: int = 30,
    wa_sensitivity: int = 150,
    wa_mult: float = 2.0,
    enable_jurik_filter: bool = False,
    jurik_period: int = 35,
    jurik_phase: float = 0.0,
    jurik_fl_period: int = 35,
) -> pd.DataFrame:
    """
    Returns a DataFrame with DPO, center, upper/lower bands, valc, bandwidth,
    and optional Wattah Atar columns (wa_trend_up, wa_trend_dn, wa_explosion).
    Uses population std-dev (ddof=0) to match NT8's built-in StdDev.
    """
    close = df['close']
    sft   = dpo_period // 2 + 1

    # DPO: close minus the slow-leg SMA shifted back by sft bars
    sma_close = close.rolling(dpo_period).mean()
    dpo       = close - sma_close.shift(sft)

    # Mobo bands on DPO
    center    = dpo.rolling(mobo_length).mean()
    std_dpo   = dpo.rolling(mobo_length).std(ddof=0)
    up_band   = center + num_dev_up * std_dpo
    dn_band   = center - num_dev_dn * std_dpo
    bandwidth = up_band - dn_band

    # Zone state: latch on breakout, carry forward inside bands
    zone = pd.Series(np.nan, index=dpo.index)
    zone[dpo > up_band] = 1.0
    zone[dpo < dn_band] = -1.0
    valc = zone.ffill().fillna(0.0)

    # Wattah Atar Explosion — mirrors NT8: EMA(EMA(fast), 9) - EMA(EMA(slow), 9)
    if enable_wattah_atar:
        fast_ema  = close.ewm(span=wa_fast, adjust=False).mean().ewm(span=9, adjust=False).mean()
        slow_ema  = close.ewm(span=wa_slow, adjust=False).mean().ewm(span=9, adjust=False).mean()
        macd      = fast_ema - slow_ema
        t1        = (macd - macd.shift(1)) * wa_sensitivity
        bb_std    = close.rolling(wa_channel).std(ddof=0)
        explosion = 2.0 * wa_mult * bb_std
        wa_up     = t1.clip(lower=0)
        wa_dn     = (-t1).clip(lower=0)
    else:
        wa_up = wa_dn = explosion = pd.Series(np.nan, index=df.index)

    if enable_jurik_filter:
        jma_vals, jma_upper, jma_lower = _compute_jma(df, jurik_period, jurik_phase, jurik_fl_period)
    else:
        nan_col = np.full(len(df), np.nan)
        jma_vals = jma_upper = jma_lower = nan_col

    return pd.DataFrame({
        'dpo':          dpo,
        'center':       center,
        'up_band':      up_band,
        'dn_band':      dn_band,
        'bandwidth':    bandwidth,
        'valc':         valc,
        'wa_trend_up':  wa_up,
        'wa_trend_dn':  wa_dn,
        'wa_explosion': explosion,
        'jma':          jma_vals,
        'jma_upper':    jma_upper,
        'jma_lower':    jma_lower,
    }, index=df.index)


# ---------------------------------------------------------------------------
# Divergence detection
# ---------------------------------------------------------------------------

def _detect_divergence(
    is_bull: bool,
    i: int,
    price_arr: np.ndarray,  # low_arr for bull, high_arr for bear
    dpo_arr: np.ndarray,
    lookback: int,
) -> bool:
    """
    Returns True if classic divergence is present ending at bar i.
    Bull: price lower swing-low, DPO higher swing-low (use low_arr).
    Bear: price higher swing-high, DPO lower swing-high (use high_arr).
    """
    pivots = []
    for j in range(i - 1, max(i - lookback, 1), -1):
        if is_bull:
            if price_arr[j] < price_arr[j - 1] and price_arr[j] < price_arr[j + 1]:
                pivots.append(j)
        else:
            if price_arr[j] > price_arr[j - 1] and price_arr[j] > price_arr[j + 1]:
                pivots.append(j)
        if len(pivots) == 2:
            break
    if len(pivots) < 2:
        return False
    j1, j2 = pivots[0], pivots[1]  # j1 is more recent
    if is_bull:
        return price_arr[j1] < price_arr[j2] and dpo_arr[j1] > dpo_arr[j2]
    else:
        return price_arr[j1] > price_arr[j2] and dpo_arr[j1] < dpo_arr[j2]


# ---------------------------------------------------------------------------
# Signal generation + trade simulation (single integrated pass)
# ---------------------------------------------------------------------------
# Combining into one pass ensures:
#   1. Cooldown (bars_between_trades) resets on trade EXIT, matching NT's
#      OnPositionUpdate behaviour — not on the signal bar.
#   2. Session-close logic (17:00 ET exit, 18:00 ET resume) mirrors NT's
#      IsExitOnSessionCloseStrategy without blocking overnight signals.

def _in_session_gap(ts: pd.Timestamp) -> bool:
    """True for bars in the CME Globex daily settlement break (22:00–22:59 UTC = 17:00–17:59 ET)."""
    return ts.hour == 22


def _generate_signals(
    df: pd.DataFrame,
    ind: pd.DataFrame,
    params: Dict[str, Any],
    tick_size: float = 0.25,
    tick_value: float = 5.0,
    commission: float = 0.0,
) -> List[Dict]:
    """Kept for backward-compat — delegates to _run_backtest_loop."""
    return _run_backtest_loop(df, ind, params, tick_size, tick_value, commission)


def _build_bar_tick_map(
    df: pd.DataFrame,
    raw_ticks: pd.DataFrame,
    bar_size: int,
) -> List[np.ndarray]:
    """
    For each bar in df, return the array of tick prices that belong to it.
    Bar boundaries are determined by tick count (bar_size ticks per bar),
    aligned from the start of the tick series — matching load_tick_bars logic.

    Returns a list of length len(df), each element a 1-D float64 array of prices.
    """
    prices = raw_ticks['price'].to_numpy(dtype=np.float64)
    times  = raw_ticks.index.to_numpy()          # datetime64[ns]
    n_bars = len(df)

    bar_tick_prices: List[np.ndarray] = []
    for b in range(n_bars):
        start_tick = b * bar_size
        end_tick   = start_tick + bar_size
        if end_tick > len(prices):
            bar_tick_prices.append(prices[start_tick:] if start_tick < len(prices) else np.empty(0))
        else:
            bar_tick_prices.append(prices[start_tick:end_tick])

    return bar_tick_prices


def _run_backtest_loop(
    df: pd.DataFrame,
    ind: pd.DataFrame,
    params: Dict[str, Any],
    tick_size: float,
    tick_value: float,
    commission: float,
    calculate_mode: str = 'on_bar_close',
) -> List[Dict]:
    """
    Integrated signal-generation + trade-simulation loop.
    Matches NT8 MoBoBandsProV101 behaviour:
      - EOD exit: force-close any open position at the 17:00 ET session close bar.
      - Session gap: no new entries between 17:00 and 18:00 ET (gap bar excluded).
      - Cooldown: bars_between_trades counted from exit bar, not signal bar.
    """
    dpo_period   = int(params['dpo_period'])
    mobo_length  = int(params['mobo_length'])
    hook_lb      = int(params.get('hook_lookback', 2))
    slope_lb     = int(params.get('slope_lookback', 5))
    slope_thresh = float(params.get('slope_threshold', 0.0))
    bw_period    = int(params.get('bw_period', 50))
    bw_mult      = float(params.get('bw_multiplier', 1.0))
    enable_bw    = bool(params.get('enable_bw_filter', True))
    enable_mid   = bool(params.get('enable_middle_band_hook', True))
    profit_tks   = int(params['profit_ticks'])
    stop_tks     = int(params['stop_ticks'])
    bars_cd      = int(params.get('bars_between_trades', 2))
    enable_longs  = bool(params.get('enable_longs', True))
    enable_shorts = bool(params.get('enable_shorts', True))

    require_color_change = bool(params.get('require_color_change', False))
    enable_div           = bool(params.get('enable_divergence_filter', False))
    div_lookback         = int(params.get('divergence_lookback', 20))
    enable_time_filter   = bool(params.get('enable_time_filter', False))
    trade_start_time     = str(params.get('trade_start_time', '09:30'))
    trade_end_time       = str(params.get('trade_end_time', '15:59'))
    enable_wa            = bool(params.get('enable_wattah_atar', False))
    wa_dead              = float(params.get('wa_dead_zone', 200))
    enable_jurik         = bool(params.get('enable_jurik_filter', False))
    jurik_period         = int(params.get('jurik_period', 35))
    jurik_phase          = float(params.get('jurik_phase', 0.0))
    jurik_fl_period      = int(params.get('jurik_fl_period', 35))

    lb  = max(hook_lb, 2)
    slb = lb - 1

    sft     = dpo_period // 2 + 1
    min_bar = dpo_period + sft + mobo_length + lb + slope_lb + 5

    o_arr = df['open'].values
    h_arr = df['high'].values
    l_arr = df['low'].values
    c_arr = df['close'].values
    idx   = df.index
    n     = len(df)

    dpo_arr   = ind['dpo'].values
    ctr_arr   = ind['center'].values
    up_arr    = ind['up_band'].values
    dn_arr    = ind['dn_band'].values
    bw_arr    = ind['bandwidth'].values
    valc_arr  = ind['valc'].values
    wa_up_arr = ind['wa_trend_up'].values
    wa_dn_arr = ind['wa_trend_dn'].values
    jma_arr        = ind['jma'].values
    jma_upper_arr  = ind['jma_upper'].values
    jma_lower_arr  = ind['jma_lower'].values

    bw_ma = pd.Series(bw_arr, index=idx).rolling(bw_period).mean().values

    pt = profit_tks * tick_size
    st = stop_tks   * tick_size

    trades: List[Dict] = []

    # Trade state
    in_trade  = False
    ep = sl = tp = entry_time = direction = band = divergence = None

    # Cooldown: bars elapsed since last exit (start ready)
    bars_since_exit = bars_cd

    start = min_bar + bw_period + 2

    for i in range(start, n):
        ts = idx[i]

        # ── Session-close exit at 17:00 ET ──────────────────────────────────
        # Exit any open position; then skip this bar for new entries.
        if _in_session_gap(ts):
            if in_trade:
                xp    = c_arr[i]
                pnl_t = (xp - ep if direction == 'long' else ep - xp) / tick_size
                trades.append(_make_trade(entry_time, ts, direction,
                                          ep, xp, pnl_t, tick_value, commission, 'EOD',
                                          band, divergence))
                in_trade = False
                bars_since_exit = 0
            continue  # no new entries during the 17:00-18:00 gap

        # ── Manage open position ─────────────────────────────────────────────
        if in_trade:
            closed = False
            if direction == 'long':
                if l_arr[i] <= sl:
                    pnl_t = (sl - ep) / tick_size
                    trades.append(_make_trade(entry_time, ts, direction,
                                              ep, sl, pnl_t, tick_value, commission, 'SL',
                                              band, divergence))
                    closed = True
                elif h_arr[i] >= tp:
                    pnl_t = (tp - ep) / tick_size
                    trades.append(_make_trade(entry_time, ts, direction,
                                              ep, tp, pnl_t, tick_value, commission, 'TP',
                                              band, divergence))
                    closed = True
            else:
                if h_arr[i] >= sl:
                    pnl_t = (ep - sl) / tick_size
                    trades.append(_make_trade(entry_time, ts, direction,
                                              ep, sl, pnl_t, tick_value, commission, 'SL',
                                              band, divergence))
                    closed = True
                elif l_arr[i] <= tp:
                    pnl_t = (ep - tp) / tick_size
                    trades.append(_make_trade(entry_time, ts, direction,
                                              ep, tp, pnl_t, tick_value, commission, 'TP',
                                              band, divergence))
                    closed = True
            if closed:
                in_trade = False
                bars_since_exit = 0  # cooldown starts from exit bar (mirrors NT OnPositionUpdate)
                # Fall through — allow signal check on same bar the position closed (mirrors NT)
            else:
                continue  # still in trade, skip signal check

        # ── Cooldown ─────────────────────────────────────────────────────────
        # NT guard: CurrentBar - lastFlatBar <= BarsBetweenTrades (skips <= bars_cd bars)
        bars_since_exit += 1
        if bars_since_exit <= bars_cd:
            continue

        # ── Signal guards ─────────────────────────────────────────────────────
        if calculate_mode == 'on_bar_close' and i >= n - 1:  # need i+1 for next-bar entry
            continue
        if i < slb + 2:
            continue

        # User-defined time filter
        if enable_time_filter:
            bar_time = f"{ts.hour:02d}:{ts.minute:02d}"
            if bar_time < trade_start_time or bar_time > trade_end_time:
                continue

        dpo_i   = dpo_arr[i]
        ctr_i   = ctr_arr[i]
        up_i    = up_arr[i]
        dn_i    = dn_arr[i]
        valc_i  = valc_arr[i]
        bw_i    = bw_arr[i]
        bw_ma_i = bw_ma[i]

        dpo_piv = dpo_arr[i - slb]
        dpo_pre = dpo_arr[i - slb - 1]
        ctr_piv = ctr_arr[i - slb]

        bw_ok = True
        if enable_bw and bw_mult > 0.0 and not np.isnan(bw_ma_i):
            bw_ok = bw_i > bw_ma_i * bw_mult

        slope = 0.0
        if slope_lb > 0 and i >= slope_lb:
            slope = (ctr_arr[i] - ctr_arr[i - slope_lb]) / slope_lb

        slope_now = slope_prev = 0.0
        color_changed = False
        if i >= 2:
            slope_now  = ctr_arr[i]     - ctr_arr[i - 1]
            slope_prev = ctr_arr[i - 1] - ctr_arr[i - 2]
            color_changed = ((slope_now > 0 and slope_prev <= 0) or
                             (slope_now < 0 and slope_prev >= 0))

        # ── SELL hook ─────────────────────────────────────────────────────────
        signal_fired = False
        if (enable_shorts
                and valc_i <= -1
                and dpo_i < ctr_i
                and dpo_i > dn_i
                and dpo_i < dpo_piv
                and dpo_piv > dpo_pre
                and (slope_thresh <= 0.0 or slope < -slope_thresh)
                and (enable_mid or dpo_piv < ctr_piv)
                and bw_ok
                and (not require_color_change or (slope_now < 0 and color_changed))
                and (not enable_wa or wa_dn_arr[i] > wa_dead)
                and (not enable_jurik or (jma_arr[i] < jma_lower_arr[i] and (i == 0 or jma_arr[i] < jma_arr[i - 1])))):
            bear_div = _detect_divergence(False, i, h_arr, dpo_arr, div_lookback)
            if not enable_div or bear_div:
                ep         = c_arr[i] if calculate_mode != 'on_bar_close' else o_arr[i + 1]
                entry_time = idx[i]   if calculate_mode != 'on_bar_close' else idx[i + 1]
                direction = 'short'
                band      = 'middle' if dpo_piv >= ctr_piv else 'lower'
                divergence = bear_div
                sl = ep + st
                tp = ep - pt
                in_trade = True
                bars_since_exit = bars_cd  # won't be used until exit, but reset for safety
                signal_fired = True

        # ── BUY hook ──────────────────────────────────────────────────────────
        if (not signal_fired
                and enable_longs
                and valc_i >= 1
                and dpo_i > ctr_i
                and dpo_i < up_i
                and dpo_i > dpo_piv
                and dpo_piv < dpo_pre
                and (slope_thresh <= 0.0 or slope > slope_thresh)
                and (enable_mid or dpo_piv > ctr_piv)
                and bw_ok
                and (not require_color_change or (slope_now > 0 and color_changed))
                and (not enable_wa or wa_up_arr[i] > wa_dead)
                and (not enable_jurik or (jma_arr[i] > jma_upper_arr[i] and (i == 0 or jma_arr[i] > jma_arr[i - 1])))):
            bull_div = _detect_divergence(True, i, l_arr, dpo_arr, div_lookback)
            if not enable_div or bull_div:
                ep         = c_arr[i] if calculate_mode != 'on_bar_close' else o_arr[i + 1]
                entry_time = idx[i]   if calculate_mode != 'on_bar_close' else idx[i + 1]
                direction = 'long'
                band      = 'middle' if dpo_piv <= ctr_piv else 'upper'
                divergence = bull_div
                sl = ep - st
                tp = ep + pt
                in_trade = True

    return trades


def _run_backtest_loop_tick(
    df: pd.DataFrame,
    ind: pd.DataFrame,
    params: Dict[str, Any],
    tick_size: float,
    tick_value: float,
    commission: float,
    bar_tick_prices: List[np.ndarray],
) -> List[Dict]:
    """
    On-each-tick backtest: for each bar, replays individual ticks.

    For each tick within bar i, we temporarily substitute the tick price as
    `close[i]` and recompute DPO/bands/valc for bar i only (bars 0..i-1 are
    fixed).  Signal conditions are checked on every tick; entry fires at the
    tick price, matching NT8 Calculate.OnEachTick behaviour.

    Open-position management (SL/TP/EOD) uses bar-level H/L exactly as in
    _run_backtest_loop because NT fills SL/TP at the bar level even in
    OnEachTick mode.
    """
    dpo_period   = int(params['dpo_period'])
    mobo_length  = int(params['mobo_length'])
    hook_lb      = int(params.get('hook_lookback', 2))
    slope_lb     = int(params.get('slope_lookback', 5))
    slope_thresh = float(params.get('slope_threshold', 0.0))
    bw_period    = int(params.get('bw_period', 50))
    bw_mult      = float(params.get('bw_multiplier', 1.0))
    enable_bw    = bool(params.get('enable_bw_filter', True))
    enable_mid   = bool(params.get('enable_middle_band_hook', True))
    profit_tks   = int(params['profit_ticks'])
    stop_tks     = int(params['stop_ticks'])
    bars_cd      = int(params.get('bars_between_trades', 2))
    enable_longs  = bool(params.get('enable_longs', True))
    enable_shorts = bool(params.get('enable_shorts', True))
    enable_div    = bool(params.get('enable_divergence_filter', False))
    div_lookback  = int(params.get('divergence_lookback', 20))
    enable_time_filter = bool(params.get('enable_time_filter', False))
    trade_start_time   = str(params.get('trade_start_time', '09:30'))
    trade_end_time     = str(params.get('trade_end_time', '15:59'))
    enable_wa            = bool(params.get('enable_wattah_atar', False))
    wa_dead              = float(params.get('wa_dead_zone', 200))
    num_dev_up           = float(params.get('num_dev_up', 0.8))
    num_dev_dn           = float(params.get('num_dev_dn', 0.8))
    require_color_change = bool(params.get('require_color_change', False))
    enable_jurik         = bool(params.get('enable_jurik_filter', False))
    jurik_period         = int(params.get('jurik_period', 35))
    jurik_phase          = float(params.get('jurik_phase', 0.0))
    jurik_fl_period      = int(params.get('jurik_fl_period', 35))

    lb  = max(hook_lb, 2)
    slb = lb - 1

    sft     = dpo_period // 2 + 1
    min_bar = dpo_period + sft + mobo_length + lb + slope_lb + 5

    o_arr = df['open'].values
    h_arr = df['high'].values
    l_arr = df['low'].values
    c_arr = df['close'].values
    idx   = df.index
    n     = len(df)

    # Precomputed base indicator arrays (bar-close values, used for pivot lookback)
    dpo_arr   = ind['dpo'].values.copy()
    ctr_arr   = ind['center'].values.copy()
    up_arr    = ind['up_band'].values.copy()
    dn_arr    = ind['dn_band'].values.copy()
    bw_arr    = ind['bandwidth'].values.copy()
    valc_arr  = ind['valc'].values.copy()
    wa_up_arr = ind['wa_trend_up'].values
    wa_dn_arr = ind['wa_trend_dn'].values
    jma_arr        = ind['jma'].values
    jma_upper_arr  = ind['jma_upper'].values
    jma_lower_arr  = ind['jma_lower'].values

    bw_ma_base = pd.Series(bw_arr, index=idx).rolling(bw_period).mean().values

    pt = profit_tks * tick_size
    st = stop_tks   * tick_size

    # Precompute rolling SMA of close for DPO (fixed for all closed bars)
    close_s  = pd.Series(c_arr, index=idx)
    sma_close_arr = close_s.rolling(dpo_period).mean().values  # length n

    # Precompute previous mobo_length dpo values for the rolling window at each bar
    # dpo_window[i] = dpo values for bars [i-mobo_length+1 .. i-1] (all closed, excluding bar i)
    # We'll compute per-tick dpo_i and update the window incrementally.

    trades: List[Dict] = []

    in_trade  = False
    ep = sl = tp = entry_time = direction = band = divergence = None
    bars_since_exit = bars_cd

    start = min_bar + bw_period + 2

    for i in range(start, n):
        ts = idx[i]

        # ── EOD exit ──────────────────────────────────────────────────────────
        if _in_session_gap(ts):
            if in_trade:
                xp    = c_arr[i]
                pnl_t = (xp - ep if direction == 'long' else ep - xp) / tick_size
                trades.append(_make_trade(entry_time, ts, direction,
                                          ep, xp, pnl_t, tick_value, commission, 'EOD',
                                          band, divergence))
                in_trade = False
                bars_since_exit = 0
            continue

        # ── Manage open position (bar-level) ──────────────────────────────────
        if in_trade:
            closed = False
            if direction == 'long':
                if l_arr[i] <= sl:
                    pnl_t = (sl - ep) / tick_size
                    trades.append(_make_trade(entry_time, ts, direction,
                                              ep, sl, pnl_t, tick_value, commission, 'SL',
                                              band, divergence))
                    closed = True
                elif h_arr[i] >= tp:
                    pnl_t = (tp - ep) / tick_size
                    trades.append(_make_trade(entry_time, ts, direction,
                                              ep, tp, pnl_t, tick_value, commission, 'TP',
                                              band, divergence))
                    closed = True
            else:
                if h_arr[i] >= sl:
                    pnl_t = (ep - sl) / tick_size
                    trades.append(_make_trade(entry_time, ts, direction,
                                              ep, sl, pnl_t, tick_value, commission, 'SL',
                                              band, divergence))
                    closed = True
                elif l_arr[i] <= tp:
                    pnl_t = (ep - tp) / tick_size
                    trades.append(_make_trade(entry_time, ts, direction,
                                              ep, tp, pnl_t, tick_value, commission, 'TP',
                                              band, divergence))
                    closed = True
            if closed:
                in_trade = False
                bars_since_exit = 0
            else:
                continue

        bars_since_exit += 1
        if bars_since_exit <= bars_cd:
            continue

        if i < slb + 2:
            continue

        if enable_time_filter:
            bar_time = f"{ts.hour:02d}:{ts.minute:02d}"
            if bar_time < trade_start_time or bar_time > trade_end_time:
                continue

        # ── Per-tick signal check ─────────────────────────────────────────────
        # Pivot lookback values come from previous bars (fixed at bar close).
        dpo_piv = dpo_arr[i - slb]
        dpo_pre = dpo_arr[i - slb - 1]
        ctr_piv = ctr_arr[i - slb]

        # Rolling window of the mobo_length dpo values ending at bar i-1
        # Used to compute per-tick center and std when tick dpo replaces bar[i].
        win_start = i - mobo_length + 1
        if win_start < 0:
            continue
        dpo_window = dpo_arr[win_start:i]  # length mobo_length - 1

        # Slope uses previous bar's center values (fixed)
        slope = 0.0
        if slope_lb > 0 and i >= slope_lb:
            slope = (ctr_arr[i] - ctr_arr[i - slope_lb]) / slope_lb

        slope_now = slope_prev = 0.0
        color_changed = False
        if i >= 2:
            slope_now  = ctr_arr[i]     - ctr_arr[i - 1]
            slope_prev = ctr_arr[i - 1] - ctr_arr[i - 2]
            color_changed = ((slope_now > 0 and slope_prev <= 0) or
                             (slope_now < 0 and slope_prev >= 0))

        # SMA of close at bar i-sft (fixed — sft >= 1, so this is a closed bar)
        sma_close_i = sma_close_arr[i - sft] if (i - sft) >= 0 else np.nan

        # BW MA uses previous bars only (bw_arr at i uses closed bar bw)
        bw_ma_i = bw_ma_base[i]

        tick_prices = bar_tick_prices[i] if i < len(bar_tick_prices) else np.empty(0)
        if len(tick_prices) == 0:
            # No tick data for this bar — fall back to bar-close check
            tick_prices = np.array([c_arr[i]])

        # valc carries across ticks within the bar (mirrors NT OnEachTick state)
        valc_intra = valc_arr[i - 1]  # start from last closed bar's state

        signal_fired_this_bar = False
        for tick_price in tick_prices:
            if signal_fired_this_bar:
                break

            if np.isnan(sma_close_i):
                continue

            # Recompute DPO, center, bands for this tick
            dpo_t = tick_price - sma_close_i

            # Rolling mobo window: dpo_window (i-mobo_length+1 .. i-1) + dpo_t
            full_win = np.append(dpo_window, dpo_t)  # length mobo_length
            ctr_t    = full_win.mean()
            std_t    = full_win.std(ddof=0)
            up_t     = ctr_t + num_dev_up * std_t
            dn_t     = ctr_t - num_dev_dn * std_t
            bw_t     = up_t - dn_t

            # valc latches on breakout, persists until next breakout (mirrors NT zone state across ticks)
            if dpo_t > up_t:
                valc_intra = 1.0
            elif dpo_t < dn_t:
                valc_intra = -1.0
            valc_t = valc_intra

            bw_ok = True
            if enable_bw and bw_mult > 0.0 and not np.isnan(bw_ma_i):
                bw_ok = bw_t > bw_ma_i * bw_mult

            # ── SELL hook ────────────────────────────────────────────────────
            if (enable_shorts
                    and valc_t <= -1
                    and dpo_t < ctr_t and dpo_t > dn_t
                    and dpo_t < dpo_piv and dpo_piv > dpo_pre
                    and (slope_thresh <= 0.0 or slope < -slope_thresh)
                    and (enable_mid or dpo_piv < ctr_piv)
                    and bw_ok
                    and (not require_color_change or (slope_now < 0 and color_changed))
                    and (not enable_wa or wa_dn_arr[i] > wa_dead)
                    and (not enable_jurik or (jma_arr[i] < jma_lower_arr[i] and (i == 0 or jma_arr[i] < jma_arr[i - 1])))):
                bear_div = _detect_divergence(False, i, h_arr, dpo_arr, div_lookback)
                if not enable_div or bear_div:
                    ep         = tick_price
                    entry_time = ts
                    direction  = 'short'
                    band_str   = 'middle' if dpo_piv >= ctr_piv else 'lower'
                    band       = band_str
                    divergence = bear_div
                    sl = ep + st
                    tp = ep - pt
                    in_trade = True
                    bars_since_exit = bars_cd
                    signal_fired_this_bar = True
                    continue

            # ── BUY hook ─────────────────────────────────────────────────────
            if (not signal_fired_this_bar
                    and enable_longs
                    and valc_t >= 1
                    and dpo_t > ctr_t and dpo_t < up_t
                    and dpo_t > dpo_piv and dpo_piv < dpo_pre
                    and (slope_thresh <= 0.0 or slope > slope_thresh)
                    and (enable_mid or dpo_piv > ctr_piv)
                    and bw_ok
                    and (not require_color_change or (slope_now > 0 and color_changed))
                    and (not enable_wa or wa_up_arr[i] > wa_dead)
                    and (not enable_jurik or (jma_arr[i] > jma_upper_arr[i] and (i == 0 or jma_arr[i] > jma_arr[i - 1])))):
                bull_div = _detect_divergence(True, i, l_arr, dpo_arr, div_lookback)
                if not enable_div or bull_div:
                    ep         = tick_price
                    entry_time = ts
                    direction  = 'long'
                    band_str   = 'middle' if dpo_piv <= ctr_piv else 'upper'
                    band       = band_str
                    divergence = bull_div
                    sl = ep - st
                    tp = ep + pt
                    in_trade = True
                    signal_fired_this_bar = True

    return trades


def _simulate_trades(
    df: pd.DataFrame,
    entries: List[Dict],
    profit_ticks: int,
    stop_ticks: int,
    tick_size: float,
    tick_value: float,
    commission: float,
) -> List[Dict]:
    """Legacy shim — entries are now trade dicts returned by _run_backtest_loop."""
    return entries


def _make_trade(entry_time, exit_time, direction, ep, xp, pnl_ticks,
                tick_value, commission, reason, band: str = '', divergence: bool = False):
    return {
        'session_date': entry_time.date(),
        'day_of_week':  entry_time.day_name(),
        'entry_time':   entry_time,
        'exit_time':    exit_time,
        'direction':    direction,
        'entry_price':  ep,
        'exit_price':   xp,
        'pnl_ticks':    pnl_ticks,
        'pnl':          pnl_ticks * tick_value - commission,
        'exit_reason':  reason,
        'band':         band,
        'divergence':   divergence,
    }


# ---------------------------------------------------------------------------
# Statistics (same monthly-Sharpe convention as rest of platform)
# ---------------------------------------------------------------------------

WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


def _daily_sortino(d_vals: np.ndarray) -> float:
    if len(d_vals) < 2:
        return 0.0
    neg = d_vals[d_vals < 0]
    if len(neg) == 0:
        return float('inf')
    downside_std = np.sqrt(np.mean(neg ** 2))
    return float(d_vals.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0


def _max_consec(is_win: np.ndarray):
    max_w = max_l = cur_w = cur_l = 0
    for w in is_win:
        if w:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        if cur_w > max_w: max_w = cur_w
        if cur_l > max_l: max_l = cur_l
    return max_w, max_l


def _max_time_to_recover(dates: list, pnls: np.ndarray) -> float:
    equity = np.cumsum(pnls)
    peak   = np.maximum.accumulate(equity)
    max_recovery = 0
    dd_start = None
    for i in range(len(equity)):
        if equity[i] < peak[i]:
            if dd_start is None:
                dd_start = dates[i - 1] if i > 0 else dates[i]
        else:
            if dd_start is not None:
                delta = (pd.Timestamp(dates[i]) - pd.Timestamp(dd_start)).days
                if delta > max_recovery:
                    max_recovery = delta
                dd_start = None
    if dd_start is not None:
        delta = (pd.Timestamp(dates[-1]) - pd.Timestamp(dd_start)).days
        if delta > max_recovery:
            max_recovery = delta
    return float(max_recovery)


def _summarise(trades: List[Dict], total_sessions: int = 0) -> dict:
    _empty = {
        'trades': 0, 'win_rate': 0.0, 'net_pnl': 0.0,
        'avg_win': 0.0, 'avg_loss': 0.0, 'profit_factor': 0.0,
        'max_drawdown': 0.0, 'sharpe': 0.0, 'sortino': 0.0,
        'ulcer_index': 0.0, 'r_squared': 0.0, 'pct_months_profit': 0.0,
        'longest_flat_days': 0, 'total_commission': 0.0,
        'start_date': '', 'end_date': '',
        **{f'{d[:3].lower()}_pnl': 0.0  for d in WEEKDAYS},
        **{f'{d[:3].lower()}_trades': 0 for d in WEEKDAYS},
    }
    if not trades:
        return _empty

    pnls   = np.array([t['pnl'] for t in trades], dtype=float)
    n      = len(pnls)
    wins   = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    even   = pnls[pnls == 0]

    cum    = np.cumsum(pnls)
    peak   = np.maximum.accumulate(cum)
    max_dd = float(-(peak - cum).max())

    daily_map: Dict = {}
    for t in trades:
        daily_map[t['session_date']] = daily_map.get(t['session_date'], 0.0) + t['pnl']
    sorted_dates = sorted(daily_map)
    d_vals       = np.array([daily_map[d] for d in sorted_dates], dtype=float)

    n_zero = max(0, total_sessions - len(d_vals))
    d_vals_padded = np.concatenate([d_vals, np.zeros(n_zero)]) if n_zero > 0 else d_vals

    trade_dates_all = [t['session_date'] for t in trades]
    first_ym = (trade_dates_all[0].year,  trade_dates_all[0].month)
    last_ym  = (trade_dates_all[-1].year, trade_dates_all[-1].month)

    monthly_full: Dict = {}
    cur_ym = first_ym
    while cur_ym <= last_ym:
        monthly_full[cur_ym] = 0.0
        y, mo = cur_ym
        cur_ym = (y + 1, 1) if mo == 12 else (y, mo + 1)

    for t in trades:
        ym = (t['session_date'].year, t['session_date'].month)
        monthly_full[ym] = monthly_full.get(ym, 0.0) + t['pnl']

    m_arr = np.array(list(monthly_full.values()), dtype=float)
    if len(m_arr) >= 6:
        m_std  = float(m_arr.std(ddof=1))
        sharpe = float(m_arr.mean() / m_std) if m_std > 0 else 1.0
    else:
        std    = d_vals_padded.std(ddof=1) if len(d_vals_padded) > 1 else 0.0
        sharpe = float((d_vals_padded.mean() / std) * np.sqrt(252)) if std > 0 else 0.0

    max_consec_w, max_consec_l = _max_consec(pnls > 0)
    n_days             = len(daily_map)
    avg_trades_per_day = n / n_days if n_days > 0 else 0.0
    profit_per_month   = float(np.mean(list(monthly_full.values()))) if monthly_full else 0.0
    n_months_pos       = sum(1 for v in monthly_full.values() if v > 0)
    pct_months_profit  = float(n_months_pos / len(monthly_full)) if monthly_full else 0.0
    max_recovery       = _max_time_to_recover(trade_dates_all, pnls)
    gross_profit       = float(wins.sum())   if len(wins)   > 0 else 0.0
    gross_loss         = float(losses.sum()) if len(losses) > 0 else 0.0

    dd_series = peak - cum
    ulcer_idx = float(np.sqrt(np.mean(dd_series ** 2)))

    x = np.arange(n, dtype=float)
    try:
        with np.errstate(invalid='ignore', divide='ignore'):
            coef  = np.polyfit(x, cum, 1)
            y_hat = np.polyval(coef, x)
    except (np.linalg.LinAlgError, ValueError):
        coef  = np.array([0.0, 0.0])
        y_hat = np.zeros_like(cum)
    ss_res = float(((cum - y_hat) ** 2).sum())
    ss_tot = float(((cum - cum.mean()) ** 2).sum())
    r_sq   = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    high_dates = []
    peak_val   = -np.inf
    for i, val in enumerate(cum):
        if val > peak_val:
            peak_val = val
            high_dates.append(trade_dates_all[i])
    longest_flat = max(
        (high_dates[j+1] - high_dates[j]).days for j in range(len(high_dates) - 1)
    ) if len(high_dates) >= 2 else 0

    stats = {
        'trades':              n,
        'num_wins':            int(len(wins)),
        'num_losses':          int(len(losses)),
        'num_even':            int(len(even)),
        'win_rate':            float(len(wins) / n),
        'gross_profit':        gross_profit,
        'gross_loss':          gross_loss,
        'net_pnl':             float(pnls.sum()),
        'avg_trade':           float(pnls.mean()),
        'avg_win':             float(wins.mean())   if len(wins)   > 0 else 0.0,
        'avg_loss':            float(losses.mean()) if len(losses) > 0 else 0.0,
        'ratio_win_loss':      float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else 0.0,
        'largest_win':         float(wins.max())    if len(wins)   > 0 else 0.0,
        'largest_loss':        float(losses.min())  if len(losses) > 0 else 0.0,
        'profit_factor':       gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf'),
        'max_drawdown':        max_dd,
        'max_consec_winners':  max_consec_w,
        'max_consec_losers':   max_consec_l,
        'avg_trades_per_day':  round(avg_trades_per_day, 2),
        'profit_per_month':    profit_per_month,
        'max_time_to_recover': max_recovery,
        'sharpe':              sharpe,
        'sortino':             _daily_sortino(d_vals_padded),
        'ulcer_index':         ulcer_idx,
        'r_squared':           r_sq,
        'pct_months_profit':   pct_months_profit,
        'longest_flat_days':   longest_flat,
        'total_commission':    0.0,
        'start_date':          str(trade_dates_all[0]),
        'end_date':            str(trade_dates_all[-1]),
    }

    for day in WEEKDAYS:
        key      = day[:3].lower()
        day_pnls = [t['pnl'] for t in trades if t['day_of_week'] == day]
        stats[f'{key}_pnl']    = float(sum(day_pnls))
        stats[f'{key}_trades'] = len(day_pnls)

    return stats


def _bootstrap_trades(
    trades: List[Dict],
    n_sims: int = 1000,
    seed: int = 42,
    total_sessions: int = 0,
) -> dict:
    _empty = {
        'bs_sharpe_p5': float('nan'), 'bs_pnl_p5':  float('nan'),
        'bs_pnl_p50':   float('nan'), 'bs_wr_p5':   float('nan'),
    }
    if len(trades) < 5:
        return _empty

    rng = np.random.default_rng(seed)
    daily_pnl: Dict = {}
    for t in trades:
        daily_pnl[t['session_date']] = daily_pnl.get(t['session_date'], 0.0) + t['pnl']
    pnls = np.array(list(daily_pnl.values()), dtype=float)

    n_zero = max(0, total_sessions - len(pnls))
    if n_zero > 0:
        pnls = np.concatenate([pnls, np.zeros(n_zero)])
    n = len(pnls)

    idx      = rng.integers(0, n, size=(n_sims, n))
    s_pnls   = pnls[idx]
    net_pnls = s_pnls.sum(axis=1)
    win_rates = (s_pnls > 0).mean(axis=1)
    stds     = s_pnls.std(axis=1, ddof=1)
    means    = s_pnls.mean(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        sharpes  = np.where(stds > 0, (means / stds) * np.sqrt(252), 0.0)
    sharpes = np.nan_to_num(sharpes, nan=0.0)

    return {
        'bs_sharpe_p5': float(np.percentile(sharpes,   5)),
        'bs_pnl_p5':    float(np.percentile(net_pnls,  5)),
        'bs_pnl_p50':   float(np.percentile(net_pnls, 50)),
        'bs_wr_p5':     float(np.percentile(win_rates,  5)),
    }


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

@register
class MoboBandsPro(BaseStrategy):
    """
    MoboBandsPro: DPO Bollinger Band hook-signal strategy.
    Ported from MoboBandsPro.cs (NinjaTrader 8).
    Default instrument: NQ=F (Mini Nasdaq-100, 5M bars).
    """

    name = "mobobands"

    # Supports 5M time bars, 1M time bars, and tick bars.
    # Bar type is selected in the dashboard sidebar; drives symbol list and data loader.
    bar_type            = 'time'                    # default; overridden by dashboard
    supported_bar_types = ['time', '1m', 'tick']    # shown in bar-type selector

    default_params: Dict[str, Any] = {
        'dpo_period': 14,
        'mobo_length': 21,
        'num_dev_up': 0.8,
        'num_dev_dn': 0.8,
        'hook_lookback': 2,
        'slope_lookback': 5,
        'slope_threshold': 0,
        'enable_middle_band_hook': True,
        'require_color_change': False,
        'enable_divergence_filter': False,
        'divergence_lookback': 20,
        'enable_bw_filter': False,
        'bw_period': 50,
        'bw_multiplier': 1,
        'profit_ticks': 40,
        'stop_ticks': 20,
        'bars_between_trades': 2,
        'enable_longs': True,
        'enable_shorts': True,
        'tick_bar_size': 233,
        'enable_time_filter': False,
        'trade_start_time': '09:30',
        'trade_end_time': '15:59',
        'calculate_mode': 'on_bar_close',
        'enable_wattah_atar': False,
        'wa_sensitivity': 150,
        'wa_fast_length': 10,
        'wa_slow_length': 30,
        'wa_channel_length': 30,
        'wa_mult': 2,
        'wa_dead_zone': 200,
        'enable_jurik_filter': False,
        'jurik_period': 35,
        'jurik_phase': 0,
        'jurik_fl_period': 35,
    }

    # NQ=F defaults — overridden by dashboard when symbol changes
    tick_size     = 0.25
    tick_value    = 5.00
    commission_rt = 3.98

    symbol  = 'NQ=F'
    db_host = None  # reads DB_HOST from .env

    # ------------------------------------------------------------------
    # param_grid — all optimizable parameters
    # ------------------------------------------------------------------

    @property
    def param_grid(self) -> Dict[str, Any]:
        return {
            # DPO shape — (min, max, step)
            'dpo_period':    (4, 40, 2),
            'mobo_length':   (5, 50, 5),
            'num_dev_up':    (0.2, 2.0, 0.2),
            'num_dev_dn':    (0.2, 2.0, 0.2),

            # Hook detection
            'hook_lookback':   (1, 8, 1),
            'slope_lookback':  (2, 10, 1),
            'slope_threshold': (0.0, 5.0, 0.5),

            # Filters — bool stays as list
            'enable_middle_band_hook':  [True, False],
            'require_color_change':     [True, False],
            'enable_divergence_filter': [True, False],
            'divergence_lookback':      (5, 40, 5),
            'enable_bw_filter': [True, False],
            'bw_period':        (10, 100, 10),
            'bw_multiplier':    (0.5, 2.0, 0.1),
            'enable_time_filter': [True, False],

            # Trade management
            'profit_ticks':        (10, 100, 10),
            'stop_ticks':          (5, 60, 5),
            'bars_between_trades': (0, 5, 1),

            # Wattah Atar filter
            'enable_wattah_atar': [True, False],

            # Jurik filter
            'enable_jurik_filter': [True, False],
            'jurik_period':        (5, 100, 5),
            'jurik_phase':         (-50.0, 50.0, 10.0),
            'jurik_fl_period':     (5, 100, 5),

            # Tick bar size — outer loop when bar_type == 'tick'; ignored otherwise
            'tick_bar_size': (100, 2000, 100),

            # Calculate mode
            'calculate_mode': ['on_bar_close', 'on_each_tick', 'on_price_change'],
        }

    # Sub-params hidden when parent filter is disabled
    param_conditional = {
        'bw_period':          ('enable_bw_filter', True),
        'bw_multiplier':      ('enable_bw_filter', True),
        'divergence_lookback': ('enable_divergence_filter', True),
        'wa_sensitivity':     ('enable_wattah_atar', True),
        'wa_fast_length':     ('enable_wattah_atar', True),
        'wa_slow_length':     ('enable_wattah_atar', True),
        'wa_channel_length':  ('enable_wattah_atar', True),
        'wa_mult':            ('enable_wattah_atar', True),
        'wa_dead_zone':       ('enable_wattah_atar', True),
        'trade_start_time':   ('enable_time_filter', True),
        'trade_end_time':     ('enable_time_filter', True),
        'jurik_period':       ('enable_jurik_filter', True),
        'jurik_phase':        ('enable_jurik_filter', True),
        'jurik_fl_period':    ('enable_jurik_filter', True),
        # calculate_mode has no dependency — always shown
    }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            'DPO / Bands':        ['dpo_period', 'mobo_length', 'num_dev_up', 'num_dev_dn'],
            'Hook Signal':        ['hook_lookback', 'slope_lookback', 'slope_threshold',
                                   'enable_middle_band_hook', 'require_color_change'],
            'Filters':            ['enable_divergence_filter', 'divergence_lookback',
                                   'enable_bw_filter', 'bw_period', 'bw_multiplier',
                                   'enable_time_filter', 'trade_start_time', 'trade_end_time'],
            'Wattah Atar':        ['enable_wattah_atar', 'wa_sensitivity', 'wa_fast_length',
                                   'wa_slow_length', 'wa_channel_length', 'wa_mult', 'wa_dead_zone'],
            'Jurik Filter':       ['enable_jurik_filter', 'jurik_period', 'jurik_phase', 'jurik_fl_period'],
            'Trade Management':   ['profit_ticks', 'stop_ticks', 'bars_between_trades',
                                   'enable_longs', 'enable_shorts'],
            'Bar Settings':       ['tick_bar_size'],
            'Calculation':        ['calculate_mode'],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            'dpo_period':              'DPO Period',
            'mobo_length':             'Mobo Length',
            'num_dev_up':              'Dev Up',
            'num_dev_dn':              'Dev Down',
            'hook_lookback':           'Hook Lookback',
            'slope_lookback':          'Slope Lookback',
            'slope_threshold':         'Slope Threshold',
            'enable_middle_band_hook':   'Allow Center Touch',
            'require_color_change':      'Require Color Change',
            'enable_divergence_filter':  'Divergence Filter On',
            'divergence_lookback':       'Divergence Lookback',
            'enable_bw_filter':          'BW Filter On',
            'bw_period':               'BW Period',
            'bw_multiplier':           'BW Multiplier',
            'profit_ticks':            'Profit Target (ticks)',
            'stop_ticks':              'Stop Loss (ticks)',
            'bars_between_trades':     'Bars Between Trades',
            'enable_longs':            'Enable Longs',
            'enable_shorts':           'Enable Shorts',
            'tick_bar_size':           'Tick Bar Size',
            'enable_time_filter':      'Time Filter On',
            'trade_start_time':        'Trade Start Time',
            'trade_end_time':          'Trade End Time',
            'enable_wattah_atar':      'Wattah Atar On',
            'wa_sensitivity':          'WA Sensitivity',
            'wa_fast_length':          'WA Fast Length',
            'wa_slow_length':          'WA Slow Length',
            'wa_channel_length':       'WA Channel Length',
            'wa_mult':                 'WA Multiplier',
            'wa_dead_zone':            'WA Dead Zone',
            'enable_jurik_filter':     'Jurik Filter On',
            'jurik_period':            'Jurik Period',
            'jurik_phase':             'Jurik Phase',
            'jurik_fl_period':         'Jurik FL Period',
            'calculate_mode':          'Calculate Mode',
        }

    @property
    def description(self) -> str:
        return "DPO Bollinger Band hook signals on NQ=F (ported from MoboBandsPro NT8 indicator)."

    # ------------------------------------------------------------------
    # Core backtest
    # ------------------------------------------------------------------

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        calc_mode = params.get('calculate_mode', 'on_bar_close')
        ind = _compute_indicators(
            data,
            int(params.get('dpo_period',  self.default_params['dpo_period'])),
            int(params.get('mobo_length', self.default_params['mobo_length'])),
            float(params.get('num_dev_up', self.default_params['num_dev_up'])),
            float(params.get('num_dev_dn', self.default_params['num_dev_dn'])),
            enable_wattah_atar=bool(params.get('enable_wattah_atar', False)),
            wa_fast=int(params.get('wa_fast_length',    self.default_params['wa_fast_length'])),
            wa_slow=int(params.get('wa_slow_length',    self.default_params['wa_slow_length'])),
            wa_channel=int(params.get('wa_channel_length', self.default_params['wa_channel_length'])),
            wa_sensitivity=int(params.get('wa_sensitivity',   self.default_params['wa_sensitivity'])),
            wa_mult=float(params.get('wa_mult',              self.default_params['wa_mult'])),
            enable_jurik_filter=bool(params.get('enable_jurik_filter', False)),
            jurik_period=int(params.get('jurik_period', self.default_params['jurik_period'])),
            jurik_phase=float(params.get('jurik_phase', self.default_params['jurik_phase'])),
            jurik_fl_period=int(params.get('jurik_fl_period', self.default_params['jurik_fl_period'])),
        )

        if calc_mode in ('on_each_tick', 'on_price_change'):
            from strategy_platform.data.loader import load_ticks_raw
            bar_size   = int(params.get('tick_bar_size', self.default_params.get('tick_bar_size', 233)))
            start_date = str(data.index[0].date())
            end_date   = str(data.index[-1].date())
            # Symbol as stored in tick_data (e.g. "MNQ").
            # params['_symbol'] is injected by the dashboard when bar_type == 'tick'.
            # Falls back to stripping suffix from strategy.symbol.
            db_symbol  = params.get('_symbol', self.symbol.replace('=F', '').replace('/', ''))
            raw_ticks  = load_ticks_raw(db_symbol, start=start_date, end=end_date, host=self.db_host)
            bar_tick_prices = _build_bar_tick_map(data, raw_ticks, bar_size)
            trades = _run_backtest_loop_tick(
                data, ind, params,
                self.tick_size, self.tick_value, self.commission_rt,
                bar_tick_prices,
            )
        else:
            trades = _run_backtest_loop(
                data, ind, params,
                self.tick_size, self.tick_value, self.commission_rt,
                calculate_mode=calc_mode,
            )

        total_sessions = int(data['close'].resample('D').last().count())
        stats  = _summarise(trades, total_sessions=total_sessions)
        bs     = _bootstrap_trades(trades, total_sessions=total_sessions)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return {**stats, **bs, 'total_trades': stats['trades'], 'trades': trades_df}

    def run_monte_carlo(
        self,
        prepared: pd.DataFrame,
        params: Dict[str, Any],
        n_sims: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Day-shuffle Monte Carlo: permutes complete trading days and re-runs the strategy."""
        df = prepared  # prepare_data not overridden; prepared is the raw DataFrame

        groups = [(date, grp) for date, grp in df.groupby(df.index.date)]
        rng = np.random.default_rng(seed)
        n   = len(groups)

        dpo_period  = int(params.get('dpo_period',  self.default_params['dpo_period']))
        mobo_length = int(params.get('mobo_length', self.default_params['mobo_length']))
        num_dev_up  = float(params.get('num_dev_up', self.default_params['num_dev_up']))
        num_dev_dn  = float(params.get('num_dev_dn', self.default_params['num_dev_dn']))
        profit_tks  = int(params.get('profit_ticks', self.default_params['profit_ticks']))
        stop_tks    = int(params.get('stop_ticks',   self.default_params['stop_ticks']))

        net_pnls: list = []
        sharpes:  list = []

        for _ in range(n_sims):
            order       = rng.permutation(n)
            shuffled_df = pd.concat([groups[i][1] for i in order])
            ind     = _compute_indicators(
                shuffled_df, dpo_period, mobo_length, num_dev_up, num_dev_dn,
                enable_wattah_atar=bool(params.get('enable_wattah_atar', False)),
                wa_fast=int(params.get('wa_fast_length',    self.default_params['wa_fast_length'])),
                wa_slow=int(params.get('wa_slow_length',    self.default_params['wa_slow_length'])),
                wa_channel=int(params.get('wa_channel_length', self.default_params['wa_channel_length'])),
                wa_sensitivity=int(params.get('wa_sensitivity',   self.default_params['wa_sensitivity'])),
                wa_mult=float(params.get('wa_mult',              self.default_params['wa_mult'])),
                enable_jurik_filter=bool(params.get('enable_jurik_filter', False)),
                jurik_period=int(params.get('jurik_period', self.default_params['jurik_period'])),
                jurik_phase=float(params.get('jurik_phase', self.default_params['jurik_phase'])),
                jurik_fl_period=int(params.get('jurik_fl_period', self.default_params['jurik_fl_period'])),
            )
            trades = _run_backtest_loop(
                shuffled_df, ind, params,
                self.tick_size, self.tick_value, self.commission_rt,
            )
            stats = _summarise(trades)
            if stats.get('trades', 0) >= 5:
                net_pnls.append(stats['net_pnl'])
                sharpes.append(stats['sharpe'])

        if not net_pnls:
            return {'mc_stability': 0.0, 'mc_sharpe_p5': float('nan'),
                    'mc_pnl_p5': float('nan'), 'mc_pnl_p50': float('nan')}

        arr = np.array(net_pnls)
        return {
            'mc_stability': float((arr > 0).mean()),
            'mc_sharpe_p5': float(np.percentile(sharpes,  5)),
            'mc_pnl_p5':    float(np.percentile(arr,      5)),
            'mc_pnl_p50':   float(np.percentile(arr,     50)),
        }
