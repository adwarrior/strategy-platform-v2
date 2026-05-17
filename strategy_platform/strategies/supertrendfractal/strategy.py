"""
SuperTrendFractal — Python port of SuperTrendFractalStrategy.cs (NinjaTrader 8).

Logic:
  1. Computes a Wilder-smoothed ATR and the standard supertrend bands
     (topSeries / bottomSeries / lineSeries) exactly as in the C# source.
  2. Detects fractal peaks (FractalLength = 3, 5, or 7) where the central
     bar's High (short signal) or Low (long signal) pierces the supertrend
     line while the trend has been stable for the full fractal window.
  3. Manages one position at a time with configurable exit modes:
       FixedTPSL  — Ticks / ATRMultiple / RiskReward sub-modes
       TrailToLine — exit when price touches or line flips
  4. Optional session filter with three trade windows + EOD forced exit.
  5. Risk sizing: qty = floor(MaxRisk / (stopDistTicks * dollarPerTick)).

Source: /home/ad/Scripts/strategies/SuperTrendFractalStrategy.cs
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from datetime import time as time_t
from typing import Any, Dict, List

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register

# Full 24h time options at 5-min granularity (288 entries) — used by all
# session-window time params so any HH:MM in 00:00–23:55 can be selected.
_HHMM_24H: List[str] = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 5)]


# ---------------------------------------------------------------------------
# Supertrend indicator
# ---------------------------------------------------------------------------

def _compute_supertrend(
    df: pd.DataFrame,
    atr_period: int,
    atr_multiplier: float,
    fractal_length: int,
) -> pd.DataFrame:
    """
    Replicates C# lines 129-155 exactly:
      - Wilder ATR seed: bar 0 = H-L; subsequent bars use
        ((min(bar+1, period)-1)*atr[prev] + TR) / min(bar+1, period)
      - topSeries / bottomSeries / lineSeries cross-condition logic
    Returns DataFrame with columns: atr, top, bottom, line
    """
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(df)

    atr_arr    = np.zeros(n, dtype=np.float64)
    top_arr    = np.zeros(n, dtype=np.float64)
    bot_arr    = np.zeros(n, dtype=np.float64)
    line_arr   = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if i == 0:
            atr_arr[i] = h[i] - l[i]
        else:
            close1    = c[i - 1]
            tr        = max(abs(l[i] - close1),
                            max(h[i] - l[i], abs(h[i] - close1)))
            denom     = min(i + 1, atr_period)
            atr_arr[i] = ((denom - 1) * atr_arr[i - 1] + tr) / denom

        mid          = (h[i] + l[i]) / 2.0
        top_arr[i]   = mid + atr_multiplier * atr_arr[i]
        bot_arr[i]   = mid - atr_multiplier * atr_arr[i]

        if i < fractal_length:
            # Not enough bars yet — initialise line to topSeries (no previous line)
            line_arr[i] = top_arr[i]
            continue

        # topSeries[i] = topValue if (topValue < line[i-1] OR close[i-1] > line[i-1]) else line[i-1]
        prev_line = line_arr[i - 1]
        prev_top  = top_arr[i - 1]   # we store raw top for comparison vs prev line
        # Note: C# compares topValue (new) vs lineSeries[1] (prev line)
        if top_arr[i] < prev_line or c[i - 1] > prev_line:
            top_s = top_arr[i]
        else:
            top_s = prev_line

        if bot_arr[i] > prev_line or c[i - 1] < prev_line:
            bot_s = bot_arr[i]
        else:
            bot_s = prev_line

        # lineSeries[i] based on previous line state
        if prev_line == top_arr[i - 1]:
            # Was in topSeries → downtrend
            line_arr[i] = top_s if c[i] <= top_s else bot_s
        elif prev_line == bot_arr[i - 1]:
            # Was in bottomSeries → uptrend
            line_arr[i] = bot_s if c[i] >= bot_s else top_s
        else:
            line_arr[i] = top_s

        # Store the smoothed top/bot for this bar so the next bar can compare
        top_arr[i]  = top_s
        bot_arr[i]  = bot_s

    return pd.DataFrame({
        'atr':    atr_arr,
        'top':    top_arr,
        'bottom': bot_arr,
        'line':   line_arr,
    }, index=df.index)


# ---------------------------------------------------------------------------
# Fractal signal detection
# ---------------------------------------------------------------------------

def _detect_signals(
    df: pd.DataFrame,
    ind: pd.DataFrame,
    fractal_length: int,
    invert_signals: bool,
) -> pd.DataFrame:
    """
    Returns DataFrame with bool columns: long_signal, short_signal.
    Mirrors C# lines 192-236.

    FractalLength is clamped to 3/5/7 — the C# setter does this.
    center = (FractalLength - 1) / 2.
    Signal fires on bar `i` (current bar in C#) but references High[center]
    which is `center` bars back.  In vectorised form we look `center` bars
    ahead of the "peak bar" when scanning — equivalent to the C# lag.
    """
    h_arr   = df['high'].values
    l_arr   = df['low'].values
    top_arr = ind['top'].values
    bot_arr = ind['bottom'].values
    line_arr = ind['line'].values
    n       = len(df)
    fl      = fractal_length
    center  = (fl - 1) // 2

    long_sig  = np.zeros(n, dtype=bool)
    short_sig = np.zeros(n, dtype=bool)

    # We need at least fl bars to have stable series; scan from bar fl onward.
    # C# fires on bar `CurrentBar` after looking back FractalLength bars.
    # So the signal fires at index i = peak_bar + center (the detection bar).
    for i in range(fl, n):
        # Window indices for the FractalLength bars ending at i
        # (C# indexing: [0] = current, [1] = 1 bar ago ... so [center] = center bars ago)
        win_start = i - fl + 1
        win_end   = i + 1  # exclusive

        # Stability check: all bars in window must be in same supertrend phase
        is_down_stable = True
        is_up_stable   = True
        for j in range(fl):
            bar_j = win_start + j
            if line_arr[bar_j] != top_arr[bar_j]:
                is_down_stable = False
            if line_arr[bar_j] != bot_arr[bar_j]:
                is_up_stable = False
            if not is_down_stable and not is_up_stable:
                break

        peak_bar = win_start + center  # the center bar of the window

        # Short: downtrend stable, peak High strictly greater than all others,
        #        and peak High > line at center
        if is_down_stable:
            is_peak_high = True
            for j in range(fl):
                if j == center:
                    continue
                bar_j = win_start + j
                if h_arr[peak_bar] <= h_arr[bar_j]:
                    is_peak_high = False
                    break
            if is_peak_high and h_arr[peak_bar] > line_arr[peak_bar]:
                short_sig[i] = True

        # Long: uptrend stable, peak Low strictly less than all others,
        #       and peak Low < line at center
        if is_up_stable:
            is_peak_low = True
            for j in range(fl):
                if j == center:
                    continue
                bar_j = win_start + j
                if l_arr[peak_bar] >= l_arr[bar_j]:
                    is_peak_low = False
                    break
            if is_peak_low and l_arr[peak_bar] < line_arr[peak_bar]:
                long_sig[i] = True

    if invert_signals:
        long_sig, short_sig = short_sig.copy(), long_sig.copy()

    return pd.DataFrame({
        'long_signal':  long_sig,
        'short_signal': short_sig,
    }, index=df.index)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp_fractal_length(val: int) -> int:
    """Mirrors C# setter: <4→3, <6→5, else 7."""
    if val < 4:
        return 3
    elif val < 6:
        return 5
    return 7


def _make_trade(entry_time, exit_time, direction, ep, xp, pnl_ticks,
                tick_value, commission, reason, qty):
    return {
        'session_date': entry_time.date(),
        'day_of_week':  pd.Timestamp(entry_time).day_name(),
        'entry_time':   entry_time,
        'exit_time':    exit_time,
        'direction':    direction,
        'entry_price':  ep,
        'exit_price':   xp,
        'pnl_ticks':    pnl_ticks,
        'pnl':          pnl_ticks * tick_value * qty - commission * qty,
        'exit_reason':  reason,
        'qty':          qty,
    }


def _time_in_window(ts: pd.Timestamp, start_str: str, stop_str: str) -> bool:
    """True if ts falls in [start, stop). If start > stop, treats window as
    spanning midnight (e.g. 18:00 → 06:00 matches 18:00-23:59 and 00:00-05:59).
    start == stop is treated as an empty window (no match)."""
    now = ts.time()
    s   = time_t(*map(int, start_str.split(':')))
    e   = time_t(*map(int, stop_str.split(':')))
    if s == e:
        return False
    if s < e:
        return s <= now < e
    return now >= s or now < e


def _session_end_for_entry(entry_ts: pd.Timestamp, eod_time: time_t,
                           session_break_hour: int = 18) -> pd.Timestamp:
    """Return the timestamp at which the futures session containing entry_ts ends.
    For CME equity futures the session runs from 18:00 ET (prev day) to 16:55 ET
    (current day). A bar at e.g. Mon 19:55 belongs to Tuesday's session, so EOD
    fires at Tue 16:55, not Mon 16:55. Matches NT's 'Break at EOD' behaviour."""
    from datetime import timedelta as _td
    if entry_ts.time() < time_t(session_break_hour, 0):
        end_date = entry_ts.date()
    else:
        end_date = entry_ts.date() + _td(days=1)
    return pd.Timestamp.combine(end_date, eod_time)


# ---------------------------------------------------------------------------
# Core backtest loop
# ---------------------------------------------------------------------------

def _run_backtest_loop(
    df: pd.DataFrame,
    ind: pd.DataFrame,
    sigs: pd.DataFrame,
    params: Dict[str, Any],
    tick_size: float,
    tick_value: float,
    commission: float,
) -> List[Dict]:
    """
    Bar-close simulation mirroring C# OnBarUpdate logic.
    All entries occur at the bar's close price (Calculate.OnBarClose equivalent).
    """
    h_arr    = df['high'].values
    l_arr    = df['low'].values
    c_arr    = df['close'].values
    o_arr    = df['open'].values
    line_arr = ind['line'].values
    atr_arr  = ind['atr'].values
    long_sig = sigs['long_signal'].values
    short_sig= sigs['short_signal'].values
    idx      = df.index
    n        = len(df)

    # -- Params --
    direction_mode    = params.get('direction', 'Both')
    exit_mode         = params.get('exit_mode', 'FixedTPSL')
    tpsl_mode         = params.get('tpsl_mode', 'Ticks')
    tp_ticks          = int(params.get('tp_ticks', 40))
    sl_ticks          = int(params.get('sl_ticks', 20))
    tp_atr_mult       = float(params.get('tp_atr_mult', 2.0))
    sl_atr_mult       = float(params.get('sl_atr_mult', 1.0))
    rr_ratio          = float(params.get('rr_ratio', 2.0))
    use_risk_sizing   = bool(params.get('use_risk_sizing', False))
    qty_fixed         = max(1, int(params.get('qty', 1)))
    max_risk          = float(params.get('max_risk', 250.0))
    bars_cd           = int(params.get('bars_between_trades', 2))
    enable_session    = bool(params.get('enable_session_filter', True))
    eod_exit_str      = params.get('eod_exit_time', '16:55')
    win1_start        = params.get('trade_window1_start', '08:00')
    win1_stop         = params.get('trade_window1_stop',  '10:00')
    enable_win2       = bool(params.get('enable_trade_window2', True))
    win2_start        = params.get('trade_window2_start', '09:30')
    win2_stop         = params.get('trade_window2_stop',  '11:30')
    enable_win3       = bool(params.get('enable_trade_window3', True))
    win3_start        = params.get('trade_window3_start', '14:00')
    win3_stop         = params.get('trade_window3_stop',  '15:55')

    eod_time = time_t(*map(int, eod_exit_str.split(':')))

    dollar_per_tick   = tick_value  # tick_value IS the $ per tick

    trades: List[Dict] = []

    in_trade      = False
    direction     = None
    ep = sl = tp  = 0.0
    entry_time    = None
    qty           = 1
    trail_line    = 0.0   # used for TrailToLine mode
    last_exit_bar = -10000
    session_end_target = None  # set on entry — matches NT's session-aware Break at EOD

    for i in range(n):
        ts  = idx[i]
        now = ts.time()

        # -- EOD forced exit (session-aware: fires only when the entry's
        # session reaches its EOD time, not whenever clock-time exceeds it) --
        if enable_session and in_trade and session_end_target is not None and ts >= session_end_target:
            xp    = c_arr[i]
            sgn   = 1 if direction == 'long' else -1
            p_t   = sgn * (xp - ep) / tick_size
            trades.append(_make_trade(entry_time, ts, direction,
                                      ep, xp, p_t, tick_value, commission, 'EOD', qty))
            in_trade      = False
            last_exit_bar = i
            session_end_target = None
            continue

        # -- TrailToLine exit management (mirrors C# step 2) --
        # 1) Intrabar stop fill: did this bar's low/high touch the trail stop
        #    that was set at the end of the PREVIOUS bar? Mirrors NT's
        #    SetStopLoss(line) behaviour — fill at the line price.
        # 2) Trail-flip: if the trend has reversed (line now on the wrong side
        #    of close), market-exit. Mirrors C# ExitLong/Short call.
        # 3) Otherwise, update sl = line_now for next bar's intrabar check.
        if exit_mode == 'TrailToLine' and in_trade:
            line_now = line_arr[i]
            if direction == 'long':
                # 1) intrabar stop fill at the previous sl
                if l_arr[i] <= sl:
                    xp  = sl
                    p_t = (xp - ep) / tick_size
                    trades.append(_make_trade(entry_time, ts, direction,
                                              ep, xp, p_t, tick_value, commission, 'StopLoss', qty))
                    in_trade = False; last_exit_bar = i; session_end_target = None; continue
                # 2) trail-flip (line crossed close)
                if line_now >= c_arr[i]:
                    xp  = c_arr[i]
                    p_t = (xp - ep) / tick_size
                    trades.append(_make_trade(entry_time, ts, direction,
                                              ep, xp, p_t, tick_value, commission, 'TrailFlip', qty))
                    in_trade = False; last_exit_bar = i; session_end_target = None; continue
                # 3) update trail
                sl = line_now
            else:  # short
                if h_arr[i] >= sl:
                    xp  = sl
                    p_t = (ep - xp) / tick_size
                    trades.append(_make_trade(entry_time, ts, direction,
                                              ep, xp, p_t, tick_value, commission, 'StopLoss', qty))
                    in_trade = False; last_exit_bar = i; session_end_target = None; continue
                if line_now <= c_arr[i]:
                    xp  = c_arr[i]
                    p_t = (ep - xp) / tick_size
                    trades.append(_make_trade(entry_time, ts, direction,
                                              ep, xp, p_t, tick_value, commission, 'TrailFlip', qty))
                    in_trade = False; last_exit_bar = i; session_end_target = None; continue
                sl = line_now

        # -- FixedTPSL bar-level exit check --
        if exit_mode == 'FixedTPSL' and in_trade:
            closed = False
            if direction == 'long':
                if l_arr[i] <= sl:
                    p_t = (sl - ep) / tick_size
                    trades.append(_make_trade(entry_time, ts, direction,
                                              ep, sl, p_t, tick_value, commission, 'SL', qty))
                    closed = True
                elif h_arr[i] >= tp:
                    p_t = (tp - ep) / tick_size
                    trades.append(_make_trade(entry_time, ts, direction,
                                              ep, tp, p_t, tick_value, commission, 'TP', qty))
                    closed = True
            else:
                if h_arr[i] >= sl:
                    p_t = (ep - sl) / tick_size
                    trades.append(_make_trade(entry_time, ts, direction,
                                              ep, sl, p_t, tick_value, commission, 'SL', qty))
                    closed = True
                elif l_arr[i] <= tp:
                    p_t = (ep - tp) / tick_size
                    trades.append(_make_trade(entry_time, ts, direction,
                                              ep, tp, p_t, tick_value, commission, 'TP', qty))
                    closed = True
            if closed:
                in_trade = False; last_exit_bar = i
            continue  # position managed; no new entry on same bar

        # -- Entry guards --
        if in_trade:
            continue

        # Cooldown (C#: CurrentBar - lastExitBar < BarsBetweenTrades)
        if (i - last_exit_bar) < bars_cd:
            continue

        # Session filter
        if enable_session:
            in_w1 = _time_in_window(ts, win1_start, win1_stop)
            in_w2 = enable_win2 and _time_in_window(ts, win2_start, win2_stop)
            in_w3 = enable_win3 and _time_in_window(ts, win3_start, win3_stop)
            if not (in_w1 or in_w2 or in_w3):
                continue

        go_long  = long_sig[i]  and direction_mode != 'ShortOnly'
        go_short = short_sig[i] and direction_mode != 'LongOnly'
        if not go_long and not go_short:
            continue

        # NT fills market orders at the OPEN of the bar AFTER the signal bar.
        # Skip if there's no next bar to fill on.
        if i + 1 >= n:
            continue

        is_long   = go_long  # long takes priority if both (rare)
        atr_now   = atr_arr[i]
        line_now  = line_arr[i]
        entry_px  = o_arr[i + 1]  # next bar's open (matches NT fill convention)

        # Resolve SL/TP ticks
        if tpsl_mode == 'Ticks':
            sl_t = sl_ticks
            tp_t = tp_ticks
        elif tpsl_mode == 'ATRMultiple':
            if tick_size <= 0:
                continue
            sl_t = (atr_now * sl_atr_mult) / tick_size
            tp_t = (atr_now * tp_atr_mult) / tick_size
        else:  # RiskReward
            sl_t = sl_ticks
            tp_t = sl_ticks * rr_ratio

        # Stop distance for sizing
        if exit_mode == 'TrailToLine':
            dist = (entry_px - line_now) if is_long else (line_now - entry_px)
            if dist <= 0:
                continue
            stop_dist_ticks = dist / tick_size
        else:
            stop_dist_ticks = sl_t

        # Sizing
        if use_risk_sizing:
            denom = stop_dist_ticks * dollar_per_tick
            if denom <= 0:
                continue
            qty = int(math.floor(max_risk / denom))
            if qty < 1:
                continue
        else:
            qty = qty_fixed

        # Set TP/SL prices
        direction = 'long' if is_long else 'short'
        ep        = entry_px
        entry_time = idx[i + 1]   # next bar's close-time label (matches NT)

        if exit_mode == 'FixedTPSL':
            if is_long:
                sl = ep - sl_t * tick_size
                tp = ep + tp_t * tick_size
            else:
                sl = ep + sl_t * tick_size
                tp = ep - tp_t * tick_size
        else:  # TrailToLine — initial stop at line
            sl = line_now
            tp = float('inf') if is_long else float('-inf')

        in_trade = True

    return trades


# ---------------------------------------------------------------------------
# Statistics — reuse mobobands pattern
# ---------------------------------------------------------------------------

WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


def _summarise(trades: List[Dict], total_sessions: int = 0) -> dict:
    _empty = {
        'trades': 0, 'win_rate': 0.0, 'net_pnl': 0.0,
        'avg_win': 0.0, 'avg_loss': 0.0, 'profit_factor': 0.0,
        'max_drawdown': 0.0, 'sharpe': 0.0, 'sortino': 0.0,
        'pct_months_profit': 0.0, 'total_commission': 0.0,
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

    cum    = np.cumsum(pnls)
    peak   = np.maximum.accumulate(cum)
    max_dd = float(-(peak - cum).max())

    daily_map: Dict = {}
    for t in trades:
        daily_map[t['session_date']] = daily_map.get(t['session_date'], 0.0) + t['pnl']
    sorted_dates = sorted(daily_map)
    d_vals = np.array([daily_map[d] for d in sorted_dates], dtype=float)
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

    neg = d_vals_padded[d_vals_padded < 0]
    if len(neg) > 0:
        ds_std  = float(np.sqrt(np.mean(neg ** 2)))
        sortino = float(d_vals_padded.mean() / ds_std * np.sqrt(252)) if ds_std > 0 else 0.0
    else:
        sortino = float('inf') if d_vals_padded.mean() > 0 else 0.0

    gross_profit = float(wins.sum())   if len(wins)   > 0 else 0.0
    gross_loss   = float(losses.sum()) if len(losses) > 0 else 0.0
    n_months_pos = sum(1 for v in monthly_full.values() if v > 0)
    pct_months   = float(n_months_pos / len(monthly_full)) if monthly_full else 0.0

    stats = {
        'trades':            n,
        'num_wins':          int(len(wins)),
        'num_losses':        int(len(losses)),
        'win_rate':          float(len(wins) / n),
        'gross_profit':      gross_profit,
        'gross_loss':        gross_loss,
        'net_pnl':           float(pnls.sum()),
        'avg_trade':         float(pnls.mean()),
        'avg_win':           float(wins.mean())   if len(wins)   > 0 else 0.0,
        'avg_loss':          float(losses.mean()) if len(losses) > 0 else 0.0,
        'profit_factor':     gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf'),
        'max_drawdown':      max_dd,
        'sharpe':            sharpe,
        'sortino':           sortino,
        'pct_months_profit': pct_months,
        'total_commission':  float(sum(t.get('qty', 1) * t['pnl'] * 0 for t in trades)),
        'start_date':        str(trade_dates_all[0]),
        'end_date':          str(trade_dates_all[-1]),
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
    pnls  = np.array(list(daily_pnl.values()), dtype=float)
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
        sharpes = np.where(stds > 0, (means / stds) * np.sqrt(252), 0.0)
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
class SuperTrendFractal(BaseStrategy):
    """
    SuperTrendFractal: peak-rejection strategy on supertrend fractals.
    Ported from SuperTrendFractalStrategy.cs (NinjaTrader 8).
    Default instrument: NQ=F (Mini Nasdaq-100, 5M bars).
    """

    name = 'supertrendfractal'

    bar_type            = 'time'
    supported_bar_types = ['time', '1m', 'tick']

    # NQ=F defaults — overridden by dashboard when symbol changes
    tick_size     = 0.25
    tick_value    = 5.00
    commission_rt = 3.98

    symbol: str = 'NQ=F'

    default_params: Dict[str, Any] = {
        'atr_multiplier': 3,
        'atr_period': 10,
        'fractal_length': 3,
        'direction': 'Both',
        'invert_signals': False,
        'exit_mode': 'TrailToLine',
        'tpsl_mode': 'Ticks',
        'tp_ticks': 40,
        'sl_ticks': 20,
        'tp_atr_mult': 2,
        'sl_atr_mult': 1,
        'rr_ratio': 2,
        'use_risk_sizing': False,
        'qty': 1,
        'max_risk': 250,
        'bars_between_trades': 2,
        'enable_session_filter': False,
        'trade_window1_start': '08:00',
        'trade_window1_stop': '10:00',
        'enable_trade_window2': False,
        'trade_window2_start': '09:30',
        'trade_window2_stop': '11:30',
        'enable_trade_window3': False,
        'trade_window3_start': '14:00',
        'trade_window3_stop': '15:55',
        'eod_exit_time': '16:55',
    }

    # ------------------------------------------------------------------
    # param_grid — (min, max, step) for numeric; list for categorical
    # ------------------------------------------------------------------

    @property
    def param_grid(self) -> Dict[str, Any]:
        return {
            # 1. Indicator
            'atr_multiplier':  (1, 5, 1),
            'atr_period':      (5, 20, 5),
            'fractal_length':  [3, 5, 7],
            # 2. Signal
            'direction':       ['Both', 'LongOnly', 'ShortOnly'],
            'invert_signals':  [False, True],
            # 3. Exit
            'exit_mode':       ['FixedTPSL', 'TrailToLine'],
            'tpsl_mode':       ['Ticks', 'ATRMultiple', 'RiskReward'],
            'tp_ticks':        (10, 80, 10),
            'sl_ticks':        (5,  40,  5),
            'tp_atr_mult':     (0.5, 4.0, 0.5),
            'sl_atr_mult':     (0.5, 2.0, 0.5),
            'rr_ratio':        (1.0, 4.0, 0.5),
            # 4. Sizing
            'use_risk_sizing': [False, True],
            'qty':             (1, 4, 1),
            'max_risk':        (100.0, 500.0, 50.0),
            # 5. Cooldown
            'bars_between_trades': (0, 10, 2),
            # 6. Session
            'enable_session_filter':  [True, False],
            'trade_window1_start':    _HHMM_24H,
            'trade_window1_stop':     _HHMM_24H,
            'enable_trade_window2':   [True, False],
            'trade_window2_start':    _HHMM_24H,
            'trade_window2_stop':     _HHMM_24H,
            'enable_trade_window3':   [True, False],
            'trade_window3_start':    _HHMM_24H,
            'trade_window3_stop':     _HHMM_24H,
            'eod_exit_time':          _HHMM_24H,
        }

    # ------------------------------------------------------------------
    # param_groups
    # ------------------------------------------------------------------

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            '1. Indicator': [
                'atr_multiplier', 'atr_period', 'fractal_length',
            ],
            '2. Signal': [
                'direction', 'invert_signals',
            ],
            '3. Exit': [
                'exit_mode', 'tpsl_mode',
                'tp_ticks', 'sl_ticks',
                'tp_atr_mult', 'sl_atr_mult',
                'rr_ratio',
            ],
            '4. Sizing': [
                'use_risk_sizing', 'qty', 'max_risk',
            ],
            '5. Cooldown': [
                'bars_between_trades',
            ],
            '6. Session': [
                'enable_session_filter',
                'trade_window1_start', 'trade_window1_stop',
                'enable_trade_window2',
                'trade_window2_start', 'trade_window2_stop',
                'enable_trade_window3',
                'trade_window3_start', 'trade_window3_stop',
                'eod_exit_time',
            ],
        }

    # ------------------------------------------------------------------
    # display_names
    # ------------------------------------------------------------------

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            'atr_multiplier':         'ATR Multiplier',
            'atr_period':             'ATR Period',
            'fractal_length':         'Fractal Length (3/5/7)',
            'direction':              'Direction',
            'invert_signals':         'Invert Signals',
            'exit_mode':              'Exit Mode',
            'tpsl_mode':              'TP/SL Mode',
            'tp_ticks':               'TP Ticks',
            'sl_ticks':               'SL Ticks',
            'tp_atr_mult':            'TP ATR Multiple',
            'sl_atr_mult':            'SL ATR Multiple',
            'rr_ratio':               'R:R Ratio',
            'use_risk_sizing':        'Use Risk Sizing',
            'qty':                    'Qty (fixed)',
            'max_risk':               'Max Risk $',
            'bars_between_trades':    'Bars Between Trades',
            'enable_session_filter':  'Enable Session Filter',
            'trade_window1_start':    'Window 1 — Start',
            'trade_window1_stop':     'Window 1 — Stop',
            'enable_trade_window2':   'Enable Window 2',
            'trade_window2_start':    'Window 2 — Start',
            'trade_window2_stop':     'Window 2 — Stop',
            'enable_trade_window3':   'Enable Window 3',
            'trade_window3_start':    'Window 3 — Start',
            'trade_window3_stop':     'Window 3 — Stop',
            'eod_exit_time':          'EOD Exit',
        }

    # ------------------------------------------------------------------
    # param_conditional — hides irrelevant params in the dashboard
    # ------------------------------------------------------------------

    param_conditional: Dict[str, tuple] = {
        # FixedTPSL sub-params
        'tpsl_mode':    ('exit_mode',       'FixedTPSL'),
        'tp_ticks':     ('tpsl_mode',       'Ticks'),
        'sl_ticks':     ('tpsl_mode',       'Ticks'),
        'tp_atr_mult':  ('tpsl_mode',       'ATRMultiple'),
        'sl_atr_mult':  ('tpsl_mode',       'ATRMultiple'),
        'rr_ratio':     ('tpsl_mode',       'RiskReward'),
        # Sizing
        'qty':          ('use_risk_sizing',  False),
        'max_risk':     ('use_risk_sizing',  True),
        # Session sub-windows
        'trade_window1_start':  ('enable_session_filter', True),
        'trade_window1_stop':   ('enable_session_filter', True),
        'enable_trade_window2': ('enable_session_filter', True),
        'trade_window2_start':  ('enable_trade_window2',  True),
        'trade_window2_stop':   ('enable_trade_window2',  True),
        'enable_trade_window3': ('enable_session_filter', True),
        'trade_window3_start':  ('enable_trade_window3',  True),
        'trade_window3_stop':   ('enable_trade_window3',  True),
        'eod_exit_time':        ('enable_session_filter', True),
    }

    @property
    def description(self) -> str:
        return "Supertrend fractal peak-rejection strategy (ported from NT8 C#)."

    # ------------------------------------------------------------------
    # Bar-type resampling helper
    # ------------------------------------------------------------------

    def _prepare_df(self, data: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
        """
        For bar_type == '1m': resample 1M → 5M before running the indicator.
        For 'time' and 'tick': use as-is.
        """
        bt = getattr(self, 'bar_type', 'time')
        if bt == '1m':
            df5 = data.resample('5min').agg({
                'open':   'first',
                'high':   'max',
                'low':    'min',
                'close':  'last',
                'volume': 'sum',
            }).dropna(subset=['close'])
            return df5
        return data

    # ------------------------------------------------------------------
    # Core backtest
    # ------------------------------------------------------------------

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        df = self._prepare_df(data, params)

        atr_period   = int(params.get('atr_period',   self.default_params['atr_period']))
        atr_mult     = float(params.get('atr_multiplier', self.default_params['atr_multiplier']))
        frac_len     = _clamp_fractal_length(int(params.get('fractal_length', self.default_params['fractal_length'])))
        invert       = bool(params.get('invert_signals', self.default_params['invert_signals']))

        ind  = _compute_supertrend(df, atr_period, atr_mult, frac_len)
        sigs = _detect_signals(df, ind, frac_len, invert)

        trades = _run_backtest_loop(
            df, ind, sigs, params,
            self.tick_size, self.tick_value, self.commission_rt,
        )

        total_sessions = int(df['close'].resample('D').last().count())
        stats     = _summarise(trades, total_sessions=total_sessions)
        bs        = _bootstrap_trades(trades, total_sessions=total_sessions)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return {**stats, **bs, 'total_trades': stats['trades'], 'trades': trades_df}

    # ------------------------------------------------------------------
    # Monte Carlo (day-shuffle)
    # ------------------------------------------------------------------

    def run_monte_carlo(
        self,
        prepared: pd.DataFrame,
        params: Dict[str, Any],
        n_sims: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Day-shuffle Monte Carlo: permutes complete trading days and re-runs."""
        df = self._prepare_df(prepared, params)

        atr_period = int(params.get('atr_period',   self.default_params['atr_period']))
        atr_mult   = float(params.get('atr_multiplier', self.default_params['atr_multiplier']))
        frac_len   = _clamp_fractal_length(int(params.get('fractal_length', self.default_params['fractal_length'])))
        invert     = bool(params.get('invert_signals', self.default_params['invert_signals']))

        groups = [(date, grp) for date, grp in df.groupby(df.index.date)]
        rng    = np.random.default_rng(seed)
        n      = len(groups)

        net_pnls: list = []
        sharpes:  list = []

        for _ in range(n_sims):
            order       = rng.permutation(n)
            shuffled_df = pd.concat([groups[i][1] for i in order])
            ind  = _compute_supertrend(shuffled_df, atr_period, atr_mult, frac_len)
            sigs = _detect_signals(shuffled_df, ind, frac_len, invert)
            trades = _run_backtest_loop(
                shuffled_df, ind, sigs, params,
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
