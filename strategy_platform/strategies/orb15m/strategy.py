"""
ORB15M — Opening Range Breakout strategy for futures (15-min OR window).

Merged port of four NinjaTrader strategies:
  - ORB_Supertrend.cs      (superset: adds SuperTrend filter/stop)
  - ORB15MOnClose.cs       (EntryMode, FVG, EMA, HTF, retest, risk controls)
  - ORB15MTrailing.cs      (base: gap filter, RR targets, BE/trailing, sizing)
  - ORB15mTrailinGapfill.cs (gap filter variant)

All parameters are individually toggleable. Every filter defaults OFF so the
base strategy runs identically to the simplest NT variant.

Bar type : time (5-min OHLCV from emini.historical_data via load_5m)
OR window: bars whose close time falls in (orb_start_time, orb_end_time]
           On 5-min: default 09:30-09:45 → bars closing at 09:35, 09:40, 09:45.
           Entries fire from the first 5-min bar AFTER orb_end_time.

Entry modes:
  OnBreakout    — close crosses OR boundary (CrossAbove / CrossBelow on bar close)
  OnCloseOutside— close outside OR; enter next bar at open
  FVGRetest     — breakout + FVG forms within lookback; limit entry at FVG fill %

Trailing: updated at each bar close (bar-level approximation, acceptable for backtesting).
"""

from __future__ import annotations

import math
from datetime import time as dtime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _rt(price: float, tick_size: float) -> float:
    """Round price to nearest tick."""
    return round(price / tick_size) * tick_size


def _parse_time(s: str) -> dtime:
    """Parse 'HH:MM' string to datetime.time."""
    h, m = s.split(":")
    return dtime(int(h), int(m))


def _in_window(t: dtime, start: dtime, end: dtime) -> bool:
    return start <= t < end


# ---------------------------------------------------------------------------
# Opening Range
# ---------------------------------------------------------------------------

def _compute_orb_levels(
    df: pd.DataFrame,
    orb_start: str,
    orb_end: str,
) -> pd.DataFrame:
    """
    Return DataFrame indexed by date with columns: or_high, or_low.

    OR bars = bars whose close time is in (orb_start, orb_end].
    On 5-min data with orb_start='09:30', orb_end='09:45':
      bars closing at 09:35, 09:40, 09:45.
    """
    s = _parse_time(orb_start)
    e = _parse_time(orb_end)
    mask = (df.index.time > s) & (df.index.time <= e)
    orb = df[mask].copy()
    orb["_date"] = orb.index.date
    levels = orb.groupby("_date").agg(or_high=("high", "max"), or_low=("low", "min"))
    return levels


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

def _compute_ema(series: pd.Series, period: int) -> np.ndarray:
    return series.ewm(span=period, adjust=False).mean().values


# ---------------------------------------------------------------------------
# SuperTrend (port of TSSuperTrend NinjaScript indicator)
# ---------------------------------------------------------------------------

def _wma(arr: np.ndarray, period: int) -> np.ndarray:
    """Weighted moving average."""
    weights = np.arange(1, period + 1, dtype=float)
    w_sum = weights.sum()
    out = np.full(len(arr), np.nan)
    for i in range(period - 1, len(arr)):
        out[i] = np.dot(arr[i - period + 1: i + 1], weights) / w_sum
    return out


def _hma(arr: np.ndarray, period: int) -> np.ndarray:
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))."""
    half = max(1, period // 2)
    sqrt_n = max(1, int(math.sqrt(period)))
    wma_half = _wma(arr, half)
    wma_full = _wma(arr, period)
    raw = 2.0 * wma_half - wma_full
    return _wma(raw, sqrt_n)


def _compute_supertrend(
    df: pd.DataFrame,
    period: int,
    multiplier: float,
    ma_type: str,
    smooth: int,
    tick_size: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute SuperTrend from OHLCV DataFrame.

    Returns:
        uptrend  : bool array — True = uptrend (close > ST line)
        st_line  : float array — the active ST line value (NaN before warmup)
    """
    closes = df["close"].values.astype(float)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    n = len(closes)

    # Smoothed average (centre of the ATR band)
    ma = ma_type.upper()
    if ma == "SMA":
        avg = df["close"].rolling(smooth).mean().values
    elif ma in ("EMA", "DEFAULT"):
        avg = df["close"].ewm(span=smooth, adjust=False).mean().values
    elif ma == "WMA":
        avg = _wma(closes, smooth)
    elif ma == "HMA":
        avg = _hma(closes, smooth)
    elif ma == "TEMA":
        e1 = df["close"].ewm(span=smooth, adjust=False).mean()
        e2 = e1.ewm(span=smooth, adjust=False).mean()
        e3 = e2.ewm(span=smooth, adjust=False).mean()
        avg = (3 * e1 - 3 * e2 + e3).values
    elif ma == "TMA":
        half = max(1, (smooth + 1) // 2)
        avg = df["close"].rolling(half).mean().rolling(half).mean().values
    else:  # VWMA / VMA / fallback → EMA
        avg = df["close"].ewm(span=smooth, adjust=False).mean().values

    # ATR (EMA-smoothed True Range, matching NT's ATR())
    prev_c = np.roll(closes, 1)
    prev_c[0] = closes[0]
    tr = np.maximum(highs - lows,
         np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    offset = atr * multiplier

    # Iterative SuperTrend band (ratchet: never moves against the trend)
    uptrend = np.ones(n, dtype=bool)
    up_line = np.zeros(n, dtype=float)   # ST line when uptrend
    dn_line = np.zeros(n, dtype=float)   # ST line when downtrend
    th = np.zeros(n, dtype=float)        # highest high in current uptrend
    tl = np.full(n, np.inf, dtype=float) # lowest low in current downtrend

    for i in range(1, n):
        if np.isnan(avg[i]) or np.isnan(offset[i]):
            uptrend[i] = uptrend[i - 1]
            up_line[i] = up_line[i - 1]
            dn_line[i] = dn_line[i - 1]
            th[i] = th[i - 1]
            tl[i] = tl[i - 1]
            continue

        prev_up = uptrend[i - 1]

        # Trend based on previous bar's ST line (matches NT logic)
        if up_line[i - 1] > 0:
            uptrend[i] = closes[i] >= up_line[i - 1]
        else:
            uptrend[i] = closes[i] > dn_line[i - 1]

        curr_up = uptrend[i]

        if curr_up and not prev_up:
            # Transition: downtrend → uptrend
            th[i] = highs[i]
            tl[i] = tl[i - 1]
            floor = tl[i - 1] if tl[i - 1] < np.inf else (avg[i] - offset[i])
            up_line[i] = max(avg[i] - offset[i], floor)
            dn_line[i] = 0.0
        elif not curr_up and prev_up:
            # Transition: uptrend → downtrend
            tl[i] = lows[i]
            th[i] = th[i - 1]
            up_line[i] = 0.0
            dn_line[i] = min(avg[i] + offset[i], th[i - 1])
        elif curr_up:
            # Continuing uptrend: stop only ratchets up
            th[i] = max(th[i - 1], highs[i])
            tl[i] = tl[i - 1]
            up_line[i] = max(avg[i] - offset[i], up_line[i - 1])
            dn_line[i] = 0.0
        else:
            # Continuing downtrend: stop only ratchets down
            tl[i] = min(tl[i - 1], lows[i])
            th[i] = th[i - 1]
            dn_line[i] = min(avg[i] + offset[i], dn_line[i - 1])
            up_line[i] = 0.0

    st_line = np.where(uptrend, up_line, dn_line)
    return uptrend, st_line


# ---------------------------------------------------------------------------
# Fair Value Gap detection (port of FVGCustom NinjaScript indicator)
# ---------------------------------------------------------------------------

def _compute_fvg(
    df: pd.DataFrame,
    use_three_bar: bool,
    min_fvg_ticks: int,
    tick_size: float,
) -> pd.DataFrame:
    """
    Detect Fair Value Gaps (FVG) per bar.

    Three-bar gap (use_three_bar=True):
      Bullish FVG: Low[i-1] > High[i-3]  — gap attributed to bar i-2
      Bearish FVG: Low[i-3] > High[i-1]  — gap attributed to bar i-2

    Two-bar gap (use_three_bar=False):
      Bullish FVG: Low[i] > High[i-2]    — gap attributed to bar i-1
      Bearish FVG: Low[i-2] > High[i]    — gap attributed to bar i-1

    Returns DataFrame with: fvg_bull, fvg_bear, fvg_top, fvg_bot (gap zone).
    """
    n = len(df)
    highs = df["high"].values
    lows  = df["low"].values

    fvg_bull = np.zeros(n, dtype=bool)
    fvg_bear = np.zeros(n, dtype=bool)
    fvg_top  = np.full(n, np.nan)
    fvg_bot  = np.full(n, np.nan)

    if use_three_bar:
        for i in range(3, n):
            up_gap = int(round((lows[i - 1] - highs[i - 3]) / tick_size))
            dn_gap = int(round((lows[i - 3] - highs[i - 1]) / tick_size))
            mid = i - 2
            if up_gap >= max(min_fvg_ticks, 1):
                fvg_bull[mid] = True
                fvg_top[mid]  = lows[i - 1]   # top of gap = bar[i-1] low
                fvg_bot[mid]  = highs[i - 3]  # bottom of gap = bar[i-3] high
            elif dn_gap >= max(min_fvg_ticks, 1):
                fvg_bear[mid] = True
                fvg_top[mid]  = highs[i - 1]
                fvg_bot[mid]  = lows[i - 3]
    else:
        for i in range(2, n):
            up_gap = int(round((lows[i] - highs[i - 2]) / tick_size))
            dn_gap = int(round((lows[i - 2] - highs[i]) / tick_size))
            mid = i - 1
            if up_gap >= max(min_fvg_ticks, 1):
                fvg_bull[mid] = True
                fvg_top[mid]  = lows[i]
                fvg_bot[mid]  = highs[i - 2]
            elif dn_gap >= max(min_fvg_ticks, 1):
                fvg_bear[mid] = True
                fvg_top[mid]  = highs[i]
                fvg_bot[mid]  = lows[i - 2]

    return pd.DataFrame(
        {"fvg_bull": fvg_bull, "fvg_bear": fvg_bear,
         "fvg_top": fvg_top, "fvg_bot": fvg_bot},
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]


def _max_consec(is_win: np.ndarray) -> tuple[int, int]:
    """Return longest winning and losing streak lengths."""
    max_w = max_l = cur_w = cur_l = 0
    for flag in is_win:
        if flag:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        max_w = max(max_w, cur_w)
        max_l = max(max_l, cur_l)
    return max_w, max_l


def _max_time_to_recover(dates: List, pnls: np.ndarray) -> float:
    """Max calendar days spent below the previous equity high."""
    if len(pnls) == 0:
        return 0.0
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    high_dates = []
    for i, is_peak in enumerate(equity == peak):
        if is_peak:
            high_dates.append(dates[i])
    if len(high_dates) < 2:
        return 0.0
    return float(max((high_dates[i + 1] - high_dates[i]).days for i in range(len(high_dates) - 1)))


def _summarise(trades: List[Dict], total_sessions: int = 0) -> Dict[str, Any]:
    """Compute strategy statistics from a list of trade dicts."""
    _empty: Dict[str, Any] = {
        "trades": 0, "num_wins": 0, "num_losses": 0, "num_even": 0,
        "win_rate": 0.0, "gross_profit": 0.0, "gross_loss": 0.0, "net_pnl": 0.0,
        "avg_trade": 0.0, "avg_win": 0.0, "avg_loss": 0.0, "ratio_win_loss": 0.0,
        "profit_factor": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "sortino": 0.0,
        "ulcer_index": 0.0, "r_squared": 0.0, "pct_months_profit": 0.0,
        "max_consec_winners": 0, "max_consec_losers": 0, "avg_trades_per_day": 0.0,
        "profit_per_month": 0.0, "max_time_to_recover": 0.0, "longest_flat_days": 0,
        "total_commission": 0.0,
        "start_date": "", "end_date": "",
        **{f"{d[:3].lower()}_pnl": 0.0   for d in _WEEKDAYS},
        **{f"{d[:3].lower()}_trades": 0  for d in _WEEKDAYS},
    }
    if not trades:
        return _empty

    pnls   = np.array([t["pnl"] for t in trades], dtype=float)
    n      = len(pnls)
    wins   = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    even   = pnls[pnls == 0]

    cum  = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    max_dd = float(-(peak - cum).max())

    # Daily map
    daily_map: Dict = {}
    for t in trades:
        d = t["session_date"]
        daily_map[d] = daily_map.get(d, 0.0) + t["pnl"]

    sorted_dates = sorted(daily_map)
    d_vals = np.array([daily_map[d] for d in sorted_dates], dtype=float)
    n_zero = max(0, total_sessions - len(d_vals))
    if n_zero > 0:
        d_vals_padded = np.concatenate([d_vals, np.zeros(n_zero)])
    else:
        d_vals_padded = d_vals

    # Monthly Sharpe
    trade_dates = [t["session_date"] for t in trades]
    first_ym = (trade_dates[0].year, trade_dates[0].month)
    last_ym  = (trade_dates[-1].year, trade_dates[-1].month)

    monthly_full: Dict = {}
    cur_ym = first_ym
    while cur_ym <= last_ym:
        monthly_full[cur_ym] = 0.0
        y, mo = cur_ym
        cur_ym = (y + 1, 1) if mo == 12 else (y, mo + 1)

    monthly: Dict = {}
    for t in trades:
        ym = (t["session_date"].year, t["session_date"].month)
        monthly[ym] = monthly.get(ym, 0.0) + t["pnl"]
    for ym in monthly_full:
        monthly_full[ym] = monthly.get(ym, 0.0)

    m_arr = np.array(list(monthly_full.values()), dtype=float)
    if len(m_arr) >= 6:
        m_std  = float(m_arr.std(ddof=1))
        sharpe = float(m_arr.mean() / m_std) if m_std > 0 else 1.0
    else:
        std    = d_vals_padded.std(ddof=1) if len(d_vals_padded) > 1 else 0.0
        sharpe = float((d_vals_padded.mean() / std) * np.sqrt(252)) if std > 0 else 0.0

    # Sortino
    neg = d_vals_padded[d_vals_padded < 0]
    down_std = float(np.sqrt(np.mean(neg ** 2))) if len(neg) > 0 else 0.0
    sortino = float((d_vals_padded.mean() / down_std) * np.sqrt(252)) if down_std > 0 else 0.0

    gross_p = float(wins.sum())   if len(wins)   > 0 else 0.0
    gross_l = float(losses.sum()) if len(losses) > 0 else 0.0
    pf      = gross_p / abs(gross_l) if gross_l != 0 else float("inf")
    max_consec_w, max_consec_l = _max_consec(pnls > 0)
    n_days = len(daily_map)
    avg_trades_per_day = n / n_days if n_days > 0 else 0.0
    profit_per_month = float(np.mean(list(monthly_full.values()))) if monthly_full else 0.0
    n_months_pos = sum(1 for v in monthly_full.values() if v > 0)
    pct_months = float(n_months_pos / len(monthly_full)) if monthly_full else 0.0
    max_recovery = _max_time_to_recover(trade_dates, pnls)
    dd_series = peak - cum
    ulcer_idx = float(np.sqrt(np.mean(dd_series ** 2)))
    total_commission = float(sum(t.get("commission", 0.0) for t in trades))

    # R²
    x    = np.arange(n, dtype=float)
    ss_tot = float(((cum - cum.mean()) ** 2).sum())
    if n >= 2 and ss_tot > 0:
        try:
            coef  = np.polyfit(x, cum, 1)
            y_hat = np.polyval(coef, x)
            ss_res = float(((cum - y_hat) ** 2).sum())
            r_sq   = float(1.0 - ss_res / ss_tot)
        except np.linalg.LinAlgError:
            r_sq = 0.0
    else:
        r_sq = 0.0

    high_dates = []
    for i, is_peak in enumerate(cum == peak):
        if is_peak:
            high_dates.append(trade_dates[i])
    if len(high_dates) >= 2:
        longest_flat = max((high_dates[i + 1] - high_dates[i]).days for i in range(len(high_dates) - 1))
    else:
        longest_flat = 0

    stats: Dict[str, Any] = {
        "trades":            n,
        "num_wins":          int(len(wins)),
        "num_losses":        int(len(losses)),
        "num_even":          int(len(even)),
        "win_rate":          float(len(wins) / n),
        "gross_profit":      gross_p,
        "gross_loss":        gross_l,
        "net_pnl":           float(pnls.sum()),
        "avg_trade":         float(pnls.mean()),
        "avg_win":           float(wins.mean())   if len(wins)   > 0 else 0.0,
        "avg_loss":          float(losses.mean()) if len(losses) > 0 else 0.0,
        "ratio_win_loss":    float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) else 0.0,
        "profit_factor":     pf,
        "max_drawdown":      max_dd,
        "sharpe":            sharpe,
        "sortino":           sortino,
        "ulcer_index":       ulcer_idx,
        "r_squared":         r_sq,
        "pct_months_profit": pct_months,
        "max_consec_winners": max_consec_w,
        "max_consec_losers":  max_consec_l,
        "avg_trades_per_day": round(avg_trades_per_day, 2),
        "profit_per_month":   profit_per_month,
        "max_time_to_recover": max_recovery,
        "longest_flat_days":   longest_flat,
        "total_commission":    total_commission,
        "start_date":        str(trade_dates[0]),
        "end_date":          str(trade_dates[-1]),
    }

    for day in _WEEKDAYS:
        key       = day[:3].lower()
        day_pnls  = [t["pnl"] for t in trades if t["day_of_week"] == day]
        stats[f"{key}_pnl"]    = float(sum(day_pnls))
        stats[f"{key}_trades"] = len(day_pnls)

    return stats


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def _run_backtest(
    df:          pd.DataFrame,
    params:      Dict[str, Any],
    tick_size:   float,
    tick_value:  float,
    commission:  float,
) -> List[Dict]:
    """
    Event-driven per-bar backtest.

    Position model: two-leg (leg1 = scalp at TP1, leg2 = runner at TP2 or trailing stop).
    State machine runs for each 5-min bar in order.
    """
    p = params

    # ---- Parse params ----
    start_t   = _parse_time(p.get("start_time",    "09:50"))
    end_t     = _parse_time(p.get("end_time",      "12:00"))
    orb_s     = _parse_time(p.get("orb_start_time","09:30"))
    orb_e     = _parse_time(p.get("orb_end_time",  "09:45"))

    allowed_days = set()
    for idx_d, day in enumerate(["monday","tuesday","wednesday","thursday","friday"]):
        if p.get(f"trade_{day}", True):
            allowed_days.add(idx_d)

    entry_mode      = p.get("entry_mode", "OnBreakout")   # OnBreakout | OnCloseOutside | FVGRetest
    require_retest  = bool(p.get("require_retest", False))
    retest_type     = p.get("retest_type", "CloseInside") # CloseInside | WickInside | Either
    max_brk_ticks   = int(p.get("max_breakout_distance_ticks", 0))
    min_or_ticks    = int(p.get("min_or_ticks", 0))
    max_or_ticks    = int(p.get("max_or_ticks", 0))

    use_htf         = bool(p.get("use_htf_confirmation", False))
    htf_mins        = int(p.get("htf_timeframe_mins", 15))

    fvg_fill_pct    = float(p.get("fvg_fill_pct", 50.0))
    fvg_lookback    = int(p.get("fvg_lookback_bars", 10))
    fvg_stop_type   = p.get("fvg_stop_type", "OppositeORBoundary")
    min_fvg_ticks   = int(p.get("min_fvg_size_ticks", 1))
    fvg_three_bar   = bool(p.get("fvg_use_three_bar", True))

    use_ema         = bool(p.get("use_ema_filter", False))
    ema_period      = int(p.get("ema_period", 20))

    use_st_filter   = bool(p.get("use_supertrend_filter", False))
    use_st_stop     = bool(p.get("use_supertrend_stop", False))
    st_period       = int(p.get("st_period", 14))
    st_mult         = float(p.get("st_multiplier", 2.618))
    st_ma_type      = p.get("st_ma_type", "HMA")
    st_smooth       = int(p.get("st_smooth", 14))

    use_gap         = bool(p.get("use_gap_filter", False))
    gap_factor_up   = float(p.get("gap_factor_up",   1.0002))
    gap_factor_dn   = float(p.get("gap_factor_down", 0.9998))

    use_pct_stop    = bool(p.get("use_pct_or_stop", False))
    stop_pct_or     = float(p.get("stop_pct_or", 1.0))
    use_rr          = bool(p.get("use_rr_targets", True))
    first_leg_rr    = float(p.get("first_leg_rr",  0.5))
    second_leg_rr   = float(p.get("second_leg_rr", 1.0))
    runner_stop_only= bool(p.get("runner_stop_only", False))

    use_be          = bool(p.get("use_breakeven", True))
    be_ticks        = int(p.get("be_offset_ticks", 0))
    be_trigger_rr   = float(p.get("be_trigger_rr", 0.0))

    use_trail       = bool(p.get("use_trailing", True))
    trail_trig_rr   = float(p.get("trail_trigger_rr", 0.0))
    trail_dist_ticks= int(p.get("trail_distance_ticks", 0))
    trail_dist_rr   = float(p.get("trail_distance_rr", 0.5))

    use_risk_size   = bool(p.get("use_risk_based_size", False))
    risk_per_trade  = float(p.get("risk_per_trade", 250.0))
    first_leg_pct   = float(p.get("first_leg_pct", 0.33))
    min_contracts   = int(p.get("min_contracts", 1))
    max_contracts   = int(p.get("max_contracts", 0))
    skip_size_zero  = bool(p.get("skip_if_size_zero", True))
    fixed_qty1      = int(p.get("qty1", 1))
    fixed_qty2      = int(p.get("qty2", 2))

    max_long_trades = int(p.get("max_long_trades", 2))
    max_short_trades= int(p.get("max_short_trades", 2))
    cooldown_bars   = int(p.get("cooldown_bars", 3))

    use_daily_guards= bool(p.get("use_daily_guards", False))
    daily_profit_lim= float(p.get("daily_profit_limit", 500.0))
    daily_loss_lim  = float(p.get("daily_loss_limit", -500.0))
    use_wl_caps     = bool(p.get("use_win_loss_caps", False))
    max_wins_sess   = int(p.get("max_wins_session", 3))
    max_losses_sess = int(p.get("max_losses_session", 3))

    ts = tick_size
    tv = tick_value

    # ---- Compute indicators ----
    n = len(df)
    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    opens  = df["open"].values.astype(float)
    closes = df["close"].values.astype(float)
    times  = df.index

    ema_vals = _compute_ema(df["close"], ema_period) if use_ema else np.full(n, np.nan)

    st_up = np.ones(n, dtype=bool)
    st_line_vals = np.zeros(n, dtype=float)
    if use_st_filter or use_st_stop:
        st_up, st_line_vals = _compute_supertrend(df, st_period, st_mult, st_ma_type, st_smooth, ts)

    fvg_df = None
    if entry_mode == "FVGRetest":
        fvg_df = _compute_fvg(df, fvg_three_bar, min_fvg_ticks, ts)
        fvg_bull_arr = fvg_df["fvg_bull"].values
        fvg_bear_arr = fvg_df["fvg_bear"].values
        fvg_top_arr  = fvg_df["fvg_top"].values
        fvg_bot_arr  = fvg_df["fvg_bot"].values
    else:
        fvg_bull_arr = fvg_bear_arr = np.zeros(n, dtype=bool)
        fvg_top_arr  = fvg_bot_arr  = np.full(n, np.nan)

    # HTF close (ffill-aligned to 5m index)
    htf_close = np.full(n, np.nan)
    if use_htf:
        rule = f"{htf_mins}min"
        htf = df.resample(rule).agg(
            open=("open","first"), high=("high","max"),
            low=("low","min"),    close=("close","last"),
        ).dropna()
        htf_c = htf["close"].reindex(df.index, method="ffill")
        htf_close = htf_c.values.astype(float)

    # OR levels per day
    orb_levels = _compute_orb_levels(df, p.get("orb_start_time","09:30"), p.get("orb_end_time","09:45"))

    # ---- Loop state ----
    trades: List[Dict] = []

    # Day-level state
    cur_date     = None
    or_high      = np.nan
    or_low       = np.nan
    long_count   = 0
    short_count  = 0
    daily_pnl    = 0.0
    wins_today   = 0
    losses_today = 0
    gap_long_ok  = True
    gap_short_ok = True
    prior_close  = np.nan   # last close before session open (for gap filter)

    # Per-trade state
    in_trade    = False
    direction   = 0       # 1=long, -1=short
    entry_px    = np.nan
    entry_bar   = -1
    stop_dist   = np.nan  # initial stop distance in price
    leg1_open   = False
    leg2_open   = False
    qty1        = fixed_qty1
    qty2        = fixed_qty2
    leg1_stop   = np.nan
    leg2_stop   = np.nan
    leg1_tp     = np.nan
    leg2_tp     = np.nan
    tp1_done    = False
    be_applied  = False
    trail_armed = False
    trail_dist_px = np.nan
    leg1_exit_px_saved     = np.nan   # persists after leg1 closes until trade recorded
    leg1_exit_reason_saved = ""

    # Cooldown
    cooldown_until = -1

    # OnCloseOutside: pending entry next bar
    pending_entry   = False
    pending_dir     = 0
    pending_stop    = np.nan
    pending_tp1     = np.nan
    pending_tp2     = np.nan
    pending_qty1    = 1
    pending_qty2    = 2
    pending_stop_d  = np.nan

    # FVGRetest: armed after breakout
    fvg_armed       = False
    fvg_dir         = 0
    fvg_armed_bar   = -1
    fvg_entry_lim   = np.nan
    fvg_stop_px     = np.nan
    fvg_tp1_px      = np.nan
    fvg_tp2_px      = np.nan
    fvg_stop_d      = np.nan

    # Retest tracking
    long_retested   = False  # price returned inside OR after long breakout
    short_retested  = False
    long_broke      = False  # first long breakout seen today
    short_broke     = False

    def _calc_sizing(stop_distance_price: float) -> Tuple[int, int]:
        """Return (qty1, qty2) based on sizing params."""
        if not use_risk_size:
            return fixed_qty1, fixed_qty2
        dollars_per_tick = tv
        stop_dist_ticks  = max(1.0, stop_distance_price / ts)
        risk_per_contract = stop_dist_ticks * dollars_per_tick
        if risk_per_contract <= 0:
            return (0, 0) if skip_size_zero else (min_contracts, min_contracts)
        total_q = int(risk_per_trade / risk_per_contract)
        total_q = max(total_q, min_contracts)
        if max_contracts > 0:
            total_q = min(total_q, max_contracts)
        q1 = max(1, int(total_q * first_leg_pct))
        q2 = max(1, total_q - q1)
        return q1, q2

    def _reset_trade():
        nonlocal in_trade, direction, entry_px, entry_bar, stop_dist
        nonlocal leg1_open, leg2_open, qty1, qty2
        nonlocal leg1_stop, leg2_stop, leg1_tp, leg2_tp
        nonlocal tp1_done, be_applied, trail_armed, trail_dist_px
        nonlocal leg1_exit_px_saved, leg1_exit_reason_saved
        in_trade    = False
        direction   = 0
        entry_px    = np.nan
        entry_bar   = -1
        stop_dist   = np.nan
        leg1_open   = False
        leg2_open   = False
        qty1        = fixed_qty1
        qty2        = fixed_qty2
        leg1_stop   = np.nan
        leg2_stop   = np.nan
        leg1_tp     = np.nan
        leg2_tp     = np.nan
        tp1_done    = False
        be_applied  = False
        trail_armed = False
        trail_dist_px          = np.nan
        leg1_exit_px_saved     = np.nan
        leg1_exit_reason_saved = ""

    def _record_trade(
        entry_time, exit_time, dirn: int,
        entry_price: float, exit_px_1: float, exit_px_2: float,
        q1: int, q2: int,
        exit_reason_1: str, exit_reason_2: str,
        session_date, day_name: str,
    ):
        """Compute combined PnL for both legs and append to trades list."""
        sign = dirn
        leg1_pnl = sign * (exit_px_1 - entry_price) / ts * tv * q1
        leg2_pnl = sign * (exit_px_2 - entry_price) / ts * tv * q2
        total_pnl = leg1_pnl + leg2_pnl - commission * (q1 + q2)
        trades.append({
            "entry_time":    entry_time,
            "exit_time":     exit_time,
            "direction":     "long" if dirn == 1 else "short",
            "entry_price":   entry_price,
            "exit_price":    exit_px_2,   # last leg exit
            "leg1_exit_px":  exit_px_1,
            "leg2_exit_px":  exit_px_2,
            "leg1_reason":   exit_reason_1,
            "leg2_reason":   exit_reason_2,
            "pnl":           total_pnl,
            "commission":    commission * (q1 + q2),
            "session_date":  session_date,
            "day_of_week":   day_name,
        })

    def _open_trade(
        bar_idx: int, dirn: int, entry_price: float,
        init_stop: float, tp1: float, tp2: float,
        q1: int, q2: int,
    ):
        nonlocal in_trade, direction, entry_px, entry_bar, stop_dist
        nonlocal leg1_open, leg2_open, qty1, qty2
        nonlocal leg1_stop, leg2_stop, leg1_tp, leg2_tp
        nonlocal tp1_done, be_applied, trail_armed, trail_dist_px
        in_trade    = True
        direction   = dirn
        entry_px    = entry_price
        entry_bar   = bar_idx
        stop_dist   = abs(entry_price - init_stop)
        leg1_open   = True
        leg2_open   = True
        qty1        = q1
        qty2        = q2
        leg1_stop   = init_stop
        leg2_stop   = init_stop
        leg1_tp     = tp1
        leg2_tp     = tp2
        tp1_done    = False
        be_applied  = False
        trail_armed = False
        trail_dist_px = np.nan
        if dirn == 1:
            nonlocal long_count
            long_count += 1
        else:
            nonlocal short_count
            short_count += 1

    # ---- Main bar loop ----
    for i in range(1, n):
        bar_time = times[i]
        bar_tod  = bar_time.time()
        bar_date = bar_time.date()
        dow      = bar_time.dayofweek
        hi = highs[i]; lo = lows[i]; op = opens[i]; cl = closes[i]

        # ----------------------------------------------------------------
        # Day reset
        # ----------------------------------------------------------------
        if bar_date != cur_date:
            cur_date     = bar_date
            long_count   = 0
            short_count  = 0
            daily_pnl    = 0.0
            wins_today   = 0
            losses_today = 0
            cooldown_until = -1
            long_broke   = False
            short_broke  = False
            long_retested  = False
            short_retested = False
            pending_entry  = False
            fvg_armed      = False

            # OR levels
            if bar_date in orb_levels.index:
                row = orb_levels.loc[bar_date]
                or_high = float(row["or_high"])
                or_low  = float(row["or_low"])
            else:
                or_high = or_low = np.nan

            # Gap filter: use prior day's last close
            if use_gap and not np.isnan(prior_close) and not np.isnan(op):
                ratio = op / prior_close
                if ratio >= gap_factor_up:
                    gap_long_ok  = True
                    gap_short_ok = False
                elif ratio <= gap_factor_dn:
                    gap_long_ok  = False
                    gap_short_ok = True
                else:
                    gap_long_ok  = True
                    gap_short_ok = True
            else:
                gap_long_ok  = True
                gap_short_ok = True

        # Track prior_close (last bar of each day)
        # Updated after each bar so we have it for the next day's reset
        prior_close = cl

        # ----------------------------------------------------------------
        # OR complete flag
        # ----------------------------------------------------------------
        or_complete = bar_tod > orb_e and not np.isnan(or_high)

        # Tick value of the OR range
        or_range = (or_high - or_low) if or_complete else np.nan

        # ----------------------------------------------------------------
        # Manage open trade: trailing update
        # ----------------------------------------------------------------
        if in_trade and tp1_done and trail_armed and not np.isnan(trail_dist_px):
            if direction == 1:
                new_stop = _rt(cl - trail_dist_px, ts)
                if leg2_open and new_stop > leg2_stop:
                    leg2_stop = new_stop
            else:
                new_stop = _rt(cl + trail_dist_px, ts)
                if leg2_open and new_stop < leg2_stop:
                    leg2_stop = new_stop

        # ----------------------------------------------------------------
        # Manage open trade: deferred BE trigger (be_trigger_rr > 0)
        # ----------------------------------------------------------------
        if in_trade and use_be and be_trigger_rr > 0 and not be_applied and leg2_open:
            ref_px = cl
            threshold = entry_px + direction * be_trigger_rr * stop_dist
            if direction == 1 and ref_px >= threshold:
                new_be = _rt(entry_px + be_ticks * ts, ts)
                new_be = _rt(min(new_be, cl - ts), ts)
                leg2_stop = new_be
                be_applied = True
                if use_trail and trail_trig_rr <= 0:
                    trail_armed = True
                    trail_dist_px = max(ts, cl - leg2_stop)
            elif direction == -1 and ref_px <= threshold:
                new_be = _rt(entry_px - be_ticks * ts, ts)
                new_be = _rt(max(new_be, cl + ts), ts)
                leg2_stop = new_be
                be_applied = True
                if use_trail and trail_trig_rr <= 0:
                    trail_armed = True
                    trail_dist_px = max(ts, leg2_stop - cl)

        # ----------------------------------------------------------------
        # Manage open trade: deferred trail arming (trail_trigger_rr > 0)
        # ----------------------------------------------------------------
        if in_trade and use_trail and trail_trig_rr > 0 and not trail_armed and tp1_done and leg2_open:
            threshold = entry_px + direction * trail_trig_rr * stop_dist
            if direction == 1 and cl >= threshold:
                trail_armed   = True
                trail_dist_px = max(ts, cl - leg2_stop)
            elif direction == -1 and cl <= threshold:
                trail_armed   = True
                trail_dist_px = max(ts, leg2_stop - cl)

        # ----------------------------------------------------------------
        # Manage open trade: SuperTrend stop (one-time close-break exit)
        # ----------------------------------------------------------------
        if in_trade and use_st_stop and leg2_open and not np.isnan(st_line_vals[i]):
            if direction == 1 and cl < st_line_vals[i]:
                leg2_stop = st_line_vals[i]
            elif direction == -1 and cl > st_line_vals[i]:
                leg2_stop = st_line_vals[i]

        # ----------------------------------------------------------------
        # Manage open trade: check exits
        # ----------------------------------------------------------------
        if in_trade:
            leg1_exit_px     = np.nan
            leg1_exit_reason = ""
            leg2_exit_px     = np.nan
            leg2_exit_reason = ""
            trade_closed     = False

            # --- Leg 1 exit ---
            if leg1_open:
                if direction == 1:
                    if hi >= leg1_tp:
                        leg1_exit_px     = leg1_tp
                        leg1_exit_reason = "tp1"
                        leg1_open = False
                    elif lo <= leg1_stop:
                        leg1_exit_px     = leg1_stop
                        leg1_exit_reason = "stop"
                        leg1_open = False
                else:
                    if lo <= leg1_tp:
                        leg1_exit_px     = leg1_tp
                        leg1_exit_reason = "tp1"
                        leg1_open = False
                    elif hi >= leg1_stop:
                        leg1_exit_px     = leg1_stop
                        leg1_exit_reason = "stop"
                        leg1_open = False

                # Persist leg1 exit info for later bars (leg2 may still be open)
                if not leg1_open:
                    leg1_exit_px_saved     = leg1_exit_px
                    leg1_exit_reason_saved = leg1_exit_reason

            # If leg1 stopped out: close both legs together (same stop)
            if leg1_exit_reason_saved == "stop" and leg2_open and not np.isnan(leg1_exit_px_saved):
                leg2_exit_px     = leg1_stop
                leg2_exit_reason = "stop"
                leg2_open = False

            # If leg1 hit TP1: apply BE to leg2 (immediate trigger)
            if leg1_exit_reason_saved == "tp1" and leg2_open:
                tp1_done = True
                if use_be and be_trigger_rr <= 0 and not be_applied:
                    new_be = _rt(entry_px + direction * be_ticks * ts, ts)
                    if direction == 1:
                        new_be = _rt(min(new_be, cl - ts), ts)
                        new_be = max(new_be, leg2_stop)
                    else:
                        new_be = _rt(max(new_be, cl + ts), ts)
                        new_be = min(new_be, leg2_stop)
                    leg2_stop  = new_be
                    be_applied = True
                    trail_dist_px = max(ts, abs(cl - leg2_stop))
                if use_trail and trail_trig_rr <= 0 and not trail_armed:
                    trail_armed   = True
                    if np.isnan(trail_dist_px):
                        trail_dist_px = max(ts, abs(cl - leg2_stop))

            # --- Leg 2 exit ---
            if leg2_open:
                if direction == 1:
                    if not runner_stop_only and hi >= leg2_tp:
                        leg2_exit_px     = leg2_tp
                        leg2_exit_reason = "tp2"
                        leg2_open = False
                    elif lo <= leg2_stop:
                        leg2_exit_px     = leg2_stop
                        leg2_exit_reason = "stop"
                        leg2_open = False
                else:
                    if not runner_stop_only and lo <= leg2_tp:
                        leg2_exit_px     = leg2_tp
                        leg2_exit_reason = "tp2"
                        leg2_open = False
                    elif hi >= leg2_stop:
                        leg2_exit_px     = leg2_stop
                        leg2_exit_reason = "stop"
                        leg2_open = False

            # EOD exit
            if bar_tod >= end_t and (leg1_open or leg2_open):
                if leg1_open:
                    leg1_exit_px_saved     = cl
                    leg1_exit_reason_saved = "eod"
                    leg1_open = False
                if leg2_open:
                    leg2_exit_px     = cl
                    leg2_exit_reason = "eod"
                    leg2_open = False

            # Record trade when both legs closed
            if not leg1_open and not leg2_open and not np.isnan(leg1_exit_px_saved) and not np.isnan(leg2_exit_px):
                _record_trade(
                    entry_time=times[entry_bar],
                    exit_time=bar_time,
                    dirn=direction,
                    entry_price=entry_px,
                    exit_px_1=leg1_exit_px_saved,
                    exit_px_2=leg2_exit_px,
                    q1=qty1, q2=qty2,
                    exit_reason_1=leg1_exit_reason_saved,
                    exit_reason_2=leg2_exit_reason,
                    session_date=bar_date,
                    day_name=bar_time.strftime("%A"),
                )
                pnl_last = trades[-1]["pnl"]
                daily_pnl += pnl_last
                if pnl_last > 0:
                    wins_today += 1
                else:
                    losses_today += 1
                cooldown_until = i + cooldown_bars
                _reset_trade()
                trade_closed = True

            if in_trade:
                continue  # still in trade, skip entry logic

        # ----------------------------------------------------------------
        # Skip entry if conditions block it
        # ----------------------------------------------------------------
        if bar_tod >= end_t:
            continue
        if not or_complete:
            continue
        if dow not in allowed_days:
            continue
        if i <= cooldown_until:
            continue
        if use_daily_guards:
            if daily_pnl >= daily_profit_lim or daily_pnl <= daily_loss_lim:
                continue
        if use_wl_caps:
            if wins_today >= max_wins_sess or losses_today >= max_losses_sess:
                continue

        # OR size filter
        if or_complete and not np.isnan(or_range):
            if min_or_ticks > 0 and or_range < min_or_ticks * ts:
                continue
            if max_or_ticks > 0 and or_range > max_or_ticks * ts:
                continue

        # ----------------------------------------------------------------
        # Execute pending OnCloseOutside entry (from previous bar)
        # ----------------------------------------------------------------
        if pending_entry and not in_trade:
            d   = pending_dir
            if (d == 1 and long_count < max_long_trades) or (d == -1 and short_count < max_short_trades):
                if (d == 1 and gap_long_ok) or (d == -1 and gap_short_ok):
                    ep = op  # enter at open of this bar
                    _open_trade(i, d, ep, pending_stop, pending_tp1, pending_tp2,
                                pending_qty1, pending_qty2)
            pending_entry = False
            if in_trade:
                continue

        # ----------------------------------------------------------------
        # FVGRetest: check if limit order is triggered
        # ----------------------------------------------------------------
        if entry_mode == "FVGRetest" and fvg_armed and not in_trade:
            if i - fvg_armed_bar > fvg_lookback:
                fvg_armed = False  # expired
            elif not np.isnan(fvg_entry_lim):
                triggered = False
                if fvg_dir == 1 and lo <= fvg_entry_lim and hi >= fvg_entry_lim:
                    triggered = True
                elif fvg_dir == -1 and hi >= fvg_entry_lim and lo <= fvg_entry_lim:
                    triggered = True
                if triggered:
                    d = fvg_dir
                    if (d == 1 and long_count < max_long_trades and gap_long_ok) or \
                       (d == -1 and short_count < max_short_trades and gap_short_ok):
                        q1_, q2_ = _calc_sizing(fvg_stop_d)
                        if not (use_risk_size and skip_size_zero and q1_ == 0):
                            _open_trade(i, d, fvg_entry_lim, fvg_stop_px,
                                        fvg_tp1_px, fvg_tp2_px, q1_, q2_)
                    fvg_armed = False
                    if in_trade:
                        continue

        # ----------------------------------------------------------------
        # Entry signal detection
        # ----------------------------------------------------------------
        if in_trade:
            continue
        if not _in_window(bar_tod, start_t, end_t):
            continue

        prev_cl = closes[i - 1]

        # Retest tracking: did price come back inside OR?
        if not np.isnan(or_high):
            if long_broke and not long_retested:
                retested = False
                if retest_type in ("CloseInside", "Either") and or_low <= prev_cl <= or_high:
                    retested = True
                if retest_type in ("WickInside", "Either") and lows[i - 1] <= or_high and highs[i - 1] >= or_low:
                    retested = True
                if retested:
                    long_retested = True
            if short_broke and not short_retested:
                retested = False
                if retest_type in ("CloseInside", "Either") and or_low <= prev_cl <= or_high:
                    retested = True
                if retest_type in ("WickInside", "Either") and lows[i - 1] <= or_high and highs[i - 1] >= or_low:
                    retested = True
                if retested:
                    short_retested = True

        long_sig  = False
        short_sig = False

        if entry_mode in ("OnBreakout", "OnCloseOutside"):
            # CrossAbove/CrossBelow on close
            if prev_cl <= or_high and cl > or_high:
                long_broke = True
                if not require_retest or long_retested:
                    long_sig = True
            if prev_cl >= or_low and cl < or_low:
                short_broke = True
                if not require_retest or short_retested:
                    short_sig = True

        elif entry_mode == "FVGRetest":
            # Detect first breakout and arm FVG scanner
            if prev_cl <= or_high and cl > or_high and not fvg_armed:
                long_broke = True
                if not require_retest or long_retested:
                    fvg_armed     = True
                    fvg_dir       = 1
                    fvg_armed_bar = i
                    fvg_entry_lim = np.nan
            if prev_cl >= or_low and cl < or_low and not fvg_armed:
                short_broke = True
                if not require_retest or short_retested:
                    fvg_armed     = True
                    fvg_dir       = -1
                    fvg_armed_bar = i
                    fvg_entry_lim = np.nan
            # Look for FVG formation
            if fvg_armed and not np.isnan(fvg_entry_lim):
                pass  # handled above
            elif fvg_armed:
                d = fvg_dir
                if d == 1 and fvg_bull_arr[i]:
                    gap_size = fvg_top_arr[i] - fvg_bot_arr[i]
                    fvg_entry_lim = _rt(fvg_top_arr[i] - gap_size * fvg_fill_pct / 100.0, ts)
                    # Stop
                    if fvg_stop_type == "PercentOfOR":
                        fvg_stop_px = _rt(fvg_entry_lim - stop_pct_or * or_range, ts)
                    elif fvg_stop_type == "FVGInvalidation":
                        fvg_stop_px = _rt(fvg_bot_arr[i] - ts, ts)
                    else:
                        fvg_stop_px = _rt(or_low, ts)
                    fvg_stop_d = max(abs(fvg_entry_lim - fvg_stop_px), ts)
                    if use_rr:
                        fvg_tp1_px = _rt(fvg_entry_lim + first_leg_rr  * fvg_stop_d, ts)
                        fvg_tp2_px = _rt(fvg_entry_lim + second_leg_rr * fvg_stop_d, ts)
                    else:
                        fvg_tp1_px = _rt(fvg_entry_lim + or_range * 0.5, ts)
                        fvg_tp2_px = _rt(fvg_entry_lim + or_range, ts)
                elif d == -1 and fvg_bear_arr[i]:
                    gap_size = fvg_top_arr[i] - fvg_bot_arr[i]
                    fvg_entry_lim = _rt(fvg_bot_arr[i] + gap_size * fvg_fill_pct / 100.0, ts)
                    if fvg_stop_type == "PercentOfOR":
                        fvg_stop_px = _rt(fvg_entry_lim + stop_pct_or * or_range, ts)
                    elif fvg_stop_type == "FVGInvalidation":
                        fvg_stop_px = _rt(fvg_top_arr[i] + ts, ts)
                    else:
                        fvg_stop_px = _rt(or_high, ts)
                    fvg_stop_d = max(abs(fvg_entry_lim - fvg_stop_px), ts)
                    if use_rr:
                        fvg_tp1_px = _rt(fvg_entry_lim - first_leg_rr  * fvg_stop_d, ts)
                        fvg_tp2_px = _rt(fvg_entry_lim - second_leg_rr * fvg_stop_d, ts)
                    else:
                        fvg_tp1_px = _rt(fvg_entry_lim - or_range * 0.5, ts)
                        fvg_tp2_px = _rt(fvg_entry_lim - or_range, ts)
            continue  # FVGRetest entries handled next bar

        # ---- Apply filters to OnBreakout / OnCloseOutside ----
        if not long_sig and not short_sig:
            continue

        # Max breakout distance
        if max_brk_ticks > 0:
            if long_sig  and (cl - or_high) > max_brk_ticks * ts:
                long_sig = False
            if short_sig and (or_low - cl)  > max_brk_ticks * ts:
                short_sig = False

        # Trade caps
        if long_count >= max_long_trades:
            long_sig = False
        if short_count >= max_short_trades:
            short_sig = False

        # Gap filter
        if not gap_long_ok:
            long_sig = False
        if not gap_short_ok:
            short_sig = False

        # EMA filter: close must be on correct side of EMA
        if use_ema and not np.isnan(ema_vals[i]):
            if long_sig  and cl < ema_vals[i]:
                long_sig = False
            if short_sig and cl > ema_vals[i]:
                short_sig = False

        # SuperTrend filter: trend must align
        if use_st_filter:
            if long_sig  and not st_up[i]:
                long_sig = False
            if short_sig and st_up[i]:
                short_sig = False

        # HTF confirmation: latest HTF bar close must also be outside OR
        if use_htf and not np.isnan(htf_close[i]):
            if long_sig  and htf_close[i] <= or_high:
                long_sig = False
            if short_sig and htf_close[i] >= or_low:
                short_sig = False

        if not long_sig and not short_sig:
            continue
        if long_sig and short_sig:
            short_sig = False  # long priority on same bar

        # ---- Compute entry price, stop, targets ----
        dirn = 1 if long_sig else -1

        # Stop placement
        if use_pct_stop:
            raw_stop_dist = max(ts, stop_pct_or * or_range)
            init_stop = _rt(cl - dirn * raw_stop_dist, ts)
        else:
            init_stop = _rt(or_low if dirn == 1 else or_high, ts)

        sd = max(abs(cl - init_stop), ts)

        # SuperTrend stop overrides initial stop
        if use_st_stop and not np.isnan(st_line_vals[i]):
            init_stop = _rt(st_line_vals[i], ts)
            sd = max(abs(cl - init_stop), ts)

        # Targets
        if use_rr:
            tp1 = _rt(cl + dirn * first_leg_rr  * sd, ts)
            tp2 = _rt(cl + dirn * second_leg_rr * sd, ts)
        else:
            tp1 = _rt(cl + dirn * or_range * 0.5, ts)
            tp2 = _rt(cl + dirn * or_range, ts)

        # Sizing
        q1_, q2_ = _calc_sizing(sd)
        if use_risk_size and skip_size_zero and q1_ == 0:
            continue

        if entry_mode == "OnBreakout":
            _open_trade(i, dirn, cl, init_stop, tp1, tp2, q1_, q2_)

        elif entry_mode == "OnCloseOutside":
            pending_entry  = True
            pending_dir    = dirn
            pending_stop   = init_stop
            pending_tp1    = tp1
            pending_tp2    = tp2
            pending_qty1   = q1_
            pending_qty2   = q2_
            pending_stop_d = sd

    return trades


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

@register
class ORB15M(BaseStrategy):
    """
    ORB15M — Opening Range Breakout (15-min OR) for futures.

    Merged port of ORB_Supertrend, ORB15MOnClose, ORB15MTrailing,
    and ORB15mTrailinGapfill NinjaTrader strategies.

    All features default OFF (except core ORB logic) so the base
    strategy matches the simplest NT variant. Enable filters via params.
    """

    name      = "orb15m"
    bar_type  = "time"

    tick_size     = 0.25
    tick_value    = 1.25   # MES defaults — pipeline overrides from get_meta(symbol) at runtime
    commission_rt = 0.62

    default_params: Dict[str, Any] = {
        'trade_monday': True,
        'trade_tuesday': True,
        'trade_wednesday': True,
        'trade_thursday': True,
        'trade_friday': True,
        'require_retest': False,
        'use_htf_confirmation': False,
        'fvg_use_three_bar': True,
        'use_ema_filter': False,
        'use_supertrend_filter': False,
        'use_supertrend_stop': False,
        'use_gap_filter': False,
        'use_pct_or_stop': False,
        'use_rr_targets': True,
        'runner_stop_only': False,
        'use_breakeven': False,
        'use_trailing': False,
        'use_risk_based_size': False,
        'skip_if_size_zero': True,
        'use_daily_guards': False,
        'use_win_loss_caps': False,
        'start_time': '09:45',
        'end_time': '12:00',
        'orb_start_time': '09:30',
        'orb_end_time': '09:45',
        'entry_mode': 'OnBreakout',
        'retest_type': 'CloseInside',
        'max_breakout_distance_ticks': 0,
        'htf_timeframe_mins': 15,
        'min_or_ticks': 0,
        'max_or_ticks': 0,
        'fvg_fill_pct': 50,
        'fvg_lookback_bars': 10,
        'fvg_stop_type': 'OppositeORBoundary',
        'min_fvg_size_ticks': 1,
        'ema_period': 20,
        'st_period': 10,
        'st_multiplier': 2.618,
        'st_ma_type': 'HMA',
        'st_smooth': 10,
        'gap_factor_up': 1.0002,
        'gap_factor_down': 0.9998,
        'stop_pct_or': 1,
        'first_leg_rr': 0.5,
        'second_leg_rr': 1,
        'be_offset_ticks': 0,
        'be_trigger_rr': 0,
        'trail_trigger_rr': 0,
        'trail_distance_ticks': 0,
        'trail_distance_rr': 0.5,
        'qty1': 1,
        'qty2': 1,
        'risk_per_trade': 250,
        'first_leg_pct': 0.33,
        'min_contracts': 1,
        'max_contracts': 0,
        'max_long_trades': 2,
        'max_short_trades': 2,
        'cooldown_bars': 0,
        'daily_profit_limit': 500,
        'daily_loss_limit': -500,
        'max_wins_session': 3,
        'max_losses_session': 3,
    }

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        def _time_range(start: str, end: str) -> List[str]:
            """Return list of 'HH:MM' strings in 5-min steps from start to end inclusive."""
            sh, sm = int(start[:2]), int(start[3:])
            eh, em = int(end[:2]),   int(end[3:])
            vals, h, m = [], sh, sm
            while h * 60 + m <= eh * 60 + em:
                vals.append(f"{h:02d}:{m:02d}")
                m += 5
                if m >= 60:
                    m, h = 0, h + 1
            return vals

        return {
            # Session timing — stay as lists (time strings, not numeric)
            "orb_start_time":            _time_range("00:00", "23:55"),
            "orb_end_time":              _time_range("00:00", "23:55"),
            "start_time":                _time_range("00:00", "23:55"),
            "end_time":                  _time_range("00:00", "23:55"),
            # Trading days — bool lists
            "trade_monday":              [True, False],
            "trade_tuesday":             [True, False],
            "trade_wednesday":           [True, False],
            "trade_thursday":            [True, False],
            "trade_friday":              [True, False],
            # Entry mode — categorical
            "entry_mode":                ["OnBreakout", "OnCloseOutside", "FVGRetest"],
            "require_retest":            [False, True],
            "retest_type":               ["CloseInside", "WickInside", "Either"],
            "max_breakout_distance_ticks": (0, 30, 2),
            # HTF confirmation
            "use_htf_confirmation":      [False, True],
            "htf_timeframe_mins":        (15, 240, 15),
            # OR size filter
            "min_or_ticks":              (0, 20, 2),
            "max_or_ticks":              (0, 100, 10),
            # FVG retest
            "fvg_fill_pct":              (10.0, 90.0, 10.0),
            "fvg_lookback_bars":         (2, 30, 2),
            "fvg_stop_type":             ["OppositeORBoundary", "PercentOfOR", "FVGInvalidation"],
            "min_fvg_size_ticks":        (1, 8, 1),
            "fvg_use_three_bar":         [True, False],
            # EMA filter
            "use_ema_filter":            [False, True],
            "ema_period":                (10, 200, 10),
            # SuperTrend
            "use_supertrend_filter":     [False, True],
            "use_supertrend_stop":       [False, True],
            "st_period":                 (5, 30, 1),
            "st_multiplier":             (1.0, 4.0, 0.1),
            "st_ma_type":                ["HMA", "EMA", "SMA"],
            "st_smooth":                 (5, 30, 1),
            # Gap filter
            "use_gap_filter":            [False, True],
            "gap_factor_up":             (1.0001, 1.0010, 0.0001),
            "gap_factor_down":           (0.9990, 0.9999, 0.0001),
            # Targets & stops
            "use_pct_or_stop":           [False, True],
            "stop_pct_or":               (0.1, 3.0, 0.1),
            "use_rr_targets":            [True, False],
            "first_leg_rr":              (0.25, 3.0, 0.25),
            "second_leg_rr":             (0.5, 5.0, 0.5),
            "runner_stop_only":          [False, True],
            # Breakeven
            "use_breakeven":             [True, False],
            "be_offset_ticks":           (0, 10, 1),
            "be_trigger_rr":             (0.0, 2.0, 0.25),
            # Trailing
            "use_trailing":              [True, False],
            "trail_trigger_rr":          (0.0, 2.0, 0.25),
            "trail_distance_ticks":      (0, 20, 2),
            "trail_distance_rr":         (0.1, 2.0, 0.1),
            # Position sizing
            "qty1":                      (1, 5, 1),
            "qty2":                      (1, 5, 1),
            "use_risk_based_size":       [False, True],
            "risk_per_trade":            (50.0, 1000.0, 50.0),
            "first_leg_pct":             (0.1, 0.9, 0.1),
            "min_contracts":             (1, 5, 1),
            "max_contracts":             (0, 10, 1),
            "skip_if_size_zero":         [True, False],
            # Entry limits
            "max_long_trades":           (1, 5, 1),
            "max_short_trades":          (1, 5, 1),
            "cooldown_bars":             (0, 10, 1),
            # Risk controls
            "use_daily_guards":          [False, True],
            "daily_profit_limit":        (100.0, 2000.0, 100.0),
            "daily_loss_limit":          (-2000.0, -100.0, 100.0),
            "use_win_loss_caps":         [False, True],
            "max_wins_session":          (1, 10, 1),
            "max_losses_session":        (1, 5, 1),
        }

    @property
    def param_group_disable_map(self) -> Dict[str, Dict[str, Any]]:
        """Explicit values to force when a group is disabled. Takes priority over saved selections."""
        return {
            "OR Size Filter": {"min_or_ticks": 0, "max_or_ticks": 0},
        }

    @property
    def param_conditional(self) -> Dict[str, tuple]:
        """
        Maps param_key -> (controlling_param, required_value).
        The param is only shown when controlling_param's current selection includes required_value.
        """
        return {
            "fvg_fill_pct":        ("entry_mode", "FVGRetest"),
            "fvg_lookback_bars":   ("entry_mode", "FVGRetest"),
            "fvg_stop_type":       ("entry_mode", "FVGRetest"),
            "min_fvg_size_ticks":  ("entry_mode", "FVGRetest"),
            "fvg_use_three_bar":   ("entry_mode", "FVGRetest"),
        }

    @property
    def param_dependencies(self) -> Dict[str, tuple]:
        """
        Maps param_key -> (controlling_param, required_value) for pipeline deduplication.
        When the controlling param != required_value, the dependent param is irrelevant
        and gets collapsed to its first grid value, eliminating redundant combinations.
        """
        return {
            # FVG retest params (from param_conditional)
            "fvg_fill_pct":          ("entry_mode",           "FVGRetest"),
            "fvg_lookback_bars":     ("entry_mode",           "FVGRetest"),
            "fvg_stop_type":         ("entry_mode",           "FVGRetest"),
            "min_fvg_size_ticks":    ("entry_mode",           "FVGRetest"),
            "fvg_use_three_bar":     ("entry_mode",           "FVGRetest"),
            # Boolean-controlled params
            "htf_timeframe_mins":    ("use_htf_confirmation", True),
            "ema_period":            ("use_ema_filter",       True),
            "st_period":             ("use_supertrend_filter", True),
            "st_multiplier":         ("use_supertrend_filter", True),
            "st_ma_type":            ("use_supertrend_filter", True),
            "st_smooth":             ("use_supertrend_filter", True),
            "gap_factor_up":         ("use_gap_filter",       True),
            "gap_factor_down":       ("use_gap_filter",       True),
            "stop_pct_or":           ("use_pct_or_stop",      True),
            "first_leg_rr":          ("use_rr_targets",       True),
            "second_leg_rr":         ("use_rr_targets",       True),
            "be_offset_ticks":       ("use_breakeven",        True),
            "be_trigger_rr":         ("use_breakeven",        True),
            "trail_trigger_rr":      ("use_trailing",         True),
            "trail_distance_ticks":  ("use_trailing",         True),
            "trail_distance_rr":     ("use_trailing",         True),
            "risk_per_trade":        ("use_risk_based_size",  True),
            "first_leg_pct":         ("use_risk_based_size",  True),
            "min_contracts":         ("use_risk_based_size",  True),
            "max_contracts":         ("use_risk_based_size",  True),
            "skip_if_size_zero":     ("use_risk_based_size",  True),
            "daily_profit_limit":    ("use_daily_guards",     True),
            "daily_loss_limit":      ("use_daily_guards",     True),
            "max_wins_session":      ("use_win_loss_caps",    True),
            "max_losses_session":    ("use_win_loss_caps",    True),
        }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            "Session & Days":   ["orb_start_time", "orb_end_time", "start_time", "end_time",
                                 "trade_monday", "trade_tuesday", "trade_wednesday", "trade_thursday", "trade_friday"],
            "Entry Mode":       ["entry_mode", "require_retest", "retest_type", "max_breakout_distance_ticks",
                                 "fvg_fill_pct", "fvg_lookback_bars", "fvg_stop_type", "min_fvg_size_ticks", "fvg_use_three_bar"],
            "HTF Confirmation": ["use_htf_confirmation", "htf_timeframe_mins"],
            "OR Size Filter":   ["min_or_ticks", "max_or_ticks"],
            "EMA Filter":       ["use_ema_filter", "ema_period"],
            "SuperTrend":       ["use_supertrend_filter", "use_supertrend_stop", "st_period", "st_multiplier", "st_ma_type", "st_smooth"],
            "Gap Filter":       ["use_gap_filter", "gap_factor_up", "gap_factor_down"],
            "Targets & Stops":  ["use_pct_or_stop", "stop_pct_or", "use_rr_targets", "first_leg_rr", "second_leg_rr", "runner_stop_only"],
            "BE & Trailing":    ["use_breakeven", "be_offset_ticks", "be_trigger_rr",
                                 "use_trailing", "trail_trigger_rr", "trail_distance_ticks", "trail_distance_rr"],
            "Position Sizing":  ["qty1", "qty2", "use_risk_based_size", "risk_per_trade", "first_leg_pct", "min_contracts", "max_contracts", "skip_if_size_zero"],
            "Risk & Limits":    ["max_long_trades", "max_short_trades", "cooldown_bars",
                                 "use_daily_guards", "daily_profit_limit", "daily_loss_limit", "use_win_loss_caps", "max_wins_session", "max_losses_session"],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            "orb_start_time":               "ORB Start Time",
            "orb_end_time":                 "ORB End Time",
            "start_time":                   "Session Start",
            "end_time":                     "Session End",
            "trade_monday":                 "Monday",
            "trade_tuesday":                "Tuesday",
            "trade_wednesday":              "Wednesday",
            "trade_thursday":               "Thursday",
            "trade_friday":                 "Friday",
            "entry_mode":                   "Entry Mode",
            "require_retest":               "Require Retest",
            "retest_type":                  "Retest Type",
            "max_breakout_distance_ticks":  "Max Breakout Distance (ticks)",
            "use_htf_confirmation":         "HTF Confirmation",
            "htf_timeframe_mins":           "HTF Timeframe (mins)",
            "min_or_ticks":                 "Min OR Size (ticks)",
            "max_or_ticks":                 "Max OR Size (ticks)",
            "fvg_fill_pct":                 "FVG Fill %",
            "fvg_lookback_bars":            "FVG Lookback Bars",
            "fvg_stop_type":                "FVG Stop Type",
            "min_fvg_size_ticks":           "Min FVG Size (ticks)",
            "fvg_use_three_bar":            "Three-Bar FVG",
            "use_ema_filter":               "EMA Filter",
            "ema_period":                   "EMA Period",
            "use_supertrend_filter":        "SuperTrend Filter",
            "use_supertrend_stop":          "SuperTrend Stop",
            "st_period":                    "ST Period",
            "st_multiplier":                "ST Multiplier",
            "st_ma_type":                   "ST MA Type",
            "st_smooth":                    "ST Smoothing",
            "use_gap_filter":               "Gap Filter",
            "gap_factor_up":                "Gap Factor Up",
            "gap_factor_down":              "Gap Factor Down",
            "use_pct_or_stop":              "Stop Mode (False=OR boundary, True=% of OR)",
            "stop_pct_or":                  "Stop % of OR",
            "use_rr_targets":               "RR Targets",
            "first_leg_rr":                 "First Leg R:R",
            "second_leg_rr":                "Second Leg R:R",
            "runner_stop_only":             "Runner Stop Only",
            "use_breakeven":                "Breakeven",
            "be_offset_ticks":              "BE Offset (ticks)",
            "be_trigger_rr":                "BE Trigger R:R",
            "use_trailing":                 "Trailing Stop",
            "trail_trigger_rr":             "Trail Trigger R:R",
            "trail_distance_ticks":         "Trail Distance (ticks)",
            "trail_distance_rr":            "Trail Distance R:R",
            "qty1":                         "Qty 1",
            "qty2":                         "Qty 2",
            "use_risk_based_size":          "Risk-Based Sizing",
            "risk_per_trade":               "Risk Per Trade ($)",
            "first_leg_pct":                "First Leg %",
            "min_contracts":                "Min Contracts",
            "max_contracts":                "Max Contracts",
            "skip_if_size_zero":            "Skip If Size Zero",
            "max_long_trades":              "Max Long Trades",
            "max_short_trades":             "Max Short Trades",
            "cooldown_bars":                "Cooldown Bars",
            "use_daily_guards":             "Daily Guards",
            "daily_profit_limit":           "Daily Profit Limit ($)",
            "daily_loss_limit":             "Daily Loss Limit ($)",
            "use_win_loss_caps":            "Win/Loss Caps",
            "max_wins_session":             "Max Wins/Session",
            "max_losses_session":           "Max Losses/Session",
        }

    @property
    def description(self) -> str:
        return "ORB15M — Opening Range Breakout (15-min OR) for futures."

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Single-run entry point."""
        trades = _run_backtest(data, params, self.tick_size, self.tick_value, self.commission_rt)
        total_sessions = int(data["close"].resample("D").last().count())
        stats = _summarise(trades, total_sessions)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return {**stats, "total_trades": stats["trades"], "trades": trades_df}

    def run_monte_carlo(
        self,
        prepared: pd.DataFrame,
        params: Dict[str, Any],
        n_sims: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Day-shuffle Monte Carlo: permute session order n_sims times and re-run backtest."""
        df = prepared
        dates  = df.index.normalize().unique().sort_values()
        groups = [(d, df[df.index.normalize() == d]) for d in dates]
        n      = len(groups)

        rng      = np.random.default_rng(seed)
        net_pnls: List[float] = []
        sharpes:  List[float] = []

        for _ in range(n_sims):
            order    = rng.permutation(n)
            shuffled = pd.concat([groups[i][1] for i in order])
            shuffled.index = df.index[:len(shuffled)]
            result = _run_backtest(shuffled, params, self.tick_size, self.tick_value, self.commission_rt)
            stats  = _summarise(result, total_sessions=n)
            if stats.get("trades", 0) >= 5:
                net_pnls.append(stats["net_pnl"])
                sharpes.append(stats["sharpe"])

        if not net_pnls:
            return {"mc_stability": 0.0, "mc_sharpe_p5": float("nan"),
                    "mc_pnl_p5": float("nan"), "mc_pnl_p50": float("nan")}

        arr = np.array(net_pnls)
        return {
            "mc_stability": float((arr > 0).mean()),
            "mc_sharpe_p5": float(np.percentile(sharpes,  5)),
            "mc_pnl_p5":    float(np.percentile(arr,      5)),
            "mc_pnl_p50":   float(np.percentile(arr,     50)),
        }
