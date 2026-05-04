"""
WickTest5M_FTFCv3 — 5-minute wick-breakout strategy on NQ=F (Mini Nasdaq-100).

Logic (ported from NinjaScript C# source):
  LONG:
    triggerPrice = High[2] + long_entry_offset * tick_size
    bar[0] range touches trigger, opens above High[2]
    Bars [1][2][3] all bullish (close > open)
    Close[1] > High[2], ascending highs: High[1] > High[2] > High[3]
    Upper wick of bar[1] < 25% of bar[1] range
    Entry at triggerPrice; SL = Low[2]; TP = entry + profit * tick_size

  SHORT (mirror):
    triggerPrice = Low[2] - short_entry_offset * tick_size
    bar[0] range touches trigger, opens below Low[2]
    Bars [1][2][3] all bearish, descending lows
    Lower wick of bar[1] < 25% of bar[1] range
    Entry at triggerPrice; SL = High[2]; TP = entry - profit * tick_size

  Filters:
    - min_bar_size_ticks: bar[2] range >= this many ticks
    - bars_between_trades: cooldown bars after a fill
    - FTFC: N sequential TFs (15M/30M/60M/240M/D) must all agree
    - trade_window: time-of-day filter (e.g. "10:00-11:30")
    - Day filter: per-weekday bool params

  EOD exit: all positions closed at 15:59 ET.

Source:  /home/ad/Scripts/strategies/Wicktest5M_v3 (NinjaScript C#)
         /home/ad/Scripts/indicators/SequentialFTFCv3 (NinjaScript C#)
Ported from: /home/ad/WickTest-Optimizer/ (standalone Python optimizer)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register


# ---------------------------------------------------------------------------
# FTFC (SequentialFTFCv3 port)
# ---------------------------------------------------------------------------

_HIGHER_TFS = ['15T', '30T', '60T', '240T', 'D']


def _build_tf_dict(df_5m: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Resample 5M OHLCV into higher TF DataFrames."""
    tf_dict: Dict[str, pd.DataFrame] = {'5T': df_5m}
    rules = {'15T': '15min', '30T': '30min', '60T': '60min', '240T': '240min', 'D': 'D'}
    for key, rule in rules.items():
        tf_dict[key] = df_5m.resample(rule).agg(
            open=('open', 'first'), high=('high', 'max'),
            low=('low', 'min'),    close=('close', 'last'),
        ).dropna()
    return tf_dict


def _compute_ftfc(
    tf_dict: Dict[str, pd.DataFrame],
    tfc_threshold: int,
    skip_tf_count: int,
) -> Tuple[pd.Series, pd.Series]:
    """
    Returns (is_bullish, is_bearish) boolean Series on the 5M index.
    tfc_threshold=0 disables the filter (always True).
    """
    index_5m = tf_dict['5T'].index
    if tfc_threshold == 0:
        ones = pd.Series(True, index=index_5m)
        return ones, ones

    available = [tf for tf in _HIGHER_TFS if tf in tf_dict]
    selected = available[skip_tf_count: skip_tf_count + tfc_threshold]
    if len(selected) < tfc_threshold:
        ones = pd.Series(True, index=index_5m)
        return ones, ones

    directions = []
    for tf in selected:
        raw = np.sign(tf_dict[tf]['close'].values - tf_dict[tf]['open'].values)
        dir_series = pd.Series(raw, index=tf_dict[tf].index)
        # Shift by 1 bar on the higher TF to avoid look-ahead (current bar not closed)
        shifted = dir_series.shift(1)
        aligned = shifted.reindex(index_5m, method='ffill')
        directions.append(aligned)

    stacked = pd.concat(directions, axis=1)
    bullish = (stacked == 1).all(axis=1).fillna(False)
    bearish = (stacked == -1).all(axis=1).fillna(False)
    return bullish, bearish


# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------

def _parse_window(window_str: str) -> Tuple[pd.Timedelta, pd.Timedelta]:
    """Parse "HH:MM-HH:MM" into two Timedeltas."""
    start_str, end_str = window_str.split('-')
    def _td(s):
        h, m = s.strip().split(':')
        return pd.Timedelta(hours=int(h), minutes=int(m))
    return _td(start_str), _td(end_str)


def _generate_signals(
    df: pd.DataFrame,
    params: Dict[str, Any],
    tick_size: float,
    ftfc_bull: pd.Series,
    ftfc_bear: pd.Series,
) -> List[Dict]:
    """
    Bar-by-bar signal scan. Returns a list of entry dicts.
    Each entry dict has: bar_idx, direction, entry_price, sl_price, tp_price.
    """
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    idx = df.index
    n = len(df)
    ts = tick_size

    window_start, window_end = _parse_window(params['trade_window'])

    allowed_days: set = set()
    day_map = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3, 'friday': 4}
    for day, num in day_map.items():
        if params.get(day, True):
            allowed_days.add(num)

    profit       = params['profit']
    stop_ticks   = params.get('stop_ticks', 0)  # 0 = use Low[2]/High[2] only
    min_bar_size = params['min_bar_size_ticks'] * ts
    bars_cd      = params['bars_between_trades']
    leo          = params['long_entry_offset']
    seo          = params['short_entry_offset']

    bull_arr = ftfc_bull.reindex(idx).fillna(False).values
    bear_arr = ftfc_bear.reindex(idx).fillna(False).values

    eod_time = pd.Timedelta(hours=15, minutes=59)

    entries = []
    bars_since_trade = bars_cd  # start ready
    entry_done_this_bar = False

    for i in range(4, n):
        entry_done_this_bar = False
        bars_since_trade += 1

        tod = pd.Timedelta(hours=idx[i].hour, minutes=idx[i].minute)

        # EOD — skip (simulator handles exit)
        if tod >= eod_time:
            continue

        # Time/day filter
        if not (window_start <= tod <= window_end):
            continue
        if idx[i].dayofweek not in allowed_days:
            continue

        # Cooldown
        if bars_since_trade < bars_cd:
            continue

        # Bar[2] size filter
        if h[i-2] - l[i-2] < min_bar_size:
            continue

        # --- LONG ---
        if bull_arr[i] and not entry_done_this_bar:
            trigger = h[i-2] + leo * ts
            if (l[i] <= trigger <= h[i]
                    and o[i] > h[i-2]
                    and c[i-1] > o[i-1] and c[i-2] > o[i-2] and c[i-3] > o[i-3]
                    and c[i-1] > h[i-2]
                    and h[i-1] > h[i-2] and h[i-2] > h[i-3]
                    and (h[i-1] - c[i-1]) < (h[i-1] - l[i-1]) * 0.25):
                sl = max(l[i-2], trigger - stop_ticks * ts) if stop_ticks > 0 else l[i-2]
                entries.append({
                    'bar_idx':    i,
                    'entry_time': idx[i],
                    'direction':  'long',
                    'entry_price': trigger,
                    'sl_price':   sl,
                    'tp_price':   trigger + profit * ts,
                })
                entry_done_this_bar = True
                bars_since_trade = 0

        # --- SHORT ---
        if bear_arr[i] and not entry_done_this_bar:
            trigger = l[i-2] - seo * ts
            if (h[i] >= trigger >= l[i]
                    and o[i] < l[i-2]
                    and c[i-1] < o[i-1] and c[i-2] < o[i-2] and c[i-3] < o[i-3]
                    and c[i-1] < l[i-2]
                    and l[i-1] < l[i-2] and l[i-2] < l[i-3]
                    and (c[i-1] - l[i-1]) < (h[i-1] - l[i-1]) * 0.25):
                sl = min(h[i-2], trigger + stop_ticks * ts) if stop_ticks > 0 else h[i-2]
                entries.append({
                    'bar_idx':    i,
                    'entry_time': idx[i],
                    'direction':  'short',
                    'entry_price': trigger,
                    'sl_price':   sl,
                    'tp_price':   trigger - profit * ts,
                })
                bars_since_trade = 0

    return entries


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------

def _simulate_trades(
    df: pd.DataFrame,
    entries: List[Dict],
    tick_size: float,
    tick_value: float,
    commission: float,
) -> List[Dict]:
    """
    Simulate fills from the pre-generated entry list.
    SL checked before TP (conservative). EOD exit at 15:59.
    """
    if not entries:
        return []

    h_arr  = df['high'].values
    l_arr  = df['low'].values
    c_arr  = df['close'].values
    idx    = df.index
    n      = len(df)
    eod_t  = pd.Timedelta(hours=15, minutes=59)

    trades = []
    entry_iter = iter(entries)
    current_entry = next(entry_iter, None)

    in_trade = False
    ep = sl = tp = entry_time = direction = None

    i = 0
    while i < n:
        ts_td = pd.Timedelta(hours=idx[i].hour, minutes=idx[i].minute)

        if in_trade:
            # EOD force-exit
            if ts_td >= eod_t:
                exit_p = c_arr[i]
                pnl_t  = (exit_p - ep if direction == 'long' else ep - exit_p) / tick_size
                trades.append(_make_trade(entry_time, idx[i], direction, ep, exit_p,
                                          pnl_t, tick_value, commission, 'EOD'))
                in_trade = False
                i += 1
                continue

            # SL first (conservative)
            if direction == 'long':
                if l_arr[i] <= sl:
                    pnl_t = (sl - ep) / tick_size
                    trades.append(_make_trade(entry_time, idx[i], direction, ep, sl,
                                              pnl_t, tick_value, commission, 'SL'))
                    in_trade = False
                elif h_arr[i] >= tp:
                    pnl_t = (tp - ep) / tick_size
                    trades.append(_make_trade(entry_time, idx[i], direction, ep, tp,
                                              pnl_t, tick_value, commission, 'TP'))
                    in_trade = False
            else:  # short
                if h_arr[i] >= sl:
                    pnl_t = (ep - sl) / tick_size
                    trades.append(_make_trade(entry_time, idx[i], direction, ep, sl,
                                              pnl_t, tick_value, commission, 'SL'))
                    in_trade = False
                elif l_arr[i] <= tp:
                    pnl_t = (ep - tp) / tick_size
                    trades.append(_make_trade(entry_time, idx[i], direction, ep, tp,
                                              pnl_t, tick_value, commission, 'TP'))
                    in_trade = False

        # Take next entry if flat and one is pending for this bar
        if not in_trade and current_entry is not None and current_entry['bar_idx'] == i:
            in_trade   = True
            direction  = current_entry['direction']
            ep         = current_entry['entry_price']
            sl         = current_entry['sl_price']
            tp         = current_entry['tp_price']
            entry_time = current_entry['entry_time']
            current_entry = next(entry_iter, None)

        i += 1

    return trades


def _make_trade(entry_time, exit_time, direction, ep, xp, pnl_ticks, tick_value, commission, reason):
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
    }


# ---------------------------------------------------------------------------
# Statistics  (same monthly-Sharpe approach as GoldBot7 for platform consistency)
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
    if n_zero > 0:
        d_vals_padded = np.concatenate([d_vals, np.zeros(n_zero)])
    else:
        d_vals_padded = d_vals

    # Monthly Sharpe (NinjaTrader-style, consistent with GoldBot7)
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

    profit_per_month  = float(np.mean(list(monthly_full.values()))) if monthly_full else 0.0
    n_months_pos      = sum(1 for v in monthly_full.values() if v > 0)
    pct_months_profit = float(n_months_pos / len(monthly_full)) if monthly_full else 0.0

    max_recovery = _max_time_to_recover(trade_dates_all, pnls)

    gross_profit = float(wins.sum())   if len(wins)   > 0 else 0.0
    gross_loss   = float(losses.sum()) if len(losses) > 0 else 0.0

    dd_series = peak - cum
    ulcer_idx = float(np.sqrt(np.mean(dd_series ** 2)))

    x      = np.arange(n, dtype=float)
    coef   = np.polyfit(x, cum, 1)
    y_hat  = np.polyval(coef, x)
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
        (high_dates[i+1] - high_dates[i]).days for i in range(len(high_dates) - 1)
    ) if len(high_dates) >= 2 else 0

    # Commission is already deducted inside each trade's pnl value
    total_commission = 0.0

    stats: Dict = {
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
        'largest_win':         float(wins.max())   if len(wins)   > 0 else 0.0,
        'largest_loss':        float(losses.min()) if len(losses) > 0 else 0.0,
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
        'total_commission':    total_commission,
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
    """Bootstrap daily P&L with replacement."""
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
    win_rates= (s_pnls > 0).mean(axis=1)
    stds     = s_pnls.std(axis=1, ddof=1)
    means    = s_pnls.mean(axis=1)
    sharpes  = np.where(stds > 0, (means / stds) * np.sqrt(252), 0.0)

    return {
        'bs_sharpe_p5': float(np.percentile(sharpes,   5)),
        'bs_pnl_p5':    float(np.percentile(net_pnls,  5)),
        'bs_pnl_p50':   float(np.percentile(net_pnls, 50)),
        'bs_wr_p5':     float(np.percentile(win_rates, 5)),
    }


# ---------------------------------------------------------------------------
# Full backtest runner
# ---------------------------------------------------------------------------

def _run_backtest(
    df: pd.DataFrame,
    params: Dict[str, Any],
    tick_size: float,
    tick_value: float,
    commission: float,
    ftfc_bull: pd.Series,
    ftfc_bear: pd.Series,
) -> List[Dict]:
    entries = _generate_signals(df, params, tick_size, ftfc_bull, ftfc_bear)
    trades  = _simulate_trades(df, entries, tick_size, tick_value, commission)
    return trades


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

@register
class WickTest5M(BaseStrategy):
    """WickTest5M_FTFCv3: 5-minute wick breakout on NQ=F (Mini Nasdaq-100)."""

    name = "wicktest5m"

    default_params: Dict[str, Any] = {
        'monday': True,
        'tuesday': True,
        'wednesday': True,
        'thursday': True,
        'friday': True,
        'profit': 80,
        'stop_ticks': 5,
        'min_bar_size_ticks': 1,
        'bars_between_trades': 1,
        'long_entry_offset': 0,
        'short_entry_offset': 1,
        'tfc_threshold': 0,
        'skip_tf_count': 0,
        'trade_window': '10:00-11:30',
    }

    tick_size     = 0.25
    tick_value    = 5.00
    commission_rt = 3.98

    symbol  = 'NQ=F'
    db_host = None  # reads DB_HOST from .env

    @property
    def param_grid(self) -> Dict[str, Any]:
        return {
            'profit':              (10, 120, 10),
            'stop_ticks':          (0, 100, 5),
            'min_bar_size_ticks':  (1, 10, 1),
            'bars_between_trades': (0, 6, 1),
            'long_entry_offset':   (0, 4, 1),
            'short_entry_offset':  (0, 4, 1),
            'tfc_threshold':       (0, 6, 1),
            'skip_tf_count':       (0, 3, 1),
            'trade_window':       [
                '09:30-10:30', '10:00-11:30', '10:00-12:00',
                '13:00-14:30', '13:30-15:00', '09:30-11:00',
            ],
        }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            "Trading Days":     ["monday", "tuesday", "wednesday", "thursday", "friday"],
            "Entry":            ["profit", "stop_ticks", "long_entry_offset", "short_entry_offset"],
            "Signal Quality":   ["min_bar_size_ticks", "bars_between_trades"],
            "FTFC Filter":      ["tfc_threshold", "skip_tf_count"],
            "Session Timing":   ["trade_window"],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            "monday":               "Monday",
            "tuesday":              "Tuesday",
            "wednesday":            "Wednesday",
            "thursday":             "Thursday",
            "friday":               "Friday",
            "profit":               "Profit Target ($)",
            "stop_ticks":           "Stop (ticks)",
            "min_bar_size_ticks":   "Min Bar Size (ticks)",
            "bars_between_trades":  "Bars Between Trades",
            "long_entry_offset":    "Long Entry Offset",
            "short_entry_offset":   "Short Entry Offset",
            "tfc_threshold":        "TFC Threshold",
            "skip_tf_count":        "Skip TF Count",
            "trade_window":         "Trade Window",
        }

    @property
    def description(self) -> str:
        return "5M wick-breakout with multi-TF FTFC filter on NQ=F (Mini Nasdaq-100)."

    # ------------------------------------------------------------------
    # Optimization hooks
    # ------------------------------------------------------------------

    def prepare_data(self, df: pd.DataFrame) -> Dict:
        """
        Pre-compute higher TF data and FTFC signals for all grid combinations.
        Called once per IS/OOS slice — avoids resampling 20k+ times.
        """
        tf_dict = _build_tf_dict(df)

        # All (tfc_threshold, skip_tf_count) combos in the grid
        ftfc_pairs = {
            (t, s)
            for t in self.param_grid.get('tfc_threshold', [0])
            for s in self.param_grid.get('skip_tf_count', [0])
        }
        ftfc_cache = {
            (t, s): _compute_ftfc(tf_dict, t, s)
            for (t, s) in ftfc_pairs
        }

        return {'df_5m': df, 'ftfc_cache': ftfc_cache}

    def run_backtest_prepared(self, prepared: Dict, params: Dict[str, Any]) -> Dict[str, Any]:
        df         = prepared['df_5m']
        t, s       = params['tfc_threshold'], params['skip_tf_count']
        bull, bear = prepared['ftfc_cache'].get(
            (t, s),
            _compute_ftfc(_build_tf_dict(df), t, s),  # fallback for custom params
        )
        trades = _run_backtest(df, params, self.tick_size, self.tick_value, self.commission_rt,
                               bull, bear)
        stats  = _summarise(trades, total_sessions=df['close'].resample('D').last().count())
        bs     = _bootstrap_trades(trades, total_sessions=df['close'].resample('D').last().count())
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return {**stats, **bs, 'total_trades': stats['trades'], 'trades': trades_df}

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Single-run entry point — builds FTFC on the fly."""
        tf_dict    = _build_tf_dict(data)
        t, s       = params.get('tfc_threshold', 0), params.get('skip_tf_count', 0)
        bull, bear = _compute_ftfc(tf_dict, t, s)
        trades     = _run_backtest(data, params, self.tick_size, self.tick_value,
                                   self.commission_rt, bull, bear)
        stats      = _summarise(trades, total_sessions=data['close'].resample('D').last().count())
        bs         = _bootstrap_trades(trades)
        trades_df  = pd.DataFrame(trades) if trades else pd.DataFrame()
        return {**stats, **bs, 'total_trades': stats['trades'], 'trades': trades_df}

    def run_monte_carlo(
        self,
        prepared: Dict,
        params: Dict[str, Any],
        n_sims: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Day-shuffle Monte Carlo: permutes complete trading days."""
        df         = prepared['df_5m']
        t, s       = params['tfc_threshold'], params['skip_tf_count']
        bull, bear = prepared['ftfc_cache'].get((t, s),
                         _compute_ftfc(_build_tf_dict(df), t, s))

        # Group 5M bars by date for day-level shuffling
        df['_date'] = df.index.date
        groups = [(date, grp.drop(columns='_date'))
                  for date, grp in df.groupby('_date')]
        df = df.drop(columns='_date')

        rng      = np.random.default_rng(seed)
        n        = len(groups)
        net_pnls = []
        sharpes  = []

        for _ in range(n_sims):
            order       = rng.permutation(n)
            shuffled_df = pd.concat([groups[i][1] for i in order])
            # Re-align FTFC signals to the shuffled index
            t2, s2        = _build_tf_dict(shuffled_df), None
            b2, be2       = _compute_ftfc(_build_tf_dict(shuffled_df), t, s)
            trades        = _run_backtest(shuffled_df, params, self.tick_size,
                                          self.tick_value, self.commission_rt, b2, be2)
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
