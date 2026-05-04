"""
GoldBot6 — Pure breakout on Prior Day Range and Previous (London) Session Range.

Ported from NinjaScript C# at /home/ad/dev/Scripts/strategies/GoldBot6.

Two variants registered:
  goldbot6_5m   — 5-minute OHLCV bars (MySQL historical_data, symbol GC=F)
  goldbot6_tick — N-tick OHLCV bars   (MySQL tick_data, symbol GC)

Logic:
  - Entry stop orders at PDH (long), PDL (short), PSH (long), PSL (short).
  - PDH/PDL = high/low of the prior 24-hr CME session (same as GoldBot7).
  - PSH/PSL = high/low of bars within the configurable London window
    (default 03:00–08:00 ET) on the current session's calendar day.
  - Up to 4 simultaneous independent positions per session (1 per signal).
  - Each position has its own fixed PT/SL (in ticks) and optional trailing stop.
  - Trailing: activates when price moves trailing_trigger_ticks in favour;
    stop then trails trailing_stop_ticks behind the running extreme (bar H/L).
  - Entry orders placed only within [start_trading, stop_trading].
  - PS entries blocked while the London window is still forming (block_ps=True).
  - Force-exit all open positions at EXIT_TIME (16:55 ET).

Parameters exposed (mirrors NinjaTrader optimisation panel):
  qty                    — contracts per entry
  stop_loss_ticks        — fixed initial stop distance
  profit_target_ticks    — fixed profit target distance
  trailing_trigger_ticks — favourable move required to activate trail
  trailing_stop_ticks    — trail distance behind running extreme
  direction              — 'both' | 'long_only' | 'short_only'
  use_pd_levels          — enable Prior Day H/L entries
  use_prev_session       — enable London-window H/L entries
  breakout_offset_ticks  — extra ticks beyond the level for the entry stop
  start_trading          — HH:MM, begin accepting fills
  stop_trading           — HH:MM, stop accepting new fills
  ps_window_start        — HH:MM, London window start
  ps_window_end          — HH:MM, London window end
  block_ps_while_forming — True → PS entries only after ps_window_end

Source: /home/ad/dev/Scripts/strategies/GoldBot6 (NinjaScript C#)
"""

from __future__ import annotations

from datetime import time as time_t, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register

# ── Constants ─────────────────────────────────────────────────────────────────

EXIT_TIME = time_t(16, 55)   # force-exit before 17:00 ET session close
WEEKDAYS  = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _round_tick(price: float, tick_size: float) -> float:
    return round(price / tick_size) * tick_size


def _parse_hhmm(s: str) -> time_t:
    parts = s.split(':')
    return time_t(int(parts[0]), int(parts[1]))


def _in_window(t: time_t, start: time_t, stop: time_t) -> bool:
    """True if t ∈ [start, stop], with midnight-wrap support."""
    if start <= stop:
        return start <= t <= stop
    return t >= start or t <= stop


def _session_date(ts: pd.Timestamp):
    """
    GC session_date = the calendar date the session CLOSES (17:00 ET).
    The CME GC session closes at 17:00 ET and reopens at 18:00 ET.
    Any bar at or after 17:00 ET belongs to the NEXT calendar day's session.
    This matches NT's PriorDayOHLC / SessionIterator for the GC ETH session.
    """
    return (ts + pd.Timedelta(days=1)).date() if ts.hour >= 17 else ts.date()


# ── Session preparation ───────────────────────────────────────────────────────

def _compute_sessions(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Split OHLCV data into GC session dicts.

    Each dict: {session_date, day_of_week, pdh, pdl, bars}.
    PDH/PDL = high/low of the prior TRADING session (skips weekends and
    any phantom non-trading sessions such as Saturday rows created by
    CT-timezone bar data crossing midnight ET).
    PSH/PSL are computed per-simulation run (they depend on ps_window params).
    """
    data = df.copy()
    data['_sd'] = pd.DatetimeIndex(data.index).map(_session_date)

    # Compute H/L for every session_date bucket that appears in the data
    hl = data.groupby('_sd').agg(sh=('high', 'max'), sl=('low', 'min'))

    # Only keep weekday sessions (Mon-Fri) — discard phantom Saturday/Sunday
    # rows that arise when CT-export bars cross 18:00 ET on a Friday night
    weekday_sessions = sorted([
        sd for sd in hl.index
        if pd.Timestamp(str(sd)).weekday() < 5  # 0=Mon … 4=Fri
    ])

    # For each weekday session, look up the PREVIOUS weekday session's H/L
    # (i.e., skip weekends properly rather than blindly shift(1))
    sessions: List[Dict[str, Any]] = []
    for i, sd in enumerate(weekday_sessions):
        if i == 0:
            continue  # no prior session available
        prior_sd = weekday_sessions[i - 1]
        pdh = float(hl.at[prior_sd, 'sh'])
        pdl = float(hl.at[prior_sd, 'sl'])
        if pd.isna(pdh) or pd.isna(pdl):
            continue

        grp = data[data['_sd'] == sd].drop(columns='_sd')
        if grp.empty:
            continue

        sessions.append({
            'session_date': sd,
            'day_of_week':  pd.Timestamp(str(sd)).day_name(),
            'pdh':          pdh,
            'pdl':          pdl,
            'bars':         grp,
        })
    return sessions


# ── Per-session simulation ────────────────────────────────────────────────────

def _simulate_session(
    session:    Dict[str, Any],
    params:     Dict[str, Any],
    tick_size:  float,
    tick_value: float,
    commission: float,
) -> List[Dict[str, Any]]:
    """
    Simulate one session. Returns 0–4 trade dicts.

    Manages four independent entry slots:
      PDH_Long, PDL_Short  (Prior Day breakouts)
      PSH_Long, PSL_Short  (London-window breakouts)

    Each slot has its own fill price, stop, target, and trailing state.
    """
    pdh = session['pdh']
    pdl = session['pdl']
    sd  = pd.Timestamp(session['session_date'])

    # ── Parse params ──────────────────────────────────────────────────────────
    direction  = params.get('direction', 'both')
    can_long   = direction in ('both', 'long_only')
    can_short  = direction in ('both', 'short_only')
    use_pd     = bool(params['use_pd_levels'])
    use_ps     = bool(params['use_prev_session'])
    qty        = int(params['qty'])
    off        = params['breakout_offset_ticks']  * tick_size
    sl_dist    = params['stop_loss_ticks']        * tick_size
    pt_dist    = params['profit_target_ticks']    * tick_size
    trig_dist  = params['trailing_trigger_ticks'] * tick_size
    trail_dist = params['trailing_stop_ticks']    * tick_size
    block_ps   = bool(params['block_ps_while_forming'])

    start_t    = _parse_hhmm(params['start_trading'])
    stop_t     = _parse_hhmm(params['stop_trading'])
    ps_start_t = _parse_hhmm(params['ps_window_start'])
    ps_end_t   = _parse_hhmm(params['ps_window_end'])
    ps_wraps   = ps_start_t > ps_end_t   # e.g. 22:00 → 02:00

    # ── Compute PSH/PSL from bars within the London window ───────────────────
    # MySQL timestamps are bar OPEN times (NT export convention).
    # ps_window_start/end params use the same open-time convention as NT
    # (e.g. ps_window_start='03:00' means include bars opening at/after 03:00).
    sd_date   = sd.date()
    prev_date = sd_date - timedelta(days=1)

    def _in_ps_window(ts: pd.Timestamp) -> bool:
        t = ts.time()   # ts is bar open time — use directly
        d = ts.date()
        if ps_wraps:
            return (d == prev_date and t >= ps_start_t) or \
                   (d == sd_date   and t <  ps_end_t)
        else:
            return d == sd_date and ps_start_t <= t < ps_end_t

    ps_mask = session['bars'].index.map(_in_ps_window)
    ps_bars = session['bars'][ps_mask]
    psh = float(ps_bars['high'].max()) if not ps_bars.empty else float('nan')
    psl = float(ps_bars['low'].min())  if not ps_bars.empty else float('nan')

    # ── Entry prices (level ± breakout offset) ────────────────────────────────
    def _entry(base: float, is_long: bool) -> Optional[float]:
        if np.isnan(base):
            return None
        return _round_tick(base + off if is_long else base - off, tick_size)

    pdh_entry = _entry(pdh, True)  if (use_pd and can_long)  else None
    pdl_entry = _entry(pdl, False) if (use_pd and can_short) else None
    psh_entry = _entry(psh, True)  if (use_ps and can_long)  else None
    psl_entry = _entry(psl, False) if (use_ps and can_short) else None

    # ── Slot initialisation ───────────────────────────────────────────────────
    # state: 'WAITING' → unfilled; 'OPEN' → in position; 'DONE' → closed
    slots: Dict[str, Dict] = {
        'PDH_Long':  {'entry': pdh_entry, 'is_long': True,  'is_ps': False},
        'PDL_Short': {'entry': pdl_entry, 'is_long': False, 'is_ps': False},
        'PSH_Long':  {'entry': psh_entry, 'is_long': True,  'is_ps': True},
        'PSL_Short': {'entry': psl_entry, 'is_long': False, 'is_ps': True},
    }
    for slot in slots.values():
        slot.update({
            'state':        'WAITING' if slot['entry'] is not None else 'DONE',
            'fill_price':   None,
            'stop_price':   None,
            'target_price': None,
            'trail_active': False,
            'first_check':  True,   # flip on first eligible bar
        })

    exit_dt = sd.replace(hour=EXIT_TIME.hour, minute=EXIT_TIME.minute,
                         second=0, microsecond=0)

    trades: List[Dict[str, Any]] = []

    # ── Bar loop ──────────────────────────────────────────────────────────────
    for ts, bar in session['bars'].iterrows():
        bar_t = ts.time()

        # Force-exit all open positions at/after EXIT_TIME
        if ts >= exit_dt:
            for sig, slot in slots.items():
                if slot['state'] == 'OPEN':
                    ep  = float(bar['close'])
                    pnl = _pnl(slot, ep, tick_size, tick_value, qty, commission)
                    trades.append(_make_trade(session, sig, slot, ep, pnl,
                                              'session_exit', commission))
                    slot['state'] = 'DONE'
            break

        # ── Manage already-open positions ─────────────────────────────────────
        for sig, slot in slots.items():
            if slot['state'] != 'OPEN':
                continue

            fp = slot['fill_price']
            sp = slot['stop_price']
            tp = slot['target_price']

            if slot['is_long']:
                # Update trailing stop
                if not slot['trail_active']:
                    if bar['high'] >= fp + trig_dist:
                        slot['trail_active'] = True
                        slot['stop_price'] = _round_tick(
                            bar['high'] - trail_dist, tick_size)
                else:
                    new_sp = _round_tick(bar['high'] - trail_dist, tick_size)
                    if new_sp > slot['stop_price']:
                        slot['stop_price'] = new_sp
                sp = slot['stop_price']

                if bar['low'] <= sp:
                    ep, reason = sp, 'stop'
                elif bar['high'] >= tp:
                    ep, reason = tp, 'target'
                else:
                    continue

            else:  # SHORT
                if not slot['trail_active']:
                    if bar['low'] <= fp - trig_dist:
                        slot['trail_active'] = True
                        slot['stop_price'] = _round_tick(
                            bar['low'] + trail_dist, tick_size)
                else:
                    new_sp = _round_tick(bar['low'] + trail_dist, tick_size)
                    if new_sp < slot['stop_price']:
                        slot['stop_price'] = new_sp
                sp = slot['stop_price']

                if bar['high'] >= sp:
                    ep, reason = sp, 'stop'
                elif bar['low'] <= tp:
                    ep, reason = tp, 'target'
                else:
                    continue

            pnl = _pnl(slot, ep, tick_size, tick_value, qty, commission)
            trades.append(_make_trade(session, sig, slot, ep, pnl, reason, commission))
            slot['state'] = 'DONE'

        # ── Try to fill WAITING entries (only within the trading window) ──────
        if not _in_window(bar_t, start_t, stop_t):
            continue

        # Determine whether the PS range is still forming at this bar
        if block_ps:
            if ps_wraps:
                ps_still_forming = (
                    (ts.date() == prev_date and bar_t >= ps_start_t) or
                    (ts.date() == sd_date   and bar_t <  ps_end_t)
                )
            else:
                ps_still_forming = (ts.date() == sd_date and bar_t < ps_end_t)
        else:
            # Without blocking: PS entries allowed once the window has started
            if ps_wraps:
                ps_still_forming = not (
                    (ts.date() == prev_date and bar_t >= ps_start_t) or
                    (ts.date() == sd_date   and bar_t <  ps_end_t)
                )
            else:
                # Block only before the window starts (levels not yet forming)
                ps_still_forming = not (
                    ts.date() == sd_date and bar_t >= ps_start_t
                )

        for sig, slot in slots.items():
            if slot['state'] != 'WAITING':
                continue
            if slot['is_ps'] and ps_still_forming:
                continue  # blocked until PS window is complete (or started)

            entry_px = slot['entry']

            # First-eligible-bar check: skip if price already past the level
            if slot['first_check']:
                slot['first_check'] = False
                close = float(bar['close'])
                if slot['is_long'] and close > entry_px:
                    slot['state'] = 'DONE'
                    continue
                if not slot['is_long'] and close < entry_px:
                    slot['state'] = 'DONE'
                    continue

            # Fill: price crosses the entry stop level
            if slot['is_long']:
                if float(bar['high']) >= entry_px:
                    slot['state']        = 'OPEN'
                    slot['fill_price']   = entry_px
                    slot['fill_time']    = ts
                    slot['stop_price']   = _round_tick(entry_px - sl_dist, tick_size)
                    slot['target_price'] = _round_tick(entry_px + pt_dist, tick_size)
                    slot['trail_active'] = False
            else:
                if float(bar['low']) <= entry_px:
                    slot['state']        = 'OPEN'
                    slot['fill_price']   = entry_px
                    slot['fill_time']    = ts
                    slot['stop_price']   = _round_tick(entry_px + sl_dist, tick_size)
                    slot['target_price'] = _round_tick(entry_px - pt_dist, tick_size)
                    slot['trail_active'] = False

    return trades


def _pnl(slot: Dict, exit_price: float, tick_size: float,
         tick_value: float, qty: int, commission: float) -> float:
    fp  = slot['fill_price']
    pts = (exit_price - fp) if slot['is_long'] else (fp - exit_price)
    return (pts / tick_size) * tick_value * qty - commission


def _make_trade(session: Dict, sig: str, slot: Dict, exit_price: float,
                pnl: float, exit_reason: str, commission: float) -> Dict[str, Any]:
    return {
        'session_date': session['session_date'],
        'day_of_week':  session['day_of_week'],
        'entry_time':   slot.get('fill_time'),
        'signal':       sig,
        'direction':    'long' if slot['is_long'] else 'short',
        'entry':        slot['fill_price'],
        'exit':         exit_price,
        'stop':         slot['stop_price'],
        'pnl':          pnl,
        'commission':   commission,
        'exit_reason':  exit_reason,
    }


# ── Full backtest runner ──────────────────────────────────────────────────────

def _run_backtest(
    sessions:   List[Dict[str, Any]],
    params:     Dict[str, Any],
    tick_size:  float,
    tick_value: float,
    commission: float,
) -> List[Dict[str, Any]]:
    trades: List[Dict[str, Any]] = []
    for s in sessions:
        trades.extend(_simulate_session(s, params, tick_size, tick_value, commission))
    return trades


# ── Statistics ────────────────────────────────────────────────────────────────

def _daily_sortino(d_vals: np.ndarray) -> float:
    if len(d_vals) < 2:
        return 0.0
    mean = d_vals.mean()
    neg  = d_vals[d_vals < 0]
    if len(neg) == 0:
        return float('inf')
    downside_std = np.sqrt(np.mean(neg ** 2))
    return float(mean / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0


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


def _summarise(trades: List[Dict[str, Any]], total_sessions: int = 0) -> dict:
    _empty: Dict[str, Any] = {
        'trades': 0, 'win_rate': 0.0, 'net_pnl': 0.0,
        'avg_win': 0.0, 'avg_loss': 0.0, 'profit_factor': 0.0,
        'max_drawdown': 0.0, 'sharpe': 0.0, 'sortino': 0.0,
        'ulcer_index': 0.0, 'r_squared': 0.0, 'pct_months_profit': 0.0,
        'longest_flat_days': 0, 'total_commission': 0.0,
        'start_date': '', 'end_date': '',
        **{f'{d[:3].lower()}_pnl':    0.0 for d in WEEKDAYS},
        **{f'{d[:3].lower()}_trades':   0  for d in WEEKDAYS},
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

    n_zero        = max(0, total_sessions - len(d_vals))
    d_vals_padded = np.concatenate([d_vals, np.zeros(n_zero)]) if n_zero > 0 else d_vals

    # NinjaTrader-style Sharpe: mean(monthly PnL) / std(monthly PnL), rf=0
    trade_dates_all = [t['session_date'] for t in trades]
    first_ym = (trade_dates_all[0].year,  trade_dates_all[0].month)
    last_ym  = (trade_dates_all[-1].year, trade_dates_all[-1].month)

    monthly_full: Dict = {}
    cur_ym = first_ym
    while cur_ym <= last_ym:
        monthly_full[cur_ym] = 0.0
        y, mo = cur_ym
        cur_ym = (y + 1, 1) if mo == 12 else (y, mo + 1)

    monthly_sums: Dict = {}
    for t in trades:
        ym = (t['session_date'].year, t['session_date'].month)
        monthly_sums[ym] = monthly_sums.get(ym, 0.0) + t['pnl']
    for ym in monthly_full:
        monthly_full[ym] = monthly_sums.get(ym, 0.0)

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

    high_dates: list = []
    peak_val = -np.inf
    for i, val in enumerate(cum):
        if val > peak_val:
            peak_val = val
            high_dates.append(trade_dates_all[i])
    longest_flat = max(
        (high_dates[i + 1] - high_dates[i]).days for i in range(len(high_dates) - 1)
    ) if len(high_dates) >= 2 else 0

    total_commission = float(sum(t.get('commission', 0.0) for t in trades))

    stats: Dict[str, Any] = {
        'trades':              n,
        'num_wins':            int(len(wins)),
        'num_losses':          int(len(losses)),
        'num_even':            int(len(even)),
        'win_rate':            float(len(wins) / n),
        'gross_profit':        gross_profit,
        'gross_loss':          gross_loss,
        'net_pnl':             float(pnls.sum()),
        'avg_trade':           float(pnls.mean()),
        'avg_win':             float(wins.mean())             if len(wins)   > 0 else 0.0,
        'avg_loss':            float(losses.mean())           if len(losses) > 0 else 0.0,
        'ratio_win_loss':      float(wins.mean() / abs(losses.mean()))
                               if len(wins) and len(losses) else 0.0,
        'largest_win':         float(wins.max())              if len(wins)   > 0 else 0.0,
        'largest_loss':        float(losses.min())            if len(losses) > 0 else 0.0,
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
    trades:         List[Dict[str, Any]],
    n_sims:         int = 1000,
    seed:           int = 42,
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

    idx       = rng.integers(0, n, size=(n_sims, n))
    s_pnls    = pnls[idx]
    net_pnls  = s_pnls.sum(axis=1)
    win_rates = (s_pnls > 0).mean(axis=1)
    stds      = s_pnls.std(axis=1, ddof=1)
    means     = s_pnls.mean(axis=1)
    sharpes   = np.where(stds > 0, (means / stds) * np.sqrt(252), 0.0)

    return {
        'bs_sharpe_p5': float(np.percentile(sharpes,   5)),
        'bs_pnl_p5':    float(np.percentile(net_pnls,  5)),
        'bs_pnl_p50':   float(np.percentile(net_pnls, 50)),
        'bs_wr_p5':     float(np.percentile(win_rates, 5)),
    }


# ── Shared defaults and grid ──────────────────────────────────────────────────

_COMMON_DEFAULTS: Dict[str, Any] = {
    # Risk / Targets
    'qty':                    1,
    'stop_loss_ticks':        100,
    'profit_target_ticks':    100,
    'trailing_trigger_ticks': 18,
    'trailing_stop_ticks':    2,
    # Entry style
    'direction':              'both',
    'use_pd_levels':          True,
    'use_prev_session':       True,
    'breakout_offset_ticks':  0,
    # Session window
    'start_trading':          '20:00',
    'stop_trading':           '15:59',
    # London window
    'ps_window_start':        '03:00',
    'ps_window_end':          '09:00',
    'block_ps_while_forming': True,
}

# Default optimisation grid (7,200 core combos with the ranges shown below).
# Expand any single-value list with --param-grid from the CLI.
_COMMON_PARAM_GRID: Dict[str, List[Any]] = {
    # Core trade geometry — highest impact
    'stop_loss_ticks':        [50, 75, 100, 125, 150],
    'profit_target_ticks':    [75, 100, 125, 150, 200],
    # Trailing stop
    'trailing_trigger_ticks': [5, 10, 20, 30],
    'trailing_stop_ticks':    [3, 5, 10, 15],
    # Entry selection
    'direction':              ['both', 'long_only', 'short_only'],
    'use_pd_levels':          [True, False],
    'use_prev_session':       [True, False],
    'breakout_offset_ticks':  [0, 1, 2],
    # Session timing
    'start_trading':          ['00:00', '18:00', '18:30'],
    'stop_trading':           ['12:00', '14:00', '16:00', '16:45'],
    # London window
    'ps_window_start':        ['02:00', '03:00', '04:00'],
    'ps_window_end':          ['07:00', '08:00', '09:00'],
    'block_ps_while_forming': [True, False],
    # Quantity
    'qty':                    [1, 2],
}


# ── Monte Carlo helper (shared by both classes) ───────────────────────────────

def _run_mc(
    strategy: BaseStrategy,
    sessions: List[Dict[str, Any]],
    params: Dict[str, Any],
    n_sims: int,
    seed: int,
) -> Dict[str, Any]:
    rng      = np.random.default_rng(seed)
    n        = len(sessions)
    net_pnls: List[float] = []
    sharpes:  List[float] = []

    for _ in range(n_sims):
        order    = rng.permutation(n)
        shuffled = [sessions[i] for i in order]
        trades   = _run_backtest(shuffled, params,
                                 strategy.tick_size, strategy.tick_value,
                                 strategy.commission_rt)
        stats = _summarise(trades, total_sessions=n)
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


def _run_prepared(
    strategy: BaseStrategy,
    sessions: List[Dict[str, Any]],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    trades    = _run_backtest(sessions, params,
                              strategy.tick_size, strategy.tick_value,
                              strategy.commission_rt)
    stats     = _summarise(trades, total_sessions=len(sessions))
    bs        = _bootstrap_trades(trades, total_sessions=len(sessions))
    trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
    return {**stats, **bs, 'total_trades': stats['trades'], 'trades': trades_df}


# ── Strategy classes ──────────────────────────────────────────────────────────

@register
class GoldBot6_5M(BaseStrategy):
    """
    GoldBot6 on 5-minute OHLCV bars from MySQL historical_data (symbol: GC=F).

    Run optimisation:
        python -m strategy_platform.optimize.pipeline --strategy goldbot6_5m

    Expand any single-value param-grid entry with --param-grid, e.g.:
        --param-grid '{"start_trading":["18:00","00:00"],"stop_trading":["16:45","12:00"]}'
    """

    name           = 'goldbot6_5m'
    bar_type       = 'time'
    symbol         = 'GC=F'
    db_host: Optional[str] = None   # reads DB_HOST from .env

    tick_size      = 0.10
    tick_value     = 10.00
    commission_rt  = 4.62

    default_params = {
        'use_pd_levels': True,
        'use_prev_session': True,
        'block_ps_while_forming': False,
        'qty': 1,
        'stop_loss_ticks': 25,
        'profit_target_ticks': 100,
        'trailing_trigger_ticks': 10,
        'trailing_stop_ticks': 1,
        'direction': 'both',
        'breakout_offset_ticks': 1,
        'start_trading': '20:00',
        'stop_trading': '15:59',
        'ps_window_start': '03:00',
        'ps_window_end': '08:00',
    }

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            'stop_loss_ticks':        (10, 200, 10),
            'profit_target_ticks':    (20, 300, 10),
            'trailing_trigger_ticks': (2, 50, 2),
            'trailing_stop_ticks':    (1, 20, 1),
            'direction':              ['both', 'long_only', 'short_only'],
            'use_pd_levels':          [True, False],
            'use_prev_session':       [True, False],
            'breakout_offset_ticks':  (0, 5, 1),
            'start_trading':          ['00:00', '18:00', '18:30', '20:00'],
            'stop_trading':           ['12:00', '14:00', '15:59', '16:00', '16:45'],
            'ps_window_start':        ['02:00', '03:00', '04:00'],
            'ps_window_end':          ['07:00', '08:00', '09:00'],
            'block_ps_while_forming': [True, False],
            'qty':                    (1, 4, 1),
        }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            "Levels":           ["use_pd_levels", "use_prev_session", "block_ps_while_forming"],
            "Targets & Stops":  ["stop_loss_ticks", "profit_target_ticks", "trailing_trigger_ticks", "trailing_stop_ticks"],
            "Entry":            ["breakout_offset_ticks", "qty"],
            "Direction":        ["direction"],
            "Session Timing":   ["start_trading", "stop_trading"],
            "London Window":    ["ps_window_start", "ps_window_end"],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            "use_pd_levels":          "PD Levels",
            "use_prev_session":       "Prev Session",
            "block_ps_while_forming": "Block PS While Forming",
            "qty":                    "Qty",
            "stop_loss_ticks":        "Stop Loss (ticks)",
            "profit_target_ticks":    "Profit Target (ticks)",
            "trailing_trigger_ticks": "Trail Trigger (ticks)",
            "trailing_stop_ticks":    "Trail Stop (ticks)",
            "direction":              "Direction",
            "breakout_offset_ticks":  "Breakout Offset (ticks)",
            "start_trading":          "Session Start",
            "stop_trading":           "Session End",
            "ps_window_start":        "London Window Start",
            "ps_window_end":          "London Window End",
        }

    @property
    def description(self) -> str:
        return (
            "GoldBot6: PDH/PDL + London-window breakout on GC=F (5M bars). "
            "Up to 4 simultaneous independent positions per session with per-entry PT/SL/trail."
        )

    def prepare_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Pre-compute sessions once so the grid search doesn't repeat it for every combo."""
        return _compute_sessions(df)

    def run_backtest_prepared(
        self,
        prepared: List[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        return _run_prepared(self, prepared, params)

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.run_backtest_prepared(_compute_sessions(data), params)

    def run_monte_carlo(
        self,
        prepared: List[Dict[str, Any]],
        params: Dict[str, Any],
        n_sims: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        return _run_mc(self, prepared, params, n_sims, seed)


@register
class GoldBot6Tick(BaseStrategy):
    """
    GoldBot6 on N-tick OHLCV bars from MySQL tick_data (symbol: GC).

    tick_bar_size is part of the optimisation grid.

    Run optimisation:
        python -m strategy_platform.optimize.pipeline --strategy goldbot6_tick

    Note: requires GC tick data in the emini.tick_data table.
    """

    name           = 'goldbot6_tick'
    bar_type       = 'tick'
    symbol         = 'MGC'
    db_host: Optional[str] = None

    tick_size      = 0.10
    tick_value     = 1.00
    commission_rt  = 1.52

    default_params = {
        'qty': 1, 'stop_loss_ticks': 100, 'profit_target_ticks': 100,
        'trailing_trigger_ticks': 18, 'trailing_stop_ticks': 2,
        'direction': 'both', 'use_pd_levels': True, 'use_prev_session': True,
        'breakout_offset_ticks': 0, 'start_trading': '20:00', 'stop_trading': '15:59',
        'ps_window_start': '03:00', 'ps_window_end': '09:00', 'block_ps_while_forming': True,
        'tick_bar_size': 100,
    }

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            'stop_loss_ticks':        (10, 200, 10),
            'profit_target_ticks':    (20, 300, 10),
            'trailing_trigger_ticks': (2, 50, 2),
            'trailing_stop_ticks':    (1, 20, 1),
            'direction':              ['both', 'long_only', 'short_only'],
            'use_pd_levels':          [True, False],
            'use_prev_session':       [True, False],
            'breakout_offset_ticks':  (0, 5, 1),
            'start_trading':          ['00:00', '18:00', '18:30', '20:00'],
            'stop_trading':           ['12:00', '14:00', '15:59', '16:00', '16:45'],
            'ps_window_start':        ['02:00', '03:00', '04:00'],
            'ps_window_end':          ['07:00', '08:00', '09:00'],
            'block_ps_while_forming': [True, False],
            'qty':                    (1, 4, 1),
            'tick_bar_size':          (50, 1000, 50),
        }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            "Bar Size":         ["tick_bar_size"],
            "Levels":           ["use_pd_levels", "use_prev_session", "block_ps_while_forming"],
            "Targets & Stops":  ["stop_loss_ticks", "profit_target_ticks", "trailing_trigger_ticks", "trailing_stop_ticks"],
            "Entry":            ["breakout_offset_ticks", "qty"],
            "Direction":        ["direction"],
            "Session Timing":   ["start_trading", "stop_trading"],
            "London Window":    ["ps_window_start", "ps_window_end"],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            "tick_bar_size":          "Tick Bar Size",
            "use_pd_levels":          "PD Levels",
            "use_prev_session":       "Prev Session",
            "block_ps_while_forming": "Block PS While Forming",
            "qty":                    "Qty",
            "stop_loss_ticks":        "Stop Loss (ticks)",
            "profit_target_ticks":    "Profit Target (ticks)",
            "trailing_trigger_ticks": "Trail Trigger (ticks)",
            "trailing_stop_ticks":    "Trail Stop (ticks)",
            "direction":              "Direction",
            "breakout_offset_ticks":  "Breakout Offset (ticks)",
            "start_trading":          "Session Start",
            "stop_trading":           "Session End",
            "ps_window_start":        "London Window Start",
            "ps_window_end":          "London Window End",
        }

    @property
    def description(self) -> str:
        return (
            "GoldBot6: PDH/PDL + London-window breakout on GC (N-tick bars). "
            "Up to 4 simultaneous independent positions per session with per-entry PT/SL/trail."
        )

    def prepare_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        return _compute_sessions(df)

    def run_backtest_prepared(
        self,
        prepared: List[Dict[str, Any]],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        return _run_prepared(self, prepared, params)

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        return self.run_backtest_prepared(_compute_sessions(data), params)

    def run_monte_carlo(
        self,
        prepared: List[Dict[str, Any]],
        params: Dict[str, Any],
        n_sims: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        return _run_mc(self, prepared, params, n_sims, seed)
