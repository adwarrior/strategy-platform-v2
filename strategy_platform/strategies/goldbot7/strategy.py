"""
GoldBot7 — PDH/PDL fib-breakout strategy on Gold futures (GC=F).

Logic (ported from NinjaScript C#):
  - At 18:05 ET, place stop-limit orders at Prior Day High (long) and
    Prior Day Low (short).
  - 1R distance = (1 - stop_fib) × prior_day_range
  - Stop   = entry ∓ 1R × stop_loss_r
  - Target = entry ± 1R × take_profit_r
  - Cancel unfilled orders at cancel_time.
  - Force-exit any open position at 16:55 ET.

Session definition:
  GC CME trades ~23 hrs: 18:00 ET (prev day) to 17:00 ET.
  session_date = calendar date of the 17:00 ET close.

Position sizing:
  Fixed $300 max risk per trade.
  qty = floor(300 / (1R × stop_loss_r × point_value))

Source: /home/ad/dev/Scripts/strategies/GoldBot7 (NinjaScript C#)
Ported from: /home/ad/Scripts/GoldBot7-Optimizer/strategies/goldbot7.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import time as time_t
from typing import Any, Dict, List, Optional

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

@register
class GoldBot7(BaseStrategy):
    """GoldBot7: Prior Day High/Low fib-breakout on GC=F (Mini Gold futures)."""

    name = "goldbot7"

    default_params: Dict[str, Any] = {
        'stop_fib':          0.9,
        'take_profit_r':     2,
        'direction':         'Both',
        'place_orders_time': '18:05',
        'cancel_orders_time': '12:00',
        'exit_time':         '16:55',
        'use_risk_sizing':   True,
        'max_risk':          300.0,
        'qty':               1,
    }

    # Instrument metadata for GC=F
    tick_size    = 0.10
    tick_value   = 10.00
    commission_rt = 4.62

    # Instrument and data defaults
    symbol  = 'GC=F'
    db_host: Optional[str] = None  # reads DB_HOST from .env

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            'stop_fib':           (0.00, 1.00, 0.05),
            'take_profit_r':      (1.00, 5.00, 0.25),
            'direction':          ['Both', 'Long Only', 'Short Only'],
            'place_orders_time':  ['18:05', '18:30', '19:00'],
            'cancel_orders_time': ['09:00', '10:00', '11:00', '12:00', '13:00'],
            'exit_time':          ['15:00', '15:30', '16:00', '16:30', '16:55'],
            'use_risk_sizing':    [True, False],
            'max_risk':           (100.0, 1000.0, 100.0),
            'qty':                (1, 5, 1),
        }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            "Entry & Stops":  ["stop_fib", "take_profit_r"],
            "Direction":      ["direction"],
            "Session Timing": ["place_orders_time", "cancel_orders_time", "exit_time"],
            "Risk":           ["use_risk_sizing", "max_risk", "qty"],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            "stop_fib":           "Stop Fibonacci",
            "take_profit_r":      "Take Profit R",
            "direction":          "Direction",
            "place_orders_time":  "Place Orders Time",
            "cancel_orders_time": "Cancel Orders Time",
            "exit_time":          "Exit Positions Time",
            "use_risk_sizing":    "Use Risk Sizing",
            "max_risk":           "Max Risk ($)",
            "qty":                "Fallback Qty",
        }

    @property
    def description(self) -> str:
        return "PDH/PDL fib-breakout on Gold futures (GC=F). Set-and-forget, session-based."

    # ------------------------------------------------------------------
    # Optimization hooks
    # ------------------------------------------------------------------

    def prepare_data(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Pre-compute sessions once so the grid search doesn't repeat it 1,200×."""
        return _compute_sessions(df)

    def run_backtest_prepared(self, sessions: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Run GoldBot7 from pre-computed sessions (used by the pipeline worker)."""
        trades    = _run_backtest(sessions, params, self.tick_size, self.tick_value, self.commission_rt)
        stats     = _summarise(trades, total_sessions=len(sessions))
        bs        = _bootstrap_trades(trades, total_sessions=len(sessions))
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return {**stats, **bs, 'total_trades': stats['trades'], 'trades': trades_df}

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience entry point — pre-computes sessions then delegates."""
        sessions = _compute_sessions(data)
        return self.run_backtest_prepared(sessions, params)

    def run_monte_carlo(
        self,
        sessions: List[Dict[str, Any]],
        params: Dict[str, Any],
        n_sims: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Day-shuffle Monte Carlo: permute session order, measure PnL stability."""
        rng      = np.random.default_rng(seed)
        n        = len(sessions)
        net_pnls = []
        sharpes  = []

        for _ in range(n_sims):
            order    = rng.permutation(n)
            shuffled = [sessions[i] for i in order]
            trades   = _run_backtest(shuffled, params, self.tick_size, self.tick_value, self.commission_rt)
            stats    = _summarise(trades, total_sessions=n)
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


# ---------------------------------------------------------------------------
# Session computation
# ---------------------------------------------------------------------------

def _session_date(ts: pd.Timestamp):
    """GC session_date = calendar date of the 17:00 ET close."""
    return (ts + pd.Timedelta(days=1)).date() if ts.hour >= 18 else ts.date()


def _compute_sessions(df_5m: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Split 5M GC data into session dicts.
    Each session carries: session_date, day_of_week, pdh, pdl, bars.
    PDH/PDL = high/low of the prior 24-hr session.
    """
    df = df_5m.copy()
    df['_sd'] = pd.DatetimeIndex(df.index).map(_session_date)

    hl = df.groupby('_sd').agg(sh=('high', 'max'), sl=('low', 'min'))
    hl['pdh'] = hl['sh'].shift(1)
    hl['pdl'] = hl['sl'].shift(1)

    sessions: List[Dict[str, Any]] = []
    for sd, grp in df.groupby('_sd'):
        row = hl.loc[sd]
        if pd.isna(row['pdh']) or pd.isna(row['pdl']):
            continue
        sessions.append({
            'session_date': sd,
            'day_of_week':  pd.Timestamp(sd).day_name(),
            'pdh':          float(row['pdh']),
            'pdl':          float(row['pdl']),
            'bars':         grp.drop(columns='_sd'),
        })
    return sessions


# ---------------------------------------------------------------------------
# Per-session simulation
# ---------------------------------------------------------------------------

def _round_tick(price: float, tick_size: float) -> float:
    return round(price / tick_size) * tick_size


def _simulate_session(
    session:    Dict[str, Any],
    params:     Dict[str, Any],
    tick_size:  float,
    tick_value: float,
    commission: float,
) -> Optional[Dict[str, Any]]:
    """Simulate one trading session. Returns a trade dict or None if no fill."""
    pdh = session['pdh']
    pdl = session['pdl']
    rng = pdh - pdl
    if rng <= 0:
        return None

    one_r = (1.0 - params['stop_fib']) * rng
    if one_r <= 0:
        return None

    point_value = tick_value / tick_size     # $100/point for GC
    sl_dist     = one_r
    tp_dist     = one_r * params['take_profit_r']

    if params.get('use_risk_sizing', True):
        max_risk     = params.get('max_risk', 300.0)
        risk_per_ctr = sl_dist * point_value
        qty          = int(max_risk / risk_per_ctr) if risk_per_ctr > 0 else 0
        if qty <= 0:
            return None
    else:
        qty = max(1, int(params.get('qty', 1)))

    long_entry   = _round_tick(pdh, tick_size)
    short_entry  = _round_tick(pdl, tick_size)
    long_stop    = _round_tick(long_entry  - sl_dist, tick_size)
    long_target  = _round_tick(long_entry  + tp_dist, tick_size)
    short_stop   = _round_tick(short_entry + sl_dist, tick_size)
    short_target = _round_tick(short_entry - tp_dist, tick_size)

    _dir = str(params.get('direction', 'Both'))
    can_long  = _dir in ('Both', 'Long Only', 'both', 'long_only')
    can_short = _dir in ('Both', 'Short Only', 'both', 'short_only')

    def _parse_time(s: str) -> time_t:
        h, m = int(s.split(':')[0]), int(s.split(':')[1])
        return time_t(h, m)

    place_t  = _parse_time(params.get('place_orders_time',  '18:05'))
    cancel_t = _parse_time(params.get('cancel_orders_time', '12:00'))
    exit_t   = _parse_time(params.get('exit_time',          '16:55'))

    sd       = pd.Timestamp(session['session_date'])
    prev_day = sd - pd.Timedelta(days=1)

    placement_dt = prev_day.replace(hour=place_t.hour,  minute=place_t.minute,  second=0, microsecond=0)
    cancel_dt    = sd.replace(      hour=cancel_t.hour, minute=cancel_t.minute, second=0, microsecond=0)
    exit_dt      = sd.replace(      hour=exit_t.hour,   minute=exit_t.minute,   second=0, microsecond=0)

    state      = 'WAITING'
    fill_price = None
    fill_ts    = None
    long_ok    = can_long   # disabled if price already above PDH at placement
    short_ok   = can_short  # disabled if price already below PDL at placement
    placed     = False      # have we checked placement-bar price yet?

    for ts, bar in session['bars'].iterrows():
        if ts < placement_dt:
            continue

        # On the first bar at/after placement time, check whether price has
        # already breached the entry levels (mirrors NinjaScript's lastPrice check).
        if not placed:
            placed = True
            if bar['close'] > long_entry:
                long_ok = False   # already above PDH — no long order
            if bar['close'] < short_entry:
                short_ok = False  # already below PDL — no short order

        # Force exit
        if ts >= exit_dt:
            if state in ('LONG', 'SHORT'):
                exit_p      = bar['close']
                pnl_pts     = (exit_p - fill_price) if state == 'LONG' else (fill_price - exit_p)
                pnl_dollars = (pnl_pts / tick_size) * tick_value * qty - commission
                sl = long_stop if state == 'LONG' else short_stop
                return _make_trade(session, state.lower(), fill_price, exit_p, sl, qty,
                                   pnl_dollars, 'session_exit', commission, fill_ts, ts)
            break

        # Look for fill
        if state == 'WAITING':
            if ts > cancel_dt:
                break

            lh = long_ok  and bar['high'] >= long_entry
            sh = short_ok and bar['low']  <= short_entry

            if lh and sh:
                if abs(bar['open'] - long_entry) <= abs(bar['open'] - short_entry):
                    sh = False
                else:
                    lh = False

            if lh:
                state, fill_price, fill_ts = 'LONG',  long_entry, ts
            elif sh:
                state, fill_price, fill_ts = 'SHORT', short_entry, ts

        # Manage open position
        elif state == 'LONG':
            if bar['low'] <= long_stop:
                exit_p, reason = long_stop,   'stop'
            elif bar['high'] >= long_target:
                exit_p, reason = long_target, 'target'
            else:
                continue
            pnl_pts     = exit_p - fill_price
            pnl_dollars = (pnl_pts / tick_size) * tick_value * qty - commission
            return _make_trade(session, 'long', fill_price, exit_p, long_stop, qty,
                               pnl_dollars, reason, commission, fill_ts, ts)

        elif state == 'SHORT':
            if bar['high'] >= short_stop:
                exit_p, reason = short_stop,   'stop'
            elif bar['low'] <= short_target:
                exit_p, reason = short_target, 'target'
            else:
                continue
            pnl_pts     = fill_price - exit_p
            pnl_dollars = (pnl_pts / tick_size) * tick_value * qty - commission
            return _make_trade(session, 'short', fill_price, exit_p, short_stop, qty,
                               pnl_dollars, reason, commission, fill_ts, ts)

    return None


def _make_trade(session, direction, entry, exit_p, stop_p, qty, pnl, exit_reason, commission=0.0,
                entry_time=None, exit_time=None):
    return {
        'session_date': session['session_date'],
        'day_of_week':  session['day_of_week'],
        'entry_time':   entry_time,
        'exit_time':    exit_time,
        'direction':    direction,
        'entry_price':  entry,
        'exit_price':   exit_p,
        'stop':         stop_p,
        'qty':          qty,
        'pnl':          pnl,
        'commission':   commission,
        'exit_reason':  exit_reason,
    }


# ---------------------------------------------------------------------------
# Full backtest runner
# ---------------------------------------------------------------------------

def _run_backtest(
    sessions:   List[Dict[str, Any]],
    params:     Dict[str, Any],
    tick_size:  float,
    tick_value: float,
    commission: float,
) -> List[Dict[str, Any]]:
    """Run GoldBot7 over all sessions. Sessions can be shuffled for Monte Carlo."""
    trades = []
    for s in sessions:
        trade = _simulate_session(s, params, tick_size, tick_value, commission)
        if trade is not None:
            trades.append(trade)
    return trades


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

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
    """
    Aggregate trade list into performance metrics including per-day breakdown.

    Parameters
    ----------
    total_sessions : total number of sessions in the period (including non-trading
        days). Used to pad daily P&L with zeros so the Sharpe/Sortino annualisation
        is correct. If 0, only trading days are used (legacy behaviour — inflated Sharpe).
    """
    _empty = {
        'trades': 0, 'win_rate': 0.0, 'net_pnl': 0.0,
        'avg_win': 0.0, 'avg_loss': 0.0, 'profit_factor': 0.0,
        'max_drawdown': 0.0, 'sharpe': 0.0, 'sortino': 0.0,
        'ulcer_index': 0.0, 'r_squared': 0.0, 'pct_months_profit': 0.0,
        'longest_flat_days': 0, 'total_commission': 0.0,
        'start_date': '', 'end_date': '',
        **{f'{d[:3].lower()}_pnl': 0.0   for d in WEEKDAYS},
        **{f'{d[:3].lower()}_trades': 0  for d in WEEKDAYS},
    }
    if not trades:
        return _empty

    pnls   = np.array([t['pnl'] for t in trades], dtype=float)
    n      = len(pnls)
    wins   = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    even   = pnls[pnls == 0]

    cum   = np.cumsum(pnls)
    peak  = np.maximum.accumulate(cum)
    max_dd = float(-(peak - cum).max())

    daily_map: Dict = {}
    for t in trades:
        daily_map[t['session_date']] = daily_map.get(t['session_date'], 0.0) + t['pnl']
    sorted_dates = sorted(daily_map)
    d_vals       = np.array([daily_map[d] for d in sorted_dates], dtype=float)

    # Pad with zeros for non-trading sessions (used by Sortino only).
    n_zero = max(0, total_sessions - len(d_vals))
    if n_zero > 0:
        d_vals_padded = np.concatenate([d_vals, np.zeros(n_zero)])
    else:
        d_vals_padded = d_vals

    # ── NinjaTrader-style Sharpe: mean(monthly_pnl) / std(monthly_pnl) ───────
    # Uses raw monthly profits, risk-free rate = 0, not annualised.
    # All calendar months between first and last trade are included (zeros for
    # months with no trades). Returns 1.0 when < 2 months of history.
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
        # NinjaTrader-style: mean(monthly_pnl) / std(monthly_pnl), rf=0, not annualised.
        # Requires ≥6 months to avoid inflated ratios from tiny monthly std.
        m_std  = float(m_arr.std(ddof=1))
        sharpe = float(m_arr.mean() / m_std) if m_std > 0 else 1.0
    else:
        # Fallback for short periods: annualised daily Sharpe (zero-padded).
        std    = d_vals_padded.std(ddof=1) if len(d_vals_padded) > 1 else 0.0
        sharpe = float((d_vals_padded.mean() / std) * np.sqrt(252)) if std > 0 else 0.0

    max_consec_w, max_consec_l = _max_consec(pnls > 0)

    n_days             = len(daily_map)
    avg_trades_per_day = n / n_days if n_days > 0 else 0.0

    # profit_per_month and pct_months_profit use monthly_full (includes zero months)
    profit_per_month  = float(np.mean(list(monthly_full.values()))) if monthly_full else 0.0
    n_months_pos      = sum(1 for v in monthly_full.values() if v > 0)
    pct_months_profit = float(n_months_pos / len(monthly_full)) if monthly_full else 0.0

    max_recovery = _max_time_to_recover(trade_dates_all, pnls)

    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss   = float(losses.sum()) if len(losses) > 0 else 0.0

    # ── Ulcer Index (RMS of drawdowns from equity peak) ──────────────────────
    dd_series  = peak - cum
    ulcer_idx  = float(np.sqrt(np.mean(dd_series ** 2)))

    # ── R Squared (linearity of equity curve) ────────────────────────────────
    x      = np.arange(n, dtype=float)
    coef   = np.polyfit(x, cum, 1)
    y_hat  = np.polyval(coef, x)
    ss_res = float(((cum - y_hat) ** 2).sum())
    ss_tot = float(((cum - cum.mean()) ** 2).sum())
    r_sq   = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # ── Longest flat period (calendar days between new equity highs) ──────────
    high_dates  = []
    peak_val    = -np.inf
    for i, val in enumerate(cum):
        if val > peak_val:
            peak_val = val
            high_dates.append(trade_dates_all[i])
    if len(high_dates) >= 2:
        longest_flat = max((high_dates[i + 1] - high_dates[i]).days
                           for i in range(len(high_dates) - 1))
    else:
        longest_flat = 0

    # ── Total commission ──────────────────────────────────────────────────────
    total_commission = float(sum(t.get('commission', 0.0) for t in trades))

    # ── Period dates ─────────────────────────────────────────────────────────
    start_date = trade_dates_all[0]
    end_date   = trade_dates_all[-1]

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
        'start_date':          str(start_date),
        'end_date':            str(end_date),
    }

    for day in WEEKDAYS:
        key       = day[:3].lower()
        day_pnls  = [t['pnl'] for t in trades if t['day_of_week'] == day]
        stats[f'{key}_pnl']    = float(sum(day_pnls))
        stats[f'{key}_trades'] = len(day_pnls)

    return stats


def _bootstrap_trades(
    trades:          List[Dict[str, Any]],
    n_sims:          int = 1000,
    seed:            int = 42,
    total_sessions:  int = 0,
) -> dict:
    """Bootstrap daily P&L with replacement, including zero-return sessions."""
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

    # Pad with zeros for non-trading sessions
    n_zero = max(0, total_sessions - len(pnls))
    if n_zero > 0:
        pnls = np.concatenate([pnls, np.zeros(n_zero)])
    n    = len(pnls)

    idx       = rng.integers(0, n, size=(n_sims, n))
    s_pnls    = pnls[idx]
    net_pnls  = s_pnls.sum(axis=1)
    win_rates = (s_pnls > 0).mean(axis=1)
    stds      = s_pnls.std(axis=1, ddof=1)
    means     = s_pnls.mean(axis=1)
    # np.where evaluates both branches, so the division runs even where
    # stds==0 (zero-variance samples) and emits a harmless 0/0 warning that
    # the where() then discards. Silence it — the result is already correct.
    with np.errstate(divide='ignore', invalid='ignore'):
        sharpes = np.where(stds > 0, (means / stds) * np.sqrt(252), 0.0)

    return {
        'bs_sharpe_p5': float(np.percentile(sharpes,   5)),
        'bs_pnl_p5':    float(np.percentile(net_pnls,  5)),
        'bs_pnl_p50':   float(np.percentile(net_pnls, 50)),
        'bs_wr_p5':     float(np.percentile(win_rates, 5)),
    }
