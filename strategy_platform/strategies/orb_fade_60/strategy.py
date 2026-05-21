"""
OrbFade60 — fade the first break of the 60-minute opening range.

Lifecycle per US cash session (NY ET):
  1. Track high/low of bars between session_open (09:30) and session_open + range_minutes (10:30).
     Lock the OR when the range window closes.
  2. After the OR is locked, watch each subsequent bar for the first time price
     pierces above OR_high or below OR_low.
  3. On first break: enter a fade trade at break_price.
       - Break UP   -> SHORT (bet on reversal back into range)
       - Break DOWN -> LONG  (bet on reversal back into range)
     Stop   = break_price +/- (stop_r * OR_size)  in the break direction (against us)
     Target = opposite side of the OR (= +1.0 * OR_size in our favour)
  4. Walk 1m bars from break+1 forward; first-touch resolves exit.
     Ambiguous bar (both stop and target tagged) -> pessimistic stop-first.
  5. Time-stop: if neither hit by break_time + outcome_minutes, exit at last close.

Optional "large_only" filter:
  Only trade when today's OR size is in the top (1 - size_percentile) of the
  trailing size_lookback_days. Uses past data only (no lookahead).

This is the mechanical-only spec discovered via the orb-reversal-ml research
project after determining the RF model added no value over the raw rule set.
"""

from __future__ import annotations

from datetime import time as time_t
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register
from strategy_platform.strategies.mobobands.strategy import _summarise, _bootstrap_trades


WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


@register
class OrbFade60(BaseStrategy):
    """60-min opening-range fade. 1m bars, ET timezone-naive."""

    name = "orb_fade_60"
    bar_type            = 'time'
    supported_bar_types = ['1m', 'time']

    default_params: Dict[str, Any] = {
        'session_open_hour':    9,
        'session_open_min':     30,
        'range_minutes':        60,
        'outcome_minutes':      60,
        'target_r_multiple':    1.0,
        'stop_r_multiple':      0.5,
        'use_size_filter':      True,
        'size_lookback_days':   90,
        'size_percentile':      0.67,
        'direction':            'Both',
        'eod_exit_time':        '16:00',
        'use_risk_sizing':      False,
        'max_risk':             300,
        'qty':                  1,
    }

    # MNQ micro Nasdaq defaults — overridden by dashboard when symbol changes
    tick_size     = 0.25
    tick_value    = 0.50      # $0.50 per 0.25-pt tick = $2 per point
    commission_rt = 0.74

    symbol  = 'MNQ=F'
    db_host: Optional[str] = None

    # ------------------------------------------------------------------
    # param_grid / groups / display
    # ------------------------------------------------------------------

    @property
    def param_grid(self) -> Dict[str, Any]:
        return {
            # 1. Session window — fixed at 09:30 ET
            'session_open_hour':   [9],
            'session_open_min':    [30],

            # 2. Range / outcome geometry
            'range_minutes':       [15, 30, 60],
            'outcome_minutes':     [30, 60, 120],

            # 3. R:R structure
            'target_r_multiple':   (0.5, 2.0, 0.25),
            'stop_r_multiple':     (0.25, 1.0, 0.25),

            # 4. Size filter
            'use_size_filter':     [True, False],
            'size_lookback_days':  [60, 90, 120, 180],
            'size_percentile':     (0.50, 0.85, 0.05),

            # 5. Direction
            'direction':           ['Both', 'Long Only', 'Short Only'],

            # 6. EOD
            'eod_exit_time':       ['15:30', '16:00', '16:30', '16:55'],

            # 7. Risk sizing
            'use_risk_sizing':     [True, False],
            'max_risk':            (50.0, 500.0, 50.0),
            'qty':                 (1, 5, 1),
        }

    param_conditional: Dict[str, Tuple[str, Any]] = {
        'size_lookback_days': ('use_size_filter', True),
        'size_percentile':    ('use_size_filter', True),
        'max_risk':           ('use_risk_sizing', True),
        'qty':                ('use_risk_sizing', False),
    }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            "1. Session":     ['session_open_hour', 'session_open_min'],
            "2. Range/Outcome": ['range_minutes', 'outcome_minutes'],
            "3. R:R":         ['target_r_multiple', 'stop_r_multiple'],
            "4. Size Filter": ['use_size_filter', 'size_lookback_days', 'size_percentile'],
            "5. Direction":   ['direction'],
            "6. EOD":         ['eod_exit_time'],
            "7. Risk":        ['use_risk_sizing', 'max_risk', 'qty'],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            'session_open_hour':  'Session Open Hour (ET)',
            'session_open_min':   'Session Open Minute',
            'range_minutes':      'OR Length (min)',
            'outcome_minutes':    'Time Stop (min after break)',
            'target_r_multiple':  'Target (× OR size)',
            'stop_r_multiple':    'Stop   (× OR size)',
            'use_size_filter':    'Use Large-Range Filter',
            'size_lookback_days': 'Size Lookback (days)',
            'size_percentile':    'Min Range Percentile',
            'direction':          'Direction',
            'eod_exit_time':      'EOD Exit Time',
            'use_risk_sizing':    'Use Risk Sizing',
            'max_risk':           'Max Risk ($)',
            'qty':                'Qty (fixed)',
        }

    @property
    def description(self) -> str:
        return ("Fade the first break of the 60-min opening range. "
                "Target = opposite side of OR; Stop = 0.5 × OR past entry; "
                "Time-stop after 60 min. Optional large-range filter.")

    # ------------------------------------------------------------------
    # Backtest / MC
    # ------------------------------------------------------------------

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        merged = {**self.default_params, **params}
        df = _ensure_1m(data)
        trades = _run_backtest_loop(
            df, merged,
            self.tick_size, self.tick_value, self.commission_rt,
        )
        total_sessions = int(df['close'].resample('D').last().count())
        stats     = _summarise(trades, total_sessions=total_sessions)
        bs        = _bootstrap_trades(trades, total_sessions=total_sessions)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return {**stats, **bs, 'total_trades': stats['trades'], 'trades': trades_df}

    def run_monte_carlo(
        self,
        prepared: pd.DataFrame,
        params: Dict[str, Any],
        n_sims: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        df = _ensure_1m(prepared)
        merged = {**self.default_params, **params}

        groups = [(d, grp) for d, grp in df.groupby(df.index.date)]
        rng = np.random.default_rng(seed)
        n   = len(groups)

        net_pnls: list = []
        sharpes:  list = []

        for _ in range(n_sims):
            order       = rng.permutation(n)
            shuffled_df = pd.concat([groups[i][1] for i in order])
            trades = _run_backtest_loop(
                shuffled_df, merged,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_1m(df: pd.DataFrame) -> pd.DataFrame:
    """Strategy logic is defined on 1m bars; resample if data is larger."""
    if len(df) < 3:
        return df
    diffs = df.index.to_series().diff().dropna()
    median_sec = diffs.median().total_seconds()
    if median_sec > 90:  # larger than ~1m bars; nothing we can do, return as-is
        return df
    if median_sec < 50:  # sub-1m -> aggregate to 1m
        return df.resample('1min').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }).dropna()
    return df


def _parse_time(s: str) -> time_t:
    h, m = int(s.split(':')[0]), int(s.split(':')[1])
    return time_t(h, m)


# ---------------------------------------------------------------------------
# Single-pass backtest loop
# ---------------------------------------------------------------------------

def _run_backtest_loop(
    df:           pd.DataFrame,
    params:       Dict[str, Any],
    tick_size:    float,
    tick_value:   float,
    commission:   float,
) -> List[Dict[str, Any]]:
    """Per-session: build OR -> watch for first break -> fade -> first-touch exit."""
    if len(df) < 100:
        return []

    # ---- Params
    open_h     = int(params['session_open_hour'])
    open_m     = int(params['session_open_min'])
    range_min  = int(params['range_minutes'])
    outcome_m  = int(params['outcome_minutes'])
    target_r   = float(params['target_r_multiple'])
    stop_r     = float(params['stop_r_multiple'])
    use_filter = bool(params['use_size_filter'])
    lookback   = int(params['size_lookback_days'])
    pct        = float(params['size_percentile'])
    direction  = str(params['direction'])
    can_long   = direction in ('Both', 'Long Only')
    can_short  = direction in ('Both', 'Short Only')
    eod_t      = _parse_time(str(params['eod_exit_time']))
    use_risk   = bool(params['use_risk_sizing'])
    max_risk   = float(params['max_risk'])
    qty_fixed  = max(1, int(params['qty']))
    point_value = tick_value / tick_size  # $ per 1 point of price movement

    session_open  = time_t(open_h, open_m)
    range_end_min = open_h * 60 + open_m + range_min
    range_end     = time_t(range_end_min // 60, range_end_min % 60)

    # ---- Pre-compute per-session OR sizes for the rolling filter
    or_sizes_by_date: Dict[Any, float] = {}
    for d, day_bars in df.groupby(df.index.date):
        rb = day_bars[(day_bars.index.time >= session_open) & (day_bars.index.time < range_end)]
        if len(rb) > 0:
            or_sizes_by_date[d] = float(rb['high'].max() - rb['low'].min())

    sorted_dates = sorted(or_sizes_by_date)

    def _passes_size_filter(today: Any) -> bool:
        if not use_filter:
            return True
        # use only PAST days (strictly < today) — no lookahead
        idx = sorted_dates.index(today) if today in or_sizes_by_date else -1
        past = sorted_dates[max(0, idx - lookback):idx]
        if len(past) < 20:
            return True  # not enough history yet; let it trade
        past_sizes = np.array([or_sizes_by_date[d] for d in past])
        threshold = float(np.quantile(past_sizes, pct))
        return or_sizes_by_date[today] >= threshold

    # ---- Per-session loop
    trades: List[Dict[str, Any]] = []

    for d, day_bars in df.groupby(df.index.date):
        rb = day_bars[(day_bars.index.time >= session_open) & (day_bars.index.time < range_end)]
        if len(rb) == 0:
            continue
        or_high = float(rb['high'].max())
        or_low  = float(rb['low'].min())
        or_size = or_high - or_low
        if or_size <= 0:
            continue

        if not _passes_size_filter(d):
            continue

        # Bars strictly after the range window
        post = day_bars[(day_bars.index.time >= range_end) & (day_bars.index.time < eod_t)]
        if len(post) == 0:
            continue

        # First break
        up_break  = post['high'] > or_high
        dn_break  = post['low']  < or_low
        any_break = up_break | dn_break
        if not any_break.any():
            continue
        first_idx = int(np.argmax(any_break.values))
        first_bar = post.iloc[first_idx]
        first_ts  = post.index[first_idx]

        is_up  = bool(up_break.iloc[first_idx])
        is_dn  = bool(dn_break.iloc[first_idx])
        if is_up and is_dn:
            continue  # ambiguous: single bar tagged both sides

        if is_up and not can_short:
            continue
        if is_dn and not can_long:
            continue

        # Entry, stop, target.
        # Entry = break-bar extreme (assumes a sell-stop / buy-stop order triggered
        # by the break, filled at the bar's worst-case price). This matches the
        # orb-reversal-ml research model. The target is always the opposite OR
        # boundary (= +1.0 * OR size from the boundary we broke, scaled by target_r).
        if is_up:
            side       = 'Short'
            entry_px   = float(first_bar['high'])
            stop_px    = entry_px + stop_r * or_size
            target_px  = or_high - target_r * or_size
        else:
            side       = 'Long'
            entry_px   = float(first_bar['low'])
            stop_px    = entry_px - stop_r * or_size
            target_px  = or_low + target_r * or_size

        stop_dist = abs(entry_px - stop_px)
        # Position sizing
        if use_risk:
            risk_per_ctr = stop_dist * point_value
            qty = int(max_risk / risk_per_ctr) if risk_per_ctr > 0 else 0
            if qty < 1:
                continue
        else:
            qty = qty_fixed

        # Walk bars after the break for first-touch resolution
        time_stop_min = (first_ts.hour * 60 + first_ts.minute) + outcome_m
        time_stop = time_t(min(23, time_stop_min // 60), time_stop_min % 60)
        # exit no later than eod_t either way
        if (time_stop.hour, time_stop.minute) > (eod_t.hour, eod_t.minute):
            time_stop = eod_t

        walk = post.iloc[first_idx + 1:]
        walk = walk[walk.index.time <= time_stop]

        exit_px:    Optional[float] = None
        exit_ts:    Optional[pd.Timestamp] = None
        exit_reason: str = 'time_stop'

        for ts, bar in walk.iterrows():
            hi, lo = float(bar['high']), float(bar['low'])
            if side == 'Short':
                hit_stop   = hi >= stop_px
                hit_target = lo <= target_px
            else:
                hit_stop   = lo <= stop_px
                hit_target = hi >= target_px

            if hit_stop and hit_target:
                # ambiguous bar: pessimistic stop-first
                exit_px, exit_ts, exit_reason = stop_px, ts, 'stop_ambiguous'
                break
            if hit_stop:
                exit_px, exit_ts, exit_reason = stop_px, ts, 'stop'
                break
            if hit_target:
                exit_px, exit_ts, exit_reason = target_px, ts, 'target'
                break

        if exit_px is None:
            if len(walk) == 0:
                continue
            exit_px = float(walk['close'].iloc[-1])
            exit_ts = walk.index[-1]
            exit_reason = 'time_stop'

        if side == 'Long':
            pnl_pts = exit_px - entry_px
        else:
            pnl_pts = entry_px - exit_px
        pnl_dollars = pnl_pts * point_value * qty - commission

        trades.append({
            'session_date': pd.Timestamp(d).date(),
            'day_of_week':  pd.Timestamp(d).day_name(),
            'side':         side,
            'entry_time':   first_ts,
            'exit_time':    exit_ts,
            'entry_price':  entry_px,
            'exit_price':   exit_px,
            'stop':         stop_px,
            'target':       target_px,
            'qty':          qty,
            'pnl':          pnl_dollars,
            'pnl_ticks':    pnl_pts / tick_size,
            'exit_reason':  exit_reason,
            'commission':   commission,
            'or_high':      or_high,
            'or_low':       or_low,
            'or_size_pts':  or_size,
        })

    return trades
