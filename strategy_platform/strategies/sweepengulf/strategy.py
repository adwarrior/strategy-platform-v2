"""
SweepEngulf — sweep + engulf two-candle liquidity-grab reversal.

One signal bar does both jobs — the sweep and the engulf:

  Bullish (long):
    1. low[0]  < low[1]    — sweeps the prior bar's low.
    2. close[0] > high[1]  — engulf: closes ABOVE the prior bar's high (close-based,
       strictly harder than a high/low overlap test — do not relax it).

  Bearish (short): exact mirror (high[0] > high[1], close[0] < low[1]).

Entry is at the NEXT bar's open (market order placed on the signal bar's close).
One position at a time, blocking — signals during an open trade are discarded.

See `/home/ad/Scripts/strategies/sweepengulf_spec.md` (single source of truth).
Section 12 of the spec lists deliberate overrides of the original Pine transcript —
those are NOT bugs, do not "fix" them back.

Single timeframe only: no higher-timeframe series, no second resample for
confirmation. Timeframe-portable — runs on whatever bar size it is given.
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

# Full 24h time options at 5-min granularity (288 entries), matches the platform
# convention used by supertrendfractal / aurora for all HH:mm params.
_HHMM_24H: List[str] = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 5)]


@register
class SweepEngulf(BaseStrategy):
    """Sweep + engulf two-candle liquidity-grab reversal. Single timeframe, ET."""

    name = "sweepengulf"
    bar_type            = '1m'   # load 1m from historical_data_1m, resample to target bar internally
    supported_bar_types = ['1m', '5m', 'time']

    # historical_data_1m is stored CENTRAL-TIME-naive. This strategy reasons about
    # ET clock times (session window, EOD), so the loader must shift CT->ET (+1h).
    # The pipeline/dashboard read this attribute and pass to_et=True to load_1m.
    # See loader.load_1m docstring + memory feedback_db_1m_is_central_time.
    db_timezone = 'ET'

    default_params: Dict[str, Any] = {
        'direction':          'Both',
        'prev_candle_dir':    'Any',
        'use_ema_filter':     False,
        'ema_length':         200,
        'use_session_filter': False,
        'session_start':      '09:30',
        'session_end':        '11:30',
        'eod_exit_time':      '16:55',
        'stop_mode':          'ATR',
        'atr_length':         14,
        'atr_mult':           1.0,
        'stop_offset_ticks':  0,
        'rr_ratio':           2.0,
        'same_bar_priority':  'StopFirst',
        'max_bars_in_trade':  0,
        'use_risk_sizing':    False,
        'max_risk':           100.0,
        'qty':                1,
        # bar-size (minutes) used when bar_type == 'time'/'1m' and we resample
        # 1m data up to the requested timeframe internally. 5 = spec default.
        'bar_size_minutes':   5,
    }

    # MNQ micro Nasdaq defaults — overridden by dashboard/pipeline via get_meta(symbol)
    tick_size     = 0.25
    tick_value    = 0.50      # $0.50 per 0.25-pt tick = $2 per point
    commission_rt = 0.74

    symbol  = 'MNQ'   # DB symbol key (NOT 'MNQ=F' — that returns 0 rows from historical_data_1m)
    db_host: Optional[str] = None

    # ------------------------------------------------------------------
    # param_grid / groups / display
    # ------------------------------------------------------------------

    @property
    def param_grid(self) -> Dict[str, Any]:
        return {
            # 1. Signal
            'direction':          ['Both', 'Long Only', 'Short Only'],
            'prev_candle_dir':    ['Any', 'Same'],

            # 2. Trend Filter
            'use_ema_filter':     [False, True],
            'ema_length':         (50, 300, 50),

            # 3. Session
            'use_session_filter': [False, True],
            'session_start':      list(_HHMM_24H),
            'session_end':        list(_HHMM_24H),
            'eod_exit_time':      list(_HHMM_24H),

            # 4. Risk
            'stop_mode':          ['ATR', 'CandleExtreme'],
            'atr_length':         (7, 28, 7),
            'atr_mult':           (0.5, 3.0, 0.5),
            'stop_offset_ticks':  (0, 8, 2),
            'rr_ratio':           (1.0, 4.0, 0.5),
            'same_bar_priority':  ['StopFirst', 'TargetFirst'],
            'max_bars_in_trade':  (0, 40, 10),

            # 5. Sizing
            'use_risk_sizing':    [False, True],
            'max_risk':           (50.0, 500.0, 50.0),
            'qty':                (1, 5, 1),
        }

    param_conditional: Dict[str, Tuple[str, Any]] = {
        'ema_length':        ('use_ema_filter', True),
        'session_start':     ('use_session_filter', True),
        'session_end':       ('use_session_filter', True),
        'atr_length':        ('stop_mode', 'ATR'),
        'atr_mult':          ('stop_mode', 'ATR'),
        'stop_offset_ticks': ('stop_mode', 'CandleExtreme'),
        'max_risk':          ('use_risk_sizing', True),
        'qty':               ('use_risk_sizing', False),
    }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            "1. Signal":       ['direction', 'prev_candle_dir'],
            "2. Trend Filter": ['use_ema_filter', 'ema_length'],
            "3. Session":      ['use_session_filter', 'session_start', 'session_end',
                                 'eod_exit_time'],
            "4. Risk":         ['stop_mode', 'atr_length', 'atr_mult', 'stop_offset_ticks',
                                 'rr_ratio', 'same_bar_priority', 'max_bars_in_trade'],
            "5. Sizing":       ['use_risk_sizing', 'max_risk', 'qty'],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            'direction':          'Direction',
            'prev_candle_dir':    'Prev Candle Direction Filter',
            'use_ema_filter':     'Use EMA Trend Filter',
            'ema_length':         'EMA Length',
            'use_session_filter': 'Use Session Filter',
            'session_start':      'Session Start (ET)',
            'session_end':        'Session End (ET)',
            'eod_exit_time':      'EOD Exit Time',
            'stop_mode':          'Stop Mode',
            'atr_length':         'ATR Length',
            'atr_mult':           'ATR Multiplier',
            'stop_offset_ticks':  'Stop Offset (ticks)',
            'rr_ratio':           'Reward:Risk Ratio',
            'same_bar_priority':  'Same-Bar Stop/Target Priority',
            'max_bars_in_trade':  'Time Stop (bars, 0=off)',
            'use_risk_sizing':    'Use Risk Sizing',
            'max_risk':           'Max Risk ($)',
            'qty':                'Qty (fixed)',
        }

    @property
    def description(self) -> str:
        return ("Sweep + engulf two-candle liquidity-grab reversal: one bar sweeps the "
                "prior bar's extreme then closes through its opposite extreme. "
                "ATR or candle-extreme stop, RR target, session-aware EOD exit.")

    # ------------------------------------------------------------------
    # Backtest / MC
    # ------------------------------------------------------------------

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        merged = {**self.default_params, **params}
        df = _ensure_bars(data, int(merged.get('bar_size_minutes', 5)))
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
        merged = {**self.default_params, **params}
        df = _ensure_bars(prepared, int(merged.get('bar_size_minutes', 5)))

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

def _ensure_bars(df: pd.DataFrame, bar_size_minutes: int) -> pd.DataFrame:
    """Strategy logic is defined on `bar_size_minutes` bars; resample 1m (or
    sub-target) input up, close-stamped (label='right', closed='right') to
    match NT's close-time bar-stamp convention. Larger-than-target input is
    returned as-is (nothing finer we can do)."""
    if len(df) < 3:
        return df
    target_sec = bar_size_minutes * 60
    diffs = df.index.to_series().diff().dropna()
    median_sec = diffs.median().total_seconds()
    if median_sec > target_sec * 1.1:  # coarser than target; nothing we can do
        return df
    if median_sec < target_sec * 0.9:  # finer than target -> aggregate, close-stamped
        rule = f'{bar_size_minutes}min'
        return df.resample(rule, label='right', closed='right').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }).dropna()
    return df


def _parse_time(s: str) -> time_t:
    h, m = int(s.split(':')[0]), int(s.split(':')[1])
    return time_t(h, m)


def _time_in_window(ts: pd.Timestamp, start_str: str, stop_str: str) -> bool:
    """True if ts falls in [start, stop). If start > stop, treats window as
    spanning midnight (e.g. 18:00 -> 06:00 matches 18:00-23:59 and 00:00-05:59).
    start == stop is treated as an empty window (no match)."""
    now = ts.time()
    s   = _parse_time(start_str)
    e   = _parse_time(stop_str)
    if s == e:
        return False
    if s < e:
        return s <= now < e
    return now >= s or now < e


def _session_end_for_entry(entry_ts: pd.Timestamp, eod_time: time_t,
                            session_break_hour: int = 18) -> pd.Timestamp:
    """Return the timestamp at which the session containing entry_ts ends.
    For CME equity futures the session runs from ~18:00 ET (prev day) to the
    EOD exit time (current day). An entry at e.g. Mon 19:55 belongs to
    Tuesday's session, so EOD fires at Tue eod_time, not Mon eod_time."""
    from datetime import timedelta as _td
    if entry_ts.time() < time_t(session_break_hour, 0):
        end_date = entry_ts.date()
    else:
        end_date = entry_ts.date() + _td(days=1)
    return pd.Timestamp.combine(end_date, eod_time)


def _compute_atr(df: pd.DataFrame, length: int) -> np.ndarray:
    """Wilder-style ATR (simple rolling mean of TR, matching the platform's
    other simple-ATR ports). NaN for the first `length` bars (warm-up)."""
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(df)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    tr_s = pd.Series(tr)
    atr = tr_s.rolling(window=length, min_periods=length).mean().values
    return atr


def _compute_ema(df: pd.DataFrame, length: int) -> np.ndarray:
    """EMA of close, NaN until `length` bars have accumulated (warm-up)."""
    ema = df['close'].ewm(span=length, adjust=False, min_periods=length).mean()
    return ema.values


# ---------------------------------------------------------------------------
# Single-pass backtest loop
# ---------------------------------------------------------------------------

def _run_backtest_loop(
    df:         pd.DataFrame,
    params:     Dict[str, Any],
    tick_size:  float,
    tick_value: float,
    commission: float,
) -> List[Dict[str, Any]]:
    """Single-timeframe sweep+engulf scan -> next-bar-open entry -> blocking
    one-position-at-a-time management with ATR/candle-extreme stop, RR target,
    time-stop, and session-aware EOD exit."""
    n = len(df)
    if n < 5:
        return []

    # ---- Params
    direction         = str(params['direction'])
    can_long          = direction in ('Both', 'Long Only', 'both', 'long_only')
    can_short         = direction in ('Both', 'Short Only', 'both', 'short_only')
    prev_candle_dir   = str(params['prev_candle_dir'])
    use_ema_filter    = bool(params['use_ema_filter'])
    ema_length        = int(params['ema_length'])
    use_session       = bool(params['use_session_filter'])
    session_start     = str(params['session_start'])
    session_end       = str(params['session_end'])
    eod_str           = str(params['eod_exit_time'])
    eod_time          = _parse_time(eod_str)
    stop_mode         = str(params['stop_mode'])
    atr_length        = int(params['atr_length'])
    atr_mult          = float(params['atr_mult'])
    stop_offset_ticks = float(params['stop_offset_ticks'])
    rr_ratio          = float(params['rr_ratio'])
    same_bar_priority = str(params['same_bar_priority'])
    max_bars_in_trade = int(params['max_bars_in_trade'])
    use_risk          = bool(params['use_risk_sizing'])
    max_risk          = float(params['max_risk'])
    qty_fixed         = max(1, int(params['qty']))
    point_value       = tick_value / tick_size  # $ per 1 point of price movement

    o_arr = df['open'].values
    h_arr = df['high'].values
    l_arr = df['low'].values
    c_arr = df['close'].values
    idx   = df.index

    # ---- Indicators (only compute what the active config needs)
    needs_atr = (stop_mode == 'ATR')
    atr_arr = _compute_atr(df, atr_length) if needs_atr else np.full(n, np.nan)
    ema_arr = _compute_ema(df, ema_length) if use_ema_filter else np.full(n, np.nan)

    trades: List[Dict[str, Any]] = []

    in_trade      = False
    side          = None
    entry_px      = entry_ts = None
    stop_px       = target_px = None
    stop_dist     = target_dist = None
    qty           = qty_fixed
    bars_in_trade = 0
    session_end_ts: Optional[pd.Timestamp] = None
    signal_time = prev_high = prev_low = None
    sweep_depth = engulf_margin = signal_bar_range = None
    prev_candle_bullish = None
    atr_at_signal = ema_at_signal = None

    i = 0
    while i < n - 1:
        if in_trade:
            hi, lo, cl = float(h_arr[i]), float(l_arr[i]), float(c_arr[i])
            ts = idx[i]
            bars_in_trade += 1

            exit_px = None
            exit_reason = None

            # (1) Time stop
            if max_bars_in_trade > 0 and bars_in_trade >= max_bars_in_trade:
                exit_px = cl
                exit_reason = 'time_stop'

            # (2) Stop / target (wick touch), ordered by same_bar_priority
            if exit_px is None:
                if side == 'Long':
                    hit_stop   = lo <= stop_px
                    hit_target = hi >= target_px
                else:
                    hit_stop   = hi >= stop_px
                    hit_target = lo <= target_px

                if hit_stop and hit_target:
                    if same_bar_priority == 'TargetFirst':
                        exit_px, exit_reason = target_px, 'target'
                    else:
                        exit_px, exit_reason = stop_px, 'stop'
                elif hit_stop:
                    exit_px, exit_reason = stop_px, 'stop'
                elif hit_target:
                    exit_px, exit_reason = target_px, 'target'

            # (3) Session-aware EOD
            if exit_px is None and session_end_ts is not None and ts >= session_end_ts:
                exit_px = cl
                exit_reason = 'eod'

            if exit_px is not None:
                if side == 'Long':
                    pnl_pts = exit_px - entry_px
                else:
                    pnl_pts = entry_px - exit_px
                pnl_dollars = pnl_pts * point_value * qty - commission

                trades.append({
                    'session_date':  pd.Timestamp(entry_ts).date(),
                    'day_of_week':   pd.Timestamp(entry_ts).day_name(),
                    'direction':     side,
                    'side':          side,
                    'signal_time':   signal_time,
                    'entry_time':    entry_ts,
                    'exit_time':     ts,
                    'entry_price':   entry_px,
                    'exit_price':    exit_px,
                    'stop':          stop_px,
                    'target':        target_px,
                    'qty':           qty,
                    'pnl':           pnl_dollars,
                    'pnl_ticks':     pnl_pts / tick_size,
                    'exit_reason':   exit_reason,
                    'commission':    commission,
                    'prev_high':          prev_high,
                    'prev_low':           prev_low,
                    'sweep_depth':        sweep_depth,
                    'engulf_margin':      engulf_margin,
                    'signal_bar_range':   signal_bar_range,
                    'prev_candle_bullish': prev_candle_bullish,
                    'stop_dist':          stop_dist,
                    'target_dist':        target_dist,
                    'bars_in_trade':      bars_in_trade,
                    'atr_at_signal':      atr_at_signal,
                    'ema_at_signal':      ema_at_signal,
                })

                in_trade = False
                side = entry_px = entry_ts = stop_px = target_px = None
                stop_dist = target_dist = None
                bars_in_trade = 0
                session_end_ts = None
                # Do not advance i here beyond the loop increment below —
                # a signal on this same bar is not evaluated (blocking model
                # discards concurrent signals; the next scan starts at i+1).

            i += 1
            continue

        # ---- Not in a trade: scan for a sweep+engulf signal on bar i (needs i-1)
        if i < 1:
            i += 1
            continue

        # Warm-up guard: only wait on indicators the active config actually needs.
        if needs_atr and np.isnan(atr_arr[i]):
            i += 1
            continue
        if use_ema_filter and np.isnan(ema_arr[i]):
            i += 1
            continue

        ph, pl = float(h_arr[i - 1]), float(l_arr[i - 1])
        po, pc = float(o_arr[i - 1]), float(c_arr[i - 1])
        sh, sl_, sc, so = float(h_arr[i]), float(l_arr[i]), float(c_arr[i]), float(o_arr[i])

        is_bull_sig = (sl_ < pl) and (sc > ph)
        is_bear_sig = (sh > ph) and (sc < pl)

        if not (is_bull_sig or is_bear_sig):
            i += 1
            continue
        if is_bull_sig and is_bear_sig:
            # degenerate — cannot happen given the strict inequalities above,
            # but guard defensively rather than assume.
            i += 1
            continue

        sig_side = 'Long' if is_bull_sig else 'Short'

        if sig_side == 'Long' and not can_long:
            i += 1
            continue
        if sig_side == 'Short' and not can_short:
            i += 1
            continue

        # prev-candle-direction filter
        if prev_candle_dir == 'Same':
            prev_bullish_flag = pc > po
            prev_bearish_flag = pc < po
            if sig_side == 'Long' and not prev_bullish_flag:
                i += 1
                continue
            if sig_side == 'Short' and not prev_bearish_flag:
                i += 1
                continue

        sig_ts = idx[i]

        # EMA trend filter (evaluated on the signal bar's own close, no shift)
        if use_ema_filter:
            ema_val = float(ema_arr[i])
            if sig_side == 'Long' and not (sc > ema_val):
                i += 1
                continue
            if sig_side == 'Short' and not (sc < ema_val):
                i += 1
                continue

        # Session filter — evaluated on the bar's close stamp
        if use_session and not _time_in_window(sig_ts, session_start, session_end):
            i += 1
            continue

        # Discard signals on the final bar (no i+1 to enter on)
        if i + 1 >= n:
            break

        cand_entry_px = float(o_arr[i + 1])
        cand_entry_ts = idx[i + 1]

        # ---- Stop / target
        if stop_mode == 'CandleExtreme':
            if sig_side == 'Long':
                cand_stop_px = sl_ - stop_offset_ticks * tick_size
            else:
                cand_stop_px = sh + stop_offset_ticks * tick_size
            cand_stop_dist = abs(cand_entry_px - cand_stop_px)
        else:  # ATR
            cand_stop_dist = float(atr_arr[i]) * atr_mult
            if sig_side == 'Long':
                cand_stop_px = cand_entry_px - cand_stop_dist
            else:
                cand_stop_px = cand_entry_px + cand_stop_dist

        if cand_stop_dist <= 0 or not np.isfinite(cand_stop_dist):
            i += 1
            continue

        cand_target_dist = cand_stop_dist * rr_ratio
        if sig_side == 'Long':
            cand_target_px = cand_entry_px + cand_target_dist
        else:
            cand_target_px = cand_entry_px - cand_target_dist

        # ---- Position sizing
        if use_risk:
            risk_per_ctr = cand_stop_dist * point_value
            cand_qty = int(max_risk / risk_per_ctr) if risk_per_ctr > 0 else 0
            if cand_qty < 1:
                i += 1
                continue
        else:
            cand_qty = qty_fixed

        # ---- Admit the trade
        in_trade   = True
        side       = sig_side
        entry_px   = cand_entry_px
        entry_ts   = cand_entry_ts
        stop_px    = cand_stop_px
        target_px  = cand_target_px
        stop_dist  = cand_stop_dist
        target_dist = cand_target_dist
        qty        = cand_qty
        bars_in_trade = 0
        session_end_ts = _session_end_for_entry(entry_ts, eod_time)

        signal_time  = sig_ts
        prev_high    = ph
        prev_low     = pl
        sweep_depth  = (pl - sl_) if sig_side == 'Long' else (sh - ph)
        engulf_margin = (sc - ph) if sig_side == 'Long' else (pl - sc)
        signal_bar_range = sh - sl_
        prev_candle_bullish = bool(pc > po)
        atr_at_signal = float(atr_arr[i]) if needs_atr else float('nan')
        ema_at_signal = float(ema_arr[i]) if use_ema_filter else float('nan')

        # The entry bar itself (i+1) must ALSO be checked for stop/target on
        # this same pass — advance to i+1 so it goes through the in_trade
        # branch on the next loop iteration.
        i += 1

    return trades
