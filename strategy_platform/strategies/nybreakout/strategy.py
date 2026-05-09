"""
NYBreakout — 9-10am NY hourly anchor + FVG-entry breakout strategy.

Ported from /home/ad/Scripts/strategies/NYBreakout.cs (NinjaScript C#).
Spec: /home/ad/Scripts/strategies/NYBreakout_spec.md.

Lifecycle per session:
  1. Track high/low of bars between anchor_start_hour and anchor_end_hour ET.
     Lock the anchor on the bar that closes at anchor_end_hour:00.
  2. Wait for a 5-min bar to close outside the locked range.
     - close > anchor_high  -> bullish bias
     - close < anchor_low   -> bearish bias
  3. Detect 3-bar FVGs in the bias direction; rank by precedence:
       overlap-with-anchor (+1000) > outside-range (+100) > closest-to-edge.
  4. Place a limit order at the best FVG; re-target if a higher-precedence
     FVG appears (when allow_limit_retarget=True).
  5. On fill: stop = 1 tick beyond candle 1 of the FVG; target = rr_target * stop_dist.
  6. Cancel pending limit + force-exit at eod_exit_time (default 16:55).

Bias may flip on opposite-side close while flat & no fill pending. Multiple trades
per session up to max_trades_per_day. BoS-cancel optional (default off).

Risk sizing matches GoldBot7: max_risk / (stop_distance * point_value); skip if qty<1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import time as time_t, datetime
from typing import Any, Dict, List, Optional, Tuple

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register
from strategy_platform.strategies.mobobands.strategy import _summarise, _bootstrap_trades


WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

@register
class NYBreakout(BaseStrategy):
    """9-10am NY hourly anchor + FVG-entry breakout. 5-min bars, ET timezone-naive."""

    name = "nybreakout"
    bar_type            = 'time'             # default; overridden by dashboard
    supported_bar_types = ['time', '1m']     # 1m is auto-resampled to 5M internally

    default_params: Dict[str, Any] = {
        'anchor_start_hour': 9,
        'anchor_end_hour': 10,
        'min_fvg_ticks': 4,
        'include_pre_trigger_fvgs': True,
        'prefer_overlap': True,
        'prefer_outside': True,
        'prefer_closest': True,
        'allow_limit_retarget': True,
        'bos_cancel_enabled': False,
        'bos_cancel_count': 3,
        'max_trades_per_day': 3,
        'entry_cutoff_time': '13:00',
        'cancel_pending_at_cutoff': False,
        'rr_target': 1,
        'eod_exit_time': '16:55',
        'direction': 'Both',
        'use_risk_sizing': False,
        'max_risk': 300,
        'qty': 1,
    }

    # MES defaults — overridden by dashboard when symbol changes
    tick_size     = 0.25
    tick_value    = 1.25
    commission_rt = 1.24

    symbol  = 'MES=F'
    db_host: Optional[str] = None

    # ------------------------------------------------------------------
    # param_grid / groups / display
    # ------------------------------------------------------------------

    @property
    def param_grid(self) -> Dict[str, Any]:
        return {
            # Anchor — fixed at 9-10am ET (NY institutional positioning hour); not swept
            'anchor_start_hour':       [9],
            'anchor_end_hour':         [10],

            # FVG Detection
            'min_fvg_ticks':           (1, 20, 1),
            'include_pre_trigger_fvgs': [True, False],

            # Entry / Selection
            'prefer_overlap':          [True, False],
            'prefer_outside':          [True, False],
            'prefer_closest':          [True, False],
            'allow_limit_retarget':    [True, False],
            'bos_cancel_enabled':      [True, False],
            'bos_cancel_count':        (1, 6, 1),
            'max_trades_per_day':      (1, 5, 1),
            'entry_cutoff_time':       ['10:00', '10:30', '11:00', '11:30', '12:00', '13:00', '14:00'],
            'cancel_pending_at_cutoff': [True, False],

            # Exit
            'rr_target':               (0.5, 5.0, 0.25),
            'eod_exit_time':           ['15:00', '15:30', '16:00', '16:30', '16:55'],

            # Direction
            'direction':               ['Both', 'Long Only', 'Short Only'],

            # Risk
            'use_risk_sizing':         [True, False],
            'max_risk':                (50.0, 500.0, 50.0),
            'qty':                     (1, 5, 1),
        }

    # Sub-params hidden when parent toggle is off / wrong state
    param_conditional: Dict[str, Tuple[str, Any]] = {
        'bos_cancel_count': ('bos_cancel_enabled', True),
        'max_risk':         ('use_risk_sizing',    True),
        'qty':              ('use_risk_sizing',    False),
    }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            "1. Anchor Range":  ['anchor_start_hour', 'anchor_end_hour'],
            "2. FVG Detection": ['min_fvg_ticks', 'include_pre_trigger_fvgs'],
            "3. Entry":         ['prefer_overlap', 'prefer_outside', 'prefer_closest',
                                 'allow_limit_retarget', 'bos_cancel_enabled', 'bos_cancel_count',
                                 'max_trades_per_day', 'entry_cutoff_time', 'cancel_pending_at_cutoff'],
            "4. Exit":          ['rr_target', 'eod_exit_time'],
            "5. Direction":     ['direction'],
            "6. Risk":          ['use_risk_sizing', 'max_risk', 'qty'],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            'anchor_start_hour':       'Anchor Start Hour (ET)',
            'anchor_end_hour':         'Anchor End Hour (ET)',
            'min_fvg_ticks':           'Min FVG Ticks',
            'include_pre_trigger_fvgs': 'Include Pre-Trigger FVGs',
            'prefer_overlap':          'Prefer Overlap',
            'prefer_outside':          'Prefer Outside',
            'prefer_closest':          'Prefer Closest',
            'allow_limit_retarget':    'Allow Limit Retarget',
            'bos_cancel_enabled':      'BoS Cancel Enabled',
            'bos_cancel_count':        'BoS Cancel Count',
            'max_trades_per_day':      'Max Trades Per Day',
            'entry_cutoff_time':       'Entry Cutoff Time',
            'cancel_pending_at_cutoff': 'Cancel Pending At Cutoff',
            'rr_target':               'RR Target',
            'eod_exit_time':           'EOD Exit Time',
            'direction':               'Direction',
            'use_risk_sizing':         'Use Risk Sizing',
            'max_risk':                'Max Risk ($)',
            'qty':                     'Qty (fixed)',
        }

    @property
    def description(self) -> str:
        return "9-10am NY hourly anchor + FVG-entry breakout (ported from NYBreakout.cs)."

    # ------------------------------------------------------------------
    # Backtest / MC
    # ------------------------------------------------------------------

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        merged = {**self.default_params, **params}
        df = _ensure_5m(data)
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
        """Day-shuffle Monte Carlo: permute trading-day order and re-run."""
        df = _ensure_5m(prepared)
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

def _ensure_5m(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect bar size from the median index spacing; resample to 5M if smaller.
    The dashboard may pass 1M bars (when symbol is in historical_data_1m); strategy
    logic is defined on 5M, so we resample once here.
    """
    if len(df) < 3:
        return df
    diffs = df.index.to_series().diff().dropna()
    median_sec = diffs.median().total_seconds()
    if median_sec < 240:  # < 4 min  -> treat as sub-5M, resample
        return df.resample('5min').agg({
            'open':   'first',
            'high':   'max',
            'low':    'min',
            'close':  'last',
            'volume': 'sum',
        }).dropna()
    return df


def _parse_time(s: str) -> time_t:
    h, m = int(s.split(':')[0]), int(s.split(':')[1])
    return time_t(h, m)


def _round_tick(price: float, tick_size: float) -> float:
    return round(price / tick_size) * tick_size


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
    """
    Single-pass simulation mirroring NYBreakout.cs OnBarUpdate.

    State is reset per ET calendar day. Each bar:
      1. Day rollover: reset all session state.
      2. EOD block: cancel limit, force-exit, return.
      3. Anchor accumulation / lock.
      4. FVG detection (3-bar pattern, look-back 2).
      5. Bias engagement / flip (only when flat + no pending limit).
      6. BoS counting (if enabled).
      7. Entry: select best FVG, place / re-target limit.
      8. Fill check (intra-bar): limit -> stop / target.
    """
    if len(df) < 5:
        return []

    # ---- Param unpack
    anchor_start = int(params['anchor_start_hour'])
    anchor_end   = int(params['anchor_end_hour'])
    min_fvg_pts  = float(params['min_fvg_ticks']) * tick_size

    include_pre_trigger = bool(params['include_pre_trigger_fvgs'])
    prefer_overlap      = bool(params['prefer_overlap'])
    prefer_outside      = bool(params['prefer_outside'])
    prefer_closest      = bool(params['prefer_closest'])
    allow_retarget      = bool(params['allow_limit_retarget'])
    bos_cancel_enabled  = bool(params['bos_cancel_enabled'])
    bos_cancel_count_p  = int(params['bos_cancel_count'])
    max_trades_per_day  = int(params['max_trades_per_day'])

    rr_target      = float(params['rr_target'])
    eod_t          = _parse_time(str(params['eod_exit_time']))
    entry_cutoff_t         = _parse_time(str(params['entry_cutoff_time']))
    cancel_pending_cutoff  = bool(params['cancel_pending_at_cutoff'])

    direction = str(params['direction'])
    can_long  = direction in ('Both', 'Long Only', 'both', 'long_only')
    can_short = direction in ('Both', 'Short Only', 'both', 'short_only')

    use_risk_sizing = bool(params['use_risk_sizing'])
    max_risk        = float(params['max_risk'])
    qty_fixed       = max(1, int(params['qty']))

    point_value = tick_value / tick_size

    # ---- Materialise as numpy arrays for speed
    idx     = df.index
    high    = df['high'].to_numpy(dtype=float)
    low     = df['low'].to_numpy(dtype=float)
    o_arr   = df['open'].to_numpy(dtype=float)
    close   = df['close'].to_numpy(dtype=float)
    n       = len(df)

    # ---- Session state (reset on day rollover)
    cur_date = None
    anchor_high = 0.0
    anchor_low  = 0.0
    anchor_locked = False
    bias = 'None'  # 'None' | 'Long' | 'Short'
    breakout_trigger_bar = -1
    fvgs: List[Dict[str, Any]] = []
    max_high_since_bias = -np.inf
    min_low_since_bias  = np.inf
    bos_count = 0
    bos_cancelled_for_day = False
    trades_today = 0
    long_taken_today = False
    short_taken_today = False
    eod_block_active = False
    used_fvg_keys: set = set()

    # Pending limit
    pending_side: Optional[str] = None    # 'Long' | 'Short' | None
    pending_entry: float = 0.0
    pending_stop:  float = 0.0
    pending_target: float = 0.0
    pending_qty:   int = 0
    pending_fvg_key: int = -1

    # Open trade
    in_trade = False
    trade_side: Optional[str] = None
    trade_entry: float = 0.0
    trade_stop:  float = 0.0
    trade_target: float = 0.0
    trade_qty:   int = 0
    trade_entry_ts: Optional[pd.Timestamp] = None
    trade_fvg_key: int = -1

    trades: List[Dict[str, Any]] = []

    def _reset_day_state():
        nonlocal anchor_high, anchor_low, anchor_locked
        nonlocal bias, breakout_trigger_bar, fvgs
        nonlocal max_high_since_bias, min_low_since_bias, bos_count, bos_cancelled_for_day
        nonlocal trades_today, long_taken_today, short_taken_today, eod_block_active, used_fvg_keys
        nonlocal pending_side, pending_entry, pending_stop, pending_target, pending_qty, pending_fvg_key
        anchor_high = 0.0
        anchor_low  = 0.0
        anchor_locked = False
        bias = 'None'
        breakout_trigger_bar = -1
        fvgs = []
        max_high_since_bias = -np.inf
        min_low_since_bias  = np.inf
        bos_count = 0
        bos_cancelled_for_day = False
        trades_today = 0
        long_taken_today = False
        short_taken_today = False
        eod_block_active = False
        used_fvg_keys = set()
        pending_side = None
        pending_entry = 0.0
        pending_stop  = 0.0
        pending_target = 0.0
        pending_qty   = 0
        pending_fvg_key = -1

    def _close_trade(exit_price: float, exit_ts, exit_reason: str):
        nonlocal in_trade, trade_side, trades_today
        nonlocal long_taken_today, short_taken_today
        used_fvg_keys.add(trade_fvg_key)
        if trade_side == 'Long':
            long_taken_today = True
        elif trade_side == 'Short':
            short_taken_today = True
        pnl_pts     = (exit_price - trade_entry) if trade_side == 'Long' else (trade_entry - exit_price)
        pnl_dollars = (pnl_pts / tick_size) * tick_value * trade_qty - commission
        sd = pd.Timestamp(exit_ts).date()
        trades.append({
            'session_date':  sd,
            'day_of_week':   pd.Timestamp(sd).day_name(),
            'side':          trade_side,
            'entry_time':    trade_entry_ts,
            'exit_time':     exit_ts,
            'entry_price':   trade_entry,
            'exit_price':    exit_price,
            'stop':          trade_stop,
            'target':        trade_target,
            'qty':           trade_qty,
            'pnl':           pnl_dollars,
            'pnl_ticks':     pnl_pts / tick_size,
            'fvg_key':       trade_fvg_key,
            'exit_reason':   exit_reason,
            'commission':    commission,
        })
        in_trade = False
        trade_side = None
        trades_today += 1

    def _select_best_fvg() -> Optional[Dict[str, Any]]:
        if bias == 'None' or not fvgs:
            return None
        eligible = []
        for f in fvgs:
            if f['direction'] != ('bull' if bias == 'Long' else 'bear'):
                continue
            if not include_pre_trigger and f['bar_index'] < breakout_trigger_bar:
                continue
            if f['bar_index'] in used_fvg_keys:
                continue
            # Reject FVGs formed while price had retraced back inside the anchor range
            if bias == 'Long'  and f['candle3_close'] < anchor_high:
                continue
            if bias == 'Short' and f['candle3_close'] > anchor_low:
                continue
            eligible.append(f)
        if not eligible:
            return None

        def _score(f: Dict[str, Any]) -> float:
            s = 0.0
            top, bot = f['top'], f['bottom']
            if bias == 'Long':
                edge = anchor_high
                # overlap: gap straddles or touches the anchor edge
                overlaps = bot <= edge <= top
                outside  = bot >= edge
                dist_pts = abs(top - edge) if top >= edge else abs(bot - edge)
            else:
                edge = anchor_low
                overlaps = bot <= edge <= top
                outside  = top <= edge
                dist_pts = abs(bot - edge) if bot <= edge else abs(top - edge)

            if prefer_overlap and overlaps:
                s += 1000.0
            if prefer_outside and outside:
                s += 100.0
            if prefer_closest:
                # closer = higher score; convert distance to ticks for stable scaling
                s += max(0.0, 100.0 - (dist_pts / tick_size))
            # recency tie-break
            s += f['bar_index'] * 1e-9
            return s

        scored = [(f, _score(f)) for f in eligible]
        scored.sort(key=lambda x: x[1], reverse=True)
        best, best_score = scored[0]
        if best_score <= 0:
            # fall back to most-recent bias-aligned FVG
            return max(eligible, key=lambda f: f['bar_index'])
        return best

    def _compute_qty(stop_dist_pts: float) -> int:
        if use_risk_sizing:
            risk_per_ctr = stop_dist_pts * point_value
            if risk_per_ctr <= 0:
                return 0
            return int(max_risk / risk_per_ctr)
        return qty_fixed

    def _place_limit(fvg: Dict[str, Any]):
        nonlocal pending_side, pending_entry, pending_stop, pending_target, pending_qty, pending_fvg_key
        if bias == 'Long':
            entry = _round_tick(fvg['top'], tick_size)
            stop  = _round_tick(fvg['candle1_low'] - tick_size, tick_size)
            stop_dist = entry - stop
            if stop_dist <= 0:
                return
            target = _round_tick(entry + rr_target * stop_dist, tick_size)
            q = _compute_qty(stop_dist)
            if q < 1:
                return
            pending_side = 'Long'
        else:
            entry = _round_tick(fvg['bottom'], tick_size)
            stop  = _round_tick(fvg['candle1_high'] + tick_size, tick_size)
            stop_dist = stop - entry
            if stop_dist <= 0:
                return
            target = _round_tick(entry - rr_target * stop_dist, tick_size)
            q = _compute_qty(stop_dist)
            if q < 1:
                return
            pending_side = 'Short'

        pending_entry  = entry
        pending_stop   = stop
        pending_target = target
        pending_qty    = q
        pending_fvg_key = fvg['bar_index']

    def _cancel_limit():
        nonlocal pending_side, pending_entry, pending_stop, pending_target, pending_qty, pending_fvg_key
        pending_side = None
        pending_entry = 0.0
        pending_stop = 0.0
        pending_target = 0.0
        pending_qty = 0
        pending_fvg_key = -1

    # ---- Main loop
    for i in range(n):
        ts: pd.Timestamp = idx[i]
        bar_date = ts.date()

        # Day rollover
        if cur_date != bar_date:
            _reset_day_state()
            cur_date = bar_date

        # EOD block
        if not eod_block_active and ts.time() >= eod_t:
            eod_block_active = True
            _cancel_limit()
            if in_trade:
                _close_trade(o_arr[i] if not np.isnan(o_arr[i]) else close[i], ts, 'eod')
        if eod_block_active:
            continue

        # Anchor accumulation / lock (BEFORE fill checks — anchor bars never trade)
        # DB convention: bars are stamped at the START of the interval (e.g. the
        # bar at 09:55 covers 09:55 -> 10:00). NT8's convention is the opposite
        # (bar at 10:00 covers 09:55 -> 10:00). For the Python path we therefore
        # accumulate bars in [anchor_start, anchor_end) by hour, and lock when
        # we first see a bar at hour == anchor_end (the first post-anchor bar).
        if not anchor_locked:
            hour = ts.hour
            if anchor_start <= hour < anchor_end:
                if anchor_high == 0.0 and anchor_low == 0.0:
                    anchor_high = high[i]
                    anchor_low  = low[i]
                else:
                    if high[i] > anchor_high: anchor_high = high[i]
                    if low[i]  < anchor_low:  anchor_low  = low[i]
            elif hour == anchor_end and ts.minute == 0:
                # With right/right resampling, Python's 10:00 bar aggregates the same
                # 1m bars as NT's close-stamped 10:00 bar (09:56–10:00). NT includes
                # this bar in anchor accumulation before locking — so must Python.
                if anchor_high == 0.0 and anchor_low == 0.0:
                    anchor_high = high[i]
                    anchor_low  = low[i]
                else:
                    if high[i] > anchor_high: anchor_high = high[i]
                    if low[i]  < anchor_low:  anchor_low  = low[i]
                if anchor_high > 0 and anchor_low > 0 and anchor_high > anchor_low:
                    anchor_locked = True
                # IMPORTANT: do NOT continue — fall through to FVG/bias/entry.
            else:
                continue
            if not anchor_locked:
                continue

        # ---- Fill check on pending limit (intra-bar)
        if pending_side is not None and not in_trade:
            if pending_side == 'Long' and low[i] <= pending_entry:
                # filled
                fill_price = pending_entry
                trade_side = 'Long'
                trade_entry = fill_price
                trade_stop = pending_stop
                trade_target = pending_target
                trade_qty = pending_qty
                trade_fvg_key = pending_fvg_key
                trade_entry_ts = ts
                in_trade = True
                long_taken_today = True
                _cancel_limit()
                # Same-bar stop/target check (assume worst-case order: stop hits first if both in range)
                if low[i] <= trade_stop:
                    _close_trade(trade_stop, ts, 'stop')
                elif high[i] >= trade_target:
                    _close_trade(trade_target, ts, 'target')
            elif pending_side == 'Short' and high[i] >= pending_entry:
                fill_price = pending_entry
                trade_side = 'Short'
                trade_entry = fill_price
                trade_stop = pending_stop
                trade_target = pending_target
                trade_qty = pending_qty
                trade_fvg_key = pending_fvg_key
                trade_entry_ts = ts
                in_trade = True
                short_taken_today = True
                _cancel_limit()
                if high[i] >= trade_stop:
                    _close_trade(trade_stop, ts, 'stop')
                elif low[i] <= trade_target:
                    _close_trade(trade_target, ts, 'target')

        # ---- Stop/target on existing open trade
        if in_trade:
            if trade_side == 'Long':
                if low[i] <= trade_stop:
                    _close_trade(trade_stop, ts, 'stop')
                elif high[i] >= trade_target:
                    _close_trade(trade_target, ts, 'target')
            else:
                if high[i] >= trade_stop:
                    _close_trade(trade_stop, ts, 'stop')
                elif low[i] <= trade_target:
                    _close_trade(trade_target, ts, 'target')

        # ---- FVG detection (3-bar pattern; need i >= 2)
        if i >= 2:
            # Bull: low[i] > high[i-2] and gap >= min_fvg_pts
            if low[i] > high[i-2] and (low[i] - high[i-2]) >= min_fvg_pts:
                fvgs.append({
                    'bar_index':     i,
                    'direction':     'bull',
                    'top':           float(low[i]),
                    'bottom':        float(high[i-2]),
                    'candle1_low':   float(low[i-2]),
                    'candle1_high':  float(high[i-2]),
                    'candle3_close': float(close[i]),
                })
            elif high[i] < low[i-2] and (low[i-2] - high[i]) >= min_fvg_pts:
                fvgs.append({
                    'bar_index':     i,
                    'direction':     'bear',
                    'top':           float(low[i-2]),
                    'bottom':        float(high[i]),
                    'candle1_low':   float(low[i-2]),
                    'candle1_high':  float(high[i-2]),
                    'candle3_close': float(close[i]),
                })

        # ---- Bias engagement / flip (only when flat + no pending limit)
        # NT skips bias on the anchor-lock bar (10:00:00 exactly) — mirror that.
        if not in_trade and pending_side is None and not (ts.hour == anchor_end and ts.minute == 0):
            if close[i] > anchor_high:
                if bias != 'Long':
                    bias = 'Long'
                    breakout_trigger_bar = i
                    max_high_since_bias = high[i]
                    min_low_since_bias  = low[i]
                    bos_count = 0
            elif close[i] < anchor_low:
                if bias != 'Short':
                    bias = 'Short'
                    breakout_trigger_bar = i
                    max_high_since_bias = high[i]
                    min_low_since_bias  = low[i]
                    bos_count = 0

        # ---- BoS counting (after bias is set)
        if bias != 'None' and not bos_cancelled_for_day:
            if bias == 'Long':
                if high[i] > max_high_since_bias:
                    if i > breakout_trigger_bar:
                        bos_count += 1
                    max_high_since_bias = high[i]
            else:
                if low[i] < min_low_since_bias:
                    if i > breakout_trigger_bar:
                        bos_count += 1
                    min_low_since_bias = low[i]

            if bos_cancel_enabled and bos_count >= bos_cancel_count_p:
                bos_cancelled_for_day = True
                _cancel_limit()

        # ---- Entry / re-target
        if ts.time() >= entry_cutoff_t and cancel_pending_cutoff and pending_side is not None:
            _cancel_limit()

        if (
            bias != 'None'
            and not bos_cancelled_for_day
            and trades_today < max_trades_per_day
            and ts.time() < entry_cutoff_t
            and not in_trade
        ):
            if bias == 'Long' and not can_long:
                continue
            if bias == 'Short' and not can_short:
                continue
            # One trade per direction per session
            if bias == 'Long' and long_taken_today:
                continue
            if bias == 'Short' and short_taken_today:
                continue

            best = _select_best_fvg()
            if best is None:
                continue

            if pending_side is None:
                _place_limit(best)
            elif allow_retarget and best['bar_index'] != pending_fvg_key:
                _cancel_limit()
                _place_limit(best)

    return trades
