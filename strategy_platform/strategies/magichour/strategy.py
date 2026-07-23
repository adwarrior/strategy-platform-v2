"""
MagicHour — premarket reference-hour reversion.

Fork of OrbFade60 (`orb_fade_60/strategy.py`). Reuses the session /
fade-to-midpoint / first-touch-exit scaffolding. See
`/home/ad/Scripts/strategies/magichour_spec.md` (single source of truth)
for the full spec this port follows.

Hour naming: the param is `range_close_hour` = the CLOSE-time label (matches NT's
RangeCloseHour) = the hour the range completes on. Internally the range opens at
open_hour = range_close_hour - 1. Default range_close_hour=8 = the 07:00-08:00 range.

Lifecycle per day (NY ET, close-stamped 5m bars), for range_close_hour = C, open = C-1:
  1. Range = high/low of the 5m bars whose OPEN time falls in [(C-1):00, C:00),
     i.e. close-stamped bars (C-1):05 ... C:00.
     The range locks at the close of the C:00-stamped bar.
  2. Watch subsequent 5m bars (first eligible stamp = C:05)
     for the first bar whose CLOSE is outside the range (breakout confirmation).
  3. Wait for the first SUBSEQUENT bar whose CLOSE is back inside the range
     (close-back-inside entry model, v1's only entry model).
  4. On that bar, place a limit order at the broken boundary:
       Break UP   -> SHORT limit at range_high
       Break DOWN -> LONG  limit at range_low
     Target = range midpoint. Stop = PRRatio (reward/pr_ratio) or RangeR
     (stop_r_multiple x range size), per `stop_mode`.
  5. Walk bars after entry; first-touch resolves exit (pessimistic stop-first
     on ambiguous bars), else time-stop after outcome_minutes, else EOD exit.

Filters: z_zone_max (breakout-extension cap), min_range_pts, and
max_breakout_delay_min (early-breakout-only), plus OrbFade60's large-range
percentile filter (default OFF here).

One trade per reference-hour setup per day (the first valid close-back-inside).
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
class MagicHour(BaseStrategy):
    """Premarket reference-hour mean reversion. 5m bars, ET timezone-naive."""

    name = "magichour"
    bar_type            = '1m'    # load 1m from historical_data_1m, resample to 5m internally
    supported_bar_types = ['1m', '5m', 'time']

    # historical_data_1m is stored CENTRAL-TIME-naive. This strategy reasons about ET
    # clock hours (reference_hour, EOD), so the loader must shift CT->ET (+1h). The
    # pipeline/dashboard read this attribute and pass to_et=True to load_1m.
    # See loader.load_1m docstring + memory feedback_db_1m_is_central_time.
    db_timezone = 'ET'

    default_params: Dict[str, Any] = {
        # CLOSE-time label of the reference hour (matches NinjaTrader RangeCloseHour):
        # the hour the range COMPLETES on. 8 = the 07:00-08:00 range (best hour).
        # Internally the range opens at (range_close_hour - 1).
        'range_close_hour':        8,
        'entry_model':             'CloseBackInside',
        'z_zone_max':              4,
        'min_range_pts':           10.0,
        # Video's "first 20 min" early-breakout filter did NOT help on MNQ H1-2025
        # (it discarded ~65% of trades and cut net P&L $3.7k->$1.3k). Default disabled.
        'max_breakout_delay_min':  0,
        'stop_mode':               'PRRatio',
        'pr_ratio':                1.5,
        'stop_r_multiple':         0.5,
        'outcome_minutes':         90,
        'use_size_filter':         False,
        'size_lookback_days':      90,
        'size_percentile':         0.67,
        'direction':               'Both',
        'eod_exit_time':           '16:00',
        'use_risk_sizing':         False,
        'max_risk':                300,
        'qty':                     1,
    }

    # MNQ micro Nasdaq defaults — overridden by dashboard when symbol changes
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
            # 1. Reference hour / entry model
            'range_close_hour':        [6, 7, 8, 9],   # close-time; 8 = best hour (07:00-08:00 range)
            'entry_model':             ['CloseBackInside'],

            # 2. Breakout filters
            'z_zone_max':              [1, 2, 3, 4, 5],
            'min_range_pts':           (5, 30, 5),
            'max_breakout_delay_min':  [0, 20, 40, 60],

            # 3. Stop / target
            'stop_mode':               ['PRRatio', 'RangeR'],
            'pr_ratio':                (1.0, 2.0, 0.25),
            'stop_r_multiple':         (0.25, 1.0, 0.25),
            'outcome_minutes':         [60, 90, 120],

            # 4. Size filter (secondary, default OFF)
            'use_size_filter':         [True, False],
            'size_lookback_days':      [60, 90, 120, 180],
            'size_percentile':         (0.50, 0.85, 0.05),

            # 5. Direction
            'direction':               ['Both', 'Long Only', 'Short Only'],

            # 6. EOD
            'eod_exit_time':           ['15:30', '16:00', '16:30', '16:55'],

            # 7. Risk sizing
            'use_risk_sizing':         [True, False],
            'max_risk':                (50.0, 500.0, 50.0),
            'qty':                     (1, 5, 1),
        }

    param_conditional: Dict[str, Tuple[str, Any]] = {
        'pr_ratio':           ('stop_mode', 'PRRatio'),
        'stop_r_multiple':    ('stop_mode', 'RangeR'),
        'size_lookback_days': ('use_size_filter', True),
        'size_percentile':    ('use_size_filter', True),
        'max_risk':           ('use_risk_sizing', True),
        'qty':                ('use_risk_sizing', False),
    }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            "1. Reference Hour": ['range_close_hour', 'entry_model'],
            "2. Breakout Filters": ['z_zone_max', 'min_range_pts', 'max_breakout_delay_min'],
            "3. Stop/Target":    ['stop_mode', 'pr_ratio', 'stop_r_multiple', 'outcome_minutes'],
            "4. Size Filter":    ['use_size_filter', 'size_lookback_days', 'size_percentile'],
            "5. Direction":      ['direction'],
            "6. EOD":            ['eod_exit_time'],
            "7. Risk":           ['use_risk_sizing', 'max_risk', 'qty'],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            'range_close_hour':       'Range Close Hour (ET)',
            'entry_model':            'Entry Model',
            'z_zone_max':             'Max Breakout Zone (1-5)',
            'min_range_pts':          'Min Range Size (pts)',
            'max_breakout_delay_min': 'Max Breakout Delay (min, 0=off)',
            'stop_mode':              'Stop Mode',
            'pr_ratio':               'Profit:Risk Ratio',
            'stop_r_multiple':        'Stop (× Range size)',
            'outcome_minutes':        'Time Stop (min after entry)',
            'use_size_filter':        'Use Large-Range Filter',
            'size_lookback_days':     'Size Lookback (days)',
            'size_percentile':        'Min Range Percentile',
            'direction':              'Direction',
            'eod_exit_time':          'EOD Exit Time',
            'use_risk_sizing':        'Use Risk Sizing',
            'max_risk':               'Max Risk ($)',
            'qty':                    'Qty (fixed)',
        }

    @property
    def description(self) -> str:
        return ("Premarket reference-hour reversion: lock the reference-hour range, "
                "wait for a close-back-inside after a breakout, fade to the midpoint. "
                "Stop = PRRatio (reward/pr_ratio) or Range-R. Z-zone / min-range / "
                "early-breakout filters.")

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
    """Strategy logic is defined on 5m bars; resample 1m (or sub-5m) input up,
    close-stamped (label='right', closed='right') to match NT convention.
    Larger-than-5m input is returned as-is (nothing finer we can do)."""
    if len(df) < 3:
        return df
    diffs = df.index.to_series().diff().dropna()
    median_sec = diffs.median().total_seconds()
    if median_sec > 330:  # larger than ~5m bars; nothing we can do, return as-is
        return df
    if median_sec < 290:  # sub-5m -> aggregate to 5m, close-stamped
        return df.resample('5min', label='right', closed='right').agg({
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
    """Per-day: build reference-hour range -> watch for breakout-confirmation
    close -> wait for close-back-inside -> fade to midpoint -> first-touch exit."""
    if len(df) < 100:
        return []

    # ---- Params
    # range_close_hour is the CLOSE-time label (NT convention); the range OPENS one hour
    # earlier. ref_hour below is the internal OPEN hour used for all anchor arithmetic.
    range_close_hour = int(params.get('range_close_hour', params.get('reference_hour', 8)))
    ref_hour     = (range_close_hour - 1) % 24
    entry_model  = str(params['entry_model'])
    z_zone_max   = int(params['z_zone_max'])
    min_range    = float(params['min_range_pts'])
    max_delay    = int(params['max_breakout_delay_min'])
    stop_mode    = str(params['stop_mode'])
    pr_ratio     = float(params['pr_ratio'])
    stop_r       = float(params['stop_r_multiple'])
    outcome_m    = int(params['outcome_minutes'])
    use_filter   = bool(params['use_size_filter'])
    lookback     = int(params['size_lookback_days'])
    pct          = float(params['size_percentile'])
    direction    = str(params['direction'])
    can_long     = direction in ('Both', 'Long Only')
    can_short    = direction in ('Both', 'Short Only')
    eod_t        = _parse_time(str(params['eod_exit_time']))
    use_risk     = bool(params['use_risk_sizing'])
    max_risk     = float(params['max_risk'])
    qty_fixed    = max(1, int(params['qty']))
    point_value  = tick_value / tick_size  # $ per 1 point of price movement

    if entry_model != 'CloseBackInside':
        return []  # only entry model implemented in v1

    # --- Anchor-relative session model (works for ALL reference hours, incl.
    # evening hours 18:00-23:00 whose trades cross midnight) ---
    #
    # Each session is anchored at date + ref_hour:00 (ET-naive). We slice by
    # ABSOLUTE timestamp, never time-of-day, so a range starting 22:00 and a fade
    # resolving 01:30 the next calendar day is one contiguous session.
    #
    # Range window: close-stamps in (anchor, anchor+60min], i.e. bars whose OPEN
    # falls in [ref_hour:00, ref_hour+1:00). Range locks at the anchor+60min bar.
    # Breakout-watch window: (anchor+60min, anchor + watch_span_min].
    #   watch_span_min = 60 (range) + WATCH_WINDOW_MIN so there is always enough
    #   room after the range for a breakout + close-back-inside + outcome walk,
    #   regardless of the daytime eod_exit_time (which is now only a *floor* cap
    #   for daytime hours, applied as an absolute clamp below).
    WATCH_WINDOW_MIN = 8 * 60  # up to 8h after range close to find breakout+entry+exit

    range_span = pd.Timedelta(minutes=60)
    watch_span = pd.Timedelta(minutes=60 + WATCH_WINDOW_MIN)

    # Build one anchor per calendar date on which ref_hour bars exist.
    # (Grouping by date to find candidate anchors is fine — the anchor timestamp
    # itself, not the date group, drives all slicing.)
    anchors: List[pd.Timestamp] = []
    for d in sorted(set(df.index.normalize())):
        anchor = d + pd.Timedelta(hours=ref_hour)
        anchors.append(anchor)

    # ---- Pre-compute per-anchor range sizes for the rolling percentile filter
    range_sizes_by_anchor: Dict[pd.Timestamp, float] = {}
    for anchor in anchors:
        rb = df[(df.index > anchor) & (df.index <= anchor + range_span)]
        if len(rb) > 0:
            range_sizes_by_anchor[anchor] = float(rb['high'].max() - rb['low'].min())

    sorted_anchors = sorted(range_sizes_by_anchor)

    def _passes_size_filter(anchor: pd.Timestamp) -> bool:
        if not use_filter:
            return True
        idx = sorted_anchors.index(anchor) if anchor in range_sizes_by_anchor else -1
        past = sorted_anchors[max(0, idx - lookback):idx]
        if len(past) < 20:
            return True
        past_sizes = np.array([range_sizes_by_anchor[a] for a in past])
        threshold = float(np.quantile(past_sizes, pct))
        return range_sizes_by_anchor[anchor] >= threshold

    # ---- Per-anchor loop
    trades: List[Dict[str, Any]] = []

    for anchor in anchors:
        d = anchor.date()
        range_start_ts = anchor
        range_end_ts   = anchor + range_span
        rb = df[(df.index > range_start_ts) & (df.index <= range_end_ts)]
        if len(rb) == 0:
            continue
        range_high = float(rb['high'].max())
        range_low  = float(rb['low'].min())
        range_size = range_high - range_low
        if range_size <= 0:
            continue
        if range_size < min_range:
            continue

        if not _passes_size_filter(anchor):
            continue

        range_mid = (range_high + range_low) / 2.0
        range_close_ts = rb.index[-1]  # timestamp of the (ref_hour+1):00-stamped bar

        # Breakout-watch bars: absolute window (range_end, anchor + watch_span].
        session_end_ts = anchor + watch_span
        post = df[(df.index > range_end_ts) & (df.index <= session_end_ts)]
        if len(post) == 0:
            continue

        # ---- Step 1: first breakout-confirmation bar (close outside range)
        up_break  = post['close'] > range_high
        dn_break  = post['close'] < range_low
        any_break = up_break | dn_break
        if not any_break.any():
            continue
        break_idx = int(np.argmax(any_break.values))
        break_bar = post.iloc[break_idx]
        break_ts  = post.index[break_idx]

        is_up  = bool(up_break.iloc[break_idx])
        is_dn  = bool(dn_break.iloc[break_idx])
        if is_up and is_dn:
            continue  # ambiguous, shouldn't happen on a single close value

        # Early-breakout timing filter
        if max_delay > 0:
            delay_min = (break_ts - range_close_ts).total_seconds() / 60.0
            if delay_min > max_delay:
                continue
        else:
            delay_min = (break_ts - range_close_ts).total_seconds() / 60.0

        # Z-zone / breakout-extension filter
        if is_up:
            break_extreme = float(break_bar['high'])
            extension_pct = (break_extreme - range_high) / range_size * 100.0
        else:
            break_extreme = float(break_bar['low'])
            extension_pct = (range_low - break_extreme) / range_size * 100.0
        extension_pct = max(0.0, extension_pct)
        z_zone = _zone_of(extension_pct)
        if z_zone > z_zone_max:
            continue

        if is_up and not can_short:
            continue
        if is_dn and not can_long:
            continue

        # ---- Step 2: first subsequent close-back-inside bar
        after_break = post.iloc[break_idx + 1:]
        if len(after_break) == 0:
            continue
        back_inside = (after_break['close'] >= range_low) & (after_break['close'] <= range_high)
        if not back_inside.any():
            continue
        entry_sig_idx = int(np.argmax(back_inside.values))
        entry_sig_ts  = after_break.index[entry_sig_idx]

        # ---- Step 3: entry = limit order at the broken boundary
        if is_up:
            side       = 'Short'
            entry_px   = range_high
            breakout_side = 'Up'
        else:
            side       = 'Long'
            entry_px   = range_low
            breakout_side = 'Down'

        target_px = range_mid
        reward    = abs(entry_px - target_px)
        if reward <= 0:
            continue

        if stop_mode == 'RangeR':
            stop_dist = stop_r * range_size
        else:  # PRRatio
            stop_dist = reward / pr_ratio if pr_ratio > 0 else 0.0
        if stop_dist <= 0:
            continue

        if side == 'Short':
            stop_px = entry_px + stop_dist
        else:
            stop_px = entry_px - stop_dist

        # Position sizing
        if use_risk:
            risk_per_ctr = stop_dist * point_value
            qty = int(max_risk / risk_per_ctr) if risk_per_ctr > 0 else 0
            if qty < 1:
                continue
        else:
            qty = qty_fixed

        # Walk bars from the close-back-inside bar forward for first-touch
        # resolution (the limit is assumed filled on/after this bar's close,
        # since the trigger to broadcast the order is that bar's close).
        # Time-stop is ABSOLUTE (entry_time + outcome_minutes), so it crosses
        # midnight cleanly. It is also clamped to the session watch window.
        time_stop_ts = entry_sig_ts + pd.Timedelta(minutes=outcome_m)
        if time_stop_ts > session_end_ts:
            time_stop_ts = session_end_ts

        walk = after_break.iloc[entry_sig_idx + 1:]
        walk = walk[walk.index <= time_stop_ts]

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
            'entry_time':   entry_sig_ts,
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
            'or_high':      range_high,
            'or_low':       range_low,
            'or_size_pts':  range_size,
            'range_close_hour':   range_close_hour,   # close-time label (NT convention)
            'reference_hour':     ref_hour,           # internal open hour (= close - 1)
            'range_high':         range_high,
            'range_low':          range_low,
            'range_mid':          range_mid,
            'range_size_pts':     range_size,
            'breakout_side':      breakout_side,
            'extension_pct':      extension_pct,
            'z_zone':             z_zone,
            'breakout_delay_min': delay_min,
            'entry_model':        entry_model,
        })

    return trades


def _zone_of(extension_pct: float) -> int:
    """Map breakout-extension % to a Z-zone: 1=0-25%, 2=25-50%, 3=50-75%,
    4=75-100%, 5=Beyond (>100%)."""
    if extension_pct <= 25.0:
        return 1
    if extension_pct <= 50.0:
        return 2
    if extension_pct <= 75.0:
        return 3
    if extension_pct <= 100.0:
        return 4
    return 5
