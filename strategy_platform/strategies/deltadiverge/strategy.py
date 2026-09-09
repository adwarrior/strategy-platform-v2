"""
DeltaDiverge — delta/price divergence at a prior-session extreme (MNQ, 1-hour bars).

From a video (NQ, hourly): a "seller exhaustion" long setup — an hourly candle
closes ABOVE the prior session's high while the candle itself is NEGATIVE
(close < open, price nominally went down) but its underlying tick-volume
DELTA was strongly POSITIVE. Aggressive buyers absorbed the down-move and
pushed the close back above the level -> read as exhaustion of the sellers
who printed the red candle. We add the mirror SHORT (video only covers longs).

BAR-LABELLING CONVENTION (read this before touching the logic)
----------------------------------------------------------------
NinjaTrader stamps bars by CLOSE time. pandas .resample(..., label=...)
can label by OPEN or CLOSE. This strategy uses **CLOSE-time labelling**
(label='right', closed='right') throughout, matching every other strategy
in this platform (magichour, mobobands, the loader's resample_ohlcv, etc.):
  - The bar stamped 14:00 covers price action from 13:00:00 to 13:59:59.999
    and its open/high/low/close reflect that hour; 14:00 is the hour it
    CLOSED. This matches NT's convention directly (bar.Time = close time).
  - A bar's signal (divergence condition) can only be evaluated once that
    bar's index timestamp has been reached, i.e. once it has printed. There
    is no ambiguity here because label='right' bars only ever "arrive"
    fully formed at simulation time (we are not walking sub-bar ticks).

NO LOOK-AHEAD
-------------
  - "Prior session high/low" is computed from bars belonging to the most
    recently COMPLETED session as of the signal bar; the signal bar itself
    is never included in that prior-session range, and the prior session is
    locked in only after its own session-end bar has closed.
  - Entry is on the NEXT bar's OPEN after the signal bar closes (never the
    signal bar's own close) — see _run_backtest_loop's `entry_px = nxt['open']`.

DELTA
-----
Hourly bar delta is loaded from emini.tick_data via
strategy_platform.strategies.deltadiverge.tick_delta.load_hourly_tick_delta,
which classifies each tick (price>=ask -> +volume, price<=bid -> -volume,
else ignored) and SUMS in SQL per UTC hour bucket, then shifts to ET-naive
(DST-aware) to align with the OHLCV bars. tick_delta buckets are OPEN-time
labelled (hour 14 = ticks from 14:00:00-14:59:59 ET); _attach_tick_delta
below shifts them +1h to align with this strategy's CLOSE-labelled bars
(an OPEN-labelled 14:00 tick bucket covers the same hour as a CLOSE-labelled
15:00 OHLCV bar).

COVERAGE GATE (tick_data is thin and INTERMITTENTLY so)
---------------------------------------------------------
tick_data volume is known to run ~42-50% of true traded volume, and some
hours have near-zero tick counts despite large true volume (near-empty, not
absent — a naive "any rows present" check would miss this). A near-empty
hour still produces a delta value that LOOKS like a real reading. We
compute coverage_ratio = tick_volume / historical_data_1m volume for the
same hour and gate on `min_coverage_ratio`. Bars failing the gate are
EXCLUDED from both signal generation and the rolling z-score baseline
window (not treated as delta=0, which would be a fabricated data point).
Outside the tick_data date range entirely (before 2024-09-12 or after
2026-04-17 for symbol MNQ) coverage_ratio is NaN and those bars are
likewise excluded and reported separately so the true tradeable range is
visible rather than silently dropped.

EXIT
----
Fixed take-profit / stop-loss in POINTS (tp_points, sl_points; MNQ = $2/pt,
converted via tick_value/tick_size, never hardcoded in dollars), plus a
flat-at-session-end fallback (eod_exit_time, ET).

See docs/deltadiverge_spec.md for the full spec.
"""

from __future__ import annotations

from datetime import time as time_t
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register
from strategy_platform.strategies.mobobands.strategy import _summarise, _bootstrap_trades
from strategy_platform.strategies.deltadiverge.tick_delta import load_hourly_tick_delta


WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


@register
class DeltaDiverge(BaseStrategy):
    """Hourly delta/price divergence at a prior-session extreme. MNQ, ET-naive."""

    name = "deltadiverge"
    bar_type            = '1m'   # load 1m from historical_data_1m, resample to 1H internally
    supported_bar_types = ['1m', 'time']

    # historical_data_1m is stored CENTRAL-TIME-naive. This strategy reasons about ET
    # session boundaries (RTH 09:30-16:00 default) and EOD flat time, so the loader
    # must shift CT->ET (+1h). db_timezone='ET' tells the dashboard/pipeline to pass
    # to_et=True to load_1m. See loader.load_1m docstring + magichour (same pattern).
    db_timezone = 'ET'

    default_params: Dict[str, Any] = {
        # ---- Trigger ----
        'trigger_type':        'zscore',   # 'net_delta' | 'zscore'
        'delta_threshold':     500.0,      # net_delta mode; video's NQ hourly value
        'zscore_threshold':    2.0,        # zscore mode
        'zscore_length':       20,         # rolling window of PRIOR hourly bars (coverage-gated)
        'min_coverage_ratio':  0.25,       # tick_volume / bar_volume; below this, bar is excluded
                                            # from BOTH signal + z-score baseline (see module docstring)

        # ---- Session definition (for prior-session high/low) ----
        'session_start':       '09:30',
        'session_end':         '16:00',

        # ---- Exit ----
        'tp_points':           40.0,
        'sl_points':            20.0,
        'eod_exit_time':        '16:00',

        # ---- Direction / instrument ----
        'direction':            'Both',    # 'Long' | 'Short' | 'Both'
        'symbol':                'MNQ',    # tick_data symbol (continuous or per-contract)
        'qty':                    1,
    }

    # MNQ micro Nasdaq defaults — overridden by dashboard when symbol changes
    tick_size     = 0.25
    tick_value    = 0.50      # $0.50 per 0.25-pt tick = $2 per point
    commission_rt = 1.02

    symbol  = 'MNQ'   # bar-data (historical_data_1m) DB symbol key
    db_host: Optional[str] = None

    # ------------------------------------------------------------------
    # param_grid / groups / display
    # ------------------------------------------------------------------

    @property
    def param_grid(self) -> Dict[str, Any]:
        return {
            'trigger_type':        ['net_delta', 'zscore'],
            'delta_threshold':     [250, 500, 750, 1000],
            'zscore_threshold':    (1.0, 3.0, 0.25),
            'zscore_length':       [10, 20, 30, 50],
            'min_coverage_ratio':  [0.15, 0.25, 0.35],

            'session_start':       ['09:30'],
            'session_end':         ['16:00'],

            'tp_points':           (10.0, 100.0, 10.0),
            'sl_points':           (10.0, 60.0, 10.0),
            'eod_exit_time':       ['15:30', '16:00', '16:55'],

            'direction':           ['Both', 'Long', 'Short'],
            'symbol':              ['MNQ', 'MNQ_H26', 'MNQ_M26'],
            'qty':                 (1, 5, 1),
        }

    param_conditional: Dict[str, Tuple[str, Any]] = {
        'delta_threshold':  ('trigger_type', 'net_delta'),
        'zscore_threshold': ('trigger_type', 'zscore'),
        'zscore_length':    ('trigger_type', 'zscore'),
    }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            "1. Trigger":       ['trigger_type', 'delta_threshold', 'zscore_threshold',
                                  'zscore_length', 'min_coverage_ratio'],
            "2. Session":       ['session_start', 'session_end'],
            "3. Exit":          ['tp_points', 'sl_points', 'eod_exit_time'],
            "4. Direction/Instrument": ['direction', 'symbol', 'qty'],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            'trigger_type':       'Trigger Type',
            'delta_threshold':    'Delta Threshold (net_delta mode)',
            'zscore_threshold':   'Z-Score Threshold',
            'zscore_length':      'Z-Score Lookback (bars)',
            'min_coverage_ratio': 'Min Tick Coverage Ratio',
            'session_start':      'Session Start (ET)',
            'session_end':        'Session End (ET)',
            'tp_points':          'Take Profit (points)',
            'sl_points':          'Stop Loss (points)',
            'eod_exit_time':      'EOD Exit Time (ET)',
            'direction':          'Direction',
            'symbol':             'Tick Symbol',
            'qty':                'Qty (fixed)',
        }

    @property
    def description(self) -> str:
        return ("Hourly delta/price divergence: close beyond the prior session's "
                "high/low on a candle whose own body ran the OTHER way, confirmed "
                "by tick-derived delta (net or z-scored) exceeding a threshold. "
                "Fixed TP/SL in points + flat-at-session-end fallback. Tick "
                "coverage is gated — thin hours are excluded from signal and "
                "z-score baseline alike.")

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        merged = {**self.default_params, **params}
        df_1h, coverage_stats = _prepare_hourly(data, merged, self.db_host)
        trades = _run_backtest_loop(
            df_1h, merged,
            self.tick_size, self.tick_value, self.commission_rt,
        )
        total_sessions = int(df_1h['close'].resample('D').last().count()) if len(df_1h) else 0
        stats     = _summarise(trades, total_sessions=total_sessions)
        bs        = _bootstrap_trades(trades, total_sessions=total_sessions)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return {
            **stats, **bs,
            'total_trades': stats['trades'],
            'trades': trades_df,
            'coverage_stats': coverage_stats,
        }

    def run_monte_carlo(
        self,
        prepared: pd.DataFrame,
        params: Dict[str, Any],
        n_sims: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        merged = {**self.default_params, **params}
        df_1h, _ = _prepare_hourly(prepared, merged, self.db_host)

        if len(df_1h) < 100:
            return {'mc_stability': 0.0, 'mc_sharpe_p5': float('nan'),
                    'mc_pnl_p5': float('nan'), 'mc_pnl_p50': float('nan')}

        groups = [(d, grp) for d, grp in df_1h.groupby(df_1h.index.date)]
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

def _parse_time(s: str) -> time_t:
    h, m = int(s.split(':')[0]), int(s.split(':')[1])
    return time_t(h, m)


def _ensure_1h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1m (or sub-1H) input up to 1-hour bars, CLOSE-stamped
    (label='right', closed='right') — see the module docstring's bar-labelling
    convention section. Already-1H-or-coarser input is returned as-is."""
    if len(df) < 3:
        return df
    diffs = df.index.to_series().diff().dropna()
    median_sec = diffs.median().total_seconds()
    if median_sec >= 3600:  # already 1H or coarser
        return df
    return df.resample('1h', label='right', closed='right').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()


def _attach_tick_delta(
    df_1h: pd.DataFrame,
    params: Dict[str, Any],
    db_host: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Join hourly tick-derived delta + coverage_ratio onto the CLOSE-labelled
    df_1h bars. Also returns coverage summary stats for reporting.

    tick_delta.load_hourly_tick_delta returns OPEN-labelled ET buckets (hour
    14 = ticks from 14:00-14:59:59 ET). df_1h is CLOSE-labelled (hour 15 =
    the bar covering 14:00-14:59:59). So an OPEN-labelled tick bucket at hour
    H joins to the CLOSE-labelled OHLCV bar at hour H+1 -- shift the tick
    index +1h before joining.
    """
    if df_1h.empty:
        df_1h = df_1h.copy()
        df_1h['delta'] = pd.Series(dtype=float)
        df_1h['tick_volume'] = pd.Series(dtype=float)
        df_1h['coverage_ratio'] = pd.Series(dtype=float)
        return df_1h, {'total_bars': 0, 'covered_bars': 0, 'coverage_pct': 0.0}

    symbol = str(params.get('symbol', 'MNQ'))
    start  = (df_1h.index[0] - pd.Timedelta(hours=2)).strftime('%Y-%m-%d')
    end    = (df_1h.index[-1] + pd.Timedelta(hours=2)).strftime('%Y-%m-%d')

    tick_df = load_hourly_tick_delta(symbol, start=start, end=end, host=db_host)

    out = df_1h.copy()
    if tick_df.empty:
        out['delta'] = np.nan
        out['tick_volume'] = np.nan
        out['coverage_ratio'] = np.nan
        return out, {
            'total_bars': len(out), 'covered_bars': 0, 'coverage_pct': 0.0,
            'note': f'no tick_data coverage for symbol={symbol} in requested window',
        }

    tick_df = tick_df.copy()
    tick_df.index = tick_df.index + pd.Timedelta(hours=1)  # open-label -> close-label alignment

    out = out.join(tick_df[['delta', 'tick_volume']], how='left')
    # coverage_ratio = tick-derived volume / bar (historical_data_1m-sourced) volume.
    # bar['volume'] here is the SUM of 1m bar volumes for this hour (from the
    # resample in _ensure_1h), i.e. the true contract volume for the hour.
    with np.errstate(invalid='ignore', divide='ignore'):
        out['coverage_ratio'] = out['tick_volume'] / out['volume'].replace(0, np.nan)

    total_bars = len(out)
    min_cov = float(params.get('min_coverage_ratio', 0.25))
    covered = out['coverage_ratio'] >= min_cov
    # NaN coverage (outside tick_data date range, or zero bar volume) never counts as covered.
    covered = covered.fillna(False)
    covered_bars = int(covered.sum())

    stats = {
        'total_bars':      total_bars,
        'covered_bars':    covered_bars,
        'coverage_pct':    float(covered_bars / total_bars) if total_bars else 0.0,
        'tick_data_start': str(tick_df.index.min()) if len(tick_df) else None,
        'tick_data_end':   str(tick_df.index.max()) if len(tick_df) else None,
        'bars_outside_tick_range': int(out['coverage_ratio'].isna().sum()),
    }
    return out, stats


def _prepare_hourly(
    data: pd.DataFrame,
    params: Dict[str, Any],
    db_host: Optional[str],
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Resample to 1H (close-stamped) and attach tick delta + coverage_ratio."""
    df_1h = _ensure_1h(data)
    df_1h, coverage_stats = _attach_tick_delta(df_1h, params, db_host)
    return df_1h, coverage_stats


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
    For each CLOSED hourly bar (the "signal bar"):
      1. Determine the prior COMPLETED session's high/low (RTH by default).
      2. LONG signal:  signal_bar.close > prior_session_high
                        AND signal_bar.close < signal_bar.open   (red candle)
                        AND delta trigger fires POSITIVE (>= threshold, gated by coverage)
      3. SHORT signal: signal_bar.close < prior_session_low
                        AND signal_bar.close > signal_bar.open   (green candle)
                        AND delta trigger fires NEGATIVE (<= -threshold, gated by coverage)
      4. Entry fills at the NEXT bar's OPEN (never the signal bar's own close —
         see module docstring's NO LOOK-AHEAD section).
      5. Fixed TP/SL in points, walked bar-by-bar (pessimistic: if a single bar's
         range touches both TP and SL, the stop is assumed to have hit first).
      6. Flat-at-session-end fallback: if neither TP nor SL is hit by
         eod_exit_time, exit at that bar's close.
    """
    if len(df) < 30:
        return []

    trigger_type    = str(params['trigger_type'])
    delta_threshold = float(params['delta_threshold'])
    z_threshold     = float(params['zscore_threshold'])
    z_length        = int(params['zscore_length'])
    min_coverage    = float(params['min_coverage_ratio'])
    session_start   = _parse_time(str(params['session_start']))
    session_end     = _parse_time(str(params['session_end']))
    tp_points       = float(params['tp_points'])
    sl_points       = float(params['sl_points'])
    eod_t           = _parse_time(str(params['eod_exit_time']))
    direction       = str(params['direction'])
    can_long        = direction in ('Both', 'Long')
    can_short       = direction in ('Both', 'Short')
    qty             = max(1, int(params['qty']))
    point_value     = tick_value / tick_size  # $ per 1 point of price movement

    if tp_points <= 0 or sl_points <= 0:
        return []

    # ---- Coverage gate: bars failing min_coverage are EXCLUDED from signal
    # generation AND from the z-score rolling baseline (never treated as
    # delta=0 — see module docstring's COVERAGE GATE section). This is a
    # boolean mask, computed once.
    covered = (df['coverage_ratio'] >= min_coverage) & df['coverage_ratio'].notna()

    # ---- Trigger series, computed only over covered bars.
    if trigger_type == 'zscore':
        delta_covered = df['delta'].where(covered)
        roll_mean = delta_covered.rolling(z_length, min_periods=z_length).mean()
        roll_std  = delta_covered.rolling(z_length, min_periods=z_length).std(ddof=0)
        with np.errstate(invalid='ignore', divide='ignore'):
            zscore = (delta_covered - roll_mean) / roll_std.replace(0, np.nan)
        trigger_val = zscore
        long_fires  = trigger_val >= z_threshold
        short_fires = trigger_val <= -z_threshold
    else:  # net_delta
        trigger_val = df['delta']
        long_fires  = trigger_val >= delta_threshold
        short_fires = trigger_val <= -delta_threshold

    long_fires  = (long_fires & covered).fillna(False)
    short_fires = (short_fires & covered).fillna(False)

    # ---- Session boundaries: RTH windows per calendar day (ET-naive index).
    # Prior session = the most recently COMPLETED session's [start,end) window
    # as of the signal bar's timestamp. A session's H/L is only available for
    # use once its own session_end bar has closed.
    dates = sorted(set(df.index.normalize()))
    session_ranges: Dict[pd.Timestamp, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for d in dates:
        s = d + pd.Timedelta(hours=session_start.hour, minutes=session_start.minute)
        e = d + pd.Timedelta(hours=session_end.hour, minutes=session_end.minute)
        session_ranges[d] = (s, e)

    # Precompute each session's high/low from bars strictly within [s, e).
    session_hi: Dict[pd.Timestamp, float] = {}
    session_lo: Dict[pd.Timestamp, float] = {}
    for d, (s, e) in session_ranges.items():
        sess_bars = df[(df.index > s) & (df.index <= e)]
        if len(sess_bars) == 0:
            continue
        session_hi[d] = float(sess_bars['high'].max())
        session_lo[d] = float(sess_bars['low'].min())

    sorted_session_days = sorted(session_hi.keys())

    def _prior_session_hilo(ts: pd.Timestamp) -> Optional[Tuple[float, float]]:
        """Most recent COMPLETED session as of ts (its session_end bar must have
        already closed, i.e. session_end <= ts's own bar-open time is not
        required here — we require session_end < ts, i.e. strictly before the
        signal bar's own close-timestamp, which by construction of df being
        walked chronologically already excludes same/future sessions)."""
        d = ts.normalize()
        # candidate days = session days strictly before ts's session END time
        for day in reversed(sorted_session_days):
            _, e = session_ranges[day]
            if e < ts:  # prior session must have fully closed before this bar
                return session_hi[day], session_lo[day]
        return None

    trades: List[Dict[str, Any]] = []
    n = len(df)
    idx = df.index

    for i in range(n - 1):  # need a next bar to enter on
        ts = idx[i]
        bar = df.iloc[i]

        prior = _prior_session_hilo(ts)
        if prior is None:
            continue
        prior_hi, prior_lo = prior

        bar_close = float(bar['close'])
        bar_open  = float(bar['open'])
        is_red    = bar_close < bar_open
        is_green  = bar_close > bar_open

        side: Optional[str] = None
        if can_long and is_red and bar_close > prior_hi and bool(long_fires.iloc[i]):
            side = 'Long'
        elif can_short and is_green and bar_close < prior_lo and bool(short_fires.iloc[i]):
            side = 'Short'

        if side is None:
            continue

        # ---- Entry fills at the NEXT bar's open (no look-ahead onto the
        # signal bar's own close).
        nxt_ts  = idx[i + 1]
        nxt_bar = df.iloc[i + 1]
        entry_px = float(nxt_bar['open'])
        entry_ts = nxt_ts

        if side == 'Long':
            tp_px = entry_px + tp_points
            sl_px = entry_px - sl_points
        else:
            tp_px = entry_px - tp_points
            sl_px = entry_px + sl_points

        # EOD flat time for the entry bar's own calendar day (session-end
        # fallback; if entry is already past this clock time same day, or the
        # trade crosses into the next day, the walk below still finds a bar to
        # exit on since it simply looks for the first bar at/after eod_t or
        # end of data).
        entry_day = entry_ts.normalize()
        eod_ts = entry_day + pd.Timedelta(hours=eod_t.hour, minutes=eod_t.minute)
        if eod_ts <= entry_ts:
            eod_ts = entry_ts + pd.Timedelta(hours=1)  # degenerate guard; walk resolves anyway

        walk = df.iloc[i + 2:]  # bars strictly after the entry bar

        exit_px: Optional[float] = None
        exit_ts: Optional[pd.Timestamp] = None
        exit_reason = 'eod'

        for wts, wbar in walk.iterrows():
            hi, lo = float(wbar['high']), float(wbar['low'])
            if side == 'Long':
                hit_sl = lo <= sl_px
                hit_tp = hi >= tp_px
            else:
                hit_sl = hi >= sl_px
                hit_tp = lo <= tp_px

            if hit_sl and hit_tp:
                exit_px, exit_ts, exit_reason = sl_px, wts, 'stop_ambiguous'
                break
            if hit_sl:
                exit_px, exit_ts, exit_reason = sl_px, wts, 'stop'
                break
            if hit_tp:
                exit_px, exit_ts, exit_reason = tp_px, wts, 'target'
                break
            if wts >= eod_ts:
                exit_px, exit_ts, exit_reason = float(wbar['close']), wts, 'eod'
                break

        if exit_px is None:
            if len(walk) == 0:
                continue
            exit_px = float(walk['close'].iloc[-1])
            exit_ts = walk.index[-1]
            exit_reason = 'eod'

        if side == 'Long':
            pnl_pts = exit_px - entry_px
        else:
            pnl_pts = entry_px - exit_px
        pnl_dollars = pnl_pts * point_value * qty - commission

        trades.append({
            'session_date': entry_day.date(),
            'day_of_week':  entry_day.day_name(),
            'side':         side,
            'signal_time':  ts,
            'entry_time':   entry_ts,
            'exit_time':    exit_ts,
            'entry_price':  entry_px,
            'exit_price':   exit_px,
            'stop':         sl_px,
            'target':       tp_px,
            'qty':          qty,
            'pnl':          pnl_dollars,
            'pnl_ticks':    pnl_pts / tick_size,
            'exit_reason':  exit_reason,
            'commission':   commission,
            'prior_session_high': prior_hi,
            'prior_session_low':  prior_lo,
            'signal_delta':       float(bar.get('delta', np.nan)),
            'signal_coverage':    float(bar.get('coverage_ratio', np.nan)),
            'trigger_type':       trigger_type,
        })

    return trades
