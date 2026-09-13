"""
ATRScalper — ATR-band limit fade (long only).

See `/home/ad/Scripts/strategies/atrscalper_spec.md` (single source of truth)
for the full spec this implementation follows.

Signal timeframe is 5m: `ATR(14)` (Wilder) on 5m bars, lower band =
`close[t-1] - atr_mult * ATR[t-1]` (both shifted one bar — see spec §2).
A resting limit BUY sits at the band, re-priced every 5m bar. Fills, stop,
and target are resolved on 1-MINUTE SUB-BARS nested inside each 5m bar
(spec §4) — never on the 5m bar alone — because the entry can plausibly
touch and continue through the 1.5-ATR stop inside the same 5m bar; a
5m-only model would silently convert that stop-out into a win
(memory: `feedback_no_bar_reconstruction_for_scalps`).

Fill-ordering rules per 1m sub-bar (pessimistic, non-negotiable):
  1. Entry before exit: if flat and low <= band, fill at exactly band.
  2. Stop before target: if both low <= stop and high >= target in the
     same sub-bar, resolve as STOP, never target.
  3. Same-sub-bar entry + stop: if the sub-bar that triggers entry also
     has low <= stop, the trade is a full stop-out on its entry bar.

Long only, one position at a time, no re-entry on the same bar after an
exit. EOD flatten at `eod_exit_time` (default 15:55 ET).
"""

from __future__ import annotations

from datetime import time as time_t
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register
from strategy_platform.strategies.mobobands.strategy import _summarise, _bootstrap_trades


@register
class ATRScalper(BaseStrategy):
    """ATR-band limit fade, long only. Signal on 5m bars, fills resolved on 1m sub-bars."""

    name = "atrscalper"
    bar_type            = '1m'    # load 1m from historical_data_1m, resample to 5m internally
    supported_bar_types = ['1m', '5m', 'time']

    # historical_data_1m is stored CENTRAL-TIME-naive. This strategy reasons about ET
    # clock hours (session window, EOD), so the loader must shift CT->ET (+1h). The
    # pipeline/dashboard read this attribute and pass to_et=True to load_1m.
    # See loader.load_1m docstring + memory feedback_db_1m_is_central_time.
    db_timezone = 'ET'

    default_params: Dict[str, Any] = {
        'atr_period':          14,
        'atr_mult':            3.1,
        'tp_atr_mult':         2.0,
        'sl_atr_mult':         1.5,
        'skip_opening_bars':   1,
        'session_start':       '09:30',
        'session_end':         '16:00',
        'eod_exit_time':       '15:55',
        'qty':                 1,
        'use_risk_sizing':     False,
        'max_risk':            100.0,
    }

    # MNQ micro Nasdaq defaults — overridden by dashboard when symbol changes
    tick_size     = 0.25
    tick_value    = 0.50      # $0.50 per 0.25-pt tick = $2 per point
    commission_rt = 1.02      # INSTRUMENT_META['MNQ']['commission']

    symbol  = 'MNQ'   # DB symbol key (NOT 'MNQ=F' — that returns 0 rows from historical_data_1m)
    db_host: Optional[str] = None

    # ------------------------------------------------------------------
    # param_grid / groups / display
    # ------------------------------------------------------------------

    @property
    def param_grid(self) -> Dict[str, Any]:
        return {
            # 1. ATR / Bands
            'atr_period':        [7, 10, 14, 20, 28],
            'atr_mult':          (2.0, 4.5, 0.1),

            # 2. Exits
            'tp_atr_mult':       (1.0, 3.5, 0.25),
            'sl_atr_mult':       (0.75, 3.0, 0.25),

            # 3. Session
            'skip_opening_bars': [0, 1, 2, 3],

            # 4. Sizing
            'use_risk_sizing':   [True, False],
            'max_risk':          (50.0, 500.0, 50.0),
            'qty':               (1, 5, 1),
        }

    param_conditional: Dict[str, Tuple[str, Any]] = {
        'max_risk': ('use_risk_sizing', True),
        'qty':      ('use_risk_sizing', False),
    }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            "1. ATR / Bands": ['atr_period', 'atr_mult'],
            "2. Exits":       ['tp_atr_mult', 'sl_atr_mult'],
            "3. Session":     ['skip_opening_bars', 'session_start', 'session_end', 'eod_exit_time'],
            "4. Sizing":      ['use_risk_sizing', 'max_risk', 'qty'],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            'atr_period':        'ATR Period (Wilder)',
            'atr_mult':          'Band ATR Multiple',
            'tp_atr_mult':       'Target (× ATR)',
            'sl_atr_mult':       'Stop (× ATR)',
            'skip_opening_bars': 'Skip Opening Bars (5m, count)',
            'session_start':     'Session Start (ET)',
            'session_end':       'Session End (ET)',
            'eod_exit_time':     'EOD Exit Time (ET)',
            'use_risk_sizing':   'Use Risk Sizing',
            'max_risk':          'Max Risk ($)',
            'qty':               'Qty (fixed)',
        }

    @property
    def description(self) -> str:
        return ("ATR-band limit fade, long only: resting limit buy at "
                "close[t-1] - atr_mult*ATR[t-1] (5m, Wilder ATR), fixed "
                "target/stop at entry. Fills resolved on 1m sub-bars, "
                "pessimistic stop-before-target ordering.")

    # ------------------------------------------------------------------
    # Backtest / MC
    # ------------------------------------------------------------------

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        merged = {**self.default_params, **params}
        df_1m, df_5m = _ensure_1m_and_5m(data)
        trades = _run_backtest_loop(
            df_1m, df_5m, merged,
            self.tick_size, self.tick_value, self.commission_rt,
        )
        total_sessions = int(df_5m['close'].resample('D').last().count())
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
        df_1m, df_5m = _ensure_1m_and_5m(prepared)
        merged = {**self.default_params, **params}

        groups_1m = dict(list(df_1m.groupby(df_1m.index.date)))
        groups_5m = dict(list(df_5m.groupby(df_5m.index.date)))
        common_dates = sorted(set(groups_1m) & set(groups_5m))
        rng = np.random.default_rng(seed)
        n   = len(common_dates)

        net_pnls: list = []
        sharpes:  list = []

        for _ in range(n_sims):
            order         = rng.permutation(n)
            shuffled_dates = [common_dates[i] for i in order]
            shuffled_1m = pd.concat([groups_1m[d] for d in shuffled_dates])
            shuffled_5m = pd.concat([groups_5m[d] for d in shuffled_dates])
            trades = _run_backtest_loop(
                shuffled_1m, shuffled_5m, merged,
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

def _ensure_1m_and_5m(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (1m frame, 5m frame). Signal layer needs 5m (close-stamped,
    label='right', closed='right' per spec §3/§4); fill resolution needs the
    native 1m frame. If input is already coarser than 1m (e.g. straight 5m
    data with no 1m available), fall back to using the 5m frame for both —
    fills then degrade to 5m-only resolution."""
    if len(df) < 3:
        return df, df

    diffs = df.index.to_series().diff().dropna()
    median_sec = diffs.median().total_seconds()

    if median_sec <= 90:  # native 1m (or finer) input
        df_1m = df
        df_5m = df.resample('5min', label='right', closed='right').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }).dropna(subset=['open'])
        return df_1m, df_5m

    # No 1m available — degrade gracefully to 5m-only for both.
    return df, df


def _parse_time(s: str) -> time_t:
    h, m = int(s.split(':')[0]), int(s.split(':')[1])
    return time_t(h, m)


def _wilder_atr(df_5m: pd.DataFrame, period: int) -> pd.Series:
    """Wilder-smoothed ATR on 5m bars."""
    high, low, close = df_5m['high'], df_5m['low'], df_5m['close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    # Wilder smoothing == EMA with alpha = 1/period (adjust=False)
    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    return atr


# ---------------------------------------------------------------------------
# Single-pass backtest loop
# ---------------------------------------------------------------------------

def _run_backtest_loop(
    df_1m:      pd.DataFrame,
    df_5m:      pd.DataFrame,
    params:     Dict[str, Any],
    tick_size:  float,
    tick_value: float,
    commission: float,
) -> List[Dict[str, Any]]:
    """Compute ATR band on 5m bars (shifted one bar per spec §2), then walk
    5m bars in order; while flat and inside the session, resolve the resting
    limit buy + subsequent stop/target/EOD exit on the 1m sub-bars nested in
    each 5m bar (spec §4), pessimistic stop-before-target ordering."""
    if len(df_5m) < 50:
        return []

    atr_period   = int(params['atr_period'])
    atr_mult     = float(params['atr_mult'])
    tp_mult      = float(params['tp_atr_mult'])
    sl_mult      = float(params['sl_atr_mult'])
    skip_bars    = int(params['skip_opening_bars'])
    session_start = _parse_time(str(params['session_start']))
    session_end   = _parse_time(str(params['session_end']))
    eod_t         = _parse_time(str(params['eod_exit_time']))
    use_risk      = bool(params['use_risk_sizing'])
    max_risk      = float(params['max_risk'])
    qty_fixed     = max(1, int(params['qty']))
    point_value   = tick_value / tick_size  # $ per 1 point of price movement

    df = df_5m.copy()
    df['atr'] = _wilder_atr(df, atr_period)

    # Static-anchor rule (spec §2): band/target/stop distances all use ATR[t-1],
    # and the band is measured from close[t-1]. Shift explicitly.
    df['atr_prev']   = df['atr'].shift(1)
    df['close_prev'] = df['close'].shift(1)
    df['band_raw']   = df['close_prev'] - atr_mult * df['atr_prev']
    # Tick rounding (spec §7): band rounds DOWN (conservative — harder to fill).
    df['band'] = np.floor(df['band_raw'] / tick_size) * tick_size

    # Count 5m bars since session open, per session date, for skip_opening_bars.
    session_date = df.index.normalize()
    in_session = np.array([
        session_start <= ts.time() <= session_end for ts in df.index
    ])
    bar_num = np.zeros(len(df), dtype=int)
    for d in pd.unique(session_date[in_session]):
        mask = (session_date == d) & in_session
        idxs = np.where(mask)[0]
        bar_num[idxs] = np.arange(len(idxs))

    trades: List[Dict[str, Any]] = []

    in_position   = False
    entry_px = stop_px = target_px = atr_at_entry = band_at_entry = None
    entry_ts: Optional[pd.Timestamp] = None
    entry_qty = 0
    mae_ticks = mfe_ticks = 0.0
    blocked_this_bar = False  # no re-entry on the same 5m bar after an exit

    n = len(df)
    for i in range(n):
        ts_5m   = df.index[i]
        row     = df.iloc[i]
        band    = row['band']
        atr_p   = row['atr_prev']

        if not in_session[i]:
            blocked_this_bar = False
            continue

        bar_start = ts_5m - pd.Timedelta(minutes=5)
        sub = df_1m[(df_1m.index > bar_start) & (df_1m.index <= ts_5m)]
        if len(sub) == 0:
            blocked_this_bar = False
            continue

        can_enter_this_bar = (
            not blocked_this_bar
            and bar_num[i] >= skip_bars
            and pd.notna(band)
            and pd.notna(atr_p)
        )
        blocked_this_bar = False  # reset; only set True below if an exit happens this bar

        for sub_ts, sub_bar in sub.iterrows():
            lo, hi = float(sub_bar['low']), float(sub_bar['high'])

            if not in_position:
                if can_enter_this_bar and lo <= band:
                    in_position   = True
                    entry_px      = float(band)
                    entry_ts      = sub_ts
                    atr_at_entry  = float(atr_p)
                    band_at_entry = float(band)
                    raw_target    = entry_px + tp_mult * atr_p
                    raw_stop      = entry_px - sl_mult * atr_p
                    # Tick rounding (spec §7): target rounds UP (harder to reach),
                    # stop rounds UP toward entry (easier to hit).
                    target_px = np.ceil(raw_target / tick_size) * tick_size
                    stop_px   = np.ceil(raw_stop   / tick_size) * tick_size
                    mae_ticks = 0.0
                    mfe_ticks = 0.0

                    if use_risk:
                        risk_per_ctr = (entry_px - stop_px) * point_value
                        entry_qty = int(max_risk / risk_per_ctr) if risk_per_ctr > 0 else 0
                        if entry_qty < 1:
                            in_position = False
                            continue
                    else:
                        entry_qty = qty_fixed

                    can_enter_this_bar = False  # one entry per 5m bar

                    # Same sub-bar entry + stop (spec §4 rule 3): full stop-out
                    # on the entry bar. Check immediately, using this same sub_bar.
                    if lo <= stop_px:
                        exit_px, exit_ts, exit_reason = float(stop_px), sub_ts, 'stop'
                        mae_ticks = (entry_px - lo) / tick_size
                        mfe_ticks = max(0.0, (hi - entry_px) / tick_size)
                        trades.append(_make_trade(
                            entry_ts, exit_ts, entry_px, exit_px, stop_px, target_px,
                            atr_at_entry, band_at_entry, exit_reason, entry_qty,
                            mae_ticks, mfe_ticks, point_value, commission, tick_size,
                        ))
                        in_position = False
                        blocked_this_bar = True
                        continue
                    # entered, not stopped on this sub-bar — fall through to next sub-bar
                continue

            # In position: update MAE/MFE, then check stop-before-target.
            mae_ticks = max(mae_ticks, (entry_px - lo) / tick_size)
            mfe_ticks = max(mfe_ticks, (hi - entry_px) / tick_size)

            hit_stop   = lo <= stop_px
            hit_target = hi >= target_px

            if hit_stop:
                exit_px, exit_ts, exit_reason = float(stop_px), sub_ts, 'stop'
            elif hit_target:
                exit_px, exit_ts, exit_reason = float(target_px), sub_ts, 'target'
            else:
                continue

            trades.append(_make_trade(
                entry_ts, exit_ts, entry_px, exit_px, stop_px, target_px,
                atr_at_entry, band_at_entry, exit_reason, entry_qty,
                mae_ticks, mfe_ticks, point_value, commission, tick_size,
            ))
            in_position = False
            blocked_this_bar = True

        # EOD flatten: if still in position and this 5m bar's close-time is at/after eod_t.
        if in_position and ts_5m.time() >= eod_t:
            exit_px = float(sub['close'].iloc[-1])
            exit_ts = sub.index[-1]
            lo_last, hi_last = float(sub['low'].iloc[-1]), float(sub['high'].iloc[-1])
            mae_ticks = max(mae_ticks, (entry_px - lo_last) / tick_size)
            mfe_ticks = max(mfe_ticks, (hi_last - entry_px) / tick_size)
            trades.append(_make_trade(
                entry_ts, exit_ts, entry_px, exit_px, stop_px, target_px,
                atr_at_entry, band_at_entry, 'eod', entry_qty,
                mae_ticks, mfe_ticks, point_value, commission, tick_size,
            ))
            in_position = False
            blocked_this_bar = True

    return trades


def _make_trade(
    entry_ts, exit_ts, entry_px, exit_px, stop_px, target_px,
    atr_at_entry, band_px, exit_reason, qty,
    mae_ticks, mfe_ticks, point_value, commission, tick_size,
) -> Dict[str, Any]:
    pnl_pts     = exit_px - entry_px
    pnl_dollars = pnl_pts * point_value * qty - commission * qty
    bars_held   = int(np.ceil((exit_ts - entry_ts).total_seconds() / 60.0)) if exit_ts > entry_ts else 0
    return {
        'session_date': pd.Timestamp(entry_ts).date(),
        'day_of_week':  pd.Timestamp(entry_ts).day_name(),
        'side':         'Long',
        'entry_time':   entry_ts,
        'exit_time':    exit_ts,
        'entry_px':     entry_px,
        'exit_px':      exit_px,
        'entry_price':  entry_px,
        'exit_price':   exit_px,
        'stop_px':      stop_px,
        'target_px':    target_px,
        'stop':         stop_px,
        'target':       target_px,
        'atr_at_entry': atr_at_entry,
        'band_px':      band_px,
        'exit_reason':  exit_reason,
        'bars_held':    bars_held,
        'mae_ticks':    mae_ticks,
        'mfe_ticks':    mfe_ticks,
        'qty':          qty,
        'pnl':          pnl_dollars,
        'pnl_ticks':    pnl_pts / tick_size,
        'commission':   commission * qty,
    }
