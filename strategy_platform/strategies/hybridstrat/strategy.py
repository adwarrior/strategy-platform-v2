"""
HybridStrat — Hurst-regime + autocorrelation-rotation trend continuation.

Source: YouTube "reverse engineered a prop trader making $33k/month" video
(anonymised "Trader Math" / TM, IQ Capital indicators). Spec (single source
of truth): `/home/ad/Scripts/strategies/hybridstrat_spec.md`. Sections 3, 5,
6, 7, 8 define this port; §4 gives the CONFIRMED (TradingView-calibrated)
indicator formulas, implemented verbatim below — do not re-derive them.

Concept (spec §1): trade a breakout of a short-term high, but only when the
Hurst proxy says the broader series is persistent (trending) while the
short-term autocorrelation says it is mildly anti-persistent (rotating).

Entry (spec §3), all true on the same bar's CLOSE:
  1. hurst[i]     >  hurst_threshold           (default 50)
  2. autocorr[i]  <= autocorr_threshold        (default -0.1)
  3. Close[i]     >  max(High[i-breakout_lookback .. i-1])   (close, not wick)
Fill at the NEXT bar's open (no look-ahead). One position at a time; signals
during an open trade are ignored, not queued.

Exit (spec §5): target and trail distance are both computed ONCE at entry
from the Wilder ATR value on the signal bar and then frozen. Target is
fixed/static (never trails). Stop trails off the highest high since entry
at that frozen distance and only ever rises. Same-bar ambiguity resolves
stop-first (conservative house convention, spec §5).

MNQ has no 5-minute table (`bar_type = '1m'`); 15m bars are resampled from
historical_data_1m internally (spec §2).

TIMEZONE HAZARD (spec §2 / feedback_db_1m_timezone_changes_mid_table):
historical_data_1m is Central before ~2026-04 and Eastern after. This
strategy has NO time-of-day logic in V1 (session_filter defaults to None,
see below) so the era split does not matter for correctness here — but do
NOT add an hour-based filter or a blanket to_et shift without handling the
split explicitly. `db_timezone` is left unset (None) on purpose: this
strategy must not request a blanket ET shift.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register
from strategy_platform.strategies.mobobands.strategy import _summarise, _bootstrap_trades


WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


@register
class HybridStrat(BaseStrategy):
    """Hurst-regime + autocorrelation-rotation breakout continuation, MNQ 15m."""

    name = "hybridstrat"
    bar_type            = '1m'    # load 1m from historical_data_1m, resample to 15m internally
    supported_bar_types = ['1m', '15m', 'time']

    # Deliberately NOT 'ET'. historical_data_1m changes timezone mid-table
    # (CT before ~2026-04, ET after) and this strategy has no session-hour
    # logic in V1 — a blanket to_et shift would be an unnecessary, unproven
    # risk. See module docstring / feedback_db_1m_timezone_changes_mid_table.
    db_timezone = None

    default_params: Dict[str, Any] = {
        'hurst_period':        20,
        'hurst_threshold':     50,
        'autocorr_lag':        5,
        'autocorr_window':     20,
        'autocorr_threshold':  -0.1,
        'breakout_lookback':   5,
        'atr_period':          20,
        'atr_target_mult':     5.0,
        'atr_stop_mult':       4.0,
        'direction':           'Long',
        'use_risk_sizing':     False,
        'max_risk_dollars':    500,
        'fixed_qty':           1,
        'session_filter':      None,   # None / 'RTH' — see module docstring re: TZ hazard
    }

    # MNQ micro Nasdaq defaults
    tick_size     = 0.25
    tick_value    = 0.50      # $0.50 per 0.25-pt tick = $2 per point
    commission_rt = 1.02      # per side per contract -> 2.04 round trip

    symbol  = 'MNQ'   # DB symbol key (NOT 'MNQ=F')
    db_host: Optional[str] = None

    # ------------------------------------------------------------------
    # param_grid / groups / display
    # ------------------------------------------------------------------

    @property
    def param_grid(self) -> Dict[str, Any]:
        return {
            # 1. Indicators (confirmed defaults bolded in spec §8; sweep for robustness only)
            'hurst_period':        [10, 15, 20, 30, 50],
            'hurst_threshold':     [45, 50, 55, 60],
            'autocorr_lag':        [3, 5, 8, 10],
            'autocorr_window':     [15, 20, 30],
            'autocorr_threshold':  [-0.3, -0.2, -0.1, 0.0],

            # 2. Breakout
            'breakout_lookback':   [3, 5, 8, 10],

            # 3. Exit
            'atr_period':          [14, 20, 30],
            'atr_target_mult':     [3, 4, 5, 6, 7],
            'atr_stop_mult':       [2, 3, 4, 5],

            # 4. Direction
            'direction':           ['Long', 'Short', 'Both'],

            # 5. Sizing
            'use_risk_sizing':     [True, False],
            'max_risk_dollars':    (100.0, 1000.0, 100.0),
            'fixed_qty':           (1, 5, 1),

            # 6. Session (default OFF — see TZ hazard in module docstring)
            'session_filter':      [None, 'RTH'],
        }

    param_conditional: Dict[str, Tuple[str, Any]] = {
        'max_risk_dollars': ('use_risk_sizing', True),
        'fixed_qty':         ('use_risk_sizing', False),
    }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            "1. Hurst":       ['hurst_period', 'hurst_threshold'],
            "2. Autocorr":    ['autocorr_lag', 'autocorr_window', 'autocorr_threshold'],
            "3. Breakout":    ['breakout_lookback'],
            "4. Exit":        ['atr_period', 'atr_target_mult', 'atr_stop_mult'],
            "5. Direction":   ['direction'],
            "6. Risk":        ['use_risk_sizing', 'max_risk_dollars', 'fixed_qty'],
            "7. Session":     ['session_filter'],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            'hurst_period':        'Hurst Lookback (bars)',
            'hurst_threshold':     'Hurst Threshold',
            'autocorr_lag':        'Autocorr Lag (bars)',
            'autocorr_window':     'Autocorr Window (bars)',
            'autocorr_threshold':  'Autocorr Threshold',
            'breakout_lookback':   'Breakout Lookback (bars, excl. current)',
            'atr_period':          'ATR Period (Wilder)',
            'atr_target_mult':     'Target (x ATR at entry, fixed)',
            'atr_stop_mult':       'Trail Distance (x ATR at entry, frozen)',
            'direction':           'Direction',
            'use_risk_sizing':     'Use Risk Sizing',
            'max_risk_dollars':    'Max Risk ($)',
            'fixed_qty':           'Qty (fixed)',
            'session_filter':      'Session Filter (None/RTH)',
        }

    @property
    def description(self) -> str:
        return ("Hurst-regime + autocorrelation-rotation breakout continuation: "
                "Hurst proxy > threshold (persistent) AND autocorrelation <= threshold "
                "(short-term anti-persistent) AND close breaks the prior N-bar high. "
                "Fixed ATR target, frozen-distance trailing stop, stop-first same-bar "
                "resolution. MNQ 15m (resampled from 1m).")

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        merged = {**self.default_params, **params}
        df = _ensure_15m(data)
        df = _compute_indicators(df, merged)
        trades = _run_backtest_loop(
            df, merged,
            self.tick_size, self.tick_value, self.commission_rt,
        )
        total_sessions = int(df['close'].resample('D').last().dropna().count())
        stats     = _summarise(trades, total_sessions=total_sessions)
        bs        = _bootstrap_trades(trades, total_sessions=total_sessions)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()

        point_value = self.tick_value / self.tick_size
        pnl_pts = [t['pnl_pts'] for t in trades] if trades else []
        wins_pts   = [p for p in pnl_pts if p > 0]
        losses_pts = [p for p in pnl_pts if p < 0]
        gross_profit_pts = sum(wins_pts)
        gross_loss_pts   = sum(losses_pts)
        profit_factor_pts = (gross_profit_pts / abs(gross_loss_pts)
                              if gross_loss_pts != 0 else float('inf'))
        expectancy_pts_mean   = float(np.mean(pnl_pts)) if pnl_pts else 0.0
        expectancy_pts_median = float(np.median(pnl_pts)) if pnl_pts else 0.0

        return {
            **stats, **bs,
            'total_trades':          stats['trades'],
            'trades':                trades_df,
            'point_value':           point_value,
            'profit_factor_pts':     profit_factor_pts,
            'expectancy_pts_mean':   expectancy_pts_mean,
            'expectancy_pts_median': expectancy_pts_median,
        }

    def run_monte_carlo(
        self,
        prepared: pd.DataFrame,
        params: Dict[str, Any],
        n_sims: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        merged = {**self.default_params, **params}
        df = _ensure_15m(prepared)
        df = _compute_indicators(df, merged)

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
# Data prep
# ---------------------------------------------------------------------------

def _ensure_15m(df: pd.DataFrame) -> pd.DataFrame:
    """Strategy logic is defined on 15m bars; resample 1m (or sub-15m) input up,
    close-stamped (label='right', closed='right') to match the platform's
    backtesting convention (spec §2 — distinct from the open-stamp convention
    used only for TradingView calibration). Larger-than-15m input is returned
    as-is."""
    if len(df) < 3:
        return df
    diffs = df.index.to_series().diff().dropna()
    median_sec = diffs.median().total_seconds()
    if median_sec > 950:   # larger than ~15m; nothing we can do, return as-is
        return df
    if median_sec < 850:   # sub-15m -> aggregate to 15m, close-stamped
        return df.resample('15min', label='right', closed='right').agg({
            'open': 'first', 'high': 'max', 'low': 'min',
            'close': 'last', 'volume': 'sum',
        }).dropna()
    return df


# ---------------------------------------------------------------------------
# Indicators — CONFIRMED formulas (spec §4), implemented verbatim
# ---------------------------------------------------------------------------

def _compute_indicators(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """Adds columns: tr, hurst, autocorr, highest_high_prior (breakout ref),
    lowest_low_prior (short-mirror breakout ref), atr_wilder.

    No look-ahead: every column at row i uses only rows <= i.
    """
    df = df.copy()
    high, low, close, open_ = df['high'], df['low'], df['close'], df['open']
    prev_close = close.shift(1)

    # True Range (shared by Hurst's simple-mean ATR and the Wilder exit ATR).
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    df['tr'] = tr

    # ---- 1) Hurst proxy (spec §4.1) ----
    # hurst[i] = 100 * log(range_n / mean_TR_n) / log(n)
    # range_n  = max(High, n) - min(Low, n), inclusive of bar i.
    # mean_TR_n = SIMPLE arithmetic mean of the last n True Ranges (NOT Wilder/RMA).
    n_h = int(params['hurst_period'])
    range_n   = high.rolling(n_h).max() - low.rolling(n_h).min()
    mean_tr_n = tr.rolling(n_h).mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = range_n / mean_tr_n
        hurst = 100.0 * np.log(ratio) / np.log(n_h)
    hurst = pd.Series(hurst, index=df.index)
    hurst[(range_n <= 0) | (mean_tr_n <= 0) | mean_tr_n.isna()] = np.nan
    df['hurst'] = hurst

    # ---- 2) Autocorrelation (spec §4.2) — NOT Pearson ----
    # bodysign[t] = sign(Close[t] - Open[t])
    # autocorr[i] = mean over the last `window` bars of (bodysign[t] * bodysign[t-lag])
    lag    = int(params['autocorr_lag'])
    window = int(params['autocorr_window'])
    bodysign = np.sign(close - open_)
    agree = bodysign * bodysign.shift(lag)
    df['autocorr'] = agree.rolling(window).mean()

    # ---- 3) Breakout reference levels (spec §3 cond 3 / §8 short mirror) ----
    # EXCLUDES the current bar: shift(1) before the rolling window.
    lb = int(params['breakout_lookback'])
    df['highest_high_prior'] = high.shift(1).rolling(lb).max()
    df['lowest_low_prior']   = low.shift(1).rolling(lb).min()

    # ---- 4) Wilder ATR for exit sizing (spec §5) ----
    atr_p = int(params['atr_period'])
    df['atr_wilder'] = tr.ewm(alpha=1.0 / atr_p, adjust=False, min_periods=atr_p).mean()

    return df


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
    """Explicit bar-by-bar walk (clarity over cleverness, per spec quality bar).

    Entry: all three conditions true on bar i's close -> fill at bar i+1's open.
    Exit: target fixed at entry; stop trails from highest/lowest since entry at
    a distance frozen at entry. Same-bar ambiguity resolves stop-first.
    One position at a time; signals while in a trade are ignored, not queued.
    """
    if len(df) < 50:
        return []

    hurst_th    = float(params['hurst_threshold'])
    ac_th       = float(params['autocorr_threshold'])
    atr_target_mult = float(params['atr_target_mult'])
    atr_stop_mult   = float(params['atr_stop_mult'])
    direction   = str(params['direction'])
    can_long    = direction in ('Long', 'Both')
    can_short   = direction in ('Short', 'Both')
    use_risk    = bool(params['use_risk_sizing'])
    max_risk    = float(params['max_risk_dollars'])
    qty_fixed   = max(1, int(params['fixed_qty']))
    point_value = tick_value / tick_size

    session_filter = params.get('session_filter')
    # session_filter='RTH' is NOT implemented in V1 (kept default None per
    # spec §"DATA/PLATFORM CONTRACT"). historical_data_1m's mid-table
    # timezone change (CT->ET ~2026-04) makes any hour-based filter unsafe
    # without explicit era handling — do not add one here without that.
    if session_filter not in (None, 'None'):
        return []

    hurst    = df['hurst'].to_numpy()
    autocorr = df['autocorr'].to_numpy()
    hh_prior = df['highest_high_prior'].to_numpy()
    ll_prior = df['lowest_low_prior'].to_numpy()
    atr      = df['atr_wilder'].to_numpy()
    o = df['open'].to_numpy()
    h = df['high'].to_numpy()
    l = df['low'].to_numpy()
    c = df['close'].to_numpy()
    idx = df.index

    n = len(df)
    trades: List[Dict[str, Any]] = []

    in_pos = False
    side = ''
    qty = 0
    entry_px = entry_ts = None
    entry_i = -1
    target_px = 0.0
    stop_px = 0.0
    trail_dist = 0.0
    extreme_since_entry = 0.0
    atr_at_entry = 0.0
    hurst_at_entry = 0.0
    autocorr_at_entry = 0.0

    for i in range(n):
        if in_pos:
            hi, lo = h[i], l[i]
            if side == 'Long':
                extreme_since_entry = max(extreme_since_entry, hi)
                new_stop = extreme_since_entry - trail_dist
                stop_px = max(stop_px, new_stop)
                hit_stop   = lo <= stop_px
                hit_target = hi >= target_px
            else:  # Short
                extreme_since_entry = min(extreme_since_entry, lo)
                new_stop = extreme_since_entry + trail_dist
                stop_px = min(stop_px, new_stop)
                hit_stop   = hi >= stop_px
                hit_target = lo <= target_px

            exit_px = None
            exit_reason = None
            if hit_stop:            # stop-first on ambiguous bars (spec §5)
                exit_px, exit_reason = stop_px, 'trail_stop'
            elif hit_target:
                exit_px, exit_reason = target_px, 'target'

            is_last_bar = (i == n - 1)
            if exit_px is None and is_last_bar:
                exit_px, exit_reason = c[i], 'eod'

            if exit_px is not None:
                if side == 'Long':
                    pnl_pts = exit_px - entry_px
                else:
                    pnl_pts = entry_px - exit_px
                pnl_dollars = pnl_pts * point_value * qty - commission * qty

                trades.append({
                    'session_date':      pd.Timestamp(entry_ts).date(),
                    'day_of_week':       pd.Timestamp(entry_ts).day_name(),
                    'side':              side,
                    'entry_time':        entry_ts,
                    'exit_time':         idx[i],
                    'entry_price':       float(entry_px),
                    'exit_price':        float(exit_px),
                    'target':            float(target_px),
                    'stop':              float(stop_px),
                    'qty':               qty,
                    'pnl':               float(pnl_dollars),
                    'pnl_pts':           float(pnl_pts),
                    'pnl_ticks':         float(pnl_pts / tick_size),
                    'exit_reason':       exit_reason,
                    'commission':        commission * qty,
                    'bars_held':         i - entry_i,
                    'atr_at_entry':      float(atr_at_entry),
                    'hurst_at_entry':    float(hurst_at_entry),
                    'autocorr_at_entry': float(autocorr_at_entry),
                })
                in_pos = False
            continue  # in-position bar consumed; no same-bar re-entry

        # ---- Not in a position: evaluate entry on this bar's close ----
        if i >= n - 1:
            break  # no next bar to fill on

        if np.isnan(hurst[i]) or np.isnan(autocorr[i]) or np.isnan(atr[i]):
            continue

        regime_ok = hurst[i] > hurst_th and autocorr[i] <= ac_th

        long_signal  = can_long  and regime_ok and not np.isnan(hh_prior[i]) and c[i] > hh_prior[i]
        short_signal = can_short and regime_ok and not np.isnan(ll_prior[i]) and c[i] < ll_prior[i]

        if not long_signal and not short_signal:
            continue

        sig_side = 'Long' if long_signal else 'Short'
        atr_sig  = atr[i]
        if atr_sig <= 0 or np.isnan(atr_sig):
            continue

        trail = atr_stop_mult * atr_sig
        risk_per_ctr = trail * point_value

        if use_risk:
            q = int(max_risk // risk_per_ctr) if risk_per_ctr > 0 else 0
            if q < 1:
                continue
        else:
            q = qty_fixed

        fill_px = o[i + 1]   # next bar's open — no look-ahead

        in_pos = True
        side = sig_side
        qty = q
        entry_px = fill_px
        entry_ts = idx[i + 1]
        entry_i = i + 1
        atr_at_entry = atr_sig
        hurst_at_entry = hurst[i]
        autocorr_at_entry = autocorr[i]
        trail_dist = trail

        if side == 'Long':
            target_px = entry_px + atr_target_mult * atr_sig
            stop_px   = entry_px - trail_dist
            extreme_since_entry = entry_px
        else:
            target_px = entry_px - atr_target_mult * atr_sig
            stop_px   = entry_px + trail_dist
            extreme_since_entry = entry_px

    return trades
