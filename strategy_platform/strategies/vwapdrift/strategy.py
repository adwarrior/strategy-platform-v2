"""
VWAPDrift — "Drift VWAP Pullback" (YouTube quant interview port).

Spec (single source of truth): /home/ad/Scripts/strategies/vwapdrift_spec.md
NT8 indicator: Scripts/indicators/VWAPDrift.cs

Session VWAP anchored 09:30 ET, computed from 5m typical-price bars (parity with
the NT indicator). Trend evaluated every 15 minutes on closed 5m bars and latched:
  LONG : close > VWAP, VWAP > VWAP 15min ago, close vs close 1h ago >= +0.1%
  SHORT: mirror with <= -0.1%
Trigger: first counter-colour 5m candle while a trend is latched (red candle in a
long trend / green in a short trend) -> market fill at the NEXT 5m bar open.
Brackets: stop 80 pts both sides, target 40 (long) / 50 (short), resolved on 1m
sub-bars, stop-first when a single 1m bar touches both (house rule).
Guardrails: no entries before anchor+60min or after 15:30 ET, flat 15:55 ET,
max 4 trades/day, stop after 2 losing trades/day, one position at a time.
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
class VwapDrift(BaseStrategy):
    """Drift VWAP Pullback: 15-min-drift-filtered session-VWAP pullback entries. ET-naive."""

    name = "vwapdrift"
    bar_type            = '1m'    # 1m from historical_data_1m: 5m signals + 1m bracket resolution
    supported_bar_types = ['1m']

    # historical_data_1m is CENTRAL-TIME-naive; all session logic here is ET clock
    # time, so the loader must shift CT->ET (see memory feedback_db_1m_is_central_time).
    db_timezone = 'ET'

    default_params: Dict[str, Any] = {
        'anchor_time':          '09:30',
        'warmup_minutes':       60,
        'slope_lookback_min':   15,
        'hour_return_pct':      0.1,
        # Max distance from VWAP the pullback candle must reach, in points.
        # 0 = OFF = the transcript's literal mechanical rule ("doesn't matter how
        # close it is to the VWOP", line 346) — any counter-colour candle triggers.
        # >0 tests the video's NARRATIVE reading ("pullback towards the VWAP"),
        # which conflicts with the mechanics; the source is genuinely ambiguous.
        # Measured on the candle's closest approach: low in a long, high in a short.
        'max_vwap_distance':    0.0,
        # 'Points' = fixed points; 'ATR' = multiple of 14-bar 5m ATR. Prefer ATR:
        # MNQ ran ~7,000 in 2020 and ~25,000 in 2026, so a fixed point distance
        # is a much tighter filter early in the sample than late.
        'vwap_distance_mode':   'ATR',
        'stop_points':          80.0,
        'target_long_points':   40.0,
        'target_short_points':  50.0,
        'max_trades_day':       4,
        'max_losses_day':       2,
        'losses_consecutive':   False,   # transcript ambiguity: False = 2 total losses/day
        'first_pullback_only':  False,   # spec §3: default re-arms after every exit
        'entry_cutoff':         '15:30',
        'eod_flat':             '15:55',
        'direction':            'Both',
        'use_risk_sizing':      False,
        'max_risk':             300,
        'qty':                  1,
    }

    # MNQ defaults — dashboard/pipeline override from INSTRUMENT_META per symbol
    tick_size     = 0.25
    tick_value    = 0.50      # $2 per point
    commission_rt = 1.02

    symbol  = 'MNQ'   # DB symbol key (NOT 'MNQ=F')
    db_host: Optional[str] = None

    # ------------------------------------------------------------------
    # param_grid / groups / display
    # ------------------------------------------------------------------

    @property
    def param_grid(self) -> Dict[str, Any]:
        return {
            # 1. VWAP / trend
            'anchor_time':          ['09:30'],
            'warmup_minutes':       [30, 60, 90],
            'slope_lookback_min':   [10, 15, 30],
            'hour_return_pct':      (0.05, 0.30, 0.05),
            'max_vwap_distance':    (0.0, 3.0, 0.25),
            'vwap_distance_mode':   ['ATR', 'Points'],

            # 2. Brackets
            'stop_points':          (40.0, 120.0, 20.0),
            'target_long_points':   (20.0, 80.0, 10.0),
            'target_short_points':  (20.0, 80.0, 10.0),

            # 3. Guardrails
            'max_trades_day':       (1, 6, 1),
            'max_losses_day':       (1, 4, 1),
            'losses_consecutive':   [True, False],
            'first_pullback_only':  [True, False],
            'entry_cutoff':         ['14:30', '15:00', '15:30'],
            'eod_flat':             ['15:55'],

            # 4. Direction
            'direction':            ['Both', 'Long Only', 'Short Only'],

            # 5. Risk
            'use_risk_sizing':      [True, False],
            'max_risk':             (50.0, 500.0, 50.0),
            'qty':                  (1, 5, 1),
        }

    param_conditional: Dict[str, Tuple[str, Any]] = {
        'max_risk': ('use_risk_sizing', True),
        'qty':      ('use_risk_sizing', False),
    }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            "1. VWAP / Trend": ['anchor_time', 'warmup_minutes', 'slope_lookback_min',
                                'hour_return_pct'],
            "2. Brackets":     ['stop_points', 'target_long_points', 'target_short_points'],
            "3. Guardrails":   ['max_trades_day', 'max_losses_day', 'losses_consecutive',
                                'first_pullback_only', 'entry_cutoff', 'eod_flat'],
            "4. Direction":    ['direction'],
            "5. Risk":         ['use_risk_sizing', 'max_risk', 'qty'],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            'anchor_time':          'VWAP Anchor (ET)',
            'warmup_minutes':       'Warm-up (min, no trades)',
            'slope_lookback_min':   'Drift Lookback (min)',
            'hour_return_pct':      '1h Momentum Threshold (%)',
            'stop_points':          'Stop (pts)',
            'target_long_points':   'Target Long (pts)',
            'target_short_points':  'Target Short (pts)',
            'max_trades_day':       'Max Trades / Day',
            'max_losses_day':       'Max Losses / Day',
            'losses_consecutive':   'Losses Must Be Consecutive',
            'first_pullback_only':  'First Pullback Only',
            'entry_cutoff':         'No New Entries After (ET)',
            'eod_flat':             'Flatten At (ET)',
            'direction':            'Direction',
            'use_risk_sizing':      'Use Risk Sizing',
            'max_risk':             'Max Risk ($)',
            'qty':                  'Qty (fixed)',
        }

    @property
    def description(self) -> str:
        return ("Drift VWAP Pullback: session VWAP anchored 09:30 ET; every 15 min latch a "
                "trend (price vs VWAP, VWAP drift, 1h momentum >= 0.1%); enter on the first "
                "counter-colour 5m pullback candle at next bar open; fixed 80pt stop / "
                "40-50pt target; prop-firm guardrails.")

    # ------------------------------------------------------------------
    # Backtest / MC
    # ------------------------------------------------------------------

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        merged = {**self.default_params, **params}
        trades = _run_backtest_loop(
            data, merged, self.tick_size, self.tick_value, self.commission_rt,
        )
        total_sessions = int(data['close'].resample('D').last().count())
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
        # NOTE: every trade here is self-contained within its day, so day-shuffle MC
        # returns near-identical totals by construction (same caveat as MagicHour —
        # prefer the trade-level bootstrap in run_backtest for confidence intervals).
        merged = {**self.default_params, **params}
        groups = [(d, grp) for d, grp in prepared.groupby(prepared.index.date)]
        rng = np.random.default_rng(seed)
        n   = len(groups)

        net_pnls: list = []
        sharpes:  list = []
        for _ in range(n_sims):
            order       = rng.permutation(n)
            shuffled_df = pd.concat([groups[i][1] for i in order])
            trades = _run_backtest_loop(
                shuffled_df, merged, self.tick_size, self.tick_value, self.commission_rt,
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


def _resample_5m(df1m: pd.DataFrame) -> pd.DataFrame:
    """1m -> 5m, close-stamped (label/closed='right') to match NT convention."""
    return df1m.resample('5min', label='right', closed='right').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna()


# ---------------------------------------------------------------------------
# Single-pass backtest loop
# ---------------------------------------------------------------------------

def _run_backtest_loop(
    df1m:       pd.DataFrame,
    params:     Dict[str, Any],
    tick_size:  float,
    tick_value: float,
    commission: float,
) -> List[Dict[str, Any]]:
    if len(df1m) < 300:
        return []

    # ---- Params
    anchor_t     = _parse_time(str(params['anchor_time']))
    warmup_min   = int(params['warmup_minutes'])
    slope_min    = int(params['slope_lookback_min'])
    slope_bars   = max(1, slope_min // 5)
    thresh_pct   = float(params['hour_return_pct']) / 100.0
    max_dist     = float(params.get('max_vwap_distance', 0.0))
    dist_mode    = str(params.get('vwap_distance_mode', 'ATR'))
    stop_pts     = float(params['stop_points'])
    tp_long      = float(params['target_long_points'])
    tp_short     = float(params['target_short_points'])
    max_trades   = int(params['max_trades_day'])
    max_losses   = int(params['max_losses_day'])
    consec       = bool(params['losses_consecutive'])
    first_only   = bool(params['first_pullback_only'])
    cutoff_t     = _parse_time(str(params['entry_cutoff']))
    eod_t        = _parse_time(str(params['eod_flat']))
    direction    = str(params['direction'])
    can_long     = direction in ('Both', 'Long Only')
    can_short    = direction in ('Both', 'Short Only')
    use_risk     = bool(params['use_risk_sizing'])
    max_risk     = float(params['max_risk'])
    qty_fixed    = max(1, int(params['qty']))
    point_value  = tick_value / tick_size

    df5 = _resample_5m(df1m)

    # ---- Session VWAP (5m typical price x volume, anchored per day; spec §1)
    # Bars are close-stamped: a bar belongs to the session if its stamp is strictly
    # after the anchor (its data starts AT the anchor).
    tod   = df5.index.time
    dates = df5.index.normalize()
    in_sess = tod > anchor_t

    tp = (df5['high'] + df5['low'] + df5['close']) / 3.0
    pv = (tp * df5['volume']).where(in_sess, 0.0)
    v  = df5['volume'].where(in_sess, 0.0)
    grp = pd.Series(dates, index=df5.index)
    cum_pv = pv.groupby(grp).cumsum()
    cum_v  = v.groupby(grp).cumsum()
    vwap = (cum_pv / cum_v.replace(0.0, np.nan)).where(in_sess)

    # 15-min drift: VWAP now vs slope_bars ago, same day only
    vwap_prev = vwap.groupby(grp).shift(slope_bars)

    # 1h momentum: close now vs close 60 minutes ago (bar-count on contiguous 5m
    # bars; same-day guard keeps overnight gaps from leaking across sessions)
    close_1h = df5['close'].groupby(grp).shift(12)
    hour_ret = df5['close'] / close_1h - 1.0

    # 14-bar ATR on 5m bars, for the ATR-relative VWAP-distance filter. Shifted
    # by 1 so the trigger bar's own range can't set its own threshold.
    prev_close = df5['close'].shift(1)
    tr = pd.concat([
        df5['high'] - df5['low'],
        (df5['high'] - prev_close).abs(),
        (df5['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr5 = tr.rolling(14, min_periods=14).mean().shift(1)

    o5 = df5['open'].values
    c5 = df5['close'].values
    idx5 = df5.index

    # 1m arrays for bracket resolution
    idx1 = df1m.index
    o1 = df1m['open'].values
    h1 = df1m['high'].values
    l1 = df1m['low'].values
    c1 = df1m['close'].values

    vwap_v      = vwap.values
    vwap_prev_v = vwap_prev.values
    hour_ret_v  = hour_ret.values
    atr_v       = atr5.values
    h5          = df5['high'].values
    l5          = df5['low'].values

    # Per-day 5m bar positions
    day_positions: Dict[Any, List[int]] = {}
    for i, d in enumerate(dates):
        day_positions.setdefault(d, []).append(i)

    trades: List[Dict[str, Any]] = []

    for d, positions in day_positions.items():
        anchor_dt = d + pd.Timedelta(hours=anchor_t.hour, minutes=anchor_t.minute)
        eval_start = anchor_dt + pd.Timedelta(minutes=warmup_min)
        eod_dt     = d + pd.Timedelta(hours=eod_t.hour, minutes=eod_t.minute)

        trend = 0            # 0 flat, +1 long, -1 short (latched between evals)
        armed = False        # first_pullback_only bookkeeping
        trades_today = 0
        losses_today = 0
        stopped_out_day = False
        in_pos_until: Optional[pd.Timestamp] = None

        for i in positions:
            ts = idx5[i]
            if ts < eval_start:
                continue
            if ts.time() > eod_t:
                break

            # ---- 15-minute trend evaluation (latched; spec §2)
            if ts.minute % 15 == 0:
                vw, vwp, hr = vwap_v[i], vwap_prev_v[i], hour_ret_v[i]
                new_trend = 0
                if np.isfinite(vw) and np.isfinite(vwp) and np.isfinite(hr):
                    if c5[i] > vw and vw > vwp and hr >= thresh_pct:
                        new_trend = 1
                    elif c5[i] < vw and vw < vwp and hr <= -thresh_pct:
                        new_trend = -1
                if new_trend != trend:
                    armed = new_trend != 0
                trend = new_trend

            # The eval_start bar (e.g. 10:30) may evaluate the trend but never trigger —
            # the no-trade window covers it (video: no trades 09:30-10:30).
            if ts <= eval_start:
                continue

            # ---- Skip bars while a position is open
            if in_pos_until is not None and ts < in_pos_until:
                continue
            in_pos_until = None

            if stopped_out_day or trades_today >= max_trades:
                continue
            if ts.time() >= cutoff_t:
                continue
            if trend == 0:
                continue
            if first_only and not armed:
                continue

            # ---- Trigger: counter-colour candle (spec §3)
            side = None
            if trend == 1 and can_long and c5[i] < o5[i]:
                side = 'Long'
            elif trend == -1 and can_short and c5[i] > o5[i]:
                side = 'Short'
            if side is None:
                continue

            # ---- Proximity filter: did the pullback actually REACH the VWAP?
            # The transcript's mechanics disclaim distance (line 346) but its
            # narrative says "pullback towards the VWAP" — this tests that reading.
            # Distance = the candle's closest approach to VWAP (low in a long,
            # high in a short); 0 if the candle traded through VWAP entirely.
            vwap_dist = np.nan
            if max_dist > 0:
                vw = vwap_v[i]
                if not np.isfinite(vw):
                    continue
                if side == 'Long':
                    dist = max(0.0, l5[i] - vw)
                else:
                    dist = max(0.0, vw - h5[i])
                if dist_mode == 'ATR':
                    a = atr_v[i]
                    if not np.isfinite(a) or a <= 0:
                        continue
                    limit = max_dist * a
                else:
                    limit = max_dist
                if dist > limit:
                    continue
                vwap_dist = float(dist)

            # ---- Fill at next 5m bar open == first 1m bar strictly after ts
            e_start = idx1.searchsorted(ts, side='right')
            if e_start >= len(idx1) or idx1[e_start] > eod_dt or idx1[e_start].normalize() != d:
                continue
            entry_px = float(o1[e_start])
            entry_ts = idx1[e_start]

            if side == 'Long':
                stop_px, target_px = entry_px - stop_pts, entry_px + tp_long
            else:
                stop_px, target_px = entry_px + stop_pts, entry_px - tp_short

            if use_risk:
                risk_per_ctr = stop_pts * point_value
                qty = int(max_risk / risk_per_ctr) if risk_per_ctr > 0 else 0
                if qty < 1:
                    continue
            else:
                qty = qty_fixed

            # ---- Walk 1m bars to first touch; stop-first on ambiguity (spec §4)
            e_end = idx1.searchsorted(eod_dt, side='right')
            exit_px: Optional[float] = None
            exit_ts: Optional[pd.Timestamp] = None
            exit_reason = 'eod'
            for j in range(e_start, e_end):
                if side == 'Long':
                    hit_stop, hit_tp = l1[j] <= stop_px, h1[j] >= target_px
                else:
                    hit_stop, hit_tp = h1[j] >= stop_px, l1[j] <= target_px
                if hit_stop:
                    exit_px, exit_ts = stop_px, idx1[j]
                    exit_reason = 'stop_ambiguous' if hit_tp else 'stop'
                    break
                if hit_tp:
                    exit_px, exit_ts, exit_reason = target_px, idx1[j], 'target'
                    break
            if exit_px is None:
                if e_end <= e_start:
                    continue
                exit_px, exit_ts = float(c1[e_end - 1]), idx1[e_end - 1]
                exit_reason = 'eod'

            pnl_pts = (exit_px - entry_px) if side == 'Long' else (entry_px - exit_px)
            pnl_dollars = pnl_pts * point_value * qty - commission * qty

            trades.append({
                'session_date': pd.Timestamp(d).date(),
                'day_of_week':  pd.Timestamp(d).day_name(),
                'side':         side,
                'entry_time':   entry_ts,
                'exit_time':    exit_ts,
                'entry_price':  entry_px,
                'exit_price':   exit_px,
                'stop':         stop_px,
                'target':       target_px,
                'qty':          qty,
                'pnl':          pnl_dollars,
                'pnl_ticks':    pnl_pts / tick_size,
                'exit_reason':  exit_reason,
                'commission':   commission * qty,
                'vwap_at_entry': float(vwap_v[i]) if np.isfinite(vwap_v[i]) else np.nan,
                'vwap_dist':     vwap_dist,
                'hour_ret_pct':  float(hour_ret_v[i] * 100.0) if np.isfinite(hour_ret_v[i]) else np.nan,
            })

            trades_today += 1
            armed = False
            in_pos_until = exit_ts  # 5m bars stamped before the exit are skipped

            if pnl_pts < 0:
                losses_today += 1
            elif consec:
                losses_today = 0
            if losses_today >= max_losses:
                stopped_out_day = True

    return trades
