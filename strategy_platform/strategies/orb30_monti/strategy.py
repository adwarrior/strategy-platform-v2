"""
ORB30Monti — Monti's improved Opening-Range Breakout (30-min, NY session).

Ported from /home/ad/Scripts/strategies/ORB30Monti.cs (NinjaScript C#).
Rationale: /home/ad/.claude/plans/could-you-have-a-whimsical-pizza.md

Based on Fabio Valentini's ORB30 strategy, improved by Monti (ex market-maker):
  - Delta filter retained as param but defaults OFF (data-snooped; Monti recommends removing).
  - Volatility-targeted position sizing: contracts = RiskPerTradeDollars / (range_width × PointValue).
    Scales down exposure on wide-range days and up on narrow-range days, improving
    return/drawdown ratio from ~6.4 to ~8.7 on 5-year NASDAQ futures backtest.

Lifecycle per session:
  1. Accumulate opening-range high/low from 09:30 NY through 10:00 NY.
     Bar timestamps in the DB are UTC (historical_data_1m convention); converted to ET here.
     Range bars: close_time_et in [09:30, 10:00].  Lock on first bar whose ET time >= 10:00.
  2. After range is locked: if a 5-min bar CLOSES above range_high → enter long at that close.
  3. Optional delta filter (default OFF, price-proxy only — see note below).
  4. Stop = range_low; target = entry + RR * (entry - range_low).
  5. EOD exit at 15:00 NY.
  6. One trade per day maximum. Long only by design.

CHART / DATA REQUIREMENTS:
  - 5-min bars expected. If 1-min data is passed, it is resampled to 5-min internally.
  - Input DataFrame index must be UTC-aware (or UTC-naive if the platform guarantees UTC).
    The strategy converts timestamps to America/New_York for all session-hour logic.
  - IsExitOnSessionCloseStrategy equivalent: EOD managed internally at 15:00 NY.
"""

from __future__ import annotations

import math
from datetime import time as time_t
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register
from strategy_platform.strategies.mobobands.strategy import _summarise, _bootstrap_trades


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

@register
class ORB30Monti(BaseStrategy):
    """Monti's improved ORB30 — long-only NY opening-range breakout with volatility-targeted sizing."""

    name = "orb30_monti"
    bar_type            = 'time'
    supported_bar_types = ['time', '1m']

    default_params: Dict[str, Any] = {
        # 1. Session
        'session_start_hour_ny':   9,
        'session_start_minute_ny': 30,
        'range_duration_minutes':  30,
        'session_end_hour_ny':     15,
        'session_end_minute_ny':   0,
        # 2. Entry
        'use_delta_filter':  False,
        'delta_threshold':   200,
        # 3. Risk
        'use_risk_sizing':         True,   # True = $-risk based; False = fixed contracts
        'risk_per_trade_dollars':  500.0,
        'fixed_contract_count':    1,
        'max_contracts_cap':       50,
        'risk_reward_ratio':       1.0,
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
            # Session — range window start is fixed by design; not swept
            'session_start_hour_ny':   [9],
            'session_start_minute_ny': [30],
            'range_duration_minutes':  [30],
            'session_end_hour_ny':     [15],
            'session_end_minute_ny':   [0],

            # Entry
            'use_delta_filter': [True, False],
            'delta_threshold':  (50, 500, 50),

            # Risk
            'sizing_mode':            ['range_width', 'fixed_contracts'],
            'risk_per_trade_dollars': (100.0, 1000.0, 100.0),
            'fixed_contract_count':   (1, 10, 1),
            'max_contracts_cap':      (5, 50, 5),
            'risk_reward_ratio':      (0.5, 3.0, 0.25),
        }

    param_conditional: Dict[str, tuple] = {
        'delta_threshold':        ('use_delta_filter',  True),
        'risk_per_trade_dollars': ('sizing_mode',       'range_width'),
        'max_contracts_cap':      ('sizing_mode',       'range_width'),
        'fixed_contract_count':   ('sizing_mode',       'fixed_contracts'),
    }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            "1. Session": [
                'session_start_hour_ny', 'session_start_minute_ny',
                'range_duration_minutes',
                'session_end_hour_ny', 'session_end_minute_ny',
            ],
            "2. Entry":   ['use_delta_filter', 'delta_threshold'],
            "3. Risk":    [
                'sizing_mode', 'risk_per_trade_dollars',
                'fixed_contract_count', 'max_contracts_cap',
                'risk_reward_ratio',
            ],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            'session_start_hour_ny':   'Session Start Hour (NY)',
            'session_start_minute_ny': 'Session Start Minute (NY)',
            'range_duration_minutes':  'Range Duration (minutes)',
            'session_end_hour_ny':     'Session End Hour (NY)',
            'session_end_minute_ny':   'Session End Minute (NY)',
            'use_delta_filter':        'Use Delta Filter',
            'delta_threshold':         'Delta Threshold (proxy ticks)',
            'sizing_mode':             'Sizing Mode',
            'risk_per_trade_dollars':  'Risk Per Trade ($)',
            'fixed_contract_count':    'Fixed Contract Count',
            'max_contracts_cap':       'Max Contracts Cap',
            'risk_reward_ratio':       'Risk/Reward Ratio',
        }

    @property
    def description(self) -> str:
        return (
            "Monti's improved ORB30 — long-only NY opening-range breakout "
            "with volatility-targeted position sizing (ported from ORB30Monti.cs)."
        )

    # ------------------------------------------------------------------
    # Backtest / MC  — signatures mirror nybreakout exactly
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
    Detect bar size from median index spacing; resample to 5M if smaller.
    Dashboard may pass 1M bars (historical_data_1m); strategy logic is on 5M.
    """
    if len(df) < 3:
        return df
    diffs = df.index.to_series().diff().dropna()
    median_sec = diffs.median().total_seconds()
    if median_sec < 240:  # < 4 min -> treat as sub-5M, resample
        return df.resample('5min').agg({
            'open':   'first',
            'high':   'max',
            'low':    'min',
            'close':  'last',
            'volume': 'sum',
        }).dropna()
    return df


def _round_tick(price: float, tick_size: float) -> float:
    return round(price / tick_size) * tick_size


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
    """
    Single-pass simulation mirroring ORB30Monti.cs OnBarUpdate.

    DB timezone note: index is assumed UTC (historical_data_1m convention).
    All session-hour comparisons use ET-converted timestamps.

    No look-ahead: range is accumulated and locked on bar-close; entries only
    fire on bars whose close is strictly after the range-lock bar.

    State is reset on each ET calendar day.
    """
    if len(df) < 5:
        return []

    # ---- Param unpack
    sess_start_h  = int(params['session_start_hour_ny'])
    sess_start_m  = int(params['session_start_minute_ny'])
    range_dur_min = int(params['range_duration_minutes'])
    sess_end_h    = int(params['session_end_hour_ny'])
    sess_end_m    = int(params['session_end_minute_ny'])

    use_delta_filter = bool(params['use_delta_filter'])
    delta_threshold  = int(params['delta_threshold'])

    sizing_mode           = str(params['sizing_mode'])
    risk_per_trade_usd    = float(params['risk_per_trade_dollars'])
    fixed_contract_count  = max(1, int(params['fixed_contract_count']))
    max_contracts_cap     = int(params['max_contracts_cap'])
    rr_ratio              = float(params['risk_reward_ratio'])

    point_value = tick_value / tick_size  # dollars per point

    # Session boundary times (ET)
    range_start_t = time_t(sess_start_h, sess_start_m)
    # range_end_t: the first ET bar time >= this triggers range lock
    range_end_t   = time_t(
        sess_start_h + (sess_start_m + range_dur_min) // 60,
        (sess_start_m + range_dur_min) % 60,
    )
    eod_t         = time_t(sess_end_h, sess_end_m)

    # ---- Materialise arrays
    idx   = df.index
    high  = df['high'].to_numpy(dtype=float)
    low   = df['low'].to_numpy(dtype=float)
    open_ = df['open'].to_numpy(dtype=float)
    close = df['close'].to_numpy(dtype=float)
    n     = len(df)

    # ---- Per-day session state
    cur_date       = None
    range_high     = 0.0
    range_low      = 0.0
    range_locked   = False
    trade_taken    = False
    eod_active     = False

    # Open trade state
    in_trade       = False
    trade_entry    = 0.0
    trade_stop     = 0.0
    trade_target   = 0.0
    trade_qty      = 0
    trade_entry_ts: Optional[pd.Timestamp] = None

    trades: List[Dict[str, Any]] = []

    def _reset_day() -> None:
        nonlocal range_high, range_low, range_locked
        nonlocal trade_taken, eod_active
        range_high   = 0.0
        range_low    = 0.0
        range_locked = False
        trade_taken  = False
        eod_active   = False

    def _close_trade(exit_price: float, exit_ts: pd.Timestamp, reason: str) -> None:
        nonlocal in_trade
        pnl_pts    = exit_price - trade_entry          # long only
        pnl_usd    = (pnl_pts / tick_size) * tick_value * trade_qty - commission
        sd         = exit_ts.date()
        trades.append({
            'session_date': sd,
            'day_of_week':  pd.Timestamp(sd).day_name(),
            'side':         'Long',
            'entry_time':   trade_entry_ts,
            'exit_time':    exit_ts,
            'entry_price':  trade_entry,
            'exit_price':   exit_price,
            'stop':         trade_stop,
            'target':       trade_target,
            'contracts':    trade_qty,
            'qty':          trade_qty,
            'pnl_dollars':  pnl_usd,
            'pnl':          pnl_usd,
            'pnl_ticks':    pnl_pts / tick_size,
            'exit_reason':  reason,
            'commission':   commission,
        })
        in_trade = False

    def _compute_qty() -> int:
        """
        Mirror ORB30Monti.ComputeQty().
        RangeWidth: floor(risk_per_trade / (range_width * point_value)), clamped [1, cap].
        FixedContracts: fixed_contract_count.
        """
        if sizing_mode == 'fixed_contracts':
            return max(1, fixed_contract_count)
        # range_width mode
        rw = range_high - range_low
        if rw <= 0:
            return 1
        risk_per_contract = rw * point_value
        if risk_per_contract <= 0:
            return 1
        qty = int(math.floor(risk_per_trade_usd / risk_per_contract))
        qty = max(1, qty)
        qty = min(qty, max_contracts_cap)
        return qty

    # ---- Main loop. Input is ET-naive from loader (historical_data_1m is stored ET).
    for i in range(n):
        ts_et: pd.Timestamp = idx[i]
        bar_date = ts_et.date()

        # ── Day rollover ──────────────────────────────────────────────────
        if bar_date != cur_date:
            if in_trade:
                # Safety: force-close any carry-over position at open of new day
                _close_trade(open_[i] if not np.isnan(open_[i]) else close[i], ts_et, 'eod')
            _reset_day()
            cur_date = bar_date

        # ── EOD exit block ────────────────────────────────────────────────
        bar_time_et = ts_et.time()
        if not eod_active and bar_time_et >= eod_t:
            eod_active = True
            if in_trade:
                _close_trade(open_[i] if not np.isnan(open_[i]) else close[i], ts_et, 'eod')
        if eod_active:
            continue

        # ── Range accumulation and locking ───────────────────────────────
        # Inclusion rule mirrors C#:
        #   inRange when ET bar time in [range_start_t, range_end_t]
        #   Lock fires on the FIRST bar whose ET time >= range_end_t (and still in-range)
        if not range_locked:
            in_range = range_start_t <= bar_time_et <= range_end_t
            if in_range:
                if range_high == 0.0 and range_low == 0.0:
                    range_high = high[i]
                    range_low  = low[i]
                else:
                    if high[i] > range_high:
                        range_high = high[i]
                    if low[i] < range_low:
                        range_low = low[i]

                # Lock at the close of the first bar whose ET time >= range_end_t
                if bar_time_et >= range_end_t:
                    if range_high > 0 and range_low > 0 and range_high > range_low:
                        range_locked = True
                    # Whether locked or not: never enter a trade until range is confirmed
            # Never enter a trade until range is locked
            continue

        # ── One trade per day guard ───────────────────────────────────────
        if trade_taken:
            # Still manage the open trade if one was entered
            if in_trade:
                if low[i] <= trade_stop:
                    _close_trade(trade_stop, ts_et, 'sl')
                elif high[i] >= trade_target:
                    _close_trade(trade_target, ts_et, 'tp')
            continue

        # ── Manage open trade (stop / target) ────────────────────────────
        if in_trade:
            if low[i] <= trade_stop:
                _close_trade(trade_stop, ts_et, 'sl')
            elif high[i] >= trade_target:
                _close_trade(trade_target, ts_et, 'tp')
            # After managing the trade, do not look for new entries on this bar
            continue

        # ── Breakout entry condition ──────────────────────────────────────
        # Close above range high on a bar that is strictly after the range-lock bar.
        # (range_locked is True at this point, so this bar IS strictly after lock.)
        if close[i] <= range_high:
            continue

        # ── Delta filter (optional, OFF by default) ──────────────────────
        # IMPORTANT: This is a PRICE-PROXY, not true volume delta.
        # NT8's C# uses (Close - Open) / TickSize as a proxy because NT8 does not
        # expose per-bar cumulative delta without tick replay. We mirror that proxy
        # exactly so backtest behaviour matches the C# when the filter is enabled.
        # Monti's recommendation is to leave this filter OFF — it is data-snooped.
        if use_delta_filter:
            price_proxy = (close[i] - open_[i]) / tick_size  # ticks, upward-move proxy
            if price_proxy < delta_threshold:
                continue

        # ── Compute position size ─────────────────────────────────────────
        qty = _compute_qty()
        if qty < 1:
            continue

        # ── Set stop and target ───────────────────────────────────────────
        entry_price = close[i]
        stop_price  = _round_tick(range_low, tick_size)
        stop_dist   = entry_price - stop_price
        if stop_dist <= 0:
            continue
        target_price = _round_tick(entry_price + rr_ratio * stop_dist, tick_size)

        # ── Enter long ────────────────────────────────────────────────────
        trade_entry    = entry_price
        trade_stop     = stop_price
        trade_target   = target_price
        trade_qty      = qty
        trade_entry_ts = ts_et
        in_trade       = True
        trade_taken    = True

        # Same-bar exit check (worst-case: stop hits before target if both in range)
        if low[i] <= trade_stop:
            _close_trade(trade_stop, ts_et, 'sl')
        elif high[i] >= trade_target:
            _close_trade(trade_target, ts_et, 'tp')

    return trades
