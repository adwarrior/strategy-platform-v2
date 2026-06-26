"""
SuperTrendFractalModern — Python port of SuperTrendFractalModernStrategy.cs (NT8).

IDENTICAL to SuperTrendFractal in every respect (fractal peak-rejection entries,
exits, sizing, sessions, daily gates) EXCEPT the Supertrend core, which is the
"Modern Adaptive" variant ported from the GBB Pine indicator:

  L2  Regime-scaled multiplier  — the band multiplier breathes with a Kaufman
      Efficiency Ratio (KER) regime signal instead of being fixed.
  L3  Hysteresis flip filter    — the line only flips after price clears the
      opposing band by HystAtr*ATR for HystBars consecutive bars.

L1 (Ehlers adaptive period) is omitted by design, matching the C#.

With EnableRegimeMult=False AND EnableHysteresis=False the core collapses to the
classic Supertrend selection used by the Modern C# (dir-driven) — note this is a
slightly different *classic* formulation than the original strategy's line chain,
exactly as in the two .cs files, so the off-switch reproduces the MODERN file's
classic path, not necessarily the original strategy bar-for-bar.

Source: /home/ad/Scripts/strategies/SuperTrendFractalModernStrategy.cs
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any, Dict, List

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register

# Reuse ALL of the original's shared machinery — signals, exits, sizing, gates,
# stats, MC. Only the supertrend computation differs.
from strategy_platform.strategies.supertrendfractal.strategy import (
    _HHMM_24H,
    _detect_signals,
    _clamp_fractal_length,
    _run_backtest_loop,
    _summarise,
    _bootstrap_trades,
)


# ---------------------------------------------------------------------------
# Modern (L2 + L3) Supertrend core — replicates the C# OnBarUpdate step 1.
# ---------------------------------------------------------------------------

def _ker(c: np.ndarray, i: int, ker_length: int) -> float:
    """Kaufman Efficiency Ratio over ker_length bars: |net move| / sum|bar moves|.
    Mirrors C# Ker() (C#[0]=current=c[i], C#[k]=c[i-k])."""
    direction = abs(c[i] - c[i - ker_length])
    vol = 0.0
    for k in range(ker_length):
        vol += abs(c[i - k] - c[i - k - 1])
    return (direction / vol) if vol > 0.0 else 0.0


def _ker_percent_rank(c: np.ndarray, i: int, ker_length: int, pct_window: int) -> float:
    """Percentile rank of current KER within the last pct_window bars (0..1).
    Mirrors C# KerPercentRank() exactly, including its index arithmetic
    (C#[i] == python c[idx - i])."""
    cur = _ker(c, i, ker_length)
    win = min(pct_window, i - ker_length)   # C#: CurrentBar - KerLength
    if win < 2:
        return cur
    below = 0
    for k in range(1, win + 1):
        # C#: d = |Close[k] - Close[k+KerLength]|  -> python: |c[i-k] - c[i-k-KerLength]|
        d = abs(c[i - k] - c[i - k - ker_length])
        v = 0.0
        for j in range(ker_length):
            # C#: |Close[k+j] - Close[k+j+1]| -> python |c[i-k-j] - c[i-k-j-1]|
            v += abs(c[i - k - j] - c[i - k - j - 1])
        past = (d / v) if v > 0.0 else 0.0
        if past < cur:
            below += 1
    return below / win


def _compute_supertrend_modern(
    df: pd.DataFrame,
    atr_period: int,
    atr_multiplier: float,
    fractal_length: int,
    params: Dict[str, Any],
) -> pd.DataFrame:
    """
    Replicates SuperTrendFractalModernStrategy.cs OnBarUpdate step 1 exactly:
      - Wilder ATR seed (same recursion as the original).
      - L2: regime-scaled multiplier via convex hinge on KER (or KER pct rank).
      - Band ratchet (same as original).
      - L3: line selection with optional hysteresis on the flip, dir-driven.
    Returns DataFrame with columns: atr, top, bottom, line.
    """
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(df)

    enable_regime  = bool(params.get('enable_regime_mult',  True))
    ker_length     = int(params.get('ker_length',           20))
    use_ker_pct    = bool(params.get('use_ker_percentile',  True))
    ker_pct_window = int(params.get('ker_pct_window',        500))
    pivot          = float(params.get('pivot',               0.5))
    trend_gain     = float(params.get('trend_gain',          0.8))
    chop_gain      = float(params.get('chop_gain',           0.5))
    mult_min       = float(params.get('mult_min',            1.0))
    mult_max       = float(params.get('mult_max',            6.0))

    enable_hyst    = bool(params.get('enable_hysteresis',    True))
    hyst_atr       = float(params.get('hyst_atr',            0.5))
    hyst_bars      = int(params.get('hyst_bars',             1))

    atr_arr  = np.zeros(n, dtype=np.float64)
    top_arr  = np.zeros(n, dtype=np.float64)
    bot_arr  = np.zeros(n, dtype=np.float64)
    line_arr = np.zeros(n, dtype=np.float64)
    dir_arr  = np.zeros(n, dtype=np.int64)

    # L3 hysteresis candidate tracking (C# instance fields)
    cand_dir = 0
    cand_count = 0

    for i in range(n):
        # Wilder ATR (identical to original).
        if i == 0:
            atr_arr[i] = h[i] - l[i]
        else:
            close1 = c[i - 1]
            tr = max(abs(l[i] - close1), max(h[i] - l[i], abs(h[i] - close1)))
            denom = min(i + 1, atr_period)
            atr_arr[i] = ((denom - 1) * atr_arr[i - 1] + tr) / denom

        # --- L2: regime-scaled multiplier (convex hinge on KER) ---
        mult_eff = atr_multiplier
        if enable_regime and i > ker_length:
            ker_sig = _ker_percent_rank(c, i, ker_length, ker_pct_window) if use_ker_pct else _ker(c, i, ker_length)
            f_trend = max(0.0, (ker_sig - pivot) / (1.0 - pivot))
            f_chop  = max(0.0, (pivot - ker_sig) / pivot)
            hinge   = atr_multiplier * (1.0 + trend_gain * f_trend + chop_gain * f_chop)
            mult_eff = min(max(hinge, mult_min), mult_max)

        mid = (h[i] + l[i]) / 2.0
        top_value = mid + mult_eff * atr_arr[i]
        bot_value = mid - mult_eff * atr_arr[i]

        if i < fractal_length:
            # No previous line yet — C# uses lineSeries[1] before this point, but
            # the strategy guards CurrentBar < FractalLength (returns early). Seed
            # with the raw bands and dir=+1 so downstream lag is consistent.
            top_arr[i]  = top_value
            bot_arr[i]  = bot_value
            line_arr[i] = bot_value      # dir defaults to +1 (up) in C#
            dir_arr[i]  = 1
            continue

        prev_line = line_arr[i - 1]
        # Band ratchet (C# lines 222-223, vs prev LINE).
        top_s = top_value if (top_value < prev_line or c[i - 1] > prev_line) else prev_line
        bot_s = bot_value if (bot_value > prev_line or c[i - 1] < prev_line) else prev_line

        # --- L3: line selection with optional hysteresis on the flip ---
        prev_dir = dir_arr[i - 1] if i > 0 else 1
        new_dir = prev_dir

        if not enable_hyst:
            if prev_dir == 1:
                new_dir = -1 if (c[i] < bot_s) else 1
            else:
                new_dir = 1 if (c[i] > top_s) else -1
        else:
            buf = hyst_atr * atr_arr[i]
            if prev_dir == 1:
                if c[i] < bot_s - buf:
                    cand_count = cand_count + 1 if cand_dir == -1 else 1
                    cand_dir = -1
                    if cand_count >= hyst_bars:
                        new_dir = -1; cand_dir = 0; cand_count = 0
                else:
                    cand_dir = 0; cand_count = 0
            else:
                if c[i] > top_s + buf:
                    cand_count = cand_count + 1 if cand_dir == 1 else 1
                    cand_dir = 1
                    if cand_count >= hyst_bars:
                        new_dir = 1; cand_dir = 0; cand_count = 0
                else:
                    cand_dir = 0; cand_count = 0

        dir_arr[i]  = new_dir
        top_arr[i]  = top_s
        bot_arr[i]  = bot_s
        line_arr[i] = bot_s if new_dir == 1 else top_s

    return pd.DataFrame({
        'atr':    atr_arr,
        'top':    top_arr,
        'bottom': bot_arr,
        'line':   line_arr,
    }, index=df.index)


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

@register
class SuperTrendFractalModern(BaseStrategy):
    """
    SuperTrendFractalModern: SuperTrendFractal with an adaptive (L2 regime mult +
    L3 hysteresis) Supertrend core. Ported from SuperTrendFractalModernStrategy.cs.
    Default instrument: NQ=F (Mini Nasdaq-100).
    """

    name = 'supertrendfractalmodern'

    bar_type            = 'tick'
    supported_bar_types = ['time', '1m', 'tick']
    tick_bar_size       = 89

    tick_size     = 0.25
    tick_value    = 5.00
    commission_rt = 3.98

    symbol: str = 'NQ=F'

    default_params: Dict[str, Any] = {
        'tick_bar_size': 89,
        'minute_bar_period': 5,
        # 1. Indicator (L0)
        'atr_multiplier': 3,
        'atr_period': 10,
        'fractal_length': 3,
        # 1b. Regime Multiplier (L2)
        'enable_regime_mult': True,
        'ker_length': 20,
        'use_ker_percentile': True,
        'ker_pct_window': 500,
        'pivot': 0.5,
        'trend_gain': 0.8,
        'chop_gain': 0.5,
        'mult_min': 1.0,
        'mult_max': 6.0,
        # 1c. Hysteresis (L3)
        'enable_hysteresis': True,
        'hyst_atr': 0.5,
        'hyst_bars': 1,
        # 2. Signal
        'direction': 'Both',
        'invert_signals': False,
        # 3. Exit
        'exit_mode': 'FixedTPTrailSL',
        'tp_ticks': 80,
        'sl_ticks': 40,
        'tp_atr_mult': 2,
        'sl_atr_mult': 1,
        'rr_ratio': 2,
        # 4. Sizing
        'use_risk_sizing': False,
        'qty': 1,
        'max_risk': 250,
        # 5. Cooldown
        'bars_between_trades': 2,
        # 6. Session
        'enable_session_filter': True,
        'trade_window1_start': '09:45',
        'trade_window1_stop': '11:00',
        'enable_trade_window2': False,
        'trade_window2_start': '09:30',
        'trade_window2_stop': '11:30',
        'enable_trade_window3': False,
        'trade_window3_start': '14:00',
        'trade_window3_stop': '15:55',
        'eod_exit_time': '16:55',
        # 7. Daily gates (0 = off)
        'daily_loss_limit': 0,
        'daily_profit_target': 0,
        'max_consec_losers': 0,
        'max_consec_winners': 0,
    }

    @property
    def param_grid(self) -> Dict[str, Any]:
        return {
            'tick_bar_size':   [89],
            # 1. Indicator
            'atr_multiplier':  (1, 5, 1),
            'atr_period':      (5, 20, 5),
            'fractal_length':  [3, 5, 7],
            # 1b. Regime Multiplier (L2) — the optimization target
            'enable_regime_mult': [True, False],
            'ker_length':      (10, 40, 10),
            'use_ker_percentile': [True, False],
            'ker_pct_window':  (200, 800, 200),
            'pivot':           (0.3, 0.7, 0.1),
            'trend_gain':      (0.0, 1.6, 0.4),
            'chop_gain':       (0.0, 1.0, 0.25),
            'mult_min':        (0.5, 2.0, 0.5),
            'mult_max':        (4.0, 8.0, 1.0),
            # 1c. Hysteresis (L3) — the optimization target
            'enable_hysteresis': [True, False],
            'hyst_atr':        (0.0, 1.5, 0.25),
            'hyst_bars':       (1, 4, 1),
            # 2. Signal
            'direction':       ['Both', 'LongOnly', 'ShortOnly'],
            'invert_signals':  [False, True],
            # 3. Exit
            'exit_mode':       ['FixedTPSL_Ticks', 'FixedTPSL_ATR', 'FixedTPSL_RR',
                                'TrailToLine', 'FixedTPTrailSL'],
            'tp_ticks':        (10, 80, 10),
            'sl_ticks':        (5,  40,  5),
            'tp_atr_mult':     (0.5, 4.0, 0.5),
            'sl_atr_mult':     (0.5, 2.0, 0.5),
            'rr_ratio':        (1.0, 4.0, 0.5),
            # 4. Sizing
            'use_risk_sizing': [False, True],
            'qty':             (1, 4, 1),
            'max_risk':        (100.0, 500.0, 50.0),
            # 5. Cooldown
            'bars_between_trades': (0, 10, 2),
            # 6. Session
            'enable_session_filter':  [True, False],
            'trade_window1_start':    _HHMM_24H,
            'trade_window1_stop':     _HHMM_24H,
            'enable_trade_window2':   [True, False],
            'trade_window2_start':    _HHMM_24H,
            'trade_window2_stop':     _HHMM_24H,
            'enable_trade_window3':   [True, False],
            'trade_window3_start':    _HHMM_24H,
            'trade_window3_stop':     _HHMM_24H,
            'eod_exit_time':          _HHMM_24H,
            # 7. Daily gates
            'daily_loss_limit':       (0.0, 1000.0, 100.0),
            'daily_profit_target':    (0.0, 1000.0, 100.0),
            'max_consec_losers':      (0, 6, 1),
            'max_consec_winners':     (0, 6, 1),
        }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            '0. Bar': ['tick_bar_size'],
            '1. Indicator': ['atr_multiplier', 'atr_period', 'fractal_length'],
            '1b. Regime Multiplier (L2)': [
                'enable_regime_mult', 'ker_length', 'use_ker_percentile',
                'ker_pct_window', 'pivot', 'trend_gain', 'chop_gain',
                'mult_min', 'mult_max',
            ],
            '1c. Hysteresis (L3)': ['enable_hysteresis', 'hyst_atr', 'hyst_bars'],
            '2. Signal': ['direction', 'invert_signals'],
            '3. Exit': ['exit_mode', 'tp_ticks', 'sl_ticks', 'tp_atr_mult',
                        'sl_atr_mult', 'rr_ratio'],
            '4. Sizing': ['use_risk_sizing', 'qty', 'max_risk'],
            '5. Cooldown': ['bars_between_trades'],
            '6. Session': [
                'enable_session_filter',
                'trade_window1_start', 'trade_window1_stop',
                'enable_trade_window2', 'trade_window2_start', 'trade_window2_stop',
                'enable_trade_window3', 'trade_window3_start', 'trade_window3_stop',
                'eod_exit_time',
            ],
            '7. Daily Gates': [
                'daily_loss_limit', 'daily_profit_target',
                'max_consec_losers', 'max_consec_winners',
            ],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            'tick_bar_size':          'Tick Bar Size',
            'atr_multiplier':         'ATR Multiplier (base)',
            'atr_period':             'ATR Period',
            'fractal_length':         'Fractal Length (3/5/7)',
            'enable_regime_mult':     'Enable Regime Multiplier',
            'ker_length':             'KER Lookback',
            'use_ker_percentile':     'Rank KER vs own distribution',
            'ker_pct_window':         'KER Percentile Window',
            'pivot':                  'Hinge Pivot',
            'trend_gain':             'Trend Gain',
            'chop_gain':              'Chop Gain',
            'mult_min':               'Mult Min',
            'mult_max':               'Mult Max',
            'enable_hysteresis':      'Enable Hysteresis',
            'hyst_atr':               'Penetration Buffer (xATR)',
            'hyst_bars':              'Persistence Bars',
            'direction':              'Direction',
            'invert_signals':         'Invert Signals',
            'exit_mode':              'Exit Mode',
            'tp_ticks':               'TP Ticks',
            'sl_ticks':               'SL Ticks',
            'tp_atr_mult':            'TP ATR Multiple',
            'sl_atr_mult':            'SL ATR Multiple',
            'rr_ratio':               'R:R Ratio',
            'use_risk_sizing':        'Use Risk Sizing',
            'qty':                    'Qty (fixed)',
            'max_risk':               'Max Risk $',
            'bars_between_trades':    'Bars Between Trades',
            'enable_session_filter':  'Enable Session Filter',
            'trade_window1_start':    'Window 1 — Start',
            'trade_window1_stop':     'Window 1 — Stop',
            'enable_trade_window2':   'Enable Window 2',
            'trade_window2_start':    'Window 2 — Start',
            'trade_window2_stop':     'Window 2 — Stop',
            'enable_trade_window3':   'Enable Window 3',
            'trade_window3_start':    'Window 3 — Start',
            'trade_window3_stop':     'Window 3 — Stop',
            'eod_exit_time':          'EOD Exit',
            'daily_loss_limit':       'Daily Loss Limit $ (0=off)',
            'daily_profit_target':    'Daily Profit Target $ (0=off)',
            'max_consec_losers':      'Max Consec Losers (0=off)',
            'max_consec_winners':     'Max Consec Winners (0=off)',
        }

    param_conditional: Dict[str, tuple] = {
        # L2 sub-params
        'ker_length':         ('enable_regime_mult', True),
        'use_ker_percentile': ('enable_regime_mult', True),
        'ker_pct_window':     ('use_ker_percentile', True),
        'pivot':              ('enable_regime_mult', True),
        'trend_gain':         ('enable_regime_mult', True),
        'chop_gain':          ('enable_regime_mult', True),
        'mult_min':           ('enable_regime_mult', True),
        'mult_max':           ('enable_regime_mult', True),
        # L3 sub-params
        'hyst_atr':           ('enable_hysteresis', True),
        'hyst_bars':          ('enable_hysteresis', True),
        # Sizing
        'qty':          ('use_risk_sizing',  False),
        'max_risk':     ('use_risk_sizing',  True),
        # Session sub-windows
        'trade_window1_start':  ('enable_session_filter', True),
        'trade_window1_stop':   ('enable_session_filter', True),
        'enable_trade_window2': ('enable_session_filter', True),
        'trade_window2_start':  ('enable_trade_window2',  True),
        'trade_window2_stop':   ('enable_trade_window2',  True),
        'enable_trade_window3': ('enable_session_filter', True),
        'trade_window3_start':  ('enable_trade_window3',  True),
        'trade_window3_stop':   ('enable_trade_window3',  True),
        'eod_exit_time':        ('enable_session_filter', True),
    }

    @property
    def description(self) -> str:
        return "Supertrend fractal with adaptive (regime-mult + hysteresis) core (ported from NT8 C#)."

    # ------------------------------------------------------------------
    # Core backtest — reuses the original's shared loop with the MODERN core.
    # ------------------------------------------------------------------

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        df = data  # tick bars are pre-aggregated by the loader; no resampling here

        atr_period = int(params.get('atr_period',   self.default_params['atr_period']))
        atr_mult   = float(params.get('atr_multiplier', self.default_params['atr_multiplier']))
        frac_len   = _clamp_fractal_length(int(params.get('fractal_length', self.default_params['fractal_length'])))
        invert     = bool(params.get('invert_signals', self.default_params['invert_signals']))

        ind  = _compute_supertrend_modern(df, atr_period, atr_mult, frac_len, params)
        sigs = _detect_signals(df, ind, frac_len, invert)
        trades = _run_backtest_loop(
            df, ind, sigs, params,
            self.tick_size, self.tick_value, self.commission_rt,
        )

        total_sessions = int(df['close'].resample('D').last().count())
        stats     = _summarise(trades, total_sessions=total_sessions)
        bs        = _bootstrap_trades(trades, total_sessions=total_sessions)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return {**stats, **bs, 'total_trades': stats['trades'], 'trades': trades_df}
