"""
CCT — VWT-Bible inspired H1 POI + reclaim + iFVG-entry strategy.

Ported from /home/ad/Scripts/strategies/CCT (NinjaScript C#).
Spec:  /home/ad/Scripts/strategies/CCT_VWT_Spec.md.

Architecture:
  - Input: 1-minute bars (primary / entry timeframe).
  - H1 series derived by resampling 1m -> 60min (POI creation/activation).
  - M5 series derived by resampling 1m -> 5min (optional P/D filter only).

Lifecycle per POI:
  1. On each H1 close, spawn a long POI at High and short POI at Low.
  2. (Optional) Virgin-wick filter: invalidate POI if any later H1 wick sweeps
     the level before it's activated.
  3. Activation: a later H1 closes beyond the POI level. Target = activation
     bar's high (long) or low (short).
  4. LTF state machine (1m) after activation:
       Phase 0 -> close beyond POI (breach)
       Phase 1 -> close back through POI (reclaim)  OR  iFVG inversion
       Phase 2 -> close beyond intent-bar high/low  OR  iFVG inversion
  5. Entry at close of trigger bar. Stop = AttemptExtreme / alt-stop / FVG
     pattern / ATR. Target = HTF Bar2 high/low / RR multiple / ATR multiple.
  6. Optional P/D filter gates entries on the M5 impulse range.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register
from strategy_platform.strategies.mobobands.strategy import _summarise, _bootstrap_trades


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

@register
class CCT(BaseStrategy):
    """H1 POI reclaim + iFVG entry, with optional virgin-wick, P/D, and ATR modes."""

    name = "cct"
    bar_type = '1m'
    supported_bar_types = ['1m']

    default_params: Dict[str, Any] = {
        'enable_long': True,
        'enable_short': True,
        'htf_minutes': 60,
        'max_pois_to_keep': 50,
        'void_self_on_win': True,
        'only_trade_newest_poi': False,
        'selection_mode': 'BestPrice',
        'require_bullish_bar_for_longs': False,
        'require_bearish_bar_for_shorts': False,
        'only_one_trade_per_poi': True,
        'require_virgin_wick': True,
        'min_rr': 1,
        'use_risk_sizing': False,
        'max_risk': 100,
        'max_contracts': 10,
        'qty': 1,
        'use_rr_target': False,
        'rr_target_multiple': 2,
        'use_alternative_stop': False,
        'use_ifvg_pattern_stop': True,
        'use_ifvg_entry': False,
        'ifvg_mode': 'Either',
        'fvg_min_gap_ticks': 11,
        'max_fvg_age_bars': 0,
        'use_atr_stop_target': False,
        'atr_stop_period': 14,
        'atr_stop_mult': 1,
        'atr_target_mult': 2,
        'require_premium_discount': False,
        'range_bars_minutes': 5,
        'impulse_min_bars': 3,
        'impulse_min_atr_mult': 2,
        'impulse_min_body_ratio': 0.6,
        'pd_atr_period': 14,
        'eq_tolerance_ticks': 0,
        'enable_session_window': False,
        'session_start': '09:30',
        'session_end': '16:00',
        'session_close_exit_time': '16:45',
        'enable_session_close_exit': True,
    }

    # Defaults overridden by dashboard per symbol
    tick_size     = 0.25
    tick_value    = 1.25
    commission_rt = 1.24

    symbol  = 'MNQ=F'
    db_host: Optional[str] = None

    # ------------------------------------------------------------------
    # param_grid / groups / display
    # ------------------------------------------------------------------

    @property
    def param_grid(self) -> Dict[str, Any]:
        return {
            'enable_long':                    [True, False],
            'enable_short':                   [True, False],

            'htf_minutes':                    [60],

            'max_pois_to_keep':               (50, 500, 50),
            'void_self_on_win':               [True, False],
            'only_trade_newest_poi':          [True, False],
            'selection_mode':                 ['BestPrice', 'MostRecent'],
            'require_bullish_bar_for_longs':  [True, False],
            'require_bearish_bar_for_shorts': [True, False],
            'only_one_trade_per_poi':         [True, False],
            'require_virgin_wick':            [True, False],

            'min_rr':                         (0.5, 3.0, 0.25),
            'use_risk_sizing':                [True, False],
            'max_risk':                       (50.0, 500.0, 50.0),
            'max_contracts':                  (1, 10, 1),
            'qty':                            (1, 5, 1),

            'use_rr_target':                  [True, False],
            'rr_target_multiple':             (1.0, 5.0, 0.5),

            'use_alternative_stop':           [True, False],
            'use_ifvg_pattern_stop':          [True, False],

            'use_ifvg_entry':                 [True, False],
            'ifvg_mode':                      ['IfvgOnly', 'Either', 'Both'],
            'fvg_min_gap_ticks':              (1, 20, 1),
            'max_fvg_age_bars':               (0, 50, 5),

            'use_atr_stop_target':            [True, False],
            'atr_stop_period':                (5, 30, 1),
            'atr_stop_mult':                  (0.5, 3.0, 0.25),
            'atr_target_mult':                (0.5, 5.0, 0.25),

            'require_premium_discount':       [True, False],
            'range_bars_minutes':             [3, 5, 15],
            'impulse_min_bars':               (2, 8, 1),
            'impulse_min_atr_mult':           (1.0, 5.0, 0.5),
            'impulse_min_body_ratio':         (0.3, 0.9, 0.1),
            'pd_atr_period':                  (5, 30, 1),
            'eq_tolerance_ticks':             (0, 20, 1),

            'enable_session_window':          [True, False],
            'session_start':                  ['07:00', '08:00', '09:00', '09:30', '10:00'],
            'session_end':                    ['15:00', '15:30', '16:00', '16:30', '17:00'],
            'session_close_exit_time':        ['15:00', '15:30', '16:00', '16:30', '16:45', '16:55'],
            'enable_session_close_exit':      [True, False],
        }

    # Sub-params hidden when parent toggle is off
    param_conditional: Dict[str, Tuple[str, Any]] = {
        'max_risk':              ('use_risk_sizing',       True),
        'qty':                   ('use_risk_sizing',       False),
        'rr_target_multiple':    ('use_rr_target',         True),
        'ifvg_mode':             ('use_ifvg_entry',        True),
        'fvg_min_gap_ticks':     ('use_ifvg_entry',        True),
        'max_fvg_age_bars':      ('use_ifvg_entry',        True),
        'use_ifvg_pattern_stop': ('use_ifvg_entry',        True),
        'session_start':         ('enable_session_window', True),
        'session_end':           ('enable_session_window', True),
        'atr_stop_period':       ('use_atr_stop_target',   True),
        'atr_stop_mult':         ('use_atr_stop_target',   True),
        'atr_target_mult':       ('use_atr_stop_target',   True),
        'range_bars_minutes':    ('require_premium_discount', True),
        'impulse_min_bars':      ('require_premium_discount', True),
        'impulse_min_atr_mult':  ('require_premium_discount', True),
        'impulse_min_body_ratio':('require_premium_discount', True),
        'pd_atr_period':         ('require_premium_discount', True),
        'eq_tolerance_ticks':    ('require_premium_discount', True),
        'session_close_exit_time': ('enable_session_close_exit', True),
    }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            "1. Direction":     ['enable_long', 'enable_short'],
            "2. HTF":           ['htf_minutes'],
            "3. POI":           ['max_pois_to_keep', 'void_self_on_win', 'only_trade_newest_poi',
                                 'selection_mode', 'require_bullish_bar_for_longs',
                                 'require_bearish_bar_for_shorts', 'only_one_trade_per_poi',
                                 'require_virgin_wick'],
            "4. Risk":          ['min_rr', 'use_risk_sizing', 'max_risk', 'max_contracts', 'qty',
                                 'use_rr_target', 'rr_target_multiple'],
            "5. Stop Placement":['use_alternative_stop', 'use_ifvg_pattern_stop'],
            "6. IFVG Entry":    ['use_ifvg_entry', 'ifvg_mode', 'fvg_min_gap_ticks', 'max_fvg_age_bars'],
            "7. ATR Stop/Target":['use_atr_stop_target', 'atr_stop_period',
                                  'atr_stop_mult', 'atr_target_mult'],
            "8. Premium/Discount":['require_premium_discount', 'range_bars_minutes',
                                   'impulse_min_bars', 'impulse_min_atr_mult',
                                   'impulse_min_body_ratio', 'pd_atr_period', 'eq_tolerance_ticks'],
            "9. Sessions":      ['enable_session_window', 'session_start', 'session_end',
                                 'enable_session_close_exit', 'session_close_exit_time'],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            'enable_long':                    'Enable Long',
            'enable_short':                   'Enable Short',
            'htf_minutes':                    'HTF Period (minutes)',
            'max_pois_to_keep':               'Max POIs to Keep',
            'void_self_on_win':               'Void Self POI on Win',
            'only_trade_newest_poi':          'Only Trade Newest POI',
            'selection_mode':                 'POI Selection Mode',
            'require_bullish_bar_for_longs':  'Require Bullish Bar for Longs',
            'require_bearish_bar_for_shorts': 'Require Bearish Bar for Shorts',
            'only_one_trade_per_poi':         'Only One Trade per POI',
            'require_virgin_wick':            'Require Virgin Wick',
            'min_rr':                         'Min R:R',
            'use_risk_sizing':                'Use Risk-Based Sizing',
            'max_risk':                       'Max Risk per Trade ($)',
            'max_contracts':                  'Max Contracts',
            'qty':                            'Qty (fixed)',
            'use_rr_target':                  'Use R:R Target',
            'rr_target_multiple':             'R:R Target Multiple',
            'use_alternative_stop':           'Use Alternative Stop',
            'use_ifvg_pattern_stop':          'Use iFVG Pattern Stop',
            'use_ifvg_entry':                 'Use iFVG Entry Filter',
            'ifvg_mode':                      'iFVG Entry Mode',
            'fvg_min_gap_ticks':              'FVG Min Gap (ticks)',
            'max_fvg_age_bars':               'Max FVG Age (bars; 0=no limit)',
            'use_atr_stop_target':            'Use ATR Stop + Target',
            'atr_stop_period':                'ATR Period',
            'atr_stop_mult':                  'ATR Stop Multiple',
            'atr_target_mult':                'ATR Target Multiple',
            'require_premium_discount':       'Require Premium/Discount Filter',
            'range_bars_minutes':             'Range Bars (minutes)',
            'impulse_min_bars':               'Impulse Min Bars',
            'impulse_min_atr_mult':           'Impulse Min ATR Multiple',
            'impulse_min_body_ratio':         'Impulse Min Body Ratio',
            'pd_atr_period':                  'P/D ATR Period',
            'eq_tolerance_ticks':             'EQ Tolerance (ticks)',
            'enable_session_window':          'Enable Session Window',
            'session_start':                  'Session Start (HH:MM)',
            'session_end':                    'Session End (HH:MM)',
            'session_close_exit_time':        'Session Close Exit Time',
            'enable_session_close_exit':      'Enable Session Close Exit',
        }

    @property
    def description(self) -> str:
        return "H1 POI reclaim + iFVG entry (VWT-Bible inspired). Ported from CCT.cs."

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
            order = rng.permutation(n)
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
    """Detect bar size; if sub-1m, resample to 1m. If already >=1m, return as-is."""
    if len(df) < 3:
        return df
    diffs = df.index.to_series().diff().dropna()
    median_sec = diffs.median().total_seconds()
    if median_sec < 55:  # < ~1 min -> upsample by aggregation
        return df.resample('1min').agg({
            'open':   'first',
            'high':   'max',
            'low':    'min',
            'close':  'last',
            'volume': 'sum',
        }).dropna()
    return df


def _resample(df: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Resample 1m bars to N-minute bars. Left-labelled."""
    rule = f'{minutes}min'
    out = df.resample(rule, label='left', closed='left').agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum',
    }).dropna()
    return out


def _parse_time(s: str):
    from datetime import time as time_t
    if not s:
        return None
    h, m = int(s.split(':')[0]), int(s.split(':')[1])
    return time_t(h, m)


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """Wilder's ATR; returns array same length as inputs (leading NaNs)."""
    n = len(high)
    if n < 2:
        return np.full(n, np.nan)
    tr = np.empty(n, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i]  - close[i-1]),
        )
    atr = np.full(n, np.nan)
    if n < period:
        return atr
    atr[period-1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    return atr


# ---------------------------------------------------------------------------
# Main backtest loop
# ---------------------------------------------------------------------------

def _run_backtest_loop(
    df:         pd.DataFrame,
    params:     Dict[str, Any],
    tick_size:  float,
    tick_value: float,
    commission: float,
) -> List[Dict[str, Any]]:
    """Single-pass simulation mirroring CCT_Strategy.cs OnBarUpdate."""
    if len(df) < 60:
        return []

    # ---- Param unpack
    enable_long  = bool(params['enable_long'])
    enable_short = bool(params['enable_short'])

    htf_minutes  = int(params['htf_minutes'])

    max_pois_to_keep  = int(params['max_pois_to_keep'])
    void_self_on_win  = bool(params['void_self_on_win'])
    only_trade_newest = bool(params['only_trade_newest_poi'])
    selection_mode    = str(params['selection_mode'])
    req_bullish_long  = bool(params['require_bullish_bar_for_longs'])
    req_bearish_short = bool(params['require_bearish_bar_for_shorts'])
    only_one_per_poi  = bool(params['only_one_trade_per_poi'])
    require_virgin    = bool(params['require_virgin_wick'])

    min_rr            = float(params['min_rr'])
    use_risk_sizing   = bool(params['use_risk_sizing'])
    max_risk          = float(params['max_risk'])
    max_contracts     = int(params['max_contracts'])
    qty_fixed         = max(1, int(params['qty']))

    use_rr_target     = bool(params['use_rr_target'])
    rr_mult           = float(params['rr_target_multiple'])

    use_alt_stop      = bool(params['use_alternative_stop'])
    use_ifvg_stop     = bool(params['use_ifvg_pattern_stop'])

    use_ifvg          = bool(params['use_ifvg_entry'])
    ifvg_mode         = str(params['ifvg_mode'])
    fvg_min_gap       = float(params['fvg_min_gap_ticks']) * tick_size
    max_fvg_age       = int(params['max_fvg_age_bars'])

    use_atr_stop_tgt  = bool(params['use_atr_stop_target'])
    atr_stop_period   = int(params['atr_stop_period'])
    atr_stop_mult     = float(params['atr_stop_mult'])
    atr_target_mult   = float(params['atr_target_mult'])

    require_pd        = bool(params['require_premium_discount'])
    range_minutes     = int(params['range_bars_minutes'])
    imp_min_bars      = int(params['impulse_min_bars'])
    imp_min_atr_mult  = float(params['impulse_min_atr_mult'])
    imp_min_body      = float(params['impulse_min_body_ratio'])
    pd_atr_period     = int(params['pd_atr_period'])
    eq_tol            = int(params['eq_tolerance_ticks']) * tick_size

    enable_sess_window = bool(params.get('enable_session_window', False))
    sess_start_t      = _parse_time(str(params.get('session_start', '09:30')))
    sess_end_t        = _parse_time(str(params.get('session_end',   '16:00')))
    sess_close_exit_t = _parse_time(str(params.get('session_close_exit_time', '16:45')))
    enable_close_exit = bool(params['enable_session_close_exit'])

    point_value = tick_value / tick_size

    # ---- Materialise 1m primary
    idx   = df.index
    o     = df['open'].to_numpy(dtype=float)
    h     = df['high'].to_numpy(dtype=float)
    l     = df['low'].to_numpy(dtype=float)
    c     = df['close'].to_numpy(dtype=float)
    n     = len(df)

    # ---- Derive H1 + (optional) range series
    htf_df = _resample(df, htf_minutes)
    htf_idx = htf_df.index
    htf_o   = htf_df['open'].to_numpy(dtype=float)
    htf_h   = htf_df['high'].to_numpy(dtype=float)
    htf_l   = htf_df['low'].to_numpy(dtype=float)
    htf_c   = htf_df['close'].to_numpy(dtype=float)
    nh      = len(htf_df)

    # Map every 1m bar -> index of the most recent CLOSED htf bar.
    # An H1 bar at timestamp T (left-labelled) covers [T, T+htf_minutes).
    # It is "closed" once the 1m clock crosses T + htf_minutes.
    htf_close_ts = htf_idx + pd.Timedelta(minutes=htf_minutes)
    bar_to_htf = np.full(n, -1, dtype=int)
    j = -1
    for i in range(n):
        # advance j while next htf bar has closed by idx[i]
        while j + 1 < nh and htf_close_ts[j + 1] <= idx[i]:
            j += 1
        bar_to_htf[i] = j

    # Pre-compute ATR on 1m for ATR stop/target
    atr_1m = _atr(h, l, c, atr_stop_period) if use_atr_stop_tgt else None

    # P/D range state
    range_high = 0.0
    range_low  = 0.0
    range_eq   = 0.0
    range_valid = False

    range_df = None
    range_h_arr = range_l_arr = range_o_arr = range_c_arr = None
    range_atr = None
    range_idx = None
    bar_to_range = None
    if require_pd:
        range_df    = _resample(df, range_minutes)
        range_idx   = range_df.index
        range_o_arr = range_df['open'].to_numpy(dtype=float)
        range_h_arr = range_df['high'].to_numpy(dtype=float)
        range_l_arr = range_df['low'].to_numpy(dtype=float)
        range_c_arr = range_df['close'].to_numpy(dtype=float)
        range_atr   = _atr(range_h_arr, range_l_arr, range_c_arr, pd_atr_period)
        # Map 1m -> most recent CLOSED range bar
        range_close_ts = range_idx + pd.Timedelta(minutes=range_minutes)
        bar_to_range = np.full(n, -1, dtype=int)
        k = -1
        for i in range(n):
            while k + 1 < len(range_idx) and range_close_ts[k + 1] <= idx[i]:
                k += 1
            bar_to_range[i] = k

    # ---- POI list (dicts; mirrors C# Poi class)
    pois: List[Dict[str, Any]] = []
    next_poi_id = 1

    # ---- Per-trade state
    in_trade = False
    trade_side: Optional[str] = None
    trade_entry = 0.0
    trade_stop  = 0.0
    trade_target = 0.0
    trade_qty   = 0
    trade_poi_id = -1
    trade_entry_ts: Optional[pd.Timestamp] = None

    last_processed_htf = -1
    trades: List[Dict[str, Any]] = []

    # ---- Helpers
    def _close_trade(exit_price: float, exit_ts, exit_reason: str):
        nonlocal in_trade, trade_side, trade_poi_id
        side = trade_side
        if side == 'Long':
            pnl_pts = exit_price - trade_entry
        else:
            pnl_pts = trade_entry - exit_price
        pnl_dollars = (pnl_pts / tick_size) * tick_value * trade_qty - commission
        is_win = pnl_dollars > 0
        sd = pd.Timestamp(exit_ts).date()
        trades.append({
            'session_date':  sd,
            'day_of_week':   pd.Timestamp(sd).day_name(),
            'side':          side,
            'entry_time':    trade_entry_ts,
            'exit_time':     exit_ts,
            'entry_price':   trade_entry,
            'exit_price':    exit_price,
            'stop':          trade_stop,
            'target':        trade_target,
            'qty':           trade_qty,
            'pnl':           pnl_dollars,
            'pnl_ticks':     pnl_pts / tick_size,
            'exit_reason':   exit_reason,
            'commission':    commission,
            'poi_id':        trade_poi_id,
        })

        # Win-side POI voiding
        if is_win:
            traded = next((p for p in pois if p['id'] == trade_poi_id), None)
            if traded is not None:
                if void_self_on_win:
                    traded['valid'] = False
                for p in pois:
                    if not p['valid']:
                        continue
                    if traded['is_long'] and p['is_long'] and p['level'] < traded['level']:
                        p['valid'] = False
                    elif not traded['is_long'] and not p['is_long'] and p['level'] > traded['level']:
                        p['valid'] = False
        in_trade = False
        trade_side = None
        trade_poi_id = -1

    def _process_new_htf_bar(hi: int):
        """When 1m crosses an H1 close: spawn new POI, run virgin sweep, run activation."""
        nonlocal next_poi_id, pois
        if hi < 0 or hi >= nh:
            return
        htf_open  = htf_o[hi]
        htf_close = htf_c[hi]
        htf_high  = htf_h[hi]
        htf_low   = htf_l[hi]
        htf_time  = htf_idx[hi]
        is_bull   = htf_close > htf_open

        # 1) Spawn long + short POI candidates from this H1 bar
        if enable_long:
            pois.append({
                'id': next_poi_id, 'is_long': True,
                'level': htf_high, 'opposing_level': htf_low,
                'activated': False, 'valid': True, 'virgin': True,
                'is_bullish_bar': is_bull,
                'creation_htf_idx': hi,
                'activation_htf_time': None,
                'activation_bar_idx': -1,   # 1m bar at which activation became valid
                'target': float('nan'),
                'phase': 0,
                'attempt_extreme': float('nan'),
                'attempt_extreme_init': False,
                'breach_bar_idx': -1,
                'has_been_traded': False,
                'has_intent_bar': False, 'intent_break_level': float('nan'),
                'has_open_below': False, 'last_open_below_low': float('nan'),
                'has_open_above': False, 'last_open_above_high': float('nan'),
                'has_fvg': False,
                'fvg_top': float('nan'), 'fvg_bottom': float('nan'),
                'fvg_pattern_low': float('nan'), 'fvg_pattern_high': float('nan'),
                'fvg_formation_bar': -1,
                'fvg_has_been_inverted': False,
            })
            next_poi_id += 1
        if enable_short:
            pois.append({
                'id': next_poi_id, 'is_long': False,
                'level': htf_low, 'opposing_level': htf_high,
                'activated': False, 'valid': True, 'virgin': True,
                'is_bullish_bar': is_bull,
                'creation_htf_idx': hi,
                'activation_htf_time': None,
                'activation_bar_idx': -1,
                'target': float('nan'),
                'phase': 0,
                'attempt_extreme': float('nan'),
                'attempt_extreme_init': False,
                'breach_bar_idx': -1,
                'has_been_traded': False,
                'has_intent_bar': False, 'intent_break_level': float('nan'),
                'has_open_below': False, 'last_open_below_low': float('nan'),
                'has_open_above': False, 'last_open_above_high': float('nan'),
                'has_fvg': False,
                'fvg_top': float('nan'), 'fvg_bottom': float('nan'),
                'fvg_pattern_low': float('nan'), 'fvg_pattern_high': float('nan'),
                'fvg_formation_bar': -1,
                'fvg_has_been_inverted': False,
            })
            next_poi_id += 1

        # 2) Activation FIRST (so a POI that activates on this bar isn't killed by virgin sweep)
        just_activated_ids = set()
        for p in pois:
            if not p['valid'] or p['activated']:
                continue
            if p['creation_htf_idx'] == hi:
                continue  # cannot self-activate on creation bar
            if p['is_long']:
                if htf_close > p['level']:
                    p['activated'] = True
                    p['target'] = htf_high
                    p['activation_htf_time'] = htf_time
                    p['phase'] = 0
                    p['attempt_extreme_init'] = False
                    p['has_intent_bar'] = False
                    p['intent_break_level'] = float('nan')
                    just_activated_ids.add(p['id'])
                    if only_trade_newest:
                        for q in pois:
                            if q['valid'] and q['is_long'] and q['id'] < p['id']:
                                q['valid'] = False
            else:
                if htf_close < p['level']:
                    p['activated'] = True
                    p['target'] = htf_low
                    p['activation_htf_time'] = htf_time
                    p['phase'] = 0
                    p['attempt_extreme_init'] = False
                    p['has_intent_bar'] = False
                    p['intent_break_level'] = float('nan')
                    just_activated_ids.add(p['id'])
                    if only_trade_newest:
                        for q in pois:
                            if q['valid'] and not q['is_long'] and q['id'] < p['id']:
                                q['valid'] = False

        # 3) Virgin-wick filter on still-non-activated POIs (skip the creation bar)
        if require_virgin:
            for p in pois:
                if not p['valid'] or p['activated'] or not p['virgin']:
                    continue
                if p['creation_htf_idx'] == hi:
                    continue
                swept = (p['is_long']     and htf_high >= p['level']) or \
                        (not p['is_long'] and htf_low  <= p['level'])
                if swept:
                    p['virgin'] = False
                    p['valid'] = False

        # 4) Prune
        if len(pois) > max_pois_to_keep:
            pois[:] = [p for p in pois if p['valid']]
            if len(pois) > max_pois_to_keep:
                # drop oldest activated first
                pois.sort(key=lambda x: (x['activation_htf_time'] or pd.Timestamp.min, x['id']))
                while len(pois) > max_pois_to_keep:
                    pois.pop(0)

    def _update_pd_range(ri: int):
        nonlocal range_high, range_low, range_eq, range_valid
        if not require_pd or ri < max(imp_min_bars, pd_atr_period):
            return
        a = range_atr[ri]
        if a is None or math.isnan(a) or a <= 0:
            return
        min_range_pts = imp_min_atr_mult * a

        # Walk back from ri: try each end-bar
        max_lookback = min(ri, 200)
        for end_idx in range(0, max_lookback - imp_min_bars + 1):
            ei = ri - end_idx  # newest bar of the run
            bullish = range_c_arr[ei] > range_o_arr[ei]
            bearish = range_c_arr[ei] < range_o_arr[ei]
            if not bullish and not bearish:
                continue

            run_len = 1
            body_sum = abs(range_c_arr[ei] - range_o_arr[ei])
            rng_sum  = range_h_arr[ei] - range_l_arr[ei]
            run_high = range_h_arr[ei]
            run_low  = range_l_arr[ei]

            k = ei - 1
            while k >= 0 and (ri - k) <= max_lookback:
                same_dir = (bullish and range_c_arr[k] > range_o_arr[k]) or \
                           (bearish and range_c_arr[k] < range_o_arr[k])
                if not same_dir:
                    break
                run_len += 1
                body_sum += abs(range_c_arr[k] - range_o_arr[k])
                rng_sum  += range_h_arr[k] - range_l_arr[k]
                if range_h_arr[k] > run_high: run_high = range_h_arr[k]
                if range_l_arr[k] < run_low:  run_low  = range_l_arr[k]
                k -= 1

            if run_len < imp_min_bars:
                continue

            first_idx = ei - run_len + 1
            total_range = abs(range_c_arr[ei] - range_o_arr[first_idx])
            if total_range < min_range_pts:
                continue

            body_ratio = (body_sum / rng_sum) if rng_sum > 0 else 0.0
            if body_ratio < imp_min_body:
                continue

            range_high = run_high
            range_low  = run_low
            range_eq   = (run_high + run_low) / 2.0
            range_valid = True
            return  # first valid impulse wins

    def _detect_fvg(p: Dict[str, Any], i: int):
        """1m FVG detection during Phase 0-1 (mirror C# DetectAndTrackFvg)."""
        if not use_ifvg or i < 2:
            return
        if p['phase'] not in (0, 1):
            return
        if p['is_long']:
            # Bearish FVG: low[i-2] > high[i]
            if l[i-2] > h[i]:
                gap_top = l[i-2]
                gap_bot = h[i]
                gap_size = gap_top - gap_bot
                if gap_size < fvg_min_gap:
                    return
                if p['has_fvg']:
                    existing_dist = abs(p['fvg_bottom'] - p['level'])
                    new_dist = abs(gap_bot - p['level'])
                    if new_dist >= existing_dist:
                        return
                p['has_fvg'] = True
                p['fvg_top'] = gap_top
                p['fvg_bottom'] = gap_bot
                p['fvg_pattern_low']  = l[i-1]
                p['fvg_pattern_high'] = h[i-1]
                p['fvg_formation_bar'] = i
                p['fvg_has_been_inverted'] = False
        else:
            # Bullish FVG: high[i-2] < low[i]
            if h[i-2] < l[i]:
                gap_bot = h[i-2]
                gap_top = l[i]
                gap_size = gap_top - gap_bot
                if gap_size < fvg_min_gap:
                    return
                if p['has_fvg']:
                    existing_dist = abs(p['fvg_top'] - p['level'])
                    new_dist = abs(gap_top - p['level'])
                    if new_dist >= existing_dist:
                        return
                p['has_fvg'] = True
                p['fvg_top'] = gap_top
                p['fvg_bottom'] = gap_bot
                p['fvg_pattern_low']  = l[i-1]
                p['fvg_pattern_high'] = h[i-1]
                p['fvg_formation_bar'] = i
                p['fvg_has_been_inverted'] = False

    def _update_intent_bar(p: Dict[str, Any], i: int):
        if p['phase'] >= 2:
            return
        if p['is_long']:
            if o[i] > p['level'] and c[i] < o[i]:
                p['has_intent_bar'] = True
                p['intent_break_level'] = h[i]
        else:
            if o[i] < p['level'] and c[i] > o[i]:
                p['has_intent_bar'] = True
                p['intent_break_level'] = l[i]

    def _update_alt_stop_bar(p: Dict[str, Any], i: int):
        if p['is_long']:
            if o[i] < p['level']:
                p['has_open_below'] = True
                p['last_open_below_low'] = l[i]
        else:
            if o[i] > p['level']:
                p['has_open_above'] = True
                p['last_open_above_high'] = h[i]

    def _ltf_update_and_signal(p: Dict[str, Any], i: int) -> bool:
        """Returns True if p signals entry on this bar."""
        # Opposing-POI invalidation
        if not p['has_been_traded'] and not math.isnan(p['opposing_level']):
            if p['is_long'] and c[i] < p['opposing_level']:
                p['valid'] = False
                return False
            if not p['is_long'] and c[i] > p['opposing_level']:
                p['valid'] = False
                return False

        signaled = False
        if p['is_long']:
            if p['phase'] == 0:
                if c[i] < p['level']:
                    p['phase'] = 1
                    p['attempt_extreme'] = l[i]
                    p['attempt_extreme_init'] = True
                    p['breach_bar_idx'] = i
            elif p['phase'] == 1:
                if not p['attempt_extreme_init']:
                    p['attempt_extreme'] = l[i]
                    p['attempt_extreme_init'] = True
                else:
                    p['attempt_extreme'] = min(p['attempt_extreme'], l[i])

                if c[i] > p['level']:
                    p['phase'] = 2
                else:
                    if use_ifvg and ifvg_mode in ('IfvgOnly', 'Either'):
                        age = (i - p['fvg_formation_bar']) if max_fvg_age > 0 else 0
                        in_age = (max_fvg_age == 0) or (age <= max_fvg_age)
                        valid = p['has_fvg'] and not p['fvg_has_been_inverted'] and in_age
                        if valid and c[i] > p['fvg_top']:
                            signaled = True
            elif p['phase'] == 2:
                if c[i] < p['level']:
                    p['phase'] = 1
                    p['attempt_extreme'] = l[i]
                    p['attempt_extreme_init'] = True
                    p['breach_bar_idx'] = i
                    p['has_fvg'] = False
                    return False

                intent_met = p['has_intent_bar'] and c[i] > p['intent_break_level']

                fvg_before_breach = p['has_fvg'] and p['breach_bar_idx'] > 0 \
                    and p['fvg_formation_bar'] < p['breach_bar_idx']
                age = (i - p['fvg_formation_bar']) if max_fvg_age > 0 else 0
                in_age = (max_fvg_age == 0) or (age <= max_fvg_age)
                fvg_valid = fvg_before_breach and not p['fvg_has_been_inverted'] and in_age
                fvg_inverted = c[i] > p['fvg_top']
                fvg_below_poi = p['fvg_bottom'] < p['level']
                needs_reclaim = fvg_below_poi and c[i] > p['level']
                ifvg_met = use_ifvg and fvg_valid and fvg_inverted and (not fvg_below_poi or needs_reclaim)

                if not use_ifvg:
                    signaled = intent_met
                else:
                    if ifvg_mode == 'IfvgOnly':
                        signaled = ifvg_met
                    elif ifvg_mode == 'Either':
                        signaled = intent_met or ifvg_met
                    elif ifvg_mode == 'Both':
                        signaled = intent_met and ifvg_met
        else:
            # SHORT mirror
            if p['phase'] == 0:
                if c[i] > p['level']:
                    p['phase'] = 1
                    p['attempt_extreme'] = h[i]
                    p['attempt_extreme_init'] = True
                    p['breach_bar_idx'] = i
            elif p['phase'] == 1:
                if not p['attempt_extreme_init']:
                    p['attempt_extreme'] = h[i]
                    p['attempt_extreme_init'] = True
                else:
                    p['attempt_extreme'] = max(p['attempt_extreme'], h[i])

                if c[i] < p['level']:
                    p['phase'] = 2
                else:
                    if use_ifvg and ifvg_mode in ('IfvgOnly', 'Either'):
                        age = (i - p['fvg_formation_bar']) if max_fvg_age > 0 else 0
                        in_age = (max_fvg_age == 0) or (age <= max_fvg_age)
                        valid = p['has_fvg'] and not p['fvg_has_been_inverted'] and in_age
                        if valid and c[i] < p['fvg_bottom']:
                            signaled = True
            elif p['phase'] == 2:
                if c[i] > p['level']:
                    p['phase'] = 1
                    p['attempt_extreme'] = h[i]
                    p['attempt_extreme_init'] = True
                    p['breach_bar_idx'] = i
                    p['has_fvg'] = False
                    return False

                intent_met = p['has_intent_bar'] and c[i] < p['intent_break_level']

                fvg_before_breach = p['has_fvg'] and p['breach_bar_idx'] > 0 \
                    and p['fvg_formation_bar'] < p['breach_bar_idx']
                age = (i - p['fvg_formation_bar']) if max_fvg_age > 0 else 0
                in_age = (max_fvg_age == 0) or (age <= max_fvg_age)
                fvg_valid = fvg_before_breach and not p['fvg_has_been_inverted'] and in_age
                fvg_inverted = c[i] < p['fvg_bottom']
                fvg_above_poi = p['fvg_top'] > p['level']
                needs_reclaim = fvg_above_poi and c[i] < p['level']
                ifvg_met = use_ifvg and fvg_valid and fvg_inverted and (not fvg_above_poi or needs_reclaim)

                if not use_ifvg:
                    signaled = intent_met
                else:
                    if ifvg_mode == 'IfvgOnly':
                        signaled = ifvg_met
                    elif ifvg_mode == 'Either':
                        signaled = intent_met or ifvg_met
                    elif ifvg_mode == 'Both':
                        signaled = intent_met and ifvg_met
        return signaled

    def _choose_poi(cands: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not cands:
            return None
        if selection_mode == 'MostRecent':
            return max(cands, key=lambda p: p['activation_htf_time'] or pd.Timestamp.min)
        # BestPrice: highest level for long, lowest for short
        best = cands[0]
        for p in cands[1:]:
            if p['is_long']:
                if p['level'] > best['level']:
                    best = p
            else:
                if p['level'] < best['level']:
                    best = p
        return best

    def _passes_pd(p: Dict[str, Any], i: int) -> bool:
        if not require_pd:
            return True
        if not range_valid:
            return False
        if p['is_long']:
            return c[i] <= range_eq + eq_tol
        else:
            return c[i] >= range_eq - eq_tol

    def _compute_qty(stop_dist_pts: float) -> int:
        if use_risk_sizing and stop_dist_pts > 0:
            risk_per_ctr = stop_dist_pts * point_value
            q = int(max_risk / risk_per_ctr) if risk_per_ctr > 0 else 0
            return min(max(q, 1), max_contracts)
        return min(qty_fixed, max_contracts)

    def _place_entry(p: Dict[str, Any], i: int) -> bool:
        nonlocal in_trade, trade_side, trade_entry, trade_stop, trade_target
        nonlocal trade_qty, trade_poi_id, trade_entry_ts

        if only_one_per_poi and p['has_been_traded']:
            return False
        if p['is_long'] and req_bullish_long and not p['is_bullish_bar']:
            return False
        if (not p['is_long']) and req_bearish_short and p['is_bullish_bar']:
            return False
        if not p['attempt_extreme_init']:
            return False

        entry_price = c[i]

        # ATR override
        if use_atr_stop_tgt:
            a = atr_1m[i] if atr_1m is not None else float('nan')
            if a is None or math.isnan(a) or a <= 0:
                return False
            if p['is_long']:
                stop_price   = entry_price - (atr_stop_mult * a)
                target_price = entry_price + (atr_target_mult * a)
            else:
                stop_price   = entry_price + (atr_stop_mult * a)
                target_price = entry_price - (atr_target_mult * a)
        else:
            # Stop
            if p['is_long']:
                if use_ifvg_stop and use_ifvg and p['has_fvg'] and not math.isnan(p['fvg_pattern_low']):
                    stop_price = p['fvg_pattern_low'] - tick_size
                elif use_alt_stop and p['has_open_below']:
                    stop_price = p['last_open_below_low'] - tick_size
                else:
                    stop_price = p['attempt_extreme'] - tick_size
            else:
                if use_ifvg_stop and use_ifvg and p['has_fvg'] and not math.isnan(p['fvg_pattern_high']):
                    stop_price = p['fvg_pattern_high'] + tick_size
                elif use_alt_stop and p['has_open_above']:
                    stop_price = p['last_open_above_high'] + tick_size
                else:
                    stop_price = p['attempt_extreme'] + tick_size

            # Target: RR or HTF
            target_price = p['target']
            if use_rr_target:
                if p['is_long']:
                    stop_dist = entry_price - stop_price
                    target_price = entry_price + (stop_dist * rr_mult)
                else:
                    stop_dist = stop_price - entry_price
                    target_price = entry_price - (stop_dist * rr_mult)

        # Geometry
        if p['is_long']:
            risk   = entry_price - stop_price
            reward = target_price - entry_price
        else:
            risk   = stop_price - entry_price
            reward = entry_price - target_price
        if risk <= 0 or reward <= 0:
            return False
        if (reward / risk) < min_rr:
            return False

        q = _compute_qty(risk)
        if q < 1:
            return False

        in_trade = True
        trade_side = 'Long' if p['is_long'] else 'Short'
        trade_entry = entry_price
        trade_stop  = stop_price
        trade_target = target_price
        trade_qty = q
        trade_poi_id = p['id']
        trade_entry_ts = idx[i]
        p['has_been_traded'] = True
        if use_ifvg and p['has_fvg']:
            p['fvg_has_been_inverted'] = True
        return True

    def _in_session(ts: pd.Timestamp) -> bool:
        if not enable_sess_window or sess_start_t is None or sess_end_t is None:
            return True
        now = ts.time()
        if sess_start_t <= sess_end_t:
            return sess_start_t <= now <= sess_end_t
        return now >= sess_start_t or now <= sess_end_t

    # ---- Main loop
    for i in range(n):
        ts = idx[i]
        cur_htf = bar_to_htf[i]

        # Process any newly-closed H1 bars (typically 0 or 1; could be >1 on data gaps)
        while last_processed_htf < cur_htf:
            last_processed_htf += 1
            _process_new_htf_bar(last_processed_htf)

        # Update P/D range from completed range bars
        if require_pd:
            cur_range = bar_to_range[i]
            if cur_range >= 0:
                _update_pd_range(cur_range)

        # In-trade: check stop/target intra-bar
        if in_trade:
            if trade_side == 'Long':
                if l[i] <= trade_stop:
                    _close_trade(trade_stop, ts, 'stop')
                elif h[i] >= trade_target:
                    _close_trade(trade_target, ts, 'target')
            else:
                if h[i] >= trade_stop:
                    _close_trade(trade_stop, ts, 'stop')
                elif l[i] <= trade_target:
                    _close_trade(trade_target, ts, 'target')

        # Session-close force-exit
        if enable_close_exit and sess_close_exit_t is not None and in_trade:
            if ts.time() >= sess_close_exit_t:
                _close_trade(c[i], ts, 'session_close')

        if in_trade:
            continue
        if not _in_session(ts):
            continue

        # Update per-POI LTF state + signal collection
        entry_signals: List[Dict[str, Any]] = []
        for p in pois:
            if not p['valid'] or not p['activated']:
                continue
            # Only run LTF AFTER activation H1 bar has closed (already ensured by bar_to_htf mapping)
            _update_intent_bar(p, i)
            if use_ifvg and p['phase'] in (0, 1):
                _detect_fvg(p, i)
            _update_alt_stop_bar(p, i)
            if _ltf_update_and_signal(p, i):
                entry_signals.append(p)

        # Apply P/D gate
        if require_pd and entry_signals:
            entry_signals = [p for p in entry_signals if _passes_pd(p, i)]

        if entry_signals:
            chosen = _choose_poi(entry_signals)
            if chosen is not None:
                _place_entry(chosen, i)

    # End-of-data: close any open trade
    if in_trade:
        _close_trade(c[-1], idx[-1], 'eod_data')

    return trades
