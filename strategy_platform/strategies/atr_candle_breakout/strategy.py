"""
ATR Candle Breakout — volatility breakout strategy.

Logic (ported from the MetaTrader EA "ATR Candle Breakout EA"):
  - Scan closed candles on the signal timeframe for unusually large candles.
  - A "signal candle" must be larger than ATR × ATR_Multiplier.
  - For longs: candle close must be near the candle high (within Close_Proximity_% of range).
  - For shorts: candle close must be near the candle low.
  - Optional filters:
      * Trend Filter: higher-TF MA direction (EMA/SMA/SMMA/LWMA)
      * MTF ATR Confirmation: higher-TF candle must also be large and same direction
      * Time Filter: trade only within server-time hour window; skip Fri/Mon edges
      * S/R Filter: avoid buying near resistance, selling near support
  - Position sizing: fixed risk per trade (money amount) / stop distance
  - Stop Loss / Take Profit: percentage of entry price
  - Optional Trailing Stop: activates after profit threshold, trails by step %

Data requirements:
  - Uses strategy_platform.data.loader (load_5m, load_1m, load_tick_bars, load_all_timeframes)
  - Supports time bars (5M, 15M, etc.), 1M bars (resampled), and N-tick bars
  - Instrument metadata (tick_size, tick_value, commission) from INSTRUMENT_META

Session / Bar handling:
  - Signal timeframe is configurable (e.g., '5T', '15T', '30T', '60T', etc.)
  - Higher timeframes for trend/MTF/ S/R are also configurable
  - All timestamps are timezone-naive ET (as stored in MySQL)

Port notes:
  - The EA uses "server time" for time filters; we map to ET-naive timestamps.
  - ATR is calculated via pandas rolling (Wilder's smoothing = EMA with alpha=1/period).
  - S/R levels: swing highs/lows over lookback bars on S/R detection timeframe.
      Zone width = ATR × S/R_zone_width_mult.
      A signal is blocked if its close falls within ±zone of a valid S/R level.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import time as time_t
from typing import Any, Dict, List, Optional, Tuple

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register
from strategy_platform.data.loader import load_all_timeframes, INSTRUMENT_META, load_5m
from strategy_platform.strategies.goldbot7.strategy import _summarise, _bootstrap_trades

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

MA_METHODS = ['EMA', 'SMA', 'SMMA', 'LWMA']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average (Wilder's smoothing = EMA with alpha=1/period)."""
    return series.ewm(alpha=1.0 / period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()


def _smma(series: pd.Series, period: int) -> pd.Series:
    """Smoothed MA (SMMA) — same as Wilder's EMA."""
    return _ema(series, period)


def _lwma(series: pd.Series, period: int) -> pd.Series:
    """Linear Weighted Moving Average."""
    weights = np.arange(1, period + 1)
    return series.rolling(window=period, min_periods=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )


def _compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    """
    Wilder's ATR (EMA of True Range with alpha=1/period).
    True Range = max(high - low, abs(high - prev_close), abs(low - prev_close)).
    """
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs()
    ], axis=1).max(axis=1)
    return _ema(tr, period)


def _ma(series: pd.Series, period: int, method: str) -> pd.Series:
    method = method.upper()
    if method == 'EMA':
        return _ema(series, period)
    elif method == 'SMA':
        return _sma(series, period)
    elif method == 'SMMA':
        return _smma(series, period)
    elif method == 'LWMA':
        return _lwma(series, period)
    else:
        raise ValueError(f"Unknown MA method: {method}")


def _parse_time(s: str) -> time_t:
    h, m = int(s.split(':')[0]), int(s.split(':')[1])
    return time_t(h, m)


def _round_tick(price: float, tick_size: float) -> float:
    return round(price / tick_size) * tick_size


def _compute_swing_levels(
    df: pd.DataFrame,
    lookback: int,
    min_touches: int,
    atr_series: pd.Series,
    zone_mult: float
) -> Tuple[List[Dict], List[Dict]]:
    """
    Detect swing highs (resistance) and swing lows (support) over lookback bars.
    A level is valid if price touched it at least min_touches times.
    Returns (support_levels, resistance_levels) where each is a list of dicts:
    { 'price': float, 'touches': int, 'zone': float }
    Zone = ATR_at_level * zone_mult (use most recent ATR).
    """
    highs = df['high'].to_numpy()
    lows = df['low'].to_numpy()
    atr_vals = atr_series.to_numpy()

    n = len(df)
    if n < lookback + 2:
        return [], []

    support_levels = []
    resistance_levels = []

    # Find swing points (simple fractal: higher/lower than neighbors within lookback)
    for i in range(lookback, n - lookback):
        # Swing high
        if highs[i] == highs[i-lookback:i+lookback+1].max():
            level_price = highs[i]
            # Count touches within zone
            zone = atr_vals[i] * zone_mult if not np.isnan(atr_vals[i]) else 0
            touches = np.sum(np.abs(highs[max(0,i-lookback):i+lookback+1] - level_price) <= zone)
            if touches >= min_touches:
                resistance_levels.append({'price': float(level_price), 'touches': int(touches), 'zone': float(zone)})
        # Swing low
        if lows[i] == lows[i-lookback:i+lookback+1].min():
            level_price = lows[i]
            zone = atr_vals[i] * zone_mult if not np.isnan(atr_vals[i]) else 0
            touches = np.sum(np.abs(lows[max(0,i-lookback):i+lookback+1] - level_price) <= zone)
            if touches >= min_touches:
                support_levels.append({'price': float(level_price), 'touches': int(touches), 'zone': float(zone)})

    # Deduplicate nearby levels (keep highest touches)
    def _dedup(levels: List[Dict]) -> List[Dict]:
        if not levels:
            return []
        levels = sorted(levels, key=lambda x: -x['touches'])
        merged = []
        for lvl in levels:
            if not any(abs(lvl['price'] - m['price']) <= max(lvl['zone'], m['zone']) for m in merged):
                merged.append(lvl)
        return merged

    return _dedup(support_levels), _dedup(resistance_levels)


def _is_near_sr(
    close: float,
    direction: str,  # 'long' or 'short'
    support: List[Dict],
    resistance: List[Dict]
) -> bool:
    """
    Check if a signal close is inside an S/R zone.
    Long blocked by resistance zone; Short blocked by support zone.
    """
    if direction == 'long':
        for r in resistance:
            if abs(close - r['price']) <= r['zone']:
                return True
    else:
        for s in support:
            if abs(close - s['price']) <= s['zone']:
                return True
    return False


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

@register
class ATRCandleBreakout(BaseStrategy):
    """
    ATR Candle Breakout: volatility breakout on large ATR-multiple candles.

    Parameters mirror the MetaTrader EA inputs:
      - Signal Timeframe, ATR Period, ATR Multiplier, Close Proximity %, Min Body %
      - Trend Filter (enable, TF, MA period, MA method)
      - MTF ATR Confirmation (enable, higher TF, ATR period, multiplier)
      - Time Filter (enable, start/end hour, Fri/Mon skip)
      - S/R Filter (enable, detection TF, lookback, zone width mult, min touches)
      - Risk: SL %, TP %, Risk per trade ($)
      - Trailing Stop (enable, activate %, step %)
    """

    name = "atr_candle_breakout"
    bar_type = 'time'  # default; overridden by dashboard for tick/1m
    supported_bar_types = ['time', '1m', 'tick']

    default_params: Dict[str, Any] = {
        # ---- Strategy Settings ----
        'signal_timeframe':        '5T',          # bar timeframe for signals ('5T','15T','30T','60T','240T')
        'atr_period':              14,            # ATR lookback on signal TF
        'atr_multiplier':          1.5,           # signal candle > ATR * this
        'close_proximity_pct':     20.0,          # close within X% of high/low (0-100)
        'min_body_to_range_pct':   0.0,           # min body/range % (0 = disabled)

        # ---- Trend Filter ----
        'enable_trend_filter':     False,
        'trend_timeframe':         '60T',
        'trend_ma_period':         50,
        'trend_ma_method':         'EMA',

        # ---- MTF ATR Confirmation ----
        'enable_mtf_atr':          False,
        'mtf_timeframe':           '60T',
        'mtf_atr_period':          14,
        'mtf_atr_multiplier':      1.0,

        # ---- Time Filter ----
        'enable_time_filter':      False,
        'trading_start_hour':      0,
        'trading_end_hour':        23,
        'skip_friday_early':       True,
        'skip_monday_late':        True,

        # ---- S/R Level Filter ----
        'enable_sr_filter':        False,
        'sr_timeframe':            '60T',
        'sr_lookback_bars':        100,
        'sr_zone_width_mult':      2.0,
        'sr_min_touches':          2,

        # ---- Risk Management ----
        'stop_loss_ticks':         20,            # stop loss in ticks
        'take_profit_ticks':       40,            # take profit in ticks
        'risk_per_trade':          300.0,         # fixed $ risk per trade
        'sl_by_atr':               False,         # if True, SL = ATR * sl_atr_mult (override stop_loss_ticks)
        'sl_atr_mult':             1.0,           # ATR multiplier for SL when sl_by_atr=True
        'tp_by_atr':               False,         # if True, TP = ATR * tp_atr_mult (override take_profit_ticks)
        'tp_atr_mult':             2.0,           # ATR multiplier for TP when tp_by_atr=True

        # ---- Trailing Stop ----
        'enable_trailing':         False,
        'trail_activate_ticks':    20,            # activate trailing after X ticks profit
        'trail_step_ticks':        10,            # trail distance in ticks

        # ---- General ----
        'direction':               'Both',        # 'Both', 'Long Only', 'Short Only'
    }

    # Instrument metadata — will be overridden by dashboard when symbol changes
    tick_size     = 0.10
    tick_value    = 10.00
    commission_rt = 4.62

    symbol  = 'GC=F'
    db_host: Optional[str] = None

    # -----------------------------------------------------------------------
    # Parameter grid / groups / display
    # -----------------------------------------------------------------------

    @property
    def param_grid(self) -> Dict[str, Any]:
        return {
            # Strategy Settings
            'signal_timeframe':        ['5T', '15T', '30T', '60T', '240T'],
            'atr_period':              (5, 30, 1),
            'atr_multiplier':          (0.5, 3.0, 0.1),
            'close_proximity_pct':     (5.0, 50.0, 5.0),
            'min_body_to_range_pct':   (0.0, 50.0, 5.0),

            # Trend Filter
            'enable_trend_filter':     [True, False],
            'trend_timeframe':         ['15T', '30T', '60T', '240T', 'D'],
            'trend_ma_period':         (10, 200, 10),
            'trend_ma_method':         MA_METHODS,

            # MTF ATR Confirmation
            'enable_mtf_atr':          [True, False],
            'mtf_timeframe':           ['15T', '30T', '60T', '240T', 'D'],
            'mtf_atr_period':          (5, 30, 1),
            'mtf_atr_multiplier':      (0.5, 3.0, 0.1),

            # Time Filter
            'enable_time_filter':      [True, False],
            'trading_start_hour':      (0, 23, 1),
            'trading_end_hour':        (0, 23, 1),
            'skip_friday_early':       [True, False],
            'skip_monday_late':        [True, False],

            # S/R Level Filter
            'enable_sr_filter':        [True, False],
            'sr_timeframe':            ['15T', '30T', '60T', '240T', 'D'],
            'sr_lookback_bars':        (20, 500, 20),
            'sr_zone_width_mult':      (0.5, 5.0, 0.5),
            'sr_min_touches':          (1, 5, 1),

            # Risk Management
            'stop_loss_ticks':         (5, 100, 5),
            'take_profit_ticks':       (10, 200, 10),
            'risk_per_trade':          (50.0, 2000.0, 50.0),
            'sl_by_atr':               [True, False],
            'sl_atr_mult':             (0.5, 3.0, 0.25),
            'tp_by_atr':               [True, False],
            'tp_atr_mult':             (0.5, 5.0, 0.25),

            # Trailing Stop
            'enable_trailing':         [True, False],
            'trail_activate_ticks':    (5, 100, 5),
            'trail_step_ticks':        (5, 50, 5),

            # General
            'direction':               ['Both', 'Long Only', 'Short Only'],
        }

    # Conditional params — hide dependent params when parent is False/off
    param_conditional: Dict[str, Tuple[str, Any]] = {
        # Trend filter deps
        'trend_timeframe':      ('enable_trend_filter', True),
        'trend_ma_period':      ('enable_trend_filter', True),
        'trend_ma_method':      ('enable_trend_filter', True),
        # MTF ATR deps
        'mtf_timeframe':        ('enable_mtf_atr', True),
        'mtf_atr_period':       ('enable_mtf_atr', True),
        'mtf_atr_multiplier':   ('enable_mtf_atr', True),
        # Time filter deps
        'trading_start_hour':   ('enable_time_filter', True),
        'trading_end_hour':     ('enable_time_filter', True),
        'skip_friday_early':    ('enable_time_filter', True),
        'skip_monday_late':     ('enable_time_filter', True),
        # S/R deps
        'sr_timeframe':         ('enable_sr_filter', True),
        'sr_lookback_bars':     ('enable_sr_filter', True),
        'sr_zone_width_mult':   ('enable_sr_filter', True),
        'sr_min_touches':       ('enable_sr_filter', True),
        # SL/TP ATR deps
        'sl_atr_mult':          ('sl_by_atr', True),
        'tp_atr_mult':          ('tp_by_atr', True),
        # Trailing deps
        'trail_activate_ticks': ('enable_trailing', True),
        'trail_step_ticks':     ('enable_trailing', True),
    }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            "1. Signal Settings": [
                'signal_timeframe', 'atr_period', 'atr_multiplier',
                'close_proximity_pct', 'min_body_to_range_pct'
            ],
            "2. Trend Filter": [
                'enable_trend_filter', 'trend_timeframe', 'trend_ma_period', 'trend_ma_method'
            ],
            "3. MTF ATR Confirmation": [
                'enable_mtf_atr', 'mtf_timeframe', 'mtf_atr_period', 'mtf_atr_multiplier'
            ],
            "4. Time Filter": [
                'enable_time_filter', 'trading_start_hour', 'trading_end_hour',
                'skip_friday_early', 'skip_monday_late'
            ],
            "5. S/R Level Filter": [
                'enable_sr_filter', 'sr_timeframe', 'sr_lookback_bars',
                'sr_zone_width_mult', 'sr_min_touches'
            ],
            "6. Risk Management": [
                'stop_loss_ticks', 'take_profit_ticks', 'risk_per_trade',
                'sl_by_atr', 'sl_atr_mult', 'tp_by_atr', 'tp_atr_mult'
            ],
            "7. Trailing Stop": [
                'enable_trailing', 'trail_activate_ticks', 'trail_step_ticks'
            ],
            "8. General": [
                'direction'
            ],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            'signal_timeframe':       'Signal Timeframe',
            'atr_period':             'ATR Period',
            'atr_multiplier':         'ATR Multiplier',
            'close_proximity_pct':    'Close Proximity %',
            'min_body_to_range_pct':  'Min Body-to-Range %',
            'enable_trend_filter':    'Enable Trend Filter',
            'trend_timeframe':        'Trend Timeframe',
            'trend_ma_period':        'Trend MA Period',
            'trend_ma_method':        'Trend MA Method',
            'enable_mtf_atr':         'Enable MTF ATR Confirmation',
            'mtf_timeframe':          'MTF Timeframe',
            'mtf_atr_period':         'MTF ATR Period',
            'mtf_atr_multiplier':     'MTF ATR Multiplier',
            'enable_time_filter':     'Enable Time Filter',
            'trading_start_hour':     'Trading Start Hour (ET)',
            'trading_end_hour':       'Trading End Hour (ET)',
            'skip_friday_early':      'Skip Friday Early',
            'skip_monday_late':       'Skip Monday Late',
            'enable_sr_filter':       'Enable S/R Filter',
            'sr_timeframe':           'S/R Detection Timeframe',
            'sr_lookback_bars':       'S/R Lookback Bars',
            'sr_zone_width_mult':     'S/R Zone Width (×ATR)',
            'sr_min_touches':         'S/R Min Touches',
            'stop_loss_ticks':        'Stop Loss (ticks)',
            'take_profit_ticks':      'Take Profit (ticks)',
            'risk_per_trade':         'Risk Per Trade ($)',
            'sl_by_atr':              'SL by ATR',
            'sl_atr_mult':            'SL ATR Multiplier',
            'tp_by_atr':              'TP by ATR',
            'tp_atr_mult':            'TP ATR Multiplier',
            'enable_trailing':        'Enable Trailing Stop',
            'trail_activate_ticks':   'Trail Activate (ticks)',
            'trail_step_ticks':       'Trail Step (ticks)',
            'direction':              'Direction',
        }

    @property
    def description(self) -> str:
        return ("ATR Candle Breakout: volatility breakout on candles exceeding ATR × multiplier. "
                "Optional trend, MTF, time, and S/R filters. Risk-based sizing.")

    # -----------------------------------------------------------------------
    # Backtest entry points
    # -----------------------------------------------------------------------

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main backtest entry. Loads required timeframes, computes indicators,
        runs single-pass simulation, returns metrics.
        """
        merged = {**self.default_params, **params}

        # Load all needed timeframes for this symbol
        # We need: signal_tf, trend_tf (if enabled), mtf_tf (if enabled), sr_tf (if enabled)
        tfs_needed = {merged['signal_timeframe']}
        if merged['enable_trend_filter']:
            tfs_needed.add(merged['trend_timeframe'])
        if merged['enable_mtf_atr']:
            tfs_needed.add(merged['mtf_timeframe'])
        if merged['enable_sr_filter']:
            tfs_needed.add(merged['sr_timeframe'])

        # Load multi-timeframe data
        all_tfs = load_all_timeframes(
            self.symbol,
            start=None, end=None,  # use full range; IS/OOS split happens upstream
            refresh=False,
            host=self.db_host
        )

        # Extract signal timeframe data
        signal_df = all_tfs.get(merged['signal_timeframe'])
        if signal_df is None or len(signal_df) < 50:
            return {'net_pnl': 0, 'total_trades': 0, 'win_rate': 0, 'sharpe': 0, 'max_drawdown': 0,
                    'trades': 0, 'mc_stability': 0, 'mc_sharpe_p5': np.nan, 'mc_pnl_p5': np.nan, 'mc_pnl_p50': np.nan}

        # Build indicator DataFrames for each needed TF
        tf_data = {}
        for tf in tfs_needed:
            df = all_tfs.get(tf)
            if df is None or len(df) < 50:
                continue
            df = df.copy()
            df['atr'] = _compute_atr(df, merged['atr_period'] if tf == merged['signal_timeframe']
                                        else (merged['mtf_atr_period'] if tf == merged['mtf_timeframe']
                                              else 14))
            if merged['enable_trend_filter'] and tf == merged['trend_timeframe']:
                df['trend_ma'] = _ma(df['close'], merged['trend_ma_period'], merged['trend_ma_method'])
            tf_data[tf] = df

        # Run simulation
        trades = self._run_simulation(tf_data, merged)

        # Total sessions for Sharpe padding (count unique dates in signal data)
        total_sessions = int(signal_df['close'].resample('D').last().count())
        stats = _summarise(trades, total_sessions=total_sessions)
        bs = _bootstrap_trades(trades, total_sessions=total_sessions)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return {**stats, **bs, 'total_trades': stats['trades'], 'trades': trades_df}

    def run_monte_carlo(
        self,
        prepared: pd.DataFrame,  # unused; we reload inside
        params: Dict[str, Any],
        n_sims: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Day-shuffle Monte Carlo on signal timeframe."""
        merged = {**self.default_params, **params}

        all_tfs = load_all_timeframes(self.symbol, host=self.db_host)
        signal_df = all_tfs.get(merged['signal_timeframe'])
        if signal_df is None or len(signal_df) < 50:
            return {'mc_stability': 0.0, 'mc_sharpe_p5': float('nan'),
                    'mc_pnl_p5': float('nan'), 'mc_pnl_p50': float('nan')}

        # Group by date for shuffling
        groups = [(d, grp) for d, grp in signal_df.groupby(signal_df.index.date)]
        rng = np.random.default_rng(seed)
        n = len(groups)

        net_pnls = []
        sharpes = []

        for _ in range(n_sims):
            order = rng.permutation(n)
            shuffled_df = pd.concat([groups[i][1] for i in order])

            # Rebuild multi-TF data for shuffled signal (simplified: just shuffle signal TF)
            # For full MTF consistency we'd need to shuffle all TFs together — skip for speed.
            # This approximates by shuffling only the signal TF entries.
            tf_data = {}
            for tf_key in ['5T', '15T', '30T', '60T', '240T', 'D']:
                if tf_key in all_tfs:
                    df = all_tfs[tf_key].copy()
                    df['atr'] = _compute_atr(df, merged['atr_period'])
                    if merged['enable_trend_filter'] and tf_key == merged['trend_timeframe']:
                        df['trend_ma'] = _ma(df['close'], merged['trend_ma_period'], merged['trend_ma_method'])
                    tf_data[tf_key] = df

            # Replace signal TF with shuffled version
            tf_data[merged['signal_timeframe']] = shuffled_df
            shuffled_df['atr'] = _compute_atr(shuffled_df, merged['atr_period'])
            if merged['enable_trend_filter'] and merged['signal_timeframe'] == merged['trend_timeframe']:
                shuffled_df['trend_ma'] = _ma(shuffled_df['close'], merged['trend_ma_period'], merged['trend_ma_method'])

            trades = self._run_simulation(tf_data, merged)
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
            'mc_sharpe_p5': float(np.percentile(sharpes, 5)),
            'mc_pnl_p5':    float(np.percentile(arr,     5)),
            'mc_pnl_p50':   float(np.percentile(arr,    50)),
        }

    # -----------------------------------------------------------------------
    # Core simulation
    # -----------------------------------------------------------------------

    def _run_simulation(
        self,
        tf_data: Dict[str, pd.DataFrame],
        params: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Single-pass backtest loop on signal timeframe.
        All other TF data is accessed via tf_data dict (aligned by timestamp).
        """
        signal_tf = params['signal_timeframe']
        df = tf_data.get(signal_tf)
        if df is None or len(df) < 50:
            return []

        # Extract arrays for speed
        idx = df.index
        open_a  = df['open'].to_numpy()
        high_a  = df['high'].to_numpy()
        low_a   = df['low'].to_numpy()
        close_a = df['close'].to_numpy()
        atr_a   = df['atr'].to_numpy()
        vol_a   = df['volume'].to_numpy() if 'volume' in df.columns else np.zeros(len(df))
        n = len(df)

        # Trend MA (if enabled)
        trend_ma = None
        if params['enable_trend_filter']:
            trend_tf = params['trend_timeframe']
            trend_df = tf_data.get(trend_tf)
            if trend_df is not None and 'trend_ma' in trend_df.columns:
                # Reindex trend MA to signal TF timestamps (forward fill)
                trend_ma = trend_df['trend_ma'].reindex(idx, method='ffill').to_numpy()

        # MTF ATR (if enabled)
        mtf_atr = None
        if params['enable_mtf_atr']:
            mtf_tf = params['mtf_timeframe']
            mtf_df = tf_data.get(mtf_tf)
            if mtf_df is not None and 'atr' in mtf_df.columns:
                mtf_atr = mtf_df['atr'].reindex(idx, method='ffill').to_numpy()

        # S/R levels (if enabled) — compute once on SR TF, then map to signal TF
        support_levels = []
        resistance_levels = []
        if params['enable_sr_filter']:
            sr_tf = params['sr_timeframe']
            sr_df = tf_data.get(sr_tf)
            if sr_df is not None and 'atr' in sr_df.columns:
                support_levels, resistance_levels = _compute_swing_levels(
                    sr_df,
                    params['sr_lookback_bars'],
                    params['sr_min_touches'],
                    sr_df['atr'],
                    params['sr_zone_width_mult']
                )

        # Param unpack
        atr_mult      = float(params['atr_multiplier'])
        proximity_pct = float(params['close_proximity_pct']) / 100.0
        min_body_pct  = float(params['min_body_to_range_pct']) / 100.0
        sl_pct        = float(params['stop_loss_pct']) / 100.0
        tp_pct        = float(params['take_profit_pct']) / 100.0
        risk_per_trade = float(params['risk_per_trade'])
        direction     = str(params['direction'])
        can_long  = direction in ('Both', 'Long Only', 'both', 'long_only')
        can_short = direction in ('Both', 'Short Only', 'both', 'short_only')

        # Time filter
        time_filter = params['enable_time_filter']
        start_hour = params['trading_start_hour'] if time_filter else 0
        end_hour   = params['trading_end_hour']   if time_filter else 23
        skip_fri   = params['skip_friday_early']  if time_filter else False
        skip_mon   = params['skip_monday_late']   if time_filter else False

        # Trailing
        enable_trail = params['enable_trailing']
        trail_act_pct = float(params['trail_activate_pct']) / 100.0
        trail_step_pct = float(params['trail_step_pct']) / 100.0

        point_value = self.tick_value / self.tick_size

        trades: List[Dict[str, Any]] = []
        in_trade = False
        trade_side = None
        trade_entry = 0.0
        trade_stop = 0.0
        trade_target = 0.0
        trade_qty = 0
        trade_entry_ts = None
        trail_active = False
        trail_price = 0.0

        for i in range(1, n):  # start at 1 so we have a "closed" bar at i-1
            ts = idx[i]
            bar_ts = idx[i-1]  # the closed candle we're evaluating

            # ---- Time filter ----
            if time_filter:
                # Skip if outside trading hours
                h = bar_ts.hour
                if h < start_hour or h >= end_hour:
                    continue
                # Skip Friday early (2h before end)
                if skip_fri and bar_ts.weekday() == 4 and h >= end_hour - 2:
                    continue
                # Skip Monday late (2h after start)
                if skip_mon and bar_ts.weekday() == 0 and h < start_hour + 2:
                    continue

            # ---- Force exit at end of data (handled after loop) ----

            # ---- Manage open trade ----
            if in_trade:
                # Check stop / target / trailing
                if trade_side == 'Long':
                    # Stop hit
                    if low_a[i] <= trade_stop:
                        pnl_pts = trade_stop - trade_entry
                        pnl_dollars = (pnl_pts / self.tick_size) * self.tick_value * trade_qty - self.commission_rt
                        trades.append(self._make_trade(bar_ts, 'long', trade_entry, trade_stop,
                                                       trade_stop, trade_target, trade_qty,
                                                       pnl_dollars, 'stop', trade_entry_ts))
                        in_trade = False
                        continue
                    # Target hit
                    if high_a[i] >= trade_target:
                        pnl_pts = trade_target - trade_entry
                        pnl_dollars = (pnl_pts / self.tick_size) * self.tick_value * trade_qty - self.commission_rt
                        trades.append(self._make_trade(bar_ts, 'long', trade_entry, trade_target,
                                                       trade_stop, trade_target, trade_qty,
                                                       pnl_dollars, 'target', trade_entry_ts))
                        in_trade = False
                        continue
                    # Trailing stop
                    if enable_trail:
                        if not trail_active:
                            # Activate when profit reaches trail_act_pct of entry
                            unrealized_pts = close_a[i] - trade_entry
                            unrealized_pct = unrealized_pts / trade_entry if trade_entry > 0 else 0
                            if unrealized_pct >= trail_act_pct:
                                trail_active = True
                                trail_price = close_a[i] - trail_step_pct * trade_entry
                        else:
                            # Trail up
                            new_trail = close_a[i] - trail_step_pct * trade_entry
                            if new_trail > trail_price:
                                trail_price = new_trail
                            # Check trail stop hit
                            if low_a[i] <= trail_price:
                                pnl_pts = trail_price - trade_entry
                                pnl_dollars = (pnl_pts / self.tick_size) * self.tick_value * trade_qty - self.commission_rt
                                trades.append(self._make_trade(bar_ts, 'long', trade_entry, trail_price,
                                                               trade_stop, trade_target, trade_qty,
                                                               pnl_dollars, 'trail', trade_entry_ts))
                                in_trade = False
                                trail_active = False
                                continue

                else:  # Short
                    # Stop hit
                    if high_a[i] >= trade_stop:
                        pnl_pts = trade_entry - trade_stop
                        pnl_dollars = (pnl_pts / self.tick_size) * self.tick_value * trade_qty - self.commission_rt
                        trades.append(self._make_trade(bar_ts, 'short', trade_entry, trade_stop,
                                                       trade_stop, trade_target, trade_qty,
                                                       pnl_dollars, 'stop', trade_entry_ts))
                        in_trade = False
                        continue
                    # Target hit
                    if low_a[i] <= trade_target:
                        pnl_pts = trade_entry - trade_target
                        pnl_dollars = (pnl_pts / self.tick_size) * self.tick_value * trade_qty - self.commission_rt
                        trades.append(self._make_trade(bar_ts, 'short', trade_entry, trade_target,
                                                       trade_stop, trade_target, trade_qty,
                                                       pnl_dollars, 'target', trade_entry_ts))
                        in_trade = False
                        continue
                    # Trailing stop
                    if enable_trail:
                        if not trail_active:
                            unrealized_pts = trade_entry - close_a[i]
                            unrealized_pct = unrealized_pts / trade_entry if trade_entry > 0 else 0
                            if unrealized_pct >= trail_act_pct:
                                trail_active = True
                                trail_price = close_a[i] + trail_step_pct * trade_entry
                        else:
                            new_trail = close_a[i] + trail_step_pct * trade_entry
                            if new_trail < trail_price:
                                trail_price = new_trail
                            if high_a[i] >= trail_price:
                                pnl_pts = trade_entry - trail_price
                                pnl_dollars = (pnl_pts / self.tick_size) * self.tick_value * trade_qty - self.commission_rt
                                trades.append(self._make_trade(bar_ts, 'short', trade_entry, trail_price,
                                                               trade_stop, trade_target, trade_qty,
                                                               pnl_dollars, 'trail', trade_entry_ts))
                                in_trade = False
                                trail_active = False
                                continue
                continue  # skip signal generation while in trade

            # ---- Generate new signal (on closed bar i-1) ----
            if np.isnan(atr_a[i-1]) or atr_a[i-1] <= 0:
                continue

            bar_range = high_a[i-1] - low_a[i-1]
            body_size = abs(close_a[i-1] - open_a[i-1])

            # 1. Candle size > ATR * multiplier
            if bar_range <= atr_a[i-1] * atr_mult:
                continue

            # 2. Body-to-range filter
            if min_body_pct > 0 and bar_range > 0:
                if (body_size / bar_range) < min_body_pct:
                    continue

            # 3. Close proximity to extreme
            is_bull = close_a[i-1] > open_a[i-1]
            is_bear = close_a[i-1] < open_a[i-1]

            if is_bull and can_long:
                # Close must be within proximity_pct of high
                prox = (high_a[i-1] - close_a[i-1]) / bar_range if bar_range > 0 else 1
                if prox > proximity_pct:
                    continue
                signal_dir = 'long'
            elif is_bear and can_short:
                prox = (close_a[i-1] - low_a[i-1]) / bar_range if bar_range > 0 else 1
                if prox > proximity_pct:
                    continue
                signal_dir = 'short'
            else:
                continue  # doji or wrong direction

            # 4. Trend filter
            if params['enable_trend_filter'] and trend_ma is not None:
                if i >= len(trend_ma) or np.isnan(trend_ma[i]):
                    continue
                trend_up = close_a[i-1] > trend_ma[i]
                trend_dn = close_a[i-1] < trend_ma[i]
                if signal_dir == 'long' and not trend_up:
                    continue
                if signal_dir == 'short' and not trend_dn:
                    continue

            # 5. MTF ATR confirmation
            if params['enable_mtf_atr'] and mtf_atr is not None:
                if i >= len(mtf_atr) or np.isnan(mtf_atr[i]):
                    continue
                # Higher TF candle must also be large and same direction
                # We approximate: check if MTF ATR * multiplier < current range (scaled)
                # Simpler: require MTF candle range > MTF_ATR * MTF_multiplier
                # Need MTF candle data — use the MTF close/open
                mtf_df = tf_data.get(params['mtf_timeframe'])
                if mtf_df is not None:
                    # Find MTF bar that contains this signal bar
                    mtf_idx = mtf_df.index.searchsorted(bar_ts, side='right') - 1
                    if mtf_idx > 0:
                        mtf_bar = mtf_df.iloc[mtf_idx]
                        mtf_range = mtf_bar['high'] - mtf_bar['low']
                        mtf_atr_val = mtf_atr[i]
                        if mtf_range <= mtf_atr_val * params['mtf_atr_multiplier']:
                            continue
                        # Direction alignment
                        mtf_bull = mtf_bar['close'] > mtf_bar['open']
                        mtf_bear = mtf_bar['close'] < mtf_bar['open']
                        if signal_dir == 'long' and not mtf_bull:
                            continue
                        if signal_dir == 'short' and not mtf_bear:
                            continue

            # 6. S/R filter
            if params['enable_sr_filter'] and (support_levels or resistance_levels):
                if _is_near_sr(close_a[i-1], signal_dir, support_levels, resistance_levels):
                    continue

            # ---- All filters passed — ENTER TRADE ----
            entry = _round_tick(close_a[i-1], self.tick_size)
            if signal_dir == 'long':
                sl = _round_tick(entry * (1 - sl_pct), self.tick_size)
                tp = _round_tick(entry * (1 + tp_pct), self.tick_size)
                sl_dist = entry - sl
            else:
                sl = _round_tick(entry * (1 + sl_pct), self.tick_size)
                tp = _round_tick(entry * (1 - tp_pct), self.tick_size)
                sl_dist = sl - entry

            if sl_dist <= 0:
                continue

            # Position sizing: risk_per_trade / (sl_dist * point_value)
            risk_per_ctr = sl_dist * point_value
            if risk_per_ctr <= 0:
                continue
            qty = int(risk_per_trade / risk_per_ctr)
            if qty <= 0:
                continue

            in_trade = True
            trade_side = signal_dir
            trade_entry = entry
            trade_stop = sl
            trade_target = tp
            trade_qty = qty
            trade_entry_ts = bar_ts
            trail_active = False
            trail_price = 0.0

        # Force exit any open trade at last bar
        if in_trade:
            last_i = n - 1
            exit_p = close_a[last_i]
            exit_ts = idx[last_i]
            if trade_side == 'long':
                pnl_pts = exit_p - trade_entry
            else:
                pnl_pts = trade_entry - exit_p
            pnl_dollars = (pnl_pts / self.tick_size) * self.tick_value * trade_qty - self.commission_rt
            trades.append(self._make_trade(exit_ts, trade_side, trade_entry, exit_p,
                                           trade_stop, trade_target, trade_qty,
                                           pnl_dollars, 'eod', trade_entry_ts))

        return trades

    def _make_trade(
        self,
        exit_ts: pd.Timestamp,
        direction: str,
        entry: float,
        exit_p: float,
        stop_p: float,
        target_p: float,
        qty: int,
        pnl: float,
        exit_reason: str,
        entry_time: pd.Timestamp
    ) -> Dict[str, Any]:
        return {
            'session_date': exit_ts.date(),
            'day_of_week': exit_ts.day_name(),
            'entry_time': entry_time,
            'exit_time': exit_ts,
            'direction': direction,
            'entry_price': entry,
            'exit_price': exit_p,
            'stop': stop_p,
            'target': target_p,
            'qty': qty,
            'pnl': pnl,
            'commission': self.commission_rt,
            'exit_reason': exit_reason,
        }