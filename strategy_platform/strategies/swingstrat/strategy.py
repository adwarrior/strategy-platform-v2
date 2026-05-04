"""
SwingStrat — Multi-timeframe SMC (Smart Money Concepts) strategy.

HTF (4H) swing leg detection with FVG-based entry on LTF (15min).
Supports two variants: swingstrat_15m and swingstrat_5m.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register

from .swing import compute_htf_legs, compute_fib_levels
from .fvg import detect_fvgs, is_fvg_invalidated

# ── Constants ──────────────────────────────────────────────────────────────

WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


# ── Statistics helpers ─────────────────────────────────────────────────────

def _daily_sortino(d_vals: np.ndarray) -> float:
    if len(d_vals) < 2:
        return 0.0
    mean = d_vals.mean()
    neg = d_vals[d_vals < 0]
    if len(neg) == 0:
        return float('inf')
    downside_std = np.sqrt(np.mean(neg ** 2))
    return float(mean / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0


def _max_consec(is_win: np.ndarray) -> Tuple[int, int]:
    max_w = max_l = cur_w = cur_l = 0
    for w in is_win:
        if w:
            cur_w += 1
            cur_l = 0
        else:
            cur_l += 1
            cur_w = 0
        if cur_w > max_w:
            max_w = cur_w
        if cur_l > max_l:
            max_l = cur_l
    return max_w, max_l


def _max_time_to_recover(dates: list, pnls: np.ndarray) -> float:
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    max_recovery = 0
    dd_start = None
    for i in range(len(equity)):
        if equity[i] < peak[i]:
            if dd_start is None:
                dd_start = dates[i - 1] if i > 0 else dates[i]
        else:
            if dd_start is not None:
                delta = (pd.Timestamp(dates[i]) - pd.Timestamp(dd_start)).days
                if delta > max_recovery:
                    max_recovery = delta
                dd_start = None
    if dd_start is not None:
        delta = (pd.Timestamp(dates[-1]) - pd.Timestamp(dd_start)).days
        if delta > max_recovery:
            max_recovery = delta
    return float(max_recovery)


def _summarise(trades: List[Dict[str, Any]], total_sessions: int = 0) -> dict:
    """
    Compute performance statistics from trades list.

    Follows goldbot6 pattern for consistency.
    """
    _empty: Dict[str, Any] = {
        'trades': 0, 'win_rate': 0.0, 'net_pnl': 0.0,
        'avg_win': 0.0, 'avg_loss': 0.0, 'profit_factor': 0.0,
        'max_drawdown': 0.0, 'sharpe': 0.0, 'sortino': 0.0,
        'ulcer_index': 0.0, 'r_squared': 0.0, 'pct_months_profit': 0.0,
        'longest_flat_days': 0, 'total_commission': 0.0,
        'start_date': '', 'end_date': '',
        **{f'{d[:3].lower()}_pnl': 0.0 for d in WEEKDAYS},
        **{f'{d[:3].lower()}_trades': 0 for d in WEEKDAYS},
    }
    if not trades:
        return _empty

    pnls = np.array([t['pnl'] for t in trades], dtype=float)
    n = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    even = pnls[pnls == 0]

    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    max_dd = float(-(peak - cum).max())

    daily_map: Dict = {}
    for t in trades:
        daily_map[t['session_date']] = daily_map.get(t['session_date'], 0.0) + t['pnl']
    sorted_dates = sorted(daily_map)
    d_vals = np.array([daily_map[d] for d in sorted_dates], dtype=float)

    n_zero = max(0, total_sessions - len(d_vals))
    d_vals_padded = np.concatenate([d_vals, np.zeros(n_zero)]) if n_zero > 0 else d_vals

    # Monthly Sharpe
    trade_dates_all = [t['session_date'] for t in trades]
    first_ym = (trade_dates_all[0].year, trade_dates_all[0].month)
    last_ym = (trade_dates_all[-1].year, trade_dates_all[-1].month)

    monthly_full: Dict = {}
    cur_ym = first_ym
    while cur_ym <= last_ym:
        monthly_full[cur_ym] = 0.0
        y, mo = cur_ym
        cur_ym = (y + 1, 1) if mo == 12 else (y, mo + 1)

    monthly_sums: Dict = {}
    for t in trades:
        ym = (t['session_date'].year, t['session_date'].month)
        monthly_sums[ym] = monthly_sums.get(ym, 0.0) + t['pnl']
    for ym in monthly_full:
        monthly_full[ym] = monthly_sums.get(ym, 0.0)

    m_arr = np.array(list(monthly_full.values()), dtype=float)
    if len(m_arr) >= 6:
        m_std = float(m_arr.std(ddof=1))
        sharpe = float(m_arr.mean() / m_std) if m_std > 0 else 1.0
    else:
        std = d_vals_padded.std(ddof=1) if len(d_vals_padded) > 1 else 0.0
        sharpe = float((d_vals_padded.mean() / std) * np.sqrt(252)) if std > 0 else 0.0

    max_consec_w, max_consec_l = _max_consec(pnls > 0)
    n_days = len(daily_map)
    avg_trades_per_day = n / n_days if n_days > 0 else 0.0
    profit_per_month = float(np.mean(list(monthly_full.values()))) if monthly_full else 0.0
    n_months_pos = sum(1 for v in monthly_full.values() if v > 0)
    pct_months_profit = float(n_months_pos / len(monthly_full)) if monthly_full else 0.0
    max_recovery = _max_time_to_recover(trade_dates_all, pnls)

    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(losses.sum()) if len(losses) > 0 else 0.0

    dd_series = peak - cum
    ulcer_idx = float(np.sqrt(np.mean(dd_series ** 2)))

    x = np.arange(n, dtype=float)
    coef = np.polyfit(x, cum, 1)
    y_hat = np.polyval(coef, x)
    ss_res = float(((cum - y_hat) ** 2).sum())
    ss_tot = float(((cum - cum.mean()) ** 2).sum())
    r_sq = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    high_dates: list = []
    peak_val = -np.inf
    for i, val in enumerate(cum):
        if val > peak_val:
            peak_val = val
            high_dates.append(trade_dates_all[i])
    longest_flat = max(
        (high_dates[i + 1] - high_dates[i]).days for i in range(len(high_dates) - 1)
    ) if len(high_dates) >= 2 else 0

    total_commission = float(sum(t.get('commission', 0.0) for t in trades))

    stats: Dict[str, Any] = {
        'trades': n,
        'num_wins': int(len(wins)),
        'num_losses': int(len(losses)),
        'num_even': int(len(even)),
        'win_rate': float(len(wins) / n) if n > 0 else 0.0,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'net_pnl': float(pnls.sum()),
        'avg_trade': float(pnls.mean()) if n > 0 else 0.0,
        'avg_win': float(wins.mean()) if len(wins) > 0 else 0.0,
        'avg_loss': float(losses.mean()) if len(losses) > 0 else 0.0,
        'ratio_win_loss': float(wins.mean() / abs(losses.mean()))
        if len(wins) and len(losses) else 0.0,
        'largest_win': float(wins.max()) if len(wins) > 0 else 0.0,
        'largest_loss': float(losses.min()) if len(losses) > 0 else 0.0,
        'profit_factor': gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf'),
        'max_drawdown': max_dd,
        'max_consec_winners': max_consec_w,
        'max_consec_losers': max_consec_l,
        'avg_trades_per_day': round(avg_trades_per_day, 2),
        'profit_per_month': profit_per_month,
        'max_time_to_recover': max_recovery,
        'sharpe': sharpe,
        'sortino': _daily_sortino(d_vals_padded),
        'ulcer_index': ulcer_idx,
        'r_squared': r_sq,
        'pct_months_profit': pct_months_profit,
        'longest_flat_days': longest_flat,
        'total_commission': total_commission,
        'start_date': str(trade_dates_all[0]),
        'end_date': str(trade_dates_all[-1]),
    }

    for day in WEEKDAYS:
        key = day[:3].lower()
        day_pnls = [t['pnl'] for t in trades if t['day_of_week'] == day]
        stats[f'{key}_pnl'] = float(sum(day_pnls))
        stats[f'{key}_trades'] = len(day_pnls)

    return stats


def _session_date(ts: pd.Timestamp) -> pd.Timestamp:
    """
    GC session_date: bar at or after 17:00 ET belongs to next day.
    """
    return (ts + pd.Timedelta(days=1)).date() if ts.hour >= 17 else ts.date()


# ── Backtest simulation ────────────────────────────────────────────────────

def _simulate_backtest(
    ltf_df: pd.DataFrame,
    htf_legs_available: pd.DataFrame,
    params: Dict[str, Any],
    tick_size: float,
    tick_value: float,
    commission_rt: float,
    ltf_minutes: int,
    htf_minutes: int,
) -> List[Dict[str, Any]]:
    """
    Simulate bar-by-bar on LTF with HTF leg state.

    State machine:
    - idle: no leg
    - leg_active: HTF leg active, waiting for fib_50 cross
    - scanning: price in zone, scanning for FVG
    - armed: FVG found or tier2 armed, limit order placed
    - in_trade: position open

    Parameters
    ----------
    ltf_df : pd.DataFrame
        LTF bars with DatetimeIndex.
    htf_legs_available : pd.DataFrame
        HTF legs with 'available_from' time (shifted by htf_minutes).
    params : dict
        Strategy parameters.
    tick_size : float
        Instrument tick size.
    tick_value : float
        Tick value in dollars.
    commission_rt : float
        Round-trip commission.
    ltf_minutes : int
        LTF bar interval in minutes.
    htf_minutes : int
        HTF bar interval in minutes.

    Returns
    -------
    list of trade dicts
    """
    # Parse params
    swing_period = params.get('swing_period', 10)
    fib_min = params.get('fib_min', 0.71)
    fib_max = params.get('fib_max', 0.75)
    fvg_proximity_ticks = params.get('fvg_proximity_ticks', 10)
    fvg_lookback_bars = params.get('fvg_lookback_bars', 20)
    sl_mode = params.get('sl_mode', 'SwingExtreme')
    enable_tier2 = params.get('enable_tier2', True)
    direction = params.get('direction', 'both')
    qty = int(params.get('qty', 1))

    can_long = direction in ('both', 'long_only')
    can_short = direction in ('both', 'short_only')

    trades: List[Dict[str, Any]] = []

    # State machine
    state = 'idle'
    current_leg: Optional[Dict[str, Any]] = None
    leg_id = -1
    fib_levels: Optional[Dict[str, float]] = None
    armed_entry: Optional[Dict[str, Any]] = None
    is_tier2 = False

    open_position: Optional[Dict[str, Any]] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None

    # Bar loop
    for idx, (ts, bar) in enumerate(ltf_df.iterrows()):
        close = float(bar['close'])
        high = float(bar['high'])
        low = float(bar['low'])

        # Get available HTF leg at this LTF bar time
        available_legs = htf_legs_available[htf_legs_available['available_from'] <= ts]
        if not available_legs.empty:
            latest_leg = available_legs.iloc[-1]
            new_leg_id = int(latest_leg['leg_id'])

            if new_leg_id != leg_id:
                # New leg detected
                leg_id = new_leg_id
                current_leg = {
                    'leg_id': new_leg_id,
                    'direction': latest_leg['direction'],
                    'origin': float(latest_leg['origin']),
                    'destination': float(latest_leg['destination']),
                }
                fib_levels = compute_fib_levels(current_leg, fib_min, fib_max)

                if open_position is None:
                    state = 'leg_active'
                    armed_entry = None
                    is_tier2 = False
                # else: stay in_trade, don't reset

        # ── Manage open position ─────────────────────────────────────────────
        if state == 'in_trade' and open_position is not None:
            is_long = open_position['is_long']
            reason = None
            exit_price = None

            # Check stop first (conservative: if both hit, SL wins)
            if is_long:
                if low <= sl_price:
                    reason = 'stop'
                    exit_price = sl_price
                elif high >= tp_price:
                    reason = 'target'
                    exit_price = tp_price
            else:  # short
                if high >= sl_price:
                    reason = 'stop'
                    exit_price = sl_price
                elif low <= tp_price:
                    reason = 'target'
                    exit_price = tp_price

            if reason is not None:
                pnl = _pnl_calc(open_position, exit_price, tick_size, tick_value, qty, commission_rt)
                trades.append({
                    'session_date': _session_date(ts),
                    'day_of_week': ts.day_name(),
                    'entry_time': open_position['entry_time'],
                    'exit_time': ts,
                    'direction': 'long' if is_long else 'short',
                    'entry': open_position['entry_price'],
                    'exit': exit_price,
                    'stop': sl_price,
                    'target': tp_price,
                    'pnl': pnl,
                    'commission': commission_rt,
                    'exit_reason': reason,
                    'is_tier2': open_position.get('is_tier2', False),
                })
                open_position = None
                armed_entry = None
                is_tier2 = False

                if reason == 'target':
                    state = 'idle'
                else:  # stop
                    state = 'leg_active' if current_leg else 'idle'

                continue

        # ── State machine transitions ──────────────────────────────────────────
        if state == 'idle':
            pass

        elif state == 'leg_active':
            # Waiting for fib_50 cross
            if current_leg is None or fib_levels is None:
                continue

            fib_50 = fib_levels['fib_50']
            direction_leg = current_leg['direction']

            if direction_leg == 'short' and not can_short:
                continue
            if direction_leg == 'long' and not can_long:
                continue

            if direction_leg == 'short' and close > fib_50:
                # Entered premium, move to scanning
                state = 'scanning'
            elif direction_leg == 'long' and close < fib_50:
                # Entered discount, move to scanning
                state = 'scanning'

        elif state == 'scanning':
            # Check if price exited zone
            if current_leg is None or fib_levels is None:
                state = 'leg_active'
                continue

            fib_50 = fib_levels['fib_50']
            fib_71 = fib_levels['fib_71']
            direction_leg = current_leg['direction']

            if direction_leg == 'short' and close < fib_50:
                # Exited zone downward, back to leg_active
                state = 'leg_active'
                armed_entry = None
                continue
            elif direction_leg == 'long' and close > fib_50:
                # Exited zone upward, back to leg_active
                state = 'leg_active'
                armed_entry = None
                continue

            # Scan for FVG
            if armed_entry is None:
                # Try to find FVG
                fvg = detect_fvgs(
                    ltf_df.iloc[:idx + 1],
                    direction_leg,
                    fib_71,
                    fib_levels['fib_75'],
                    fvg_lookback_bars,
                    fvg_proximity_ticks,
                    tick_size,
                )

                if fvg is not None:
                    armed_entry = fvg.copy()
                    armed_entry['armed_time'] = ts
                    is_tier2 = False
                    state = 'armed'
                elif enable_tier2 and direction_leg == 'short' and close >= fib_71:
                    # Tier 2: arm at fib_75 for SHORT when price reaches [fib_71, fib_75]
                    armed_entry = {
                        'entry': fib_levels['fib_75'],
                        'fvg_bottom': fib_71,
                        'fvg_top': fib_levels['fib_75'],
                        'armed_time': ts,
                        'fvg_type': 'tier2',
                    }
                    is_tier2 = True
                    state = 'armed'
                elif enable_tier2 and direction_leg == 'long' and close <= fib_71:
                    # Tier 2: arm at fib_75 for LONG when price reaches [fib_71, fib_75]
                    armed_entry = {
                        'entry': fib_levels['fib_75'],
                        'fvg_bottom': fib_levels['fib_75'],
                        'fvg_top': fib_71,
                        'armed_time': ts,
                        'fvg_type': 'tier2',
                    }
                    is_tier2 = True
                    state = 'armed'

        elif state == 'armed':
            # Check for invalidation or entry
            if current_leg is None or armed_entry is None or fib_levels is None:
                state = 'scanning'
                armed_entry = None
                is_tier2 = False
                continue

            fib_50 = fib_levels['fib_50']
            fib_71 = fib_levels['fib_71']
            direction_leg = current_leg['direction']
            entry_price = armed_entry['entry']

            # Check if exited zone
            if direction_leg == 'short' and close < fib_50:
                state = 'leg_active'
                armed_entry = None
                is_tier2 = False
                continue
            elif direction_leg == 'long' and close > fib_50:
                state = 'leg_active'
                armed_entry = None
                is_tier2 = False
                continue

            # Check if FVG invalidated
            if not is_tier2 and is_fvg_invalidated(close, armed_entry, direction_leg):
                state = 'scanning'
                armed_entry = None
                is_tier2 = False
                continue

            # Check for entry fill
            fill = False
            if direction_leg == 'short' and high >= entry_price:
                fill = True
            elif direction_leg == 'long' and low <= entry_price:
                fill = True

            if fill:
                # Entry!
                is_long = direction_leg == 'long'
                open_position = {
                    'entry_price': entry_price,
                    'entry_time': ts,
                    'is_long': is_long,
                    'is_tier2': is_tier2,
                }

                tp_price = fib_levels['tp']
                if sl_mode == 'SwingExtreme':
                    sl_price = fib_levels['sl']
                else:  # FVGInvalidation
                    if direction_leg == 'short':
                        sl_price = armed_entry['fvg_top'] + tick_size
                    else:
                        sl_price = armed_entry['fvg_bottom'] - tick_size

                state = 'in_trade'
                armed_entry = None
                is_tier2 = False

    return trades


def _pnl_calc(
    position: Dict[str, Any],
    exit_price: float,
    tick_size: float,
    tick_value: float,
    qty: int,
    commission: float,
) -> float:
    """Calculate PnL for a trade."""
    entry = position['entry_price']
    is_long = position['is_long']
    pts = (exit_price - entry) if is_long else (entry - exit_price)
    return (pts / tick_size) * tick_value * qty - commission


# ── Strategy Classes ───────────────────────────────────────────────────────

@register
class SwingStrat15M(BaseStrategy):
    """
    SwingStrat on 15-minute LTF bars with 240-minute (4H) HTF legs.

    GC Gold futures (GC=F from MySQL historical_data).
    """

    name = 'swingstrat_15m'
    bar_type = 'time'

    tick_size = 0.10
    tick_value = 10.00
    commission_rt = 4.62

    default_params = {
        'swing_period': 10,
        'fib_min': 0.71,
        'fib_max': 0.75,
        'fvg_proximity_ticks': 10,
        'fvg_lookback_bars': 20,
        'sl_mode': 'SwingExtreme',
        'enable_tier2': True,
        'direction': 'both',
        'qty': 1,
    }

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            'swing_period':        (3, 20, 1),
            'fib_min':             (0.50, 0.80, 0.05),
            'fib_max':             (0.65, 0.95, 0.05),
            'fvg_proximity_ticks': (2, 30, 2),
            'sl_mode':             ['SwingExtreme', 'FVGInvalidation'],
            'enable_tier2':        [True, False],
            'direction':           ['both', 'long_only', 'short_only'],
            'qty':                 (1, 4, 1),
        }

    @property
    def description(self) -> str:
        return (
            'SwingStrat: SMC strategy with 4H swing legs and 15M FVG entries on GC=F'
        )

    def prepare_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Resample 5m data to LTF (15m) and HTF (240m).

        Returns dict with 'ltf' and 'htf' DataFrames.
        """
        ltf_minutes = 15
        htf_minutes = 240

        ltf_df = (
            df.resample('15T', label='left', closed='left')
            .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            .dropna()
        )

        htf_df = (
            df.resample('240T', label='left', closed='left')
            .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            .dropna()
        )

        return {'ltf': ltf_df, 'htf': htf_df}

    def run_backtest_prepared(
        self,
        prepared: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run backtest on prepared (LTF, HTF) data."""
        ltf_df = prepared['ltf']
        htf_df = prepared['htf']

        if ltf_df.empty or htf_df.empty:
            return {
                'net_pnl': 0.0, 'total_trades': 0, 'win_rate': 0.0, 'sharpe': 0.0,
                'max_drawdown': 0.0, 'trades': pd.DataFrame(), 'equity_curve': pd.Series(),
            }

        # Compute HTF legs
        htf_legs = compute_htf_legs(htf_df, params.get('swing_period', 10))
        if htf_legs.empty:
            return {
                'net_pnl': 0.0, 'total_trades': 0, 'win_rate': 0.0, 'sharpe': 0.0,
                'max_drawdown': 0.0, 'trades': pd.DataFrame(), 'equity_curve': pd.Series(),
            }

        # Shift HTF legs index to account for no-lookahead: leg available AFTER the 4H bar closes
        htf_minutes = 240
        htf_legs['available_from'] = htf_legs.index + pd.Timedelta(minutes=htf_minutes)

        # Simulate backtest
        trades = _simulate_backtest(
            ltf_df, htf_legs, params,
            self.tick_size, self.tick_value, self.commission_rt,
            ltf_minutes=15, htf_minutes=htf_minutes,
        )

        # Compute stats
        stats = _summarise(trades, total_sessions=len(ltf_df) // 96)  # Rough estimate
        
        # Build equity curve
        if trades:
            trade_dates = [t['exit_time'] for t in trades]
            pnls = np.array([t['pnl'] for t in trades])
            equity = np.cumsum(pnls)
            equity_curve = pd.Series(equity, index=trade_dates)
        else:
            equity_curve = pd.Series(dtype=float)

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return {
            **stats,
            'total_trades': stats['trades'],
            'trades': trades_df,
            'equity_curve': equity_curve,
        }

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        prepared = self.prepare_data(data)
        return self.run_backtest_prepared(prepared, params)


@register
class SwingStrat5M(BaseStrategy):
    """
    SwingStrat on 5-minute LTF bars with 60-minute (1H) HTF legs.

    GC Gold futures (GC=F from MySQL historical_data).
    """

    name = 'swingstrat_5m'
    bar_type = 'time'

    tick_size = 0.10
    tick_value = 10.00
    commission_rt = 4.62

    default_params = {
        'swing_period': 10,
        'fib_min': 0.71,
        'fib_max': 0.75,
        'fvg_proximity_ticks': 10,
        'fvg_lookback_bars': 20,
        'sl_mode': 'SwingExtreme',
        'enable_tier2': True,
        'direction': 'both',
        'qty': 1,
    }

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            'swing_period':        (3, 20, 1),
            'fib_min':             (0.50, 0.80, 0.05),
            'fib_max':             (0.65, 0.95, 0.05),
            'fvg_proximity_ticks': (2, 30, 2),
            'sl_mode':             ['SwingExtreme', 'FVGInvalidation'],
            'enable_tier2':        [True, False],
            'direction':           ['both', 'long_only', 'short_only'],
            'qty':                 (1, 4, 1),
        }

    @property
    def description(self) -> str:
        return (
            'SwingStrat: SMC strategy with 1H swing legs and 5M FVG entries on GC=F'
        )

    def prepare_data(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Resample 5m data to LTF (5m) and HTF (60m).

        Returns dict with 'ltf' and 'htf' DataFrames.
        """
        ltf_minutes = 5
        htf_minutes = 60

        ltf_df = df.copy()  # Already 5m

        htf_df = (
            df.resample('60T', label='left', closed='left')
            .agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'})
            .dropna()
        )

        return {'ltf': ltf_df, 'htf': htf_df}

    def run_backtest_prepared(
        self,
        prepared: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run backtest on prepared (LTF, HTF) data."""
        ltf_df = prepared['ltf']
        htf_df = prepared['htf']

        if ltf_df.empty or htf_df.empty:
            return {
                'net_pnl': 0.0, 'total_trades': 0, 'win_rate': 0.0, 'sharpe': 0.0,
                'max_drawdown': 0.0, 'trades': pd.DataFrame(), 'equity_curve': pd.Series(),
            }

        # Compute HTF legs
        htf_legs = compute_htf_legs(htf_df, params.get('swing_period', 10))
        if htf_legs.empty:
            return {
                'net_pnl': 0.0, 'total_trades': 0, 'win_rate': 0.0, 'sharpe': 0.0,
                'max_drawdown': 0.0, 'trades': pd.DataFrame(), 'equity_curve': pd.Series(),
            }

        # Shift HTF legs index to account for no-lookahead
        htf_minutes = 60
        htf_legs['available_from'] = htf_legs.index + pd.Timedelta(minutes=htf_minutes)

        # Simulate backtest
        trades = _simulate_backtest(
            ltf_df, htf_legs, params,
            self.tick_size, self.tick_value, self.commission_rt,
            ltf_minutes=5, htf_minutes=htf_minutes,
        )

        # Compute stats
        stats = _summarise(trades, total_sessions=len(ltf_df) // 288)  # Rough estimate

        # Build equity curve
        if trades:
            trade_dates = [t['exit_time'] for t in trades]
            pnls = np.array([t['pnl'] for t in trades])
            equity = np.cumsum(pnls)
            equity_curve = pd.Series(equity, index=trade_dates)
        else:
            equity_curve = pd.Series(dtype=float)

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return {
            **stats,
            'total_trades': stats['trades'],
            'trades': trades_df,
            'equity_curve': equity_curve,
        }

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        prepared = self.prepare_data(data)
        return self.run_backtest_prepared(prepared, params)
