"""
WaeJurikPro — Waddah Attar Explosion V2 + Adaptive Jurik Filter strategy.

Logic:
  Waddah Attar Explosion (WAE V2) detects momentum bursts by comparing the
  change in an EMA differential against a Bollinger Band width (explosion)
  and an ATR-based dead zone. Directional signals require the trend histogram
  to exceed both the explosion and the dead zone, with a rising signal MA.

  The Adaptive Jurik Moving Average (JMA) acts as a trend filter. Entry is
  gated by whether price is above/below the JMA quantile bands (band filter)
  and/or whether JMA slope is favourable (slope/colour filter).

  Entry:  market open of bar i+1 (signal fires on bar i close).
  Exit:   fixed profit_ticks / stop_ticks bracket; EOD exit ~15:55 ET.
  Cooldown: bars_between_trades bars must elapse after exit before next entry.

Default instrument: MNQ (Micro Nasdaq-100, tick bars).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _is_eod_bar(ts: pd.Timestamp, bar_type: str) -> bool:
    """
    True for bars that fall in the EOD close window.

    Tick / 1M bars: UTC hour >= 21 covers the 15:55-16:00 ET window
    (21:00 UTC = 16:00 ET in winter, 20:00 UTC = 16:00 ET in summer).
    We use 20:55 UTC (hour==20, minute>=55) to match the CME close bar.
    5M time bars use the same hour==20, minute>=55 rule.
    """
    if bar_type in ('tick', '1m'):
        return ts.hour >= 21 or (ts.hour == 20 and ts.minute >= 55)
    # 5M / time bars: 20:55 UTC = 15:55 ET
    return ts.hour == 20 and ts.minute >= 55


def _in_session_gap(ts: pd.Timestamp) -> bool:
    """True for bars in the CME Globex daily settlement break (22:00-22:59 UTC = 17:00-17:59 ET)."""
    return ts.hour == 22


# ---------------------------------------------------------------------------
# WAE V2 indicator
# ---------------------------------------------------------------------------

def _wma_series(s: pd.Series, period: int) -> pd.Series:
    """Weighted moving average using linearly increasing weights."""
    weights = np.arange(1, period + 1, dtype=float)
    total   = weights.sum()
    return s.rolling(period).apply(lambda x: (x * weights).sum() / total, raw=True)


def _compute_wae(
    df: pd.DataFrame,
    fast_ma: int,
    slow_ma: int,
    signal_ma: int,
    bands_length: int,
    bands_dev: float,
    sensitive: int,
    dead_zone_period: int,
    atr_mult: float,
) -> pd.DataFrame:
    """
    Compute Waddah Attar Explosion V2 columns.

    Returns DataFrame with columns:
        trend_up, trend_dn, trend_up_signal, trend_dn_signal,
        explosion, dead_zone
    """
    close  = df['close']
    high   = df['high']
    low    = df['low']

    # EMA differential change scaled by sensitivity
    ema_fast = close.ewm(span=fast_ma, adjust=False).mean()
    ema_slow = close.ewm(span=slow_ma, adjust=False).mean()
    diff     = ema_fast - ema_slow
    trend    = (diff - diff.shift(1)) * sensitive

    trend_up = trend.clip(lower=0.0)
    trend_dn = (-trend).clip(lower=0.0)

    trend_up_signal = _wma_series(trend_up, signal_ma)
    trend_dn_signal = _wma_series(trend_dn, signal_ma)

    # Bollinger band width (explosion): 2 * dev * rolling_std(close, bands_length)
    explosion = 2.0 * bands_dev * close.rolling(bands_length).std(ddof=0)

    # ATR-based dead zone
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr      = tr.rolling(dead_zone_period).mean()
    dead_zone = atr * atr_mult

    return pd.DataFrame({
        'trend_up':         trend_up,
        'trend_dn':         trend_dn,
        'trend_up_signal':  trend_up_signal,
        'trend_dn_signal':  trend_dn_signal,
        'explosion':        explosion,
        'dead_zone':        dead_zone,
    }, index=df.index)


# ---------------------------------------------------------------------------
# JMA (Adaptive Jurik Moving Average) — stateful bar-by-bar loop
# ---------------------------------------------------------------------------

def _compute_jma(
    prices: np.ndarray,
    period: int,
    phase: float,
) -> np.ndarray:
    """
    Compute Jurik Moving Average approximation for an array of prices.

    The filter is stateful: e0, e1, e2, jma carry across bars.
    Must be computed in a scalar loop — no vectorised equivalent.

    Parameters
    ----------
    prices : 1-D float64 array
    period : JMA period (>= 1)
    phase  : JMA phase (-100 to +100, controls lag/overshoot balance)

    Returns
    -------
    jma_out : 1-D float64 array of same length as prices, NaN until warm-up.
    """
    n = len(prices)
    jma_out = np.full(n, np.nan, dtype=float)

    if n == 0:
        return jma_out

    # Fixed coefficients
    phase_ratio = max(-100.0, min(100.0, float(phase))) / 100.0 + 1.5

    denom = 0.45 * (period - 1)
    beta  = denom / (denom + 2.0) if denom > 0.0 else 0.0

    r    = math.sqrt(max(0.5 * (period - 1), 1.0))
    len1 = (math.log(r) / math.log(2.0) + 2.0) if r > 1.0 else 2.0
    pow1 = max(len1 - 2.0, 0.5)
    alpha = beta ** pow1

    # State variables — initialise to first price
    p0   = float(prices[0])
    e0   = p0
    e1   = 0.0
    e2   = 0.0
    jma  = p0

    for i in range(n):
        price = float(prices[i])

        e0_prev = e0
        e1_prev = e1
        e2_prev = e2
        jma_prev = jma

        e0 = (1.0 - beta) * price + beta * e0_prev
        e1 = (price - e0) * (1.0 - beta) + beta * e1_prev
        ma1 = e0 + phase_ratio * e1

        k  = 1.0 - alpha
        e2 = (ma1 - jma_prev) * k * k + alpha * alpha * e2_prev
        jma = jma_prev + e2

        jma_out[i] = jma

    return jma_out


# ---------------------------------------------------------------------------
# Heiken Ashi trend-biased extreme price
# ---------------------------------------------------------------------------

def _ha_trend_price(df: pd.DataFrame) -> np.ndarray:
    """
    Compute Heiken Ashi trend-biased extreme price for each bar.

    ha_close = (open + high + low + close) / 4
    ha_open  = (prev_ha_open + prev_ha_close) / 2   (init to ha_close[0])
    price    = max(ha_close, high) if ha_close >= ha_open else min(ha_close, low)
    """
    o = df['open'].values
    h = df['high'].values
    l = df['low'].values
    c = df['close'].values
    n = len(df)

    ha_close = (o + h + l + c) / 4.0
    ha_open  = np.empty(n, dtype=float)
    ha_open[0] = ha_close[0]
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0

    price = np.where(ha_close >= ha_open, np.maximum(ha_close, h), np.minimum(ha_close, l))
    return price


# ---------------------------------------------------------------------------
# Jurik filter indicator suite
# ---------------------------------------------------------------------------

def _compute_jurik_filter(
    df: pd.DataFrame,
    period: int,
    phase: float,
    levels_period: int,
    up_pct: float,
    down_pct: float,
) -> pd.DataFrame:
    """
    Compute JMA line, quantile bands, and colour (slope) for each bar.

    Returns DataFrame with columns:
        jma_line, jma_upper, jma_lower, jurik_green, jurik_red
    """
    prices = _ha_trend_price(df)
    jma_arr = _compute_jma(prices, period, phase)

    n = len(df)
    jma_upper = np.full(n, np.nan, dtype=float)
    jma_lower = np.full(n, np.nan, dtype=float)

    # Quantile bands: rolling window of levels_period JMA values
    for i in range(levels_period - 1, n):
        window = jma_arr[i - levels_period + 1: i + 1]
        if not np.any(np.isnan(window)):
            jma_upper[i] = np.percentile(window, up_pct,   interpolation='linear')
            jma_lower[i] = np.percentile(window, down_pct, interpolation='linear')

    # Slope / colour
    jurik_green = np.zeros(n, dtype=bool)
    jurik_red   = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not (math.isnan(jma_arr[i]) or math.isnan(jma_arr[i - 1])):
            jurik_green[i] = jma_arr[i] > jma_arr[i - 1]
            jurik_red[i]   = jma_arr[i] < jma_arr[i - 1]

    return pd.DataFrame({
        'jma_line':    jma_arr,
        'jma_upper':   jma_upper,
        'jma_lower':   jma_lower,
        'jurik_green': jurik_green,
        'jurik_red':   jurik_red,
    }, index=df.index)


# ---------------------------------------------------------------------------
# Trade record helper
# ---------------------------------------------------------------------------

def _record_trade(
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    direction: str,
    ep: float,
    xp: float,
    pnl_ticks: float,
    tick_value: float,
    commission: float,
    reason: str,
) -> Dict[str, Any]:
    return {
        'session_date': entry_time.date(),
        'day_of_week':  entry_time.day_name(),
        'entry_time':   entry_time,
        'exit_time':    exit_time,
        'direction':    direction,
        'entry_price':  ep,
        'exit_price':   xp,
        'pnl_ticks':    pnl_ticks,
        'pnl':          pnl_ticks * tick_value - commission,
        'exit_reason':  reason,
    }


# ---------------------------------------------------------------------------
# Backtest loop
# ---------------------------------------------------------------------------

def _run_backtest_loop(
    df: pd.DataFrame,
    wae: pd.DataFrame,
    jur: pd.DataFrame,
    params: Dict[str, Any],
    tick_size: float,
    tick_value: float,
    commission: float,
    bar_type: str = 'tick',
) -> List[Dict]:
    """
    Signal generation + trade simulation in a single pass.

    WAE signals use a one-shot re-arm mechanism:
      - buy_signal_allowed resets to True when trend_up <= explosion
      - fires once (sets allowed=False) and waits for reset before firing again
      - mirror logic for sell

    Entry at open of bar i+1 (next-bar entry, on_bar_close mode).
    Exit: TP / SL bracket checked against H/L of each bar.
    EOD: force-close at EOD bar; no new entries in session gap (22:xx UTC).
    """
    profit_tks    = int(params.get('profit_ticks',        40))
    stop_tks      = int(params.get('stop_ticks',          20))
    bars_cd       = int(params.get('bars_between_trades',  2))
    enable_longs  = bool(params.get('enable_longs',      True))
    enable_shorts = bool(params.get('enable_shorts',     True))

    enable_jbf    = bool(params.get('enable_jurik_band_filter',  True))
    enable_jsf    = bool(params.get('enable_jurik_slope_filter', False))

    enable_tf     = bool(params.get('enable_time_filter', False))
    t_start       = str(params.get('trade_start_time',  '09:30'))
    t_end         = str(params.get('trade_end_time',    '15:59'))

    pt = profit_tks * tick_size
    st = stop_tks   * tick_size

    o_arr  = df['open'].values
    h_arr  = df['high'].values
    l_arr  = df['low'].values
    idx    = df.index
    n      = len(df)

    tu_arr  = wae['trend_up'].values
    td_arr  = wae['trend_dn'].values
    tus_arr = wae['trend_up_signal'].values
    tds_arr = wae['trend_dn_signal'].values
    exp_arr = wae['explosion'].values
    dz_arr  = wae['dead_zone'].values

    jma_arr   = jur['jma_line'].values
    jupp_arr  = jur['jma_upper'].values
    jlo_arr   = jur['jma_lower'].values
    jgrn_arr  = jur['jurik_green'].values
    jred_arr  = jur['jurik_red'].values

    # Determine the warm-up length: enough bars for all indicators
    # WAE needs max(slow_ma, bands_length, dead_zone_period, signal_ma)
    # Jurik needs levels_period bars after JMA warms up
    wae_warm   = max(
        int(params.get('wae_slow_ma',          40)),
        int(params.get('wae_bands_length',      20)),
        int(params.get('wae_dead_zone_period',  14)),
        int(params.get('wae_signal_ma',         14)),
    )
    jur_warm   = int(params.get('jurik_levels_period', 35))
    min_bar    = wae_warm + jur_warm + 5

    trades: List[Dict] = []

    in_trade         = False
    ep = sl = tp     = None
    entry_time       = None
    direction        = None
    bars_since_exit  = bars_cd  # start ready

    # WAE re-arm state
    buy_signal_allowed  = True
    sell_signal_allowed = True

    for i in range(min_bar, n - 1):  # -1: need bar i+1 for entry
        ts = idx[i]

        # Session gap — exit and skip
        if _in_session_gap(ts):
            if in_trade:
                xp_    = o_arr[i]
                pnl_t  = (xp_ - ep if direction == 'long' else ep - xp_) / tick_size
                trades.append(_record_trade(
                    entry_time, ts, direction, ep, xp_, pnl_t, tick_value, commission, 'EOD'))
                in_trade = False
                bars_since_exit = 0
            continue

        # EOD exit — close at this bar's open, no new entries this bar
        if _is_eod_bar(ts, bar_type):
            if in_trade:
                xp_   = o_arr[i]
                pnl_t = (xp_ - ep if direction == 'long' else ep - xp_) / tick_size
                trades.append(_record_trade(
                    entry_time, ts, direction, ep, xp_, pnl_t, tick_value, commission, 'EOD'))
                in_trade = False
                bars_since_exit = 0
            continue

        # Manage open position
        if in_trade:
            closed = False
            if direction == 'long':
                if l_arr[i] <= sl:
                    pnl_t = (sl - ep) / tick_size
                    trades.append(_record_trade(
                        entry_time, ts, direction, ep, sl, pnl_t, tick_value, commission, 'SL'))
                    closed = True
                elif h_arr[i] >= tp:
                    pnl_t = (tp - ep) / tick_size
                    trades.append(_record_trade(
                        entry_time, ts, direction, ep, tp, pnl_t, tick_value, commission, 'TP'))
                    closed = True
            else:
                if h_arr[i] >= sl:
                    pnl_t = (ep - sl) / tick_size
                    trades.append(_record_trade(
                        entry_time, ts, direction, ep, sl, pnl_t, tick_value, commission, 'SL'))
                    closed = True
                elif l_arr[i] <= tp:
                    pnl_t = (ep - tp) / tick_size
                    trades.append(_record_trade(
                        entry_time, ts, direction, ep, tp, pnl_t, tick_value, commission, 'TP'))
                    closed = True
            if closed:
                in_trade = False
                bars_since_exit = 0
            else:
                # Still in trade; still update WAE re-arm state and skip signal check
                if tu_arr[i] <= exp_arr[i]:
                    buy_signal_allowed = True
                if td_arr[i] <= exp_arr[i]:
                    sell_signal_allowed = True
                continue

        # Cooldown
        bars_since_exit += 1
        if bars_since_exit <= bars_cd:
            # Still update re-arm state
            if tu_arr[i] <= exp_arr[i]:
                buy_signal_allowed = True
            if td_arr[i] <= exp_arr[i]:
                sell_signal_allowed = True
            continue

        # NaN guard
        if (math.isnan(exp_arr[i]) or math.isnan(dz_arr[i])
                or math.isnan(tu_arr[i]) or math.isnan(td_arr[i])
                or math.isnan(tus_arr[i]) or math.isnan(tds_arr[i])):
            continue

        # Time filter
        if enable_tf:
            bar_time = f"{ts.hour:02d}:{ts.minute:02d}"
            if bar_time < t_start or bar_time > t_end:
                # Update re-arm state even when filtered
                if tu_arr[i] <= exp_arr[i]:
                    buy_signal_allowed = True
                if td_arr[i] <= exp_arr[i]:
                    sell_signal_allowed = True
                continue

        tu_i   = tu_arr[i]
        td_i   = td_arr[i]
        tus_i  = tus_arr[i]
        tds_i  = tds_arr[i]
        exp_i  = exp_arr[i]
        dz_i   = dz_arr[i]
        tu_p   = tu_arr[i - 1]
        td_p   = td_arr[i - 1]
        exp_p  = exp_arr[i - 1]
        tus_p  = tus_arr[i - 1]
        tds_p  = tds_arr[i - 1]

        # Update WAE re-arm state BEFORE signal check
        if tu_i <= exp_i:
            buy_signal_allowed = True
        if td_i <= exp_i:
            sell_signal_allowed = True

        signal_fired = False

        # ── WAE BUY signal ─────────────────────────────────────────────────
        wae_buy = (
            buy_signal_allowed
            and tu_i > tu_p          # bright bar (rising)
            and tu_i > exp_i         # above explosion
            and tu_i > dz_i          # above dead zone
            and tus_i > exp_i        # signal MA above explosion
            and exp_i > dz_i         # explosion above dead zone
            and exp_i > exp_p        # explosion rising
            and tus_i > tus_p        # signal MA rising
        )

        if enable_longs and wae_buy:
            # Jurik band filter
            jbf_ok = True
            if enable_jbf:
                jbf_ok = (not math.isnan(jma_arr[i]) and not math.isnan(jupp_arr[i])
                          and jma_arr[i] > jupp_arr[i])
            # Jurik slope filter
            jsf_ok = True
            if enable_jsf:
                jsf_ok = bool(jgrn_arr[i])

            if jbf_ok and jsf_ok:
                next_ts   = idx[i + 1]
                ep        = o_arr[i + 1]
                entry_time = next_ts
                direction  = 'long'
                sl         = ep - st
                tp         = ep + pt
                in_trade   = True
                buy_signal_allowed = False   # re-arm required
                signal_fired = True

        # ── WAE SELL signal ────────────────────────────────────────────────
        if not signal_fired:
            wae_sell = (
                sell_signal_allowed
                and td_i > td_p          # bright bar (rising)
                and td_i > exp_i         # above explosion
                and td_i > dz_i          # above dead zone
                and tds_i > exp_i        # signal MA above explosion
                and exp_i > dz_i         # explosion above dead zone
                and exp_i > exp_p        # explosion rising
                and tds_i > tds_p        # signal MA rising
            )

            if enable_shorts and wae_sell:
                jbf_ok = True
                if enable_jbf:
                    jbf_ok = (not math.isnan(jma_arr[i]) and not math.isnan(jlo_arr[i])
                              and jma_arr[i] < jlo_arr[i])
                jsf_ok = True
                if enable_jsf:
                    jsf_ok = bool(jred_arr[i])

                if jbf_ok and jsf_ok:
                    next_ts    = idx[i + 1]
                    ep         = o_arr[i + 1]
                    entry_time = next_ts
                    direction  = 'short'
                    sl         = ep + st
                    tp         = ep - pt
                    in_trade   = True
                    sell_signal_allowed = False
                    signal_fired = True

    return trades


# ---------------------------------------------------------------------------
# Statistics (mirrors mobobands convention)
# ---------------------------------------------------------------------------

WEEKDAYS = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']


def _daily_sortino(d_vals: np.ndarray) -> float:
    if len(d_vals) < 2:
        return 0.0
    neg = d_vals[d_vals < 0]
    if len(neg) == 0:
        return float('inf')
    downside_std = float(np.sqrt(np.mean(neg ** 2)))
    return float(d_vals.mean() / downside_std * np.sqrt(252)) if downside_std > 0 else 0.0


def _max_consec(is_win: np.ndarray):
    max_w = max_l = cur_w = cur_l = 0
    for w in is_win:
        if w:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        if cur_w > max_w: max_w = cur_w
        if cur_l > max_l: max_l = cur_l
    return max_w, max_l


def _max_time_to_recover(dates: list, pnls: np.ndarray) -> float:
    equity   = np.cumsum(pnls)
    peak     = np.maximum.accumulate(equity)
    max_rec  = 0
    dd_start = None
    for i in range(len(equity)):
        if equity[i] < peak[i]:
            if dd_start is None:
                dd_start = dates[i - 1] if i > 0 else dates[i]
        else:
            if dd_start is not None:
                delta = (pd.Timestamp(dates[i]) - pd.Timestamp(dd_start)).days
                if delta > max_rec:
                    max_rec = delta
                dd_start = None
    if dd_start is not None:
        delta = (pd.Timestamp(dates[-1]) - pd.Timestamp(dd_start)).days
        if delta > max_rec:
            max_rec = delta
    return float(max_rec)


def _summarise(trades: List[Dict], total_sessions: int = 0) -> dict:
    _empty = {
        'trades': 0, 'win_rate': 0.0, 'net_pnl': 0.0,
        'avg_win': 0.0, 'avg_loss': 0.0, 'profit_factor': 0.0,
        'max_drawdown': 0.0, 'sharpe': 0.0, 'sortino': 0.0,
        'ulcer_index': 0.0, 'r_squared': 0.0, 'pct_months_profit': 0.0,
        'longest_flat_days': 0, 'total_commission': 0.0,
        'start_date': '', 'end_date': '',
        **{f'{d[:3].lower()}_pnl': 0.0  for d in WEEKDAYS},
        **{f'{d[:3].lower()}_trades': 0 for d in WEEKDAYS},
    }
    if not trades:
        return _empty

    pnls   = np.array([t['pnl'] for t in trades], dtype=float)
    n      = len(pnls)
    wins   = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    even   = pnls[pnls == 0]

    cum  = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    max_dd = float(-(peak - cum).max())

    daily_map: Dict[Any, float] = {}
    for t in trades:
        daily_map[t['session_date']] = daily_map.get(t['session_date'], 0.0) + t['pnl']
    sorted_dates = sorted(daily_map)
    d_vals = np.array([daily_map[d] for d in sorted_dates], dtype=float)

    n_zero = max(0, total_sessions - len(d_vals))
    d_vals_padded = np.concatenate([d_vals, np.zeros(n_zero)]) if n_zero > 0 else d_vals

    trade_dates_all = [t['session_date'] for t in trades]
    first_ym = (trade_dates_all[0].year,  trade_dates_all[0].month)
    last_ym  = (trade_dates_all[-1].year, trade_dates_all[-1].month)

    monthly_full: Dict[Any, float] = {}
    cur_ym = first_ym
    while cur_ym <= last_ym:
        monthly_full[cur_ym] = 0.0
        y, mo = cur_ym
        cur_ym = (y + 1, 1) if mo == 12 else (y, mo + 1)
    for t in trades:
        ym = (t['session_date'].year, t['session_date'].month)
        monthly_full[ym] = monthly_full.get(ym, 0.0) + t['pnl']

    m_arr = np.array(list(monthly_full.values()), dtype=float)
    if len(m_arr) >= 6:
        m_std  = float(m_arr.std(ddof=1))
        sharpe = float(m_arr.mean() / m_std) if m_std > 0 else 1.0
    else:
        std    = d_vals_padded.std(ddof=1) if len(d_vals_padded) > 1 else 0.0
        sharpe = float((d_vals_padded.mean() / std) * np.sqrt(252)) if std > 0 else 0.0

    max_consec_w, max_consec_l = _max_consec(pnls > 0)
    n_days             = len(daily_map)
    avg_trades_per_day = n / n_days if n_days > 0 else 0.0
    profit_per_month   = float(np.mean(list(monthly_full.values()))) if monthly_full else 0.0
    n_months_pos       = sum(1 for v in monthly_full.values() if v > 0)
    pct_months_profit  = float(n_months_pos / len(monthly_full)) if monthly_full else 0.0
    max_recovery       = _max_time_to_recover(trade_dates_all, pnls)
    gross_profit       = float(wins.sum())   if len(wins)   > 0 else 0.0
    gross_loss         = float(losses.sum()) if len(losses) > 0 else 0.0

    dd_series = peak - cum
    ulcer_idx = float(np.sqrt(np.mean(dd_series ** 2)))

    x = np.arange(n, dtype=float)
    try:
        with np.errstate(invalid='ignore', divide='ignore'):
            coef  = np.polyfit(x, cum, 1)
            y_hat = np.polyval(coef, x)
    except (np.linalg.LinAlgError, ValueError):
        coef  = np.array([0.0, 0.0])
        y_hat = np.zeros_like(cum)
    ss_res = float(((cum - y_hat) ** 2).sum())
    ss_tot = float(((cum - cum.mean()) ** 2).sum())
    r_sq   = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    high_dates = []
    peak_val   = -np.inf
    for i, val in enumerate(cum):
        if val > peak_val:
            peak_val = val
            high_dates.append(trade_dates_all[i])
    longest_flat = max(
        (high_dates[j + 1] - high_dates[j]).days for j in range(len(high_dates) - 1)
    ) if len(high_dates) >= 2 else 0

    stats: Dict[str, Any] = {
        'trades':              n,
        'num_wins':            int(len(wins)),
        'num_losses':          int(len(losses)),
        'num_even':            int(len(even)),
        'win_rate':            float(len(wins) / n),
        'gross_profit':        gross_profit,
        'gross_loss':          gross_loss,
        'net_pnl':             float(pnls.sum()),
        'avg_trade':           float(pnls.mean()),
        'avg_win':             float(wins.mean())   if len(wins)   > 0 else 0.0,
        'avg_loss':            float(losses.mean()) if len(losses) > 0 else 0.0,
        'ratio_win_loss':      float(wins.mean() / abs(losses.mean()))
                               if len(wins) and len(losses) else 0.0,
        'largest_win':         float(wins.max())    if len(wins)   > 0 else 0.0,
        'largest_loss':        float(losses.min())  if len(losses) > 0 else 0.0,
        'profit_factor':       gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf'),
        'max_drawdown':        max_dd,
        'max_consec_winners':  max_consec_w,
        'max_consec_losers':   max_consec_l,
        'avg_trades_per_day':  round(avg_trades_per_day, 2),
        'profit_per_month':    profit_per_month,
        'max_time_to_recover': max_recovery,
        'sharpe':              sharpe,
        'sortino':             _daily_sortino(d_vals_padded),
        'ulcer_index':         ulcer_idx,
        'r_squared':           r_sq,
        'pct_months_profit':   pct_months_profit,
        'longest_flat_days':   longest_flat,
        'total_commission':    0.0,
        'start_date':          str(trade_dates_all[0]),
        'end_date':            str(trade_dates_all[-1]),
    }
    for day in WEEKDAYS:
        key      = day[:3].lower()
        day_pnls = [t['pnl'] for t in trades if t['day_of_week'] == day]
        stats[f'{key}_pnl']    = float(sum(day_pnls))
        stats[f'{key}_trades'] = len(day_pnls)

    return stats


def _bootstrap_trades(
    trades: List[Dict],
    n_sims: int = 1000,
    seed: int = 42,
    total_sessions: int = 0,
) -> dict:
    _empty = {
        'bs_sharpe_p5': float('nan'), 'bs_pnl_p5':  float('nan'),
        'bs_pnl_p50':   float('nan'), 'bs_wr_p5':   float('nan'),
    }
    if len(trades) < 5:
        return _empty

    rng = np.random.default_rng(seed)
    daily_pnl: Dict[Any, float] = {}
    for t in trades:
        daily_pnl[t['session_date']] = daily_pnl.get(t['session_date'], 0.0) + t['pnl']
    pnls = np.array(list(daily_pnl.values()), dtype=float)

    n_zero = max(0, total_sessions - len(pnls))
    if n_zero > 0:
        pnls = np.concatenate([pnls, np.zeros(n_zero)])
    m = len(pnls)

    idx_    = rng.integers(0, m, size=(n_sims, m))
    s_pnls  = pnls[idx_]
    net_pnls  = s_pnls.sum(axis=1)
    win_rates = (s_pnls > 0).mean(axis=1)
    stds      = s_pnls.std(axis=1, ddof=1)
    means     = s_pnls.mean(axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        sharpes = np.where(stds > 0, (means / stds) * np.sqrt(252), 0.0)
    sharpes = np.nan_to_num(sharpes, nan=0.0)

    return {
        'bs_sharpe_p5': float(np.percentile(sharpes,    5)),
        'bs_pnl_p5':    float(np.percentile(net_pnls,   5)),
        'bs_pnl_p50':   float(np.percentile(net_pnls,  50)),
        'bs_wr_p5':     float(np.percentile(win_rates,  5)),
    }


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

@register
class WaeJurikPro(BaseStrategy):
    """
    WaeJurikPro: Waddah Attar Explosion V2 with Adaptive Jurik Filter.

    WAE V2 detects momentum breakouts above a Bollinger explosion threshold
    and ATR dead zone. The Jurik filter gates entries based on JMA quantile
    band position and/or slope direction. Default instrument: MNQ tick bars.
    """

    name = "WaeJurikPro"

    bar_type             = 'tick'
    supported_bar_types  = ['time', '1m', 'tick']

    default_params: Dict[str, Any] = {
        # Trade
        'profit_ticks':         40,
        'stop_ticks':           20,
        'bars_between_trades':  2,
        'enable_longs':         True,
        'enable_shorts':        True,
        # Time filter
        'enable_time_filter':   False,
        'trade_start_time':     '09:30',
        'trade_end_time':       '15:59',
        # WAE params
        'wae_fast_ma':          20,
        'wae_slow_ma':          40,
        'wae_signal_ma':        14,
        'wae_bands_length':     20,
        'wae_bands_dev':        4.0,
        'wae_sensitive':        150,
        'wae_dead_zone_period': 14,
        'wae_atr_mult':         4.0,
        # Jurik filter
        'enable_jurik_band_filter':  True,
        'enable_jurik_slope_filter': False,
        'jurik_period':         35,
        'jurik_phase':          0.0,
        'jurik_levels_period':  35,
        'jurik_up_pct':         90.0,
        'jurik_down_pct':       10.0,
        # Tick bar size (used when bar_type == 'tick')
        'tick_bar_size':        233,
    }

    # MNQ defaults — overridden by dashboard when symbol changes
    tick_size     = 0.25
    tick_value    = 0.50
    commission_rt = 0.50

    symbol  = 'MNQ'
    db_host = None

    # ------------------------------------------------------------------
    # param_grid
    # ------------------------------------------------------------------

    @property
    def param_grid(self) -> Dict[str, Any]:
        return {
            # WAE core
            'wae_fast_ma':          (5,  30,  5),
            'wae_slow_ma':          (20, 60,  5),
            'wae_signal_ma':        (5,  20,  5),
            'wae_bands_length':     (10, 40,  5),
            'wae_bands_dev':        (2.0, 6.0, 0.5),
            'wae_sensitive':        (50,  300, 50),
            'wae_dead_zone_period': (7,   21,  7),
            'wae_atr_mult':         (2.0, 6.0, 1.0),
            # Jurik
            'enable_jurik_band_filter':  [True, False],
            'enable_jurik_slope_filter': [True, False],
            'jurik_period':         (10, 60, 10),
            'jurik_phase':          (-50.0, 50.0, 25.0),
            'jurik_levels_period':  (10, 60, 10),
            'jurik_up_pct':         (70.0, 95.0, 5.0),
            'jurik_down_pct':       (5.0,  30.0, 5.0),
            # Trade
            'profit_ticks':         (20, 80, 10),
            'stop_ticks':           (10, 40,  5),
            'bars_between_trades':  (1,   5,  1),
            # Bools
            'enable_longs':         [True, False],
            'enable_shorts':        [True, False],
            'enable_time_filter':   [True, False],
            # Tick bar size
            'tick_bar_size':        [89, 144, 233, 377, 610],
        }

    param_conditional = {
        'jurik_period':              ('enable_jurik_band_filter', True),
        'jurik_phase':               ('enable_jurik_band_filter', True),
        'jurik_levels_period':       ('enable_jurik_band_filter', True),
        'jurik_up_pct':              ('enable_jurik_band_filter', True),
        'jurik_down_pct':            ('enable_jurik_band_filter', True),
        'enable_jurik_slope_filter': ('enable_jurik_band_filter', True),
        'trade_start_time':          ('enable_time_filter', True),
        'trade_end_time':            ('enable_time_filter', True),
    }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            'WAE Parameters': [
                'wae_fast_ma', 'wae_slow_ma', 'wae_signal_ma',
                'wae_bands_length', 'wae_bands_dev', 'wae_sensitive',
                'wae_dead_zone_period', 'wae_atr_mult',
            ],
            'Jurik Filter': [
                'enable_jurik_band_filter', 'enable_jurik_slope_filter',
                'jurik_period', 'jurik_phase',
                'jurik_levels_period', 'jurik_up_pct', 'jurik_down_pct',
            ],
            'Trade': [
                'profit_ticks', 'stop_ticks', 'bars_between_trades',
                'enable_longs', 'enable_shorts',
            ],
            'Time Filter': [
                'enable_time_filter', 'trade_start_time', 'trade_end_time',
            ],
            'Tick Bar': ['tick_bar_size'],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            'wae_fast_ma':               'WAE Fast MA',
            'wae_slow_ma':               'WAE Slow MA',
            'wae_signal_ma':             'Signal MA',
            'wae_bands_length':          'Bands Length',
            'wae_bands_dev':             'Bands Dev',
            'wae_sensitive':             'Sensitive',
            'wae_dead_zone_period':      'Dead Zone Period',
            'wae_atr_mult':              'ATR Mult',
            'enable_jurik_band_filter':  'Jurik Band Filter',
            'enable_jurik_slope_filter': 'Jurik Slope/Colour Filter',
            'jurik_period':              'Jurik Period',
            'jurik_phase':               'Jurik Phase',
            'jurik_levels_period':       'Levels Period',
            'jurik_up_pct':              'Upper Band %',
            'jurik_down_pct':            'Lower Band %',
            'profit_ticks':              'Profit Ticks',
            'stop_ticks':                'Stop Ticks',
            'bars_between_trades':       'Bars Between Trades',
            'enable_longs':              'Enable Longs',
            'enable_shorts':             'Enable Shorts',
            'enable_time_filter':        'Time Filter',
            'trade_start_time':          'Start Time',
            'trade_end_time':            'End Time',
            'tick_bar_size':             'Tick Bar Size',
        }

    @property
    def description(self) -> str:
        return (
            "Waddah Attar Explosion V2 momentum signals filtered by the "
            "Adaptive Jurik Moving Average on MNQ tick bars."
        )

    # ------------------------------------------------------------------
    # Core backtest
    # ------------------------------------------------------------------

    def _build_indicators(self, data: pd.DataFrame, params: Dict[str, Any]):
        """Compute WAE and Jurik indicators from OHLCV data and params."""
        wae = _compute_wae(
            data,
            fast_ma          = int(params.get('wae_fast_ma',         self.default_params['wae_fast_ma'])),
            slow_ma          = int(params.get('wae_slow_ma',         self.default_params['wae_slow_ma'])),
            signal_ma        = int(params.get('wae_signal_ma',       self.default_params['wae_signal_ma'])),
            bands_length     = int(params.get('wae_bands_length',    self.default_params['wae_bands_length'])),
            bands_dev        = float(params.get('wae_bands_dev',     self.default_params['wae_bands_dev'])),
            sensitive        = int(params.get('wae_sensitive',       self.default_params['wae_sensitive'])),
            dead_zone_period = int(params.get('wae_dead_zone_period',self.default_params['wae_dead_zone_period'])),
            atr_mult         = float(params.get('wae_atr_mult',      self.default_params['wae_atr_mult'])),
        )
        jur = _compute_jurik_filter(
            data,
            period        = int(params.get('jurik_period',        self.default_params['jurik_period'])),
            phase         = float(params.get('jurik_phase',       self.default_params['jurik_phase'])),
            levels_period = int(params.get('jurik_levels_period', self.default_params['jurik_levels_period'])),
            up_pct        = float(params.get('jurik_up_pct',      self.default_params['jurik_up_pct'])),
            down_pct      = float(params.get('jurik_down_pct',    self.default_params['jurik_down_pct'])),
        )
        return wae, jur

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single backtest on OHLCV data with the given parameter set."""
        wae, jur = self._build_indicators(data, params)

        current_bar_type = params.get('_bar_type', self.bar_type)
        trades = _run_backtest_loop(
            data, wae, jur, params,
            self.tick_size, self.tick_value, self.commission_rt,
            bar_type=current_bar_type,
        )

        total_sessions = int(data['close'].resample('D').last().count())
        stats  = _summarise(trades, total_sessions=total_sessions)
        bs     = _bootstrap_trades(trades, total_sessions=total_sessions)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        return {**stats, **bs, 'total_trades': stats['trades'], 'trades': trades_df}

    # ------------------------------------------------------------------
    # Monte Carlo
    # ------------------------------------------------------------------

    def run_monte_carlo(
        self,
        prepared: pd.DataFrame,
        params: Dict[str, Any],
        n_sims: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Day-shuffle Monte Carlo: permutes complete trading days and re-runs the strategy."""
        df     = prepared
        groups = [(date, grp) for date, grp in df.groupby(df.index.date)]
        rng    = np.random.default_rng(seed)
        n      = len(groups)

        current_bar_type = params.get('_bar_type', self.bar_type)
        net_pnls: List[float] = []
        sharpes:  List[float] = []

        for _ in range(n_sims):
            order       = rng.permutation(n)
            shuffled_df = pd.concat([groups[i][1] for i in order])
            wae, jur    = self._build_indicators(shuffled_df, params)
            trades      = _run_backtest_loop(
                shuffled_df, wae, jur, params,
                self.tick_size, self.tick_value, self.commission_rt,
                bar_type=current_bar_type,
            )
            stats = _summarise(trades)
            if stats.get('trades', 0) >= 5:
                net_pnls.append(stats['net_pnl'])
                sharpes.append(stats['sharpe'])

        if not net_pnls:
            return {
                'mc_stability': 0.0,
                'mc_sharpe_p5': float('nan'),
                'mc_pnl_p5':    float('nan'),
                'mc_pnl_p50':   float('nan'),
            }

        arr = np.array(net_pnls)
        return {
            'mc_stability': float((arr > 0).mean()),
            'mc_sharpe_p5': float(np.percentile(sharpes,  5)),
            'mc_pnl_p5':    float(np.percentile(arr,      5)),
            'mc_pnl_p50':   float(np.percentile(arr,     50)),
        }
