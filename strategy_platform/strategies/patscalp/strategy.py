"""
PATScalpHybrid — tick-bar scalping strategy on MNQ (Micro Nasdaq-100).

Ported from:
  NinjaScript source : /home/ad/Scripts/strategies/PATScalpHybrid
  Standalone optimizer: /home/ad/Scripts/PATScalp-Optimizer/

Bar type: N-tick OHLCV bars (bar_type = "tick").
tick_bar_size is an outer optimization axis — PATS signals are recomputed
once per bar size, then all inner parameter combos run against those signals.

Entry logic:
  - PATS long/short arrow fires on close of signal bar
  - Entry stop-order at prior bar's high (long) or low (short) ± entry_offset_ticks
  - Two-leg position: scalp leg (TP1) + runner leg (TP2 or trailing stop)
  - After TP1 hit: move runner stop to entry + be_plus_ticks

Filters available as grid parameters:
  ema_touch_mode, use_ema_side_gate, ema_slope_period, min_ema_slope_ticks,
  use_trend_structure, trend_swing_lookback, reject_inside_bars,
  reject_doji_bars, doji_max_body_pct, use_max_stop_filter, max_stop_ticks,
  use_adverse_cancel, adverse_threshold_pct, stop_anchor,
  enable_longs, enable_shorts
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import pandas as pd

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register


# ---------------------------------------------------------------------------
# PATS indicator (2-leg price-action structure)
# ---------------------------------------------------------------------------

def _compute_pats_signals(df: pd.DataFrame, tick_size: float = 0.25) -> pd.DataFrame:
    """
    Compute PATS long/short arrow signals over a DataFrame of OHLCV tick bars.

    Returns a DataFrame with columns long_signal (bool) and short_signal (bool),
    sharing the same index as df.
    """
    n      = len(df)
    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    opens  = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    ts     = tick_size

    def rt(p):
        return round(p / ts) * ts

    def gte_tick(a, b):
        return rt(a) >= rt(b) - 1e-10

    def lte_tick(a, b):
        return rt(a) <= rt(b) + 1e-10

    def eq_tick(a, b):
        return abs(rt(a) - rt(b)) < 0.5 * ts

    long_signal  = np.zeros(n, dtype=bool)
    short_signal = np.zeros(n, dtype=bool)

    # ── LONG state ──
    pivot_bar_l = -1; pivot_high_l = np.nan; cand_bar_l = -1; cand_high_l = np.nan
    pullback_active_l = False; leg1_done_l = False; leg1_bar_l = -1
    arrow_done_l = False; await_reset_l = False; bear_after_leg1_l = False
    last_pivot_bar_l = -1

    # ── SHORT state ──
    pivot_bar_s = -1; pivot_low_s = np.nan; cand_bar_s = -1; cand_low_s = np.nan
    pullback_active_s = False; leg1_done_s = False; leg1_bar_s = -1
    arrow_done_s = False; await_reset_s = False; bull_after_leg1_s = False
    last_pivot_bar_s = -1

    for i in range(2, n):
        hi = highs[i]; lo = lows[i]; op = opens[i]; cl = closes[i]
        hi1 = highs[i-1]; lo1 = lows[i-1]; cl1 = closes[i-1]; op1 = opens[i-1]

        # ══ LONGS ══
        did_reset_l = False
        if not await_reset_l and pivot_bar_l >= 0:
            if eq_tick(hi, pivot_high_l) or gte_tick(hi, pivot_high_l + ts):
                pivot_bar_l = i; pivot_high_l = hi; cand_bar_l = -1; cand_high_l = np.nan
                pullback_active_l = False; leg1_done_l = False; leg1_bar_l = -1
                arrow_done_l = False; await_reset_l = False; bear_after_leg1_l = False
                last_pivot_bar_l = i; did_reset_l = True

        if not did_reset_l and (await_reset_l or pivot_bar_l < 0):
            if cand_bar_l < 0 or eq_tick(hi, cand_high_l) or gte_tick(hi, cand_high_l + ts):
                cand_bar_l = i; cand_high_l = hi
            if cand_bar_l >= 0 and i > cand_bar_l:
                if lte_tick(lo, rt(lows[cand_bar_l] - ts)):
                    pivot_bar_l = cand_bar_l; pivot_high_l = highs[cand_bar_l]
                    cand_bar_l = -1; cand_high_l = np.nan
                    pullback_active_l = False; leg1_done_l = False; leg1_bar_l = -1
                    arrow_done_l = False; await_reset_l = False; bear_after_leg1_l = False
                    last_pivot_bar_l = pivot_bar_l; did_reset_l = True

        seeking_leg2_l = leg1_done_l and not arrow_done_l
        if (not did_reset_l and not await_reset_l and pivot_bar_l >= 0
                and (i > last_pivot_bar_l or seeking_leg2_l)):
            if not pullback_active_l:
                if i > last_pivot_bar_l and cl1 < op1:
                    pullback_active_l = True
            if pullback_active_l:
                up_trig = rt(hi1 + ts)
                broke_up = gte_tick(hi, up_trig)
                if not leg1_done_l and broke_up:
                    leg1_done_l = True; leg1_bar_l = i; bear_after_leg1_l = False
                if leg1_done_l and not arrow_done_l and (i-1) > leg1_bar_l and cl1 < op1:
                    bear_after_leg1_l = True
                if leg1_done_l and not arrow_done_l and bear_after_leg1_l and broke_up:
                    long_signal[i] = True; arrow_done_l = True; await_reset_l = True

        # ══ SHORTS ══
        did_reset_s = False
        if not await_reset_s and pivot_bar_s >= 0:
            if eq_tick(lo, pivot_low_s) or lte_tick(lo, pivot_low_s - ts):
                pivot_bar_s = i; pivot_low_s = lo; cand_bar_s = -1; cand_low_s = np.nan
                pullback_active_s = False; leg1_done_s = False; leg1_bar_s = -1
                arrow_done_s = False; await_reset_s = False; bull_after_leg1_s = False
                last_pivot_bar_s = i; did_reset_s = True

        if not did_reset_s and (await_reset_s or pivot_bar_s < 0):
            if cand_bar_s < 0 or eq_tick(lo, cand_low_s) or lte_tick(lo, cand_low_s - ts):
                cand_bar_s = i; cand_low_s = lo
            if cand_bar_s >= 0 and i > cand_bar_s:
                if gte_tick(hi, rt(highs[cand_bar_s] + ts)):
                    pivot_bar_s = cand_bar_s; pivot_low_s = lows[cand_bar_s]
                    cand_bar_s = -1; cand_low_s = np.nan
                    pullback_active_s = False; leg1_done_s = False; leg1_bar_s = -1
                    arrow_done_s = False; await_reset_s = False; bull_after_leg1_s = False
                    last_pivot_bar_s = pivot_bar_s; did_reset_s = True

        seeking_leg2_s = leg1_done_s and not arrow_done_s
        if (not did_reset_s and not await_reset_s and pivot_bar_s >= 0
                and (i > last_pivot_bar_s or seeking_leg2_s)):
            if not pullback_active_s:
                if i > last_pivot_bar_s and cl1 > op1:
                    pullback_active_s = True
            if pullback_active_s:
                dn_trig  = rt(lo1 - ts)
                broke_dn = lte_tick(lo, dn_trig)
                if not leg1_done_s and broke_dn:
                    leg1_done_s = True; leg1_bar_s = i; bull_after_leg1_s = False
                if leg1_done_s and not arrow_done_s and (i-1) > leg1_bar_s and cl1 > op1:
                    bull_after_leg1_s = True
                if leg1_done_s and not arrow_done_s and bull_after_leg1_s and broke_dn:
                    short_signal[i] = True; arrow_done_s = True; await_reset_s = True

    return pd.DataFrame(
        {"long_signal": long_signal, "short_signal": short_signal},
        index=df.index,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ema(series: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(series), np.nan)
    if len(series) < period:
        return out
    k = 2.0 / (period + 1)
    out[period - 1] = series[:period].mean()
    for i in range(period, len(series)):
        out[i] = series[i] * k + out[i-1] * (1 - k)
    return out


def _parse_time(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 3600 + int(m) * 60


def _in_window(dt: pd.Timestamp, start_s: int, end_s: int) -> bool:
    t = dt.hour * 3600 + dt.minute * 60 + dt.second
    if start_s <= end_s:
        return start_s <= t <= end_s
    return t >= start_s or t <= end_s


def _check_trend_structure(highs, lows, i, is_long, lookback):
    start = max(0, i - lookback)
    window_len = i - start + 1
    if window_len < 4:
        return True
    mid = start + window_len // 2
    earlier_hi = highs[start:mid].max(); earlier_lo = lows[start:mid].min()
    recent_hi  = highs[mid:i+1].max();  recent_lo  = lows[mid:i+1].min()
    if is_long:
        return recent_hi > earlier_hi and recent_lo > earlier_lo
    else:
        return recent_lo < earlier_lo and recent_hi < earlier_hi


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

def _run_backtest(
    df:         pd.DataFrame,
    signals:    pd.DataFrame,
    params:     Dict[str, Any],
    tick_size:  float,
    tick_value: float,
) -> Dict[str, Any]:
    ts = tick_size
    tv = tick_value

    highs  = df["high"].to_numpy()
    lows   = df["low"].to_numpy()
    opens  = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    times  = df.index

    ema_period  = int(params.get("ema_period", 21))
    ema_vals    = _ema(closes, ema_period)
    start_s     = _parse_time(params.get("trade_start", "09:30"))
    end_s       = _parse_time(params.get("trade_end",   "12:00"))

    tp1_ticks          = int(params.get("tp1_ticks", 4))
    tp2_ticks          = int(params.get("tp2_ticks", 12))
    entry_offset       = int(params.get("entry_offset_ticks", 1))
    stop_offset        = int(params.get("stop_offset_ticks",  1))
    be_plus            = int(params.get("be_plus_ticks", 0))
    ema_touch_mode     = params.get("ema_touch_mode", "off")
    use_ema_side_gate  = bool(params.get("use_ema_side_gate", False))
    use_trend_struct   = bool(params.get("use_trend_structure", False))
    trend_lookback     = int(params.get("trend_swing_lookback", 20))
    reject_inside      = bool(params.get("reject_inside_bars", False))
    reject_doji        = bool(params.get("reject_doji_bars", False))
    doji_pct           = int(params.get("doji_max_body_pct", 25))
    use_max_stop         = bool(params.get("use_max_stop_filter", False))
    max_stop_ticks       = int(params.get("max_stop_ticks", 12))
    use_adverse_cancel    = bool(params.get("use_adverse_cancel", False))
    adverse_threshold_pct = int(params.get("adverse_threshold_pct", 50))
    enable_longs          = bool(params.get("enable_longs", True))
    enable_shorts         = bool(params.get("enable_shorts", True))
    ema_slope_period      = int(params.get("ema_slope_period", 0))
    min_ema_slope_ticks_p = int(params.get("min_ema_slope_ticks", 0))
    stop_anchor           = params.get("stop_anchor", "prior_bar")

    def rt(p): return round(p / ts) * ts

    n = len(df)
    trades = []
    trade = None
    scalp_done = runner_done = False
    runner_stop = np.nan
    last_signal_bar = -1

    # entry / exit price
    entry_price = stop_price = tp1_price = tp2_price = np.nan
    direction = "long"
    entry_dt  = None

    for i in range(max(ema_period, 3), n):
        hi = highs[i]; lo = lows[i]; op = opens[i]; cl = closes[i]; dt = times[i]
        hi1 = highs[i-1]; lo1 = lows[i-1]; cl1 = closes[i-1]; op1 = opens[i-1]

        # ── Manage open trade ──
        if trade is not None:
            is_long = (direction == "long")

            if not scalp_done:
                if is_long:
                    if hi >= tp1_price:
                        trade.update(exit_dt_scalp=dt, exit_price_scalp=tp1_price, exit_reason_scalp="tp1")
                        scalp_done = True
                    elif lo <= stop_price:
                        trade.update(exit_dt_scalp=dt, exit_price_scalp=stop_price, exit_reason_scalp="stop")
                        scalp_done = True
                else:
                    if lo <= tp1_price:
                        trade.update(exit_dt_scalp=dt, exit_price_scalp=tp1_price, exit_reason_scalp="tp1")
                        scalp_done = True
                    elif hi >= stop_price:
                        trade.update(exit_dt_scalp=dt, exit_price_scalp=stop_price, exit_reason_scalp="stop")
                        scalp_done = True

                if scalp_done and trade["exit_reason_scalp"] == "stop" and not runner_done:
                    trade.update(exit_dt_runner=dt, exit_price_runner=stop_price, exit_reason_runner="stop")
                    runner_done = True

            if scalp_done and trade["exit_reason_scalp"] == "tp1" and not runner_done:
                if not trade.get("be_moved"):
                    be = rt(entry_price + (1 if is_long else -1) * be_plus * ts)
                    runner_stop = be
                    trade["be_moved"] = True

            if not runner_done:
                eff_stop = runner_stop if not np.isnan(runner_stop) else stop_price
                if is_long:
                    if hi >= tp2_price:
                        trade.update(exit_dt_runner=dt, exit_price_runner=tp2_price, exit_reason_runner="tp2")
                        runner_done = True
                    elif lo <= eff_stop:
                        trade.update(exit_dt_runner=dt, exit_price_runner=eff_stop, exit_reason_runner="stop")
                        runner_done = True
                else:
                    if lo <= tp2_price:
                        trade.update(exit_dt_runner=dt, exit_price_runner=tp2_price, exit_reason_runner="tp2")
                        runner_done = True
                    elif hi >= eff_stop:
                        trade.update(exit_dt_runner=dt, exit_price_runner=eff_stop, exit_reason_runner="stop")
                        runner_done = True

            if scalp_done and runner_done:
                # Compute PnL
                s_ticks = (trade["exit_price_scalp"] - entry_price) / ts
                r_ticks = (trade["exit_price_runner"] - entry_price) / ts
                if not is_long:
                    s_ticks = -s_ticks
                    r_ticks = -r_ticks
                scalp_pnl  = s_ticks * tv
                runner_pnl = r_ticks * tv
                trade["scalp_pnl"]  = scalp_pnl
                trade["runner_pnl"] = runner_pnl
                trade["total_pnl"]  = scalp_pnl + runner_pnl
                trades.append(trade)
                trade = None; scalp_done = runner_done = False; runner_stop = np.nan

        if trade is not None:
            continue

        if not _in_window(dt, start_s, end_s):
            continue

        long_sig  = enable_longs  and bool(signals["long_signal"].iloc[i])
        short_sig = enable_shorts and bool(signals["short_signal"].iloc[i])
        if not long_sig and not short_sig:
            continue
        if i == last_signal_bar:
            continue

        if use_trend_struct:
            if long_sig  and not _check_trend_structure(highs, lows, i, True,  trend_lookback):
                long_sig = False
            if short_sig and not _check_trend_structure(highs, lows, i, False, trend_lookback):
                short_sig = False

        if not long_sig and not short_sig:
            continue

        if ema_touch_mode != "off" and not np.isnan(ema_vals[i]):
            ema_now  = ema_vals[i]
            ema_prev = ema_vals[i-1]
            touch_signal = lows[i]   <= ema_now  and highs[i]   >= ema_now
            touch_prior  = lows[i-1] <= ema_prev and highs[i-1] >= ema_prev
            if ema_touch_mode == "signal_bar":
                pass_ema = touch_signal
            elif ema_touch_mode == "prior_bar":
                pass_ema = touch_prior
            else:
                pass_ema = touch_signal or touch_prior
            if not pass_ema:
                long_sig = short_sig = False

        if not long_sig and not short_sig:
            continue

        if use_ema_side_gate and not np.isnan(ema_vals[i]):
            ema_now = ema_vals[i]
            if long_sig  and cl < ema_now: long_sig  = False
            if short_sig and cl > ema_now: short_sig = False

        if ema_slope_period > 0 and min_ema_slope_ticks_p > 0:
            if i > ema_slope_period and not np.isnan(ema_vals[i - ema_slope_period]):
                slope = ema_vals[i] - ema_vals[i - ema_slope_period]
                min_slope = min_ema_slope_ticks_p * ts
                if long_sig  and slope <  min_slope:  long_sig  = False
                if short_sig and slope > -min_slope: short_sig = False
            else:
                long_sig = short_sig = False

        if reject_inside and i >= 2:
            if hi <= highs[i-1] and lo >= lows[i-1]:
                long_sig = short_sig = False

        if reject_doji:
            bar_range = max(hi - lo, ts)
            body = abs(cl - op)
            if (body / bar_range) * 100 <= doji_pct:
                long_sig = short_sig = False

        if not long_sig and not short_sig:
            continue
        if long_sig and short_sig:
            continue

        is_long = long_sig
        entry_price = rt(hi1 + entry_offset * ts) if is_long else rt(lo1 - entry_offset * ts)
        if stop_anchor == "signal_bar":
            stop_price = rt(lows[i] - stop_offset * ts) if is_long else rt(highs[i] + stop_offset * ts)
        else:
            stop_price = rt(lo1 - stop_offset * ts) if is_long else rt(hi1 + stop_offset * ts)
        stop_dist   = abs(entry_price - stop_price)

        if use_max_stop and stop_dist > max_stop_ticks * ts:
            continue

        if use_adverse_cancel and stop_dist > 0:
            adverse_dist = (adverse_threshold_pct / 100.0) * stop_dist
            if is_long     and (entry_price - lo) >= adverse_dist:
                continue
            if not is_long and (hi - entry_price) >= adverse_dist:
                continue

        tp1_price = rt(entry_price + tp1_ticks * ts) if is_long else rt(entry_price - tp1_ticks * ts)
        tp2_price = rt(entry_price + tp2_ticks * ts) if is_long else rt(entry_price - tp2_ticks * ts)

        if is_long and hi >= tp1_price:
            continue
        if not is_long and lo <= tp1_price:
            continue
        if is_long and hi < entry_price:
            continue
        if not is_long and lo > entry_price:
            continue

        direction = "long" if is_long else "short"
        entry_dt  = dt
        trade = {
            "entry_dt": dt, "direction": direction,
            "entry_price": entry_price, "stop_price": stop_price,
            "tp1_price": tp1_price, "tp2_price": tp2_price,
            "exit_dt_scalp": None, "exit_price_scalp": None, "exit_reason_scalp": "",
            "exit_dt_runner": None, "exit_price_runner": None, "exit_reason_runner": "",
            "be_moved": False,
        }
        runner_stop = stop_price
        scalp_done = runner_done = False
        last_signal_bar = i

    if not trades:
        return {
            "net_pnl": 0.0, "total_trades": 0, "win_rate": 0.0,
            "sharpe": 0.0, "max_drawdown": 0.0, "profit_factor": 0.0,
            "avg_trade": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "trades": pd.DataFrame(),
        }

    trades_df = pd.DataFrame(trades)
    pnl = trades_df["total_pnl"]
    wins = pnl > 0

    eq = pnl.cumsum()
    peak = eq.cummax()
    max_dd = float((eq - peak).min())

    sharpe = float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 0 else 0.0
    pf = float(pnl[wins].sum() / abs(pnl[~wins].sum())) if (~wins).any() and pnl[~wins].sum() != 0 else float("inf")

    # Normalise column names for the platform contract
    trades_df = trades_df.rename(columns={
        "entry_dt": "entry_time",
        "exit_dt_scalp": "exit_time",
        "exit_price_scalp": "exit_price",
        "total_pnl": "pnl",
    })
    trades_df["pnl_ticks"] = trades_df["pnl"] / tick_value

    return {
        "net_pnl":      float(pnl.sum()),
        "gross_profit": float(pnl[wins].sum()) if wins.any() else 0.0,
        "gross_loss":   float(pnl[~wins].sum()) if (~wins).any() else 0.0,
        "total_trades": len(pnl),
        "num_wins":     int(wins.sum()),
        "num_losses":   int((~wins).sum()),
        "win_rate":     float(wins.mean()),
        "avg_trade":    float(pnl.mean()),
        "avg_win":      float(pnl[wins].mean()) if wins.any() else 0.0,
        "avg_loss":     float(pnl[~wins].mean()) if (~wins).any() else 0.0,
        "profit_factor": pf,
        "max_drawdown": abs(max_dd),
        "sharpe":       sharpe,
        "trades":       trades_df,
    }


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------

@register
class PATScalpHybrid(BaseStrategy):
    """
    PATScalpHybrid — tick-bar PATS scalping strategy on MNQ.

    bar_type = "tick": the pipeline loads N-tick OHLCV bars from emini.tick_data.
    tick_bar_size is an outer optimization axis; PATS signals are recomputed
    once per bar size inside prepare_data().
    """

    name      = "patscalp"
    bar_type  = "tick"
    symbol    = "MNQ"

    tick_size     = 0.25
    tick_value    = 0.50
    commission_rt = 0.50

    # Defaults match NinjaTrader PATScalpHybrid SetDefaults exactly.
    default_params: Dict[str, Any] = {
        "tick_bar_size":         1300,
        "tp1_ticks":             4,
        "tp2_ticks":             12,
        "entry_offset_ticks":    1,
        "stop_offset_ticks":     1,
        "stop_anchor":           "prior_bar",
        "be_plus_ticks":         0,
        "ema_period":            21,
        "ema_touch_mode":        "either",
        "use_ema_side_gate":     True,
        "ema_slope_period":      8,
        "min_ema_slope_ticks":   2,
        "use_trend_structure":   True,
        "trend_swing_lookback":  20,
        "reject_inside_bars":    True,
        "reject_doji_bars":      True,
        "doji_max_body_pct":     25,
        "use_max_stop_filter":   True,
        "max_stop_ticks":        12,
        "use_adverse_cancel":    True,
        "adverse_threshold_pct": 50,
        "enable_longs":          True,
        "enable_shorts":         True,
        "trade_start":           "09:30",
        "trade_end":             "12:00",
    }

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        return {
            # ── Bar & entry ──────────────────────────────────────────────────
            "tick_bar_size":         (100, 3000, 100),
            "tp1_ticks":             (2, 20, 2),
            "tp2_ticks":             (6, 40, 2),
            "entry_offset_ticks":    (0, 4, 1),
            "stop_offset_ticks":     (0, 4, 1),
            "stop_anchor":           ["prior_bar", "signal_bar"],
            "be_plus_ticks":         (0, 6, 1),
            # ── EMA filters ──────────────────────────────────────────────────
            "ema_touch_mode":        ["off", "either", "signal_bar", "prior_bar"],
            "use_ema_side_gate":     [True, False],
            "ema_slope_period":      (0, 20, 2),
            "min_ema_slope_ticks":   (0, 6, 1),
            # ── Signal quality filters ────────────────────────────────────────
            "reject_inside_bars":    [False, True],
            "reject_doji_bars":      [False, True],
            "doji_max_body_pct":     (10, 50, 5),
            "use_trend_structure":   [True, False],
            "trend_swing_lookback":  (5, 40, 5),
            # ── Stop & risk filters ───────────────────────────────────────────
            "use_max_stop_filter":   [False, True],
            "max_stop_ticks":        (4, 30, 2),
            "use_adverse_cancel":    [False, True],
            "adverse_threshold_pct": (10, 90, 10),
            # ── Direction ────────────────────────────────────────────────────
            "enable_longs":          [True, False],
            "enable_shorts":         [True, False],
        }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            "Bar & Entry":      ["tick_bar_size", "tp1_ticks", "tp2_ticks", "entry_offset_ticks", "stop_offset_ticks", "stop_anchor", "be_plus_ticks"],
            "EMA Filters":      ["ema_period", "ema_touch_mode", "use_ema_side_gate", "ema_slope_period", "min_ema_slope_ticks"],
            "Signal Quality":   ["use_trend_structure", "trend_swing_lookback", "reject_inside_bars", "reject_doji_bars", "doji_max_body_pct"],
            "Stop & Risk":      ["use_max_stop_filter", "max_stop_ticks", "use_adverse_cancel", "adverse_threshold_pct"],
            "Direction":        ["enable_longs", "enable_shorts"],
            "Session Timing":   ["trade_start", "trade_end"],
        }

    @property
    def display_names(self) -> Dict[str, str]:
        return {
            "tick_bar_size":         "Tick Bar Size",
            "tp1_ticks":             "TP1 (ticks)",
            "tp2_ticks":             "TP2 (ticks)",
            "entry_offset_ticks":    "Entry Offset (ticks)",
            "stop_offset_ticks":     "Stop Offset (ticks)",
            "stop_anchor":           "Stop Anchor",
            "be_plus_ticks":         "BE+ (ticks)",
            "ema_period":            "EMA Period",
            "ema_touch_mode":        "EMA Touch Mode",
            "use_ema_side_gate":     "EMA Side Gate",
            "ema_slope_period":      "EMA Slope Period",
            "min_ema_slope_ticks":   "Min EMA Slope (ticks)",
            "use_trend_structure":   "Trend Structure",
            "trend_swing_lookback":  "Trend Swing Lookback",
            "reject_inside_bars":    "Reject Inside Bars",
            "reject_doji_bars":      "Reject Doji Bars",
            "doji_max_body_pct":     "Doji Max Body %",
            "use_max_stop_filter":   "Max Stop Filter",
            "max_stop_ticks":        "Max Stop (ticks)",
            "use_adverse_cancel":    "Adverse Cancel",
            "adverse_threshold_pct": "Adverse Threshold %",
            "enable_longs":          "Enable Longs",
            "enable_shorts":         "Enable Shorts",
            "trade_start":           "Trade Start",
            "trade_end":             "Trade End",
        }

    @property
    def description(self) -> str:
        return "PATScalp Hybrid — PATS 2-leg signal, tick bars, MNQ"

    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Pre-compute PATS signals once per IS/OOS slice.

        Returns (bars_df, signals_df) — both passed to run_backtest_prepared.
        """
        signals = _compute_pats_signals(df, self.tick_size)
        return df, signals

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run backtest from raw bars (computes PATS signals internally)."""
        signals = _compute_pats_signals(data, self.tick_size)
        return _run_backtest(data, signals, params, self.tick_size, self.tick_value)

    def run_backtest_prepared(
        self,
        prepared: Tuple[pd.DataFrame, pd.DataFrame],
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run backtest from pre-computed (bars, signals) tuple."""
        bars, signals = prepared
        return _run_backtest(bars, signals, params, self.tick_size, self.tick_value)

    def run_monte_carlo(
        self,
        prepared: Tuple[pd.DataFrame, pd.DataFrame],
        params: Dict[str, Any],
        n_sims: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Day-shuffle Monte Carlo for tick bar data.

        Groups bars by calendar date, shuffles day order n_sims times,
        recomputes PATS signals on each shuffled sequence (required because
        PATS is a sequential state machine), then runs the backtest.

        Stability = fraction of sims with positive net P&L. Target > 0.60.
        """
        bars, _ = prepared  # signals are recomputed per shuffle

        # Group bars by date
        dates  = bars.index.normalize().unique().sort_values()
        groups = [(d, bars[bars.index.normalize() == d]) for d in dates]
        n      = len(groups)

        rng      = np.random.default_rng(seed)
        net_pnls = []
        sharpes  = []

        for _ in range(n_sims):
            order        = rng.permutation(n)
            shuffled     = pd.concat([groups[i][1] for i in order])
            # Reset index to sequential timestamps so the backtest time-window
            # filter still sees intraday times (dates are now arbitrary)
            shuffled.index = bars.index[:len(shuffled)]
            signals      = _compute_pats_signals(shuffled, self.tick_size)
            result       = _run_backtest(shuffled, signals, params,
                                         self.tick_size, self.tick_value)
            if result.get('total_trades', 0) >= 5:
                net_pnls.append(result['net_pnl'])
                sharpes.append(result['sharpe'])

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
