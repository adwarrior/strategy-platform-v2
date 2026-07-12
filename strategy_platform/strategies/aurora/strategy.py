"""
Aurora Heatshelves — Python port of AuroraHeatshelvesStrategy.cs (NinjaTrader 8).

This is the TRADING LAYER on top of the footprint engine (footprint.py) and the
raw-tick loader (tick_loader.py). It wraps the engine in an event loop over raw
ticks and exposes the platform ``BaseStrategy`` interface.

Faithful port of the C# trade-layer methods:
  - OnBarUpdate          (main loop, lines 420-462)         -> run_backtest loop
  - EffectiveLookback    (lines 397-416)                    -> _effective_lookback
  - UpdateTradingLayer   (line 721)
  - DetectFlipsAndBreaks (lines 747-813)
  - ArmBestKeyWall       (lines 818-871)
  - EligibleForEntry     (lines 889-922)
  - EnterMarket          (lines 924-932)
  - MarkTraded           (lines 934-948)
  - ClearArm             (lines 950-953)
  - StopPtsNow/TargetPtsNow (lines 955-963)
  - SizeForStop          (lines 965-977)
  - OnExecutionUpdate    (fill bookkeeping, lines 984-1000)

NO-LOOK-AHEAD model (mirrors the C#):
  * Ticks are accumulated per 1-min bar via FootprintEngine.on_tick.
  * Shelves are processed only AFTER a bar closes (process_closed_bar on the
    just-closed bar), then the key walls + trading layer (arm/flip) run.
  * A resting intercept limit armed off the just-closed bar is eligible to fill
    on the NEXT bar's ticks (conservative trade-through fill).
  * TP/SL are inferred from the within-bar tick path of subsequent bars.

THREE parity-critical carry-forwards (see report):
  1. Real OHLC + ATR are passed into process_closed_bar for every closed bar.
  2. eng.eff_lookback = EffectiveLookback() is set BEFORE each process_closed_bar.
  3. eng.last_processed_bar is advanced each bar.

Source: /home/ad/Scripts/strategies/AuroraHeatshelvesStrategy.cs

2026-07-09: synced to the C# 2026-07-06 engine re-sync — MergeMaxRows shelf
height cap + center-distance merge gate, corrupt-tick guard, opt-in
consolidation detector — plus the reliability filters (entry_min_touches,
entry_min_age_bars, fast_tape_atr_mult, all default OFF) and per-trade wall
metadata (wall_kind/mid/touches/age/flipped, mirroring the NT fill log).
"""

from __future__ import annotations

import math
from datetime import time as time_t, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from strategy_platform.base_strategy import BaseStrategy
from strategy_platform.registry import register
from strategy_platform.strategies.aurora.footprint import FootprintEngine, NodeKind

# Full 24h time options at 5-min granularity for dashboard time params.
_HHMM_24H: List[str] = [f"{h:02d}:{m:02d}" for h in range(24) for m in range(0, 60, 5)]


def _parse_hhmm(s: str) -> Optional[time_t]:
    """Parse 'HH:MM' into a datetime.time. 'off'/'' -> None (disabled)."""
    if s is None:
        return None
    s = str(s).strip().lower()
    if s in ("", "off", "none"):
        return None
    parts = s.split(":")
    return time_t(int(parts[0]), int(parts[1]))


def _level_key(mid: float, is_buy: bool, tick_size: float) -> str:
    """Port of C# LevelKey (lines 715-718)."""
    return f"{round(mid / tick_size)}|{1 if is_buy else 0}"


class _LevelState:
    """Port of C# LevelState (lines 109-121)."""
    __slots__ = ("mid", "buy", "kind", "traded", "retired", "ref", "rearm_ready")

    def __init__(self, mid: float = 0.0, buy: bool = False,
                 kind: Optional[NodeKind] = None, ref=None):
        self.mid = mid
        self.buy = buy
        self.kind = kind
        self.traded = 0
        self.retired = False
        self.ref = ref
        self.rearm_ready = True


# ---------------------------------------------------------------------------
# ATR(14) Wilder — matches NinjaTrader's ATR(14) indicator
# ---------------------------------------------------------------------------

def _wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                period: int = 14) -> np.ndarray:
    """NinjaTrader ATR(14): bar 0 ATR = High-Low; subsequent bars use the
    Wilder running-average seeded over the first `period` bars, exactly as
    NT's ATR indicator (and the existing STF port's _compute_supertrend seed):
        atr[i] = ((min(i+1, period) - 1) * atr[i-1] + TR) / min(i+1, period)
    TR = max(High-Low, |High-PrevClose|, |Low-PrevClose|).
    """
    n = len(high)
    atr = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if i == 0:
            atr[i] = high[i] - low[i]
        else:
            c1 = close[i - 1]
            tr = max(high[i] - low[i], abs(high[i] - c1), abs(low[i] - c1))
            denom = min(i + 1, period)
            atr[i] = ((denom - 1) * atr[i - 1] + tr) / denom
    return atr


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _summarise(trades: List[Dict]) -> Dict[str, Any]:
    if not trades:
        return {
            'net_pnl': 0.0, 'total_trades': 0, 'win_rate': 0.0,
            'sharpe': 0.0, 'max_drawdown': 0.0,
            'avg_win': 0.0, 'avg_loss': 0.0, 'profit_factor': 0.0,
        }
    pnls = np.array([t['pnl'] for t in trades], dtype=float)
    n = len(pnls)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    max_dd = float(-(peak - cum).max())

    # Daily Sharpe (annualised), grouped by calendar date of entry.
    daily: Dict = {}
    for t in trades:
        d = pd.Timestamp(t['entry_time']).date()
        daily[d] = daily.get(d, 0.0) + t['pnl']
    d_vals = np.array(list(daily.values()), dtype=float)
    if len(d_vals) > 1:
        std = d_vals.std(ddof=1)
        sharpe = float((d_vals.mean() / std) * np.sqrt(252)) if std > 0 else 0.0
    else:
        sharpe = 0.0

    gross_profit = float(wins.sum()) if len(wins) else 0.0
    gross_loss = float(losses.sum()) if len(losses) else 0.0

    return {
        'net_pnl': float(pnls.sum()),
        'total_trades': n,
        'win_rate': float(len(wins) / n),
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'avg_win': float(wins.mean()) if len(wins) else 0.0,
        'avg_loss': float(losses.mean()) if len(losses) else 0.0,
        'profit_factor': gross_profit / abs(gross_loss) if gross_loss != 0 else float('inf'),
    }


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------

@register
class Aurora(BaseStrategy):
    """
    Aurora Heatshelves intercept scalp — Python port (MNQ 1m, raw ticks).

    Consumes a RAW-TICK DataFrame (ts ET-naive index; cols price/bid/ask/volume)
    from tick_loader.load_raw_ticks, builds 1-min footprint bars, and runs the
    footprint engine + trading layer to produce trades.
    """

    name = 'aurora'

    bar_type = 'tick'
    supported_bar_types = ['tick']
    calculate_mode = 'on_each_tick'
    # Footprint strategy: consumes RAW ticks with bid/ask, never OHLC bars.
    # The platform must call self.load_data() instead of its generic loaders —
    # and load_data reads ONLY tick_data_full (full volume). The legacy
    # tick_data table understates volume ~44% (dedupe artifact), which
    # silently halves every wall for a footprint engine.
    data_kind = 'raw_tick'
    tick_table = 'tick_data_full'

    # MNQ defaults
    tick_size = 0.25
    tick_value = 0.50          # $ per tick (MNQ: $0.50/tick, $2/point)
    # NT runs net (IncludeCommission=true, C# line 299). Real Trades.csv shows
    # $1.02 per contract round-trip. Subtracted in _close_position as
    # commission_rt * qty so net P&L matches NT.
    commission_rt = 1.02       # $ per contract round-trip (matches NT net P&L)

    symbol: str = 'MNQ=F'

    default_params: Dict[str, Any] = {
        # 0. Bar type the engine runs on, built from the same raw ticks.
        #    'Nmin' = N-minute time bars (NT close-time labelling);
        #    'Nt'   = N-tick bars (label = last tick's timestamp, NT-style).
        #    Everything downstream (walls, arming, fills, session windows) is
        #    bar-type agnostic — it works off bar index + bar_time.
        'bar_spec': '1min',
        # 1. Engine (mirror the indicator — NT SetDefaults lines 302-316)
        'ticks_per_row': 25,
        'lookback': 180,
        'lookback_cap_days': 5.0,
        'age_half_life': 60,
        'vol_frac': 0.55,
        'max_shelves': 250,
        'absorb_ratio': 0.25,
        'break_buf': 0.10,
        'allow_flip': True,
        'key_per_side': 2,
        'min_gap_atr': 0.6,
        'max_dist_pct': 3.0,
        'show_balanced': True,
        'show_absorption': True,
        'show_init': True,
        # 1. Engine — 2026-07-06 re-sync (MergeMaxRows + opt-in consolidation)
        'merge_max_rows': 3,
        'show_consolidation': False,
        'consol_min_bars': 12,
        'consol_vol_mult': 1.8,
        # 2. Entry (NT lines 319-332)
        'entry_offset_ticks': 2,
        'trade_bal': True,
        'trade_absorb': True,
        'trade_init': True,
        'flip_to_market': True,
        'rearm_atr': 1.0,
        'flip_tol_pct': 0.001,
        # 2. Entry — reliability filters (2026-07-06, default OFF like the C#)
        'entry_min_touches': 0,
        'entry_min_age_bars': 0,
        'fast_tape_atr_mult': 0.0,
        # 3. Exits
        'tp_early_pts': 20.0,
        'sl_early_pts': 20.0,
        'tp_late_pts': 10.0,
        'sl_late_pts': 10.0,
        'tighten_time': '11:00',     # 'off' disables the early->late switch
        # 4. Sizing
        'use_risk_sizing': True,
        'contracts': 1,
        'risk_dollars': 100.0,
        'max_contracts': 5,
        # 5. Session
        'entry_start': '09:30',
        'entry_end': '12:00',
        'flat_by': '15:55',
    }

    # ------------------------------------------------------------------
    # Param grid — the 5 swept knobs plus headroom
    # ------------------------------------------------------------------

    @property
    def param_grid(self) -> Dict[str, Any]:
        return {
            'entry_offset_ticks': (0, 6, 1),
            'tp_early_pts': (10.0, 40.0, 5.0),
            'sl_early_pts': (10.0, 40.0, 5.0),
            'tp_late_pts':  (5.0, 20.0, 5.0),
            'sl_late_pts':  (5.0, 20.0, 5.0),
            'rearm_atr':    (0.5, 2.0, 0.5),
            'tighten_time': ['off', '10:30', '11:00', '11:30'],
            'entry_min_touches': (0, 5, 1),
            'entry_min_age_bars': (0, 20, 5),
            'fast_tape_atr_mult': (0.0, 4.0, 1.0),
        }

    @property
    def param_groups(self) -> Dict[str, List[str]]:
        return {
            '1. Engine': [
                'bar_spec',
                'ticks_per_row', 'lookback', 'lookback_cap_days', 'age_half_life',
                'vol_frac', 'max_shelves', 'absorb_ratio', 'break_buf',
                'allow_flip', 'key_per_side', 'min_gap_atr', 'max_dist_pct',
                'show_balanced', 'show_absorption', 'show_init',
                'merge_max_rows', 'show_consolidation', 'consol_min_bars',
                'consol_vol_mult',
            ],
            '2. Entry': [
                'entry_offset_ticks', 'trade_bal', 'trade_absorb', 'trade_init',
                'flip_to_market', 'rearm_atr', 'flip_tol_pct',
                'entry_min_touches', 'entry_min_age_bars', 'fast_tape_atr_mult',
            ],
            '3. Exits': [
                'tp_early_pts', 'sl_early_pts', 'tp_late_pts', 'sl_late_pts',
                'tighten_time',
            ],
            '4. Sizing': [
                'use_risk_sizing', 'contracts', 'risk_dollars', 'max_contracts',
            ],
            '5. Session': [
                'entry_start', 'entry_end', 'flat_by',
            ],
        }

    @property
    def description(self) -> str:
        return "Aurora Heatshelves footprint intercept scalp (ported from NT8 C#)."

    # ------------------------------------------------------------------
    # EffectiveLookback — C# lines 397-416 (1-min bars here)
    # ------------------------------------------------------------------

    @staticmethod
    def _effective_lookback(lookback: int, cap_days: float,
                            minutes_per_bar: float = 1.0) -> int:
        target = int(lookback)
        if cap_days <= 0.0:
            return target
        if minutes_per_bar <= 0.0:
            return target
        cap_bars = cap_days * 6.5 * 60.0 / minutes_per_bar
        capped = int(round(cap_bars))
        if capped < 1:
            capped = 1
        return min(target, capped)

    # ------------------------------------------------------------------
    # Platform data hook (data_kind='raw_tick')
    # ------------------------------------------------------------------

    def load_data(self, symbol: str, start: str, end: str,
                  host: Optional[str] = None) -> pd.DataFrame:
        """Load raw ticks from tick_data_full for a platform symbol + range.

        The platform trades continuous symbols ('MNQ=F'); tick_data_full is
        keyed by CONTRACT ('MNQ_M26', 'MNQ_U26'). Per business day the
        contract is chosen data-driven: the one whose loaded coverage spans
        the day, and on overlap days (roll week) the one with more ticks.
        Refuses loudly when the full-volume table has no coverage — NEVER
        falls back to the thinned legacy tick_data table.
        """
        from sqlalchemy import text
        from strategy_platform.data import loader as _dl
        from .tick_loader import load_raw_ticks

        prefix = str(symbol).split('=')[0].replace('_F', '').upper()
        s_date = str(start).split('T')[0]
        e_date = str(end).split('T')[0]

        engine = _dl._engine(host)
        with engine.connect() as conn:
            cov = conn.execute(
                text("SELECT symbol, MIN(ts), MAX(ts) FROM " + self.tick_table +
                     " WHERE symbol LIKE :p GROUP BY symbol"),
                {"p": prefix + r"\_%"}).fetchall()
        if not cov:
            raise ValueError(
                f"{self.tick_table} holds no '{prefix}_*' contracts at all — "
                "re-export ticks from NinjaTrader and load them first.")
        cov_desc = ", ".join(
            f"{r[0]} {pd.Timestamp(r[1]).date()}→{pd.Timestamp(r[2]).date()}" for r in cov)

        days = pd.bdate_range(s_date, e_date)
        if len(days) == 0:
            days = pd.DatetimeIndex([pd.Timestamp(s_date)])
        frames = []
        with engine.connect() as conn:
            for day in days:
                d = day.date()
                cands = [r for r in cov
                         if pd.Timestamp(r[1]).date() <= d <= pd.Timestamp(r[2]).date()]
                if not cands:
                    continue
                if len(cands) == 1:
                    pick = cands[0][0]
                else:
                    # roll week: both contracts loaded — take the dominant one
                    pick, best_n = None, -1
                    for r in cands:
                        n = conn.execute(
                            text("SELECT COUNT(*) FROM " + self.tick_table +
                                 " WHERE symbol=:s AND ts>=:a AND ts<:b"),
                            {"s": r[0], "a": str(d),
                             "b": str(d + timedelta(days=1))}).scalar()
                        if n > best_n:
                            pick, best_n = r[0], int(n)
                t = load_raw_ticks(pick, str(d), str(d), host=host, table=self.tick_table)
                if len(t):
                    frames.append(t)
        if not frames:
            raise ValueError(
                f"{self.tick_table} has no {prefix} ticks between {s_date} and "
                f"{e_date}. Loaded coverage: {cov_desc}. Re-export the missing "
                "range from NinjaTrader before backtesting it.")
        df = pd.concat(frames).sort_index()
        # Trim to the requested intraday window (index is ET-naive).
        df = df.loc[str(start).replace('T', ' '):str(end).replace('T', ' ')]
        if df.empty:
            raise ValueError(
                f"No {prefix} ticks left after trimming to {start}→{end} "
                f"(coverage: {cov_desc}).")
        return df

    # ------------------------------------------------------------------
    # Core backtest
    # ------------------------------------------------------------------

    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        p = {**self.default_params, **(params or {})}
        ts = self.tick_size
        point_value = self.tick_value / ts  # $ per point (MNQ: 0.50/0.25 = 2.0)

        # ---- Engine params (subset the engine understands) ----
        eng_params = {
            'tick_size': ts,
            'ticks_per_row': int(p['ticks_per_row']),
            'lookback': int(p['lookback']),
            'age_half_life': int(p['age_half_life']),
            'vol_frac': float(p['vol_frac']),
            'max_shelves': int(p['max_shelves']),
            'absorb_ratio': float(p['absorb_ratio']),
            'break_buf': float(p['break_buf']),
            'allow_flip': bool(p['allow_flip']),
            'show_balanced': bool(p['show_balanced']),
            'show_absorption': bool(p['show_absorption']),
            'show_init': bool(p['show_init']),
            'key_per_side': int(p['key_per_side']),
            'min_gap_atr': float(p['min_gap_atr']),
            'max_dist_pct': float(p['max_dist_pct']),
            'merge_max_rows': int(p['merge_max_rows']),
            'show_consolidation': bool(p['show_consolidation']),
            'consol_min_bars': int(p['consol_min_bars']),
            'consol_vol_mult': float(p['consol_vol_mult']),
        }
        eng = FootprintEngine(eng_params)

        # ---- Trading params ----
        entry_off = int(p['entry_offset_ticks']) * ts
        trade_bal = bool(p['trade_bal'])
        trade_absorb = bool(p['trade_absorb'])
        trade_init = bool(p['trade_init'])
        flip_to_market = bool(p['flip_to_market'])
        rearm_atr = float(p['rearm_atr'])
        flip_tol_pct = float(p['flip_tol_pct'])
        entry_min_touches = int(p['entry_min_touches'])
        entry_min_age_bars = int(p['entry_min_age_bars'])
        fast_tape_atr_mult = float(p['fast_tape_atr_mult'])
        tp_early = float(p['tp_early_pts'])
        sl_early = float(p['sl_early_pts'])
        tp_late = float(p['tp_late_pts'])
        sl_late = float(p['sl_late_pts'])
        tighten_t = _parse_hhmm(p.get('tighten_time'))
        use_risk = bool(p['use_risk_sizing'])
        contracts = int(p['contracts'])
        risk_dollars = float(p['risk_dollars'])
        max_contracts = int(p['max_contracts'])
        entry_start = _parse_hhmm(p['entry_start'])
        entry_end = _parse_hhmm(p['entry_end'])
        flat_by = _parse_hhmm(p['flat_by'])

        # ---- Bar spec: 'Nmin' time bars or 'Nt' tick bars -----------------
        bar_spec = str(p.get('bar_spec', '1min')).strip().lower()
        if bar_spec.endswith('t'):
            bar_tick_count = int(bar_spec[:-1])
            bar_minutes = 0.0                      # undefined for tick bars
        elif bar_spec.endswith('min'):
            bar_tick_count = 0
            bar_minutes = float(bar_spec[:-3])
        else:
            raise ValueError(f"bar_spec must look like '5min' or '1000t', got {bar_spec!r}")

        # Tick bars: minutes_per_bar undefined -> raw lookback, mirroring the
        # C# EffectiveLookback fallback for non-time bar types.
        eff_lookback = self._effective_lookback(
            int(p['lookback']), float(p['lookback_cap_days']),
            minutes_per_bar=bar_minutes)

        if data is None or data.empty:
            return {**_summarise([]), 'trades': pd.DataFrame()}

        # ---- Build bars from raw ticks ----
        price = data['price'].to_numpy(float)
        bid = data['bid'].to_numpy(float) if 'bid' in data.columns else np.zeros(len(data))
        ask = data['ask'].to_numpy(float) if 'ask' in data.columns else np.zeros(len(data))
        vol = data['volume'].to_numpy(float) if 'volume' in data.columns else np.ones(len(data))
        index = data.index
        if bar_tick_count > 0:
            # N-tick bars: every N consecutive tick EVENTS form one bar
            # (matches NT tick-bar construction; a bar may straddle the
            # maintenance break — acceptable approximation, no session reset).
            codes = np.arange(len(price)) // bar_tick_count
            n_bars = int(codes[-1]) + 1
            bar_starts = None                      # unused on the tick path
        else:
            bar_floor = index.floor(f'{int(bar_minutes)}min')
            # Map each tick to a sequential bar index (0,1,2,...) in time order.
            codes, bar_starts = pd.factorize(bar_floor, sort=True)
            n_bars = len(bar_starts)

        # Per-bar OHLC from tick prices.
        bar_hi = np.full(n_bars, -np.inf)
        bar_lo = np.full(n_bars, np.inf)
        bar_open = np.full(n_bars, np.nan)
        bar_close = np.full(n_bars, np.nan)
        for i in range(len(price)):
            b = codes[i]
            pr = price[i]
            if pr > bar_hi[b]:
                bar_hi[b] = pr
            if pr < bar_lo[b]:
                bar_lo[b] = pr
            if np.isnan(bar_open[b]):
                bar_open[b] = pr
            bar_close[b] = pr
        # PARITY FIX (2026-07-01): NinjaTrader labels a bar by its CLOSE time
        # (end of the minute), while pandas .floor() labels by the OPEN
        # minute. The ticks in [09:29:00, 09:30:00) are NT's "09:30" bar, not
        # "09:29". Without this shift every bar decision (arm time, session
        # window, tighten time) ran one minute early vs NT — the trades drifted
        # ~1 min, trade-by-trade match collapsed, and the wrong early-session
        # levels armed (Feb PF 0.92 vs NT 1.87). Shift the LABEL forward one
        # bar interval to match NT's close-time convention; OHLC grouping is
        # unchanged. Tick bars are labelled by their LAST tick's timestamp —
        # NT's convention for non-time bars (already a "close" time, no shift).
        if bar_tick_count > 0:
            last_idx = np.minimum(
                (np.arange(n_bars) + 1) * bar_tick_count - 1, len(index) - 1)
            bar_time = pd.DatetimeIndex(index[last_idx])
        else:
            bar_time = pd.DatetimeIndex(bar_starts) + pd.Timedelta(minutes=bar_minutes)

        atr_arr = _wilder_atr(bar_hi, bar_lo, bar_close, period=14)

        # ---- Trading-layer state (mirror C# fields) ----
        levels: Dict[str, _LevelState] = {}
        armed = False
        armed_is_buy = False
        armed_level_mid = 0.0
        armed_limit = 0.0
        armed_key: Optional[str] = None
        # C# binds StopPtsNow()/TargetPtsNow()/SizeForStop() to the resting order
        # at ARM time (ArmBestKeyWall lines 863-866); the fill inherits them. We
        # freeze them here so a level armed before tighten_time keeps its EARLY
        # brackets/size even if it fills after tighten_time.
        armed_stop_pts = 0.0
        armed_tgt_pts = 0.0
        armed_qty = 0
        # C# entryShelf: shelf behind the most recent arm. NEVER cleared (a
        # fill always belongs to the latest submit), so not touched by clear_arm.
        armed_shelf = None

        # Open position state
        in_pos = False
        pos_dir: Optional[str] = None   # 'long' / 'short'
        pos_entry_px = 0.0
        pos_qty = 0
        pos_entry_time = None
        pos_tp = 0.0
        pos_sl = 0.0
        pos_wall: Dict[str, Any] = {}   # wall metadata frozen at fill time

        trades: List[Dict] = []

        def clear_arm():
            nonlocal armed, armed_key, armed_level_mid, armed_limit
            nonlocal armed_stop_pts, armed_tgt_pts, armed_qty
            armed = False
            armed_key = None
            armed_level_mid = 0.0
            armed_limit = 0.0
            armed_stop_pts = 0.0
            armed_tgt_pts = 0.0
            armed_qty = 0

        def mark_traded(mid: float, is_buy: bool):
            k = _level_key(mid, is_buy, ts)
            st = levels.get(k)
            if st is None:
                st = _LevelState(mid=mid, buy=is_buy)
                levels[k] = st
            st.traded += 1
            if st.kind != NodeKind.ABSORPTION:
                st.retired = True
            else:
                st.rearm_ready = False

        def stop_pts_now(decision_time) -> float:
            if tighten_t is not None and decision_time.time() >= tighten_t:
                return sl_late
            return sl_early

        def target_pts_now(decision_time) -> float:
            if tighten_t is not None and decision_time.time() >= tighten_t:
                return tp_late
            return tp_early

        def size_for_stop(stop_pts: float) -> int:
            if not use_risk:
                return max(1, contracts)
            risk_per_contract = stop_pts * point_value
            if risk_per_contract <= 0:
                return 1
            qty = int(math.floor(risk_dollars / risk_per_contract))
            if qty < 1:
                qty = 1
            if qty > max_contracts:
                qty = max_contracts
            return qty

        def eligible_for_entry(s, cur_bar: int) -> bool:
            """Port of C# EligibleForEntry (2026-07-06 version with seasoning
            filters). `cur_bar` is the forming bar index (C# CurrentBar =
            closed_bar + 1), used for the wall-age gate."""
            # Seasoning filters (default off): a wall must have PROVEN itself
            # by defending (touches) and/or surviving (age) before an
            # intercept may rest on it.
            if entry_min_touches > 0 and s.touch < entry_min_touches:
                return False
            if entry_min_age_bars > 0 and cur_bar - s.orig < entry_min_age_bars:
                return False

            k = _level_key(s.mid, s.is_buy, ts)
            st = levels.get(k)
            if s.kind == NodeKind.BALANCED:
                if not trade_bal:
                    return False
                return st is None or (not st.retired and st.traded == 0)
            if s.kind == NodeKind.ABSORPTION:
                if not trade_absorb:
                    return False
                if st is None:
                    return True
                return (not st.retired) and st.rearm_ready
            if s.kind == NodeKind.INITIATIVE:
                if not trade_init:
                    return False
                if not s.flp:
                    return False
                return st is None or (not st.retired and st.traded == 0)
            return False

        def detect_flips_and_breaks(cl: float, atr_safe: float, decision_time):
            """Port of C# DetectFlipsAndBreaks (lines 747-813)."""
            nonlocal in_pos, pos_dir, pos_entry_px, pos_qty, pos_entry_time, pos_tp, pos_sl
            # Register / refresh every current key wall in the level map.
            for s in eng.key_shelves:
                k = _level_key(s.mid, s.is_buy, ts)
                st = levels.get(k)
                if st is None:
                    st = _LevelState(mid=s.mid, buy=s.is_buy, kind=s.kind, ref=s)
                    levels[k] = st
                else:
                    if st.kind != NodeKind.ABSORPTION and s.kind == NodeKind.ABSORPTION:
                        st.kind = NodeKind.ABSORPTION
                    st.ref = s
                # ABSORB re-arm: price moved ReArmAtr*ATR away -> fresh touch allowed.
                if not st.rearm_ready:
                    if abs(cl - st.mid) >= rearm_atr * atr_safe:
                        st.rearm_ready = True

            # Flip-to-market — faithful port of C# DetectFlipsAndBreaks lines 815-828.
            # ROOT-CAUSE FIX (2026-07-02, from NT ELIGCHK trace): NT's EnterMarket
            # calls EnterLong/Short WHILE a working intercept-limit order already
            # rests for the same signal — NT REJECTS that stacked entry, so NO
            # position opens and NO AuroraFlipMkt trade is ever logged (Feb 718/718
            # + May 605/605 are all AuroraIntercept). BUT MarkTraded runs
            # unconditionally right after, so the armed Absorption level's Traded++
            # and ReArmReady=False — this THROTTLES the level to ineligible on the
            # next bar until price leaves by ReArmAtr*ATR and returns.
            #
            # The earlier port mistakes: (a) _enter_market actually opened a
            # position -> phantom trades; (b) then a tight absolute-tick gate
            # suppressed the flip entirely -> mark_traded never ran -> the level
            # stayed permanently eligible, so the port armed walls (e.g. BUY 27731
            # on 2026-05-01 09:47/09:49) that NT had throttled off -> ~259 missed
            # NT trades. Correct behavior: fire on NT's exact condition, mark the
            # level (throttle), clear the arm, but DON'T open a position.
            if flip_to_market and armed and not in_pos:
                tol = cl * flip_tol_pct                 # C#: shelf near armed mid
                for s in eng.key_shelves:
                    if s.kind != NodeKind.ABSORPTION:
                        continue
                    if s.is_buy != armed_is_buy:
                        continue
                    if abs(s.mid - armed_level_mid) <= tol:
                        # NT's EnterLong/Short is rejected (working entry rests) so
                        # no fill; only the throttle side-effect happens.
                        mark_traded(armed_level_mid, armed_is_buy)
                        clear_arm()
                        break

            # Retire levels whose backing shelf broke or left the key list.
            live_keys = {_level_key(s.mid, s.is_buy, ts) for s in eng.key_shelves}
            dead = []
            for k, st in levels.items():
                shelf_broken = st.ref is not None and st.ref.brk
                gone = k not in live_keys
                if shelf_broken or gone:
                    dead.append(k)
            for k in dead:
                del levels[k]

        # NOTE: C# EnterMarket (lines 984-992) is intentionally NOT ported as a
        # position-opening call. In NT it fires only from the flip-to-market path
        # while a working intercept-limit already rests, so NT rejects the stacked
        # entry and only its MarkTraded side-effect survives. The flip block above
        # reproduces exactly that: mark_traded (throttle) + clear_arm, no position.

        def arm_best_key_wall(cl: float, atr_safe: float, decision_time, cur_bar: int):
            """Port of C# ArmBestKeyWall (lines 818-871)."""
            nonlocal armed, armed_is_buy, armed_level_mid, armed_limit, armed_key
            nonlocal armed_stop_pts, armed_tgt_pts, armed_qty, armed_shelf
            best = None
            best_score = -1.0
            for s in eng.key_shelves:
                if not eligible_for_entry(s, cur_bar):
                    continue
                # NOTE: eng.score uses last_processed_bar (=closed_bar) as the
                # current bar, whereas C# Score uses CurrentBar (=closed_bar + 1).
                # This 1-bar age-decay basis offset is inherent to the engine
                # seam: eng.score is locked to last_processed_bar and the engine
                # (footprint.py) must not be modified, so it cannot be corrected
                # here without changing the engine signature. (Fix-pass FIX 4.)
                sc = eng.score(s, cl, atr_safe)
                if sc > best_score:
                    best_score = sc
                    best = s

            if best is None:
                clear_arm()
                return

            best_key = _level_key(best.mid, best.is_buy, ts)
            # Already armed on this exact level -> leave it resting.
            if armed and best_key == armed_key:
                return

            is_buy = best.is_buy
            limit = (best.mid + entry_off) if is_buy else (best.mid - entry_off)
            limit = round(limit / ts) * ts

            armed = True
            armed_is_buy = is_buy
            armed_level_mid = best.mid
            armed_limit = limit
            armed_key = best_key
            # C# entryShelf: live ref to the backing shelf, read at fill time
            # so every trade carries the wall's kind/touches/age as they were
            # when money went in (mirrors the NT fill log).
            armed_shelf = best
            # C# lines 863-866: brackets + size are evaluated NOW (arm time) and
            # bound to the resting order. Freeze them on the arm state so the
            # fill inherits the ARM-time regime, not the fill-time regime.
            armed_stop_pts = stop_pts_now(decision_time)
            armed_tgt_pts = target_pts_now(decision_time)
            armed_qty = size_for_stop(armed_stop_pts)

        def update_trading_layer(cl: float, atr_safe: float, decision_time,
                                 closed_bar: int):
            """Port of C# UpdateTradingLayer (2026-07-06 version with the
            fast-tape standdown, step 3b)."""
            detect_flips_and_breaks(cl, atr_safe, decision_time)
            if in_pos:
                return
            # Outside entry window: cancel arm, no new entries.
            in_window = True
            if entry_start is not None and entry_end is not None:
                tod = decision_time.time()
                in_window = entry_start <= tod < entry_end
            if not in_window:
                clear_arm()
                return
            # (3b) FAST-TAPE STANDDOWN (opt-in): when the just-closed bar's
            # range blows past fast_tape_atr_mult x ATR, pull any resting
            # entry and stand aside this bar (C# UpdateTradingLayer step 3b).
            if fast_tape_atr_mult > 0 and closed_bar >= 1:
                if bar_hi[closed_bar] - bar_lo[closed_bar] > fast_tape_atr_mult * atr_safe:
                    clear_arm()
                    return
            arm_best_key_wall(cl, atr_safe, decision_time, closed_bar + 1)

        # ---- Main event loop ----
        # We walk ticks in time order. A bar "closes" when the tick's bar code
        # advances. At that point we process all newly-closed bars through the
        # engine + trading layer (mirrors C# OnBarUpdate first-tick-of-bar),
        # using the just-closed bar's close/ATR and the NEW bar's timestamp as
        # the decision time. The armed limit / open position is then checked
        # tick-by-tick against the CURRENT (forming) bar's ticks.

        def process_through(closed_bar: int, decision_time):
            """Process engine + trading layer up to and including `closed_bar`."""
            # C# carry-forward #2: set eff_lookback BEFORE process_closed_bar.
            eng.eff_lookback = eff_lookback
            b = eng.last_processed_bar + 1
            while b <= closed_bar:
                rows = eng.bar_rows.get(b, {})
                # Carry-forward #1: real OHLC + ATR for this closed bar.
                process_closed_bar = eng.process_closed_bar
                process_closed_bar(b, rows,
                                   hi=float(bar_hi[b]), lo=float(bar_lo[b]),
                                   cl=float(bar_close[b]), atr=float(atr_arr[b]))
                # Carry-forward #3: advance last_processed_bar each bar.
                eng.last_processed_bar = b
                eng.bar_rows.pop(b, None)
                b += 1
            # Trading layer off the just-closed bar (C# uses Close[0]/atr[0] of
            # the forming bar; we use the just-closed bar's close + ATR, which is
            # the parity-stable, look-ahead-free equivalent).
            cl = float(bar_close[closed_bar])
            atr_raw = float(atr_arr[closed_bar])
            atr_safe = (10 * ts) if (atr_raw <= 0 or math.isnan(atr_raw)) else atr_raw
            eng.refresh_key_shelves(cl, atr_safe)
            update_trading_layer(cl, atr_safe, decision_time, closed_bar)

        def try_fill_and_exit(tick_price: float, tick_time, cur_bar: int):
            """Check the armed resting limit for a trade-through fill, and any
            open position for TP/SL, against the current tick."""
            nonlocal in_pos, pos_dir, pos_entry_px, pos_qty, pos_entry_time
            nonlocal pos_tp, pos_sl, armed, armed_key

            # 1) Open-position exit check (TP/SL via the within-bar tick path).
            if in_pos:
                if pos_dir == 'long':
                    if tick_price <= pos_sl:
                        _close_position(pos_sl, tick_time, 'SL')
                    elif tick_price >= pos_tp:
                        _close_position(pos_tp, tick_time, 'TP')
                else:
                    if tick_price >= pos_sl:
                        _close_position(pos_sl, tick_time, 'SL')
                    elif tick_price <= pos_tp:
                        _close_position(pos_tp, tick_time, 'TP')
                return

            # 2) Resting limit fill — requires trade-through (conservative).
            if armed:
                hit = False
                if armed_is_buy and tick_price <= armed_limit:
                    hit = True
                elif (not armed_is_buy) and tick_price >= armed_limit:
                    hit = True
                if hit:
                    _fill_limit(tick_time, cur_bar)

        def _fill_limit(fill_time, cur_bar: int):
            """A resting intercept limit filled. The position inherits the
            brackets + size that were FROZEN at arm time (C# binds
            StopPtsNow()/TargetPtsNow()/SizeForStop() to the order in
            ArmBestKeyWall, lines 863-866). A level armed before tighten_time
            keeps its EARLY stop/target/size even if filled after it."""
            nonlocal in_pos, pos_dir, pos_entry_px, pos_qty, pos_entry_time
            nonlocal pos_tp, pos_sl, pos_wall
            is_buy = armed_is_buy
            stop_pts = armed_stop_pts
            tgt_pts = armed_tgt_pts
            qty = armed_qty
            in_pos = True
            pos_dir = 'long' if is_buy else 'short'
            pos_entry_px = armed_limit
            pos_qty = qty
            pos_entry_time = fill_time
            if is_buy:
                pos_tp = armed_limit + tgt_pts
                pos_sl = armed_limit - stop_pts
            else:
                pos_tp = armed_limit - tgt_pts
                pos_sl = armed_limit + stop_pts
            # Wall metadata AT FILL TIME (mirrors the NT fill logger, which
            # reads the live entryShelf ref when the execution lands).
            s = armed_shelf
            pos_wall = {
                'wall_kind': s.kind.name.capitalize() if s is not None else '',
                'wall_mid': s.mid if s is not None else np.nan,
                'wall_touches': s.touch if s is not None else 0,
                'wall_age_bars': (cur_bar - s.orig) if s is not None else 0,
                'wall_flipped': bool(s.flp) if s is not None else False,
            }
            # OnExecutionUpdate: mark the level traded and clear the arm.
            mark_traded(armed_level_mid, is_buy)
            clear_arm()

        def _close_position(exit_px: float, exit_time, reason: str):
            nonlocal in_pos, pos_dir, pos_entry_px, pos_qty, pos_entry_time, pos_tp, pos_sl
            sgn = 1.0 if pos_dir == 'long' else -1.0
            pnl_points = sgn * (exit_px - pos_entry_px)
            pnl = pnl_points * point_value * pos_qty - self.commission_rt * pos_qty
            trades.append({
                'entry_time': pos_entry_time,
                'exit_time': exit_time,
                'direction': pos_dir,
                'entry_price': pos_entry_px,
                'exit_price': exit_px,
                'pnl': pnl,
                'qty': pos_qty,
                'reason': reason,
                **pos_wall,
            })
            in_pos = False
            pos_dir = None

        # Iterate ticks; feed engine; drive bar closes and fills.
        cur_code = -1
        warmup_bars = 20  # C# CurrentBar < 20 guard
        flat_fired_for_date = set()

        for i in range(len(price)):
            b = codes[i]
            t = index[i]
            # C# stamps executions with Time[0] = the forming bar's CLOSE-time
            # label (NT bars are close-stamped), not the raw tick instant. Use the
            # +1-min-shifted bar label so matched entries/exits carry NT's minute,
            # not the raw sub-minute tick time (which reads ~1 bar early).
            bar_label = bar_time[b]

            # New bar opened -> close the previous bar(s) and run the layer.
            if b != cur_code:
                if cur_code >= 0 and b >= warmup_bars:
                    # C#: OnBarUpdate returns when CurrentBar < 20 and first acts
                    # at CurrentBar=20 -> arms off closed bar 19. Here the forming
                    # bar `b` == CurrentBar, so the first arming runs when b==20
                    # (closed bar b-1 == 19). The decision time is the NEW bar's
                    # timestamp. Process all bars up to b-1.
                    process_through(b - 1, bar_time[b])
                elif cur_code >= 0:
                    # Still in warmup: advance the engine but skip trading layer.
                    eng.eff_lookback = eff_lookback
                    bb = eng.last_processed_bar + 1
                    while bb <= b - 1:
                        rows = eng.bar_rows.get(bb, {})
                        eng.process_closed_bar(
                            bb, rows, hi=float(bar_hi[bb]), lo=float(bar_lo[bb]),
                            cl=float(bar_close[bb]), atr=float(atr_arr[bb]))
                        eng.last_processed_bar = bb
                        eng.bar_rows.pop(bb, None)
                        bb += 1
                cur_code = b

            # Session flatten (C# runs every tick): force-exit + cancel arm at flat_by.
            if flat_by is not None and t.time() >= flat_by:
                if in_pos:
                    _close_position(price[i], bar_label, 'EOD')
                clear_arm()
                # After flat_by, suppress new arming for the rest of this bar's
                # processing by skipping fills (handled below via window check).

            # Feed the engine this tick (accumulates into the forming bar `b`).
            eng.on_tick(t, price[i], bid[i], ask[i], vol[i], b)

            # Fill the resting limit / manage the open position on this tick,
            # but only once warmup is satisfied and not past flat_by.
            if b >= warmup_bars:
                if flat_by is not None and t.time() >= flat_by:
                    pass  # no new fills after flat
                else:
                    try_fill_and_exit(price[i], bar_label, b)

        # Close any still-open position at the last tick (defensive; should be
        # rare given the flat_by EOD rule).
        if in_pos and len(price) > 0:
            _close_position(price[-1], index[-1], 'EOD')

        stats = _summarise(trades)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame(
            columns=['entry_time', 'exit_time', 'direction', 'entry_price',
                     'exit_price', 'pnl', 'qty', 'reason', 'wall_kind',
                     'wall_mid', 'wall_touches', 'wall_age_bars', 'wall_flipped'])
        return {**stats, 'trades': trades_df}
