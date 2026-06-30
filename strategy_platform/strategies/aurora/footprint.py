"""
Aurora Heatshelves — footprint engine primitives.

Python port of the engine inner types and tick-capture logic from
AuroraHeatshelvesStrategy.cs (lines 46-95, 354-395).

This module ships:
  - NodeKind        (C# enum NodeKind, lines 46)
  - Row             (C# class Row, lines 48-55)
  - Shelf           (C# class Shelf, lines 57-83, incl. RankStrength)
  - row_key()       (C# RowKey, lines 390-395)
  - FootprintEngine (tick accumulation only; ProcessClosedBar / AddOrMerge /
                      SelectKeys / RefreshKeyShelves are stubbed — Tasks 3-4)

Source: /home/ad/Scripts/strategies/AuroraHeatshelvesStrategy.cs
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List, Optional


class NodeKind(Enum):
    """C# enum NodeKind (line 46): { Balanced, Initiative, Absorption }."""
    BALANCED = auto()
    INITIATIVE = auto()
    ABSORPTION = auto()


@dataclass
class Row:
    """C# class Row (lines 48-55): per-(bar, price-row) buy/sell/neutral volume."""
    up: float = 0.0
    down: float = 0.0
    neutral: float = 0.0

    @property
    def total(self) -> float:
        return self.up + self.down + self.neutral

    @property
    def delta(self) -> float:
        return self.up - self.down


@dataclass
class Shelf:
    """C# class Shelf (lines 57-83): a merged liquidity/footprint shelf.

    Field name map (C# -> Python):
      Top->top, Bot->bot, Vol->vol, Del->delta, Buy->is_buy, Kind->kind,
      Last->bar (the field RankStrength ages from), Orig->orig, Brk->brk,
      End->end, Touch->touch, Flp->flp, Qual->qual.
    """
    top: float
    bot: float
    vol: float
    delta: float
    is_buy: bool
    kind: NodeKind
    bar: float          # C# `Last` — the field RankStrength ages from
    orig: int = 0       # C# `Orig` — bar the shelf was first created
    brk: bool = False   # C# `Brk` — has the level been broken
    end: int = 0        # C# `End`
    touch: int = 0      # C# `Touch`
    flp: bool = False   # C# `Flp`
    qual: float = 0.0   # C# `Qual` — shelf quality score (this task owns it)

    @property
    def mid(self) -> float:
        return (self.top + self.bot) / 2.0

    def rank_strength(self, avg_vol: float, cur_bar: int, half_life_bars: float) -> float:
        """Port of C# Shelf.RankStrength (lines 75-82)."""
        norm = self.vol / avg_vol if avg_vol > 0 else self.vol
        base_s = norm * (1.0 + 0.25 * self.qual)
        age = cur_bar - self.bar
        decay = math.pow(0.5, age / half_life_bars) if half_life_bars > 0 else 1.0
        return base_s * decay


def row_key(price: float, tick_size: float, ticks_per_row: int) -> float:
    """Port of C# RowKey (lines 390-395).

    rowHeight = TicksPerRow * TickSize
    snapped   = Math.Round(price / TickSize) * TickSize
    return Math.Floor(snapped / rowHeight + 1e-9) * rowHeight
    """
    row_height = ticks_per_row * tick_size
    snapped = round(price / tick_size) * tick_size
    return math.floor(snapped / row_height + 1e-9) * row_height


class FootprintEngine:
    """Port of the engine fields/methods in AuroraHeatshelvesStrategy.cs.

    This task ships only `params` plumbing and tick accumulation
    (`on_tick` building `bar_rows[bar][key]`, ported from C# OnMarketData,
    lines 354-388). All other methods are stubs to be filled in Tasks 3-4:
    process_closed_bar, add_or_merge, select_keys, refresh_key_shelves.
    """

    # C# const VolEmaAlpha = 2.0 / (50 + 1) (line 90)
    VOL_EMA_ALPHA: float = 2.0 / (50 + 1)

    # Engine input defaults, ported verbatim from the C# SetDefaults block
    # (lines 302-316). Used when a key is absent from `params`.
    _DEFAULTS: Dict[str, Any] = {
        "tick_size": 0.25,        # MNQ tick size (not a NinjaScript input; instrument)
        "ticks_per_row": 25,      # TicksPerRow
        "lookback": 180,          # Lookback
        "age_half_life": 60,      # AgeHalfLife
        "vol_frac": 0.55,         # VolFrac (volume gate fraction)
        "max_shelves": 250,       # MaxShelves
        "absorb_ratio": 0.25,     # AbsorbRatio (absorption max |d|/volume)
        "break_buf": 0.10,        # BreakBuf (break buffer xATR)
        "allow_flip": True,       # AllowFlip
        "show_balanced": True,    # ShowBalanced
        "show_absorption": True,  # ShowAbsorption
        "show_init": True,        # ShowInit
        "key_per_side": 2,        # KeyPerSide
        "min_gap_atr": 0.6,       # MinGapATR
        "max_dist_pct": 3.0,      # MaxDistPct
    }

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        # merge caller params over the C# defaults
        self.params: Dict[str, Any] = dict(self._DEFAULTS)
        if params:
            self.params.update(params)

        # C# engine fields (lines 86-99)
        self.shelves: List[Shelf] = []
        self.key_shelves: List[Shelf] = []  # C# keyShelves (line 99-ish), set by refresh_key_shelves
        self.avg_bar_vol: float = 0.0
        self.bar_vol_count: int = 0
        self.eff_lookback: int = self.params["lookback"]  # C# effLookback

        # per-bar footprint rows, keyed by bar index then row key
        self.bar_rows: Dict[int, Dict[float, Row]] = {}
        self.added_this_bar: set = set()
        self.last_processed_bar: int = -1
        self.last_trade_price: Optional[float] = None  # C# double.NaN

    def on_tick(self, ts, price: float, bid: float, ask: float, volume: float, cur_bar: int) -> None:
        """Port of C# OnMarketData (lines 354-388).

        NOTE: the C# version reads `CurrentBar` directly from the NinjaTrader
        Bars object; this Python port takes the current bar index explicitly
        as `cur_bar` since there is no live bar-series object here.
        """
        if volume <= 0:
            return

        rows = self.bar_rows.get(cur_bar)
        if rows is None:
            rows = {}
            self.bar_rows[cur_bar] = rows

        tick_size = self.params["tick_size"]
        ticks_per_row = self.params["ticks_per_row"]
        key = row_key(price, tick_size, ticks_per_row)

        r = rows.get(key)
        if r is None:
            r = Row()
            rows[key] = r

        have_quotes = ask > 0 and bid > 0 and ask >= bid
        if have_quotes and price >= ask:
            r.up += volume
        elif have_quotes and price <= bid:
            r.down += volume
        elif not have_quotes and self.last_trade_price is not None and price > self.last_trade_price:
            r.up += volume
        elif not have_quotes and self.last_trade_price is not None and price < self.last_trade_price:
            r.down += volume
        else:
            r.neutral += volume

        self.last_trade_price = price

    # -- Closed-bar processing (Task 3) -------------------------------

    def process_closed_bar(
        self,
        bar: int,
        rows: Dict[float, Row],
        hi: Optional[float] = None,
        lo: Optional[float] = None,
        cl: Optional[float] = None,
        atr: Optional[float] = None,
    ) -> None:
        """Port of C# ProcessClosedBar (lines 466-624).

        The C# version reads High/Low/Close/ATR from the NinjaTrader bar
        series (indexed by `ago = CurrentBar - bar`) and the live `CurrentBar`.
        Here those are passed explicitly:
          - hi/lo/cl: the closed bar's OHLC. When omitted they are synthesized
            from the footprint row keys (full-row span; close at the high).
          - atr: the closed bar's ATR. When omitted/invalid, C# falls back to
            10 * TickSize (line 471).
        `bar` doubles as the "current bar" for ranking/age (C# uses it in the
        hard-cap RankStrength call, line 620).
        """
        tick_size = self.params["tick_size"]
        ticks_per_row = self.params["ticks_per_row"]
        row_height = ticks_per_row * tick_size

        # Synthesize OHLC from row keys when not provided. Rows span
        # [key, key + row_height]; close defaults to the high so closePos = 1.
        if hi is None or lo is None or cl is None:
            if rows:
                keys = list(rows.keys())
                syn_lo = min(keys)
                syn_hi = max(keys) + row_height
            else:
                syn_lo = 0.0
                syn_hi = 0.0
            if lo is None:
                lo = syn_lo
            if hi is None:
                hi = syn_hi
            if cl is None:
                cl = hi

        # C# line 471: atrSafe fallback when ATR is non-positive / NaN.
        if atr is None or atr <= 0 or math.isnan(atr):
            atr_safe = 10 * tick_size
        else:
            atr_safe = atr

        # bar volume EMA (lines 473-477)
        bar_vol = sum(r.total for r in rows.values())
        if self.bar_vol_count == 0:
            self.avg_bar_vol = bar_vol
        else:
            self.avg_bar_vol += self.VOL_EMA_ALPHA * (bar_vol - self.avg_bar_vol)
        self.bar_vol_count += 1

        buf = atr_safe * self.params["break_buf"]

        # 1) DEFEND / BREAK / FLIP pass (lines 481-517)
        allow_flip = self.params["allow_flip"]
        for s in self.shelves:
            touched = lo <= s.top and hi >= s.bot
            if not s.brk:
                if s.is_buy and cl < s.bot - buf:
                    s.brk = True
                    s.end = bar
                    s.bar = bar  # C# s.Last = bar
                elif (not s.is_buy) and cl > s.top + buf:
                    s.brk = True
                    s.end = bar
                    s.bar = bar
                elif touched and (cl > s.top if s.is_buy else cl < s.bot):
                    poke = (s.mid - lo) if s.is_buy else (hi - s.mid)
                    sharp = max(0.0, poke) / max(atr_safe, tick_size)
                    s.touch += 1
                    s.qual += 1.0 + min(sharp, 2.0)
                    s.bar = bar  # C# s.Last = bar
            elif allow_flip and not s.flp:
                if s.is_buy and touched and cl < s.bot:
                    s.is_buy = False
                    s.brk = False
                    s.flp = True
                    s.touch = 1
                    s.qual = 1.0
                    s.delta = 0.0
                    s.orig = bar
                    s.bar = bar
                elif (not s.is_buy) and touched and cl > s.top:
                    s.is_buy = True
                    s.brk = False
                    s.flp = True
                    s.touch = 1
                    s.qual = 1.0
                    s.delta = 0.0
                    s.orig = bar
                    s.bar = bar

        # 2) NODE DETECTION (lines 519-611)
        self.added_this_bar.clear()
        if rows:
            vol_frac = self.params["vol_frac"]
            absorb_ratio = self.params["absorb_ratio"]
            show_balanced = self.params["show_balanced"]
            show_absorption = self.params["show_absorption"]
            show_init = self.params["show_init"]

            b_max = max(r.total for r in rows.values())
            vol_gate = b_max * vol_frac
            third = (hi - lo) / 3.0

            close_pos = (cl - lo) / max(hi - lo, tick_size)
            buy_followed_up = close_pos >= 0.66
            sell_followed_dn = close_pos <= 0.34

            best_abs_b = best_abs_s = best_ini_b = best_ini_s = 0.0
            abs_b_key = abs_s_key = ini_b_key = ini_s_key = float("nan")
            abs_b = abs_s = ini_b = ini_s = None

            for k, r in rows.items():
                tv = r.total
                if tv < vol_gate:
                    continue
                dd = r.delta
                mid = k + ticks_per_row * tick_size / 2.0
                ratio = abs(dd) / max(tv, 1e-10)
                if ratio <= absorb_ratio:
                    if mid <= lo + third and tv > best_abs_b:
                        best_abs_b = tv
                        abs_b = r
                        abs_b_key = k
                    elif mid >= hi - third and tv > best_abs_s:
                        best_abs_s = tv
                        abs_s = r
                        abs_s_key = k
                else:
                    if dd > 0:
                        if buy_followed_up and dd > best_ini_b:
                            best_ini_b = dd
                            ini_b = r
                            ini_b_key = k
                        elif (not buy_followed_up) and mid <= lo + third and tv > best_abs_b:
                            best_abs_b = tv
                            abs_b = r
                            abs_b_key = k
                        elif (not buy_followed_up) and mid >= hi - third and tv > best_abs_s:
                            best_abs_s = tv
                            abs_s = r
                            abs_s_key = k
                    elif dd < 0:
                        if sell_followed_dn and -dd > best_ini_s:
                            best_ini_s = -dd
                            ini_s = r
                            ini_s_key = k
                        elif (not sell_followed_dn) and mid <= lo + third and tv > best_abs_b:
                            best_abs_b = tv
                            abs_b = r
                            abs_b_key = k
                        elif (not sell_followed_dn) and mid >= hi - third and tv > best_abs_s:
                            best_abs_s = tv
                            abs_s = r
                            abs_s_key = k

            row_h = row_height
            if abs_b is not None:
                k = NodeKind.ABSORPTION if close_pos >= 0.5 else NodeKind.BALANCED
                show = show_absorption if k == NodeKind.ABSORPTION else show_balanced
                if show:
                    self.add_or_merge(bar, abs_b_key + row_h, abs_b_key,
                                      abs_b.total, abs_b.delta, True, k)
            if abs_s is not None:
                k = NodeKind.ABSORPTION if close_pos <= 0.5 else NodeKind.BALANCED
                show = show_absorption if k == NodeKind.ABSORPTION else show_balanced
                if show:
                    self.add_or_merge(bar, abs_s_key + row_h, abs_s_key,
                                      abs_s.total, abs_s.delta, False, k)
            if show_init and ini_b is not None:
                self.add_or_merge(bar, ini_b_key + row_h, ini_b_key,
                                  ini_b.total, ini_b.delta, True, NodeKind.INITIATIVE)
            if show_init and ini_s is not None:
                self.add_or_merge(bar, ini_s_key + row_h, ini_s_key,
                                  ini_s.total, ini_s.delta, False, NodeKind.INITIATIVE)

            # 2b) TRUE ABSORPTION — strong delta that FAILED to follow through (lines 581-610)
            if show_absorption and b_max > 0:
                bar_delta = sum(r.delta for r in rows.values())
                if bar_delta > 0 and close_pos <= 0.34:
                    # SUPPLY at the high
                    a_s_vol = 0.0
                    a_s = None
                    a_s_key = float("nan")
                    for k, r in rows.items():
                        if r.total < vol_gate:
                            continue
                        m = k + row_h / 2.0
                        if m >= hi - third and r.total > a_s_vol:
                            a_s_vol = r.total
                            a_s = r
                            a_s_key = k
                    if a_s is not None:
                        self.add_or_merge(bar, a_s_key + row_h, a_s_key,
                                          a_s.total, a_s.delta, False, NodeKind.ABSORPTION)
                elif bar_delta < 0 and close_pos >= 0.66:
                    # DEMAND at the low
                    a_b_vol = 0.0
                    a_b = None
                    a_b_key = float("nan")
                    for k, r in rows.items():
                        if r.total < vol_gate:
                            continue
                        m = k + row_h / 2.0
                        if m <= lo + third and r.total > a_b_vol:
                            a_b_vol = r.total
                            a_b = r
                            a_b_key = k
                    if a_b is not None:
                        self.add_or_merge(bar, a_b_key + row_h, a_b_key,
                                          a_b.total, a_b.delta, True, NodeKind.ABSORPTION)

        # 3) PRUNE by inactivity (lines 613-615)
        cut = bar - self.eff_lookback
        self.shelves = [s for s in self.shelves if s.bar >= cut]

        # 4) HARD CAP (lines 617-623)
        max_shelves = self.params["max_shelves"]
        age_half_life = self.params["age_half_life"]
        while len(self.shelves) > max_shelves:
            broken = [s for s in self.shelves if s.brk]
            if broken:
                victim = min(broken,
                             key=lambda s: s.rank_strength(self.avg_bar_vol, bar, age_half_life))
            else:
                victim = min(self.shelves,
                             key=lambda s: s.rank_strength(self.avg_bar_vol, bar, age_half_life))
            self.shelves.remove(victim)

    def add_or_merge(
        self,
        bar: int,
        top: float,
        bot: float,
        vol: float,
        delta: float,
        is_buy: bool,
        kind: NodeKind,
    ) -> None:
        """Port of C# AddOrMerge (lines 626-658).

        Dedupes per bar (rounded bot | side | kind). Merges into an existing
        shelf ONLY when it is not broken, same side, same kind, and overlapping
        — otherwise appends a fresh shelf. On merge, Qual += 1.0 (line 656).
        """
        tick_size = self.params["tick_size"]
        dedupe = "{}|{}|{}".format(round(bot / tick_size), 1 if is_buy else 0, kind.value)
        if dedupe in self.added_this_bar:
            return
        self.added_this_bar.add(dedupe)

        hit = None
        for s in self.shelves:
            if (not s.brk) and s.is_buy == is_buy and s.kind == kind \
                    and bot <= s.top and top >= s.bot:
                hit = s
                break

        if hit is None:
            self.shelves.append(Shelf(
                orig=bar, bar=bar,
                top=top, bot=bot, vol=vol, delta=delta,
                is_buy=is_buy, kind=kind, brk=False, end=bar,
                touch=0, qual=0.0, flp=False,
            ))
            return

        hit.vol += vol
        hit.delta += delta
        hit.top = max(hit.top, top)
        hit.bot = min(hit.bot, bot)
        hit.touch += 1
        hit.qual += 1.0
        hit.bar = bar  # C# hit.Last = bar

    # -- Key-shelf selection (Task 4) ----------------------------------

    def score(self, s: Shelf, cl: float, atr_safe: float) -> float:
        """Port of C# Score (lines 662-666)."""
        d_atr = abs(s.mid - cl) / atr_safe
        age_half_life = self.params["age_half_life"]
        return s.rank_strength(self.avg_bar_vol, self.last_processed_bar, age_half_life) / (1.0 + d_atr)

    def in_range(self, s: Shelf, cl: float) -> bool:
        """Port of C# InRange (lines 668-671)."""
        max_dist_pct = self.params["max_dist_pct"]
        return max_dist_pct <= 0 or abs(s.mid - cl) / cl <= max_dist_pct / 100.0

    def select_keys(
        self,
        pool: List[Shelf],
        cnt: int,
        min_gap: float,
        cl: float,
        atr_safe: float,
    ) -> List[Shelf]:
        """Port of C# SelectKeys (lines 673-695)."""
        outp: List[Shelf] = []
        take = min(cnt, len(pool))
        used: set = set()
        for _ in range(take):
            best = -1.0
            bp: Optional[Shelf] = None
            for cand in pool:
                if id(cand) in used:
                    continue
                clustered = False
                if min_gap > 0:
                    for q in outp:
                        if abs(cand.mid - q.mid) < min_gap:
                            clustered = True
                            break
                if clustered:
                    continue
                sc = self.score(cand, cl, atr_safe)
                if sc > best:
                    best = sc
                    bp = cand
            if bp is not None:
                used.add(id(bp))
                outp.append(bp)
        return outp

    def refresh_key_shelves(self, cl: float, atr: float) -> None:
        """Port of C# RefreshKeyShelves (lines 697-713)."""
        # C# line 471-pattern atrSafe fallback is applied by the caller in
        # ProcessClosedBar; RefreshKeyShelves itself receives atrSafe already.
        # Mirror that contract: treat non-positive/NaN atr the same way.
        tick_size = self.params["tick_size"]
        if atr is None or atr <= 0 or math.isnan(atr):
            atr_safe = 10 * tick_size
        else:
            atr_safe = atr

        sup_pool: List[Shelf] = []
        dem_pool: List[Shelf] = []
        for s in self.shelves:
            if s.brk or not self.in_range(s, cl):
                continue
            if s.is_buy and s.mid < cl:
                dem_pool.append(s)
            elif (not s.is_buy) and s.mid > cl:
                sup_pool.append(s)

        key_per_side = self.params["key_per_side"]
        min_gap = self.params["min_gap_atr"] * atr_safe
        sup_keys = self.select_keys(sup_pool, key_per_side, min_gap, cl, atr_safe)
        dem_keys = self.select_keys(dem_pool, key_per_side, min_gap, cl, atr_safe)
        self.key_shelves = sup_keys + dem_keys
