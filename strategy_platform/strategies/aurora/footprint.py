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
from dataclasses import dataclass, field
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
    """C# class Shelf (lines 57-83): a merged liquidity/footprint shelf."""
    top: float
    bot: float
    vol: float
    delta: float
    is_buy: bool
    kind: NodeKind
    bar: float          # C# `Last` — the field RankStrength ages from
    flp: bool = False   # C# `Flp`

    @property
    def mid(self) -> float:
        return (self.top + self.bot) / 2.0

    def rank_strength(self, avg_vol: float, cur_bar: int, half_life_bars: float) -> float:
        """Port of C# Shelf.RankStrength (lines 75-82). Note: C# uses `Qual`
        (shelf quality score) which is not part of this task's Shelf fields;
        Qual is assumed 0.0 here pending whichever later task introduces it."""
        norm = self.vol / avg_vol if avg_vol > 0 else self.vol
        qual = 0.0
        base_s = norm * (1.0 + 0.25 * qual)
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

    def __init__(self, params: Dict[str, Any]):
        self.params = params

        # C# engine fields (lines 86-99)
        self.shelves: List[Shelf] = []
        self.avg_bar_vol: float = 0.0
        self.bar_vol_count: int = 0

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

    # -- Stubs: filled in Tasks 3-4 -----------------------------------

    def process_closed_bar(self, bar, rows) -> None:
        raise NotImplementedError("process_closed_bar is implemented in Task 3")

    def add_or_merge(self, *args, **kwargs) -> None:
        raise NotImplementedError("add_or_merge is implemented in Task 3")

    def select_keys(self, *args, **kwargs) -> None:
        raise NotImplementedError("select_keys is implemented in Task 4")

    def refresh_key_shelves(self, cl, atr) -> None:
        raise NotImplementedError("refresh_key_shelves is implemented in Task 4")
