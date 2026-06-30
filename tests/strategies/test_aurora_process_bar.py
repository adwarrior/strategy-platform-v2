"""Task 3: ProcessClosedBar — DEFEND/BREAK/FLIP + node detection + AddOrMerge.

Ports of AuroraHeatshelvesStrategy.cs ProcessClosedBar (466-624) and
AddOrMerge (626-660). Pure-logic, no DB.

The C# ProcessClosedBar reads bar OHLC and ATR from the NinjaTrader bar
series; the Python port takes them explicitly (hi/lo/cl/atr), defaulting
to a synthetic span derived from the row keys when omitted.
"""

from strategy_platform.strategies.aurora.footprint import FootprintEngine, Row, NodeKind


def _row(up, down):
    r = Row()
    r.up = up
    r.down = down
    return r


def test_absorption_classification_low_delta_high_vol():
    """A high-volume row with near-zero net delta is ABSORPTION (|d|/vol below threshold).

    vol 980, |delta| 20 -> ratio ~0.0204, well under AbsorbRatio 0.25.
    Row sits at the bottom third (demand) and the bar closes in the upper
    half (closePos >= 0.5) so the node is tagged ABSORPTION, not BALANCED.
    """
    eng = FootprintEngine()  # MNQ defaults: tick_size 0.25, ticks_per_row 25
    # one row at key 100.0 spanning [100.0, 106.25]; place it in the bottom
    # third of a bar that closes high so it lands as demand absorption.
    rows = {100.0: _row(up=500, down=480)}  # vol 980, |delta| 20
    eng.process_closed_bar(bar=1, rows=rows, hi=120.0, lo=100.0, cl=119.0)
    kinds = [s.kind for s in eng.shelves]
    assert NodeKind.ABSORPTION in kinds


def test_add_or_merge_merges_same_kind_overlap():
    eng = FootprintEngine()
    eng.add_or_merge(1, 101.0, 100.0, 50, 10, True, NodeKind.ABSORPTION)
    n_before = len(eng.shelves)
    eng.add_or_merge(1, 100.5, 99.5, 30, 5, True, NodeKind.ABSORPTION)  # overlaps + same kind
    assert len(eng.shelves) == n_before  # merged, not appended


def test_add_or_merge_increments_qual_on_merge():
    """C# AddOrMerge does hit.Qual += 1.0 on a merge (line 656)."""
    eng = FootprintEngine()
    eng.add_or_merge(1, 101.0, 100.0, 50, 10, True, NodeKind.ABSORPTION)
    assert eng.shelves[0].qual == 0.0  # fresh shelf starts at 0
    eng.add_or_merge(1, 100.5, 99.5, 30, 5, True, NodeKind.ABSORPTION)
    assert eng.shelves[0].qual == 1.0  # merged -> +1.0


def test_add_or_merge_no_merge_different_kind():
    eng = FootprintEngine()
    eng.add_or_merge(1, 101.0, 100.0, 50, 10, True, NodeKind.ABSORPTION)
    eng.add_or_merge(1, 100.5, 99.5, 30, 5, True, NodeKind.INITIATIVE)  # overlaps, different kind
    assert len(eng.shelves) == 2  # not merged


def test_add_or_merge_no_merge_different_side():
    eng = FootprintEngine()
    eng.add_or_merge(1, 101.0, 100.0, 50, 10, True, NodeKind.ABSORPTION)
    eng.add_or_merge(1, 100.5, 99.5, 30, 5, False, NodeKind.ABSORPTION)  # overlaps, different side
    assert len(eng.shelves) == 2  # not merged


def test_break_marks_shelf_broken():
    """A buy (demand) shelf with close below bot - buffer is broken."""
    eng = FootprintEngine()
    eng.add_or_merge(1, 101.0, 100.0, 50, 10, True, NodeKind.ABSORPTION)
    s = eng.shelves[0]
    assert not s.brk
    # close well below bot (100.0) by more than the break buffer (atr*0.10)
    eng.process_closed_bar(bar=2, rows={}, hi=99.5, lo=95.0, cl=95.0, atr=1.0)
    assert s.brk


def test_rank_strength_reads_qual_field():
    """RankStrength formula baseS = norm*(1 + 0.25*Qual) must use the live qual field."""
    from strategy_platform.strategies.aurora.footprint import Shelf
    s = Shelf(top=101.0, bot=100.0, vol=10.0, delta=0.0, is_buy=True,
              kind=NodeKind.ABSORPTION, bar=5)
    s.qual = 4.0
    # avg_vol == vol -> norm == 1; half_life large -> decay ~1 at age 0
    val = s.rank_strength(avg_vol=10.0, cur_bar=5, half_life_bars=60)
    assert abs(val - (1.0 * (1.0 + 0.25 * 4.0))) < 1e-9  # == 2.0
