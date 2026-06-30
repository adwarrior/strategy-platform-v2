"""
Aurora Heatshelves — key-shelf selection tests.

Covers FootprintEngine.score / in_range / select_keys / refresh_key_shelves,
ported from AuroraHeatshelvesStrategy.cs Score/InRange/SelectKeys/
RefreshKeyShelves (lines 662-713).
"""

from strategy_platform.strategies.aurora.footprint import FootprintEngine, Shelf, NodeKind


def test_select_keys_respects_count_and_min_gap():
    eng = FootprintEngine(params={"key_per_side": 2})
    # three demand shelves below price; min_gap should drop the one too close to a stronger pick
    eng.shelves = [
        Shelf(top=99.0, bot=98.0, vol=900, delta=200, is_buy=True, kind=NodeKind.ABSORPTION, bar=10),
        Shelf(top=98.9, bot=97.9, vol=100, delta=10,  is_buy=True, kind=NodeKind.ABSORPTION, bar=10),
        Shelf(top=95.0, bot=94.0, vol=800, delta=150, is_buy=True, kind=NodeKind.ABSORPTION, bar=10),
    ]
    eng.refresh_key_shelves(cl=100.0, atr=1.0)
    demand_keys = [s for s in eng.key_shelves if s.is_buy]
    assert len(demand_keys) <= 2
    # strongest (vol 900) must be selected; near-duplicate (vol 100) dropped by min-gap
    assert any(abs(s.vol - 900) < 1 for s in demand_keys)
    assert all(abs(s.vol - 100) > 1 for s in demand_keys)
