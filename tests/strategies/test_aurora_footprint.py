from strategy_platform.strategies.aurora.footprint import Row, Shelf, NodeKind, row_key

def test_row_delta_and_total():
    r = Row(); r.up = 7; r.down = 3; r.neutral = 2
    assert r.total == 12
    assert r.delta == 4

def test_row_key_snaps_to_row_height():
    # tick_size 0.25, ticks_per_row 4 -> row height 1.0; price 100.6 snaps to 100.5 grid then floors to row
    # NT: snapped = round(price/tick)*tick ; key = floor(snapped/rowH)*rowH
    assert row_key(100.6, 0.25, 4) == 100.0
    assert row_key(101.0, 0.25, 4) == 101.0

def test_shelf_mid():
    s = Shelf(top=101.0, bot=100.0, vol=50, delta=10, is_buy=True, kind=NodeKind.ABSORPTION, bar=5)
    assert s.mid == 100.5
