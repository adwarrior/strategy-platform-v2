import pandas as pd
import numpy as np
from strategy_platform.strategies.aurora.tick_loader import classify_delta

def test_classify_delta_uses_nt_rule():
    df = pd.DataFrame({
        "price":  [100.0, 100.0, 100.0, 100.0],
        "bid":    [ 99.75, 99.75,  0.0,   0.0 ],
        "ask":    [100.0, 100.25,  0.0,   0.0 ],
        "volume": [  5,     3,      4,     2  ],
    })
    # tick 0: price>=ask -> +5 ; tick 1: bid<price<ask -> neutral 0
    # tick 2: no quotes, no prev last -> neutral 0
    # tick 3: no quotes, price(100)==prev last(100) -> neutral 0
    out = classify_delta(df)
    assert list(out) == [5, 0, 0, 0]

def test_classify_delta_tick_rule_fallback():
    df = pd.DataFrame({
        "price":  [100.0, 100.25, 100.0],
        "bid":    [  0.0,   0.0,    0.0 ],
        "ask":    [  0.0,   0.0,    0.0 ],
        "volume": [  4,     6,      2  ],
    })
    # t0 neutral (no prev); t1 price up vs 100 -> +6 ; t2 price down vs 100.25 -> -2
    out = classify_delta(df)
    assert list(out) == [0, 6, -2]
