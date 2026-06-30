"""Smoke tests for the Aurora trading layer + run_backtest event loop (Task 5).

These check only the result CONTRACT (keys, trades-frame shape, lowercase
directions) on a tiny synthetic flat tick stream — they do NOT assert specific
trades. Real-data parity is Task 6.
"""
import pandas as pd

from strategy_platform.registry import StrategyRegistry
import strategy_platform.strategies.aurora.strategy  # noqa: F401  ensure registration


def test_aurora_registered():
    assert "aurora" in StrategyRegistry.list_strategies()


def test_run_backtest_returns_contract_keys():
    strat = StrategyRegistry.get("aurora")()
    # 2 minutes of flat synthetic ticks -> valid (likely empty) result, correct shape
    idx = pd.date_range("2026-02-03 09:30:00", periods=240, freq="500ms")
    data = pd.DataFrame({"price": 18000.0, "bid": 17999.75, "ask": 18000.0,
                         "volume": 1}, index=idx)
    res = strat.run_backtest(data, strat.params)
    for k in ("net_pnl", "total_trades", "win_rate", "sharpe", "max_drawdown"):
        assert k in res
    assert isinstance(res.get("trades"), pd.DataFrame)
    if not res["trades"].empty:
        assert set(res["trades"]["direction"].unique()) <= {"long", "short"}
        for col in ("entry_time", "exit_time", "direction", "entry_price",
                    "exit_price", "pnl", "qty", "reason"):
            assert col in res["trades"].columns


def test_default_params_match_nt():
    strat = StrategyRegistry.get("aurora")()
    p = strat.default_params
    assert p["entry_offset_ticks"] == 2
    assert p["tp_early_pts"] == 20.0 and p["sl_early_pts"] == 20.0
    assert p["tp_late_pts"] == 10.0 and p["sl_late_pts"] == 10.0
    assert p["rearm_atr"] == 1.0
    assert p["tighten_time"] == "11:00"
    assert p["ticks_per_row"] == 25
    assert p["key_per_side"] == 2
    assert p["entry_start"] == "09:30"
    assert p["entry_end"] == "12:00"
    assert p["flat_by"] == "15:55"
