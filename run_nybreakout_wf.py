"""Walk-forward stress-test of NYBreakout on MNQ (micro, 1m UTC -> 5m ET).

Validates the MNQ edge across many rolling IS/OOS windows rather than a single
2024-26 holdout. Uses a reduced grid focused on the winning region (full 768
grid x ~24 slices would be enormous), and a 1-year IS / 3-month OOS / 3-month
step cadence.

Usage:
    python3 run_nybreakout_wf.py MNQ
"""
from __future__ import annotations

import sys
from datetime import datetime

import pandas as pd

import strategy_platform  # noqa: F401  triggers @register
from strategy_platform.data.loader import load_1m
from strategy_platform.optimize import walk_forward as WF
from strategy_platform.registry import StrategyRegistry

DATA_START = "2020-01-01"
DATA_END   = None

# Reduced grid: lock the structural winners, sweep the parameters most likely
# to vary across regimes. ~24 combos/slice keeps the WF tractable.
WF_GRID = {
    'anchor_start_hour':        [9],
    'anchor_end_hour':          [10],
    'min_fvg_ticks':            [2, 4],
    'enable_rebalance':         [False],
    'include_pre_trigger_fvgs': [True, False],
    'prefer_overlap':           [True],
    'prefer_outside':           [True],
    'prefer_closest':           [True],
    'allow_limit_retarget':     [True],
    'bos_cancel_enabled':       [False],
    'bos_cancel_count':         [3],
    'max_trades_per_day':       [1, 3],
    'entry_cutoff_time':        ['11:00', '12:00', '13:00'],
    'cancel_pending_at_cutoff': [False],
    'rr_target':                [1.0, 1.5],
    'eod_exit_time':            ['16:55'],
    'direction':                ['Both'],
    'use_risk_sizing':          [True],
    'max_risk':                 [300.0],
    'qty':                      [1],
}


def load_5m_et(symbol, host):
    print(f"Loading {symbol} 1M (UTC) ...", flush=True)
    df1 = load_1m(symbol, start=DATA_START, end=DATA_END, host=host)
    if df1.empty:
        raise SystemExit(f"No 1m data for {symbol}.")
    idx = df1.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df1 = df1.copy()
    df1.index = idx.tz_convert("US/Eastern").tz_localize(None)
    df5 = (df1.resample("5min", label="right", closed="right")
           .agg({"open":"first","high":"max","low":"min","close":"last","volume":"sum"})
           .dropna())
    print(f"  -> {len(df5):,} 5m ET bars ({df5.index[0]} -> {df5.index[-1]})", flush=True)
    return df5


def main():
    symbol = (sys.argv[1] if len(sys.argv) > 1 else "MNQ").upper()
    t0 = datetime.now()
    strat = StrategyRegistry.get("nybreakout")()
    host  = getattr(strat, "db_host", None)

    df5 = load_5m_et(symbol, host)
    WF.load_5m = lambda *a, **k: df5  # drive WF with prepared ET frame

    # _compute_slices needs explicit date strings (it can't derive them from the
    # monkeypatched loader). Use the prepared frame's actual span.
    d_start = df5.index[0].strftime("%Y-%m-%d")
    d_end   = df5.index[-1].strftime("%Y-%m-%d")
    print(f"  WF date range: {d_start} -> {d_end}", flush=True)

    res = WF.run_walk_forward(
        strategy_name       = "nybreakout",
        symbol              = symbol,
        bar_type            = "time",
        data_start          = d_start,
        data_end            = d_end,
        is_window_days      = 365,
        oos_window_days     = 90,
        step_days           = 90,
        rank_by             = "sharpe",
        param_grid_override = WF_GRID,
    )
    n_slices = len(res.get("slices", []))
    print(f"\nDONE {symbol} WF in {(datetime.now()-t0).seconds//60}m. slices={n_slices}", flush=True)


if __name__ == "__main__":
    main()
