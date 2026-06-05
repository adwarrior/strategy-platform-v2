"""Full IS/MC/OOS optimization of NYBreakout on micro futures (MNQ, MGC).

Micro 1-minute data lives in emini.historical_data_1m, stored in **UTC**.
NYBreakout's anchor logic is defined in **ET** (9-10am NY). So we must:
  1. load 1m (UTC, tz-naive),
  2. tz_localize UTC -> tz_convert US/Eastern -> drop tz (keep ET wall-clock),
  3. resample to 5m ET (label/closed='right' to match NT close-stamped bars),
then hand the prepared 5m-ET frame to the platform's run_pipeline via a
pre-prepared DataFrame so it does NOT re-load from MySQL.

This mirrors openretest/strategy.py's prepare_data tz handling and the
run_IS_*_focused.py runner pattern.

Usage:
    python3 run_nybreakout_micro.py MNQ
    python3 run_nybreakout_micro.py MGC
"""
from __future__ import annotations

import sys
from datetime import datetime

import pandas as pd

import strategy_platform  # noqa: F401  triggers @register for all strategies
from strategy_platform.data.loader import load_1m, get_meta
from strategy_platform.optimize import pipeline as P
from strategy_platform.registry import StrategyRegistry


# 1m micro coverage in historical_data_1m: 2020-01-01 -> ~2026-05-29
DATA_START = "2020-01-01"
DATA_END   = None          # to latest
TRAIN_PCT  = 0.70          # 70% IS / 30% OOS by date range


def load_5m_et(symbol: str, host: str | None) -> pd.DataFrame:
    """Load 1m UTC micro bars and return 5m ET-naive OHLCV."""
    print(f"Loading {symbol} 1M (UTC) {DATA_START} -> {DATA_END or 'latest'} ...", flush=True)
    df1 = load_1m(symbol, start=DATA_START, end=DATA_END, host=host)
    if df1.empty:
        raise SystemExit(f"No 1m data for {symbol}.")
    print(f"  {len(df1):,} 1m bars  ({df1.index[0]} -> {df1.index[-1]})", flush=True)

    idx = df1.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    df1 = df1.copy()
    df1.index = idx.tz_convert("US/Eastern").tz_localize(None)

    df5 = (
        df1.resample("5min", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min",
              "close": "last", "volume": "sum"})
        .dropna()
    )
    print(f"  -> {len(df5):,} 5m ET bars  ({df5.index[0]} -> {df5.index[-1]})", flush=True)
    return df5


def main() -> None:
    symbol = (sys.argv[1] if len(sys.argv) > 1 else "MNQ").upper()
    t0 = datetime.now()

    cls   = StrategyRegistry.get("nybreakout")
    strat = cls()
    host  = getattr(strat, "db_host", None)

    df5 = load_5m_et(symbol, host)

    # run_pipeline re-loads from MySQL via load_5m (which has no micros) and does
    # no tz conversion — so we drive the pipeline with our prepared ET frame by
    # monkeypatching the loader it calls to return our df. bar_type stays 'time'.
    P.load_5m = lambda *a, **k: df5  # type: ignore

    meta = get_meta(symbol)
    print(f"  meta: tick_size={meta['tick_size']} tick_value=${meta['tick_value']} "
          f"commission=${meta['commission']}", flush=True)

    FOCUSED_GRID = {
        'anchor_start_hour':        [9],
        'anchor_end_hour':          [10],
        'min_fvg_ticks':            [2, 4, 6, 8],
        'enable_rebalance':         [False],
        'include_pre_trigger_fvgs': [True, False],
        'prefer_overlap':           [True, False],
        'prefer_outside':           [True],
        'prefer_closest':           [True],
        'allow_limit_retarget':     [True, False],
        'bos_cancel_enabled':       [False],
        'bos_cancel_count':         [3],
        'max_trades_per_day':       [1, 3],
        'entry_cutoff_time':        ['11:00', '12:00', '13:00'],
        'cancel_pending_at_cutoff': [False],
        'rr_target':                [1.0, 1.5, 2.0, 3.0],
        'eod_exit_time':            ['16:55'],
        'direction':                ['Both'],
        'use_risk_sizing':          [True],
        'max_risk':                 [300.0],
        'qty':                      [1],
    }

    results = P.run_pipeline(
        strategy_name       = "nybreakout",
        symbol              = symbol,
        bar_type            = "time",
        train_pct           = TRAIN_PCT,
        rank_by             = "sharpe",
        param_grid_override = FOCUSED_GRID,
        mc_sims             = 30,   # 200 was pathologically slow single-process; 30 gives same stability signal
    )

    print(f"\nDONE {symbol} in {(datetime.now()-t0).seconds//60}m. "
          f"Result frames: {list(results.keys())}", flush=True)
    for k, v in results.items():
        try:
            print(f"  {k}: {len(v)} rows")
        except Exception:
            pass


if __name__ == "__main__":
    main()
