"""Fast IS+OOS sweep runner for NYBreakout — bypasses pipeline.py MC bottleneck.

Used by per-group coordinate-descent optimization. Returns top combos by
IS-Sharpe with their OOS metrics. Pick winner by OOS-Sharpe with min-trades guard.
"""

from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, '/home/ad/strategy-platform-v2')

import pandas as pd

from strategy_platform.data.loader import load_5m, INSTRUMENT_META
from strategy_platform.strategies.nybreakout.strategy import NYBreakout


_SHARED: Dict[str, Any] = {}


def _init_worker(is_df: pd.DataFrame, oos_df: pd.DataFrame, locked: Dict[str, Any]) -> None:
    s = NYBreakout()
    m = INSTRUMENT_META['MNQ']
    s.tick_size = m['tick_size']
    s.tick_value = m['tick_value']
    s.commission_rt = m['commission']
    _SHARED['strategy'] = s
    _SHARED['is_df'] = is_df
    _SHARED['oos_df'] = oos_df
    _SHARED['locked'] = locked


def _run_one(combo: Dict[str, Any]) -> Dict[str, Any]:
    s = _SHARED['strategy']
    p = {**_SHARED['locked'], **combo}
    out: Dict[str, Any] = dict(combo)
    try:
        r_is = s.run_backtest(_SHARED['is_df'], p)
        r_oos = s.run_backtest(_SHARED['oos_df'], p)
        out.update({
            'is_trades': r_is['total_trades'],
            'is_win_rate': r_is['win_rate'],
            'is_net_pnl': r_is['net_pnl'],
            'is_sharpe': r_is['sharpe'],
            'is_pf': r_is['profit_factor'],
            'is_mdd': r_is['max_drawdown'],
            'oos_trades': r_oos['total_trades'],
            'oos_win_rate': r_oos['win_rate'],
            'oos_net_pnl': r_oos['net_pnl'],
            'oos_sharpe': r_oos['sharpe'],
            'oos_pf': r_oos['profit_factor'],
            'oos_mdd': r_oos['max_drawdown'],
        })
    except Exception as e:
        out['error'] = str(e)
    return out


def _expand(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(grid.keys())
    vals = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]
    return [dict(zip(keys, c)) for c in itertools.product(*vals)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2024-01-01')
    p.add_argument('--end', default='2026-04-17')
    p.add_argument('--train-pct', type=float, default=0.70)
    p.add_argument('--locked', required=True, help='JSON dict of locked params')
    p.add_argument('--grid', required=True, help='JSON dict of grid params (lists)')
    p.add_argument('--min-is-trades', type=int, default=30)
    p.add_argument('--out', required=True, help='Output CSV path')
    p.add_argument('--workers', type=int, default=4)
    args = p.parse_args()

    locked = json.loads(args.locked)
    grid = json.loads(args.grid)
    combos = _expand(grid)
    print(f"[sweep] locked={locked}", flush=True)
    print(f"[sweep] grid keys={list(grid.keys())} -> {len(combos)} combos", flush=True)

    df = load_5m('MNQ', start=args.start, end=args.end)
    n = len(df)
    is_end = int(n * args.train_pct)
    is_df = df.iloc[:is_end].copy()
    oos_df = df.iloc[is_end:].copy()
    print(f"[sweep] IS: {is_df.index[0]} -> {is_df.index[-1]}  ({len(is_df):,} bars)", flush=True)
    print(f"[sweep] OOS: {oos_df.index[0]} -> {oos_df.index[-1]}  ({len(oos_df):,} bars)", flush=True)

    with mp.Pool(args.workers, initializer=_init_worker, initargs=(is_df, oos_df, locked)) as pool:
        results: List[Dict[str, Any]] = []
        for i, r in enumerate(pool.imap_unordered(_run_one, combos), 1):
            results.append(r)
            if i % max(1, len(combos) // 10) == 0 or i == len(combos):
                print(f"  {i}/{len(combos)}  ({i*100//len(combos)}%)", flush=True)

    df_out = pd.DataFrame(results)
    df_out.to_csv(args.out, index=False)
    print(f"\n[sweep] saved: {args.out}", flush=True)

    df_valid = df_out[df_out.get('error').isna()] if 'error' in df_out.columns else df_out
    df_valid = df_valid[df_valid['is_trades'] >= args.min_is_trades]
    if len(df_valid) == 0:
        print("[sweep] no combos passed min-is-trades filter")
        return

    keys_to_show = list(grid.keys()) + ['is_trades', 'is_sharpe', 'is_pf', 'is_net_pnl',
                                         'oos_trades', 'oos_sharpe', 'oos_pf', 'oos_net_pnl']
    keys_to_show = [k for k in keys_to_show if k in df_valid.columns]

    print("\n== Top 10 by IS Sharpe ==")
    print(df_valid.sort_values('is_sharpe', ascending=False)[keys_to_show].head(10).to_string(index=False))

    print("\n== Top 10 by OOS Sharpe ==")
    print(df_valid.sort_values('oos_sharpe', ascending=False)[keys_to_show].head(10).to_string(index=False))


if __name__ == '__main__':
    main()
