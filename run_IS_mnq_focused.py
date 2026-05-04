#!/usr/bin/env python3
"""
IS-only focused grid optimization for MoboBands on MNQ 233-tick bars.
Period: Sep 2024 – Jun 2025. Ranked by profit_factor.

Fixed params: calculate_mode=on_each_tick, tick_bar_size=233,
              all filters at defaults except those swept below.
Swept params:  dpo_period, mobo_length, num_dev_up, num_dev_dn,
               profit_ticks, stop_ticks, bars_between_trades  (32,400 combos)

Commission override: $1.02 RT (NinjaTrader $0.51/leg for MNQ).
"""
from __future__ import annotations

import itertools
import multiprocessing as mp
import os
import sys
from datetime import datetime

sys.path.insert(0, '/home/ad/strategy-platform')
os.chdir('/home/ad/strategy-platform')

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

import numpy as np
import strategy_platform          # triggers @register for all strategies
from strategy_platform.registry import StrategyRegistry
from strategy_platform.data.loader import load_tick_bars

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOL       = 'MNQ'
BAR_SIZE     = 233
IS_START     = '2024-09-12'
IS_END       = '2025-06-30'
MIN_TRADES   = 20
NWORKERS     = max(1, mp.cpu_count() - 1)
CHUNK_SIZE   = 200
OUTFILE      = 'reports/IS_optimization_MNQ_233tick.csv'
CHECKPOINT   = 'reports/checkpoint_IS_mnq_focused.csv'

TICK_SIZE    = 0.25
TICK_VALUE   = 0.50   # MNQ micro contract
COMMISSION   = 1.02   # $0.51/leg × 2 (NinjaTrader per-leg rate)

# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

FOCUSED_GRID: dict = {
    'dpo_period':          [10, 12, 14, 16, 18, 20],
    'mobo_length':         [14, 21, 28],
    'num_dev_up':          [0.5, 0.8, 1.0, 1.2, 1.5],
    'num_dev_dn':          [0.5, 0.8, 1.0, 1.2, 1.5],
    'profit_ticks':        [20, 30, 40, 50, 60, 80],
    'stop_ticks':          [10, 15, 20, 25, 30, 40],
    'bars_between_trades': [0, 1, 2, 3],
}

FIXED_PARAMS: dict = {
    'tick_bar_size':            BAR_SIZE,
    'calculate_mode':           'on_bar_close',  # fast sweep; top 10 re-run on_each_tick below
    'hook_lookback':            2,
    'slope_lookback':           5,
    'slope_threshold':          0.0,
    'enable_middle_band_hook':  True,
    'require_color_change':     False,
    'enable_divergence_filter': False,
    'divergence_lookback':      20,
    'enable_bw_filter':         True,
    'bw_period':                50,
    'bw_multiplier':            1.0,
    'enable_time_filter':       False,
    'enable_wattah_atar':       False,
    'enable_longs':             True,
    'enable_shorts':            True,
    'wa_fast_length':           10,
    'wa_slow_length':           30,
    'wa_channel_length':        30,
    'wa_sensitivity':           150,
    'wa_mult':                  2.0,
    'wa_dead_zone':             200,
}

# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

_state: dict = {}


def _init_worker(df_is: pd.DataFrame, strategy_name: str) -> None:
    cls = StrategyRegistry.get(strategy_name)
    strat = cls()
    strat.tick_size     = TICK_SIZE
    strat.tick_value    = TICK_VALUE
    strat.commission_rt = COMMISSION
    _state['df']    = df_is
    _state['strat'] = strat


def _run_one(params: dict):
    strat = _state['strat']
    try:
        result = strat.run_backtest(_state['df'], params)
        trades = result.get('total_trades', result.get('trades', 0))
        if trades < MIN_TRADES:
            return None
        pf = result.get('profit_factor', 0.0)
        if pf == float('inf'):
            pf = 999.0
        return {
            **{k: params[k] for k in FOCUSED_GRID},
            'trades':        trades,
            'win_rate':      round(result.get('win_rate', 0.0), 4),
            'net_pnl':       round(result.get('net_pnl', 0.0), 2),
            'profit_factor': round(pf, 4),
            'sharpe':        round(result.get('sharpe', 0.0), 4),
        }
    except Exception as exc:
        print(f"  [warn] combo failed: {exc}", flush=True)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.makedirs('reports', exist_ok=True)
    t0 = datetime.now()
    print(f"\n{'='*60}")
    print(f"  MoboBands IS Optimization — MNQ 233-tick  {IS_START}→{IS_END}")
    print(f"  Started: {t0.strftime('%Y-%m-%d %H:%M:%S')}  workers={NWORKERS}")
    print(f"{'='*60}\n")

    # --- resume from checkpoint ---
    done_keys: set = set()
    prior_rows: list = []
    if os.path.exists(CHECKPOINT):
        cp = pd.read_csv(CHECKPOINT)
        prior_rows = cp.to_dict('records')
        for r in prior_rows:
            done_keys.add(tuple(r[k] for k in FOCUSED_GRID))
        print(f"Resuming from checkpoint: {len(done_keys):,} combos already done\n")

    # --- wait for DB to be reachable (retry for up to 30 min) ---
    import pymysql, time as _time
    db_host = os.getenv('DB_HOST', '192.168.1.228')
    for attempt in range(30):
        try:
            pymysql.connect(host=db_host, user=os.getenv('DB_USER','adam'),
                            password=os.getenv('DB_PASSWORD',''), db='emini').close()
            break
        except Exception as e:
            print(f"  DB not reachable ({e}), retrying in 60s ({attempt+1}/30)...", flush=True)
            _time.sleep(60)
    else:
        print("ERROR: DB unreachable after 30 minutes — aborting.")
        sys.exit(1)

    # --- load data in monthly batches to avoid MySQL connection timeout ---
    print(f"Loading {SYMBOL} {BAR_SIZE}-tick bars {IS_START} → {IS_END} (monthly batches)...")
    months = pd.date_range(start=IS_START, end=IS_END, freq='MS')
    batch_ends = list(months[1:].strftime('%Y-%m-%d')) + [
        (pd.Timestamp(IS_END) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    ]
    batch_starts = [IS_START] + list(months[1:].strftime('%Y-%m-%d'))

    chunks = []
    for b_start, b_end in zip(batch_starts, batch_ends):
        print(f"  batch {b_start} → {b_end}", flush=True)
        chunk = load_tick_bars(SYMBOL, bar_size=BAR_SIZE, start=b_start, end=b_end)
        if not chunk.empty:
            chunks.append(chunk)

    if not chunks:
        print("ERROR: no data returned — check DB connection and date range.")
        sys.exit(1)

    df = pd.concat(chunks).sort_index()
    df = df[~df.index.duplicated(keep='first')]
    print(f"  {len(df):,} bars total  ({df.index[0].date()} → {df.index[-1].date()})\n")

    # --- build combo list ---
    grid_keys = list(FOCUSED_GRID.keys())
    all_combos = [
        {**FIXED_PARAMS, **dict(zip(grid_keys, vals))}
        for vals in itertools.product(*FOCUSED_GRID.values())
    ]
    pending = [
        c for c in all_combos
        if tuple(c[k] for k in FOCUSED_GRID) not in done_keys
    ]
    total = len(all_combos)
    print(f"Grid: {total:,} total combos, {len(pending):,} pending\n")

    # --- parallel sweep ---
    results: list = list(prior_rows)
    with mp.Pool(
        NWORKERS,
        initializer=_init_worker,
        initargs=(df, 'mobobands'),
    ) as pool:
        for i in range(0, len(pending), CHUNK_SIZE):
            batch  = pending[i : i + CHUNK_SIZE]
            batch_results = pool.map(_run_one, batch)
            new    = [r for r in batch_results if r is not None]
            results.extend(new)
            done   = i + len(batch)
            elapsed = (datetime.now() - t0).seconds
            print(
                f"  [{done:,}/{len(pending):,}]  valid={len(results):,}  "
                f"elapsed={elapsed//60}m{elapsed%60:02d}s",
                flush=True,
            )
            pd.DataFrame(results).to_csv(CHECKPOINT, index=False)

    # --- sort and save ---
    df_out = pd.DataFrame(results)
    if df_out.empty:
        print("\nNo results passed MIN_TRADES filter.")
        return

    df_out = df_out.sort_values('profit_factor', ascending=False).reset_index(drop=True)
    df_out.to_csv(OUTFILE, index=False)
    print(f"\nSaved {len(df_out):,} results → {OUTFILE}")

    # --- Top 10 table ---
    display_cols = [
        'dpo_period', 'mobo_length', 'num_dev_up', 'num_dev_dn',
        'profit_ticks', 'stop_ticks', 'bars_between_trades',
        'trades', 'win_rate', 'net_pnl', 'profit_factor', 'sharpe',
    ]
    print("\n" + "="*60)
    print("  TOP 10 by profit_factor")
    print("="*60)
    print(df_out[display_cols].head(10).to_string(index=True))

    # --- Top 5 NT cross-check block ---
    print("\n" + "="*60)
    print("  TOP 5 — NT Strategy Analyzer Parameters")
    print("="*60)
    for rank, (_, row) in enumerate(df_out.head(5).iterrows(), start=1):
        print(f"\n  Rank {rank}  PF={row['profit_factor']:.3f}  "
              f"WR={row['win_rate']:.1%}  PnL=${row['net_pnl']:,.0f}  "
              f"Trades={int(row['trades'])}")
        for k in FOCUSED_GRID:
            default = {
                'dpo_period': 14, 'mobo_length': 21, 'num_dev_up': 0.8,
                'num_dev_dn': 0.8, 'profit_ticks': 40, 'stop_ticks': 40,
                'bars_between_trades': 2,
            }
            marker = '  *' if row[k] != default[k] else '   '
            print(f"  {marker} {k}: {row[k]}")

    # --- Phase 2: re-run top 10 with on_each_tick for accurate entry prices ---
    print("\n" + "="*60)
    print("  PHASE 2 — re-running top 10 with on_each_tick")
    print("="*60)
    top10_params = [
        {**FIXED_PARAMS, 'calculate_mode': 'on_each_tick',
         **{k: row[k] for k in FOCUSED_GRID}}
        for _, row in df_out.head(10).iterrows()
    ]
    strat = StrategyRegistry.get('mobobands')()
    strat.tick_size = TICK_SIZE; strat.tick_value = TICK_VALUE; strat.commission_rt = COMMISSION
    tick_rows = []
    for i, p in enumerate(top10_params, start=1):
        result = strat.run_backtest(df, p)
        trades = result.get('total_trades', 0)
        pf = result.get('profit_factor', 0.0)
        if pf == float('inf'): pf = 999.0
        tick_rows.append({
            **{k: p[k] for k in FOCUSED_GRID},
            'trades':        trades,
            'win_rate':      round(result.get('win_rate', 0.0), 4),
            'net_pnl':       round(result.get('net_pnl', 0.0), 2),
            'profit_factor': round(pf, 4),
            'sharpe':        round(result.get('sharpe', 0.0), 4),
        })
        print(f"  [{i}/10] PF={pf:.3f}  WR={result.get('win_rate',0):.1%}  "
              f"PnL=${result.get('net_pnl',0):,.0f}  Trades={trades}", flush=True)

    df_tick = pd.DataFrame(tick_rows).sort_values('profit_factor', ascending=False).reset_index(drop=True)
    tick_outfile = OUTFILE.replace('.csv', '_tick_top10.csv')
    df_tick.to_csv(tick_outfile, index=False)
    print(f"\nSaved on_each_tick top-10 → {tick_outfile}")

    print("\n" + "="*60)
    print("  TOP 5 — NT Strategy Analyzer Parameters (on_each_tick verified)")
    print("="*60)
    defaults = {'dpo_period': 14, 'mobo_length': 21, 'num_dev_up': 0.8,
                'num_dev_dn': 0.8, 'profit_ticks': 40, 'stop_ticks': 40,
                'bars_between_trades': 2}
    for rank, (_, row) in enumerate(df_tick.head(5).iterrows(), start=1):
        print(f"\n  Rank {rank}  PF={row['profit_factor']:.3f}  "
              f"WR={row['win_rate']:.1%}  PnL=${row['net_pnl']:,.0f}  Trades={int(row['trades'])}")
        for k in FOCUSED_GRID:
            marker = '  *' if row[k] != defaults[k] else '   '
            print(f"  {marker} {k}: {row[k]}")

    elapsed_total = (datetime.now() - t0).seconds
    print(f"\nTotal runtime: {elapsed_total//60}m {elapsed_total%60:02d}s")

    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)
        print("Checkpoint cleaned up.")


if __name__ == '__main__':
    main()
