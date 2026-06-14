"""
Generic 4-stage optimization pipeline for the strategy platform.

Stages
------
1. IS grid search   — parallel sweep of all param combinations on in-sample data.
                      Each combo includes a bootstrap stability check.
2. Monte Carlo      — day-shuffle (or strategy-specific) MC on top N IS combos.
3. OOS validation   — walk-forward check on top MC-stable combos.
4. Reports          — CSV results + Sharpe heatmap saved to reports/.

Checkpoint / resume
-------------------
Results are checkpointed every CHECKPOINT_INTERVAL combos to:
    reports/checkpoint_<strategy>_<symbol>_<grid_hash>.csv
Re-running picks up from where it left off.  Checkpoint is deleted on success.

Usage (CLI)
-----------
    python -m strategy_platform.optimize.pipeline --strategy goldbot7
    python -m strategy_platform.optimize.pipeline --strategy goldbot7 --refresh
    python -m strategy_platform.optimize.pipeline --strategy goldbot7 --symbol NQ=F

Usage (library)
---------------
    from strategy_platform.optimize.pipeline import run_pipeline
    run_pipeline("goldbot7", symbol="GC=F", data_start="2024-03-01")
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
import os
import sys
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import strategy_platform  # noqa: F401 — triggers auto-registration of all strategies
from strategy_platform.data.loader import get_meta, is_oos_split, load_1m, load_5m, load_tick_bars, resample_ohlcv
from strategy_platform.registry import StrategyRegistry
from strategy_platform import results_store

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

BOOTSTRAP_SIMS      = 1000
MONTE_CARLO_SIMS    = 200
MC_TOP_N            = 20
OOS_TOP_N           = 5
MIN_TRADES          = 20
CHECKPOINT_INTERVAL = 50

# Supported IS ranking metrics: name -> (column, ascending)
RANK_METRICS: Dict[str, tuple] = {
    'sharpe':        ('sharpe',        False),
    'profit_factor': ('profit_factor', False),
    'sortino':       ('sortino',       False),
    'net_pnl':       ('net_pnl',       False),
    'max_drawdown':  ('max_drawdown',  True),   # lower is better
}

REPORTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'reports')


def strategy_reports_dir(strategy_name: str) -> str:
    """Per-strategy subfolder under reports/. Created on demand.

    All strategy-tagged outputs (BT/RUN/IS/MC/OOS/AR/WF/checkpoint/heatmap) live
    here. Files without a strategy tag (sweep findings, cross-strategy syntheses)
    belong in reports/_synthesis or reports/_shared.
    """
    d = os.path.join(REPORTS_DIR, strategy_name)
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grid_combinations(grid: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
    keys = list(grid.keys())
    for values in itertools.product(*grid.values()):
        yield dict(zip(keys, values))


def _deduplicated_combinations(
    grid: Dict[str, List[Any]],
    param_dependencies: Dict[str, tuple],
) -> Iterator[Dict[str, Any]]:
    """
    Yield unique parameter combinations, collapsing dependent params when their
    controlling param does not have the required value.

    For example, if param_dependencies = {"ema_period": ("use_ema_filter", True)},
    then when use_ema_filter=False, ema_period is forced to its first grid value
    and duplicate combos are skipped — saving compute on irrelevant sweeps.
    """
    if not param_dependencies:
        yield from _grid_combinations(grid)
        return

    seen: set = set()
    for combo in _grid_combinations(grid):
        canonical = dict(combo)
        for dep_param, (ctrl_param, required_val) in param_dependencies.items():
            if dep_param not in canonical or ctrl_param not in canonical:
                continue
            if canonical[ctrl_param] != required_val:
                # Controlling param is not in its required state — collapse dependent param
                first_val = grid[dep_param][0] if dep_param in grid else canonical[dep_param]
                canonical[dep_param] = first_val
        key = tuple(canonical[k] for k in sorted(canonical.keys()))
        if key not in seen:
            seen.add(key)
            yield canonical


def _grid_hash(grid: dict, data_start: Optional[str] = None, data_end: Optional[str] = None) -> str:
    key = {"grid": grid, "start": data_start, "end": data_end}
    return hashlib.md5(
        json.dumps(key, sort_keys=True, default=str).encode()
    ).hexdigest()[:8]


def _combo_key(combo: dict, param_keys: List[str]) -> tuple:
    return tuple(combo[k] for k in param_keys)


def _load_checkpoint(path: str, param_keys: List[str]) -> tuple[set, list]:
    """Return (set of completed combo keys, list of prior result dicts)."""
    if not os.path.exists(path):
        return set(), []
    try:
        df    = pd.read_csv(path)
        keys  = {tuple(row[k] for k in param_keys) for _, row in df.iterrows()}
        return keys, df.to_dict('records')
    except Exception:
        return set(), []


# ---------------------------------------------------------------------------
# Multiprocessing worker
# ---------------------------------------------------------------------------

_worker_state: dict = {}


def _init_worker(prepared_is, strategy_name: str, symbol_meta: dict) -> None:
    _worker_state['prepared_is']   = prepared_is
    _worker_state['strategy_name'] = strategy_name
    _worker_state['symbol_meta']   = symbol_meta


def _run_combo(params: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate one parameter combination (called inside worker pool)."""
    cls      = StrategyRegistry.get(_worker_state['strategy_name'])
    strategy = cls()
    meta = _worker_state.get('symbol_meta', {})
    if meta:
        strategy.tick_size     = meta['tick_size']
        strategy.tick_value    = meta['tick_value']
        strategy.commission_rt = meta['commission']
    # Merge swept params on top of defaults so the strategy always receives a
    # complete params dict even when only a subset of groups are being swept.
    full_params = {**strategy.default_params, **params}
    result      = strategy.run_backtest_prepared(_worker_state['prepared_is'], full_params)
    # Strip the trades DataFrame — too large to serialise across processes
    result.pop('trades', None)
    row = dict(params)
    row.update(result)
    return row


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    strategy_name:      str,
    symbol:             Optional[str] = None,
    bar_type:           Optional[str] = None,
    timeframe_mins:     Optional[int] = None,
    data_start:         Optional[str] = None,
    data_end:           Optional[str] = None,
    refresh:            bool = False,
    train_pct:          float = 0.70,
    param_grid_override: Optional[Dict[str, List[Any]]] = None,
    mc_sims:            int = MONTE_CARLO_SIMS,
    mc_top_n:           int = MC_TOP_N,
    oos_top_n:          int = OOS_TOP_N,
    min_trades:         int = MIN_TRADES,
    rank_by:            str = 'sharpe',
    run_settings:       Optional[Dict[str, Any]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Run the full 4-stage IS/MC/OOS/Bootstrap pipeline for *strategy_name*.

    Parameters
    ----------
    strategy_name : registered strategy name, e.g. "goldbot7"
    symbol        : instrument symbol, e.g. "GC=F" (default: first in strategy.param_grid or raise)
    data_start    : ISO date string for data start
    data_end      : ISO date string for data end
    refresh       : force MySQL reload (bypass Parquet cache)
    train_pct     : IS fraction (default 0.70)

    Returns
    -------
    dict with keys "is_results", "mc_results", "oos_results" (all DataFrames)
    """
    strategy_dir = strategy_reports_dir(strategy_name)
    ts = datetime.now().strftime('%Y%m%d_%H%M')

    cls      = StrategyRegistry.get(strategy_name)
    strategy = cls()
    param_grid  = param_grid_override if param_grid_override is not None else strategy.param_grid
    param_keys  = list(param_grid.keys())
    db_host     = getattr(strategy, 'db_host', None)

    if not param_grid:
        raise ValueError(f"Strategy '{strategy_name}' has an empty param_grid.")

    # Determine symbol — use strategy attribute or require argument
    if symbol is None:
        symbol = getattr(strategy, 'symbol', None)
    if symbol is None:
        raise ValueError(
            f"No symbol provided and strategy '{strategy_name}' has no 'symbol' attribute. "
            "Pass --symbol or set strategy.symbol."
        )

    sym_safe = symbol.replace('=', '_')

    # Validate rank_by
    if rank_by not in RANK_METRICS:
        raise ValueError(
            f"Unknown rank_by='{rank_by}'. Valid options: {list(RANK_METRICS.keys())}"
        )
    rank_col, rank_asc = RANK_METRICS[rank_by]

    # Apply symbol metadata to the strategy instance (tick_size, tick_value, commission_rt)
    symbol_meta = get_meta(symbol)
    strategy.tick_size     = symbol_meta['tick_size']
    strategy.tick_value    = symbol_meta['tick_value']
    strategy.commission_rt = symbol_meta['commission']
    print(f"  Symbol metadata: tick_size={symbol_meta['tick_size']}, "
          f"tick_value=${symbol_meta['tick_value']}, commission_rt=${symbol_meta['commission']}")

    # ------------------------------------------------------------------
    # Stage 1: Load data
    # ------------------------------------------------------------------
    if bar_type is not None:
        strategy.bar_type = bar_type   # CLI / caller override
    _bar_type        = getattr(strategy, 'bar_type', 'time')
    is_tick_strategy = _bar_type == 'tick'
    is_1m_strategy   = _bar_type == '1m'
    tick_bar_sizes   = param_grid.pop('tick_bar_size', None) if is_tick_strategy else None
    # Update param_keys after possible pop
    param_keys = list(param_grid.keys())
    if tick_bar_sizes is not None:
        param_keys = ['tick_bar_size'] + param_keys

    # Human-readable bar-type descriptor, persisted so the results browser can
    # show the timeframe a run used (mirrors the backtest `meta.data_source`).
    if is_tick_strategy:
        _sizes = tick_bar_sizes or []
        bar_type_desc = ("Tick bars (" + ", ".join(f"{s}t" for s in _sizes) + ")"
                         if _sizes else "Tick bars")
        bar_period = None
    elif is_1m_strategy:
        bar_type_desc, bar_period = "MySQL 1M", "1m"
    else:
        bar_type_desc, bar_period = "MySQL 5M", "5m"

    print(f"\n[1/4] Loading {symbol} data for '{strategy_name}'...")

    if is_tick_strategy:
        # For tick strategies we load one representative bar size just to get date metadata.
        # The actual per-bar-size loading happens inside the grid search.
        _sample_size = tick_bar_sizes[0] if tick_bar_sizes else 1300
        df_sample    = load_tick_bars(symbol, bar_size=_sample_size,
                                      start=data_start, end=data_end, host=db_host)
        print(f"      {len(df_sample):,} tick bars (sample at {_sample_size}-tick, for date metadata)")
    elif is_1m_strategy:
        df_sample = load_1m(symbol, start=data_start, end=data_end, host=db_host)
    else:
        df_sample = load_5m(symbol, start=data_start, end=data_end, refresh=refresh, host=db_host)

    if df_sample.empty:
        raise ValueError(
            f"No data found for symbol '{symbol}' (bar_type='{_bar_type}') between {data_start} and {data_end}. "
            "Check your date range, symbol name, and DB connection."
        )

    # Resample the loaded base to the user-selected primary timeframe.
    # Native base is 5M (time) / 1M (1m); a no-op when timeframe_mins matches.
    # Tick strategies are unaffected — their timeframe is tick-count based.
    _native_min = 1 if is_1m_strategy else 5
    if not is_tick_strategy and timeframe_mins and timeframe_mins != _native_min:
        df_sample     = resample_ohlcv(df_sample, timeframe_mins)
        bar_type_desc = f"{bar_type_desc} → {timeframe_mins}M"
        bar_period    = f"{timeframe_mins}m"

    if is_1m_strategy:
        print(f"      {len(df_sample):,} {timeframe_mins or 1}M bars  ({df_sample.index[0].date()} → {df_sample.index[-1].date()})")
    elif not is_tick_strategy:
        print(f"      {len(df_sample):,} {timeframe_mins or 5}M bars  ({df_sample.index[0].date()} → {df_sample.index[-1].date()})")

    _, _, cutoff = is_oos_split(df_sample, train_pct=train_pct)

    run_meta = {
        '_run_ts':      ts,
        '_data_start':  str(df_sample.index[0].date()),
        '_data_end':    str(df_sample.index[-1].date()),
        '_is_cutoff':   str(cutoff.date()),
        '_oos_start':   str(df_sample[df_sample.index >= cutoff].index[0].date()),
        '_oos_end':     str(df_sample.index[-1].date()),
    }

    # ------------------------------------------------------------------
    # Stage 2: IS grid search
    # ------------------------------------------------------------------

    if is_tick_strategy and tick_bar_sizes:
        # Outer loop over tick_bar_size: load bars + prepare once per size,
        # then sweep the inner parameter grid.
        inner_param_keys = [k for k in param_keys if k != 'tick_bar_size']
        inner_grid       = {k: param_grid[k] for k in inner_param_keys}
        _param_deps = getattr(strategy, 'param_dependencies', {})
        inner_combos     = list(_deduplicated_combinations(inner_grid, _param_deps))
        total            = len(tick_bar_sizes) * len(inner_combos)

        gh              = _grid_hash({**{'tick_bar_size': tick_bar_sizes}, **param_grid}, data_start, data_end)
        checkpoint_path = os.path.join(strategy_dir, f'checkpoint_{strategy_name}_{sym_safe}_{gh}.csv')
        completed_keys, prior_results = _load_checkpoint(checkpoint_path, param_keys)
        new_results = list(prior_results)
        done_count  = len(completed_keys)

        print(f"\n[2/4] Grid search: {total:,} combinations "
              f"({len(tick_bar_sizes)} bar sizes × {len(inner_combos):,} inner combos)")
        if done_count:
            print(f"      Resuming: {done_count:,} done.")

        n_workers = max(1, mp.cpu_count() - 1)

        for bar_size in tick_bar_sizes:
            print(f"\n  ── bar_size={bar_size} ──")
            bars_all = load_tick_bars(symbol, bar_size=bar_size,
                                      start=data_start, end=data_end, host=db_host)
            df_is, df_oos, _ = is_oos_split(bars_all, train_pct=train_pct)
            print(f"      IS: {len(df_is):,} bars  OOS: {len(df_oos):,} bars")

            prepared_is_bs = strategy.prepare_data(df_is)

            # Tag inner combos with this bar size
            tagged_combos = [{**c, 'tick_bar_size': bar_size} for c in inner_combos
                             if _combo_key({**c, 'tick_bar_size': bar_size}, param_keys)
                             not in completed_keys]

            with mp.Pool(
                processes   = n_workers,
                initializer = _init_worker,
                initargs    = (prepared_is_bs, strategy_name, symbol_meta),
            ) as pool:
                for j, result in enumerate(
                    pool.imap_unordered(_run_combo, tagged_combos, chunksize=10), 1
                ):
                    new_results.append(result)
                    done_count += 1
                    if done_count % CHECKPOINT_INTERVAL == 0 or j == len(tagged_combos):
                        pd.DataFrame(new_results).to_csv(checkpoint_path, index=False)
                        print(f"      {done_count:>6,}/{total:,}  ({done_count/total*100:.1f}%)  — checkpoint saved")

        # Lazy cache: load and prepare a bar size only when first needed (avoids
        # holding all 100+ sizes in memory simultaneously).
        _tick_cache: Dict[int, tuple] = {}

        def _get_tick_prepared(bar_size: int):
            if bar_size not in _tick_cache:
                bars_all = load_tick_bars(symbol, bar_size=bar_size,
                                          start=data_start, end=data_end, host=db_host)
                df_is_bs, df_oos_bs, _ = is_oos_split(bars_all, train_pct=train_pct)
                _tick_cache[bar_size] = (
                    strategy.prepare_data(df_is_bs),
                    strategy.prepare_data(df_oos_bs),
                )
            return _tick_cache[bar_size]

    else:
        df_is, df_oos, cutoff = is_oos_split(df_sample, train_pct=train_pct)
        print(f"      IS:  {len(df_is):,} bars  (up to {cutoff.date()})")
        print(f"      OOS: {len(df_oos):,} bars  ({cutoff.date()} →)")

        run_meta.update({
            '_is_start': str(df_is.index[0].date()),
            '_is_end':   str(df_is.index[-1].date()),
        })

        print(f"      Pre-processing IS and OOS data...")
        prepared_is  = strategy.prepare_data(df_is)
        prepared_oos = strategy.prepare_data(df_oos)

        if isinstance(prepared_is, list):
            print(f"      IS:  {len(prepared_is):,} units")
            print(f"      OOS: {len(prepared_oos):,} units")

        _param_deps = getattr(strategy, 'param_dependencies', {})
        all_combos = list(_deduplicated_combinations(param_grid, _param_deps))
        total      = len(all_combos)

        gh              = _grid_hash(param_grid, data_start, data_end)
        checkpoint_path = os.path.join(strategy_dir, f'checkpoint_{strategy_name}_{sym_safe}_{gh}.csv')
        completed_keys, prior_results = _load_checkpoint(checkpoint_path, param_keys)
        pending_combos  = [c for c in all_combos if _combo_key(c, param_keys) not in completed_keys]

        print(f"\n[2/4] Grid search: {total:,} combinations")
        if completed_keys:
            print(f"      Resuming: {len(completed_keys):,} done, {len(pending_combos):,} remaining.")
        else:
            print(f"      Running all {total:,} combinations.")

        n_workers   = max(1, mp.cpu_count() - 1)
        new_results = list(prior_results)

        print(f"      {n_workers} worker(s)...")
        with mp.Pool(
            processes   = n_workers,
            initializer = _init_worker,
            initargs    = (prepared_is, strategy_name, symbol_meta),
        ) as pool:
            for i, result in enumerate(
                pool.imap_unordered(_run_combo, pending_combos, chunksize=10), 1
            ):
                new_results.append(result)

                if i % CHECKPOINT_INTERVAL == 0 or i == len(pending_combos):
                    pd.DataFrame(new_results).to_csv(checkpoint_path, index=False)
                    done = len(completed_keys) + i
                    print(f"      {done:>6,}/{total:,}  ({done/total*100:.1f}%)  — checkpoint saved")

    df_results = pd.DataFrame(new_results)
    trades_col = 'total_trades' if 'total_trades' in df_results.columns else 'trades'
    df_valid   = df_results[df_results.get(trades_col, 0) >= min_trades].copy()
    if df_valid.empty:
        print(f"\n  WARNING: No combos with ≥{min_trades} trades. Lowering to 10.")
        df_valid = df_results[df_results.get(trades_col, 0) >= 10].copy()

    if df_valid.empty:
        print(f"\n  ERROR: No combos had any trades. Check your date range and parameters.")
        sys.exit(1)

    if rank_col not in df_valid.columns:
        print(f"  WARNING: rank_by='{rank_by}' column '{rank_col}' not found in results. Falling back to sharpe.")
        rank_col, rank_asc = RANK_METRICS['sharpe']
    df_valid = df_valid.sort_values(rank_col, ascending=rank_asc)

    for k, v in run_meta.items():
        df_valid[k] = v

    is_csv = os.path.join(strategy_dir, f'IS_{strategy_name}_{sym_safe}_{ts}.csv')
    df_valid.to_csv(is_csv, index=False)
    print(f"\n  IS results saved: {is_csv}")

    rank_label = rank_by.replace('_', ' ').title()
    _print_top(df_valid, param_keys, n=20, label=f"Top 20 IS by {rank_label}")
    _save_heatmap(df_valid, param_keys, ts, strategy_name, symbol, rank_col)

    # ------------------------------------------------------------------
    # Stage 3: Monte Carlo on top IS combos
    # ------------------------------------------------------------------
    print(f"\n[3/4] Monte Carlo ({mc_sims} sims × top {mc_top_n} combos)...")
    mc_rows = []

    for rank, (_, row) in enumerate(df_valid.head(mc_top_n).iterrows(), 1):
        params      = {k: row[k] for k in param_keys}
        full_params = {**strategy.default_params, **params}
        if is_tick_strategy and tick_bar_sizes:
            bar_size_val = int(row.get('tick_bar_size', tick_bar_sizes[0]))
            mc_prepared, _ = _get_tick_prepared(bar_size_val)
            mc = strategy.run_monte_carlo(mc_prepared, full_params, n_sims=mc_sims)
        else:
            mc = strategy.run_monte_carlo(prepared_is, full_params, n_sims=mc_sims)

        mc_row = dict(params)
        for col in ['total_trades', 'trades', 'net_pnl', 'sharpe', 'max_drawdown', 'bs_sharpe_p5']:
            if col in row:
                mc_row[col] = row[col]
        mc_row.update(mc)
        mc_rows.append(mc_row)

        stability_str = f"{mc['mc_stability']:.2f}" if not pd.isna(mc['mc_stability']) else "n/a"
        pnl_p5_str    = f"${mc['mc_pnl_p5']:,.0f}" if not pd.isna(mc['mc_pnl_p5']) else "n/a"
        pnl_p50_str   = f"${mc['mc_pnl_p50']:,.0f}" if not pd.isna(mc['mc_pnl_p50']) else "n/a"
        print(f"      [{rank:>2}/{MC_TOP_N}]  stability={stability_str}  "
              f"mc_pnl_p5={pnl_p5_str}  mc_pnl_p50={pnl_p50_str}")

    if not mc_rows:
        print("\n  ERROR: No combos survived to Monte Carlo stage (all had too few trades).")
        sys.exit(1)
    df_mc  = pd.DataFrame(mc_rows).sort_values('mc_stability', ascending=False)
    for k, v in run_meta.items():
        df_mc[k] = v
    mc_csv = os.path.join(strategy_dir, f'MC_{strategy_name}_{sym_safe}_{ts}.csv')
    df_mc.to_csv(mc_csv, index=False)
    print(f"\n  MC results saved: {mc_csv}")

    # ------------------------------------------------------------------
    # Stage 4: OOS validation
    # ------------------------------------------------------------------
    print(f"\n[4/4] OOS validation (top {oos_top_n} by MC stability)...")
    oos_rows = []

    for _, row in df_mc.head(oos_top_n).iterrows():
        params      = {k: row[k] for k in param_keys}
        full_params = {**strategy.default_params, **params}
        if is_tick_strategy and tick_bar_sizes:
            bar_size_val = int(row.get('tick_bar_size', tick_bar_sizes[0]))
            _, oos_prepared = _get_tick_prepared(bar_size_val)
            oos_res = strategy.run_backtest_prepared(oos_prepared, full_params)
        else:
            oos_res = strategy.run_backtest_prepared(prepared_oos, full_params)
        oos_res.pop('trades', None)

        oos_row  = dict(params)
        for col in ['mc_stability', 'mc_sharpe_p5', 'mc_pnl_p50']:
            if col in row:
                oos_row[col] = row[col]
        oos_row.update({f'oos_{k}': v for k, v in oos_res.items()})
        oos_rows.append(oos_row)

    df_oos_out = pd.DataFrame(oos_rows)
    for k, v in run_meta.items():
        df_oos_out[k] = v
    oos_csv    = os.path.join(strategy_dir, f'OOS_{strategy_name}_{sym_safe}_{ts}.csv')
    df_oos_out.to_csv(oos_csv, index=False)
    print(f"\n  OOS results saved: {oos_csv}")
    _print_top(df_oos_out, param_keys, n=OOS_TOP_N, label="OOS Validation")

    # Persist to the shared results store so multiple machines see the same history.
    try:
        results_store.save_optimizer_run(
            strategy_name=strategy_name,
            symbol=symbol,
            run_ts=ts,
            run_meta=run_meta,
            settings={
                "data_source": bar_type_desc,
                "bar_type": _bar_type,
                "bar_period": bar_period,
                "tick_bar_sizes": tick_bar_sizes,
                "data_start": data_start,
                "data_end": data_end,
                "refresh": refresh,
                "train_pct": train_pct,
                "mc_sims": mc_sims,
                "mc_top_n": mc_top_n,
                "oos_top_n": oos_top_n,
                "min_trades": min_trades,
                "rank_by": rank_by,
                "param_keys": param_keys,
                **(run_settings or {}),
            },
            stage_frames={
                "IS": df_valid,
                "MC": df_mc,
                "OOS": df_oos_out,
            },
        )
        print("  Shared results store updated.")
    except Exception as e:
        print(f"  WARNING: failed to update shared results store: {e}")

    # Cleanup checkpoint
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"\n  Checkpoint removed.")

    print(f"\nDone. Reports saved to {strategy_dir}/")

    return {
        'is_results':  df_valid,
        'mc_results':  df_mc,
        'oos_results': df_oos_out,
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _print_top(df: pd.DataFrame, param_keys: List[str], n: int, label: str) -> None:
    display_cols = param_keys + [
        c for c in ['total_trades', 'trades', 'win_rate', 'net_pnl', 'sharpe',
                    'profit_factor', 'max_drawdown', 'bs_sharpe_p5', 'bs_pnl_p5',
                    'mc_stability', 'mc_pnl_p5', 'mc_pnl_p50',
                    'oos_net_pnl', 'oos_sharpe', 'oos_win_rate']
        if c in df.columns
    ]
    print(f"\n── {label} {'─' * max(0, 60 - len(label))}")
    print(df[display_cols].head(n).to_string(index=False))


def _save_heatmap(
    df: pd.DataFrame,
    param_keys: List[str],
    ts: str,
    strategy_name: str,
    symbol: str,
    metric: str = 'sharpe',
) -> None:
    """Save a heatmap for the first two numeric param dimensions, coloured by *metric*."""
    numeric_keys = [k for k in param_keys if pd.api.types.is_numeric_dtype(df[k])]
    if len(numeric_keys) < 2:
        return
    if metric not in df.columns:
        metric = 'sharpe'
    x_key, y_key = numeric_keys[0], numeric_keys[1]
    metric_label = metric.replace('_', ' ').title()
    try:
        pivot = df.groupby([y_key, x_key])[metric].mean().unstack()
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(pivot.values, aspect='auto', cmap='RdYlGn')
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)
        ax.set_title(f"{symbol} {strategy_name} — Avg {metric_label} by {y_key} × {x_key} (IS)")
        plt.colorbar(im, ax=ax, label=f'Avg {metric_label}')
        path = os.path.join(strategy_reports_dir(strategy_name), f'heatmap_{strategy_name}_{symbol.replace("=","_")}_{ts}.png')
        plt.savefig(path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  Heatmap saved: {path}")
    except Exception as e:
        print(f"  Heatmap skipped: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the strategy optimization pipeline.")
    parser.add_argument('--strategy',      required=True,       help='Strategy name (e.g. goldbot7)')
    parser.add_argument('--symbol',        default=None,        help='Symbol override (e.g. GC=F)')
    parser.add_argument('--bar-type',      default=None,        choices=['time', '1m', 'tick'],
                                                                 help='Override strategy bar type (time / 1m / tick)')
    parser.add_argument('--timeframe-mins', type=int, default=None,
                                                                 help='Primary timeframe in minutes; resamples the base (5M/1M) to this size (e.g. 15, 30, 60, 240)')
    parser.add_argument('--start',         default=None,        help='Data start date (ISO, e.g. 2024-03-01)')
    parser.add_argument('--end',           default=None,        help='Data end date (ISO)')
    parser.add_argument('--refresh',       action='store_true', help='Force MySQL reload')
    parser.add_argument('--train-pct',     type=float, default=0.70,              help='IS fraction (default 0.70)')
    parser.add_argument('--mc-sims',       type=int,   default=MONTE_CARLO_SIMS,  help=f'Monte Carlo simulations (default {MONTE_CARLO_SIMS})')
    parser.add_argument('--mc-top-n',      type=int,   default=MC_TOP_N,          help=f'Top IS combos to MC-test (default {MC_TOP_N})')
    parser.add_argument('--oos-top-n',     type=int,   default=OOS_TOP_N,         help=f'Top MC combos to OOS-validate (default {OOS_TOP_N})')
    parser.add_argument('--min-trades',    type=int,   default=MIN_TRADES,        help=f'Min trades to include a combo (default {MIN_TRADES})')
    parser.add_argument('--rank-by',       default='sharpe', choices=list(RANK_METRICS.keys()),
                                                                                   help='IS ranking metric (default: sharpe)')
    parser.add_argument('--param-grid',    default=None, help='JSON param grid override, e.g. \'{"stop_fib":[0.90,0.95]}\'')
    parser.add_argument('--run-settings',  default=None, help='JSON blob of Configure & Run settings to persist with the run')
    args = parser.parse_args()

    param_grid_override = json.loads(args.param_grid) if args.param_grid else None
    run_settings = json.loads(args.run_settings) if args.run_settings else None

    run_pipeline(
        strategy_name       = args.strategy,
        symbol              = args.symbol,
        bar_type            = args.bar_type,
        timeframe_mins      = args.timeframe_mins,
        data_start          = args.start,
        data_end            = args.end,
        refresh             = args.refresh,
        train_pct           = args.train_pct,
        mc_sims             = args.mc_sims,
        mc_top_n            = args.mc_top_n,
        oos_top_n           = args.oos_top_n,
        min_trades          = args.min_trades,
        rank_by             = args.rank_by,
        param_grid_override = param_grid_override,
        run_settings        = run_settings,
    )
