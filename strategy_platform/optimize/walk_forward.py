"""
Sliding-window walk-forward optimisation (WFO) for the strategy platform.

Algorithm
---------
For each step k:
  IS  window: [data_start + k*step, data_start + k*step + is_window]
  OOS window: [is_end, is_end + oos_window]

IS grid search is parallelised with the same mp.Pool pattern as pipeline.py.
No MC stage — too expensive across many slices.  Best IS combo goes straight
to OOS.

Usage (CLI)
-----------
    python -m strategy_platform.optimize.walk_forward \\
        --strategy patscalp --symbol MNQ \\
        --start 2024-09-01 --end 2026-01-01

Usage (library)
---------------
    from strategy_platform.optimize.walk_forward import run_walk_forward
    result = run_walk_forward("patscalp", "MNQ", data_start="2024-09-01", data_end="2026-01-01")
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import strategy_platform  # noqa: F401 — triggers auto-registration of all strategies
from strategy_platform.data.loader import get_meta, load_1m, load_5m, load_tick_bars
from strategy_platform.registry import StrategyRegistry

# Re-use helpers and constants from pipeline — no duplication.
from strategy_platform.optimize.pipeline import (
    RANK_METRICS,
    REPORTS_DIR,
    _deduplicated_combinations,
    _grid_combinations,  # noqa: F401 — imported for completeness; _deduplicated_combinations wraps it
    _init_worker,
    _run_combo,
)

# ---------------------------------------------------------------------------
# JSON serialisation helper
# ---------------------------------------------------------------------------

class _SafeEncoder(json.JSONEncoder):
    """Convert Timestamps → ISO strings and NaN/Inf → null."""

    def default(self, o: Any) -> Any:
        if isinstance(o, pd.Timestamp):
            return o.isoformat()
        if isinstance(o, (datetime, date)):
            return o.isoformat()
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if (math.isnan(v) or math.isinf(v)) else v
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return super().default(o)

    def encode(self, o: Any) -> str:
        # Intercept float NaN/Inf at the top level too
        if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
            return "null"
        return super().encode(o)

    def iterencode(self, o: Any, _one_shot: bool = False):
        # Recursively sanitise dicts/lists
        return super().iterencode(self._sanitise(o), _one_shot)

    def _sanitise(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._sanitise(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitise(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            return None if (math.isnan(v) or math.isinf(v)) else v
        return obj


# ---------------------------------------------------------------------------
# Slice boundary computation
# ---------------------------------------------------------------------------

def _compute_slices(
    data_start: str,
    data_end: str,
    is_window_days: int,
    oos_window_days: int,
    step_days: int,
) -> List[Dict[str, str]]:
    """Return list of {is_start, is_end, oos_start, oos_end} boundary dicts."""
    start = pd.Timestamp(data_start)
    end   = pd.Timestamp(data_end)
    slices: List[Dict[str, str]] = []
    k = 0
    while True:
        is_start  = start + timedelta(days=k * step_days)
        is_end    = is_start + timedelta(days=is_window_days)
        oos_start = is_end
        oos_end   = oos_start + timedelta(days=oos_window_days)
        if oos_end > end:
            break
        slices.append({
            "is_start":  is_start.strftime("%Y-%m-%d"),
            "is_end":    is_end.strftime("%Y-%m-%d"),
            "oos_start": oos_start.strftime("%Y-%m-%d"),
            "oos_end":   oos_end.strftime("%Y-%m-%d"),
        })
        k += 1
    return slices


# ---------------------------------------------------------------------------
# Dataframe slicing by date
# ---------------------------------------------------------------------------

def _slice_df(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """
    Return rows where index is in [start, end).

    Uses the DatetimeIndex directly (same convention as pipeline.py's
    is_oos_split: end boundary is exclusive so adjacent slices don't overlap).
    """
    ts_start = pd.Timestamp(start)
    ts_end   = pd.Timestamp(end)
    return df[(df.index >= ts_start) & (df.index < ts_end)]


# ---------------------------------------------------------------------------
# Metrics extraction
# ---------------------------------------------------------------------------

_METRIC_COLS = ["sharpe", "sortino", "net_pnl", "profit_factor", "max_drawdown",
                "win_rate", "trades", "total_trades"]


def _extract_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
    """Pull the scalar metrics we care about from a backtest result dict."""
    metrics: Dict[str, Any] = {}
    for col in _METRIC_COLS:
        if col in result:
            v = result[col]
            metrics[col] = None if (isinstance(v, float) and (math.isnan(v) or math.isinf(v))) else v
    # Normalise trade count to a single 'trades' key
    if "trades" not in metrics and "total_trades" in metrics:
        metrics["trades"] = metrics["total_trades"]
    return metrics


def _trades_to_records(trades: Any) -> List[Dict[str, Any]]:
    """Convert trades (DataFrame or list of dicts) to a JSON-safe list."""
    if trades is None:
        return []
    if isinstance(trades, pd.DataFrame):
        records = trades.to_dict("records")
    else:
        records = list(trades)

    safe: List[Dict[str, Any]] = []
    for rec in records:
        row: Dict[str, Any] = {}
        for k, v in rec.items():
            key = str(k)
            if isinstance(v, pd.Timestamp):
                row[key] = v.isoformat()
            elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                row[key] = None
            elif isinstance(v, (np.integer,)):
                row[key] = int(v)
            elif isinstance(v, (np.floating,)):
                row[key] = None if (math.isnan(float(v)) or math.isinf(float(v))) else float(v)
            else:
                row[key] = v
        safe.append(row)
    return safe


# ---------------------------------------------------------------------------
# IS sweep for a single slice
# ---------------------------------------------------------------------------

def _run_is_sweep(
    prepared_is: Any,
    strategy_name: str,
    symbol_meta: dict,
    param_grid: Dict[str, List[Any]],
    param_dependencies: Dict[str, tuple],
    min_trades: int,
    rank_col: str,
    rank_asc: bool,
    n_workers: int,
) -> tuple[Optional[Dict[str, Any]], Dict[str, Any], int]:
    """
    Parallel IS sweep.  Returns (best_params, best_is_metrics, n_combos).
    best_params is None when no combo clears min_trades.
    """
    all_combos = list(_deduplicated_combinations(param_grid, param_dependencies))
    n_combos   = len(all_combos)

    if n_combos == 0:
        return None, {}, 0

    with mp.Pool(
        processes   = n_workers,
        initializer = _init_worker,
        initargs    = (prepared_is, strategy_name, symbol_meta),
    ) as pool:
        results = list(pool.imap_unordered(_run_combo, all_combos, chunksize=10))

    df = pd.DataFrame(results)
    trades_col = "total_trades" if "total_trades" in df.columns else "trades"
    if trades_col not in df.columns:
        return None, {}, n_combos

    df_valid = df[df[trades_col] >= min_trades].copy()
    if df_valid.empty:
        return None, {}, n_combos

    if rank_col not in df_valid.columns:
        # Fall back gracefully
        rank_col, rank_asc = RANK_METRICS["sharpe"]
    if rank_col not in df_valid.columns:
        return None, {}, n_combos

    df_valid = df_valid.sort_values(rank_col, ascending=rank_asc)
    best_row  = df_valid.iloc[0]
    param_keys = list(param_grid.keys())
    best_params = {k: best_row[k] for k in param_keys if k in best_row}
    best_metrics = _extract_metrics({str(k): v for k, v in best_row.to_dict().items()})
    return best_params, best_metrics, n_combos


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_walk_forward(
    strategy_name: str,
    symbol: str,
    bar_type: Optional[str] = None,
    data_start: Optional[str] = None,
    data_end: Optional[str] = None,
    is_window_days: int = 180,
    oos_window_days: int = 30,
    step_days: int = 30,
    rank_by: str = "sharpe",
    min_trades: int = 20,
    param_grid_override: Optional[Dict[str, List[Any]]] = None,
    refresh: bool = False,
    run_settings: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run sliding-window walk-forward optimisation for *strategy_name*.

    Parameters
    ----------
    strategy_name       : Registered strategy name, e.g. "patscalp".
    symbol              : Instrument symbol, e.g. "MNQ".
    bar_type            : Override strategy bar_type ('time' | '1m' | 'tick').
    data_start          : ISO date — first day of first IS slice.
    data_end            : ISO date — last day eligible for OOS coverage.
    is_window_days      : Length of the IS window (calendar days).
    oos_window_days     : Length of each OOS slice (calendar days).
    step_days           : How far to advance the window each step.
    rank_by             : IS ranking metric key (see RANK_METRICS).
    min_trades          : Minimum IS trades for a combo to be considered valid.
    param_grid_override : Replace strategy.param_grid entirely if provided.
    refresh             : Force MySQL reload (bypass Parquet cache).
    run_settings        : Arbitrary metadata blob stored in the JSON output.

    Returns
    -------
    dict with keys "meta" and "slices" matching the WF JSON schema.

    Raises
    ------
    ValueError  if fewer than 2 slices can be formed from the given date range.
    """
    if data_start is None or data_end is None:
        raise ValueError("data_start and data_end are required.")

    os.makedirs(REPORTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    # ------------------------------------------------------------------
    # Strategy & grid setup
    # ------------------------------------------------------------------
    cls      = StrategyRegistry.get(strategy_name)
    strategy = cls()

    if bar_type is not None:
        strategy.bar_type = bar_type
    _bar_type = getattr(strategy, "bar_type", "time")

    param_grid = param_grid_override if param_grid_override is not None else strategy.param_grid
    if not param_grid:
        raise ValueError(f"Strategy '{strategy_name}' has an empty param_grid.")

    param_keys         = list(param_grid.keys())
    param_dependencies = getattr(strategy, "param_dependencies", {})
    db_host            = getattr(strategy, "db_host", None)

    # Validate rank_by
    if rank_by not in RANK_METRICS:
        raise ValueError(
            f"Unknown rank_by='{rank_by}'. Valid options: {list(RANK_METRICS.keys())}"
        )
    rank_col, rank_asc = RANK_METRICS[rank_by]

    sym_safe = symbol.replace("=", "_")

    # Apply symbol metadata
    symbol_meta = get_meta(symbol)
    strategy.tick_size     = symbol_meta["tick_size"]
    strategy.tick_value    = symbol_meta["tick_value"]
    strategy.commission_rt = symbol_meta["commission"]
    print(f"  Symbol metadata: tick_size={symbol_meta['tick_size']}, "
          f"tick_value=${symbol_meta['tick_value']}, commission_rt=${symbol_meta['commission']}")

    # ------------------------------------------------------------------
    # Load full-range data ONCE
    # ------------------------------------------------------------------
    is_tick_strategy = _bar_type == "tick"
    is_1m_strategy   = _bar_type == "1m"

    # For tick strategies: tick_bar_size must be a param_grid key or a single
    # value in strategy.default_params.  We handle the single-size case here
    # (multi-size tick-bar WFO would be a separate concern).
    tick_bar_size: Optional[int] = None
    if is_tick_strategy:
        if "tick_bar_size" in param_grid:
            sizes = param_grid.pop("tick_bar_size")
            param_keys = list(param_grid.keys())
            # For WFO we use only the first tick_bar_size to keep runtime sane.
            # The caller can override with param_grid_override to pin a specific size.
            tick_bar_size = int(sizes[0])
            print(f"  [WFO] tick strategy: using tick_bar_size={tick_bar_size} "
                  f"(first value only; multi-size WFO not supported).")
        else:
            tick_bar_size = int(getattr(strategy, "default_params", {}).get("tick_bar_size", 1300))

    print(f"\n[WFO] Loading full-range {symbol} data ({data_start} → {data_end})...")

    if is_tick_strategy:
        df_full = load_tick_bars(symbol, bar_size=tick_bar_size,
                                 start=data_start, end=data_end, host=db_host)
    elif is_1m_strategy:
        df_full = load_1m(symbol, start=data_start, end=data_end, host=db_host)
    else:
        df_full = load_5m(symbol, start=data_start, end=data_end,
                          refresh=refresh, host=db_host)

    if df_full.empty:
        raise ValueError(
            f"No data for '{symbol}' (bar_type='{_bar_type}') between {data_start} and {data_end}."
        )
    print(f"      {len(df_full):,} bars  ({df_full.index[0].date()} → {df_full.index[-1].date()})")

    # ------------------------------------------------------------------
    # Compute slice boundaries
    # ------------------------------------------------------------------
    slices_meta = _compute_slices(data_start, data_end, is_window_days, oos_window_days, step_days)
    n_slices    = len(slices_meta)

    if n_slices < 2:
        raise ValueError(
            f"Only {n_slices} slice(s) can be formed. "
            f"Widen the date range or reduce is_window_days/oos_window_days."
        )

    print(f"[WFO] {n_slices} slices  (IS={is_window_days}d  OOS={oos_window_days}d  step={step_days}d)")

    n_workers = max(1, mp.cpu_count() - 1)

    # ------------------------------------------------------------------
    # Per-slice IS + OOS
    # ------------------------------------------------------------------
    slice_records: List[Dict[str, Any]] = []
    oos_pnl_total  = 0.0
    is_pnl_total   = 0.0

    for k, bounds in enumerate(slices_meta):
        is_start  = bounds["is_start"]
        is_end    = bounds["is_end"]
        oos_start = bounds["oos_start"]
        oos_end   = bounds["oos_end"]

        df_is  = _slice_df(df_full, is_start, is_end)
        df_oos = _slice_df(df_full, oos_start, oos_end)

        if df_is.empty or df_oos.empty:
            _which = []
            if df_is.empty:
                _which.append(f"IS {is_start}..{is_end}")
            if df_oos.empty:
                _which.append(f"OOS {oos_start}..{oos_end}")
            print(f"[Slice {k+1}/{n_slices}] SKIP — DB has 0 bars for: {' and '.join(_which)} "
                  f"(data gap, not a slice bug — reimport this date range to fix)")
            slice_records.append({
                "slice_idx":   k,
                "is_start":    is_start,
                "is_end":      is_end,
                "oos_start":   oos_start,
                "oos_end":     oos_end,
                "best_params": None,
                "is_metrics":  {},
                "oos_metrics": {},
                "oos_trades":  [],
            })
            continue

        # Prepare IS data
        prepared_is = strategy.prepare_data(df_is)

        # IS sweep
        best_params, is_metrics, n_combos = _run_is_sweep(
            prepared_is   = prepared_is,
            strategy_name = strategy_name,
            symbol_meta   = symbol_meta,
            param_grid    = param_grid,
            param_dependencies = param_dependencies,
            min_trades    = min_trades,
            rank_col      = rank_col,
            rank_asc      = rank_asc,
            n_workers     = n_workers,
        )

        if best_params is None:
            best_rank_val = float("nan")
            print(f"[Slice {k+1}/{n_slices}] IS {is_start}..{is_end} "
                  f"({n_combos} combos) → NO VALID COMBOS (all < {min_trades} trades) "
                  f"→ OOS {oos_start}..{oos_end} → skipped")
            slice_records.append({
                "slice_idx":   k,
                "is_start":    is_start,
                "is_end":      is_end,
                "oos_start":   oos_start,
                "oos_end":     oos_end,
                "best_params": None,
                "is_metrics":  {},
                "oos_metrics": {},
                "oos_trades":  [],
            })
            continue

        best_rank_val = is_metrics.get(rank_col, float("nan"))
        is_net_pnl    = is_metrics.get("net_pnl", 0.0) or 0.0

        # OOS evaluation with best combo
        full_params  = {**strategy.default_params, **best_params}
        prepared_oos = strategy.prepare_data(df_oos)
        oos_result   = strategy.run_backtest_prepared(prepared_oos, full_params)

        oos_trades_raw = oos_result.pop("trades", None)
        oos_metrics    = _extract_metrics(oos_result)
        oos_trades     = _trades_to_records(oos_trades_raw)
        oos_net_pnl    = oos_metrics.get("net_pnl", 0.0) or 0.0

        oos_pnl_total += oos_net_pnl
        is_pnl_total  += is_net_pnl

        oos_trades_count = len(oos_trades) if oos_trades else oos_metrics.get("trades", 0)
        rank_str = f"{best_rank_val:.3f}" if isinstance(best_rank_val, float) and not math.isnan(best_rank_val) else "n/a"
        print(
            f"[Slice {k+1}/{n_slices}] IS {is_start}..{is_end} "
            f"({n_combos} combos) → best {rank_by}={rank_str} "
            f"→ OOS {oos_start}..{oos_end} "
            f"→ net_pnl=${oos_net_pnl:,.0f} trades={oos_trades_count}"
        )

        slice_records.append({
            "slice_idx":   k,
            "is_start":    is_start,
            "is_end":      is_end,
            "oos_start":   oos_start,
            "oos_end":     oos_end,
            "best_params": best_params,
            "is_metrics":  is_metrics,
            "oos_metrics": oos_metrics,
            "oos_trades":  oos_trades,
        })

    # ------------------------------------------------------------------
    # Walk-forward efficiency
    # ------------------------------------------------------------------
    if is_pnl_total != 0.0:
        wfe = oos_pnl_total / is_pnl_total
    else:
        wfe = None

    print(f"\n[WFO] Complete.  OOS net_pnl=${oos_pnl_total:,.0f}  "
          f"IS net_pnl=${is_pnl_total:,.0f}  "
          f"WFE={wfe:.3f}" if wfe is not None else
          f"\n[WFO] Complete.  OOS net_pnl=${oos_pnl_total:,.0f}  WFE=n/a (IS pnl=0)")

    # ------------------------------------------------------------------
    # Assemble output
    # ------------------------------------------------------------------
    output: Dict[str, Any] = {
        "meta": {
            "strategy":        strategy_name,
            "symbol":          symbol,
            "bar_type":        _bar_type,
            "data_start":      data_start,
            "data_end":        data_end,
            "is_window_days":  is_window_days,
            "oos_window_days": oos_window_days,
            "step_days":       step_days,
            "rank_by":         rank_by,
            "min_trades":      min_trades,
            "n_slices":        n_slices,
            "wfe":             wfe,
            "param_keys":      param_keys,
            "run_settings":    run_settings or {},
            "ts":              ts,
        },
        "slices": slice_records,
    }

    # ------------------------------------------------------------------
    # Save JSON report
    # ------------------------------------------------------------------
    report_path = os.path.join(REPORTS_DIR, f"WF_{strategy_name}_{sym_safe}_{ts}.json")
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, cls=_SafeEncoder, indent=2)
    print(f"[WFO] Report saved: {report_path}")

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sliding-window walk-forward optimisation."
    )
    parser.add_argument("--strategy",        required=True,  help="Strategy name (e.g. patscalp)")
    parser.add_argument("--symbol",          required=True,  help="Symbol (e.g. MNQ)")
    parser.add_argument("--bar-type",        default=None,   choices=["time", "1m", "tick"],
                                                             help="Override strategy bar_type")
    parser.add_argument("--start",           required=True,  help="Data start date (ISO, e.g. 2024-09-01)")
    parser.add_argument("--end",             required=True,  help="Data end date (ISO, e.g. 2026-01-01)")
    parser.add_argument("--is-window-days",  type=int, default=180, help="IS window length in days (default 180)")
    parser.add_argument("--oos-window-days", type=int, default=30,  help="OOS slice length in days (default 30)")
    parser.add_argument("--step-days",       type=int, default=30,  help="Step between windows in days (default 30)")
    parser.add_argument("--rank-by",         default="sharpe", choices=list(RANK_METRICS.keys()),
                                                             help="IS ranking metric (default: sharpe)")
    parser.add_argument("--min-trades",      type=int, default=20,  help="Min IS trades to include a combo (default 20)")
    parser.add_argument("--param-grid",      default=None,   help='JSON param grid override, e.g. \'{"stop_ticks":[8,10,12]}\'')
    parser.add_argument("--run-settings",    default=None,   help="JSON blob of metadata to persist with the run")
    parser.add_argument("--refresh",         action="store_true", help="Force MySQL reload")
    args = parser.parse_args()

    _param_grid_override = json.loads(args.param_grid) if args.param_grid else None
    _run_settings        = json.loads(args.run_settings) if args.run_settings else None

    run_walk_forward(
        strategy_name       = args.strategy,
        symbol              = args.symbol,
        bar_type            = args.bar_type,
        data_start          = args.start,
        data_end            = args.end,
        is_window_days      = args.is_window_days,
        oos_window_days     = args.oos_window_days,
        step_days           = args.step_days,
        rank_by             = args.rank_by,
        min_trades          = args.min_trades,
        param_grid_override = _param_grid_override,
        refresh             = args.refresh,
        run_settings        = _run_settings,
    )
