"""
Aggregate Stage B outputs across all core sub-runs.

Reads optimize_runs/stage_b_run_log.txt (one line per core: '<core_label>  <oos_csv>'),
concatenates the OOS CSVs, applies a DD cap, ranks by oos_sortino + oos_net_pnl,
emits top-N finalists ready for walk-forward validation.

Outputs:
  optimize_runs/stage_b_finalists.json   - full param dicts for top N
  optimize_runs/wfo_grid.json            - union grid for walk-forward
  optimize_runs/stage_b_summary.txt      - human-readable log

Usage:
    python optimize_runs/aggregate_stage_b.py [TOP_N] [DD_CAP_DOLLARS]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT_DIR = ROOT / "optimize_runs"

DEFAULT_DD_CAP = 2000.0
DEFAULT_TOP_N = 5

PARAM_KEYS = [
    "atr_multiplier", "atr_period", "fractal_length",
    "direction", "invert_signals",
    "exit_mode", "tpsl_mode",
    "tp_ticks", "sl_ticks",
    "tp_atr_mult", "sl_atr_mult",
    "rr_ratio",
    "use_risk_sizing", "qty", "max_risk",
    "bars_between_trades",
    "enable_session_filter",
    "trade_window1_start", "trade_window1_stop",
    "enable_trade_window2", "trade_window2_start", "trade_window2_stop",
    "enable_trade_window3", "trade_window3_start", "trade_window3_stop",
    "eod_exit_time",
]


def main() -> int:
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TOP_N
    dd_cap = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DD_CAP

    log = OUT_DIR / "stage_b_run_log.txt"
    if not log.exists():
        print(f"ERROR: {log} missing", file=sys.stderr)
        return 2

    csvs: list[Path] = []
    for line in log.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            csv = REPORTS / parts[1]
            if csv.exists():
                csvs.append(csv)
    if not csvs:
        print("ERROR: no Stage B OOS CSVs found", file=sys.stderr)
        return 3

    print(f"  aggregating {len(csvs)} Stage B OOS files")
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    print(f"  total rows: {len(df)}")

    if "oos_max_drawdown" not in df.columns:
        print("  ERROR: oos_max_drawdown column missing", file=sys.stderr)
        return 4

    df["_dd_abs"] = df["oos_max_drawdown"].abs()
    before = len(df)
    df = df[df["_dd_abs"] <= dd_cap].copy()
    print(f"  after |DD|<=${dd_cap:.0f}: {len(df)}/{before}")
    if len(df) == 0:
        print("  ERROR: no Stage B combos passed DD filter", file=sys.stderr)
        return 5

    # Filter to profitable only
    before = len(df)
    df = df[df["oos_net_pnl"] > 0]
    print(f"  after profitable: {len(df)}/{before}")
    if len(df) == 0:
        print("  ERROR: no Stage B combos profitable", file=sys.stderr)
        return 6

    # Rank: Sortino primary, net_pnl secondary
    sort_cols = []
    for col in ("oos_sortino", "oos_sharpe", "oos_net_pnl"):
        if col in df.columns:
            sort_cols.append(col)
    df = df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))

    have = [k for k in PARAM_KEYS if k in df.columns]
    top = df.head(top_n)
    finalists = []
    summary = [
        f"Stage B finalists (top {top_n}, |DD|<=${dd_cap:.0f}, profitable, by OOS Sortino/net_pnl)",
        f"  Aggregated CSVs: {[c.name for c in csvs]}",
        "",
    ]
    for i, (_, row) in enumerate(top.iterrows(), 1):
        params = {k: row[k] for k in have}
        for k, v in params.items():
            if hasattr(v, "item"):
                params[k] = v.item()
            if isinstance(v, bool):
                params[k] = bool(v)
        finalists.append(params)
        summary.append(
            f"  #{i}: sortino={row.get('oos_sortino', 0):.3f}  "
            f"sharpe={row.get('oos_sharpe', 0):.3f}  "
            f"net_pnl=${row.get('oos_net_pnl', 0):,.0f}  "
            f"dd=${row.get('oos_max_drawdown', 0):,.0f}  "
            f"trades={int(row.get('oos_total_trades', 0))}  "
            f"wr={row.get('oos_win_rate', 0):.1%}  "
            f"win1={row.get('trade_window1_start')}->{row.get('trade_window1_stop')}  "
            f"exit={row.get('exit_mode')}/{row.get('tpsl_mode')}  "
            f"atr={row.get('atr_multiplier')}x{int(row.get('atr_period', 0))} "
            f"frac={int(row.get('fractal_length', 0))} "
            f"tp_atr={row.get('tp_atr_mult')} sl_atr={row.get('sl_atr_mult')} "
            f"cd={int(row.get('bars_between_trades', 0))}"
        )

    (OUT_DIR / "stage_b_finalists.json").write_text(json.dumps(finalists, indent=2))

    # Build union grid for WFO (each param: list of unique values across finalists)
    union: dict[str, list] = {}
    for p in finalists:
        for k, v in p.items():
            union.setdefault(k, [])
            if v not in union[k]:
                union[k].append(v)
    (OUT_DIR / "wfo_grid.json").write_text(json.dumps(union, indent=2))

    (OUT_DIR / "stage_b_summary.txt").write_text("\n".join(summary))
    print("\n".join(summary))
    print(f"\n  wrote: stage_b_finalists.json  ({len(finalists)} configs)")
    print("  wrote: wfo_grid.json (union of finalists)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
