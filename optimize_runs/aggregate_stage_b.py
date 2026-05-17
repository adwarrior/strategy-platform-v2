"""
Aggregate Stage B outputs across all core sub-runs.

After each Stage B sub-run, a new OOS_supertrendfractal_MNQ_*.csv is written
to reports/. This script:
  1. Reads optimize_runs/stage_b_run_log.txt to find the OOS CSVs for this
     Stage B batch (each line: 'core_<i>  <oos_csv_name>').
  2. Concatenates them, applies DD <= $2000 filter, ranks by OOS Sortino +
     net_pnl, emits the top N finalists.

Outputs:
  optimize_runs/stage_b_finalists.json   — full param dicts for top N
  optimize_runs/wfo_grid.json            — union grid for walk-forward
  optimize_runs/stage_b_summary.txt      — human-readable log
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT_DIR = ROOT / "optimize_runs"

DD_LIMIT = 2000.0
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

    log = OUT_DIR / "stage_b_run_log.txt"
    if not log.exists():
        print(f"ERROR: {log} missing")
        return 2

    csvs: list[Path] = []
    for line in log.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            csv = REPORTS / parts[1]
            if csv.exists():
                csvs.append(csv)
    if not csvs:
        print("ERROR: no Stage B OOS CSVs found")
        return 3

    print(f"  aggregating {len(csvs)} Stage B OOS files")
    df = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
    print(f"  total rows: {len(df)}")

    if "max_drawdown" in df.columns:
        df["max_drawdown"] = df["max_drawdown"].abs()
        before = len(df)
        df = df[df["max_drawdown"] <= DD_LIMIT]
        print(f"  after DD<=${DD_LIMIT:.0f}: {len(df)}/{before}")
    if len(df) == 0:
        print("  ERROR: no Stage B combos passed DD filter")
        return 4

    sort_cols = []
    for col in ("sortino", "net_pnl"):
        if col in df.columns:
            sort_cols.append(col)
    if not sort_cols:
        sort_cols = ["sharpe"]
    df = df.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))

    have = [k for k in PARAM_KEYS if k in df.columns]
    top = df.head(top_n)
    finalists = []
    summary = [f"Stage B finalists (top {top_n}, DD<=${DD_LIMIT:.0f}, by OOS Sortino/net_pnl)", ""]
    for i, (_, row) in enumerate(top.iterrows(), 1):
        params = {k: row[k] for k in have}
        for k, v in params.items():
            if hasattr(v, "item"):
                params[k] = v.item()
        finalists.append(params)
        summary.append(
            f"  #{i}: sortino={row.get('sortino', 0):.3f}  "
            f"net_pnl=${row.get('net_pnl', 0):.0f}  "
            f"dd=${row.get('max_drawdown', 0):.0f}  "
            f"trades={int(row.get('total_trades', row.get('trades', 0)))}  "
            f"win1={row.get('trade_window1_start')}->{row.get('trade_window1_stop')}  "
            f"exit={row.get('exit_mode')}/{row.get('tpsl_mode')}"
        )

    (OUT_DIR / "stage_b_finalists.json").write_text(json.dumps(finalists, indent=2))

    # Build union grid for WFO
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
    print(f"  wrote: wfo_grid.json (union of finalists)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
