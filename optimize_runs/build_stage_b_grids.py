"""
Build per-core Stage B param-grid JSON files for the session sweep.

Reads optimize_runs/stage_a_top_cores.json (list of core param dicts), and for
each core emits optimize_runs/stage_b_core_{i}.json — a param-grid override
that locks the core params and sweeps a coarse 12x12 session-window grid.

Stage B sessions tested per core:
  start ∈ {00:00, 02:00, 04:00, ..., 22:00}   (12 starts)
  stop  ∈ {04:00, 06:00, 08:00, ..., 23:55}   (12 stops)
  enable_session_filter = [True]
  windows 2 & 3 disabled
  144 combos per core × N cores
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CORES_FILE = ROOT / "stage_a_top_cores.json"

STARTS = [f"{h:02d}:00" for h in range(0, 24, 2)]  # 00:00,02:00,...,22:00 (12)
STOPS_RAW = [f"{h:02d}:00" for h in range(4, 24, 2)] + ["23:55"]  # 04..22 step 2 + 23:55 (11)
# pad to 12 by adding 02:00 (overnight wrap)
STOPS = STOPS_RAW + ["02:00"]


def _as_list(v):
    return [v] if not isinstance(v, list) else v


def main() -> int:
    if not CORES_FILE.exists():
        print(f"ERROR: {CORES_FILE} missing — run filter_stage_a.py first")
        return 2
    cores = json.loads(CORES_FILE.read_text())
    if not cores:
        print("ERROR: no cores in stage_a_top_cores.json")
        return 3

    for i, core in enumerate(cores, 1):
        grid = {k: _as_list(v) for k, v in core.items()}
        # Session sweep
        grid["enable_session_filter"] = [True]
        grid["trade_window1_start"] = STARTS
        grid["trade_window1_stop"] = STOPS
        grid["enable_trade_window2"] = [False]
        grid["trade_window2_start"] = ["09:30"]
        grid["trade_window2_stop"] = ["11:30"]
        grid["enable_trade_window3"] = [False]
        grid["trade_window3_start"] = ["14:00"]
        grid["trade_window3_stop"] = ["15:55"]
        grid["eod_exit_time"] = ["16:55"]

        out = ROOT / f"stage_b_core_{i}.json"
        out.write_text(json.dumps(grid, indent=2))
        print(f"  wrote: {out.name} (144 session combos)")

    print(f"\n  {len(cores)} Stage B grids ready")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
