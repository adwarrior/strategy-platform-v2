"""
Post-filter Stage A pipeline results.

Reads the latest IS_supertrendfractal_MNQ_*.csv and OOS_supertrendfractal_MNQ_*.csv
in reports/, applies max_drawdown <= $2000 to both halves, ranks survivors by
OOS Sortino (then OOS net_pnl as tiebreak), and emits:

  optimize_runs/stage_a_top_cores.json   - top-N "core" param dicts
  optimize_runs/stage_a_filter_log.txt   - human-readable summary

A "core" config is the 5-tuple (atr_multiplier, atr_period, fractal_length,
exit_mode + tpsl_mode + tp/sl/rr params, bars_between_trades). Stage B then
sweeps session windows on these locked cores.

Usage:
    python optimize_runs/filter_stage_a.py [TOP_N]
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
OUT_DIR = ROOT / "optimize_runs"

DD_LIMIT = 2000.0  # max_drawdown cap in $
DEFAULT_TOP_N = 5


def _latest(pattern: str) -> Path | None:
    files = sorted(glob.glob(str(REPORTS / pattern)))
    return Path(files[-1]) if files else None


def main() -> int:
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TOP_N

    oos_csv = _latest("OOS_supertrendfractal_MNQ_*.csv")
    is_csv = _latest("IS_supertrendfractal_MNQ_*.csv")
    if not oos_csv:
        print(f"ERROR: no OOS_supertrendfractal_MNQ_*.csv in {REPORTS}", file=sys.stderr)
        return 2
    if not is_csv:
        print(f"ERROR: no IS_supertrendfractal_MNQ_*.csv in {REPORTS}", file=sys.stderr)
        return 2

    print(f"  IS  src: {is_csv.name}")
    print(f"  OOS src: {oos_csv.name}")

    oos = pd.read_csv(oos_csv)
    is_df = pd.read_csv(is_csv)

    print(f"  IS rows: {len(is_df)}  OOS rows: {len(oos)}")

    # ---- DD filter on OOS ----
    dd_col = "max_drawdown" if "max_drawdown" in oos.columns else None
    if not dd_col:
        print(f"  WARN: no max_drawdown column in OOS — skipping DD filter")
        survivors = oos.copy()
    else:
        oos[dd_col] = oos[dd_col].abs()
        before = len(oos)
        survivors = oos[oos[dd_col] <= DD_LIMIT].copy()
        print(f"  OOS DD<=${DD_LIMIT:.0f}: {len(survivors)}/{before} survive")

    # ---- Also filter on IS DD if available ----
    if dd_col and "max_drawdown" in is_df.columns:
        is_df[dd_col] = is_df[dd_col].abs()
        # Join key — assume same param columns present in both
        param_cols = [
            c for c in is_df.columns
            if c not in {"sharpe", "sortino", "net_pnl", "profit_factor",
                         "max_drawdown", "win_rate", "trades", "total_trades",
                         "trades_df", "tick_bar_size"}
            and not c.startswith("mc_")
        ]
        is_keep = is_df[is_df[dd_col] <= DD_LIMIT][param_cols].drop_duplicates()
        before = len(survivors)
        survivors = survivors.merge(is_keep, on=param_cols, how="inner")
        print(f"  IS  DD<=${DD_LIMIT:.0f}: {len(survivors)}/{before} survive")

    if len(survivors) == 0:
        print("  ERROR: no combos passed DD filter on both IS and OOS.", file=sys.stderr)
        # Still write an empty seed so downstream can detect & abort
        (OUT_DIR / "stage_a_top_cores.json").write_text("[]")
        return 3

    # ---- Rank by OOS Sortino desc, then OOS net_pnl desc ----
    rank_cols = []
    if "sortino" in survivors.columns:
        rank_cols.append(("sortino", False))
    if "net_pnl" in survivors.columns:
        rank_cols.append(("net_pnl", False))
    if not rank_cols:
        rank_cols = [("sharpe", False)]
    sort_by = [c for c, _ in rank_cols]
    ascending = [a for _, a in rank_cols]
    survivors = survivors.sort_values(by=sort_by, ascending=ascending)

    # ---- Pick top N, extracting only the "core" param keys ----
    CORE_KEYS = [
        "atr_multiplier", "atr_period", "fractal_length",
        "direction", "invert_signals",
        "exit_mode", "tpsl_mode",
        "tp_ticks", "sl_ticks",
        "tp_atr_mult", "sl_atr_mult",
        "rr_ratio",
        "use_risk_sizing", "qty", "max_risk",
        "bars_between_trades",
    ]
    have_keys = [k for k in CORE_KEYS if k in survivors.columns]

    top = survivors.head(top_n)
    cores: list[dict] = []
    log_lines: list[str] = [
        f"Stage A filter — DD<=${DD_LIMIT:.0f}, ranked by OOS Sortino/net_pnl",
        f"  IS src : {is_csv.name}",
        f"  OOS src: {oos_csv.name}",
        f"  survivors after DD filter: {len(survivors)}",
        "",
        "Top cores:",
    ]
    for i, (_, row) in enumerate(top.iterrows(), 1):
        core = {k: row[k] for k in have_keys}
        # Clean up types: numpy ints/floats -> python natives
        for k, v in core.items():
            if hasattr(v, "item"):
                core[k] = v.item()
        cores.append(core)
        log_lines.append(
            f"  #{i}: sortino={row.get('sortino', float('nan')):.3f}  "
            f"net_pnl=${row.get('net_pnl', 0):.0f}  "
            f"dd=${row.get('max_drawdown', 0):.0f}  "
            f"trades={int(row.get('total_trades', row.get('trades', 0)))}  "
            f"params={core}"
        )

    (OUT_DIR / "stage_a_top_cores.json").write_text(json.dumps(cores, indent=2))
    (OUT_DIR / "stage_a_filter_log.txt").write_text("\n".join(log_lines))
    print("\n".join(log_lines))
    print(f"\n  wrote: optimize_runs/stage_a_top_cores.json  ({len(cores)} cores)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
