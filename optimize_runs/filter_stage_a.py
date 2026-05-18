"""
Post-filter Stage A pipeline results.

Reads the latest IS_supertrendfractal_MNQ_*.csv and OOS_supertrendfractal_MNQ_*.csv
in reports/, applies a max-drawdown cap on both halves, dedups by core-param
fingerprint, ranks survivors by oos_sortino (then oos_net_pnl), and emits:

  optimize_runs/stage_a_top_cores.json   - top-N "core" param dicts
  optimize_runs/stage_a_filter_log.txt   - human-readable summary

A "core" config is the indicator + exit-tree + cooldown 5-tuple. Stage B then
sweeps session windows on these locked cores.

Note: Stage A DD cap is intentionally looser than the final prop cap of $2,500
because session filtering in Stage B typically cuts DD 20-30%, AR refines
further. Final-prop cap is enforced post-WFO.

Usage:
    python optimize_runs/filter_stage_a.py [TOP_N] [DD_CAP_DOLLARS]
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

DEFAULT_DD_CAP = 3000.0
DEFAULT_TOP_N = 5

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


def _latest(pattern: str) -> Path | None:
    files = sorted(glob.glob(str(REPORTS / pattern)))
    return Path(files[-1]) if files else None


def _canonical_core(row: pd.Series) -> tuple:
    """Build a fingerprint that ignores irrelevant params per exit_mode/tpsl_mode."""
    exit_mode = row.get("exit_mode")
    tpsl_mode = row.get("tpsl_mode")
    # Always-relevant
    base = (
        row.get("atr_multiplier"), row.get("atr_period"), row.get("fractal_length"),
        row.get("direction"), row.get("invert_signals"),
        exit_mode,
        row.get("bars_between_trades"),
    )
    # Exit-tree relevant subset
    if exit_mode == "FixedTPSL":
        if tpsl_mode == "Ticks":
            tail = ("Ticks", row.get("tp_ticks"), row.get("sl_ticks"))
        elif tpsl_mode == "ATRMultiple":
            tail = ("ATR", row.get("tp_atr_mult"), row.get("sl_atr_mult"))
        elif tpsl_mode == "RiskReward":
            tail = ("RR", row.get("rr_ratio"))
        else:
            tail = (tpsl_mode,)
    else:
        tail = ("Trail",)
    return base + tail


def main() -> int:
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TOP_N
    dd_cap = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DD_CAP

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
    print(f"  DD cap: ${dd_cap:.0f}   top_n: {top_n}")

    oos = pd.read_csv(oos_csv)
    is_df = pd.read_csv(is_csv)
    print(f"  IS rows: {len(is_df)}  OOS rows: {len(oos)}")

    # ---- DD filter (OOS uses oos_max_drawdown; IS uses bare max_drawdown) ----
    if "oos_max_drawdown" not in oos.columns:
        print("  ERROR: oos_max_drawdown column missing in OOS CSV", file=sys.stderr)
        return 3
    oos["_dd_abs"] = oos["oos_max_drawdown"].abs()
    before = len(oos)
    oos_keep = oos[oos["_dd_abs"] <= dd_cap].copy()
    print(f"  OOS |DD|<=${dd_cap:.0f}: {len(oos_keep)}/{before}")

    # Cross-check on IS DD too
    if "max_drawdown" in is_df.columns:
        is_df["_dd_abs"] = is_df["max_drawdown"].abs()
        merge_keys = [k for k in CORE_KEYS if k in is_df.columns and k in oos_keep.columns]
        is_pass = is_df[is_df["_dd_abs"] <= dd_cap][merge_keys].drop_duplicates()
        before = len(oos_keep)
        oos_keep = oos_keep.merge(is_pass, on=merge_keys, how="inner")
        print(f"  IS  |DD|<=${dd_cap:.0f}: {len(oos_keep)}/{before} also pass IS DD")

    if len(oos_keep) == 0:
        print("  ERROR: no combos passed DD filter on both IS and OOS", file=sys.stderr)
        (OUT_DIR / "stage_a_top_cores.json").write_text("[]")
        return 4

    # ---- Rank by OOS Sortino desc, then OOS net_pnl desc ----
    rank_cols = []
    for c in ("oos_sortino", "oos_sharpe", "oos_net_pnl"):
        if c in oos_keep.columns:
            rank_cols.append(c)
    oos_keep = oos_keep.sort_values(by=rank_cols, ascending=[False] * len(rank_cols))

    # ---- Dedup by canonical core fingerprint; pick top_n unique cores ----
    seen: set = set()
    cores: list[dict] = []
    log_lines: list[str] = [
        f"Stage A filter — |DD|<=${dd_cap:.0f}, dedup by core, top {top_n} by OOS Sortino",
        f"  IS  src : {is_csv.name}",
        f"  OOS src: {oos_csv.name}",
        f"  unique DD-passing OOS rows: {len(oos_keep)}",
        "",
    ]
    have_keys = [k for k in CORE_KEYS if k in oos_keep.columns]
    for _, row in oos_keep.iterrows():
        fp = _canonical_core(row)
        if fp in seen:
            continue
        seen.add(fp)
        params = {k: row[k] for k in have_keys}
        for k, v in params.items():
            if hasattr(v, "item"):
                params[k] = v.item()
        # Convert numpy bool to native bool, etc.
        for k, v in params.items():
            if isinstance(v, (bool,)):
                params[k] = bool(v)
        cores.append(params)
        log_lines.append(
            f"  #{len(cores)}: sortino={row.get('oos_sortino', 0):.3f}  "
            f"sharpe={row.get('oos_sharpe', 0):.3f}  "
            f"net_pnl=${row.get('oos_net_pnl', 0):,.0f}  "
            f"dd=${row.get('oos_max_drawdown', 0):,.0f}  "
            f"trades={int(row.get('oos_total_trades', 0))}  "
            f"wr={row.get('oos_win_rate', 0):.1%}  "
            f"exit={row.get('exit_mode')}/{row.get('tpsl_mode')}  "
            f"atr={row.get('atr_multiplier')}x/{int(row.get('atr_period', 0))} frac={int(row.get('fractal_length', 0))} "
            f"cd={int(row.get('bars_between_trades', 0))}"
        )
        if len(cores) >= top_n:
            break

    (OUT_DIR / "stage_a_top_cores.json").write_text(json.dumps(cores, indent=2))
    (OUT_DIR / "stage_a_filter_log.txt").write_text("\n".join(log_lines))
    print("\n".join(log_lines))
    print(f"\n  wrote: optimize_runs/stage_a_top_cores.json  ({len(cores)} unique cores)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
