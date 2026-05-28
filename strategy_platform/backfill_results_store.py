"""
Backfill local reports/ artifacts into the shared MySQL results store.

Usage:
    python -m strategy_platform.backfill_results_store
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional, Tuple

import pandas as pd

import strategy_platform  # noqa: F401 - ensure strategies register
from strategy_platform.optimize.pipeline import REPORTS_DIR, strategy_reports_dir
from strategy_platform.registry import StrategyRegistry
from strategy_platform import results_store

RUN_RE = re.compile(r'^(IS|MC|OOS)_(.+)_(\d{8}_\d{4})\.csv$')
BT_RE = re.compile(r'^BT_(.+)_(\d{8}_\d{6})\.json$')


def _scan_all_reports() -> list[tuple[str, str]]:
    """Yield (name, full_path) for every file in REPORTS_DIR (root + 1-level subdirs).

    Strategy-tagged files live under reports/<strategy>/ after the 2026-05 reorg;
    we also scan the flat root for files migrated mid-flight or backwards-compat.
    """
    out = []
    if not os.path.isdir(REPORTS_DIR):
        return out
    for name in os.listdir(REPORTS_DIR):
        full = os.path.join(REPORTS_DIR, name)
        if os.path.isfile(full):
            out.append((name, full))
        elif os.path.isdir(full) and not name.startswith("_") and name != "configs":
            # strategy subfolder — scan its files only (not its sub-sub)
            try:
                for nm in os.listdir(full):
                    fp = os.path.join(full, nm)
                    if os.path.isfile(fp):
                        out.append((nm, fp))
            except Exception:
                pass
    return out


def _strategy_names() -> list[str]:
    return sorted(StrategyRegistry.list_strategies(), key=len, reverse=True)


def _split_strategy_and_sym_safe(middle: str) -> Optional[Tuple[str, str]]:
    for strategy_name in _strategy_names():
        prefix = f"{strategy_name}_"
        if middle.startswith(prefix):
            return strategy_name, middle[len(prefix):]
        if middle == strategy_name:
            return strategy_name, ""
    return None


def _symbol_from_sym_safe(sym_safe: str) -> str:
    return sym_safe.replace("_", "=")


def _run_label_path(strategy_name: str, sym_safe: str, run_ts: str) -> str:
    # Check new strategy subfolder first, fall back to flat root.
    name = f"RUN_{strategy_name}_{sym_safe}_{run_ts}.label"
    for d in (strategy_reports_dir(strategy_name), REPORTS_DIR):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(strategy_reports_dir(strategy_name), name)


def _bt_label_path(strategy_name: str, sym_safe: str, bt_ts: str) -> str:
    name = f"BT_{strategy_name}_{sym_safe}_{bt_ts}.label"
    for d in (strategy_reports_dir(strategy_name), REPORTS_DIR):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return os.path.join(strategy_reports_dir(strategy_name), name)


def _read_label(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return ""


def _load_stage_frames(strategy_name: str, sym_safe: str, run_ts: str) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for stage in ["IS", "MC", "OOS"]:
        name = f"{stage}_{strategy_name}_{sym_safe}_{run_ts}.csv"
        for d in (strategy_reports_dir(strategy_name), REPORTS_DIR):
            path = os.path.join(d, name)
            if os.path.exists(path):
                out[stage] = pd.read_csv(path)
                break
    return out


def _run_meta_from_frames(stage_frames: Dict[str, pd.DataFrame]) -> dict:
    for df in stage_frames.values():
        if df is not None and not df.empty:
            row = df.iloc[0].to_dict()
            return {k: v for k, v in row.items() if str(k).startswith("_")}
    return {}


def backfill_optimizer_runs() -> int:
    count = 0
    seen = set()
    for name, _full in _scan_all_reports():
        match = RUN_RE.match(name)
        if not match or match.group(1) != "IS":
            continue
        middle = match.group(2)
        run_ts = match.group(3)
        split = _split_strategy_and_sym_safe(middle)
        if split is None:
            continue
        strategy_name, sym_safe = split
        key = (strategy_name, sym_safe, run_ts)
        if key in seen:
            continue
        seen.add(key)

        stage_frames = _load_stage_frames(strategy_name, sym_safe, run_ts)
        if not stage_frames:
            continue

        results_store.save_optimizer_run(
            strategy_name=strategy_name,
            symbol=_symbol_from_sym_safe(sym_safe),
            run_ts=run_ts,
            run_meta=_run_meta_from_frames(stage_frames),
            settings={"backfilled_from_reports": True},
            stage_frames=stage_frames,
            label=_read_label(_run_label_path(strategy_name, sym_safe, run_ts)) or None,
        )
        count += 1
    return count


def backfill_backtests() -> int:
    count = 0
    for name, path in _scan_all_reports():
        match = BT_RE.match(name)
        if not match:
            continue
        middle = match.group(1)
        bt_ts = match.group(2)
        split = _split_strategy_and_sym_safe(middle)
        if split is None:
            continue
        strategy_name, sym_safe = split
        try:
            with open(path) as f:
                payload = json.load(f)
        except Exception:
            continue

        results_store.save_backtest(
            strategy_name=strategy_name,
            symbol=_symbol_from_sym_safe(sym_safe),
            bt_ts=bt_ts,
            payload=payload,
            label=_read_label(_bt_label_path(strategy_name, sym_safe, bt_ts)) or None,
        )
        count += 1
    return count


def main() -> None:
    results_store.ensure_results_store()
    run_count = backfill_optimizer_runs()
    bt_count = backfill_backtests()
    print(f"Backfilled optimizer runs: {run_count}")
    print(f"Backfilled saved backtests: {bt_count}")


if __name__ == "__main__":
    main()
