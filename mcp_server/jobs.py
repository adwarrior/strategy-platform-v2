#!/usr/bin/env python3
"""Job-registry helpers for fire-and-poll optimization.

Pure helpers: no subprocess spawning, no FastMCP imports. The MCP tools in
server.py and the runner in opt_runner.py build on these.
"""
from __future__ import annotations

import json
import os
import secrets
from datetime import datetime
from typing import Optional

# Repo root on sys.path so strategy_platform imports resolve when this module
# is imported from server.py (which already inserts it) or directly in tests.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
import sys
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from strategy_platform.registry import StrategyRegistry
from strategy_platform.optimize.pipeline import _deduplicated_combinations
from strategy_platform import results_store

JOBS_DIR = os.path.join(_HERE, "jobs")


def sym_safe(symbol: str) -> str:
    return symbol.replace("=", "_")


def new_job_id() -> str:
    return secrets.token_hex(4)


def make_run_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M")  # local wall-clock intentional: matches run_pipeline's datetime.now().strftime('%Y%m%d_%H%M') for run_ts parity


def _ensure_dir() -> None:
    os.makedirs(JOBS_DIR, exist_ok=True)


def job_path(job_id: str) -> str:
    return os.path.join(JOBS_DIR, f"job_{job_id}.json")


def log_path(job_id: str) -> str:
    return os.path.join(JOBS_DIR, f"opt_{job_id}.log")


def write_job(record: dict) -> None:
    _ensure_dir()
    with open(job_path(record["job_id"]), "w") as f:
        json.dump(record, f, indent=2)


def read_job(job_id: str) -> Optional[dict]:
    try:
        with open(job_path(job_id)) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def all_jobs() -> list[dict]:
    if not os.path.isdir(JOBS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(JOBS_DIR)):
        if name.startswith("job_") and name.endswith(".json"):
            try:
                with open(os.path.join(JOBS_DIR, name)) as f:
                    out.append(json.load(f))
            except (json.JSONDecodeError, OSError):
                continue
    return out


def pid_alive(pid: int) -> bool:
    if not pid or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists but not ours to signal
    return True


def count_combos(strategy: str, grid: Optional[dict]) -> int:
    inst = StrategyRegistry.get(strategy)()
    effective = grid if grid is not None else inst.param_grid
    deps = inst.param_dependencies
    return sum(1 for _ in _deduplicated_combinations(effective, deps))


def run_in_db(strategy: str, sym_safe_val: str, run_ts: str) -> bool:
    try:
        return run_ts in results_store.list_optimizer_run_timestamps(strategy, sym_safe_val)
    except Exception:
        return False


def tail(path: str, n: int = 20) -> str:
    try:
        with open(path, errors="replace") as f:
            lines = f.readlines()
        return "".join(lines[-n:])
    except FileNotFoundError:
        return ""


def compute_status(record: dict) -> str:
    if run_in_db(record["strategy"], record["sym_safe"], record["run_ts"]):
        return "done"
    if pid_alive(int(record.get("pid", 0))):
        return "running"
    return "failed"
