# Fire-and-Poll Optimization for the Local MCP Server — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `start_optimization` / `check_optimization` / `list_jobs` MCP tools so Claude can launch a full optimizer sweep that runs detached, survives the session ending, and lands results in the existing `sp_optimizer_runs` DB.

**Architecture:** A thin CLI wrapper (`opt_runner.py`) calls the platform's existing `pipeline.run_pipeline(...)` as a **detached subprocess**; the pipeline persists its own results. A file-based job registry (`jobs/job_<id>.json` + `opt_<id>.log`) tracks each launch. Polling detects "done" by checking whether the run's `run_ts` has appeared in `sp_optimizer_runs` — the DB is the single source of truth for completion.

**Tech Stack:** Python 3.10, `mcp` SDK (FastMCP), the existing `strategy_platform` package (`registry`, `optimize.pipeline`, `results_store`, `data.loader`), `subprocess` (detached via `start_new_session=True`).

## Global Constraints

- **Local only.** stdio MCP server; no network egress; no auth.
- **No logic duplication.** Reuse `pipeline.run_pipeline(...)`; the runner only marshals args. Reuse `pipeline._deduplicated_combinations(grid, param_dependencies)` for the combo count.
- **Completion is DB-driven.** "Done" ⟺ `run_ts` present in `sp_optimizer_runs` (via `results_store.list_optimizer_run_timestamps(strategy, sym_safe)`). Job-file `status` is advisory only.
- **`run_ts` format is `YYYYmmdd_HHMM`** (matches `run_pipeline`, which calls `datetime.now().strftime('%Y%m%d_%H%M')`).
- **`sym_safe` convention:** `symbol.replace('=', '_')`.
- **Mega-sweep guard:** `MAX_COMBOS = 10000`. Over the cap, `start_optimization` refuses unless `confirm_large=True`.
- **Context discipline:** any tabular/log payload returned to Claude is capped (≤50 result rows, ≤20 log lines).
- **Strategy metadata are `@property`** (not methods): `inst.params`, `inst.param_grid`, `inst.param_dependencies`, `inst.description`, `inst.display_names` — access as attributes.
- **Run from the repo root** so `import strategy_platform` resolves; `server.py` already inserts the repo root on `sys.path` and loads `.env`.
- **Files live in** `mcp_server/` inside `strategy-platform-v2` (per the existing local-MCP layout).

---

## File Structure

- `mcp_server/opt_runner.py` — **new**. CLI wrapper around `run_pipeline`. No business logic.
- `mcp_server/jobs.py` — **new**. Job-registry helpers (paths, write/read/list job records, pid-alive check, combo count, DB-completion check, log tail). Pure functions, independently testable, no MCP/FastMCP imports.
- `mcp_server/server.py` — **modify**. Add 3 tools (`start_optimization`, `check_optimization`, `list_jobs`) that are thin adapters over `jobs.py` + subprocess spawn.
- `mcp_server/jobs/` — **new dir** (runtime). Holds `job_<id>.json` and `opt_<id>.log`. Git-ignored.
- `mcp_server/test_optimize.py` — **new**. End-to-end test: tiny grid → poll → assert DB run + OOS rows. Plus unit tests for `jobs.py` helpers.

Rationale for `jobs.py`: the registry/status logic is the only non-trivial code here and benefits from being unit-testable without spawning subprocesses or importing FastMCP. `server.py` tools stay ~15-line adapters; `opt_runner.py` stays a dumb marshaller.

---

### Task 1: Job registry helpers (`jobs.py`)

Pure helper module — no subprocess spawning, no MCP imports. This is the testable core.

**Files:**
- Create: `mcp_server/jobs.py`
- Test: `mcp_server/test_optimize.py` (unit tests for this task)

**Interfaces:**
- Consumes: `strategy_platform.optimize.pipeline._deduplicated_combinations`, `strategy_platform.registry.StrategyRegistry`, `strategy_platform.results_store.list_optimizer_run_timestamps`.
- Produces (later tasks rely on these exact signatures):
  - `JOBS_DIR: str` — absolute path to `mcp_server/jobs`.
  - `new_job_id() -> str` — short hex id.
  - `make_run_ts() -> str` — `YYYYmmdd_HHMM`.
  - `job_path(job_id: str) -> str`, `log_path(job_id: str) -> str`.
  - `write_job(record: dict) -> None` — writes `job_<id>.json`.
  - `read_job(job_id: str) -> Optional[dict]`.
  - `all_jobs() -> list[dict]`.
  - `pid_alive(pid: int) -> bool`.
  - `count_combos(strategy: str, grid: Optional[dict]) -> int` — dedup combo count for `grid` (or the strategy's full grid when `grid` is None).
  - `run_in_db(strategy: str, sym_safe: str, run_ts: str) -> bool`.
  - `tail(path: str, n: int = 20) -> str`.
  - `compute_status(record: dict) -> str` — returns `"done" | "running" | "failed"` using the decision order: DB run present → done; else pid alive → running; else failed.
  - `sym_safe(symbol: str) -> str`.

- [ ] **Step 1: Write failing unit tests**

Create `mcp_server/test_optimize.py` with the unit-test portion:

```python
import os
import time
import json
import pytest

import jobs  # mcp_server/ is the CWD when running these tests


def test_sym_safe():
    assert jobs.sym_safe("NQ=F") == "NQ_F"
    assert jobs.sym_safe("MNQ") == "MNQ"


def test_make_run_ts_format():
    ts = jobs.make_run_ts()
    assert len(ts) == 13 and ts[8] == "_"        # YYYYmmdd_HHMM
    assert ts[:8].isdigit() and ts[9:].isdigit()


def test_new_job_id_unique():
    assert jobs.new_job_id() != jobs.new_job_id()


def test_write_read_all_jobs(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs, "JOBS_DIR", str(tmp_path))
    rec = {"job_id": "abc123", "pid": 1, "run_ts": "20260101_0000",
           "strategy": "s", "symbol": "MNQ", "sym_safe": "MNQ"}
    jobs.write_job(rec)
    assert jobs.read_job("abc123") == rec
    assert rec in jobs.all_jobs()
    assert jobs.read_job("nope") is None


def test_pid_alive_self_and_dead():
    assert jobs.pid_alive(os.getpid()) is True
    assert jobs.pid_alive(2_000_000_000) is False    # implausibly high pid


def test_tail(tmp_path):
    p = tmp_path / "x.log"
    p.write_text("\n".join(str(i) for i in range(100)))
    out = jobs.tail(str(p), n=5)
    assert out.strip().splitlines() == ["95", "96", "97", "98", "99"]


def test_tail_missing_file_returns_empty():
    assert jobs.tail("/no/such/file.log") == ""


def test_count_combos_small_grid(monkeypatch):
    # 2 x 3 = 6 combos, no dependencies
    class FakeStrat:
        param_grid = {"a": [1, 2], "b": [1, 2, 3]}
        param_dependencies = {}
    monkeypatch.setattr(jobs.StrategyRegistry, "get", staticmethod(lambda n: (lambda: FakeStrat())))
    assert jobs.count_combos("whatever", {"a": [1, 2], "b": [1, 2, 3]}) == 6


def test_compute_status_running(monkeypatch):
    monkeypatch.setattr(jobs, "run_in_db", lambda *a, **k: False)
    monkeypatch.setattr(jobs, "pid_alive", lambda pid: True)
    rec = {"strategy": "s", "sym_safe": "MNQ", "run_ts": "20260101_0000", "pid": 123}
    assert jobs.compute_status(rec) == "running"


def test_compute_status_done(monkeypatch):
    monkeypatch.setattr(jobs, "run_in_db", lambda *a, **k: True)
    rec = {"strategy": "s", "sym_safe": "MNQ", "run_ts": "20260101_0000", "pid": 123}
    assert jobs.compute_status(rec) == "done"


def test_compute_status_failed(monkeypatch):
    monkeypatch.setattr(jobs, "run_in_db", lambda *a, **k: False)
    monkeypatch.setattr(jobs, "pid_alive", lambda pid: False)
    rec = {"strategy": "s", "sym_safe": "MNQ", "run_ts": "20260101_0000", "pid": 123}
    assert jobs.compute_status(rec) == "failed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ad/strategy-platform-v2/mcp_server && python3 -m pytest test_optimize.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'jobs'`.

- [ ] **Step 3: Write `jobs.py`**

```python
#!/usr/bin/env python3
"""Job-registry helpers for fire-and-poll optimization.

Pure helpers: no subprocess spawning, no FastMCP imports. The MCP tools in
server.py and the runner in opt_runner.py build on these.
"""
from __future__ import annotations

import json
import os
import secrets
import signal
from datetime import datetime, timezone
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
    return datetime.now().strftime("%Y%m%d_%H%M")


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


def all_jobs() -> list:
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
    effective = grid if grid else inst.param_grid
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ad/strategy-platform-v2/mcp_server && python3 -m pytest test_optimize.py -v`
Expected: PASS (10 tests). If `pytest` is missing: `python3 -m pip install --user pytest`.

- [ ] **Step 5: Commit**

```bash
cd /home/ad/strategy-platform-v2
git add mcp_server/jobs.py mcp_server/test_optimize.py
git commit -m "feat(mcp): job-registry helpers for fire-and-poll optimization"
```

---

### Task 2: Detached runner (`opt_runner.py`)

A dumb CLI that calls `run_pipeline`. Verified by running it directly on a tiny grid and asserting it writes a run to the DB.

**Files:**
- Create: `mcp_server/opt_runner.py`
- Test: `mcp_server/test_optimize.py` (add the runner test below)

**Interfaces:**
- Consumes: `strategy_platform.optimize.pipeline.run_pipeline(strategy_name, symbol, timeframe_mins, data_start, data_end, train_pct, rank_by, min_trades, param_grid_override, run_settings)`; `jobs.run_in_db`.
- Produces: an executable module runnable as `python3 mcp_server/opt_runner.py --strategy ... --symbol ... --run-ts ... [--grid-file PATH] ...`. On success the run identified by `--run-ts` exists in `sp_optimizer_runs`.

- [ ] **Step 1: Write the failing test**

Add to `mcp_server/test_optimize.py`:

```python
import subprocess
import sys


# A strategy + symbol + small date range known to produce a run quickly.
# WaeJurikPro on MNQ 5m over one week ran in the local smoke test.
SMALL = dict(strategy="WaeJurikPro", symbol="MNQ",
             data_start="2026-05-01", data_end="2026-05-08")


def _tiny_grid_for(strategy):
    """Pick the first value of each grid param -> a 1-combo grid (fast)."""
    inst = jobs.StrategyRegistry.get(strategy)()
    return {k: [v[0]] for k, v in inst.param_grid.items()}


@pytest.mark.slow
def test_opt_runner_writes_db_run():
    run_ts = jobs.make_run_ts() + "_t1"     # unique suffix avoids collisions
    grid = _tiny_grid_for(SMALL["strategy"])
    grid_file = os.path.join(jobs.JOBS_DIR, f"grid_{run_ts}.json")
    jobs._ensure_dir()
    with open(grid_file, "w") as f:
        json.dump(grid, f)

    runner = os.path.join(os.path.dirname(os.path.abspath(jobs.__file__)), "opt_runner.py")
    proc = subprocess.run(
        [sys.executable, runner,
         "--strategy", SMALL["strategy"], "--symbol", SMALL["symbol"],
         "--run-ts", run_ts, "--timeframe-mins", "5",
         "--data-start", SMALL["data_start"], "--data-end", SMALL["data_end"],
         "--grid-file", grid_file],
        capture_output=True, text=True, timeout=600,
        cwd=os.path.dirname(runner),
    )
    assert proc.returncode == 0, f"runner failed:\nSTDOUT{proc.stdout}\nSTDERR{proc.stderr}"
    assert jobs.run_in_db(SMALL["strategy"], "MNQ", run_ts), \
        "run_ts not found in sp_optimizer_runs after runner completed"
```

Note: `run_pipeline` accepts an arbitrary `run_ts`? It generates its own `ts` internally. See Step 3 — the runner passes `run_ts` through `run_settings` is NOT enough; instead the runner sets the pipeline's timestamp. Confirm the mechanism in Step 3 before implementing.

- [ ] **Step 2: Confirm how `run_pipeline` sets its `run_ts`, then run the test to verify it fails**

First, read how the pipeline derives and saves `ts`:

Run: `grep -nE "ts = datetime|run_ts|save_optimizer_run\(|def run_pipeline" /home/ad/strategy-platform-v2/strategy_platform/optimize/pipeline.py`

- `run_pipeline` builds `ts = datetime.now().strftime('%Y%m%d_%H%M')` internally (line ~224) and passes it as `run_ts` to `save_optimizer_run`. It does **not** accept a `run_ts` argument.

**Implication (resolve before coding):** the parent (`start_optimization`) cannot know the exact `run_ts` in advance because the child generates it. Two safe options — the runner is where we fix this:
  - **(A) Runner prints the run_ts.** Let `run_pipeline` generate `ts`, have the runner capture the saved run_ts and print a final line `RUN_TS=<ts>`; the parent reads it from the log on first poll. But the parent needs run_ts *immediately* for the job record.
  - **(B) Runner forces the run_ts.** Pass `run_ts` into `run_pipeline` via a new optional kwarg, OR set it deterministically. Cleanest: add `run_ts: Optional[str] = None` to `run_pipeline` (default = current behavior) and use it when provided. This is a **small, backward-compatible change to `pipeline.py`** and makes the parent fully deterministic.

**Decision:** Use **(B)** — add an optional `run_ts` param to `run_pipeline`. It is backward-compatible (defaults to the generated value) and removes all run-id ambiguity from poll logic. This is the one platform-file edit in this plan and is covered by Step 3.

Run: `cd /home/ad/strategy-platform-v2/mcp_server && python3 -m pytest test_optimize.py::test_opt_runner_writes_db_run -v`
Expected: FAIL — `opt_runner.py` does not exist.

- [ ] **Step 3: Make `run_pipeline` accept an optional `run_ts`, then write `opt_runner.py`**

First, the backward-compatible platform edit. In `strategy_platform/optimize/pipeline.py`:

Modify the signature (around line 205) to add the parameter:

```python
    run_settings:       Optional[Dict[str, Any]] = None,
    run_ts:             Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
```

And change the timestamp line (around line 224) from:

```python
    ts = datetime.now().strftime('%Y%m%d_%H%M')
```

to:

```python
    ts = run_ts or datetime.now().strftime('%Y%m%d_%H%M')
```

(That `ts` is already threaded through to `save_optimizer_run(...)`, so no other pipeline change is needed.)

Now create `mcp_server/opt_runner.py`:

```python
#!/usr/bin/env python3
"""Detached CLI wrapper around pipeline.run_pipeline.

Spawned by the start_optimization MCP tool. Contains no business logic: it
marshals args and calls run_pipeline, which persists results to the DB.
stdout/stderr are redirected to the job log by the parent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
except Exception:
    pass

from strategy_platform.optimize.pipeline import run_pipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--run-ts", required=True)
    ap.add_argument("--timeframe-mins", type=int, default=5)
    ap.add_argument("--data-start", default=None)
    ap.add_argument("--data-end", default=None)
    ap.add_argument("--train-pct", type=float, default=0.70)
    ap.add_argument("--rank-by", default="sharpe")
    ap.add_argument("--min-trades", type=int, default=None)
    ap.add_argument("--grid-file", default=None,
                    help="Path to JSON file with a param_grid override dict.")
    args = ap.parse_args()

    grid = None
    if args.grid_file:
        with open(args.grid_file) as f:
            grid = json.load(f)

    print(f"[opt_runner] starting {args.strategy} {args.symbol} run_ts={args.run_ts}", flush=True)
    kwargs = dict(
        strategy_name=args.strategy,
        symbol=args.symbol,
        timeframe_mins=args.timeframe_mins,
        data_start=args.data_start,
        data_end=args.data_end,
        train_pct=args.train_pct,
        rank_by=args.rank_by,
        param_grid_override=grid,
        run_ts=args.run_ts,
    )
    if args.min_trades is not None:
        kwargs["min_trades"] = args.min_trades

    run_pipeline(**kwargs)
    print(f"[opt_runner] done run_ts={args.run_ts}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the runner test to verify it passes**

Run: `cd /home/ad/strategy-platform-v2/mcp_server && python3 -m pytest test_optimize.py::test_opt_runner_writes_db_run -v -m slow`
Expected: PASS. The run completes (1-combo grid, one week of MNQ 5m) and `run_ts` appears in `sp_optimizer_runs`.
(Requires DB reachability at `DB_HOST` from `.env`. If the DB is down, this test errors — that's environmental, not a code failure.)

- [ ] **Step 5: Verify the pipeline change is backward-compatible**

Run: `cd /home/ad/strategy-platform-v2 && python3 -c "import inspect; from strategy_platform.optimize.pipeline import run_pipeline; sig=inspect.signature(run_pipeline); assert sig.parameters['run_ts'].default is None; print('run_ts param OK, default None')"`
Expected: `run_ts param OK, default None`

- [ ] **Step 6: Commit**

```bash
cd /home/ad/strategy-platform-v2
git add mcp_server/opt_runner.py strategy_platform/optimize/pipeline.py mcp_server/test_optimize.py
git commit -m "feat(mcp): detached opt_runner + optional run_ts in run_pipeline"
```

---

### Task 3: MCP tools (`start_optimization`, `check_optimization`, `list_jobs`)

Wire the helpers + runner into FastMCP tools in `server.py`. Verified by an end-to-end test that launches a tiny sweep via the tool and polls to completion.

**Files:**
- Modify: `mcp_server/server.py` (add imports + 3 `@mcp.tool()` functions, before the `if __name__ == "__main__"` block)
- Modify: `mcp_server/test_optimize.py` (add the e2e test below)
- Modify: `.gitignore` (ignore the runtime jobs dir)

**Interfaces:**
- Consumes: everything from `jobs.py` (Task 1) and `opt_runner.py` (Task 2).
- Produces (Claude-facing tools):
  - `start_optimization(strategy, symbol, timeframe="5m", data_start=None, data_end=None, param_grid=None, train_pct=0.70, rank_by="sharpe", min_trades=None, confirm_large=False) -> dict`
  - `check_optimization(job_id) -> dict`
  - `list_jobs() -> dict`

- [ ] **Step 1: Write the failing e2e test**

Add to `mcp_server/test_optimize.py`:

```python
@pytest.mark.slow
def test_start_and_check_optimization_e2e():
    import server
    grid = _tiny_grid_for(SMALL["strategy"])
    started = server.start_optimization(
        strategy=SMALL["strategy"], symbol=SMALL["symbol"],
        timeframe="5m", data_start=SMALL["data_start"], data_end=SMALL["data_end"],
        param_grid=grid,
    )
    assert started["status"] == "running"
    assert "job_id" in started and "run_ts" in started
    job_id = started["job_id"]

    deadline = time.time() + 600
    status = None
    while time.time() < deadline:
        res = server.check_optimization(job_id)
        status = res["status"]
        if status in ("done", "failed"):
            break
        time.sleep(5)
    assert status == "done", f"ended as {status}: {res}"
    assert res.get("top_results") is not None

    listed = server.list_jobs()
    assert any(j["job_id"] == job_id for j in listed["jobs"])


def test_start_optimization_refuses_large(monkeypatch):
    import server
    monkeypatch.setattr(server.jobs, "count_combos", lambda *a, **k: 999999)
    res = server.start_optimization(strategy=SMALL["strategy"], symbol="MNQ")
    assert res.get("refused") is True
    assert res["combos"] == 999999
```

- [ ] **Step 2: Run the e2e + refusal tests to verify they fail**

Run: `cd /home/ad/strategy-platform-v2/mcp_server && python3 -m pytest test_optimize.py::test_start_optimization_refuses_large -v`
Expected: FAIL — `AttributeError: module 'server' has no attribute 'start_optimization'`.

- [ ] **Step 3: Add the tools to `server.py`**

Add this import near the other platform imports at the top of `mcp_server/server.py` (after `from strategy_platform import results_store`):

```python
import subprocess  # noqa: E402
import jobs  # noqa: E402  (mcp_server/jobs.py — registry helpers)
```

Also add the timeframe→minutes helper and the combo cap constant near the other module constants (after `_MAX_TRADE_ROWS = 50`):

```python
MAX_COMBOS = 10000
_MAX_OPT_ROWS = 50
_MAX_LOG_LINES = 20


def _tf_to_minutes(timeframe: str) -> int:
    """Map a timeframe string to integer minutes for run_pipeline.

    Accepts '1m'/'5m'/'15m'/'60m'/'240m' or bare '5'. Tick timeframes are not
    supported for optimization here (the optimizer pipeline is time-bar based).
    """
    tf = str(timeframe).strip().lower()
    if tf.endswith("min"):
        return int(tf[:-3])
    if tf.endswith("m"):
        return int(tf[:-1])
    return int(tf)
```

Add these three tools just before the `if __name__ == "__main__":` block:

```python
# =========================================================================
# Optimization (fire-and-poll)
# =========================================================================
@mcp.tool()
def start_optimization(strategy: str, symbol: str, timeframe: str = "5m",
                       data_start: Optional[str] = None,
                       data_end: Optional[str] = None,
                       param_grid: Optional[dict] = None,
                       train_pct: float = 0.70, rank_by: str = "sharpe",
                       min_trades: Optional[int] = None,
                       confirm_large: bool = False) -> dict:
    """Launch a full optimizer sweep in the background and return immediately.

    Runs detached (survives this session). Results land in the results_store
    DB; poll with check_optimization(job_id) or list_jobs(). param_grid is a
    dict of param->list of values (see get_strategy_params); omit it to use
    the strategy's full grid. timeframe is a minute bar ('5m','15m',...).

    Refuses if the grid exceeds MAX_COMBOS (10000) unless confirm_large=True.
    """
    # Fail fast on bad inputs (before spawning anything).
    try:
        StrategyRegistry.get(strategy)
    except Exception as e:
        return {"error": f"unknown strategy '{strategy}': {e}"}
    try:
        loader.get_meta(symbol)
    except Exception as e:
        return {"error": f"unknown/unsupported symbol '{symbol}': {e}"}

    try:
        combos = jobs.count_combos(strategy, param_grid)
    except Exception as e:
        return {"error": f"could not evaluate grid: {e}"}
    if combos > MAX_COMBOS and not confirm_large:
        return {"refused": True, "combos": combos, "max_combos": MAX_COMBOS,
                "message": (f"Grid has {combos} combinations (cap {MAX_COMBOS}). "
                            "Re-call with confirm_large=True to launch anyway.")}

    job_id = jobs.new_job_id()
    run_ts = jobs.make_run_ts()
    jobs._ensure_dir()

    grid_file = None
    if param_grid:
        grid_file = os.path.join(jobs.JOBS_DIR, f"grid_{job_id}.json")
        with open(grid_file, "w") as f:
            import json as _json
            _json.dump(param_grid, f)

    runner = os.path.join(os.path.dirname(os.path.abspath(__file__)), "opt_runner.py")
    cmd = [sys.executable, runner,
           "--strategy", strategy, "--symbol", symbol, "--run-ts", run_ts,
           "--timeframe-mins", str(_tf_to_minutes(timeframe)),
           "--train-pct", str(train_pct), "--rank-by", rank_by]
    if data_start:
        cmd += ["--data-start", data_start]
    if data_end:
        cmd += ["--data-end", data_end]
    if min_trades is not None:
        cmd += ["--min-trades", str(min_trades)]
    if grid_file:
        cmd += ["--grid-file", grid_file]

    lp = jobs.log_path(job_id)
    logf = open(lp, "w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            start_new_session=True, cwd=_REPO_ROOT)

    record = {
        "job_id": job_id, "pid": proc.pid, "run_ts": run_ts,
        "strategy": strategy, "symbol": symbol, "sym_safe": jobs.sym_safe(symbol),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "log_path": lp, "combos": combos, "status": "running",
    }
    jobs.write_job(record)
    return {"job_id": job_id, "run_ts": run_ts, "status": "running",
            "combos": combos, "log_path": lp}


@mcp.tool()
def check_optimization(job_id: str) -> dict:
    """Poll an optimization launched by start_optimization.

    Returns status 'running' (with elapsed + log tail), 'done' (with top OOS
    results), or 'failed' (with log tail).
    """
    rec = jobs.read_job(job_id)
    if rec is None:
        return {"error": f"no job '{job_id}'"}
    status = jobs.compute_status(rec)
    rec["status"] = status
    jobs.write_job(rec)

    out = {"job_id": job_id, "status": status, "strategy": rec["strategy"],
           "symbol": rec["symbol"], "run_ts": rec["run_ts"]}

    if status == "done":
        df = results_store.load_optimizer_stage(
            rec["strategy"], rec["sym_safe"], rec["run_ts"], "oos")
        if df is not None and not df.empty:
            out["top_results"] = (df.head(_MAX_OPT_ROWS)
                                  .astype(str).to_dict(orient="records"))
            out["oos_rows"] = int(len(df))
        else:
            out["top_results"] = []
            out["note"] = "run recorded but OOS stage empty"
    else:
        out["log_tail"] = jobs.tail(rec["log_path"], _MAX_LOG_LINES)
    return out


@mcp.tool()
def list_jobs() -> dict:
    """List all known optimization jobs with their current status."""
    result = []
    for rec in jobs.all_jobs():
        result.append({
            "job_id": rec.get("job_id"), "strategy": rec.get("strategy"),
            "symbol": rec.get("symbol"), "run_ts": rec.get("run_ts"),
            "started_at": rec.get("started_at"),
            "status": jobs.compute_status(rec),
        })
    return {"count": len(result), "jobs": result}
```

- [ ] **Step 4: Ignore the runtime jobs dir**

Append to `/home/ad/strategy-platform-v2/.gitignore`:

```
mcp_server/jobs/
```

- [ ] **Step 5: Run the refusal test (fast) then the full e2e (slow)**

Run: `cd /home/ad/strategy-platform-v2/mcp_server && python3 -m pytest test_optimize.py::test_start_optimization_refuses_large -v`
Expected: PASS.

Run: `cd /home/ad/strategy-platform-v2/mcp_server && python3 -m pytest test_optimize.py -v -m slow`
Expected: PASS — sweep launches, polls to `done`, results returned, job listed.

- [ ] **Step 6: Verify the MCP server still starts and exposes the new tools**

Run: `cd /home/ad/strategy-platform-v2 && timeout 5 python3 mcp_server/server.py < /dev/null; echo "exit=$?"`
Expected: starts and waits for stdio (no import error / traceback); `timeout` kills it, `exit=124` is fine. A traceback is a failure.

Then re-confirm registration health:
Run: `claude mcp list 2>&1 | grep strategy-platform`
Expected: `strategy-platform: … ✔ Connected` (the new tools appear as `mcp__strategy-platform__*` in a fresh Claude session).

- [ ] **Step 7: Commit**

```bash
cd /home/ad/strategy-platform-v2
git add mcp_server/server.py mcp_server/test_optimize.py .gitignore
git commit -m "feat(mcp): start_optimization / check_optimization / list_jobs tools"
```

---

### Task 4: Update README

Document the three new tools and the fire-and-poll model.

**Files:**
- Modify: `mcp_server/README.md`

- [ ] **Step 1: Add an optimization section to the README**

Insert after the tools table in `mcp_server/README.md`:

```markdown
## Optimization (fire-and-poll)

Full optimizer sweeps run for minutes to hours, so they don't fit a single
request/response tool call. Instead:

| Tool | Purpose |
|------|---------|
| `start_optimization` | Launch a sweep **detached**; returns a `job_id` immediately. Refuses grids over `MAX_COMBOS` (10000) unless `confirm_large=True`. |
| `check_optimization` | Poll a `job_id` → `running` (elapsed + log tail), `done` (top OOS results), or `failed` (log tail). |
| `list_jobs` | All known jobs with current status (works across sessions). |

The sweep runs as a detached subprocess wrapping `pipeline.run_pipeline`,
which persists to `sp_optimizer_runs`. "Done" is detected when the run's
`run_ts` appears in that table — so completion survives the Claude session
ending. Job records live in `mcp_server/jobs/` (git-ignored).

Example flow: "optimize mobobands on MCL 5m for Jan–Apr" → `start_optimization`
→ later "is that sweep done?" → `check_optimization` → results, also visible
in the dashboard like any other optimizer run.
```

- [ ] **Step 2: Commit**

```bash
cd /home/ad/strategy-platform-v2
git add mcp_server/README.md
git commit -m "docs(mcp): document fire-and-poll optimization tools"
```

---

## Self-Review

**Spec coverage:**
- `start_optimization` (validate, combo cap, detached spawn, job file, immediate return) → Task 3. ✓
- `check_optimization` (DB→done / pid→running / else→failed; capped results + log tail) → Task 3, logic in `jobs.compute_status` (Task 1). ✓
- `list_jobs` (cross-session, recomputed status) → Task 3. ✓
- `opt_runner.py` thin wrapper → Task 2. ✓
- Job registry (`job_<id>.json` + `opt_<id>.log`) → Task 1 + Task 3. ✓
- `MAX_COMBOS=10000` + `confirm_large` → Task 3 constant + refusal test. ✓
- DB-driven completion via `list_optimizer_run_timestamps` → `jobs.run_in_db` (Task 1). ✓
- `run_ts` format `YYYYmmdd_HHMM` → `jobs.make_run_ts` (Task 1); determinism secured by optional `run_ts` in `run_pipeline` (Task 2). ✓
- Context caps (≤50 rows, ≤20 log lines) → Task 3 constants. ✓
- Testing (tiny grid → poll → assert DB run + OOS rows) → Task 3 e2e. ✓
- Out-of-scope items (progress %, stop/pause, scheduling) → none added. ✓

**Resolved during planning:** the spec assumed the parent knows `run_ts` up front, but `run_pipeline` generated its own timestamp internally. Task 2 adds a backward-compatible optional `run_ts` parameter to `run_pipeline` so the job record is deterministic — this is the only platform-file edit and is explicitly tested for backward-compatibility (Task 2, Step 5).

**Placeholder scan:** none — all steps contain concrete code/commands.

**Type consistency:** `jobs.*` signatures in Task 1's Produces block are used verbatim in Tasks 2–3 (`count_combos`, `run_in_db`, `compute_status`, `make_run_ts`, `sym_safe`, `log_path`, `write_job`, `read_job`, `all_jobs`, `tail`, `_ensure_dir`, `JOBS_DIR`). `run_pipeline(... run_ts=...)` added in Task 2 is consumed by `opt_runner.py` in the same task. Tool signatures match the spec.
