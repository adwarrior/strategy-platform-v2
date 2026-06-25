# Fire-and-poll optimization for the local MCP server

**Date:** 2026-06-25
**Status:** Approved — pending implementation
**Repo:** `strategy-platform-v2`
**Builds on:** the local MCP server at `mcp_server/` (read/run/results tools)

## Goal

Add two MCP tools — `start_optimization` and `check_optimization` — plus a
small job registry and a `list_jobs` tool, so Claude can launch a full
optimizer sweep that:

- returns control **immediately** (no multi-hour hang on the tool call),
- runs **detached**, surviving the Claude/MCP session ending,
- lands its results in the existing `sp_optimizer_runs` DB, where the
  already-built `list_optimizer_runs` / `load_optimizer_run` tools read them.

Single backtests already fit the MCP request/response shape and exist. Full
optimizations do not — they run for minutes to hours. This closes that gap
without re-implementing the optimizer.

## Why this shape

Two facts about the existing platform make fire-and-poll clean:

1. The Streamlit dashboard already launches optimizations as **detached
   `subprocess.Popen` jobs** (and has `_terminate_proc_tree` to kill them).
   So a subprocess launcher is the proven, in-house pattern — not a new idea.
2. `strategy_platform.optimize.pipeline.run_pipeline(...)` is the real
   entrypoint and **persists its own results** via
   `results_store.save_optimizer_run(...)` when it finishes.

Therefore: launch `run_pipeline` in a detached subprocess; "done" is detected
by the run's `run_ts` appearing in `sp_optimizer_runs`. No new optimization
logic, same engine, and results show up in the dashboard exactly as a
dashboard-launched run would.

## Components

```
mcp_server/
  server.py        ← + start_optimization, check_optimization, list_jobs
  opt_runner.py    ← NEW: thin CLI wrapper around pipeline.run_pipeline
  jobs/            ← NEW: job_<id>.json registry + opt_<id>.log per run
```

### `opt_runner.py` (new)

A minimal CLI invoked as a detached subprocess. Parses args, calls
`pipeline.run_pipeline(...)` with them, and exits. All stdout/stderr is
redirected by the parent to `jobs/opt_<id>.log`. It contains **no** logic
beyond arg-marshalling — `run_pipeline` does the work and the saving.

Args mirror the relevant `run_pipeline` parameters:
`strategy_name, symbol, timeframe_mins, data_start, data_end, train_pct,
rank_by, min_trades`, and a JSON-encoded `param_grid_override` (passed via a
temp file path to avoid shell-quoting issues with large grids).

### Job registry (`jobs/`)

One `job_<id>.json` per launch:

```json
{
  "job_id": "a1b2c3",
  "pid": 48213,
  "run_ts": "20260625_1642",
  "strategy": "mobobands",
  "symbol": "MCL",
  "sym_safe": "MCL",
  "started_at": "2026-06-25T16:42:01Z",
  "log_path": "/.../mcp_server/jobs/opt_a1b2c3.log",
  "status": "running"
}
```

`status` is advisory; the authoritative completion signal is always the DB.

## Tools

### `start_optimization`

```
start_optimization(
    strategy, symbol,
    timeframe="5m",
    data_start=None, data_end=None,
    param_grid=None,          # dict param->list; None = strategy's full grid
    train_pct=0.70,
    rank_by="sharpe",
    min_trades=None,          # platform default if None
    confirm_large=False,
) -> dict
```

Behaviour:

1. Validate `strategy` (registry), `symbol` (loader meta), `rank_by` in the
   platform's `RANK_METRICS`. Fail fast with a clear error before spawning.
2. Resolve the effective grid (`param_grid` or the strategy's full grid) and
   count deduplicated combinations via the platform's
   `_deduplicated_combinations`. If the count exceeds **`MAX_COMBOS` (default
   10000)** and `confirm_large` is not `True`, **refuse** with the count and a
   message to re-call with `confirm_large=True`.
3. Generate `job_id` and `run_ts` (`YYYYmmdd_HHMM`, matching `run_pipeline`).
4. Write the grid override to a temp JSON file; spawn `opt_runner.py`
   **detached** (`start_new_session=True`), redirecting stdout/stderr to
   `jobs/opt_<id>.log`.
5. Write `job_<id>.json`.
6. Return immediately: `{job_id, run_ts, status:"running", combos, log_path}`.

### `check_optimization`

```
check_optimization(job_id) -> dict
```

Decision order:

1. `run_ts` present in `sp_optimizer_runs` (via
   `list_optimizer_run_timestamps`)? → **`done`**. Return the top-N rows of
   the OOS stage (capped at 50) plus the run's saved settings, and update the
   job file `status="done"`.
2. Else process alive (pid signal-0 check)? → **`running`**. Return elapsed
   time and the last ~20 lines of the log.
3. Else (pid dead, no `run_ts`) → **`failed`**. Return the log tail so the
   traceback is visible; update job file `status="failed"`.

### `list_jobs`

```
list_jobs() -> dict
```

Read all `job_<id>.json` files and return each with its **freshly recomputed**
status (same logic as `check_optimization`), so "what optimizations are
running?" works across sessions.

## Error handling & safety

- **No accidental mega-sweeps.** Combination cap (`MAX_COMBOS=10000`) gates
  `start_optimization`; over the cap requires `confirm_large=True`.
- **Fail fast.** Unknown strategy/symbol/rank_by errors in the parent before
  any subprocess is spawned.
- **No stuck jobs.** A dead pid with no DB run is reported `failed` with the
  log tail — never an eternal "running".
- **No kill tool in v1** (YAGNI). The pid is surfaced; a runaway sweep can be
  `kill <pid>`-ed manually. A `stop_optimization` tool can be added later if
  it proves needed.

## Testing

`mcp_server/test_optimize.py`:

1. Launch a **deliberately tiny grid** (a handful of combos, ~seconds) on a
   small date range via `start_optimization`.
2. Poll `check_optimization` until `status == "done"` (with a timeout).
3. Assert the `run_ts` appears in `sp_optimizer_runs` and the OOS stage has
   rows.

This exercises the real detached-spawn → pipeline → DB-persist → poll loop
end to end, quickly. The test must not require `confirm_large` (tiny grid).

## Out of scope (YAGNI)

Progress percentage, stop/pause/resume, scheduling, autoresearch wiring,
multi-symbol batch launches. Logs plus the combination count are sufficient
visibility for v1.
