import os
import time
import json
import subprocess
import sys
import pytest

import jobs  # mcp_server/ is the CWD when running these tests


# A strategy + symbol + small date range known to produce a run quickly.
# mobobands (time-bar) on MNQ 5m routes through load_1m→resample; yields ~1180 bars.
# WaeJurikPro was a tick-bar strategy — MNQ tick data ends 2026-04-17 so that combo had no data.
SMALL = dict(strategy="mobobands", symbol="MNQ",
             data_start="2026-04-01", data_end="2026-04-08")


def _tiny_grid_for(strategy):
    """Pick the first value of each grid param -> a 1-combo grid (fast).

    NOTE: not safe for strategies whose boolean filter params over-filter when all True
    (e.g. mobobands) — use an explicit minimal grid instead.
    """
    inst = jobs.StrategyRegistry.get(strategy)()
    return {k: [v[0]] for k, v in inst.param_grid.items()}


# Hard-coded 1-combo grid for the slow integration test.
# _tiny_grid_for("mobobands") enables all boolean filters simultaneously,
# which over-filters to zero trades on any short date range.
# This grid turns every enable_* filter off so the base signal fires freely.
_MOBOBANDS_MINIMAL_GRID = {
    "dpo_period": [14], "mobo_length": [20], "num_dev_up": [1.0],
    "num_dev_dn": [1.0], "hook_lookback": [3], "slope_lookback": [3],
    "slope_threshold": [0.0],
    "enable_middle_band_hook": [False], "require_color_change": [False],
    "enable_divergence_filter": [False], "divergence_lookback": [10],
    "enable_bw_filter": [False], "bw_period": [20], "bw_multiplier": [1.0],
    "enable_time_filter": [False],
    "profit_ticks": [20], "stop_ticks": [10], "bars_between_trades": [0],
    "enable_wattah_atar": [False], "enable_jurik_filter": [False],
    "jurik_period": [14], "jurik_phase": [0.0], "jurik_fl_period": [14],
    "enable_adx_filter": [False], "adx_period": [14], "adx_threshold": [20.0],
    "tick_bar_size": [100], "calculate_mode": ["on_bar_close"],
}


@pytest.mark.slow
def test_opt_runner_writes_db_run():
    run_ts = jobs.make_run_ts()  # 13-char YYYYmmdd_HHMM — DB column is VARCHAR(13)
    grid = _MOBOBANDS_MINIMAL_GRID
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


def test_count_combos_uses_pipeline_dedup(monkeypatch):
    called = {}
    def fake_dedup(grid, deps):
        called["yes"] = (grid, deps)
        yield {}; yield {}; yield {}   # 3 combos
    class FakeStrat:
        param_grid = {"a": [1]}
        param_dependencies = {"x": ("y", True)}
    monkeypatch.setattr(jobs.StrategyRegistry, "get", staticmethod(lambda n: (lambda: FakeStrat())))
    monkeypatch.setattr(jobs, "_deduplicated_combinations", fake_dedup)
    assert jobs.count_combos("whatever", {"a": [1]}) == 3
    assert "yes" in called


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


# =========================================================================
# Task-3 tests: start_optimization / check_optimization / list_jobs
# =========================================================================

def test_start_optimization_refuses_large(monkeypatch):
    import server
    monkeypatch.setattr(server.jobs, "count_combos", lambda *a, **k: 999999)
    spawned = {"n": 0}
    def fake_popen(*a, **k):
        spawned["n"] += 1
        raise AssertionError("Popen should not be called on refusal")
    monkeypatch.setattr(server.subprocess, "Popen", fake_popen)
    res = server.start_optimization(strategy="mobobands", symbol="MNQ")
    assert res.get("refused") is True
    assert res["combos"] == 999999
    assert spawned["n"] == 0


@pytest.mark.slow
def test_start_and_check_optimization_e2e():
    import server
    grid = _MOBOBANDS_MINIMAL_GRID
    started = server.start_optimization(
        strategy=SMALL["strategy"], symbol=SMALL["symbol"],
        timeframe="5m", data_start=SMALL["data_start"], data_end=SMALL["data_end"],
        param_grid=grid,
    )
    assert started["status"] == "running", f"unexpected start result: {started}"
    assert "job_id" in started and "run_ts" in started
    job_id = started["job_id"]

    deadline = time.time() + 600
    status = None
    res = {}
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
