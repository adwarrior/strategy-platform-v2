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
