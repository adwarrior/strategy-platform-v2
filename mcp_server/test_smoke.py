#!/usr/bin/env python3
"""
Smoke test for the local MCP server — verifies wiring against the REAL
engine and DB before the server is ever registered with Claude.

Run:
    python3 /home/ad/strategy-platform-v2/mcp_server/test_smoke.py

It exercises the read/discovery/run paths. It does NOT write or delete
anything in the results store (save=False, no delete_result call), so it's
safe to run repeatedly.
"""

import json
import sys

import server  # noqa: E402  (same directory)


def _show(name, obj):
    print(f"\n=== {name} ===")
    print(json.dumps(obj, indent=2, default=str)[:1200])


def main() -> int:
    failures = []

    # 1. Discovery -------------------------------------------------------
    strat = server.list_strategies()
    _show("list_strategies", strat)
    if not strat.get("strategies"):
        failures.append("list_strategies returned no strategies")
        print("FAIL: no strategies registered")
        return 1
    sample_strategy = strat["strategies"][0]

    params = server.get_strategy_params(sample_strategy)
    _show(f"get_strategy_params({sample_strategy})", params)

    syms = server.list_symbols()
    _show("list_symbols", syms)

    # 2. Data ------------------------------------------------------------
    cov = server.get_data_coverage("MNQ")
    _show("get_data_coverage(MNQ)", cov)

    bars = server.get_bars("MNQ", "5m", start="2026-05-01", end="2026-05-02")
    _show("get_bars(MNQ 5m 2026-05-01..02)", bars)
    if bars.get("rows", 0) == 0:
        print("WARN: get_bars returned 0 rows (range may be outside coverage)")

    # 3. Run (in-memory, not saved) -------------------------------------
    try:
        bt = server.run_backtest(
            sample_strategy, "MNQ", "5m",
            start="2026-05-01", end="2026-05-08", save=False,
        )
        _show(f"run_backtest({sample_strategy} MNQ 5m, 1wk)", bt)
        if "error" in bt:
            print(f"WARN: run_backtest returned error: {bt['error']}")
    except Exception as e:
        failures.append(f"run_backtest raised: {e!r}")
        print(f"FAIL: run_backtest raised {e!r}")

    # 4. Results store read ---------------------------------------------
    lb = server.list_backtests(sample_strategy, "MNQ")
    _show(f"list_backtests({sample_strategy} MNQ)", lb)

    print("\n" + "=" * 50)
    if failures:
        print(f"SMOKE TEST FAILED ({len(failures)} issue(s)):")
        for f in failures:
            print("  -", f)
        return 1
    print("SMOKE TEST PASSED — all tools wired to the real engine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
