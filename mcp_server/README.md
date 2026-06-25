# Local MCP server for strategy-platform-v2

A local, stdio-based [MCP](https://modelcontextprotocol.io) server that gives
Claude tools pointed at **your** engine and **your** MySQL data. It imports the
platform in-process, so there's no logic duplication and **nothing leaves this
machine** (unlike a remote MCP such as trader-dev).

## What it exposes

| Tool | Purpose |
|------|---------|
| `list_strategies` | All registered strategy names |
| `get_strategy_params` | Defaults, param grid, display names, description |
| `list_symbols` | Symbols known to the data loader |
| `get_data_coverage` | Tick/commission meta + tick-data coverage for a symbol |
| `get_bars` | Small OHLCV preview (row-capped at 50) |
| `run_backtest` | Run on the real engine; returns metrics + capped trades. `save=True` persists to results_store |
| `list_backtests` / `load_backtest` | Browse saved manual backtests |
| `list_optimizer_runs` / `load_optimizer_run` | Browse saved optimizer runs (stage a/b) |
| `set_label` | Label a saved backtest or optimizer run |
| `delete_result` | Delete a saved result — **refuses unless `confirm=True`** |

`timeframe` accepts `1m`, `5m`, `15m`, `60m`, … or `Ntick` (e.g. `1300tick`).
Dates are ISO `YYYY-MM-DD`.

## How it works

```
Claude  ──stdio──▶  mcp_server/server.py  ──import──▶  strategy_platform/*
                    (thin adapters only)              registry / loader /
                                                      results_store / engine
```

`server.py` adds the repo root to `sys.path` and loads `.env`, so it uses the
same DB config (`DB_HOST=192.168.1.228`, etc.) as the dashboard. Each tool is a
~10-line adapter: validate → call a platform function → JSON-serialise.
Bar/trade payloads are capped so a large DataFrame never floods context.

## Requirements

System `python3` (3.10) with the platform deps already installed, plus the MCP
SDK:

```bash
python3 -m pip install --user "mcp>=1.2"
```

## Run / test

```bash
# Smoke test against the real engine + DB (read/run only; writes nothing):
python3 mcp_server/test_smoke.py
```

## Registration with Claude

Registered at **user** scope (available in all projects):

```bash
claude mcp add --transport stdio --scope user strategy-platform -- \
    python3 /home/ad/strategy-platform-v2/mcp_server/server.py
```

Verify: `claude mcp list` should show `strategy-platform … ✔ Connected`.
Tools appear as `mcp__strategy-platform__*` in a new Claude session.

## Notes

- **Local only.** stdio transport, no network egress, no auth needed.
- **Safety.** Read/run/save are free; `delete_result` requires `confirm=True`.
- **Always in sync.** Because it imports the live platform, changes to a
  strategy are reflected immediately — no regeneration step.
