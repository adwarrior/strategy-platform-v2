# Rewrite of the `port-strategy` skill + reusable parity harness

**Date:** 2026-06-25
**Status:** Approved — pending implementation
**Skill source:** `~/Scripts/claude-config/skills/port-strategy/SKILL.md` (deploy via `apply.sh`)
**Harness:** `strategy-platform-v2/scripts/parity_check.py`

## Goal

Replace the stale 169-line "bootstrap a multi-agent NT→Python pipeline" prompt
with a mode-based skill that authors/ports strategies **into the platform that
already exists** (`strategy-platform-v2`), gated on a two-tier, trade-by-trade
parity check encoded as a reusable harness.

The old skill describes building infrastructure (registry, loader, optimizer,
agent system) that is already built. Running it today would duplicate or
contradict the platform. The rewrite re-points it at the real workflow:
**author/port a strategy, register it, and prove the NT8 and Python versions
match before declaring done.**

## Why a parity gate is the centerpiece

A port that backtests well in Python but diverges from its NT8 source gives
false confidence — worse than no port. The user's history confirms parity is
where ports live or die: OpenRetest ("tick parity established, 4 Python bugs
fixed"), MoboBands (EOD-cutoff + UTC divergence bugs), ORB30 Monti ("parity
verified bar-for-bar"). The `PARITY_REPORT_3_per_trade.md` post-mortem shows the
two biggest parity-killers are **not logic bugs**: contract-series mismatch
(continuous MNQ vs `MNQ JUN26` — entirely different OHLC) and NT data outages.
The harness must surface those as distinct causes before logic debugging begins.

## Architecture

```
SKILL.md (port-strategy, rewritten)
  Step 0: detect mode (port / author / reverse / fix-parity)
  Step 1: load conventions (DB tz, UTC export, tick-load, commission,
          BaseStrategy contract, expose-every-NT-input)
  Step 2: author/port the side(s) that are missing
  Step 3: register the Python side in strategy_platform (@register)
  Step 4: call scripts/parity_check.py
            Tier 1: Python(MySQL)      vs NT trade log   (same data)
            Tier 2: Python(NT .txt export) vs NT trade log (pure logic)
  GATE:   not "done" until Tier-2 trades match within tolerance
```

## The four modes

All modes converge on Steps 3–4 (register + parity-gate).

| Mode | Input | Produces |
|------|-------|----------|
| `port` (C#→Python) | `.cs` in `~/Scripts/strategies/` | Python `BaseStrategy` in the platform |
| `author` | written strategy idea | NT8 C# **and** Python pair |
| `reverse` (Python→C#) | existing Python strategy | NT8 C# for execution |
| `fix-parity` | both versions already exist | (no authoring) parity run + debug |

Mode detection in Step 0: if a `.cs` path is given → `port`; if a Python
strategy name that already has an NT `.cs` → `fix-parity`; if only a Python
strategy → `reverse`; if only a prose description → `author`. When ambiguous,
ask.

## The reusable harness: `scripts/parity_check.py`

Generalizes the existing `parity_orb30monti.py` (MySQL data) and
`parity_orb30monti_nt.py` (NT native export) into one tool.

### Interface

```python
def parity(
    strategy_name: str,           # registered platform strategy
    params: dict,                 # NT-matching param set
    nt_trade_log: str,            # path to NT per-trade export
    symbol: str,                  # e.g. "MNQ"
    timeframe: str,               # e.g. "5m"
    start: str, end: str,         # window matching the NT run
    nt_export_file: str | None = None,  # NT native OHLC .txt for Tier 2
    tolerance: dict | None = None,      # price/time/pnl tolerances
) -> dict:                        # {tier1: {...}, tier2: {...}, verdict: "pass"|"fail", report_path}
```

### Tier 1 — same data

Load the same window from MySQL via `loader.load_5m`/`load_1m`/`load_tick_bars`,
run the registered Python strategy, diff its trades against the NT trade log.

### Tier 2 — pure logic (required for "done")

Parse NT's **native export**, convert UTC→ET-naive, build bars at the strategy's
bar type, run Python on **that** data, diff against the NT trade log. Any
remaining drift is pure strategy-logic difference, not data difference.

The native export comes in two shapes the parser must handle:
- **1-minute OHLC**, format `'YYYYMMDD HHMMSS;O;H;L;C;V'` (semicolon) → for
  time-bar strategies; resample to the strategy timeframe.
- **Tick**, format `'YYYYMMDD HHMMSS fraction;price;...;volume'` (subsecond
  timestamp, one print per line; e.g.
  `NinjaResults/NQ 09-26_16-24.Last.txt`) → for tick-bar strategies; aggregate
  to N-tick bars (the `bar_type='tick'` / `tick_bar_size` path), **not** a
  minute resample.

The harness selects bar-building based on the registered strategy's `bar_type`.

### NT trade-log format

The harness reads NT's per-trade CSV export (e.g. `STF_89Tick_Trades.csv`),
**not** the tab-separated `Performance` summary (which is metrics-only and used
only for a coarse cross-check). Schema, with the parse traps:
- columns include `Market pos.` (Long/Short → direction), `Entry price`,
  `Exit price`, `Entry time`, `Exit time`, `Profit` (dollar string like
  `-$18.98` → strip `$`/`,`), `MAE`, `MFE`, `Bars`;
- **dates are day-first `DD/MM/YYYY HH:MM:SS`** — parse with `dayfirst=True`, a
  known trap;
- the summary CSV (`*_Summary.csv`) is the metrics source for the coarse check.

### Trade matching (encodes the learned key)

For **time-bar** strategies, NT exports `entry_time` as the bar **open** time
while Python uses the sub-bar timestamp; correct match for timeframe `T`:
`NT_entry_time == Python_entry_time.ceil(T) - T`, same direction (the validated
`ceil('5min') - 5min` rule, generalised).

For **tick-bar** strategies there is no fixed minute grid, so match on nearest
entry time within a small time window (e.g. ±1 bar's typical duration) plus same
direction, then confirm with entry price within tolerance.

Compare matched trades on: entry time, exit time, direction, entry price, exit
price, `pnl_ticks` — each within `tolerance`.

### Pre-flight guards (report distinct causes before logic debugging)

Before attributing mismatches to logic, the harness checks and flags:
1. **Contract-series mismatch** — if matched-bar price deltas are
   systematically large and roughly constant/decaying (the continuous-vs-`JUN26`
   forward-premium signature), warn that the two sides are on different OHLC
   series. This is a data problem, not a logic bug.
2. **Date-coverage gaps** — count NT-only and Python-only trading days; a block
   of NT-only or Python-only days signals a data outage on one side.

### Output

Writes `PARITY_REPORT.md` (to the platform `reports/` dir) with: matched /
NT-only / Python-only counts, per-trade diff table for matched trades,
pre-flight findings, and the pass/fail verdict. Returns the same as a dict.

A `pass` requires the Tier-2 matched-trade set to agree within tolerance with
**no** unexplained NT-only/Python-only trades inside the common date range.

## Conventions the skill loads up-front

Encoded as guardrails (from the user's memory + platform code), kept in
`references/conventions.md` so `SKILL.md` stays lean:

- **DB timezone:** `historical_data` is ET; `historical_data_1m` and `tick_data`
  are UTC — `tz_convert` before any session-hour logic.
- **NT export:** UTC regardless of chart TZ; format `'YYYYMMDD HHMMSS;O;H;L;C;V'`
  (semicolon-separated). User exports via Tools→Historical Data, "Get Data"
  first to fill gaps.
- **Tick data:** load monthly-and-concat to avoid OOM (esp. MNQ);
  `bar_type='tick'` strategies must include `tick_bar_size` in `param_grid`.
- **Commission:** per-instrument round-trip rate from `Commisions.txt`.
- **BaseStrategy contract:** set `name`, `default_params`, `tick_size`,
  `tick_value`, `commission_rt`; implement `run_backtest(data, params) -> dict`
  (must return `net_pnl`, `total_trades`, `win_rate`, `sharpe`, `max_drawdown`);
  implement the `param_grid` property; decorate with `@register`.
- **Expose every NT input** as a platform param so all are optimizable.
- **Fast iteration:** use the local MCP `run_backtest` tool for Python-side
  backtests while debugging divergences.

## Files & maintenance

- **Skill source:** edit `~/Scripts/claude-config/skills/port-strategy/SKILL.md`
  and `references/conventions.md`, then run the repo's `apply.sh` to deploy to
  `~/.claude/skills/port-strategy/`. Direct edits to `~/.claude/skills/` get
  clobbered by sync (the Markov-skill lesson).
- **Harness:** `strategy-platform-v2/scripts/parity_check.py`, committed with
  the platform.

## Testing

Two layers, because the once-used ORB30 reference files
(`/home/ad/Scripts/Results/Ninja.txt`, `/home/ad/data/MNQ 06-26.Last.txt`) were
transient and no longer exist on disk.

1. **Guaranteed gate — synthetic unit tests.** The harness's logic is tested
   against hand-built inputs, with no dependency on any external NT file:
   - the trade-matching key (`Python_entry_time.ceil(T) - T`, same direction)
     matches and mismatches correctly for a few constructed trade pairs;
   - the NT native-export parser reads the `'YYYYMMDD HHMMSS;O;H;L;C;V'` format
     and converts UTC→ET-naive correctly;
   - the pre-flight guards fire: a constant/large price-delta set triggers the
     contract-series warning; a block of one-sided trading days triggers the
     date-coverage warning.
   These run in CI-style isolation and are the harness's real regression test.

2. **Live integration check — SuperTrendFractal.** A usable matched pair exists:
   - NT per-trade log: `~/Scripts/Results/NinjaResults/STF_89Tick_Trades.csv`
     (76 trades, NQ SEP26, 16–24 Jun 2026);
   - NT native tick export: `~/Scripts/Results/NinjaResults/NQ 09-26_16-24.Last.txt`;
   - registered Python port: `supertrendfractal` (tick-bar, NQ, $5/tick);
   - live config in `STF_HANDOFF.md` (ATR mult 3, ATR period 10, fractal len 3,
     Both, FixedTPTrailSL TP 80t / trail SL, cooldown 2, session 09:45–11:00,
     EOD 16:55).

   Run `parity_check.py` end-to-end on this as the live smoke test. It exercises
   the harder paths: the CSV trade-log parser (day-first dates, `$` stripping),
   tick-export → 89-tick-bar building, and the tick-bar matching rule.

   **Caveat — treat result as a smoke signal, not a hard gate.** `STF_HANDOFF.md`
   asserts "parity confirmed" but not bar-for-bar, and the live run was on NQ
   while the platform's tick history is MNQ (a documented proxy with ~2.15×
   tick tempo). A residual mismatch here may reflect that data confound rather
   than a harness bug — which is exactly what the pre-flight contract-series /
   coverage guards should surface. The synthetic unit tests (layer 1) remain the
   hard gate; this layer proves the parsers and bar-builders work on real files.

## Out of scope (YAGNI)

- The old multi-agent manager/translator/optimizer/analyst system — the
  platform + local MCP already provide this.
- Auto-running optimizations — that is the local MCP's `start_optimization`.
- Live execution — NinjaTrader remains execution-only; this skill stops at a
  verified, registered, optimizable Python port (and/or generated C#).
```
