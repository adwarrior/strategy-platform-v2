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

Parse NT's **native export** (format `'YYYYMMDD HHMMSS;O;H;L;C;V'`, semicolon,
UTC), convert UTC→ET-naive, resample to the strategy timeframe, run Python on
**that** data, diff against the NT trade log. Any remaining drift is pure
strategy-logic difference, not data difference.

### Trade matching (encodes the learned key)

NT exports `entry_time` as the **5-min bar open time**; Python uses the 1-min
sub-bar timestamp. Correct match:
`NT_entry_time == Python_entry_time.ceil('5min') - 5min`, same direction.
For a timeframe `T`, generalise to `Python_entry_time.ceil(T) - T`.
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

2. **Best-effort integration check.** At implementation time, discover whether a
   usable NT trade-log + NT native-export pair still exists for any ported
   strategy (search `~/Scripts/Results/` and `~/Scripts/Results/NinjaResults/`,
   e.g. the `Ninja*.txt` logs and `*.Last.txt` exports that ARE present). If a
   matched pair is found, run `parity_check.py` end-to-end on it as a live smoke
   test. If none is found, skip this layer (the synthetic tests are the gate)
   and note in the harness README that a fresh NT export is needed to run a full
   live parity check. Do not hard-code the absent ORB30 paths.

## Out of scope (YAGNI)

- The old multi-agent manager/translator/optimizer/analyst system — the
  platform + local MCP already provide this.
- Auto-running optimizations — that is the local MCP's `start_optimization`.
- Live execution — NinjaTrader remains execution-only; this skill stops at a
  verified, registered, optimizable Python port (and/or generated C#).
```
