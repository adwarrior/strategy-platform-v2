# Aurora bar-type sweep — design (2026-07-11, user-approved)

**Question:** is 1-minute the right bar type for the Aurora intercept-scalp, or do
other time frames / tick bars do better?

**Decisions (user):**
- Test both families: time bars 1m (baseline), 2m, 3m, 5m; tick bars 1000t,
  1597t, 2500t. Nothing under 1000t — smaller tick bars close so often the
  trade count explodes.
- Keep the live 07-08 config verbatim on every bar type (no param rescaling) —
  mirrors what switching the chart's bar type in NinjaTrader would do. The
  engine's lookback day-cap applies to time frames; tick bars use the raw bar
  count (C# EffectiveLookback fallback).

**Implementation:**
- `bar_spec` param on the Aurora port (`'Nmin'` / `'Nt'`, default `'1min'`).
  Time bars: pandas floor + close-time label shift (validated parity fix).
  Tick bars: every N tick events = one bar, labelled by last tick's timestamp
  (NT convention). Bars may straddle the maintenance break (accepted
  approximation). Everything downstream is bar-type agnostic.
- `scripts/sweep_bartype.py` — same harness as sweep_aurora.py: per-config
  JSONL + resume, spans per contract, May+Jul pass chained to June OOS pass.
  Results: `results/sweep_bartype*.{log,jsonl}`.

**Verification:** 16 aurora tests pass; May-1 smoke — 1min reproduces the
pre-change result exactly (PF 1.65, net +364); 5min/1000t/1597t run sanely
(6/40/31 trades).

**Caveats:** port fills are deliberately conservative; tick-bar commission drag
grows with trade count; per-bar-close touch semantics mean tick bars judge
touches far more often in fast tape.
