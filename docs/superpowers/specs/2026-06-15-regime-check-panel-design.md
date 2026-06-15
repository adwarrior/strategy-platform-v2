# Regime Check panel — design

**Date:** 2026-06-15
**Goal:** Let the user run the Markov 2.0 regime IS-vs-OOS comparison (Step 0 of the 3-period
validation workflow) from *inside* the strategy-platform dashboard, where sweeps are launched —
instead of the standalone `markov-hedge-fund-method` CLI skill, which can't be invoked from the platform.

## Problem

Sweeps run in the Streamlit dashboard (Configure & Run tab). The regime-comparison tool lives in a
separate CLI skill. The user cannot, and will not, drop to a terminal before every sweep. The check
needs to be one click away from the Run button, reusing the symbol/dates already set for the sweep.

## What it does

For the symbol + date range + In-Sample split % already chosen in Configure & Run, derive the IS and
OOS windows, load price history, label each day Bull/Bear/Sideways (Markov 2.0 method), and report
whether the two windows have comparable regime mix:

- Regime distribution (Bear / Sideways / Bull %) for IS and OOS
- Dominant regime + net return per window
- Jensen–Shannon divergence between IS and OOS (0 = identical, 1 = disjoint)
- Verdict: **PROCEED / WARN / RED FLAG** + advice (e.g. "add a confirm window dominated by Bear")

Non-blocking: the verdict is informational; the Run button is unaffected.

## Architecture (3 isolated units)

1. **`strategy_platform/regime/markov_regime.py`** — pure logic, no I/O, no Streamlit.
   Lifted from the skill's honest core: `label_regimes` (20d rolling return, ±5% threshold),
   stride-sampled `build_transition_matrix`, regime-mix, JSD, verdict. Single function
   `compare_windows(daily_close, is_start, is_end, oos_start, oos_end) -> dict`.
   One source of truth for the math; unit-testable with a synthetic series.

2. **`strategy_platform/regime/data.py`** — `load_daily(symbol, start, end) -> pd.Series`.
   Auto-picks the source: `load_1m` for MES/MNQ/MGC (5m table lacks them), `load_5m` otherwise.
   Resamples to daily close. The only DB-touching piece.

3. **`_render_regime_check(symbol, run_start, run_end, train_pct)` in `app.py`** —
   an `st.expander("🎲 Regime Check (IS vs OOS)")` placed just above the Run/Stop controls
   (~line 2646) in the Configure & Run tab. Derives IS = [start, split_date], OOS = (split_date, end]
   from the In-Sample split %, calls units 1+2 inside a button-gated block (don't hit the DB on every
   rerun), renders the verdict with `st.success/warning/error` and a small mix table.

## Data flow

symbol + run_start + run_end + train_pct  →  derive split_date  →  load_daily(symbol, run_start, run_end)
→  compare_windows(daily_close, IS, OOS)  →  render verdict + mix + advice.

## Error handling

- Missing dates / split → info message, do nothing.
- `load_daily` empty or DB error → `st.error` with the reason; no crash.
- < ~40 daily obs in a window → warn that the regime estimate is unreliable but still show it.
- Symbol not in either table → caught by loader, surfaced as `st.error`.

## Testing

- Unit: `compare_windows` on a synthetic bull→bear→flat series returns the expected dominant regimes
  and a sensible verdict.
- Integration: import-check `app.py` compiles; manual one-click check on MNQ in the running dashboard.

## Out of scope (YAGNI)

- No auto-gate at Run time (user chose non-blocking).
- No 3rd "confirm" window input in v1 (advice still names the regime to confirm; can add later).
- No new tab; lives in Configure & Run.
