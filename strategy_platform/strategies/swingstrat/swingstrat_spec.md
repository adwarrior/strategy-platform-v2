# SwingStrat (SMC "Ara 4HR" method) — Spec

Single source of truth for the multi-timeframe Smart Money Concepts strategy registered as
`swingstrat_15m` / `swingstrat_5m`, and the NinjaTrader C# twin `SwingStrat.cs`.

Source method: transcript `Scripts/Notes/Transcribes/Ara 4HR Strat`. The video's 4-item
checklist is **HTF alignment → liquidity sweep → BOS + imbalance → 71% retracement**.

## TL;DR of audit (2026-06-18)

The existing Python `swingstrat` **already implements 3 of the 4 checklist items**
(HTF premium/discount, BOS leg, FVG imbalance, 71% Fib entry, Fib SL/TP). The **only missing
item is the liquidity-sweep pre-condition (#2)**. This spec documents the whole strategy and
defines the sweep gate to be added (optional, default OFF).

---

## Timeframes

- **HTF = 4H** (`htf_minutes=60` resample base; legs computed on the HTF series).
- **LTF = 15M** (`swingstrat_15m`) or **5M** (`swingstrat_5m`) — execution.
- No-lookahead: HTF legs are shifted `available_from = confirmed_time + htf_minutes`, so an
  LTF bar can only act on an HTF leg after the HTF bar that confirmed it has fully closed.

## State machine (per LTF bar)

`leg_active → scanning → armed → in_trade → (flat, back to leg_active)`

1. **leg_active** — an HTF leg is available; wait for price to cross **fib_50** into the
   correct zone (short: price must be in premium > fib_50; long: in discount < fib_50).
   This is checklist item #1, **HTF alignment / premium-discount**.
2. **scanning** — price is in the [fib_50 → fib_71] retracement zone; scan the last
   `fvg_lookback_bars` LTF bars for an FVG near fib_71 (checklist #3, imbalance).
   - If FVG found near fib_71 → **arm** a limit at the FVG edge.
   - Tier-2 fallback (`enable_tier2`): if price reaches the [fib_71, fib_75] band with no
     qualifying FVG, arm a limit at fib_75. (This is an extension beyond the video.)
3. **armed** — limit order resting. Cancel/invalidate if the FVG is breached
   (`is_fvg_invalidated`) or price closes back beyond fib_50 (leg context lost).
4. **in_trade** — entry filled at the armed price (checklist #4, 71% retracement).
   - **SL** = `fib range extreme` (origin of leg) in `SwingExtreme` mode, OR the FVG far edge
     in `FVGInvalidation` mode.
   - **TP** = `destination` of leg (the swing the move came from) = Fib-range opposite end.
   - Conservative fill ordering: if both SL and TP are touchable in one bar, **SL wins**.

## Leg / BOS definition (existing, unchanged)

- Fractal swing: high/low extreme over ±`swing_period` bars, **confirmed at i+swing_period**
  (no lookahead).
- **BOS DOWN** (new SHORT leg): HTF `close < last confirmed swing low`.
  origin = last swing high, destination = last swing low.
- **BOS UP** (new LONG leg): HTF `close > last confirmed swing high`.
  origin = last swing low, destination = last swing high.
- Each extreme consumed once (`last_swing_* = None` after a leg fires).

## Fib levels (existing, unchanged)

Measured origin→destination of the leg. `fib_50` = premium/discount divider; entry zone
`fib_min..fib_max` defaults **0.71..0.75**; `tp = destination`, `sl = origin`. Gives the
video's ~2.45 R:R by construction.

---

## NEW — Liquidity-sweep gate (checklist #2)  *[to be added, default OFF]*

**Concept (from video):** before a valid BOS leg, price should *sweep* a prior major
high/low — a **wick pierces the level but the bar body does not close beyond it** — purging
resting liquidity. A *body close* beyond the level is a break (continuation), not a sweep.

**Deterministic rule (HTF, evaluated at the bar a leg is confirmed):**

For a leg confirmed at HTF bar index `c` with direction `dir`:
- Look back over the `sweep_lookback` HTF bars **ending at the leg's BOS bar** (i.e. bars
  `[c - sweep_lookback, c]`).
- The level to be swept is **the leg's origin extreme** (`leg['origin']`):
  - **SHORT leg** → origin = the swing **high** the impulse fell from (buy-side liquidity
    that price ran above and rejected to kick off the down-move).
  - **LONG leg** → origin = the swing **low** the impulse rose from (sell-side liquidity
    that price ran below and rejected to kick off the up-move).
  - Rationale: the origin extreme is the level whose liquidity grab *launched* the very
    impulse that created the FVG — so it is both causally-correct (pre-dates the FVG, no
    look-ahead) and the closest relevant extreme to the FVG/entry zone. Already computed in
    `compute_htf_legs`, so no new pivot logic.
- A **valid sweep** exists within the lookback if there is at least one HTF bar `k` where:
  - SHORT: `high[k] > swept_level` **AND** `close[k] <= swept_level`
    (wicked above the high, closed back below — buy-side sweep).
  - LONG: `low[k] < swept_level` **AND** `close[k] >= swept_level`
    (wicked below the low, closed back above — sell-side sweep).
- If `require_liquidity_sweep` is **True** and no valid sweep is found in the window, the leg
  is **rejected** (no trade context created from it). If **False** (default), legs behave
  exactly as today — **zero change to existing results.**

**Why this rule:** it reuses the already-computed fractal series (no new pivot logic),
encodes the video's "wick-beyond, no body-close" definition exactly, and is bar-deterministic
so Python and NinjaTrader can agree.

**New params:**
| param | type | default | group | meaning |
|---|---|---|---|---|
| `require_liquidity_sweep` | bool | `False` | "Liquidity Sweep" | gate legs on a prior sweep |
| `sweep_lookback` | int | `10` | "Liquidity Sweep" | HTF bars before BOS to search for the sweep |

`param_grid`: `require_liquidity_sweep: [False, True]`, `sweep_lookback: (5, 20, 5)`.
`param_conditional`: `sweep_lookback` only shown/swept when `require_liquidity_sweep` enabled.

---

## NinjaTrader C# twin — `SwingStrat.cs`

Mirror the above on NT8, following NYBreakout/SuperTrendFractal conventions:
- `AddDataSeries(BarsPeriodType.Minute, 240)` for the 4H bias series; chart/exec = 15M (or 5M).
  Guard logic with `BarsInProgress`.
- Fractal swing detection on the 4H series; BOS on 4H close vs last confirmed swing.
- Premium/discount via fib_50 of the active leg; FVG scan on exec series; arm a limit
  (`EnterLongLimit`/`EnterShortLimit`) at the FVG edge / fib_75 (Tier-2).
- SL/TP via `SetStopLoss`/`SetProfitTarget` at fib range extremes (or FVG-invalidation SL).
- Liquidity-sweep gate as above, `RequireLiquiditySweep` (default false) + `SweepLookback`.
- Risk sizing (`use_risk_sizing`/`max_risk`/`qty`, skip if qty<1), session-aware EOD
  (mirror SuperTrendFractal's `SessionEndForEntry`), **uncheck "Break at EOD"** in Analyzer.

## Parity test plan

MNQ, 15M exec + 4H bias, single-contract NT CSV for the test window. Compare aggregate stats
(trades, net PnL, PF, win rate) — pass threshold ~15% absolute PnL delta, robust relative
ranking. Sweep gate OFF for the first parity pass (matches current Python), then ON.

## Open questions / risks

- **Sweep "swept_level" selection** — RESOLVED 2026-06-18: use the **leg's origin extreme**
  (`leg['origin']`). It is the level whose liquidity grab launched the impulse that formed the
  FVG → causally correct, no look-ahead, closest relevant extreme to the entry. Default-off +
  sweepable, so the backtest will reveal whether the gate adds edge.
- `tick_size` in `detect_fvgs` defaults 0.10 (Gold) — must be injected from `INSTRUMENT_META`
  at runtime for MNQ (0.25). Verify the pipeline/dashboard applies symbol meta (it does for
  other strategies; confirm during Phase 2).
