# DeltaDiverge — spec

Hourly delta/price divergence at a prior-session extreme. MNQ futures, 1-hour bars.
Registered as `deltadiverge` in `strategy_platform/strategies/deltadiverge/strategy.py`.

## Idea

From a video (NQ, hourly bars): a "seller exhaustion" LONG setup — the hourly
candle closes **above** the prior session's high while the candle body itself
is **negative** (close < open — price nominally fell during the hour), but the
underlying tick-derived volume delta for that bar is strongly **positive**.
Aggressive buyers absorbed the down-move and the close still ended up above
the level: read as the sellers who printed the red candle running out of
supply. The video only shows longs; we add the mirror short.

- **LONG**: `close > prior_session_high` AND `close < open` (red candle) AND
  bar delta trigger fires **positive**.
- **SHORT**: `close < prior_session_low` AND `close > open` (green candle) AND
  bar delta trigger fires **negative**.

## Bar-labelling convention (chosen, and why it matters)

NinjaTrader stamps bars by **close** time. This strategy uses **close-time
labelling** throughout (`pandas.resample(..., label='right', closed='right')`),
matching every other strategy in this platform (magichour, mobobands, the
loader's `resample_ohlcv`). A bar stamped `14:00` covers price action from
`13:00:00.000` to `13:59:59.999` — `14:00` is the hour it *closed*, not opened.

This choice is load-bearing for two things:
1. **Signal timing** — a bar's condition is only evaluated once that bar's
   index timestamp exists (i.e. it has fully printed).
2. **The tick-delta join** — `tick_delta.load_hourly_tick_delta` buckets ticks
   by **open**-time (SQL `FLOOR(unix_ts/3600)`, i.e. hour `14` = ticks from
   `14:00:00` to `14:59:59`). To align with the close-labelled OHLCV bars, the
   tick-bucket index is shifted **+1 hour** before joining (`_attach_tick_delta`
   in `strategy.py`) — an open-labelled `14:00` tick bucket describes the same
   clock hour as the close-labelled `15:00` OHLCV bar.

## No-look-ahead rules

- "Prior session high/low" only uses a session whose **own session-end bar has
  already closed** strictly before the signal bar's timestamp. The signal
  bar's own session (if it's mid-session) is never included.
- The signal bar itself must be **closed** before its condition can be
  evaluated — the loop walks the DataFrame index in order and only looks at
  `df.iloc[i]` (already-closed bars) to decide whether to trade.
- **Entry fills on the next bar's open** (`df.iloc[i+1]['open']`), never on
  the signal bar's own close. This is enforced in `_run_backtest_loop`.

## Data

- **Bars**: `historical_data_1m`, symbol key `MNQ`, loaded via
  `load_1m(symbol, to_et=True)` (table is stored **Central-Time-naive**;
  `to_et=True` applies the fixed +1h CT→ET shift). Resampled 1m → 1H,
  close-stamped, in `strategy.py::_ensure_1h`.
- **Delta**: `tick_data` — columns `(id, symbol, ts, price, bid, ask, volume)`,
  **no `datetime` column**, and **stored in UTC** (confirmed empirically: the
  CME 17:00–18:00 ET maintenance-break gap lands at UTC hour 21–22 in March,
  i.e. before US DST begins, matching UTC = ET+5). Each tick is classified
  `price >= ask → +volume` (buy-initiated), `price <= bid → -volume`
  (sell-initiated), else ignored (~0.04% of MNQ ticks, per prior verification).
  Delta is **summed per UTC hour bucket in SQL** (`strategy_platform/strategies/
  deltadiverge/tick_delta.py`) — raw ticks never land in pandas — then the
  bucket index is converted UTC→ET (DST-aware, via a proper tz round-trip, not
  a fixed offset, since `tick_data` timestamps are true UTC unlike the CT-naive
  bar table).

### Query performance / caching

A single aggregate query over the full ~19-month tick history does not return
in reasonable time — measured: a 1-week window took ~42s, a 3-month window
took **347s**. `load_hourly_tick_delta` therefore chunks by **calendar month**
(one `GROUP BY` query per month, ~batch-safe) and caches each *complete* past
month to Parquet under `.cache/deltadiverge_tick_delta/`. The in-progress
current month is never cached. Repeat runs over a previously-queried window
are near-instant; a first run over a new multi-month window is still slow
(minutes) — this is a known cost of the coverage gate requiring per-hour
tick-derived volume, not a bug.

## Coverage gate (required — tick_data is intermittently thin)

`tick_data` volume is known to run **~42–50%** of true traded volume (a known
thinning artefact — `historical_data_1m` volume is trustworthy, `tick_data`
volume is not, in absolute terms). Coverage is also **intermittent**: some
hours have near-zero tick counts despite large true volume (e.g. one measured
case: 21 ticks against 49,056 contract-volume in the same hour). A near-empty
hour still produces a delta value that looks like a legitimate reading, so an
"is there any data at all" check does not catch it.

For every hourly bar:

```
coverage_ratio = tick_derived_volume / historical_data_1m_volume_for_that_hour
```

- Parameter `min_coverage_ratio` (default **0.25** — comfortably below the
  normal 0.42–0.50 band, comfortably above near-empty cases).
- Bars with `coverage_ratio < min_coverage_ratio`, or with no tick_data
  coverage at all (outside the tick-data date range — before 2024-09-12 or
  after 2026-04-17 for symbol `MNQ` — or zero bar volume), are **excluded**
  from both:
  1. **Signal generation** (`long_fires` / `short_fires` masked `False`), and
  2. **The rolling z-score baseline window** — excluded bars' delta is
     replaced with `NaN` before the rolling mean/std is computed
     (`df['delta'].where(covered)`), so a thin hour cannot silently shrink the
     z-score denominator. They are never treated as `delta = 0`, which would
     be a fabricated data point.
- `run_backtest` returns a `coverage_stats` dict (`total_bars`, `covered_bars`,
  `coverage_pct`, `bars_outside_tick_range`, tick-data start/end actually
  seen) so the true tradeable sample size is visible, not silently dropped.
  Bars outside the tradeable tick-data range are reported, not hidden.

## Trigger types (both required, both implemented)

- `net_delta`: raw `|bar delta| >= delta_threshold`. The video's approach
  (his stated value: 500 on hourly NQ). Simple but tick_data's known ~44%
  volume-thinning makes the absolute magnitude untrustworthy — this mode
  exists for completeness/comparison, not because it's expected to be robust.
- `zscore`: bar delta standardised over a rolling window of `zscore_length`
  **prior, coverage-gated** hourly bars (mirrors the DeltaFlowProfileV2
  indicator's DeltaStrength measure). Triggers at `|z| >= zscore_threshold`.
  Preferred given the volume-thinning caveat, since it's a relative measure.

## Exit

- Fixed **take-profit** (`tp_points`) and **stop-loss** (`sl_points`), both in
  **points** (not dollars) — converted to $ via `tick_value/tick_size`
  (MNQ = $2/point), never a hardcoded dollar figure.
- Bar-by-bar walk from the bar after entry; if a single bar's range touches
  both TP and SL, the **stop is assumed to hit first** (pessimistic,
  `exit_reason = 'stop_ambiguous'`).
- **Flat-at-session-end fallback** (`eod_exit_time`, ET): if neither TP nor SL
  is hit by that clock time, exit at that bar's close (`exit_reason = 'eod'`).

## Parameters

| Param | Meaning | Default |
|---|---|---|
| `trigger_type` | `'net_delta'` \| `'zscore'` | `'zscore'` |
| `delta_threshold` | net_delta mode threshold | 500.0 |
| `zscore_threshold` | zscore mode threshold | 2.0 |
| `zscore_length` | rolling window (coverage-gated prior bars) | 20 |
| `min_coverage_ratio` | tick_volume / bar_volume gate | 0.25 |
| `session_start` / `session_end` | RTH window defining "prior session" | 09:30 / 16:00 ET |
| `tp_points` / `sl_points` | fixed exit distances, points | 40.0 / 20.0 |
| `eod_exit_time` | flat-by fallback, ET | 16:00 |
| `direction` | `'Long'` \| `'Short'` \| `'Both'` | `'Both'` |
| `symbol` | tick_data symbol key | `'MNQ'` (continuous); `'MNQ_H26'`/`'MNQ_M26'` also exposed for per-contract testing |
| `qty` | fixed contracts | 1 |

## Assumptions made (not specified in the brief, resolved here)

1. **"Prior session" = the most recently fully-closed RTH session as of the
   signal bar**, not "yesterday's session" by calendar date — this matters
   for the first session of a new week/after a holiday and for evening bars,
   where a naive calendar-date lookup would either miss a session or grab a
   stale one. Implemented via an explicit per-session-day `(start, end)`
   table plus a scan for the latest session whose `end < signal_ts`.
2. **The candle-body condition (`close < open` for longs) is on the SIGNAL
   bar itself**, not some other bar — the video's framing ("the candle is
   negative but delta is positive") only makes sense read as one bar's own
   OHLC vs. that same bar's own delta.
3. **Coverage gate excludes bars from the z-score baseline entirely** (NaN,
   not zero) per the explicit instruction that a thin hour must not be able
   to shrink the rolling stdev and manufacture a false outlier later.
4. **Tick-delta bucket alignment is +1h (open-label → close-label)**, derived
   from first principles of the two labelling conventions — not verified
   against a second source, since NT bar-close labelling is the platform-wide
   convention already used elsewhere (see Bar-labelling convention above).
5. **EOD fallback anchors to the entry bar's own calendar day** at
   `eod_exit_time`; if entry happens after that clock time the walk simply
   exits at the first bar whose timestamp is `>= entry_ts + 1h` as a
   degenerate guard (extremely rare given RTH-session-relative signals, but
   prevents an infinite/unbounded hold).
6. Trades are **one at a time** (no pyramiding, no simultaneous long+short) —
   not specified either way in the brief; matches every other strategy in
   this platform.
