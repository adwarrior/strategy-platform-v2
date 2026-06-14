# Multi-timeframe backtesting

The dashboard's **Minutes per bar** selector lets any time-based strategy run on
1, 2, 3, 4, 5, 15, 30, 60, or 240-minute bars. The selected size is the
strategy's **primary timeframe** — the bars it actually operates on.

## How it works

`bar_type` still selects the *base data source*; `timeframe_mins` is the
*resample target* applied after load, before the strategy sees the data:

| Selected | bar_type | Base table        | Resample          |
|----------|----------|-------------------|-------------------|
| 1–4M     | `1m`     | `historical_data_1m` (MES/MNQ/MGC) | 1M → N (no-op at 1M) |
| 5M       | `time`   | `historical_data` (5M, all symbols)| none (native)     |
| 15/30/60/240M | `time` | `historical_data` (5M)         | 5M → N            |
| tick     | `tick`   | `tick_data`       | n/a (tick-count based) |

Resampling uses `loader.resample_ohlcv(df, minutes)` with the platform
convention (`label='right', closed='right'`). It is a no-op when the target
equals the native base size, so the existing 5M/1M behaviour is unchanged.

Plumbed through: dashboard → `--timeframe-mins` → `run_pipeline` /
`run_walk_forward` → resample after the base load, before `is_oos_split` and
`prepare_data`.

## Strategy compatibility

Strategies receive an OHLCV DataFrame at the selected timeframe. Three classes:

- **Single-timeframe** (most strategies): operate only on the bars handed to
  them. Work at any timeframe. Bar-count indicator periods (ATR length,
  lookback windows) scale with the timeframe — re-optimise per timeframe.
- **Multi-timeframe** (`wicktest5m`): build a higher-TF confirmation stack
  internally. Refactored to be **base-relative** — `_build_tf_dict` /
  `_compute_ftfc` detect the incoming bar size and build the stack relative to
  it (e.g. a 15M base builds 30/60/240/D above). At a 5M base the output is
  byte-identical to the original implementation.
- **Floor-resamplers** (`nybreakout`, `orb30_monti`, `orb_fade_60`, `cct`):
  call `_ensure_5m` / `_ensure_1m`, which only resample *up* when handed data
  finer than their floor. At a coarser selected timeframe they pass it through
  and run on it.

`waejurikpro` has no internal timeframe stack — it is already
timeframe-agnostic.

## Caveats

- A strategy's *behaviour* changes across timeframes (bar-count periods mean
  different clock spans). Parameters optimised on 5M should be re-optimised for
  other timeframes.
- 1–4M timeframes only have meaningful data for MES/MNQ/MGC (1-minute DB). The
  symbol-coverage warning in the dashboard still applies.
- Clock-anchored strategies (specific entry/exit times) remain correct because
  they read the DatetimeIndex, but exact-timestamp matches can fall between
  coarse bars — verify time-of-day logic uses windows, not exact stamps.
