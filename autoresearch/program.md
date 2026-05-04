# Autoresearch — WickTest5M_FTFCv3

## Goal
Improve the Sharpe ratio of the WickTest5M strategy by iteratively proposing
and testing small parameter changes. Run continuously until stopped.

## What you are optimising
File: `strategy_platform/strategies/wicktest5m/strategy.py`
The `default_params` dict is the **current best** configuration.
The `param_grid` dict defines the **allowed range** for each parameter.

## How each generation works
1. Read the current `default_params` from strategy.py.
2. Propose ONE small change (see rules below).
3. Write the updated `default_params` back to strategy.py.
4. Run: `python autoresearch/run_experiment.py --strategy wicktest5m --symbol NQ=F`
5. Parse the JSON result from stdout.
6. If `sharpe` improved AND `trades >= 20`: keep the change (git commit).
   Otherwise: revert strategy.py to the previous version (git checkout).
7. Log the result to `autoresearch/results.tsv`.
8. Repeat.

## Rules for proposing changes
- Change **only** `default_params` values, nothing else.
- Each generation changes **exactly one parameter**.
- The new value must be inside the range defined by `param_grid`.
- Do not set `tfc_threshold = 0` unless explicitly exploring the no-filter case.
- Valid `trade_window` values are the strings listed in `param_grid`.
- Day filter bools (`monday`–`friday`) may be toggled but only one at a time.
- Do not change `skip_tf_count` and `tfc_threshold` simultaneously.

## Rejection rules (revert immediately without running)
- Proposed value is outside the `param_grid` range.
- `profit` < 10 or `profit` > 200.
- `min_bar_size_ticks` < 1.
- `bars_between_trades` < 1.

## Look-ahead bias check
If `sharpe` > 5.0 or `win_rate` > 0.90, treat as suspicious — reject and revert.
These values are unrealistic for a real intraday strategy.

## results.tsv format (append one row per generation)
```
gen\ttimestamp\tsharpe\tnet_pnl\ttrades\twin_rate\tparam_changed\told_value\tnew_value\tkept
```

## Strategy summary (for context)
- Instrument: NQ=F (Mini Nasdaq-100 futures), tick=$5, commission=$3.98 RT
- Timeframe: 5-minute bars, intraday
- Entry: wick breakout — current bar touches High[2]+offset (long) or Low[2]-offset (short)
         with three preceding bullish/bearish bars and small upper/lower wick on bar[1]
- Exit: profit target (fixed ticks) + stop at Low[2]/High[2] + EOD exit at 15:59
- Filter: FTFC — N sequential TFs (15M→30M→60M→240M→D) must all agree in direction
- Data source: MySQL at 172.23.48.1, table emini.historical_data
- IS split: 70% of date range

## Starting Sharpe
Record the Sharpe from generation 0 (before any changes) as the baseline.
Only accept changes that beat this baseline.
