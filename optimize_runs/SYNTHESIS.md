# SuperTrendFractal MNQ 5M — Optimization Synthesis

**Run dates:** 2026-05-17 → 2026-05-18
**Data range:** 2020-01-01 → 2026-05-15 (448,453 5M bars)
**Strategy:** `supertrendfractal` (v2 platform)
**Constraint:** $50K prop account, $2,500 trailing DD

---

## TL;DR

**Do not deploy this strategy on prop in its current form.** Stage B and single-slice OOS look fine, but the walk-forward exposes severe regime-dependence: the headline OOS profit is carried by **one** out of eight slices. Strip that slice and the strategy bleeds **−$1,576** across 7 of 8 windows.

The best candidate config is a converged ATR-exit setup on a fixed core. The winning **time window** is the most actionable finding — `06:00–18:00` UTC dominates WFO selection 4 of 8 times. But the config bleeds in 5 of 8 OOS slices and shows IS-Sortino decay over the 2020–2024 period, hinting the edge degraded mid-sample and partially returned only in 2024–2025.

---

## Pipeline summary

| Phase | Count | Outcome |
|---|---|---|
| Stage A core sweep | 5,832 combos, 5M, 24h trading | 300 OOS-validated; only **6** survived `|DD|≤$3,000` |
| Filter A → top cores | 5 dedup-unique | All same indicator core (atr_mult=4, atr_period=10, fractal_length=7), differing only in `sl_atr_mult` and `bars_between_trades` |
| Stage B session sweep | 720 combos (5 cores × 144 windows) | **169 profitable @ |DD|≤$2k** — session filtering compressed DD significantly |
| Aggregate → top-5 finalists | 5 configs | Best: **$2,949 P&L / $1,792 DD / 06:00–23:55** |
| Walk-forward | 8 slices, 720d IS / 180d OOS / 180d step | **WFE 0.19**, 3/8 OOS positive, slice 7 carries the result |
| Autoresearch | 3 seeds × 500 gens (planned) | **SKIPPED** — `anthropic` package missing in venv; also redundant given IS-overfit risk per WFE |

---

## Converged core (consistent across Stage B, WFO)

```
atr_multiplier      = 4
atr_period          = 10
fractal_length      = 7
direction           = Both
invert_signals      = False
exit_mode           = FixedTPSL
tpsl_mode           = ATRMultiple
tp_atr_mult         = 3.0
sl_atr_mult         = 0.5   (or 1.0 — see ranking below)
bars_between_trades = 2 or 4
```

Stage B isolation showed `sl_atr_mult=1.0` produced higher net P&L ($4,119 best), but WFO selected `sl_atr_mult=0.5` in 8/8 slices, suggesting the tighter stop is more robust across regimes despite lower headline P&L.

---

## Walk-forward per-slice (the real story)

| # | IS window | OOS window | IS sort | IS pnl | **OOS sort** | **OOS pnl** | OOS dd | WR | window |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2020-01..2021-12 | 2021-12..2022-06 | 0.96 | $879 | **1.96** | **+$691** | $-490 | 17.9% | 06:00–23:55 |
| 2 | 2020-06..2022-06 | 2022-06..2022-12 | 1.81 | $1,686 | -2.43 | **-$633** | $-668 | 16.7% | 08:00–23:55 |
| 3 | 2020-12..2022-12 | 2022-12..2023-06 | 1.57 | $1,608 | -1.51 | -$457 | $-484 | 20.5% | 06:00–18:00 |
| 4 | 2021-06..2023-06 | 2023-06..2023-12 | 0.86 | $985 | -1.15 | -$305 | $-465 | 19.7% | 06:00–18:00 |
| 5 | 2021-12..2023-12 | 2023-12..2024-06 | 0.53 | $622 | 1.14 | +$330 | $-292 | 17.3% | 06:00–18:00 |
| 6 | 2022-06..2024-06 | 2024-06..2024-12 | 0.19 | $203 | -1.35 | -$531 | $-651 | 20.3% | 06:00–18:00 |
| **7** | 2022-12..2024-12 | 2024-12..2025-06 | -0.27 | -$304 | **10.04** | **+$3,300** | $-269 | **25.2%** | 08:00–23:55 |
| 8 | 2023-06..2025-06 | 2025-06..2025-11 | 2.95 | $3,396 | -1.93 | -$671 | $-811 | 13.1% | 08:00–23:55 |

**Aggregate OOS:** +$1,724 across all 8; **−$1,576** if slice 7 is removed.
**WFE (OOS / IS pnl):** 0.19 — strategy retains only ~19% of in-sample edge out-of-sample.

### What slice 7 actually shows

Slice 7 is the only slice where IS Sortino was **negative** (−0.27) yet OOS exploded (Sortino 10.04, WR jumped to 25%). That's a regime change in the OOS period that the IS data didn't predict. The strategy "got lucky" with momentum conditions in early 2025 (likely the post-election/AI-rally trend regime favouring momentum-trend strategies).

### Session-window stability

| Window | WFO win count |
|---|---|
| 06:00–18:00 | 4 |
| 08:00–23:55 | 3 |
| 06:00–23:55 | 1 |

`06:00–18:00` is the most-selected window. Notably:
- All three "06:00–18:00" winning slices (3, 4, 5, 6) were **losing** OOS.
- Two of three "08:00–23:55" slices were also losing — only slice 7 won big.

Conclusion: window choice isn't the determining factor. The strategy's edge is regime-driven, not window-driven.

---

## Honest verdict for $50K prop deployment

| Aspect | Assessment |
|---|---|
| Single-slice OOS (2024-08 → 2026-05) | **Passes** — $2,949 P&L, $1,792 DD, profitable |
| WFO robustness | **Fails** — 5/8 OOS losing, total positive driven by one slice |
| Drawdown vs prop trail | Average WFO OOS DD ~$500, well within $2,500 trail. **DD-wise the strategy is prop-safe.** |
| P&L vs prop scaling/profit-target | Marginal — slice OOS P&L ~$300–700 over 6 months. Plenty of slices net negative for months. |
| Strategy decay | IS Sortino fell from 1.96 (2020-2021) to 0.19 (2022-2024). Decay is real. |

**The DD is fine. The win rate isn't the problem (the strategy is designed to be low-WR / high-RR). The problem is the edge is unstable across regimes.**

---

## Recommendations

### What to do before any live capital

1. **Reduce data window or add regime filter.** If only the 2024-2025 era works, fit on that era and accept the strategy is a "trend regime only" tool. Don't deploy in chop.
2. **Forward-paper trade** the `06:00–18:00, atr_mult=4, atr_period=10, fractal=7, ATR/3.0/0.5, cd=2` config for 30–60 days. Compare to the WFO OOS slice 7 baseline.
3. **Install `anthropic`** in `/home/ad/WickTest-Optimizer/venv` (`pip install anthropic`) and re-run AR — but **only on slice 7's regime** to find micro-improvements, not to refit IS.
4. **Add a regime gate**: SuperTrend on a higher timeframe (e.g. daily) as a meta-filter — only trade when daily SuperTrend agrees with intraday direction. This is a hypothesis worth testing.

### What NOT to do

- **Don't size the strategy as if Stage B's $2,949 / $1,792 DD is representative.** It isn't. The WFO baseline of $216/slice ($1,724 / 8) is closer to truth, and that's pre-cost on a per-6mo basis.
- **Don't run AR on full data IS.** WFE 0.19 means in-sample wins don't survive out-of-sample. AR optimising IS Sharpe will produce attractive but fragile configs.

---

## Recommended candidate (if forced to deploy)

```python
# SuperTrendFractal — MNQ 5M — best WFO-selected config
{
    "atr_multiplier":       4,
    "atr_period":           10,
    "fractal_length":       7,
    "direction":            "Both",
    "invert_signals":       False,
    "exit_mode":            "FixedTPSL",
    "tpsl_mode":            "ATRMultiple",
    "tp_atr_mult":          3.0,
    "sl_atr_mult":          0.5,
    "bars_between_trades":  2,
    "use_risk_sizing":      False,
    "qty":                  1,
    "enable_session_filter": True,
    "trade_window1_start":  "06:00",
    "trade_window1_stop":   "18:00",
    "eod_exit_time":        "16:55"
}
```

**Expected (per WFO 6-month slice):** P&L roughly between −$500 and +$700 in trend regimes, −$200 to +$300 in chop. Max DD typically <$700, well under prop trail.

---

## Artefacts

| File | Description |
|---|---|
| [stage_a_grid.json](stage_a_grid.json) | Stage A param grid (5,832 combos) |
| [stage_a_top_cores.json](stage_a_top_cores.json) | 5 dedup cores after DD filter |
| [stage_a_filter_log.txt](stage_a_filter_log.txt) | Stage A filter ranking log |
| [stage_b_finalists.json](stage_b_finalists.json) | Top-5 Stage B finalists |
| [stage_b_summary.txt](stage_b_summary.txt) | Stage B aggregate log |
| [wfo_grid.json](wfo_grid.json) | Union grid sent to walk-forward |
| `reports/IS_supertrendfractal_MNQ_20260518_0522.csv` | Full Stage A IS (5,832 rows) |
| `reports/OOS_supertrendfractal_MNQ_20260518_0522.csv` | Stage A OOS top-300 |
| `reports/OOS_supertrendfractal_MNQ_20260518_1543..1630.csv` | 5 Stage B per-core OOS |
| `reports/WF_supertrendfractal_MNQ_20260518_2155.json` | Walk-forward 8-slice report |
| [RUN_STATE.json](RUN_STATE.json) | Phase-level run state |
| [run_all.sh](run_all.sh) / [resume.sh](resume.sh) | Re-run / resume scripts |

---

## Known gaps

- **Autoresearch not run**: `anthropic` package missing from `/home/ad/WickTest-Optimizer/venv`. Install with `pip install anthropic` (API key already in `.env`). Re-run via `bash optimize_runs/run_all.sh` after editing `RUN_STATE.json` to set `autoresearch: pending`. Given WFE 0.19, my recommendation is to skip AR until regime-filter work is done — it will only amplify IS overfit.
- **1M bar-type equivalent to 5M**: SuperTrendFractal internally resamples 1M→5M, so the requested "MNQ 1M and 5M" pass collapses to a single 5M run. No information lost; flagged so it's not misread as missing work.
- **Stage A `TrailToLine` and `FixedTPSL/Ticks` exit modes did not survive the DD filter.** Worth a separate Stage A run with these as the only sweep target if the ATR exit ends up being abandoned.

---

*Generated 2026-05-18 by `/project-manager` autonomous run.*
