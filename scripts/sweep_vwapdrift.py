"""VWAPDrift OFAT sweep — can any single lever lift a break-even baseline?

Baseline (the video's exact config) measured 2026-08-11 on MNQ 2020->2026-08:
4,646 trades, WR 62.6%, PF 1.05, ~$2.5/trade. The claims reproduce; the edge
doesn't pay. This sweep asks ONE question per line: does changing exactly one
knob lift PF materially, in ALL THREE periods?

OFAT, not a Cartesian grid, on purpose. A full grid over these params is tens
of thousands of combos against a PF-1.05 baseline — that finds noise, not edge.
Any winner here still has to survive a focused grid afterwards.

3-period split (see memory feedback_three_period_validation):
  IS      2020-01-01 -> 2022-12-31   tune
  OOS     2023-01-01 -> 2024-12-31   validate (contains the losing 2024)
  CONF    2025-01-01 -> 2026-08-07   confirm
A lever is only interesting if PF > baseline in all three, not just on ALL.

Data is loaded ONCE in the parent and shared with forked workers (copy-on-write)
— 2.3M 1m bars, so re-loading per worker would blow out RAM and time.

Usage: python scripts/sweep_vwapdrift.py
Env:   SWEEP_WORKERS (default 4), SWEEP_OUT (default results/sweep_vwapdrift.jsonl)
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategy_platform.data.loader import load_1m                       # noqa: E402
from strategy_platform.strategies.vwapdrift.strategy import VwapDrift   # noqa: E402

SYMBOL = os.environ.get("SWEEP_SYMBOL", "MNQ")

# (name, start, end) — scored separately, never pooled for the verdict.
PERIODS = [
    ("IS",   "2020-01-01", "2022-12-31"),
    ("OOS",  "2023-01-01", "2024-12-31"),
    ("CONF", "2025-01-01", "2026-08-07"),
]

# The video's exact config. Every variation is {**BASE, **ov}, so each line
# answers "what if the published strategy changed only this one thing".
BASE = dict(VwapDrift.default_params)

VARIATIONS = []  # (group, label, overrides)


def add(group, label, ov):
    VARIATIONS.append((group, label, ov))


add("baseline", "video-config", {})

# --- Session window. The user's manual edge lives 09:30-11:30 ET and the
# Aurora work showed the session window is the single biggest lever; the
# video trades 10:30-15:30 flat. warmup_minutes shifts the open edge,
# entry_cutoff the close edge.
for v in [60, 90, 120, 180]:
    add("warmup", f"warmup={v}m", {"warmup_minutes": v})

for v in ["11:30", "12:00", "13:00", "14:00", "15:30"]:
    add("cutoff", f"cutoff={v}", {"entry_cutoff": v})

# The morning-only book: open at the video's 10:30, close early.
for v in ["11:30", "12:00", "13:00"]:
    add("morning_only", f"10:30-{v}", {"warmup_minutes": 60, "entry_cutoff": v})

# --- Drift strength. 0.1% was the video's optimised value; a stronger
# momentum gate should trade less and (if the edge is real) trade better.
for v in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    add("momentum", f"1h>={v}%", {"hour_return_pct": v})

for v in [10, 15, 30, 60]:
    add("drift_lookback", f"drift={v}m", {"slope_lookback_min": v})

# --- Brackets. The inverted RR is the whole reason expectancy is thin:
# avg win $84 vs avg loss $134. Test symmetric and positive-RR variants.
for sl, tpl, tps in [
    (80, 40, 50),    # video
    (80, 80, 80),    # symmetric
    (60, 60, 60),
    (40, 40, 40),
    (80, 120, 120),  # positive RR
    (60, 120, 120),
    (40, 80, 80),
    (100, 50, 50),   # even more inverted
]:
    add("brackets", f"sl{sl}/tp{tpl}-{tps}",
        {"stop_points": float(sl), "target_long_points": float(tpl),
         "target_short_points": float(tps)})

# --- Guardrails. Fewer trades/day concentrates on the earliest (best?) signal;
# first_pullback_only is the spec's documented ambiguity in the transcript.
for v in [1, 2, 3, 4, 6]:
    add("max_trades", f"maxtrades={v}", {"max_trades_day": v})

add("first_pullback", "first-pullback-only", {"first_pullback_only": True})
add("losses", "consecutive-losses", {"losses_consecutive": True})

for v in [1, 2, 3]:
    add("max_losses", f"maxlosses={v}", {"max_losses_day": v})

# --- Direction. Long/short split was near-even overall, but 2024's loss
# may be one-sided.
for v in ["Long Only", "Short Only"]:
    add("direction", v, {"direction": v})

# --- Distance-to-VWAP (added 2026-08-11 at user's reading of the video).
# The transcript's MECHANICS disclaim distance ("doesn't matter how close it is
# to the VWOP", line 346) but its NARRATIVE says "pullback towards the VWAP".
# The source genuinely conflicts, so measure it rather than argue about it.
# ATR-relative is the honest test: MNQ ran ~7,000 in 2020 and ~25,000 in 2026,
# so a fixed point cap is a far tighter filter early in the sample than late.
for v in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
    add("vwap_dist_atr", f"dist<={v}ATR",
        {"max_vwap_distance": v, "vwap_distance_mode": "ATR"})

for v in [5, 10, 15, 25, 50]:
    add("vwap_dist_pts", f"dist<={v}pts",
        {"max_vwap_distance": float(v), "vwap_distance_mode": "Points"})

# The distance filter combined with the only lever that was positive in all
# three periods (slower drift), in case they're complementary.
for v in [0.5, 1.0, 2.0]:
    add("dist_x_drift", f"dist<={v}ATR+drift60",
        {"max_vwap_distance": v, "vwap_distance_mode": "ATR",
         "slope_lookback_min": 60})


def summarise(trades: pd.DataFrame) -> dict:
    """Per-period stats. PF and expectancy per trade are the decision metrics —
    net alone rewards a variation for simply trading more."""
    if trades is None or len(trades) == 0:
        return dict(n=0, win=0.0, pf=0.0, net=0.0, exp=0.0, dd=0.0)
    p = trades["pnl"].to_numpy(float)
    gp = p[p > 0].sum()
    gl = p[p < 0].sum()
    eq = p.cumsum()
    dd = float((eq - np.maximum.accumulate(eq)).min())
    return dict(
        n=int(len(p)),
        win=float((p > 0).mean()),
        pf=float(gp / abs(gl)) if gl != 0 else float("inf"),
        net=float(p.sum()),
        exp=float(p.mean()),
        dd=dd,
    )


_SLICES = None  # {period_name: df} — inherited by forked workers via COW


def _run_variation(args):
    label, ov = args
    full = {**BASE, **ov}
    strat = VwapDrift()
    out = {}
    for name, _, _ in PERIODS:
        df = _SLICES.get(name)
        if df is None or len(df) == 0:
            out[name] = summarise(None)
            continue
        res = strat.run_backtest(df, full)
        out[name] = summarise(res.get("trades"))
    return label, out


def main():
    global _SLICES
    from multiprocessing import Pool

    out_path = Path(os.environ.get(
        "SWEEP_OUT",
        Path(__file__).resolve().parents[1] / "results" / "sweep_vwapdrift.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            results[rec["label"]] = rec["result"]
        print(f"Resuming: {len(results)} configs already in {out_path}", flush=True)

    tasks = [(label, ov) for _, label, ov in VARIATIONS if label not in results]

    if tasks:
        span_start = min(p[1] for p in PERIODS)
        span_end = max(p[2] for p in PERIODS)
        print(f"Loading {SYMBOL} 1m {span_start} -> {span_end} ...", flush=True)
        # db_timezone='ET' on the strategy => loader must shift CT->ET.
        df = load_1m(SYMBOL, start=span_start, end=span_end, to_et=True)
        _SLICES = {name: df[s:e] for name, s, e in PERIODS}
        for name, s, e in PERIODS:
            print(f"  {name:5s} {s} -> {e}: {len(_SLICES[name]):,} bars", flush=True)

        nproc = min(int(os.environ.get("SWEEP_WORKERS", "4")), len(tasks))
        print(f"Sweeping {len(tasks)} configs with {nproc} workers", flush=True)
        with Pool(nproc) as pool:
            for label, r in pool.imap_unordered(_run_variation, tasks):
                results[label] = r
                with open(out_path, "a") as f:
                    f.write(json.dumps({"label": label, "result": r}) + "\n")
                print(f"[{len(results)}/{len(VARIATIONS)}] {label:22s} "
                      + "  ".join(f"{n}:PF={r[n]['pf']:.2f}" for n, _, _ in PERIODS),
                      flush=True)

    _report(results)


def _report(results):
    base = results.get("video-config")
    base_pf = {n: base[n]["pf"] for n, _, _ in PERIODS} if base else {}

    print("\n" + "=" * 100)
    print("VWAPDrift OFAT sweep — PF per period (baseline = the video's config)")
    print("=" * 100)

    cur_group = None
    for group, label, _ov in VARIATIONS:
        if label not in results:
            continue
        if group != cur_group:
            print(f"\n--- {group} ---")
            cur_group = group
        r = results[label]
        cells = "  ".join(
            f"{n}: n={r[n]['n']:4d} PF={r[n]['pf']:5.2f} ${r[n]['exp']:+6.1f}/t"
            for n, _, _ in PERIODS)
        star = "  <<baseline" if label == "video-config" else ""
        print(f"  {label:22s} {cells}{star}")

    # A lever only counts if it beats the baseline in EVERY period.
    print("\n" + "=" * 100)
    print("SURVIVORS — PF above baseline in all three periods, and PF>1.2 everywhere")
    print("=" * 100)
    survivors = []
    for group, label, ov in VARIATIONS:
        if label == "video-config" or label not in results:
            continue
        r = results[label]
        if all(r[n]["n"] >= 30 for n, _, _ in PERIODS) and \
           all(r[n]["pf"] > base_pf.get(n, 0) for n, _, _ in PERIODS) and \
           all(r[n]["pf"] > 1.2 for n, _, _ in PERIODS):
            survivors.append((group, label, r, ov))

    if not survivors:
        print("  NONE. No single lever lifts PF above 1.2 in all three periods.")
        print("  => the baseline is break-even because the signal is thin, not")
        print("     because one parameter is mis-set. Shelve, or rethink the entry.")
    else:
        for group, label, r, ov in survivors:
            print(f"  [{group}] {label:22s} " + "  ".join(
                f"{n}: PF={r[n]['pf']:.2f} ${r[n]['exp']:+.1f}/t net=${r[n]['net']:+,.0f}"
                for n, _, _ in PERIODS))
            print(f"      params: {ov}")
        print("\n  NEXT: these are OFAT winners only. Run a focused grid around them")
        print("  before believing any of it (single-lever survivors still overfit).")


if __name__ == "__main__":
    main()
