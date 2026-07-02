"""Aurora param robustness sweep (OFAT) on full-volume May ticks.

Runs the design-doc one-factor-at-a-time grid: for each param, hold the others at
default and sweep that param's values (default always included as the anchor).
Ranks each value by PF / win% / net$ so we can pick a robust region, then apply
the winner to the NT strategy.

EFFICIENCY: load each trading day's ticks ONCE, then run every config against the
in-memory day (configs share the tick load). ~21 day-loads instead of 29x21.
Memory stays bounded (one day at a time) — full-volume table would OOM otherwise.

Usage: python scripts/sweep_aurora.py            # May, full-volume
Env:   AURORA_TICK_TABLE=tick_data_full (default here), SWEEP_SYMBOL, SWEEP_START/END
"""
import os
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategy_platform.strategies.aurora.strategy import Aurora           # noqa: E402
from strategy_platform.strategies.aurora.tick_loader import load_raw_ticks  # noqa: E402

TABLE = os.environ.get("AURORA_TICK_TABLE", "tick_data_full")
SYMBOL = os.environ.get("SWEEP_SYMBOL", "MNQ_M26")
START = os.environ.get("SWEEP_START", "2026-05-01")
END = os.environ.get("SWEEP_END", "2026-05-29")

# OFAT grid (design doc 2026-06-30). Each entry: param -> list of override dicts,
# each dict a full set of param overrides for that variation. Paired TP/SL are
# single variations. Default is always present as the anchor.
DEFAULT = dict(Aurora.default_params)

VARIATIONS = []  # (group, label, overrides)
def add(group, label, ov):
    VARIATIONS.append((group, label, ov))

# baseline anchor
add("baseline", "default", {})

for v in [0, 1, 2, 3, 5, 8]:
    add("entry_offset_ticks", f"off={v}", {"entry_offset_ticks": v})

for tp, sl in [(10, 10), (15, 15), (20, 20), (25, 25), (30, 30), (20, 15), (15, 20)]:
    add("tp_sl_early", f"early {tp}/{sl}", {"tp_early_pts": float(tp), "sl_early_pts": float(sl)})

for tp, sl in [(5, 5), (8, 8), (10, 10), (12, 12), (15, 15)]:
    add("tp_sl_late", f"late {tp}/{sl}", {"tp_late_pts": float(tp), "sl_late_pts": float(sl)})

for v in [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]:
    add("rearm_atr", f"rearm={v}", {"rearm_atr": v})

for v in ["off", "10:00", "10:30", "11:00", "11:30"]:
    add("tighten_time", f"tighten={v}", {"tighten_time": v})


def summarise(trades):
    if not trades:
        return dict(n=0, win=0.0, pf=0.0, net=0.0)
    import numpy as np
    p = np.array([t["pnl"] for t in trades], float)
    gp = p[p > 0].sum(); gl = p[p < 0].sum()
    return dict(n=len(p), win=float((p > 0).mean()),
                pf=float(gp / abs(gl)) if gl != 0 else float("inf"),
                net=float(p.sum()))


_TICKS = None  # per-worker cached month ticks (loaded once per process)


def _init_worker(sym, start, end, table):
    global _TICKS
    if table == "tick_data_full":
        # full-volume: concat per-day to bound peak memory during load
        frames = []
        for day in pd.bdate_range(start, end):
            d = day.strftime("%Y-%m-%d")
            t = load_raw_ticks(sym, d, d, table=table)
            if len(t):
                frames.append(t)
        _TICKS = pd.concat(frames) if frames else None
    else:
        _TICKS = load_raw_ticks(sym, start, end, table=table)


def _run_variation(args):
    label, ov = args
    res = Aurora().run_backtest(_TICKS, ov)
    return label, summarise([{"pnl": p} for p in res["trades"]["pnl"].tolist()])


def main():
    from multiprocessing import Pool
    tasks = [(label, ov) for _, label, ov in VARIATIONS]
    nproc = min(int(os.environ.get("SWEEP_WORKERS", "2")), len(tasks))
    print(f"Sweeping {len(tasks)} configs on {SYMBOL} {START}..{END} "
          f"({TABLE}) with {nproc} workers", flush=True)
    with Pool(nproc, initializer=_init_worker, initargs=(SYMBOL, START, END, TABLE)) as pool:
        results = dict(pool.map(_run_variation, tasks))

    print("\n===== OFAT SWEEP RESULTS =====")
    cur_group = None
    for group, label, ov in VARIATIONS:
        if group != cur_group:
            print(f"\n--- {group} ---")
            cur_group = group
        s = results[label]
        star = "  <<" if label == "default" else ""
        print(f"  {label:16s} n={s['n']:4d}  win={s['win']*100:5.1f}%  "
              f"PF={s['pf']:5.2f}  net=${s['net']:+8,.0f}{star}")


if __name__ == "__main__":
    main()
