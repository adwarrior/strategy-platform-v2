"""Aurora param robustness sweep (OFAT) on full-volume ticks, May + July spans.

Anchored on the LIVE forward-test config (2026-07-08: EntryMinTouches=1,
TradeBalanced=OFF, FlipToMarket=OFF) — every variation is applied ON TOP of
that base, so each line answers "what if the live strategy changed only this
knob". Live-log parity (2026-07-09) validated the port against the SimAurora
fill logs (07-07 100% match), so results are trustworthy as a CONSERVATIVE
estimate: the port's known biases (extra early arms, wall-band drift on
multi-hour walls) lose coin-flips live tends to win.

Spans run separately per contract (May MNQ_M26, July MNQ_U26) — different
price levels + missing June make concatenation nonsense — and are reported
per-span plus combined.

EFFICIENCY: each worker loads every span's ticks ONCE (day-chunked to bound
peak memory), then runs its share of configs against the in-memory spans.

Usage: python scripts/sweep_aurora.py
Env:   AURORA_TICK_TABLE (default tick_data_full), SWEEP_WORKERS (default 2)
"""
import os
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategy_platform.strategies.aurora.strategy import Aurora           # noqa: E402
from strategy_platform.strategies.aurora.tick_loader import load_raw_ticks  # noqa: E402

TABLE = os.environ.get("AURORA_TICK_TABLE", "tick_data_full")

# (name, symbol, start, end) — separate contracts, never concatenated.
# Override via SWEEP_SPANS="name:symbol:start:end,name:symbol:start:end"
# (e.g. the June pass runs each roll leg as its own span). Keep total span
# size ~1 month per worker: each worker holds every span's ticks in RAM.
SPANS = [
    ("May", "MNQ_M26", "2026-05-01", "2026-05-29"),
    ("Jul", "MNQ_U26", "2026-07-01", "2026-07-08"),
]
if os.environ.get("SWEEP_SPANS"):
    SPANS = [tuple(x.split(":")) for x in os.environ["SWEEP_SPANS"].split(",")]

# The live forward-test config (NT chart 2026-07-08). Every variation is
# {**BASE, **overrides} so the anchor is what actually trades live.
BASE = {"entry_min_touches": 1, "trade_bal": False, "flip_to_market": False}

VARIATIONS = []  # (group, label, overrides-on-top-of-BASE)
def add(group, label, ov):
    VARIATIONS.append((group, label, ov))

# anchors
add("baseline", "live-config", {})
add("baseline", "old-defaults", {"entry_min_touches": 0, "trade_bal": True,
                                 "flip_to_market": True})

# The forward-test levers.
add("trade_bal", "bal=ON", {"trade_bal": True})

for v in [0, 1, 2, 3]:
    add("entry_min_touches", f"touches>={v}", {"entry_min_touches": v})

for v in [0, 5, 15, 30]:
    add("entry_min_age_bars", f"age>={v}", {"entry_min_age_bars": v})

for v in [0.0, 1.5, 2.0, 3.0]:
    add("fast_tape_atr_mult", f"fast_tape={v}", {"fast_tape_atr_mult": v})

# Original OFAT knobs (design doc 2026-06-30), now around the live base.
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


def summarise(pnls):
    import numpy as np
    if not len(pnls):
        return dict(n=0, win=0.0, pf=0.0, net=0.0)
    p = np.asarray(pnls, float)
    gp = p[p > 0].sum(); gl = p[p < 0].sum()
    return dict(n=len(p), win=float((p > 0).mean()),
                pf=float(gp / abs(gl)) if gl != 0 else float("inf"),
                net=float(p.sum()))


_SPAN_TICKS = None  # per-worker cache: {span_name: ticks} loaded once


def _init_worker(spans, table):
    global _SPAN_TICKS
    _SPAN_TICKS = {}
    for name, sym, start, end in spans:
        frames = []
        for day in pd.bdate_range(start, end):
            d = day.strftime("%Y-%m-%d")
            t = load_raw_ticks(sym, d, d, table=table)
            if len(t):
                frames.append(t)
        _SPAN_TICKS[name] = pd.concat(frames) if frames else None
        print(f"  worker loaded {name} {sym}: "
              f"{sum(len(f) for f in frames):,} ticks", flush=True)


def _run_variation(args):
    label, ov = args
    full = {**BASE, **ov}
    out = {}
    all_pnls = []
    for name, _, _, _ in SPANS:
        ticks = _SPAN_TICKS.get(name)
        if ticks is None:
            out[name] = summarise([])
            continue
        res = Aurora().run_backtest(ticks, full)
        pnls = res["trades"]["pnl"].tolist()
        out[name] = summarise(pnls)
        all_pnls.extend(pnls)
    out["ALL"] = summarise(all_pnls)
    return label, out


def main():
    from multiprocessing import Pool
    tasks = [(label, ov) for _, label, ov in VARIATIONS]
    nproc = min(int(os.environ.get("SWEEP_WORKERS", "2")), len(tasks))
    print(f"Sweeping {len(tasks)} configs on {[s[:2] for s in SPANS]} "
          f"({TABLE}) with {nproc} workers, base={BASE}", flush=True)
    results = {}
    with Pool(nproc, initializer=_init_worker, initargs=(SPANS, TABLE)) as pool:
        for label, r in pool.imap_unordered(_run_variation, tasks):
            results[label] = r
            print(f"[{len(results)}/{len(tasks)}] {label}: "
                  f"ALL n={r['ALL']['n']} PF={r['ALL']['pf']:.2f} "
                  f"net=${r['ALL']['net']:+,.0f}", flush=True)

    print("\n===== OFAT SWEEP RESULTS (base = live 07-08 config) =====")
    cur_group = None
    for group, label, ov in VARIATIONS:
        if group != cur_group:
            print(f"\n--- {group} ---")
            cur_group = group
        r = results[label]
        star = "  <<" if label == "live-config" else ""
        cells = "  ".join(
            f"{nm}: n={s['n']:3d} win={s['win']*100:4.1f}% PF={s['pf']:5.2f} "
            f"net=${s['net']:+8,.0f}"
            for nm, s in ((n, r[n]) for n in [s[0] for s in SPANS] + ["ALL"]))
        print(f"  {label:16s} {cells}{star}", flush=True)


if __name__ == "__main__":
    main()
