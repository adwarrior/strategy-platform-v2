"""Aurora bar-type sweep: 1-minute baseline vs other time frames + tick bars.

Design (2026-07-11, user-approved): run the LIVE forward-test config
(EntryMinTouches=1, TradeBalanced=OFF, FlipToMarket=OFF) unchanged on each
bar type — exactly what would happen if the chart's bar type were switched
in NinjaTrader. Params are NOT rescaled per bar type; the engine's own
lookback day-cap applies to time frames and tick bars use the raw bar count
(mirrors C# EffectiveLookback). Nothing under 1000 ticks per bar: smaller
tick bars close so often the trade count explodes (user call).

Same harness as sweep_aurora.py: per-config JSONL persistence + resume on
re-run (power-cut safe), spans run separately per contract, per-span +
combined summary table at the end.

Usage: python scripts/sweep_bartype.py
Env:   AURORA_TICK_TABLE (default tick_data_full), SWEEP_WORKERS (default 2),
       SWEEP_SPANS="name:symbol:start:end,...", SWEEP_OUT=<jsonl path>
"""
import os
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategy_platform.strategies.aurora.strategy import Aurora           # noqa: E402
from strategy_platform.strategies.aurora.tick_loader import load_raw_ticks  # noqa: E402

TABLE = os.environ.get("AURORA_TICK_TABLE", "tick_data_full")

SPANS = [
    ("May", "MNQ_M26", "2026-05-01", "2026-05-29"),
    ("Jul", "MNQ_U26", "2026-07-01", "2026-07-08"),
]
if os.environ.get("SWEEP_SPANS"):
    SPANS = [tuple(x.split(":")) for x in os.environ["SWEEP_SPANS"].split(",")]

# The live forward-test config (NT chart 2026-07-08).
BASE = {"entry_min_touches": 1, "trade_bal": False, "flip_to_market": False}

VARIATIONS = []  # (group, label, overrides-on-top-of-BASE)
def add(group, label, ov):
    VARIATIONS.append((group, label, ov))

add("time", "1min (live)", {"bar_spec": "1min"})
for v in [2, 3, 5]:
    add("time", f"{v}min", {"bar_spec": f"{v}min"})
for v in [1000, 1597, 2500]:
    add("tick", f"{v}t", {"bar_spec": f"{v}t"})


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
    import json
    from multiprocessing import Pool
    tasks = [(label, ov) for _, label, ov in VARIATIONS]
    nproc = min(int(os.environ.get("SWEEP_WORKERS", "2")), len(tasks))
    # Survives power loss: each finished config is appended immediately, and
    # already-present labels are skipped so a re-run resumes where it died.
    out_path = Path(os.environ.get(
        "SWEEP_OUT", Path(__file__).resolve().parents[1] / "results" / "sweep_bartype.jsonl"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            rec = json.loads(line)
            results[rec["label"]] = rec["result"]
        tasks = [t for t in tasks if t[0] not in results]
        print(f"Resuming: {len(results)} configs already in {out_path}", flush=True)
    print(f"Bar-type sweep: {len(tasks)} configs on {[s[:2] for s in SPANS]} "
          f"({TABLE}) with {nproc} workers, base={BASE}", flush=True)
    if tasks:
        with Pool(nproc, initializer=_init_worker, initargs=(SPANS, TABLE)) as pool:
            for label, r in pool.imap_unordered(_run_variation, tasks):
                results[label] = r
                with open(out_path, "a") as f:
                    f.write(json.dumps({"label": label, "result": r}) + "\n")
                print(f"[{len(results)}/{len(VARIATIONS)}] {label}: "
                      f"ALL n={r['ALL']['n']} PF={r['ALL']['pf']:.2f} "
                      f"net=${r['ALL']['net']:+,.0f}", flush=True)

    print("\n===== BAR-TYPE SWEEP RESULTS (live 07-08 config on every bar type) =====")
    cur_group = None
    for group, label, ov in VARIATIONS:
        if group != cur_group:
            print(f"\n--- {group} ---")
            cur_group = group
        r = results[label]
        star = "  <<" if label == "1min (live)" else ""
        cells = "  ".join(
            f"{nm}: n={s['n']:3d} win={s['win']*100:4.1f}% PF={s['pf']:5.2f} "
            f"net=${s['net']:+8,.0f}"
            for nm, s in ((n, r[n]) for n in [s[0] for s in SPANS] + ["ALL"]))
        print(f"  {label:16s} {cells}{star}", flush=True)


if __name__ == "__main__":
    main()
