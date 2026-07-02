"""Aurora parity — NT phantom-exit audit.

Tests whether NT's Strategy-Analyzer exit prices actually occurred in the tick
stream during each trade's life. A "phantom" exit = NT booked a fill at a price
the ticks NEVER spanned between entry and exit (widened +/-90s for the NT
close-label / tick_data skew). This is the decisive test for the STRATEGIC FORK
(see memory project_aurora_python_parity):
  - high phantom rate  -> NT's backtest edge is a fill-model artifact (option B);
                          the conservative Python port is the honest number.
  - low phantom rate   -> the residual is footprint-fidelity reproduction (A).

Motivating case (2026-05-01 09:49 short): NT booked a +20pt "Profit target" exit
at 27757.75, but price hit the STOP (27797.50) and never reached the target.

Usage: AURORA_TICK_TABLE=tick_data_full python scripts/aurora_parity/phantom_exits.py
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strategy_platform.strategies.aurora.tick_loader import load_raw_ticks  # noqa: E402

os.environ.setdefault("AURORA_TICK_TABLE", "tick_data_full")
NT_CSV = os.environ.get("NT_TRADES_CSV", "/home/ad/Scripts/Results/Aurora/Trades_May.csv")
SYMBOL = os.environ.get("SWEEP_SYMBOL", "MNQ_M26")
START = os.environ.get("SWEEP_START", "2026-05-01")
END = os.environ.get("SWEEP_END", "2026-05-29")
TABLE = os.environ.get("AURORA_TICK_TABLE", "tick_data_full")
TOL = 0.13  # ~half a tick of slack when testing whether price spanned the exit


def main():
    nt = pd.read_csv(NT_CSV)
    nt["Entry time"] = pd.to_datetime(nt["Entry time"], format="%d/%m/%Y %H:%M:%S")
    nt["Exit time"] = pd.to_datetime(nt["Exit time"], format="%d/%m/%Y %H:%M:%S")

    tot = phantom = tp_tot = phantom_tp = 0
    for day in pd.bdate_range(START, END):
        d = day.strftime("%Y-%m-%d")
        ticks = load_raw_ticks(SYMBOL, d, d, table=TABLE)
        if ticks.empty:
            continue
        px = ticks["price"]
        ntd = nt[nt["Entry time"].dt.strftime("%Y-%m-%d") == d]
        for _, r in ntd.iterrows():
            lo = r["Entry time"] - pd.Timedelta(seconds=90)
            hi = r["Exit time"] + pd.Timedelta(seconds=90)
            w = px[(px.index >= lo) & (px.index <= hi)]
            if len(w) == 0:
                continue
            tot += 1
            xp = float(r["Exit price"])
            reached = (w.min() - TOL) <= xp <= (w.max() + TOL)
            if r["Exit name"] == "Profit target":
                tp_tot += 1
                if not reached:
                    phantom_tp += 1
            if not reached:
                phantom += 1

    print(f"NT trades checked: {tot}", flush=True)
    print(f"  exits whose price the ticks NEVER spanned: {phantom} "
          f"({phantom / max(tot, 1) * 100:.0f}%)", flush=True)
    print(f"  Profit-target exits: {tp_tot}, of which phantom (target never reached): "
          f"{phantom_tp} ({phantom_tp / max(tp_tot, 1) * 100:.0f}%)", flush=True)


if __name__ == "__main__":
    main()
