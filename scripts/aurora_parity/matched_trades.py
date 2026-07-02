"""Aurora parity — matched-trade PnL: NT vs the Python port.

For every NT trade (Trades_May.csv), find the port trade with the same direction
whose entry time is within 90s. On the trades BOTH engines take (matched), compare
PnL and win rate. This isolates the FILL/EXIT model from trade SELECTION: if the
matched trades disagree in PnL, the gap is fills, not which trades get taken.

May 1-15 result (2026-07-02): 214 matched, NT +$1953/57% vs PORT -$1281/49%,
88% same entry -> the divergence is fill/exit modeling. See memory
project_aurora_python_parity (STRATEGIC FORK / option B).

Usage: AURORA_TICK_TABLE=tick_data_full python scripts/aurora_parity/matched_trades.py
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strategy_platform.strategies.aurora.strategy import Aurora           # noqa: E402
from strategy_platform.strategies.aurora.tick_loader import load_raw_ticks  # noqa: E402

os.environ.setdefault("AURORA_TICK_TABLE", "tick_data_full")
NT_CSV = os.environ.get("NT_TRADES_CSV", "/home/ad/Scripts/Results/Aurora/Trades_May.csv")
SYMBOL = os.environ.get("SWEEP_SYMBOL", "MNQ_M26")
START = os.environ.get("SWEEP_START", "2026-05-01")
END = os.environ.get("SWEEP_END", "2026-05-29")
TABLE = os.environ.get("AURORA_TICK_TABLE", "tick_data_full")


def _nt_pnl(series: pd.Series) -> pd.Series:
    pnl = series.astype(str).str.replace(r"[\$,()]", "", regex=True).astype(float)
    pnl[series.astype(str).str.contains(r"\(")] *= -1
    return pnl


def main():
    nt = pd.read_csv(NT_CSV)
    nt["Entry time"] = pd.to_datetime(nt["Entry time"], format="%d/%m/%Y %H:%M:%S")
    nt["ntpnl"] = _nt_pnl(nt["Profit"])

    mt_py, mt_nt, missed = [], [], []
    for day in pd.bdate_range(START, END):
        d = day.strftime("%Y-%m-%d")
        ticks = load_raw_ticks(SYMBOL, d, d, table=TABLE)
        if ticks.empty:
            continue
        py = Aurora().run_backtest(ticks, {})["trades"]
        py = py[py["entry_time"].astype(str).str.startswith(d)].reset_index(drop=True)
        if len(py):
            py["entry_time"] = pd.to_datetime(py["entry_time"])
        ntd = nt[nt["Entry time"].dt.strftime("%Y-%m-%d") == d].reset_index(drop=True)
        used = set()
        for _, n in ntd.iterrows():
            cand = py[(py["direction"].str.lower() == n["Market pos."].lower())
                      & (~py.index.isin(used))].copy() if len(py) else py
            if len(cand) == 0:
                missed.append(n["ntpnl"])
                continue
            cand["dt"] = (cand["entry_time"] - n["Entry time"]).abs()
            j = cand["dt"].idxmin()
            if cand.loc[j, "dt"].total_seconds() <= 90:
                used.add(j)
                mt_py.append(py.loc[j, "pnl"])
                mt_nt.append(n["ntpnl"])
            else:
                missed.append(n["ntpnl"])

    mt_py, mt_nt, missed = np.array(mt_py), np.array(mt_nt), np.array(missed)
    print(f"MATCHED n={len(mt_py)}", flush=True)
    print(f"  NT   pnl ${mt_nt.sum():+.0f}  win={(mt_nt > 0).mean() * 100:.0f}%", flush=True)
    print(f"  PORT pnl ${mt_py.sum():+.0f}  win={(mt_py > 0).mean() * 100:.0f}%", flush=True)
    print(f"  same win/loss sign: {(np.sign(mt_py) == np.sign(mt_nt)).mean() * 100:.0f}%", flush=True)
    print(f"MISSED-by-port NT trades n={len(missed)}  NT pnl ${missed.sum():+.0f}  "
          f"win={(missed > 0).mean() * 100:.0f}%", flush=True)


if __name__ == "__main__":
    main()
