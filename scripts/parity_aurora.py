"""Aurora Heatshelves — trade-by-trade parity: Python port vs the 5 NT Trades.csv runs.

Phase 1 GATE of the param-robustness sweep (design:
docs/superpowers/specs/2026-06-30-aurora-param-robustness-sweep-design.md).

For each of the 5 NT runs: load the NT trade log, run the Python Aurora on the SAME
MNQ raw bid/ask ticks from the emini `tick_data` table (DB-tick Tier-1), and match
trade-by-trade with parity_check.match_trades (tick-bar branch: nearest-in-time within
a window + entry-price tolerance). Pass = trade counts and entry/exit alignment hold.

TRAPS handled:
  - NT exports dates as dd/MM/yyyy (day-first). Parsing as %m/%d silently swaps
    month/day for days<=12 and scrambles the match. We parse dayfirst=True.
  - tick_data is UTC + ~1min skew -> tick_loader ET-normalizes; NT trade times are ET.
  - NT logs 'Long'/'Short'; Python emits 'long'/'short'. match_trades normalizes both.
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategy_platform.strategies.aurora.strategy import Aurora           # noqa: E402
from strategy_platform.strategies.aurora.tick_loader import load_raw_ticks  # noqa: E402
from scripts.parity_check import match_trades, preflight_guards           # noqa: E402

NT_DIR = Path("/home/ad/Scripts/Results/Aurora")

# (label, NT Trades.csv, DB tick symbol, start, end)  — dates inclusive.
RUNS = [
    ("Feb",       "Trades_Feb.csv",           "MNQ_H26", "2026-02-02", "2026-02-27"),
    ("Mar-front", "Trades_Mar_Frontkend.csv", "MNQ_H26", "2026-03-02", "2026-03-16"),
    ("Mar-back",  "Trades_Mar_backend.csv",   "MNQ_M26", "2026-03-17", "2026-03-31"),
    ("Apr",       "Trades_Apr.csv",           "MNQ_M26", "2026-04-01", "2026-04-30"),
    ("May",       "Trades_May.csv",           "MNQ_M26", "2026-05-01", "2026-05-29"),
]

# tick-bar match tolerances (design: DB ticks, not identical to NT's own feed)
TIME_WINDOW_S = 90     # entry within 90s (tick-vs-1min-bar skew, see memory)
PRICE_TOL = 1.0        # entry price within 1.0 pt


def load_nt_trades(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    out = pd.DataFrame({
        "entry_time": pd.to_datetime(df["Entry time"], dayfirst=True),
        "exit_time":  pd.to_datetime(df["Exit time"],  dayfirst=True),
        "direction":  df["Market pos."],
        "entry_price": df["Entry price"].astype(float),
        "exit_price":  df["Exit price"].astype(float),
    })
    return out


def run_one(label, csv, sym, start, end):
    nt = load_nt_trades(NT_DIR / csv)
    ticks = load_raw_ticks(sym, start, end)
    if ticks.empty:
        return {"label": label, "status": "DATA-BLOCKED", "detail": f"no ticks {sym} {start}..{end}"}

    res = Aurora().run_backtest(ticks, {})           # defaults — matches the NT baseline runs
    py = res["trades"]
    if py.empty:
        return {"label": label, "status": "FAIL", "detail": "Python produced 0 trades",
                "nt_n": len(nt), "py_n": 0}

    m = match_trades(nt, py[["entry_time", "exit_time", "direction",
                             "entry_price", "exit_price"]],
                     timeframe_min=None, time_window_s=TIME_WINDOW_S, price_tol=PRICE_TOL)
    matched = m["matched"]
    n_match = len(matched)
    warns = preflight_guards(matched, nt, py)
    ent_err = (matched["py_entry_price"] - matched["nt_entry_price"]).abs().mean() if n_match else float("nan")

    # Aggregate (distribution-level) stats for both sides.
    nt_agg = nt_stats(NT_DIR / csv)
    py_win = float((py["pnl"] > 0).mean()) if len(py) else 0.0
    gp = float(py.loc[py["pnl"] > 0, "pnl"].sum())
    gl = float(py.loc[py["pnl"] < 0, "pnl"].sum())
    py_pf = gp / abs(gl) if gl != 0 else float("inf")
    py_net = float(py["pnl"].sum())

    return {
        "label": label, "status": "OK", "nt_n": len(nt), "py_n": len(py),
        "matched": n_match, "nt_only": len(m["nt_only"]), "py_only": len(m["py_only"]),
        "match_rate": n_match / len(nt) if len(nt) else 0.0,
        "mean_entry_err": ent_err, "warnings": warns,
        "nt_win": nt_agg["win"], "nt_pf": nt_agg["pf"], "nt_net": nt_agg["net"],
        "py_win": py_win, "py_pf": py_pf, "py_net": py_net,
    }


def nt_stats(csv_path: Path) -> dict:
    """Win%/PF/net$ straight from an NT Trades.csv (Profit col is net $)."""
    df = pd.read_csv(csv_path)
    pnl = df["Profit"].astype(str).str.replace(r"[$,()]", "", regex=True)
    neg = df["Profit"].astype(str).str.contains(r"\(")
    pnl = pnl.astype(float)
    pnl[neg] = -pnl[neg]
    gp = pnl[pnl > 0].sum()
    gl = pnl[pnl < 0].sum()
    return {"win": float((pnl > 0).mean()), "pf": float(gp / abs(gl)) if gl != 0 else float("inf"),
            "net": float(pnl.sum())}


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    rows = []
    for label, csv, sym, start, end in RUNS:
        if only and only.lower() not in label.lower():
            continue
        print(f"\n=== {label} ({sym} {start}..{end}) ===", flush=True)
        r = run_one(label, csv, sym, start, end)
        rows.append(r)
        if r["status"] != "OK":
            print(f"  {r['status']}: {r.get('detail','')}")
            continue
        print(f"  NT={r['nt_n']}  PY={r['py_n']}  matched={r['matched']} "
              f"({r['match_rate']*100:.1f}% of NT)  nt_only={r['nt_only']}  py_only={r['py_only']}")
        print(f"  mean entry-price err = {r['mean_entry_err']:.3f} pt")
        print(f"  AGG  win%  NT {r['nt_win']*100:5.1f} vs PY {r['py_win']*100:5.1f}   "
              f"PF  NT {r['nt_pf']:.2f} vs PY {r['py_pf']:.2f}   "
              f"net$ NT {r['nt_net']:+,.0f} vs PY {r['py_net']:+,.0f}")
        for w in r["warnings"]:
            print(f"  ! {w}")

    print("\n===== AGGREGATE PARITY (distribution-level) =====")
    hdr = (f"{'run':10s} {'cnt NT/PY':>11s} {'win NT/PY':>13s} "
           f"{'PF NT/PY':>13s} {'net$ NT/PY':>21s}")
    print(hdr)
    for r in rows:
        if r["status"] != "OK":
            print(f"{r['label']:10s}  {r['status']}")
            continue
        print(f"{r['label']:10s} {r['nt_n']:5d}/{r['py_n']:<5d} "
              f"{r['nt_win']*100:5.1f}/{r['py_win']*100:<5.1f}%  "
              f"{r['nt_pf']:5.2f}/{r['py_pf']:<5.2f}  "
              f"{r['nt_net']:+9,.0f}/{r['py_net']:+9,.0f}")


if __name__ == "__main__":
    main()
