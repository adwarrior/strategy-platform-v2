"""Aurora port vs LIVE forward-test fill logs (AuroraFills_*.csv).

Ground truth = the strategy's own fill logger running live on SimAurora
(State.Realtime), NOT NT backtest fills — see memory project_aurora_python_parity
(2026-07-09 pivot). Entry rows carry the wall's kind/mid/touches/age at fill
time, so this compares WALL SELECTION separately from fill/outcome parity.

Usage:
    AURORA_SYMBOL=MNQ_U26 python scripts/aurora_parity/live_log_parity.py \
        2026-07-07 "/home/ad/Scripts/Results/Aurora/images/July 7th/AuroraFills_20260707.csv" \
        entry_min_touches=1 flip_to_market=False

    python scripts/aurora_parity/live_log_parity.py \
        2026-07-08 ".../July 8th/AuroraFills_20260708.csv" \
        entry_min_touches=1 trade_bal=False flip_to_market=False

The port replays ticks from the PRIOR evening 18:00 ET (session open) so shelf
memory / ATR / volume EMA are saturated by 09:30, mirroring the live run that
was left running (entries only 09:30-12:00 via default params).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from strategy_platform.strategies.aurora.strategy import Aurora           # noqa: E402
from strategy_platform.strategies.aurora.tick_loader import load_raw_ticks  # noqa: E402

ENTRY_ORDERS = ("AuroraIntercept", "AuroraFlipMkt")


def parse_fill_log(path: str) -> pd.DataFrame:
    """Reconstruct round-trip trades from the per-fill CSV.

    positionAfter is unreliable (NT's OnExecutionUpdate marketPosition arg);
    track signed net position instead. A trade = flat -> nonzero -> flat.
    Wall metadata comes from the trade's first entry fill.
    """
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"])
    trades = []
    net = 0
    cur = None
    for _, r in df.iterrows():
        signed = int(r["qty"]) * (1 if r["side"] == "Buy" else -1)
        if r["order"] in ENTRY_ORDERS:
            if cur is None:
                cur = {
                    "entry_time": r["time"],
                    "direction": "long" if signed > 0 else "short",
                    "entry_px_qty": [],
                    "exit_px_qty": [],
                    "wall_kind": r["wallKind"],
                    "wall_mid": float(r["wallMid"]),
                    "wall_touches": int(r["wallTouches"]),
                    "wall_age_bars": int(r["wallAgeBars"]),
                    "order": r["order"],
                }
            cur["entry_px_qty"].append((float(r["price"]), abs(signed)))
        elif cur is not None:
            cur["exit_px_qty"].append((float(r["price"]), abs(signed)))
            cur["exit_time"] = r["time"]
            cur["exit_reason"] = r["order"]
        net += signed
        if cur is not None and net == 0:
            eq = sum(q for _, q in cur["entry_px_qty"])
            ep = sum(p * q for p, q in cur["entry_px_qty"]) / eq
            xq = sum(q for _, q in cur["exit_px_qty"]) or 1
            xp = sum(p * q for p, q in cur["exit_px_qty"]) / xq
            sgn = 1.0 if cur["direction"] == "long" else -1.0
            trades.append({
                "entry_time": cur["entry_time"],
                "exit_time": cur.get("exit_time"),
                "direction": cur["direction"],
                "entry_price": ep,
                "exit_price": xp,
                "qty": eq,
                "pnl_pts": sgn * (xp - ep) * eq,
                "wall_kind": cur["wall_kind"],
                "wall_mid": cur["wall_mid"],
                "wall_touches": cur["wall_touches"],
                "wall_age_bars": cur["wall_age_bars"],
                "reason": cur.get("exit_reason", ""),
                "order": cur["order"],
            })
            cur = None
    return pd.DataFrame(trades)


def run_port(day: str, symbol: str, overrides: dict) -> pd.DataFrame:
    day_ts = pd.Timestamp(day)
    prev = day_ts - pd.Timedelta(days=1)
    # tick_loader filters by whole UTC days; slice to the session in ET after.
    df = load_raw_ticks(symbol, str(prev.date()), str(day_ts.date()),
                        table="tick_data_full")
    if df.empty:
        raise SystemExit(f"no ticks for {symbol} around {day} in tick_data_full")
    start = prev + pd.Timedelta(hours=18)          # prior session open, ET
    end = day_ts + pd.Timedelta(hours=16)          # past flat_by 15:55
    df = df.loc[(df.index >= start) & (df.index < end)]
    print(f"ticks {df.index[0]} -> {df.index[-1]}  n={len(df):,}")
    res = Aurora().run_backtest(df, overrides)
    t = res["trades"]
    return t[pd.to_datetime(t["entry_time"]).dt.date == day_ts.date()].reset_index(drop=True)


def bar_label(ts: pd.Timestamp) -> pd.Timestamp:
    """NT close-time bar label of a raw fill instant (09:30:29 -> 09:31)."""
    f = ts.floor("1min")
    return f + pd.Timedelta(minutes=1) if ts != f else ts


def match(live: pd.DataFrame, port: pd.DataFrame, tol_min: int = 2):
    """Greedy same-direction match by wall + time, not time alone.

    Time-only nearest-entry pairing mis-attributed outcomes (2026-07-09 dig):
    it paired a live winner with the port's EXTRA earlier trade on the same
    wall while the port's true twin (+40, same wall/price, 1 bar later) went
    to "port-only", and in an 11:11-11:15 cluster it crossed two walls 34pt
    apart. Cost = wall-mid error + time distance + a penalty when the port's
    bar-label window CLOSED before the live fill instant (a resting-limit
    fill logged live at T cannot be the same fill as a port trade whose bar
    ended before T; allow slack for feed-clock skew but prefer causal pairs).
    Pairs are taken globally cheapest-first (avoids greedy row-order steals).
    """
    cands = []
    for i, lv in live.iterrows():
        lt = bar_label(lv["entry_time"])
        for j, pt in port.iterrows():
            if pt["direction"] != lv["direction"]:
                continue
            d = abs((pd.Timestamp(pt["entry_time"]) - lt).total_seconds()) / 60.0
            if d > tol_min:
                continue
            mid_err = abs(lv["wall_mid"] - pt["wall_mid"])
            causal_pen = 2.0 if pd.Timestamp(pt["entry_time"]) < lv["entry_time"] else 0.0
            cands.append((mid_err + 0.5 * d + causal_pen, i, j, d))
    used, used_live = set(), set()
    rows = []
    for cost, i, j, d in sorted(cands, key=lambda c: c[0]):
        if i in used_live or j in used:
            continue
        used_live.add(i)
        used.add(j)
        lv = live.loc[i]
        pt = port.loc[j]
        rows.append({
            "live_entry": lv["entry_time"], "port_entry": pt["entry_time"],
            "dt_min": d, "dir": lv["direction"],
            "live_kind": lv["wall_kind"], "port_kind": pt["wall_kind"],
            "live_mid": lv["wall_mid"], "port_mid": pt["wall_mid"],
            "mid_err": abs(lv["wall_mid"] - pt["wall_mid"]),
            "px_err": abs(lv["entry_price"] - pt["entry_price"]),
            "live_touch": lv["wall_touches"], "port_touch": pt["wall_touches"],
            "live_pts": lv["pnl_pts"],
            "port_pts": (1 if pt["direction"] == "long" else -1)
                        * (pt["exit_price"] - pt["entry_price"]) * pt["qty"],
        })
    rows.sort(key=lambda r: r["live_entry"])
    return pd.DataFrame(rows), used


def main():
    day, log_path = sys.argv[1], sys.argv[2]
    overrides = {}
    for a in sys.argv[3:]:
        k, v = a.split("=", 1)
        overrides[k] = (v.lower() == "true") if v.lower() in ("true", "false") \
            else (float(v) if "." in v else int(v))
    symbol = os.environ.get("AURORA_SYMBOL", "MNQ_U26")

    live = parse_fill_log(log_path)
    print(f"LIVE  {day}: {len(live)} trades, {live['pnl_pts'].sum():+.2f} pt-contracts, "
          f"win {(live['pnl_pts'] > 0).mean():.0%}")
    print(f"params: {overrides}")

    port = run_port(day, symbol, overrides)
    ppts = ((port["direction"].eq("long").astype(int) * 2 - 1)
            * (port["exit_price"] - port["entry_price"]) * port["qty"])
    print(f"PORT  {day}: {len(port)} trades, {ppts.sum():+.2f} pt-contracts, "
          f"win {(ppts > 0).mean():.0%}" if len(port) else "PORT: 0 trades")

    m, used = match(live, port)
    print(f"\nMATCH: {len(m)}/{len(live)} live trades matched "
          f"({len(m)/max(len(live),1):.0%}); port-only: {len(port)-len(used)}")
    if len(m):
        kind_ok = (m["live_kind"] == m["port_kind"]).mean()
        sign_ok = (np.sign(m["live_pts"]) == np.sign(m["port_pts"])).mean()
        print(f"wall kind agree: {kind_ok:.0%} | wall mid err median: "
              f"{m['mid_err'].median():.2f} | entry px err median: "
              f"{m['px_err'].median():.2f} | outcome sign agree: {sign_ok:.0%}")
        pd.set_option("display.width", 200)
        print("\n" + m.to_string(index=False))
    if len(port) - len(used) > 0:
        extra = port.loc[[j for j in port.index if j not in used],
                         ["entry_time", "direction", "wall_kind", "wall_mid", "wall_touches"]]
        print("\nPORT-ONLY trades:\n" + extra.to_string(index=False))
    unmatched = [i for i, lv in live.iterrows()
                 if not any(abs((pd.Timestamp(r["live_entry"]) - lv["entry_time"]).total_seconds()) < 1
                            for _, r in m.iterrows())]
    if unmatched:
        print("\nLIVE-ONLY trades:\n" + live.loc[unmatched, ["entry_time", "direction",
              "wall_kind", "wall_mid", "wall_touches"]].to_string(index=False))


if __name__ == "__main__":
    main()
