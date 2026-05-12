"""
Walk-forward tick-size sweep — MoBoBands on MCL.

Folds (5):
  1: IS 2024-08-01→2024-09-30  OOS 2024-10-01→2024-10-31
  2: IS 2024-09-01→2024-10-31  OOS 2024-11-01→2024-11-30
  3: IS 2024-10-01→2024-11-30  OOS 2024-12-01→2024-12-31
  4: IS 2024-11-01→2024-12-31  OOS 2025-01-01→2025-01-31
  5: IS 2025-07-01→2025-08-31  OOS 2025-09-01→2025-09-30

Tick sizes: 233, 377, 512, 610, 987
"""

from __future__ import annotations

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = "/home/ad/strategy-platform-v2"
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, ".env"))

import numpy as np
import pandas as pd
from datetime import date

from strategy_platform.data.loader import load_tick_bars, INSTRUMENT_META
from strategy_platform.strategies.mobobands.strategy import MoboBandsPro, _summarise

# ── instrument meta ──────────────────────────────────────────────────────────
META = INSTRUMENT_META.get("MCL")
if META is None or META.get("tick_value") != 1.0:
    print("WARNING: INSTRUMENT_META['MCL'] missing or unexpected — using hardcoded values.")
    META = {"tick_size": 0.01, "tick_value": 1.0, "commission": 0.50}
else:
    print(f"MCL meta: tick_size={META['tick_size']}, tick_value={META['tick_value']}, "
          f"commission={META['commission']}")

TICK_SIZE   = META["tick_size"]
TICK_VALUE  = META["tick_value"]
COMMISSION  = META["commission"]   # per round-trip

# ── strategy instance ─────────────────────────────────────────────────────────
strat = MoboBandsPro()
strat.tick_size     = TICK_SIZE
strat.tick_value    = TICK_VALUE
strat.commission_rt = COMMISSION
strat.symbol        = "MCL"

# ── baseline params (merged on top of default_params) ────────────────────────
BASELINE = {
    "mobo_length":               21,
    "num_dev_up":                1.2,
    "num_dev_dn":                1.2,
    "profit_ticks":              50,
    "stop_ticks":                15,
    "require_color_change":      False,
    "enable_divergence_filter":  False,
    "enable_time_filter":        False,
    "enable_wattah_atar":        False,
    "calculate_mode":            "on_bar_close",
    "_symbol":                   "MCL",
}

# ── walk-forward fold definitions ─────────────────────────────────────────────
FOLDS = [
    {"fold": 1, "is_start": "2024-08-01", "is_end": "2024-09-30",
               "oos_start": "2024-10-01", "oos_end": "2024-10-31"},
    {"fold": 2, "is_start": "2024-09-01", "is_end": "2024-10-31",
               "oos_start": "2024-11-01", "oos_end": "2024-11-30"},
    {"fold": 3, "is_start": "2024-10-01", "is_end": "2024-11-30",
               "oos_start": "2024-12-01", "oos_end": "2024-12-31"},
    {"fold": 4, "is_start": "2024-11-01", "is_end": "2024-12-31",
               "oos_start": "2025-01-01", "oos_end": "2025-01-31"},
    {"fold": 5, "is_start": "2025-07-01", "is_end": "2025-08-31",
               "oos_start": "2025-09-01", "oos_end": "2025-09-30"},
]

TICK_SIZES = [233, 377, 512, 610, 987]

# ── session bucketing ─────────────────────────────────────────────────────────
def session_bucket(et_hour: int) -> str:
    """Map ET hour-of-day to session label."""
    if et_hour >= 19 or et_hour < 2:
        return "Asia"
    elif 2 <= et_hour < 8:
        return "London"
    elif 8 <= et_hour < 12:
        return "NY_AM"
    elif 12 <= et_hour < 16:
        return "NY_PM"
    else:   # 16 <= et_hour < 19
        return "Globex"


# ── quick stats helper (avoids re-importing _summarise for subsections) ───────
def fold_stats(trades: list) -> dict:
    if not trades:
        return {
            "sharpe": 0.0, "pf": 0.0, "net_pnl": 0.0,
            "trades": 0, "max_dd": 0.0, "win_rate": 0.0,
        }
    pnls = np.array([t["pnl"] for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    cum  = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    max_dd = float((peak - cum).max())

    gross_profit = float(wins.sum())   if len(wins)   > 0 else 0.0
    gross_loss   = float(-losses.sum()) if len(losses) > 0 else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)

    win_rate = float(len(wins) / len(pnls))

    # Monthly Sharpe (same convention as _summarise)
    daily_map: dict = {}
    for t in trades:
        d = t["entry_time"].date()
        daily_map[d] = daily_map.get(d, 0.0) + t["pnl"]
    d_vals = np.array(list(daily_map.values()), dtype=float)
    std = d_vals.std(ddof=1) if len(d_vals) > 1 else 0.0
    sharpe = float((d_vals.mean() / std) * np.sqrt(252)) if std > 0 else 0.0

    return {
        "sharpe": round(sharpe, 4),
        "pf": round(pf, 4),
        "net_pnl": round(float(pnls.sum()), 2),
        "trades": len(trades),
        "max_dd": round(max_dd, 2),
        "win_rate": round(win_rate, 4),
    }


# ── data cache: avoid re-fetching the same wide date range twice ──────────────
_bar_cache: dict = {}

def get_bars(tick_size: int, start: str, end: str) -> pd.DataFrame:
    key = (tick_size, start, end)
    if key not in _bar_cache:
        print(f"  Loading MCL {tick_size}-tick bars {start} → {end} ...")
        df = load_tick_bars("MCL", bar_size=tick_size, start=start, end=end)
        _bar_cache[key] = df
    return _bar_cache[key]


# We cache per tick_size for the widest spans needed, then slice.
# Widest IS+OOS spans per fold:
WIDE_SPANS = {
    233:  ("2024-08-01", "2025-09-30"),
    377:  ("2024-08-01", "2025-09-30"),
    512:  ("2024-08-01", "2025-09-30"),
    610:  ("2024-08-01", "2025-09-30"),
    987:  ("2024-08-01", "2025-09-30"),
}

def get_full_bars(tick_size: int) -> pd.DataFrame:
    s, e = WIDE_SPANS[tick_size]
    key = (tick_size, "FULL")
    if key not in _bar_cache:
        print(f"  Fetching full MCL {tick_size}-tick bars {s} → {e} ...")
        df = load_tick_bars("MCL", bar_size=tick_size, start=s, end=e)
        _bar_cache[key] = df
    return _bar_cache[key]


# ── main sweep ────────────────────────────────────────────────────────────────
rows_main    = []   # per-fold per-tick-size
rows_session = []   # per-trade with session label

print("=" * 70)
print("Walk-Forward Sweep: MoBoBands / MCL")
print("=" * 70)

for tick_size in TICK_SIZES:
    print(f"\n--- Tick size: {tick_size} ---")
    full_df = get_full_bars(tick_size)
    print(f"  Total bars loaded: {len(full_df):,}  "
          f"({full_df.index[0]} → {full_df.index[-1]})")

    params = {**strat.default_params, **BASELINE, "tick_bar_size": tick_size}

    for fold in FOLDS:
        fn       = fold["fold"]
        is_start = fold["is_start"]
        is_end   = fold["is_end"]
        oos_start = fold["oos_start"]
        oos_end   = fold["oos_end"]

        # Slice — index is UTC but we compare date strings (both resolve correctly
        # because session gaps mean bars never straddle midnight by more than ~1 bar).
        is_df  = full_df.loc[is_start:is_end]
        oos_df = full_df.loc[oos_start:oos_end]

        if len(is_df) < 50:
            print(f"  Fold {fn} IS: only {len(is_df)} bars — SKIP")
            continue
        if len(oos_df) < 10:
            print(f"  Fold {fn} OOS: only {len(oos_df)} bars — SKIP")
            continue

        print(f"  Fold {fn}: IS={len(is_df):,} bars, OOS={len(oos_df):,} bars", end="")

        # IS run (informational only)
        is_result  = strat.run_backtest(is_df,  params)
        is_trades  = is_result.get("trades", pd.DataFrame())
        if isinstance(is_trades, pd.DataFrame) and not is_trades.empty:
            is_trade_list = is_trades.to_dict("records")
        else:
            is_trade_list = []
        is_s = fold_stats(is_trade_list)

        # OOS run (ranking metric)
        oos_result = strat.run_backtest(oos_df, params)
        oos_trades = oos_result.get("trades", pd.DataFrame())
        if isinstance(oos_trades, pd.DataFrame) and not oos_trades.empty:
            oos_trade_list = oos_trades.to_dict("records")
        else:
            oos_trade_list = []
        oos_s = fold_stats(oos_trade_list)

        flag = " [<30 trades - UNRELIABLE]" if oos_s["trades"] < 30 else ""
        print(f" | IS trades={is_s['trades']} OOS trades={oos_s['trades']}{flag}")

        rows_main.append({
            "instrument":   "MCL",
            "strategy":     "mobobands",
            "fold":         fn,
            "tick_size":    tick_size,
            "is_start":     is_start,
            "is_end":       is_end,
            "oos_start":    oos_start,
            "oos_end":      oos_end,
            "is_sharpe":    is_s["sharpe"],
            "is_pf":        is_s["pf"],
            "is_net_pnl":   is_s["net_pnl"],
            "is_trades":    is_s["trades"],
            "is_max_dd":    is_s["max_dd"],
            "oos_sharpe":   oos_s["sharpe"],
            "oos_pf":       oos_s["pf"],
            "oos_net_pnl":  oos_s["net_pnl"],
            "oos_trades":   oos_s["trades"],
            "oos_max_dd":   oos_s["max_dd"],
            "oos_win_rate": oos_s["win_rate"],
            "unreliable":   oos_s["trades"] < 30,
        })

        # Session bucketing — OOS trades only
        for t in oos_trade_list:
            entry_utc = pd.Timestamp(t["entry_time"])
            if entry_utc.tzinfo is None:
                entry_utc = entry_utc.tz_localize("UTC")
            entry_et  = entry_utc.tz_convert("US/Eastern")
            bucket    = session_bucket(entry_et.hour)
            rows_session.append({
                "tick_size":   tick_size,
                "fold":        fn,
                "oos_start":   oos_start,
                "oos_end":     oos_end,
                "entry_utc":   str(entry_utc),
                "entry_et":    str(entry_et),
                "session":     bucket,
                "direction":   t.get("direction", ""),
                "pnl":         t["pnl"],
                "exit_reason": t.get("exit_reason", ""),
                "win":         1 if t["pnl"] > 0 else 0,
            })


# ── write TSV 1: per-fold stats ───────────────────────────────────────────────
OUT1 = os.path.join(ROOT, "reports", "wf_mobobands_MCL_2026-05-12.tsv")
df_main = pd.DataFrame(rows_main)
df_main.to_csv(OUT1, sep="\t", index=False)
print(f"\nWrote {OUT1}  ({len(df_main)} rows)")

# ── write TSV 2: session bucket aggregates ────────────────────────────────────
OUT2 = os.path.join(ROOT, "reports", "wf_mobobands_MCL_sessions.tsv")
df_sess = pd.DataFrame(rows_session)

if not df_sess.empty:
    sess_agg_rows = []
    SESSIONS = ["Asia", "London", "NY_AM", "NY_PM", "Globex"]
    for tick_size in TICK_SIZES:
        ts_df = df_sess[df_sess["tick_size"] == tick_size]
        for sess in SESSIONS:
            s_df = ts_df[ts_df["session"] == sess]
            if s_df.empty:
                sess_agg_rows.append({
                    "tick_size": tick_size, "session": sess,
                    "trades": 0, "win_rate": 0.0,
                    "net_pnl": 0.0, "pf": 0.0, "sharpe": 0.0,
                })
                continue
            pnls = s_df["pnl"].values.astype(float)
            wins = pnls[pnls > 0]
            losses = pnls[pnls < 0]
            gp = float(wins.sum())   if len(wins)   > 0 else 0.0
            gl = float(-losses.sum()) if len(losses) > 0 else 0.0
            pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
            wr = float(s_df["win"].mean())
            # Simple Sharpe from trade PnL sequence
            std = pnls.std(ddof=1) if len(pnls) > 1 else 0.0
            sharpe = float(pnls.mean() / std) if std > 0 else 0.0
            sess_agg_rows.append({
                "tick_size": tick_size,
                "session":   sess,
                "trades":    len(pnls),
                "win_rate":  round(wr, 4),
                "net_pnl":   round(float(pnls.sum()), 2),
                "pf":        round(pf, 4),
                "sharpe":    round(sharpe, 4),
            })
    df_sess_agg = pd.DataFrame(sess_agg_rows)
    df_sess_agg.to_csv(OUT2, sep="\t", index=False)
    print(f"Wrote {OUT2}  ({len(df_sess_agg)} rows)")
else:
    print("No session data — TSV 2 not written (no OOS trades).")
    df_sess_agg = pd.DataFrame()

# ── composite scores per tick_size ────────────────────────────────────────────
print("\n--- Composite OOS Scores (all folds) ---")
score_rows = []
for tick_size in TICK_SIZES:
    sub = df_main[df_main["tick_size"] == tick_size]
    reliable = sub[~sub["unreliable"]]
    all_sharpes = sub["oos_sharpe"].values
    rel_sharpes = reliable["oos_sharpe"].values
    mean_s = float(np.mean(all_sharpes)) if len(all_sharpes) > 0 else 0.0
    std_s  = float(np.std(all_sharpes, ddof=1)) if len(all_sharpes) > 1 else 0.0
    composite = mean_s / (1 + std_s)
    pct_pos  = float((sub["oos_net_pnl"] > 0).mean())
    n_reliable = int((~sub["unreliable"]).sum())
    score_rows.append({
        "tick_size": tick_size,
        "n_folds": len(sub),
        "n_reliable": n_reliable,
        "mean_oos_sharpe": round(mean_s, 4),
        "std_oos_sharpe":  round(std_s, 4),
        "composite":       round(composite, 4),
        "pct_pos_pnl":     round(pct_pos, 4),
        "mean_oos_pnl":    round(float(sub["oos_net_pnl"].mean()), 2),
        "mean_oos_trades": round(float(sub["oos_trades"].mean()), 1),
    })
    print(f"  {tick_size}-tick: mean_sharpe={mean_s:.3f}  std={std_s:.3f}  "
          f"composite={composite:.3f}  pct_pos={pct_pos:.0%}  "
          f"mean_trades={sub['oos_trades'].mean():.1f}  reliable_folds={n_reliable}/{len(sub)}")

df_scores = pd.DataFrame(score_rows).sort_values("composite", ascending=False)
best_row   = df_scores.iloc[0]
best_tick  = int(best_row["tick_size"])

# ── best session for best tick_size ──────────────────────────────────────────
best_session = "N/A"
if not df_sess_agg.empty:
    best_sess_df = df_sess_agg[df_sess_agg["tick_size"] == best_tick].sort_values(
        "sharpe", ascending=False
    )
    if not best_sess_df.empty:
        best_session = best_sess_df.iloc[0]["session"]

# ── write report: summary.md ──────────────────────────────────────────────────
OUT3 = os.path.join(ROOT, "reports", "wf_mobobands_MCL_summary.md")

lines = []
lines.append("# MoBoBands MCL Walk-Forward Summary — 2026-05-12\n")
lines.append(f"**Instrument:** MCL  |  **Strategy:** MoBoBands  |  "
             f"**commission:** ${COMMISSION:.2f} RT  |  "
             f"**tick_value:** ${TICK_VALUE:.2f}\n")
lines.append(f"**Baseline params:** mobo_length=21, num_dev_up=1.2, num_dev_dn=1.2, "
             f"profit_ticks=50, stop_ticks=15, all filters OFF\n")
lines.append("\n## Composite OOS Scores\n")
lines.append("| tick_size | folds | reliable | mean_oos_sharpe | std_oos_sharpe | composite | pct_pos_pnl | mean_oos_trades |\n")
lines.append("|-----------|-------|----------|-----------------|----------------|-----------|-------------|----------------|\n")
for _, r in df_scores.iterrows():
    lines.append(f"| {int(r['tick_size'])} | {int(r['n_folds'])} | {int(r['n_reliable'])} | "
                 f"{r['mean_oos_sharpe']:.4f} | {r['std_oos_sharpe']:.4f} | "
                 f"{r['composite']:.4f} | {r['pct_pos_pnl']:.0%} | {r['mean_oos_trades']:.1f} |\n")

lines.append("\n## Per-Fold Detail\n")
lines.append("| fold | tick_size | is_trades | oos_trades | oos_sharpe | oos_pf | oos_net_pnl | oos_max_dd | oos_win_rate | unreliable |\n")
lines.append("|------|-----------|-----------|------------|------------|--------|-------------|------------|--------------|------------|\n")
for _, r in df_main.sort_values(["tick_size", "fold"]).iterrows():
    flag = "YES" if r["unreliable"] else ""
    lines.append(f"| {int(r['fold'])} | {int(r['tick_size'])} | {int(r['is_trades'])} | "
                 f"{int(r['oos_trades'])} | {r['oos_sharpe']:.4f} | {r['oos_pf']:.4f} | "
                 f"{r['oos_net_pnl']:.2f} | {r['oos_max_dd']:.2f} | "
                 f"{r['oos_win_rate']:.4f} | {flag} |\n")

if not df_sess_agg.empty:
    lines.append("\n## Session Bucket Breakdown\n")
    lines.append("| tick_size | session | trades | win_rate | net_pnl | pf | sharpe |\n")
    lines.append("|-----------|---------|--------|----------|---------|-----|--------|\n")
    for _, r in df_sess_agg.sort_values(["tick_size", "session"]).iterrows():
        lines.append(f"| {int(r['tick_size'])} | {r['session']} | {int(r['trades'])} | "
                     f"{r['win_rate']:.1%} | {r['net_pnl']:.2f} | {r['pf']:.4f} | {r['sharpe']:.4f} |\n")

# ── recommendation ────────────────────────────────────────────────────────────
lines.append("\n## Recommendation\n")

no_edge = all(df_scores["composite"] <= 0) or all(df_scores["mean_oos_sharpe"] <= 0)

if no_edge:
    lines.append("**no edge found** — No tick size produced a positive composite OOS score. "
                 "MoBoBands on MCL at these baseline params shows no stable edge across the "
                 "five walk-forward folds. Do not trade until params are revisited.\n")
else:
    lines.append(f"**Best tick size: {best_tick}**\n\n")
    lines.append(f"Composite stability score: `mean_oos_sharpe / (1 + stdev_oos_sharpe)` = "
                 f"{best_row['composite']:.4f}\n\n")
    lines.append(f"Mean OOS Sharpe = {best_row['mean_oos_sharpe']:.4f}, "
                 f"std = {best_row['std_oos_sharpe']:.4f}, "
                 f"{best_row['pct_pos_pnl']:.0%} of folds positive PnL, "
                 f"mean {best_row['mean_oos_trades']:.1f} trades/fold.\n\n")
    if best_row["n_reliable"] < best_row["n_folds"]:
        n_unrel = int(best_row["n_folds"] - best_row["n_reliable"])
        lines.append(f"**Warning:** {n_unrel} of {int(best_row['n_folds'])} folds had <30 OOS trades "
                     f"and are flagged as unreliable.\n\n")
    lines.append(f"Best session window: **{best_session}** (highest per-trade Sharpe across OOS folds).\n")

lines.append("\n---\n")
lines.append("*Folds with <30 OOS trades flagged as UNRELIABLE. "
             "IS results are informational only — ranking is OOS only.*\n")

with open(OUT3, "w") as f:
    f.writelines(lines)
print(f"Wrote {OUT3}")

print("\nDone.")
