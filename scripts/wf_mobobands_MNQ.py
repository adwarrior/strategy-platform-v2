"""
Walk-forward tick-size sweep: MoBoBands on MNQ.

Outputs:
  reports/wf_mobobands_MNQ_2026-05-12.tsv      — per-fold per-tick-size IS+OOS stats
  reports/wf_mobobands_MNQ_sessions.tsv         — OOS trades bucketed by ET session hour
  reports/wf_mobobands_MNQ_summary.md           — best tick size + session recommendation

Commission note:
  loader.py INSTRUMENT_META['MNQ']['commission'] = $0.50 RT.
  Project memory (Commisions.txt lookup) says $1.02 RT.
  This script uses $0.50 (loader value = ground truth in codebase). FLAGGED.
"""

from __future__ import annotations

import os
import sys
import warnings
from typing import Any, Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Path setup ──────────────────────────────────────────────────────────────
ROOT = "/home/ad/strategy-platform-v2"
sys.path.insert(0, ROOT)

from strategy_platform.data.loader import load_tick_bars, INSTRUMENT_META
from strategy_platform.strategies.mobobands.strategy import MoboBandsPro

# ── Constants ────────────────────────────────────────────────────────────────
SYMBOL       = "MNQ"
STRATEGY_KEY = "mobobands"
REPORTS_DIR  = os.path.join(ROOT, "reports")

META         = INSTRUMENT_META[SYMBOL]
TICK_SIZE    = META["tick_size"]    # 0.25
TICK_VALUE   = META["tick_value"]   # 0.50
COMMISSION   = META["commission"]   # 0.50 RT  (see docstring — memory says 1.02, flagged)

TICK_SIZES = [233, 377, 512, 610, 987]

# WF folds: (is_start, is_end, oos_start, oos_end)
FOLDS = [
    ("2024-10-01", "2024-11-30", "2024-12-01", "2024-12-31"),
    ("2024-12-01", "2025-01-31", "2025-02-01", "2025-02-28"),
    ("2025-02-01", "2025-03-31", "2025-04-01", "2025-04-30"),
    ("2025-04-01", "2025-05-31", "2025-06-01", "2025-06-30"),
    ("2025-08-01", "2025-09-30", "2025-10-01", "2025-10-31"),
]

# Baseline params — override defaults; all filters OFF per plan.
BASELINE_OVERRIDE = {
    "mobo_length":               21,
    "num_dev_up":                1.2,
    "num_dev_dn":                1.2,
    "profit_ticks":              50,
    "stop_ticks":                15,
    "require_color_change":      False,
    "enable_divergence_filter":  False,
    "enable_time_filter":        False,
    "enable_wattah_atar":        False,
    "enable_bw_filter":          False,
    "enable_jurik_filter":       False,
    "enable_adx_filter":        False,
    "calculate_mode":            "on_bar_close",
}

# Session buckets (ET hour ranges, inclusive start exclusive end)
SESSION_BUCKETS = {
    "Asia":      (19, 2),   # wraps midnight: 19:00–01:59 ET
    "London":    (2,  8),
    "NY_AM":     (8,  12),
    "NY_PM":     (12, 16),
    "Globex":    (16, 19),
}


def _session_label(hour_et: int) -> str:
    if hour_et >= 19 or hour_et < 2:
        return "Asia"
    if 2 <= hour_et < 8:
        return "London"
    if 8 <= hour_et < 12:
        return "NY_AM"
    if 12 <= hour_et < 16:
        return "NY_PM"
    return "Globex"


def _summarise_trades(trades: List[Dict]) -> Dict[str, Any]:
    """Light-weight stats for a fold leg (IS or OOS)."""
    if not trades:
        return {
            "trades": 0, "net_pnl": 0.0, "win_rate": 0.0,
            "profit_factor": 0.0, "max_drawdown": 0.0, "sharpe": 0.0,
        }
    pnls   = np.array([t["pnl"] for t in trades], dtype=float)
    wins   = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    cum    = np.cumsum(pnls)
    peak   = np.maximum.accumulate(cum)
    max_dd = float((peak - cum).max())

    gross_p = wins.sum()   if len(wins)   > 0 else 0.0
    gross_l = abs(losses.sum()) if len(losses) > 0 else 0.0
    pf      = gross_p / gross_l if gross_l > 0 else (float("inf") if gross_p > 0 else 0.0)

    # Daily PnL → annualised Sharpe
    daily: Dict = {}
    for t in trades:
        d = t["session_date"]
        daily[d] = daily.get(d, 0.0) + t["pnl"]
    d_vals = np.array(list(daily.values()), dtype=float)
    std    = d_vals.std(ddof=1) if len(d_vals) > 1 else 0.0
    sharpe = float(d_vals.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    return {
        "trades":        len(trades),
        "net_pnl":       float(pnls.sum()),
        "win_rate":      float((pnls > 0).mean()),
        "profit_factor": round(pf, 4),
        "max_drawdown":  round(max_dd, 2),
        "sharpe":        round(sharpe, 4),
    }


def _run_fold(
    strategy: MoboBandsPro,
    df_full: pd.DataFrame,
    fold_idx: int,
    tick_sz: int,
    is_start: str, is_end: str,
    oos_start: str, oos_end: str,
) -> tuple[Dict, List[Dict]]:
    """Run one IS+OOS fold. Returns (row_dict, oos_trade_list)."""

    # Slice using UTC index (tick bars are UTC-indexed)
    is_s_utc  = pd.Timestamp(is_start,  tz="UTC")
    is_e_utc  = pd.Timestamp(is_end,    tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    oos_s_utc = pd.Timestamp(oos_start, tz="UTC")
    oos_e_utc = pd.Timestamp(oos_end,   tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

    df_is  = df_full.loc[is_s_utc:is_e_utc]
    df_oos = df_full.loc[oos_s_utc:oos_e_utc]

    params = {**strategy.default_params, **BASELINE_OVERRIDE}

    # Set instrument attrs on strategy so run_backtest uses correct tick_size/value/commission
    strategy.tick_size    = TICK_SIZE
    strategy.tick_value   = TICK_VALUE
    strategy.commission_rt = COMMISSION

    def _backtest(df: pd.DataFrame) -> List[Dict]:
        if len(df) < 50:
            return []
        try:
            result = strategy.run_backtest(df, params)
            # run_backtest returns {stats..., 'trades': pd.DataFrame}
            trades_df = result.get("trades", pd.DataFrame())
            if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
                return trades_df.to_dict("records")
            return []
        except Exception as exc:
            print(f"    [fold {fold_idx} tick={tick_sz}] backtest error: {exc}")
            return []

    is_trades  = _backtest(df_is)
    oos_trades = _backtest(df_oos)

    is_stats  = _summarise_trades(is_trades)
    oos_stats = _summarise_trades(oos_trades)

    low_trade_flag = oos_stats["trades"] < 30

    row = {
        "instrument": SYMBOL,
        "strategy":   STRATEGY_KEY,
        "fold":       fold_idx,
        "tick_size":  tick_sz,
        "is_start":   is_start,
        "is_end":     is_end,
        "oos_start":  oos_start,
        "oos_end":    oos_end,
        # IS
        "is_trades":        is_stats["trades"],
        "is_net_pnl":       round(is_stats["net_pnl"], 2),
        "is_win_rate":      round(is_stats["win_rate"], 4),
        "is_profit_factor": is_stats["profit_factor"],
        "is_max_drawdown":  is_stats["max_drawdown"],
        "is_sharpe":        is_stats["sharpe"],
        # OOS
        "oos_trades":        oos_stats["trades"],
        "oos_net_pnl":       round(oos_stats["net_pnl"], 2),
        "oos_win_rate":      round(oos_stats["win_rate"], 4),
        "oos_profit_factor": oos_stats["profit_factor"],
        "oos_max_drawdown":  oos_stats["max_drawdown"],
        "oos_sharpe":        oos_stats["sharpe"],
        "low_trade_flag":   low_trade_flag,
    }

    # Tag each OOS trade with fold + tick_size + ET session
    for t in oos_trades:
        entry_et = t["entry_time"]
        # entry_time from _make_trade is a pandas Timestamp.
        # If it's tz-naive (strategy strips tz), we must localize+convert.
        if entry_et.tzinfo is None:
            # Tick bar index was UTC-aware; strategy may strip tz in _run_backtest_loop.
            # Treat as UTC then convert.
            entry_et = entry_et.tz_localize("UTC").tz_convert("America/New_York")
        else:
            entry_et = entry_et.tz_convert("America/New_York")
        t["_fold"]      = fold_idx
        t["_tick_size"] = tick_sz
        t["_session"]   = _session_label(entry_et.hour)

    return row, oos_trades


def _check_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure index is UTC-aware (load_tick_bars returns tz-naive UTC in older versions)."""
    if df.index.tz is None:
        df = df.copy()
        df.index = df.index.tz_localize("UTC")
    return df


def main():
    strategy = MoboBandsPro()

    all_rows:       List[Dict]  = []
    all_oos_trades: List[Dict]  = []

    # Load per-fold (3mo at a time) to avoid OOM on MNQ's 130M+ ticks
    import gc

    for tick_sz in TICK_SIZES:
        print(f"\n{'='*60}")
        print(f"  Tick size: {tick_sz}")
        print(f"{'='*60}")

        for i, (is_s, is_e, oos_s, oos_e) in enumerate(FOLDS, start=1):
            print(f"  Fold {i}: loading {is_s} → {oos_e} ...", flush=True)
            df_fold = load_tick_bars(SYMBOL, bar_size=tick_sz,
                                     start=is_s, end=oos_e)
            df_fold = _check_tz(df_fold)
            print(f"    Loaded {len(df_fold):,} bars. IS {is_s}→{is_e}  OOS {oos_s}→{oos_e}",
                  flush=True)
            row, oos_trades = _run_fold(
                strategy, df_fold, i, tick_sz,
                is_s, is_e, oos_s, oos_e,
            )
            all_rows.append(row)
            all_oos_trades.extend(oos_trades)
            flag = " [LOW TRADES]" if row["low_trade_flag"] else ""
            print(f"    IS={row['is_trades']} trades  OOS={row['oos_trades']} trades  "
                  f"OOS Sharpe={row['oos_sharpe']:.3f}{flag}", flush=True)
            del df_fold
            gc.collect()

    # ── Write main TSV ────────────────────────────────────────────────────────
    df_main = pd.DataFrame(all_rows)
    main_path = os.path.join(REPORTS_DIR, "wf_mobobands_MNQ_2026-05-12.tsv")
    df_main.to_csv(main_path, sep="\t", index=False)
    print(f"\nWrote: {main_path}")

    # ── Session TSV ───────────────────────────────────────────────────────────
    if all_oos_trades:
        sess_rows = []
        for tick_sz in TICK_SIZES:
            tick_trades = [t for t in all_oos_trades if t["_tick_size"] == tick_sz]
            for sess in ["Asia", "London", "NY_AM", "NY_PM", "Globex"]:
                s_trades = [t for t in tick_trades if t["_session"] == sess]
                if not s_trades:
                    sess_rows.append({
                        "tick_size": tick_sz, "session": sess,
                        "trades": 0, "win_rate": 0.0,
                        "expectancy": 0.0, "profit_factor": 0.0, "net_pnl": 0.0,
                    })
                    continue
                pnls  = np.array([t["pnl"] for t in s_trades], dtype=float)
                wins  = pnls[pnls > 0]
                losses= pnls[pnls < 0]
                gp    = wins.sum() if len(wins) > 0 else 0.0
                gl    = abs(losses.sum()) if len(losses) > 0 else 0.0
                pf    = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
                sess_rows.append({
                    "tick_size":      tick_sz,
                    "session":        sess,
                    "trades":         len(s_trades),
                    "win_rate":       round(float((pnls > 0).mean()), 4),
                    "expectancy":     round(float(pnls.mean()), 2),
                    "profit_factor":  round(pf, 4),
                    "net_pnl":        round(float(pnls.sum()), 2),
                })
        df_sess = pd.DataFrame(sess_rows)
        sess_path = os.path.join(REPORTS_DIR, "wf_mobobands_MNQ_sessions.tsv")
        df_sess.to_csv(sess_path, sep="\t", index=False)
        print(f"Wrote: {sess_path}")
    else:
        df_sess = pd.DataFrame()
        print("No OOS trades — sessions TSV skipped.")

    # ── Summary MD ────────────────────────────────────────────────────────────
    _write_summary(df_main, df_sess)


def _write_summary(df_main: pd.DataFrame, df_sess: pd.DataFrame):
    lines = ["# MoBoBands MNQ Walk-Forward Summary — 2026-05-12", ""]

    lines += [
        "## Commission flag",
        "",
        "`INSTRUMENT_META['MNQ']['commission']` = **$0.50 RT** (loader.py).",
        "Project memory (Commisions.txt) says **$1.02 RT**. Results use $0.50.",
        "Re-run with corrected value if $1.02 is confirmed as current broker rate.",
        "",
    ]

    lines += [
        "## OOS aggregate by tick size",
        "",
        "| tick_size | mean_oos_sharpe | stdev_oos_sharpe | composite | folds_pos_pnl | low_trade_folds |",
        "|----------:|----------------:|-----------------:|----------:|--------------:|----------------:|",
    ]

    best_composite = -np.inf
    best_tick      = None
    tick_composites = {}

    for tick_sz in TICK_SIZES:
        sub = df_main[df_main["tick_size"] == tick_sz]
        sharpes   = sub["oos_sharpe"].values
        pnls      = sub["oos_net_pnl"].values
        low_flags = sub["low_trade_flag"].sum()
        mean_s    = float(np.mean(sharpes))
        std_s     = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
        composite = mean_s / (1.0 + std_s)
        folds_pos = int((pnls > 0).sum())
        tick_composites[tick_sz] = composite
        lines.append(
            f"| {tick_sz:9d} | {mean_s:15.4f} | {std_s:16.4f} | "
            f"{composite:9.4f} | {folds_pos:13d} | {low_flags:15d} |"
        )
        if composite > best_composite:
            best_composite = composite
            best_tick      = tick_sz

    lines += ["", ""]

    # Session analysis for best tick
    if best_tick and not df_sess.empty:
        lines += [
            f"## Session breakdown — best tick size {best_tick}",
            "",
            "| session | trades | win_rate | expectancy | profit_factor | net_pnl |",
            "|:--------|-------:|---------:|-----------:|--------------:|--------:|",
        ]
        sub_sess = df_sess[df_sess["tick_size"] == best_tick].sort_values("net_pnl", ascending=False)
        best_session = None
        best_sess_pnl = -np.inf
        for _, r in sub_sess.iterrows():
            lines.append(
                f"| {r['session']:7s} | {int(r['trades']):6d} | {r['win_rate']:8.1%} | "
                f"{r['expectancy']:10.2f} | {r['profit_factor']:13.4f} | {r['net_pnl']:7.2f} |"
            )
            if r["net_pnl"] > best_sess_pnl and r["trades"] >= 30:
                best_sess_pnl = r["net_pnl"]
                best_session  = r["session"]
        lines += ["", ""]

    # Verdict
    lines += ["## Verdict", ""]
    if best_tick and best_composite > 0:
        sess_str = f", best session window **{best_session}**" if best_session else ""
        lines.append(
            f"Best tick size for MNQ mobobands: **{best_tick}** "
            f"(composite stability = {best_composite:.4f}{sess_str}). "
            f"Mean OOS Sharpe = {np.mean(df_main[df_main['tick_size']==best_tick]['oos_sharpe'].values):.4f}. "
            f"Commission used: $0.50 RT — verify against current broker rate ($1.02 per memory)."
        )
    else:
        lines.append(
            "No tick size produced positive composite OOS Sharpe. "
            "**No edge found** with baseline params on this WF span. "
            "Consider adjusting profit_ticks/stop_ticks before discarding."
        )

    lines += ["", f"*Folds flagged with <30 OOS trades are noted in the main TSV.*", ""]

    summary_path = os.path.join(REPORTS_DIR, "wf_mobobands_MNQ_summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
