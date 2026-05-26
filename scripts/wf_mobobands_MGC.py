"""
Walk-forward sweep — MoBoBands / MGC
Tick sizes: 233, 377, 512, 610, 987
3 folds (post-gap span: 2025-04-01 → 2025-10-31)

Fold 1: IS 2025-04-01→2025-05-31, OOS 2025-06-01→2025-06-30
Fold 2: IS 2025-06-01→2025-07-31, OOS 2025-08-01→2025-08-31
Fold 3: IS 2025-08-01→2025-09-30, OOS 2025-10-01→2025-10-31

Commission: from INSTRUMENT_META['MGC'] = $1.52 RT
Tick data is UTC; session bucketing converts to US/Eastern.
"""

from __future__ import annotations

import os
import sys
import math
import warnings
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path("/home/ad/strategy-platform-v2")
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from strategy_platform.data.loader import load_tick_bars, INSTRUMENT_META
from strategy_platform.strategies.mobobands.strategy import MoboBandsPro as MobobandsStrategy

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SYMBOL      = "MGC"
STRATEGY    = "mobobands"
TICK_SIZES  = [233, 377, 512, 610, 987]
MIN_TRADES  = 30

# Verify INSTRUMENT_META for MGC
_meta = INSTRUMENT_META.get(SYMBOL)
if _meta is None:
    print(f"WARNING: INSTRUMENT_META missing '{SYMBOL}'. Using fallback commission $1.52")
    MGC_TICK_SIZE   = 0.10
    MGC_TICK_VALUE  = 1.00
    MGC_COMMISSION  = 1.52
else:
    MGC_TICK_SIZE   = _meta["tick_size"]
    MGC_TICK_VALUE  = _meta["tick_value"]
    MGC_COMMISSION  = _meta["commission"]
    if abs(MGC_COMMISSION - 1.52) > 0.01:
        print(f"NOTE: INSTRUMENT_META['MGC'].commission = {MGC_COMMISSION} (expected 1.52 from memory)")

print(f"MGC meta: tick_size={MGC_TICK_SIZE}, tick_value={MGC_TICK_VALUE}, commission_rt={MGC_COMMISSION}")

# ---------------------------------------------------------------------------
# Walk-forward fold definitions
# ---------------------------------------------------------------------------
FOLDS = [
    {"fold": 1, "is_start": "2025-04-01", "is_end": "2025-05-31",
               "oos_start": "2025-06-01", "oos_end": "2025-06-30"},
    {"fold": 2, "is_start": "2025-06-01", "is_end": "2025-07-31",
               "oos_start": "2025-08-01", "oos_end": "2025-08-31"},
    {"fold": 3, "is_start": "2025-08-01", "is_end": "2025-09-30",
               "oos_start": "2025-10-01", "oos_end": "2025-10-31"},
]

# ---------------------------------------------------------------------------
# Baseline params — merge over default_params so no required key is dropped
# ---------------------------------------------------------------------------
SWEEP_PARAMS_OVERRIDE = {
    "mobo_length":              21,
    "num_dev_up":               1.2,
    "num_dev_dn":               1.2,
    "profit_ticks":             50,
    "stop_ticks":               15,
    "require_color_change":     False,
    "enable_divergence_filter": False,
    "enable_time_filter":       False,
    "enable_wattah_atar":       False,
    "enable_bw_filter":         False,
    "enable_jurik_filter":      False,
    "enable_adx_filter":        False,
    "calculate_mode":           "on_bar_close",
}

# ---------------------------------------------------------------------------
# Session bucket helper (ET entry hour)
# ---------------------------------------------------------------------------
SESSION_LABELS = ["Asia", "London", "NY_AM", "NY_PM", "Globex"]

def session_for_hour(hour_et: int) -> str:
    """Return session label for an ET hour (0-23)."""
    if 2 <= hour_et < 8:
        return "London"
    if 8 <= hour_et < 12:
        return "NY_AM"
    if 12 <= hour_et < 16:
        return "NY_PM"
    if 16 <= hour_et < 19:
        return "Globex"
    # 19-24 and 0-2  → Asia
    return "Asia"

# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def compute_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    """Extract standardised metrics from run_backtest result dict."""
    trades_val = result.get("total_trades", result.get("trades", 0))
    # 'trades' key may be DataFrame or int depending on call path
    if isinstance(trades_val, pd.DataFrame):
        n_trades = len(trades_val)
    else:
        n_trades = int(trades_val)

    return {
        "sharpe":       float(result.get("sharpe", 0.0)),
        "pf":           float(result.get("profit_factor", 0.0)),
        "net_pnl":      float(result.get("net_pnl", 0.0)),
        "trades":       n_trades,
        "max_dd":       float(result.get("max_drawdown", 0.0)),
        "win_rate":     float(result.get("win_rate", 0.0)),
    }


def session_breakdown(trades_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Bucket trades by ET entry hour → session stats."""
    out: Dict[str, Dict] = {s: {"trades": 0, "wins": 0, "net_pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0}
                             for s in SESSION_LABELS}

    if trades_df is None or len(trades_df) == 0:
        return out

    # entry_time inherits the strategy's bar index, which is ET-naive
    # (loader returns ET-naive for emini MNQ/MES/MGC). Use as-is.
    _et_src = trades_df["entry_time"]
    if _et_src.dt.tz is not None:
        _et_src = _et_src.dt.tz_convert("US/Eastern").dt.tz_localize(None)
    et_times = _et_src
    for i, row in trades_df.iterrows():
        hour = et_times.iloc[i].hour
        sess = session_for_hour(hour)
        out[sess]["trades"]  += 1
        out[sess]["net_pnl"] += row["pnl"]
        if row["pnl"] > 0:
            out[sess]["wins"]         += 1
            out[sess]["gross_profit"] += row["pnl"]
        else:
            out[sess]["gross_loss"]   += row["pnl"]

    for s in SESSION_LABELS:
        b = out[s]
        n = b["trades"]
        b["win_rate"]   = b["wins"] / n if n > 0 else 0.0
        b["expectancy"] = b["net_pnl"] / n if n > 0 else 0.0
        gl = abs(b["gross_loss"])
        b["pf"]         = b["gross_profit"] / gl if gl > 0 else (float("inf") if b["gross_profit"] > 0 else 0.0)

    return out

# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def build_strategy(tick_size_bars: int) -> MobobandsStrategy:
    """Instantiate strategy with MGC meta patched in."""
    s = MobobandsStrategy()
    s.tick_size     = MGC_TICK_SIZE
    s.tick_value    = MGC_TICK_VALUE
    s.commission_rt = MGC_COMMISSION
    s.symbol        = SYMBOL
    s.bar_type      = "tick"
    return s


def build_params(tick_bar_size: int) -> Dict[str, Any]:
    strat = MobobandsStrategy()
    params = dict(strat.default_params)       # start from full defaults
    params.update(SWEEP_PARAMS_OVERRIDE)      # apply our overrides
    params["tick_bar_size"] = tick_bar_size   # sweep variable
    params["_symbol"]       = SYMBOL
    return params


def run_fold(strat: MobobandsStrategy, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
    """Run backtest on a slice; return metrics dict + trades DataFrame."""
    if len(data) < 50:
        return {"sharpe": 0.0, "pf": 0.0, "net_pnl": 0.0,
                "trades": 0, "max_dd": 0.0, "win_rate": 0.0,
                "_trades_df": pd.DataFrame(), "_flag": "INSUFFICIENT_BARS"}
    try:
        result = strat.run_backtest(data, params)
    except Exception as e:
        warnings.warn(f"run_backtest error: {e}")
        return {"sharpe": 0.0, "pf": 0.0, "net_pnl": 0.0,
                "trades": 0, "max_dd": 0.0, "win_rate": 0.0,
                "_trades_df": pd.DataFrame(), "_flag": f"ERROR:{e}"}

    m = compute_metrics(result)
    trades_df = result.get("trades", pd.DataFrame())
    if not isinstance(trades_df, pd.DataFrame):
        trades_df = pd.DataFrame()

    flag = ""
    if m["trades"] < MIN_TRADES:
        flag = f"LOW_TRADES({m['trades']})"

    m["_trades_df"] = trades_df
    m["_flag"]      = flag
    return m


def main():
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    fold_rows:    List[Dict] = []
    session_rows: List[Dict] = []

    # Cache tick bar data per (tick_size, period) to avoid re-querying
    # Load all data once per tick_size for the full span, then slice
    full_span_start = "2025-04-01"
    full_span_end   = "2025-10-31"

    for tick_sz in TICK_SIZES:
        print(f"\n=== Tick size: {tick_sz} ===")
        print(f"  Loading {SYMBOL} tick bars ({full_span_start} → {full_span_end})...")

        try:
            full_data = load_tick_bars(
                symbol   = SYMBOL,
                bar_size = tick_sz,
                start    = full_span_start,
                end      = full_span_end,
                host     = os.getenv("DB_HOST", "192.168.1.228"),
            )
        except Exception as e:
            print(f"  ERROR loading data for tick_sz={tick_sz}: {e}")
            # Record error rows for all folds
            for fold in FOLDS:
                fold_rows.append({
                    "instrument": SYMBOL, "strategy": STRATEGY,
                    "fold": fold["fold"], "tick_size": tick_sz,
                    "is_start": fold["is_start"], "is_end": fold["is_end"],
                    "oos_start": fold["oos_start"], "oos_end": fold["oos_end"],
                    "is_sharpe": None, "is_pf": None, "is_net_pnl": None,
                    "is_trades": None, "is_max_dd": None,
                    "oos_sharpe": None, "oos_pf": None, "oos_net_pnl": None,
                    "oos_trades": None, "oos_max_dd": None, "oos_win_rate": None,
                    "flag": f"DATA_ERROR:{e}",
                })
            continue

        print(f"  Loaded {len(full_data)} bars total.")

        # full_data index is UTC naive or UTC — normalise
        if full_data.index.tz is None:
            full_data.index = full_data.index.tz_localize("UTC")

        strat  = build_strategy(tick_sz)
        params = build_params(tick_sz)

        for fold in FOLDS:
            f_num = fold["fold"]
            print(f"  Fold {f_num}: IS {fold['is_start']}→{fold['is_end']}  OOS {fold['oos_start']}→{fold['oos_end']}")

            # Slice IS
            is_start = pd.Timestamp(fold["is_start"], tz="UTC")
            is_end   = pd.Timestamp(fold["is_end"],   tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            is_data  = full_data.loc[is_start:is_end].copy()

            # Slice OOS
            oos_start = pd.Timestamp(fold["oos_start"], tz="UTC")
            oos_end   = pd.Timestamp(fold["oos_end"],   tz="UTC") + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            oos_data  = full_data.loc[oos_start:oos_end].copy()

            print(f"    IS bars: {len(is_data)}, OOS bars: {len(oos_data)}")

            # --- IS run ---
            is_m = run_fold(strat, is_data, params)

            # --- OOS run ---
            oos_m = run_fold(strat, oos_data, params)

            flag_parts = []
            if is_m.get("_flag"):  flag_parts.append(f"IS:{is_m['_flag']}")
            if oos_m.get("_flag"): flag_parts.append(f"OOS:{oos_m['_flag']}")
            flag = " | ".join(flag_parts)

            row = {
                "instrument": SYMBOL,
                "strategy":   STRATEGY,
                "fold":       f_num,
                "tick_size":  tick_sz,
                "is_start":   fold["is_start"],
                "is_end":     fold["is_end"],
                "oos_start":  fold["oos_start"],
                "oos_end":    fold["oos_end"],
                # IS stats
                "is_sharpe":  round(is_m["sharpe"],   4),
                "is_pf":      round(is_m["pf"],        4),
                "is_net_pnl": round(is_m["net_pnl"],   2),
                "is_trades":  is_m["trades"],
                "is_max_dd":  round(is_m["max_dd"],    2),
                # OOS stats
                "oos_sharpe":   round(oos_m["sharpe"],   4),
                "oos_pf":       round(oos_m["pf"],        4),
                "oos_net_pnl":  round(oos_m["net_pnl"],   2),
                "oos_trades":   oos_m["trades"],
                "oos_max_dd":   round(oos_m["max_dd"],    2),
                "oos_win_rate": round(oos_m["win_rate"],  4),
                "flag":         flag,
            }
            fold_rows.append(row)
            print(f"    OOS: sharpe={row['oos_sharpe']:.3f}  pf={row['oos_pf']:.3f}  "
                  f"pnl={row['oos_net_pnl']:.0f}  trades={row['oos_trades']}  {flag}")

            # --- Session breakdown (OOS trades only) ---
            oos_trades_df = oos_m.get("_trades_df", pd.DataFrame())
            sess_stats    = session_breakdown(oos_trades_df)

            for sess, stats in sess_stats.items():
                session_rows.append({
                    "instrument": SYMBOL,
                    "strategy":   STRATEGY,
                    "fold":       f_num,
                    "tick_size":  tick_sz,
                    "oos_start":  fold["oos_start"],
                    "oos_end":    fold["oos_end"],
                    "session":    sess,
                    "trades":     stats["trades"],
                    "win_rate":   round(stats["win_rate"],   4),
                    "expectancy": round(stats["expectancy"], 4),
                    "pf":         round(stats["pf"],         4) if math.isfinite(stats["pf"]) else None,
                    "net_pnl":    round(stats["net_pnl"],    2),
                })

    # ---------------------------------------------------------------------------
    # Write fold TSV
    # ---------------------------------------------------------------------------
    fold_tsv = reports_dir / "wf_mobobands_MGC_2026-05-12.tsv"
    fold_df  = pd.DataFrame(fold_rows)
    fold_df.to_csv(fold_tsv, sep="\t", index=False)
    print(f"\nWrote: {fold_tsv}")

    # ---------------------------------------------------------------------------
    # Write session TSV
    # ---------------------------------------------------------------------------
    sess_tsv = reports_dir / "wf_mobobands_MGC_sessions.tsv"
    sess_df  = pd.DataFrame(session_rows)
    sess_df.to_csv(sess_tsv, sep="\t", index=False)
    print(f"Wrote: {sess_tsv}")

    # ---------------------------------------------------------------------------
    # Aggregate OOS scoring per tick_size
    # ---------------------------------------------------------------------------
    summary_rows = []
    for tick_sz in TICK_SIZES:
        sub = fold_df[fold_df["tick_size"] == tick_sz]
        # Exclude rows where OOS trades is None (data errors)
        valid = sub[sub["oos_trades"].notna() & (sub["oos_trades"] >= MIN_TRADES)]

        n_folds      = len(sub)
        n_valid      = len(valid)
        sharpes      = valid["oos_sharpe"].astype(float).tolist()
        mean_sharpe  = float(np.mean(sharpes))  if sharpes else 0.0
        std_sharpe   = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
        composite    = mean_sharpe / (1 + std_sharpe) if sharpes else 0.0
        pct_pos_pnl  = float((valid["oos_net_pnl"] > 0).sum()) / n_valid if n_valid > 0 else 0.0
        total_pnl    = float(valid["oos_net_pnl"].sum())
        total_trades = int(valid["oos_trades"].sum()) if n_valid > 0 else 0

        summary_rows.append({
            "tick_size":    tick_sz,
            "n_folds":      n_folds,
            "n_valid_folds": n_valid,
            "mean_oos_sharpe": round(mean_sharpe, 4),
            "std_oos_sharpe":  round(std_sharpe,  4),
            "composite_score": round(composite,   4),
            "pct_folds_pos_pnl": round(pct_pos_pnl, 3),
            "total_oos_pnl":   round(total_pnl, 2),
            "total_oos_trades": total_trades,
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("composite_score", ascending=False)

    # Best tick size by composite score
    best_row = summary_df.iloc[0] if len(summary_df) > 0 else None

    # Best session across all tick sizes (by total net_pnl among sessions with >=10 trades per fold-slot)
    if len(sess_df) > 0:
        sess_agg = (
            sess_df.groupby(["tick_size", "session"])
            .agg(
                total_trades=("trades", "sum"),
                total_pnl=("net_pnl", "sum"),
                mean_win_rate=("win_rate", "mean"),
                mean_pf=("pf", "mean"),
            )
            .reset_index()
        )
        # Filter low-sample sessions
        sess_agg = sess_agg[sess_agg["total_trades"] >= 10]
        if len(sess_agg) > 0:
            best_sess_row = sess_agg.sort_values("total_pnl", ascending=False).iloc[0]
            best_session  = str(best_sess_row["session"])
            best_sess_ts  = int(best_sess_row["tick_size"])
            best_sess_pnl = float(best_sess_row["total_pnl"])
        else:
            best_session  = "N/A (insufficient trades per session)"
            best_sess_ts  = None
            best_sess_pnl = 0.0
    else:
        best_session  = "N/A"
        best_sess_ts  = None
        best_sess_pnl = 0.0

    # ---------------------------------------------------------------------------
    # Write summary markdown
    # ---------------------------------------------------------------------------
    summary_md = reports_dir / "wf_mobobands_MGC_summary.md"

    edge_found = (
        best_row is not None
        and best_row["n_valid_folds"] >= 2
        and best_row["mean_oos_sharpe"] > 0.3
        and best_row["pct_folds_pos_pnl"] >= 0.5
    )

    lines = [
        "# MoBoBands / MGC Walk-Forward Summary",
        f"**Run date:** 2026-05-12",
        f"**Commission:** ${MGC_COMMISSION:.2f} RT  |  tick_size={MGC_TICK_SIZE}  tick_value=${MGC_TICK_VALUE:.2f}",
        "",
        "## Statistical power caveat",
        "MGC has only **3 folds** (post-gap span 2025-04-01 → 2025-10-31). "
        "Conclusions carry reduced statistical power — treat as directional, not conclusive.",
        "",
        "## OOS aggregate scores by tick size",
        "",
        "| tick_size | valid_folds | mean_oos_sharpe | std_oos_sharpe | composite (mean/(1+std)) | pct_folds_pos_pnl | total_oos_pnl | total_oos_trades |",
        "|-----------|-------------|-----------------|----------------|--------------------------|-------------------|---------------|-----------------|",
    ]
    for _, r in summary_df.iterrows():
        lines.append(
            f"| {int(r['tick_size'])} | {int(r['n_valid_folds'])}/{int(r['n_folds'])} "
            f"| {r['mean_oos_sharpe']:.4f} | {r['std_oos_sharpe']:.4f} "
            f"| {r['composite_score']:.4f} | {r['pct_folds_pos_pnl']:.0%} "
            f"| ${r['total_oos_pnl']:.0f} | {int(r['total_oos_trades'])} |"
        )

    lines += [
        "",
        "## Per-fold detail",
        "",
        "| fold | tick_size | is_trades | oos_trades | oos_sharpe | oos_pf | oos_net_pnl | oos_win_rate | flag |",
        "|------|-----------|-----------|------------|------------|--------|-------------|--------------|------|",
    ]
    for _, r in fold_df.sort_values(["tick_size", "fold"]).iterrows():
        lines.append(
            f"| {int(r['fold'])} | {int(r['tick_size'])} "
            f"| {int(r['is_trades']) if pd.notna(r['is_trades']) else 'N/A'} "
            f"| {int(r['oos_trades']) if pd.notna(r['oos_trades']) else 'N/A'} "
            f"| {r['oos_sharpe'] if pd.notna(r['oos_sharpe']) else 'N/A'} "
            f"| {r['oos_pf'] if pd.notna(r['oos_pf']) else 'N/A'} "
            f"| ${r['oos_net_pnl'] if pd.notna(r['oos_net_pnl']) else 'N/A'} "
            f"| {r['oos_win_rate'] if pd.notna(r['oos_win_rate']) else 'N/A'} "
            f"| {r['flag'] if r['flag'] else '-'} |"
        )

    lines += ["", "## Session breakdown (OOS trades, all folds combined)", ""]
    if len(sess_df) > 0 and "sess_agg" in dir():
        lines += [
            "| tick_size | session | total_trades | total_pnl | mean_win_rate | mean_pf |",
            "|-----------|---------|--------------|-----------|---------------|---------|",
        ]
        for _, r in sess_agg.sort_values(["tick_size", "total_pnl"], ascending=[True, False]).iterrows():
            pf_str = f"{r['mean_pf']:.3f}" if pd.notna(r["mean_pf"]) else "N/A"
            lines.append(
                f"| {int(r['tick_size'])} | {r['session']} | {int(r['total_trades'])} "
                f"| ${r['total_pnl']:.0f} | {r['mean_win_rate']:.1%} | {pf_str} |"
            )

    lines += ["", "## Recommendation", ""]

    if not edge_found:
        lines.append("**No edge found.** No tick size produced mean OOS Sharpe > 0.3 with >= 2 valid folds and >= 50% folds profitable. Do not trade MGC with mobobands at this param baseline.")
    else:
        best_ts = int(best_row["tick_size"])
        lines += [
            f"**Best tick size: {best_ts}**",
            f"- Composite stability score: {best_row['composite_score']:.4f}",
            f"- Mean OOS Sharpe: {best_row['mean_oos_sharpe']:.4f}  (std: {best_row['std_oos_sharpe']:.4f})",
            f"- {int(best_row['pct_folds_pos_pnl']*100)}% of valid folds profitable",
            f"- Total OOS PnL: ${best_row['total_oos_pnl']:.0f}  over {int(best_row['total_oos_trades'])} trades",
            "",
            f"**Best session: {best_session}** (tick_size={best_sess_ts}, total OOS PnL ${best_sess_pnl:.0f})",
            "",
            "Caveat: 3-fold WF is weak evidence. Confirm with param sweep or holdout test before live deployment.",
        ]

    summary_md.write_text("\n".join(lines))
    print(f"Wrote: {summary_md}")

    # Print headline
    print("\n" + "="*60)
    print("HEADLINE")
    print("="*60)
    if not edge_found:
        print("No edge found for MGC/mobobands across tick sizes tested.")
    else:
        print(f"Best tick size: {int(best_row['tick_size'])}  "
              f"composite={best_row['composite_score']:.3f}  "
              f"mean_oos_sharpe={best_row['mean_oos_sharpe']:.3f}")
        print(f"Best session: {best_session} (tick_size={best_sess_ts})")

    print(f"\nOutputs:")
    print(f"  {fold_tsv}")
    print(f"  {sess_tsv}")
    print(f"  {summary_md}")


if __name__ == "__main__":
    main()
