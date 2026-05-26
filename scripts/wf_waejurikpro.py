"""
Walk-forward tick-size sweep — WaeJurikPro on MCL/MGC/MNQ.

Usage:
  python3 scripts/wf_waejurikpro.py --symbol MCL
  python3 scripts/wf_waejurikpro.py --symbol MGC
  python3 scripts/wf_waejurikpro.py --symbol MNQ

Mirrors the mobobands sweep:
  - 5 tick sizes (233, 377, 512, 610, 987)
  - Same fold structure (IS=2mo / OOS=1mo, rolling)
  - All filters off (baseline only — tick size is the sweep dimension)
  - Per-fold loading on MNQ (avoid OOM from full-span loading)
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
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
from strategy_platform.strategies.waejurikpro.strategy import WaeJurikPro

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STRATEGY    = "waejurikpro"
TICK_SIZES  = [233, 377, 512, 610, 987]
MIN_TRADES  = 30

# Per-symbol WF fold definitions
FOLDS_BY_SYMBOL: Dict[str, List[Dict[str, Any]]] = {
    "MCL": [
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
    ],
    "MGC": [
        {"fold": 1, "is_start": "2025-04-01", "is_end": "2025-05-31",
                   "oos_start": "2025-06-01", "oos_end": "2025-06-30"},
        {"fold": 2, "is_start": "2025-06-01", "is_end": "2025-07-31",
                   "oos_start": "2025-08-01", "oos_end": "2025-08-31"},
        {"fold": 3, "is_start": "2025-08-01", "is_end": "2025-09-30",
                   "oos_start": "2025-10-01", "oos_end": "2025-10-31"},
    ],
    "MNQ": [
        {"fold": 1, "is_start": "2024-10-01", "is_end": "2024-11-30",
                   "oos_start": "2024-12-01", "oos_end": "2024-12-31"},
        {"fold": 2, "is_start": "2024-12-01", "is_end": "2025-01-31",
                   "oos_start": "2025-02-01", "oos_end": "2025-02-28"},
        {"fold": 3, "is_start": "2025-02-01", "is_end": "2025-03-31",
                   "oos_start": "2025-04-01", "oos_end": "2025-04-30"},
        {"fold": 4, "is_start": "2025-04-01", "is_end": "2025-05-31",
                   "oos_start": "2025-06-01", "oos_end": "2025-06-30"},
        {"fold": 5, "is_start": "2025-08-01", "is_end": "2025-09-30",
                   "oos_start": "2025-10-01", "oos_end": "2025-10-31"},
    ],
}

# Symbols that should use per-fold loading (large datasets)
PER_FOLD_LOAD_SYMBOLS = {"MNQ"}

# Baseline params — merge over default_params; all filters off
SWEEP_PARAMS_OVERRIDE: Dict[str, Any] = {
    # Trade
    "profit_ticks":         40,
    "stop_ticks":           20,
    "bars_between_trades":  2,
    "enable_longs":         True,
    "enable_shorts":        True,
    # Time filter OFF — we want 24h
    "enable_time_filter":   False,
    # Jurik filter OFF — baseline test of WAE alone
    "enable_jurik_band_filter":  False,
    "enable_jurik_slope_filter": False,
    # calculate_mode bar close (data already aggregated)
    "calculate_mode":       "on_bar_close",
}

SESSION_LABELS = ["Asia", "London", "NY_AM", "NY_PM", "Globex"]


def session_for_hour(hour_et: int) -> str:
    if 2 <= hour_et < 8:
        return "London"
    if 8 <= hour_et < 12:
        return "NY_AM"
    if 12 <= hour_et < 16:
        return "NY_PM"
    if 16 <= hour_et < 19:
        return "Globex"
    return "Asia"


def compute_metrics(result: Dict[str, Any]) -> Dict[str, float]:
    trades_val = result.get("total_trades", result.get("trades", 0))
    if isinstance(trades_val, pd.DataFrame):
        n_trades = len(trades_val)
    else:
        n_trades = int(trades_val)
    return {
        "sharpe":    float(result.get("sharpe", 0.0)),
        "pf":        float(result.get("profit_factor", 0.0)),
        "net_pnl":   float(result.get("net_pnl", 0.0)),
        "trades":    n_trades,
        "max_dd":    float(result.get("max_drawdown", 0.0)),
        "win_rate":  float(result.get("win_rate", 0.0)),
    }


def session_breakdown(trades_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict] = {s: {"trades": 0, "wins": 0, "net_pnl": 0.0,
                                 "gross_profit": 0.0, "gross_loss": 0.0}
                             for s in SESSION_LABELS}
    if trades_df is None or len(trades_df) == 0 or "entry_time" not in trades_df.columns:
        return _finalise_sessions(out)

    # entry_time inherits the strategy's bar index, which is ET-naive
    # (loader returns ET-naive for emini MNQ/MES/MGC). Use as-is.
    _et_src = trades_df["entry_time"]
    if _et_src.dt.tz is not None:
        _et_src = _et_src.dt.tz_convert("US/Eastern").dt.tz_localize(None)
    et_times = _et_src
    pnl_col = "pnl" if "pnl" in trades_df.columns else "net_pnl"
    if pnl_col not in trades_df.columns:
        return _finalise_sessions(out)

    for i, row in trades_df.iterrows():
        hour = et_times.iloc[i].hour
        sess = session_for_hour(hour)
        pnl = float(row[pnl_col])
        out[sess]["trades"]  += 1
        out[sess]["net_pnl"] += pnl
        if pnl > 0:
            out[sess]["wins"]         += 1
            out[sess]["gross_profit"] += pnl
        else:
            out[sess]["gross_loss"]   += pnl
    return _finalise_sessions(out)


def _finalise_sessions(out: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    for s in SESSION_LABELS:
        b = out[s]
        n = b["trades"]
        b["win_rate"]   = b["wins"] / n if n > 0 else 0.0
        b["expectancy"] = b["net_pnl"] / n if n > 0 else 0.0
        gl = abs(b["gross_loss"])
        b["pf"] = b["gross_profit"] / gl if gl > 0 else (float("inf") if b["gross_profit"] > 0 else 0.0)
    return out


def build_strategy(symbol: str, meta: Dict[str, Any]) -> WaeJurikPro:
    s = WaeJurikPro()
    s.tick_size     = meta["tick_size"]
    s.tick_value    = meta["tick_value"]
    s.commission_rt = meta["commission"]
    s.symbol        = symbol
    s.bar_type      = "tick"
    return s


def build_params(tick_bar_size: int, symbol: str) -> Dict[str, Any]:
    strat = WaeJurikPro()
    params = dict(strat.default_params)
    params.update(SWEEP_PARAMS_OVERRIDE)
    params["tick_bar_size"] = tick_bar_size
    params["_symbol"]       = symbol
    return params


def run_fold(strat: WaeJurikPro, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
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
                "_trades_df": pd.DataFrame(), "_flag": f"ERROR:{type(e).__name__}:{e}"}
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


def normalise_tz(df: pd.DataFrame) -> pd.DataFrame:
    """Strip any stray tz info — loader returns ET-naive for emini data."""
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    return df


def run_one_combo(strat, tick_sz, fold, full_data, symbol, fold_rows, session_rows):
    """Run IS+OOS for a single fold; append rows to lists."""
    f_num = fold["fold"]
    params = build_params(tick_sz, symbol)
    print(f"  Fold {f_num}: IS {fold['is_start']}→{fold['is_end']}  OOS {fold['oos_start']}→{fold['oos_end']}",
          flush=True)

    # Slice bounds in ET-naive (matches loader's ET-naive bar index)
    is_start = pd.Timestamp(fold["is_start"])
    is_end   = pd.Timestamp(fold["is_end"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    oos_start = pd.Timestamp(fold["oos_start"])
    oos_end   = pd.Timestamp(fold["oos_end"]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    is_data  = full_data.loc[is_start:is_end].copy()
    oos_data = full_data.loc[oos_start:oos_end].copy()
    print(f"    IS bars: {len(is_data)}, OOS bars: {len(oos_data)}", flush=True)

    is_m  = run_fold(strat, is_data,  params)
    oos_m = run_fold(strat, oos_data, params)

    flag_parts = []
    if is_m.get("_flag"):  flag_parts.append(f"IS:{is_m['_flag']}")
    if oos_m.get("_flag"): flag_parts.append(f"OOS:{oos_m['_flag']}")
    flag = " | ".join(flag_parts)

    row = {
        "instrument": symbol, "strategy": STRATEGY,
        "fold": f_num, "tick_size": tick_sz,
        "is_start": fold["is_start"], "is_end": fold["is_end"],
        "oos_start": fold["oos_start"], "oos_end": fold["oos_end"],
        "is_sharpe":  round(is_m["sharpe"], 4),
        "is_pf":      round(is_m["pf"], 4),
        "is_net_pnl": round(is_m["net_pnl"], 2),
        "is_trades":  is_m["trades"],
        "is_max_dd":  round(is_m["max_dd"], 2),
        "oos_sharpe":   round(oos_m["sharpe"], 4),
        "oos_pf":       round(oos_m["pf"], 4),
        "oos_net_pnl":  round(oos_m["net_pnl"], 2),
        "oos_trades":   oos_m["trades"],
        "oos_max_dd":   round(oos_m["max_dd"], 2),
        "oos_win_rate": round(oos_m["win_rate"], 4),
        "flag":         flag,
    }
    fold_rows.append(row)
    print(f"    OOS: sharpe={row['oos_sharpe']:.3f}  pf={row['oos_pf']:.3f}  "
          f"pnl={row['oos_net_pnl']:.0f}  trades={row['oos_trades']}  {flag}", flush=True)

    sess_stats = session_breakdown(oos_m.get("_trades_df", pd.DataFrame()))
    for sess, stats in sess_stats.items():
        session_rows.append({
            "instrument": symbol, "strategy": STRATEGY,
            "fold": f_num, "tick_size": tick_sz,
            "oos_start": fold["oos_start"], "oos_end": fold["oos_end"],
            "session": sess, "trades": stats["trades"],
            "win_rate": round(stats["win_rate"], 4),
            "expectancy": round(stats["expectancy"], 4),
            "pf": round(stats["pf"], 4) if math.isfinite(stats["pf"]) else None,
            "net_pnl": round(stats["net_pnl"], 2),
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, choices=["MCL", "MGC", "MNQ"])
    args = ap.parse_args()
    symbol = args.symbol

    folds = FOLDS_BY_SYMBOL[symbol]
    meta  = INSTRUMENT_META.get(symbol)
    if meta is None:
        raise RuntimeError(f"INSTRUMENT_META missing '{symbol}'")
    print(f"{symbol} meta: tick_size={meta['tick_size']}, tick_value={meta['tick_value']}, "
          f"commission_rt={meta['commission']}")

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    fold_rows:    List[Dict] = []
    session_rows: List[Dict] = []

    use_per_fold = symbol in PER_FOLD_LOAD_SYMBOLS

    strat = build_strategy(symbol, meta)
    db_host = os.getenv("DB_HOST", "192.168.1.228")

    for tick_sz in TICK_SIZES:
        print(f"\n{'='*60}")
        print(f"  Tick size: {tick_sz}  |  symbol: {symbol}")
        print(f"{'='*60}", flush=True)

        if use_per_fold:
            # Load per-fold (3mo window) to avoid OOM
            for fold in folds:
                print(f"  Loading {fold['is_start']} → {fold['oos_end']} ...", flush=True)
                try:
                    full_data = load_tick_bars(symbol=symbol, bar_size=tick_sz,
                                                start=fold["is_start"], end=fold["oos_end"],
                                                host=db_host)
                except Exception as e:
                    print(f"  ERROR: {e}")
                    fold_rows.append({
                        "instrument": symbol, "strategy": STRATEGY,
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
                full_data = normalise_tz(full_data)
                print(f"    Loaded {len(full_data)} bars.", flush=True)
                run_one_combo(strat, tick_sz, fold, full_data, symbol, fold_rows, session_rows)
                del full_data
                gc.collect()
        else:
            # Single load for full span, slice per fold
            full_start = folds[0]["is_start"]
            full_end   = folds[-1]["oos_end"]
            print(f"  Loading {full_start} → {full_end} ...", flush=True)
            try:
                full_data = load_tick_bars(symbol=symbol, bar_size=tick_sz,
                                            start=full_start, end=full_end, host=db_host)
            except Exception as e:
                print(f"  ERROR: {e}")
                for fold in folds:
                    fold_rows.append({
                        "instrument": symbol, "strategy": STRATEGY,
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
            full_data = normalise_tz(full_data)
            print(f"  Loaded {len(full_data)} bars.", flush=True)
            for fold in folds:
                run_one_combo(strat, tick_sz, fold, full_data, symbol, fold_rows, session_rows)
            del full_data
            gc.collect()

        # Write incrementally so progress survives OOM
        _write_outputs(reports_dir, symbol, fold_rows, session_rows, interim=True)

    # Final writes + summary
    _write_outputs(reports_dir, symbol, fold_rows, session_rows, interim=False)
    print("\nDone.")


def _write_outputs(reports_dir, symbol, fold_rows, session_rows, interim):
    fold_tsv = reports_dir / f"wf_waejurikpro_{symbol}_2026-05-13.tsv"
    sess_tsv = reports_dir / f"wf_waejurikpro_{symbol}_sessions.tsv"
    fold_df = pd.DataFrame(fold_rows)
    sess_df = pd.DataFrame(session_rows)
    fold_df.to_csv(fold_tsv, sep="\t", index=False)
    sess_df.to_csv(sess_tsv, sep="\t", index=False)
    if interim:
        print(f"  [interim write] {len(fold_df)} fold rows / {len(sess_df)} session rows", flush=True)
        return
    print(f"\nWrote: {fold_tsv}")
    print(f"Wrote: {sess_tsv}")

    # Aggregate summary
    summary_rows = []
    for tick_sz in TICK_SIZES:
        sub = fold_df[fold_df["tick_size"] == tick_sz]
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
            "tick_size": tick_sz, "n_folds": n_folds, "n_valid_folds": n_valid,
            "mean_oos_sharpe": round(mean_sharpe, 4),
            "std_oos_sharpe":  round(std_sharpe, 4),
            "composite_score": round(composite, 4),
            "pct_folds_pos_pnl": round(pct_pos_pnl, 3),
            "total_oos_pnl":   round(total_pnl, 2),
            "total_oos_trades": total_trades,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("composite_score", ascending=False)
    best_row = summary_df.iloc[0] if len(summary_df) > 0 else None

    edge_found = (
        best_row is not None
        and best_row["n_valid_folds"] >= 2
        and best_row["mean_oos_sharpe"] > 0.3
        and best_row["pct_folds_pos_pnl"] >= 0.5
    )

    # Best session aggregate
    best_session, best_sess_ts, best_sess_pnl = "N/A", None, 0.0
    sess_agg = None
    if len(sess_df) > 0:
        sess_agg = (sess_df.groupby(["tick_size", "session"])
                    .agg(total_trades=("trades", "sum"),
                         total_pnl=("net_pnl", "sum"),
                         mean_win_rate=("win_rate", "mean"),
                         mean_pf=("pf", "mean"))
                    .reset_index())
        sess_agg = sess_agg[sess_agg["total_trades"] >= 10]
        if len(sess_agg) > 0:
            r = sess_agg.sort_values("total_pnl", ascending=False).iloc[0]
            best_session  = str(r["session"])
            best_sess_ts  = int(r["tick_size"])
            best_sess_pnl = float(r["total_pnl"])

    summary_md = reports_dir / f"wf_waejurikpro_{symbol}_summary.md"
    out: List[str] = [
        f"# WaeJurikPro / {symbol} Walk-Forward Summary",
        f"**Run date:** 2026-05-13",
        f"**Commission:** ${INSTRUMENT_META[symbol]['commission']:.2f} RT  "
        f"|  tick_size={INSTRUMENT_META[symbol]['tick_size']}  "
        f"tick_value=${INSTRUMENT_META[symbol]['tick_value']:.2f}",
        f"**Folds:** {len(FOLDS_BY_SYMBOL[symbol])}  "
        f"|  Tick sizes tested: {TICK_SIZES}",
        f"**Baseline:** all filters off — `enable_jurik_band_filter`, "
        f"`enable_jurik_slope_filter`, `enable_time_filter` = False",
        "",
        "## OOS aggregate scores by tick size",
        "",
        "| tick_size | valid_folds | mean_oos_sharpe | std_oos_sharpe | composite | pct_folds_pos_pnl | total_oos_pnl | total_oos_trades |",
        "|-----------|-------------|-----------------|----------------|-----------|-------------------|---------------|-----------------|",
    ]
    for _, r in summary_df.iterrows():
        out.append(
            f"| {int(r['tick_size'])} | {int(r['n_valid_folds'])}/{int(r['n_folds'])} "
            f"| {r['mean_oos_sharpe']:.4f} | {r['std_oos_sharpe']:.4f} "
            f"| {r['composite_score']:.4f} | {r['pct_folds_pos_pnl']:.0%} "
            f"| ${r['total_oos_pnl']:.0f} | {int(r['total_oos_trades'])} |"
        )

    out += ["", "## Per-fold detail", "",
            "| fold | tick_size | is_trades | oos_trades | oos_sharpe | oos_pf | oos_net_pnl | oos_win_rate | flag |",
            "|------|-----------|-----------|------------|------------|--------|-------------|--------------|------|"]
    for _, r in fold_df.sort_values(["tick_size", "fold"]).iterrows():
        out.append(
            f"| {int(r['fold'])} | {int(r['tick_size'])} "
            f"| {int(r['is_trades']) if pd.notna(r['is_trades']) else 'N/A'} "
            f"| {int(r['oos_trades']) if pd.notna(r['oos_trades']) else 'N/A'} "
            f"| {r['oos_sharpe'] if pd.notna(r['oos_sharpe']) else 'N/A'} "
            f"| {r['oos_pf'] if pd.notna(r['oos_pf']) else 'N/A'} "
            f"| ${r['oos_net_pnl'] if pd.notna(r['oos_net_pnl']) else 'N/A'} "
            f"| {r['oos_win_rate'] if pd.notna(r['oos_win_rate']) else 'N/A'} "
            f"| {r['flag'] if r['flag'] else '-'} |"
        )

    out += ["", "## Session breakdown (OOS, all folds combined, ≥10 trades)", ""]
    if sess_agg is not None and len(sess_agg) > 0:
        out += ["| tick_size | session | total_trades | total_pnl | mean_win_rate | mean_pf |",
                "|-----------|---------|--------------|-----------|---------------|---------|"]
        for _, r in sess_agg.sort_values(["tick_size", "total_pnl"], ascending=[True, False]).iterrows():
            pf_str = f"{r['mean_pf']:.3f}" if pd.notna(r["mean_pf"]) else "N/A"
            out.append(
                f"| {int(r['tick_size'])} | {r['session']} | {int(r['total_trades'])} "
                f"| ${r['total_pnl']:.0f} | {r['mean_win_rate']:.1%} | {pf_str} |"
            )

    out += ["", "## Recommendation", ""]
    if not edge_found:
        out.append(f"**No edge found.** No tick size produced mean OOS Sharpe > 0.3 with "
                   f"≥ 2 valid folds and ≥ 50% folds profitable. Do not trade {symbol} with "
                   f"waejurikpro at this baseline.")
    else:
        bt = int(best_row["tick_size"])
        out += [
            f"**Best tick size: {bt}**",
            f"- Composite stability score: {best_row['composite_score']:.4f}",
            f"- Mean OOS Sharpe: {best_row['mean_oos_sharpe']:.4f}  (std {best_row['std_oos_sharpe']:.4f})",
            f"- {int(best_row['pct_folds_pos_pnl']*100)}% of valid folds profitable",
            f"- Total OOS PnL: ${best_row['total_oos_pnl']:.0f}  over {int(best_row['total_oos_trades'])} trades",
            "",
            f"**Best session: {best_session}** (tick_size={best_sess_ts}, "
            f"total OOS PnL ${best_sess_pnl:.0f})",
        ]
    summary_md.write_text("\n".join(out))
    print(f"Wrote: {summary_md}")

    print("\n" + "="*60)
    print(f"HEADLINE — {symbol}")
    print("="*60)
    if not edge_found:
        print(f"No edge found for {symbol}/waejurikpro across tick sizes tested.")
    else:
        print(f"Best tick size: {int(best_row['tick_size'])}  "
              f"composite={best_row['composite_score']:.3f}  "
              f"mean_oos_sharpe={best_row['mean_oos_sharpe']:.3f}")
        print(f"Best session: {best_session} (tick_size={best_sess_ts})")


if __name__ == "__main__":
    main()
