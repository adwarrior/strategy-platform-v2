"""
Trimmed walk-forward tick-size sweep — WaeJurikPro on MNQ ONLY.

Resumes Phase 2 of the tick-size sweep (the only incomplete piece). The full
5-fold run OOM'd on 2026-05-16 at fold 2/233-tick (peak RSS 10.9GB on 11GB box).

Per the resume manifest's OOM mitigation: MNQ is trimmed to **4 folds** and
every tick-size/fold combination is loaded individually (3-month window),
processed, then explicitly freed with gc.collect() before the next load.
Outputs are written incrementally so any crash preserves completed folds.

Reuses all logic from wf_waejurikpro.py; only overrides the MNQ fold list and
output date stamp so it does not clobber the partial 2026-05-13 artifacts.

Usage:
  python3 scripts/wf_waejurikpro_MNQ_trimmed.py
"""

from __future__ import annotations

import gc
import sys
from pathlib import Path

ROOT = Path("/home/ad/strategy-platform-v2")
sys.path.insert(0, str(ROOT))

import wf_waejurikpro as base  # noqa: E402  (reuse all helpers)

# ---------------------------------------------------------------------------
# Trim MNQ to 4 folds (drop fold 5 — the 2025-08→10 window — to cut compute).
# Keeps the four contiguous early folds for the cleanest regime comparison
# against the partial 233-tick result we already have for folds 1-2.
# ---------------------------------------------------------------------------
MNQ_TRIMMED_FOLDS = base.FOLDS_BY_SYMBOL["MNQ"][:4]


def main() -> None:
    symbol = "MNQ"
    base.FOLDS_BY_SYMBOL[symbol] = MNQ_TRIMMED_FOLDS

    meta = base.INSTRUMENT_META.get(symbol)
    if meta is None:
        raise RuntimeError(f"INSTRUMENT_META missing '{symbol}'")
    print(f"{symbol} meta: tick_size={meta['tick_size']}, "
          f"tick_value={meta['tick_value']}, commission_rt={meta['commission']}")
    print(f"Trimmed run: {len(MNQ_TRIMMED_FOLDS)} folds, "
          f"tick sizes {base.TICK_SIZES}\n", flush=True)

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    fold_rows: list = []
    session_rows: list = []

    strat = base.build_strategy(symbol, meta)
    db_host = base.os.getenv("DB_HOST", "192.168.1.228")

    for tick_sz in base.TICK_SIZES:
        print(f"\n{'='*60}\n  Tick size: {tick_sz}  |  symbol: {symbol}\n{'='*60}",
              flush=True)
        for fold in MNQ_TRIMMED_FOLDS:
            print(f"  Loading {fold['is_start']} → {fold['oos_end']} ...", flush=True)
            try:
                full_data = base.load_tick_bars(
                    symbol=symbol, bar_size=tick_sz,
                    start=fold["is_start"], end=fold["oos_end"], host=db_host)
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR: {e}")
                fold_rows.append({
                    "instrument": symbol, "strategy": base.STRATEGY,
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
            full_data = base.normalise_tz(full_data)
            print(f"    Loaded {len(full_data)} bars.", flush=True)
            base.run_one_combo(strat, tick_sz, fold, full_data, symbol,
                               fold_rows, session_rows)
            del full_data
            gc.collect()

        # Incremental write after each tick size so progress survives a crash.
        _write(reports_dir, symbol, fold_rows, session_rows, interim=True)

    _write(reports_dir, symbol, fold_rows, session_rows, interim=False)
    print("\nDone.")


def _write(reports_dir, symbol, fold_rows, session_rows, interim):
    """Write to *_trimmed_* filenames so partial 2026-05-13 artifacts stay intact."""
    import pandas as pd
    fold_tsv = reports_dir / f"wf_waejurikpro_{symbol}_trimmed_2026-06-07.tsv"
    sess_tsv = reports_dir / f"wf_waejurikpro_{symbol}_trimmed_sessions.tsv"
    pd.DataFrame(fold_rows).to_csv(fold_tsv, sep="\t", index=False)
    pd.DataFrame(session_rows).to_csv(sess_tsv, sep="\t", index=False)
    if interim:
        print(f"  [interim write] {len(fold_rows)} fold rows / "
              f"{len(session_rows)} session rows", flush=True)
        return
    # Reuse the base summary writer, but point it at trimmed filenames by
    # temporarily swapping the date stamp it embeds.
    print(f"\nWrote: {fold_tsv}\nWrote: {sess_tsv}")
    _write_summary(reports_dir, symbol, fold_rows, session_rows)


def _write_summary(reports_dir, symbol, fold_rows, session_rows):
    """Trimmed-aware summary — same scoring as base._write_outputs."""
    import math
    import numpy as np
    import pandas as pd
    fold_df = pd.DataFrame(fold_rows)
    sess_df = pd.DataFrame(session_rows)
    MIN_TRADES = base.MIN_TRADES
    TICK_SIZES = base.TICK_SIZES

    summary_rows = []
    for tick_sz in TICK_SIZES:
        sub = fold_df[fold_df["tick_size"] == tick_sz]
        valid = sub[sub["oos_trades"].notna() & (sub["oos_trades"] >= MIN_TRADES)]
        sharpes = valid["oos_sharpe"].astype(float).tolist()
        mean_sharpe = float(np.mean(sharpes)) if sharpes else 0.0
        std_sharpe = float(np.std(sharpes, ddof=1)) if len(sharpes) > 1 else 0.0
        composite = mean_sharpe / (1 + std_sharpe) if sharpes else 0.0
        n_valid = len(valid)
        pct_pos = float((valid["oos_net_pnl"] > 0).sum()) / n_valid if n_valid else 0.0
        summary_rows.append({
            "tick_size": tick_sz, "n_folds": len(sub), "n_valid_folds": n_valid,
            "mean_oos_sharpe": round(mean_sharpe, 4),
            "std_oos_sharpe": round(std_sharpe, 4),
            "composite_score": round(composite, 4),
            "pct_folds_pos_pnl": round(pct_pos, 3),
            "total_oos_pnl": round(float(valid["oos_net_pnl"].sum()), 2),
            "total_oos_trades": int(valid["oos_trades"].sum()) if n_valid else 0,
        })
    summary_df = pd.DataFrame(summary_rows).sort_values("composite_score", ascending=False)
    best = summary_df.iloc[0] if len(summary_df) else None
    edge = (best is not None and best["n_valid_folds"] >= 2
            and best["mean_oos_sharpe"] > 0.3 and best["pct_folds_pos_pnl"] >= 0.5)

    out = [
        f"# WaeJurikPro / {symbol} Walk-Forward Summary (TRIMMED — resumed)",
        "**Run date:** 2026-06-07",
        f"**Folds:** {len(base.FOLDS_BY_SYMBOL[symbol])} (trimmed from 5 to avoid OOM)  "
        f"|  Tick sizes: {TICK_SIZES}",
        "**Baseline:** all filters off (jurik band/slope, time = False)",
        "**Context:** Completes Phase 2 of the tick-size sweep. The full 5-fold "
        "run OOM'd 2026-05-16 at fold 2/233-tick (folds 1-2 lost money).",
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
            f"| ${r['total_oos_pnl']:.0f} | {int(r['total_oos_trades'])} |")

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
            f"| {r['flag'] if r['flag'] else '-'} |")

    out += ["", "## Recommendation", ""]
    if not edge:
        out.append(
            f"**No edge found.** No tick size produced mean OOS Sharpe > 0.3 with "
            f"≥ 2 valid folds and ≥ 50% folds profitable. Confirms the partial-run "
            f"and mobobands-MNQ priors: do not trade {symbol} with waejurikpro at "
            f"this baseline.")
    else:
        bt = int(best["tick_size"])
        out += [
            f"**Best tick size: {bt}**",
            f"- Composite: {best['composite_score']:.4f}",
            f"- Mean OOS Sharpe: {best['mean_oos_sharpe']:.4f} (std {best['std_oos_sharpe']:.4f})",
            f"- {int(best['pct_folds_pos_pnl']*100)}% of valid folds profitable",
            f"- Total OOS PnL: ${best['total_oos_pnl']:.0f} over {int(best['total_oos_trades'])} trades",
        ]
    summary_md = reports_dir / f"wf_waejurikpro_{symbol}_trimmed_summary.md"
    summary_md.write_text("\n".join(out))
    print(f"Wrote: {summary_md}")
    print("\n" + "=" * 60 + f"\nHEADLINE — {symbol} (trimmed)\n" + "=" * 60)
    if not edge:
        print(f"No edge found for {symbol}/waejurikpro across tick sizes tested.")
    else:
        print(f"Best tick size: {int(best['tick_size'])}  "
              f"composite={best['composite_score']:.3f}")


if __name__ == "__main__":
    main()
