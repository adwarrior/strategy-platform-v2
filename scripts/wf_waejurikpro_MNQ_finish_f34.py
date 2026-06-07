"""
Finish WaeJurik MNQ trimmed sweep — Folds 3 & 4 only.

Folds 1-2 completed and are saved in:
  reports/wf_waejurikpro_MNQ_trimmed_2026-06-07.tsv
  reports/wf_waejurikpro_MNQ_trimmed_sessions.tsv

The pull-once design OOM'd on Fold 3's load_ticks_raw (holds the full ~30M-tick
frame in RAM → 9.6GB, oom-killed on the 11GB box). For the remaining 2 folds we
fall back to the memory-SAFE streaming loader (load_tick_bars streams 500k-tick
chunks and discards them), one tick size at a time. Slower (re-pulls per tick
size) but bounded memory. Results append to the existing TSVs, then the full
summary is regenerated from all 4 folds.

Convention: keep the UTC-naive bar index (matches the rest of the sweep — see
wf_waejurikpro_MNQ_trimmed.py for the tz note).

Usage:
  python3 scripts/wf_waejurikpro_MNQ_finish_f34.py
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path("/home/ad/strategy-platform-v2")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import wf_waejurikpro as base
import wf_waejurikpro_MNQ_trimmed as trim

REPORTS = ROOT / "reports"
FOLD_TSV = REPORTS / "wf_waejurikpro_MNQ_trimmed_2026-06-07.tsv"
SESS_TSV = REPORTS / "wf_waejurikpro_MNQ_trimmed_sessions.tsv"

REMAINING_FOLDS = base.FOLDS_BY_SYMBOL["MNQ"][2:4]  # folds 3 and 4


def _load_bars_monthly(symbol, tick_sz, start, end, host):
    """Load tick bars one calendar month at a time and concat.

    Fold 3 spans 46M ticks (Apr 2025 alone = 18.5M) — a single load_tick_bars
    call buffers enough to OOM the 11GB box. Per-month loads cap peak RAM at
    ~4GB. Minor caveat: the fixed-N tick counter restarts each month, so a
    handful of bars at month boundaries differ from one continuous load. Out of
    ~100k bars this is negligible and does not affect the edge verdict.
    """
    months = pd.date_range(pd.Timestamp(start).replace(day=1),
                           pd.Timestamp(end), freq="MS")
    pieces = []
    for ms in months:
        m_start = max(pd.Timestamp(start), ms)
        m_end = min(pd.Timestamp(end), ms + pd.offsets.MonthEnd(1))
        df = base.load_tick_bars(symbol=symbol, bar_size=tick_sz,
                                 start=str(m_start.date()),
                                 end=str(m_end.date()), host=host)
        if len(df):
            pieces.append(df)
        del df
        gc.collect()
    if not pieces:
        return pd.DataFrame(columns=["open", "high", "low", "close",
                                     "volume", "tick_count"])
    out = pd.concat(pieces).sort_index()
    return out


def main() -> None:
    symbol = "MNQ"
    # Restore the 4-fold list so the summary writer reports all four.
    base.FOLDS_BY_SYMBOL[symbol] = base.FOLDS_BY_SYMBOL[symbol][:4]
    meta = base.INSTRUMENT_META[symbol]
    strat = base.build_strategy(symbol, meta)
    host = os.getenv("DB_HOST", "192.168.1.228")

    # Load existing folds 1-2 so the final summary covers all four.
    fold_rows = pd.read_csv(FOLD_TSV, sep="\t").to_dict("records")
    session_rows = pd.read_csv(SESS_TSV, sep="\t").to_dict("records")
    print(f"Loaded {len(fold_rows)} existing fold rows (folds 1-2).", flush=True)

    for fold in REMAINING_FOLDS:
        print(f"\n{'='*60}\n  Fold {fold['fold']} (streaming, memory-safe)\n{'='*60}",
              flush=True)
        for tick_sz in base.TICK_SIZES:
            print(f"  tick_size={tick_sz}: loading {fold['is_start']}→{fold['oos_end']} "
                  f"in monthly chunks...", flush=True)
            try:
                bars = _load_bars_monthly(symbol, tick_sz, fold["is_start"],
                                          fold["oos_end"], host)
            except Exception as e:  # noqa: BLE001
                print(f"    DATA_ERROR: {e}", flush=True)
                fold_rows.append(trim._data_error_row(symbol, fold, tick_sz, e))
                continue
            # load_tick_bars returns UTC-naive — keep as-is (sweep convention).
            base.run_one_combo(strat, tick_sz, fold, bars, symbol,
                               fold_rows, session_rows)
            del bars
            gc.collect()
        # Incremental save after each fold.
        pd.DataFrame(fold_rows).to_csv(FOLD_TSV, sep="\t", index=False)
        pd.DataFrame(session_rows).to_csv(SESS_TSV, sep="\t", index=False)
        print(f"  [saved] {len(fold_rows)} fold rows", flush=True)

    # Final summary across all 4 folds.
    trim._write_summary(REPORTS, symbol, fold_rows, session_rows)
    print("\nDone — all 4 folds complete.")


if __name__ == "__main__":
    main()
