#!/usr/bin/env python3
"""Two-tier trade-by-trade parity harness for NinjaTrader ↔ platform ports.

Tier 1: Python run on platform MySQL data vs the NT trade log (same data).
Tier 2: Python run on NT's own native export vs the NT trade log (pure logic).

Pure helpers are unit-tested with synthetic inputs; parity() orchestrates a
full check and writes a PARITY_REPORT.md.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import pandas as pd

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _money(s: str) -> float:
    """'-$18.98' -> -18.98 ; '$1,126.02' -> 1126.02."""
    s = str(s).strip().replace("$", "").replace(",", "")
    return float(s) if s not in ("", "-") else 0.0


def parse_nt_trade_log(path: str) -> pd.DataFrame:
    """Parse an NT per-trade CSV export into a normalized trades frame.

    Returns columns: entry_time, exit_time, direction, entry_price,
    exit_price, pnl. NT dates are day-first DD/MM/YYYY.
    """
    raw = pd.read_csv(path)
    out = pd.DataFrame({
        "entry_time": pd.to_datetime(raw["Entry time"], dayfirst=True),
        "exit_time": pd.to_datetime(raw["Exit time"], dayfirst=True),
        "direction": raw["Market pos."].astype(str).str.strip(),
        "entry_price": raw["Entry price"].astype(float),
        "exit_price": raw["Exit price"].astype(float),
        "pnl": raw["Profit"].map(_money),
    })
    return out


def _utc_to_et_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Localize a naive UTC index to UTC, convert to ET, drop tz (ET-naive)."""
    return idx.tz_localize("UTC").tz_convert("America/New_York").tz_localize(None)


def parse_nt_ohlc_export(path: str) -> pd.DataFrame:
    """Parse NT 1-min OHLC export 'YYYYMMDD HHMMSS;O;H;L;C;V' (UTC) to ET-naive."""
    df = pd.read_csv(path, sep=";", header=None,
                     names=["ts", "open", "high", "low", "close", "volume"],
                     dtype={"ts": str})
    idx = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S")
    df.index = _utc_to_et_naive(pd.DatetimeIndex(idx))
    return df[["open", "high", "low", "close", "volume"]].astype(float)


def parse_nt_tick_export(path: str) -> pd.DataFrame:
    """Parse NT tick export 'YYYYMMDD HHMMSS<frac>;price;...;volume' (UTC) to ET-naive.

    The timestamp field is 'YYYYMMDD HHMMSS NNNNNNN' (space-separated subsecond
    fraction in 0.1-microsecond units). price is the first numeric after ';';
    volume is the last field.
    """
    rows = []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(";")
            ts_tokens = parts[0].split()              # ['YYYYMMDD', 'HHMMSS', 'FRACTION']
            ymd, hms = ts_tokens[0], ts_tokens[1]
            frac = ts_tokens[2] if len(ts_tokens) > 2 else "0"
            base = pd.to_datetime(ymd + " " + hms, format="%Y%m%d %H%M%S")
            ts = base + pd.to_timedelta(int(frac or 0) * 100, unit="ns")
            price = float(parts[1])
            vol = float(parts[-1]) if parts[-1].strip() else 0.0
            rows.append((ts, price, vol))
    out = pd.DataFrame(rows, columns=["ts", "price", "volume"]).set_index("ts")
    out.index = _utc_to_et_naive(pd.DatetimeIndex(out.index))
    return out[["price", "volume"]]


def ticks_to_bars(ticks: pd.DataFrame, bar_size: int) -> pd.DataFrame:
    """Aggregate a tick frame (price, volume; time index) into N-tick OHLCV bars.

    Each consecutive block of `bar_size` ticks becomes one bar; the bar's index
    is its first tick's timestamp. A trailing partial block is dropped (matches
    NT, which only forms complete bars).
    """
    n = len(ticks) // bar_size
    bars = []
    times = []
    p = ticks["price"].to_numpy()
    v = ticks["volume"].to_numpy()
    t = ticks.index.to_numpy()
    for i in range(n):
        block = p[i * bar_size:(i + 1) * bar_size]
        vblock = v[i * bar_size:(i + 1) * bar_size]
        bars.append((block[0], block.max(), block.min(), block[-1], vblock.sum()))
        times.append(t[i * bar_size])
    return pd.DataFrame(bars, index=pd.DatetimeIndex(times),
                        columns=["open", "high", "low", "close", "volume"])


def _norm_direction(df: pd.DataFrame) -> pd.Series:
    if "direction" in df.columns:
        return df["direction"].astype(str).str.strip()
    return df["side"].astype(str).str.strip()


def match_trades(nt: pd.DataFrame, py: pd.DataFrame,
                 timeframe_min: Optional[int], time_window_s: int,
                 price_tol: float) -> dict:
    """Pair Python trades to NT trades. See module docstring for the match key."""
    nt = nt.reset_index(drop=True).copy()
    py = py.reset_index(drop=True).copy()
    nt["_dir"] = _norm_direction(nt)
    py["_dir"] = _norm_direction(py)

    if timeframe_min is not None:
        freq = f"{timeframe_min}min"
        py["_key_time"] = py["entry_time"].dt.ceil(freq) - pd.Timedelta(minutes=timeframe_min)
    matched_rows, used_py = [], set()
    nt_only_idx = []
    for i, ntr in nt.iterrows():
        hit = None
        if timeframe_min is not None:
            # Time-bar branch: exact key-time match, first found
            for j, pyr in py.iterrows():
                if j in used_py or pyr["_dir"] != ntr["_dir"]:
                    continue
                if pyr["_key_time"] == ntr["entry_time"]:
                    hit = j
                    break
        else:
            # Tick-bar branch: collect ALL candidates, pick NEAREST in time
            best_dt = None
            for j, pyr in py.iterrows():
                if j in used_py or pyr["_dir"] != ntr["_dir"]:
                    continue
                dt = abs((pyr["entry_time"] - ntr["entry_time"]).total_seconds())
                if dt <= time_window_s and abs(pyr["entry_price"] - ntr["entry_price"]) <= price_tol:
                    if best_dt is None or dt < best_dt:
                        best_dt = dt
                        hit = j
        if hit is None:
            nt_only_idx.append(i)
        else:
            used_py.add(hit)
            pyr = py.loc[hit]
            matched_rows.append({
                "nt_entry_time": ntr["entry_time"], "py_entry_time": pyr["entry_time"],
                "direction": ntr["_dir"],
                "nt_entry_price": ntr["entry_price"], "py_entry_price": pyr["entry_price"],
                "nt_exit_price": ntr["exit_price"], "py_exit_price": pyr["exit_price"],
            })

    _scratch = [c for c in ("_dir", "_key_time") if c in nt.columns]
    nt_only = nt.loc[nt_only_idx].drop(columns=_scratch).reset_index(drop=True)

    py_only_idx = [j for j in py.index if j not in used_py]
    py_only = py.loc[py_only_idx].drop(
        columns=[c for c in ("_dir", "_key_time") if c in py.columns]
    ).reset_index(drop=True)

    return {
        "matched": pd.DataFrame(matched_rows),
        "nt_only": nt_only,
        "py_only": py_only,
    }


def preflight_guards(matched: pd.DataFrame, nt: pd.DataFrame, py: pd.DataFrame) -> list:
    warns = []
    # (a) contract-series: large + low-variance entry-price deltas across matches
    if not matched.empty and {"nt_entry_price", "py_entry_price"}.issubset(matched.columns):
        delta = (matched["py_entry_price"] - matched["nt_entry_price"]).abs()
        if len(delta) >= 2 and delta.mean() > 5.0 and delta.std(ddof=0) < 0.5 * delta.mean():
            warns.append(
                f"CONTRACT-SERIES WARNING: matched entry prices differ by a "
                f"large, near-constant amount (mean {delta.mean():.1f}). The two "
                f"sides may be on different contract series (e.g. continuous vs "
                f"a dated month). Resolve data before attributing logic drift."
            )
    # (b) coverage: one-sided trading days
    nt_days = set(pd.to_datetime(nt["entry_time"]).dt.normalize()) if not nt.empty else set()
    py_days = set(pd.to_datetime(py["entry_time"]).dt.normalize()) if not py.empty else set()
    nt_only_days = sorted(nt_days - py_days)
    py_only_days = sorted(py_days - nt_days)
    if nt_only_days or py_only_days:
        warns.append(
            f"COVERAGE WARNING: {len(nt_only_days)} NT-only and "
            f"{len(py_only_days)} Python-only trading day(s). A one-sided block "
            f"signals a data outage on one side, not a logic bug."
        )
    return warns


from strategy_platform.registry import StrategyRegistry  # noqa: E402
from strategy_platform.data import loader  # noqa: E402


def _load_platform_bars(symbol: str, timeframe_min: Optional[int],
                        bar_size: Optional[int], start: str, end: str) -> pd.DataFrame:
    """Tier-1 data: load from the platform MySQL store at the right bar type."""
    if bar_size is not None:               # tick-bar strategy
        return loader.load_tick_bars(symbol, bar_size, start=start, end=end)
    if timeframe_min == 1:
        return loader.load_1m(symbol, start=start, end=end)
    base = loader.load_5m(symbol, start=start, end=end)
    if timeframe_min and timeframe_min != 5:
        return loader.resample_ohlcv(base, timeframe_min)
    return base


def _run_python(strategy_name: str, params: dict, bars: pd.DataFrame) -> pd.DataFrame:
    """Run a registered strategy on a prepared bars frame; return its trades frame."""
    strat = StrategyRegistry.get(strategy_name)(params)
    full = {**strat.params, **params}
    result = strat.run_backtest(bars, full)
    trades = result.get("trades")
    return trades if isinstance(trades, pd.DataFrame) else pd.DataFrame()


def _diff_summary(nt: pd.DataFrame, py: pd.DataFrame, m: dict) -> dict:
    return {"nt_trades": int(len(nt)), "py_trades": int(len(py)),
            "matched": int(len(m["matched"])),
            "nt_only": int(len(m["nt_only"])), "py_only": int(len(m["py_only"]))}


def parity(strategy_name: str, params: dict, nt_trade_log: str, symbol: str,
           timeframe_min: Optional[int], start: str, end: str,
           nt_export_file: Optional[str] = None, bar_size: Optional[int] = None,
           tolerance: Optional[dict] = None, report_dir: Optional[str] = None) -> dict:
    """Two-tier parity check. Tier 2 (native export) is required for a 'pass'."""
    tol = tolerance or {"price": 1.0, "time_window_s": 5}
    nt = parse_nt_trade_log(nt_trade_log)

    # Tier 1 — platform MySQL data
    bars1 = _load_platform_bars(symbol, timeframe_min, bar_size, start, end)
    py1 = _run_python(strategy_name, params, bars1)
    m1 = match_trades(nt, py1, timeframe_min, tol["time_window_s"], tol["price"])
    tier1 = _diff_summary(nt, py1, m1)

    # Tier 2 — NT native export
    tier2, warns = None, []
    if nt_export_file:
        if bar_size is not None:
            ticks = parse_nt_tick_export(nt_export_file)
            bars2 = ticks_to_bars(ticks, bar_size)
        else:
            bars2 = parse_nt_ohlc_export(nt_export_file)
            if timeframe_min and timeframe_min != 1:
                bars2 = loader.resample_ohlcv(bars2, timeframe_min)
        py2 = _run_python(strategy_name, params, bars2)
        m2 = match_trades(nt, py2, timeframe_min, tol["time_window_s"], tol["price"])
        tier2 = _diff_summary(nt, py2, m2)
        warns = preflight_guards(m2["matched"], nt, py2)

    # Verdict
    if tier2 is None:
        verdict = "data-blocked"        # no native export -> cannot prove logic parity
    elif warns:
        verdict = "data-blocked"        # a data confound is flagged; resolve before judging
    elif tier2["nt_only"] == 0 and tier2["py_only"] == 0:
        verdict = "pass"
    else:
        verdict = "fail"

    report_path = _write_report(report_dir, strategy_name, symbol, tier1, tier2, warns, verdict)
    return {"tier1": tier1, "tier2": tier2, "warnings": warns,
            "verdict": verdict, "report_path": report_path}


def _write_report(report_dir, strategy, symbol, tier1, tier2, warns, verdict) -> str:
    report_dir = report_dir or os.path.join(_REPO_ROOT, "reports")
    os.makedirs(report_dir, exist_ok=True)
    path = os.path.join(report_dir, f"PARITY_REPORT_{strategy}_{symbol}.md")
    lines = [f"# Parity Report — {strategy} on {symbol}", "",
             f"**Verdict:** {verdict}", "", "## Tier 1 (platform MySQL data)",
             f"```\n{tier1}\n```", "", "## Tier 2 (NT native export)",
             f"```\n{tier2}\n```", "", "## Pre-flight warnings"]
    lines += [f"- {w}" for w in warns] or ["- none"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="NT<->platform parity check")
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--nt-log", required=True)
    ap.add_argument("--nt-export", default=None)
    ap.add_argument("--timeframe-min", type=int, default=None)
    ap.add_argument("--bar-size", type=int, default=None)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    args = ap.parse_args(argv)
    res = parity(strategy_name=args.strategy, params={}, nt_trade_log=args.nt_log,
                 symbol=args.symbol, timeframe_min=args.timeframe_min,
                 start=args.start, end=args.end, nt_export_file=args.nt_export,
                 bar_size=args.bar_size)
    print(f"verdict={res['verdict']} report={res['report_path']}")
    return 0 if res["verdict"] in ("pass", "data-blocked") else 1


if __name__ == "__main__":
    sys.exit(main())
