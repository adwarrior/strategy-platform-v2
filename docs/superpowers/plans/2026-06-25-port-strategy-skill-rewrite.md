# Port-Strategy Skill Rewrite + Parity Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stale `port-strategy` skill with a mode-based skill that ports/authors strategies into `strategy-platform-v2`, gated on a reusable two-tier trade-by-trade parity harness (`scripts/parity_check.py`).

**Architecture:** Build the harness bottom-up as small, independently-tested pure functions (NT trade-log CSV parser → NT native-export parsers → trade matcher → pre-flight guards → `parity()` orchestrator), then rewrite the skill prose (`SKILL.md` + `references/conventions.md`) that calls it, and deploy via `apply.sh`. Harness logic is tested with synthetic inputs (the hard gate); a real SuperTrendFractal NT pair provides a live smoke test.

**Tech Stack:** Python 3.10, pandas, the existing `strategy_platform` package (`registry`, `data.loader`, `base_strategy`), pytest. The skill itself is Markdown deployed by the repo's `apply.sh`.

## Global Constraints

- **Harness location:** `strategy-platform-v2/scripts/parity_check.py` (committed with the platform). Tests in `strategy-platform-v2/scripts/test_parity_check.py`.
- **Skill source location:** edit `~/Scripts/claude-config/skills/port-strategy/SKILL.md` and `~/Scripts/claude-config/skills/port-strategy/references/conventions.md`; deploy with `~/Scripts/claude-config/apply.sh` (rsyncs `skills/` → `~/.claude/skills/`). NEVER edit `~/.claude/skills/port-strategy/` directly — `apply.sh`/sync clobbers it.
- **Platform trade DataFrame columns** (what `run_backtest(...)["trades"]` yields, per `orb30_monti`): `side` ('Long'/'Short'), `entry_time`, `exit_time`, `entry_price`, `exit_price`, `pnl`, `pnl_ticks`. The harness diffs against these exact names.
- **NT per-trade CSV** (e.g. `STF_89Tick_Trades.csv`) columns: `Market pos.` (Long/Short), `Entry price`, `Exit price`, `Entry time`, `Exit time`, `Profit` (dollar string like `-$18.98`), `MAE`, `MFE`, `Bars`. **Dates are day-first `DD/MM/YYYY HH:MM:SS` → parse with `dayfirst=True`.** Dollar strings: strip `$` and `,`, keep sign.
- **NT native export, two shapes:** 1-min OHLC `'YYYYMMDD HHMMSS;O;H;L;C;V'` (semicolon); tick `'YYYYMMDD HHMMSS<fraction>;price;...;volume'` (subsecond timestamp, one print/line). Both UTC → convert to **ET-naive** (`America/New_York`, tz dropped) to match platform convention.
- **Bar building selected by the registered strategy's `bar_type`** ('time' → resample minutes; 'tick' → aggregate to N-tick bars using the strategy's `tick_bar_size` param).
- **Tick→bar aggregation** must be a harness-local helper (the DB `load_tick_bars` reads MySQL, not a tick list/file).
- **Trade matching:** time-bar → `NT_entry_time == Python_entry_time.ceil(T) - T`, same direction. tick-bar → nearest entry time within a window + same direction, confirmed by entry price within tolerance.
- **Pre-flight guards** run before logic-mismatch attribution: contract-series mismatch (systematically large/constant matched-bar price deltas) and date-coverage gaps (one-sided trading days).
- **Parity verdict:** `pass` requires Tier-2 matched trades agree within tolerance AND no unexplained one-sided trades inside the common date range.
- **Run tests from** `strategy-platform-v2/scripts/`: `cd /home/ad/strategy-platform-v2/scripts && python3 -m pytest test_parity_check.py -v`. `pytest` is installed.
- **Live smoke-test inputs (SuperTrendFractal):** NT log `~/Scripts/Results/NinjaResults/STF_89Tick_Trades.csv`; NT tick export `~/Scripts/Results/NinjaResults/NQ 09-26_16-24.Last.txt`; Python strategy `supertrendfractal` (tick-bar, NQ). Treat as smoke signal only (NQ-live vs MNQ-DB confound documented in `STF_HANDOFF.md`).
- **Four skill modes:** `port` (C#→Python), `author` (idea→C#+Python pair), `reverse` (Python→C#), `fix-parity` (both exist → harness+debug). All converge on register + parity-gate.

---

## File Structure

- `strategy-platform-v2/scripts/parity_check.py` — **new**. The harness: parsers, tick aggregator, matcher, guards, `parity()` orchestrator, and a `__main__` CLI. One cohesive file (~300 lines) since the pieces share types and are always used together.
- `strategy-platform-v2/scripts/test_parity_check.py` — **new**. Synthetic unit tests (hard gate) + one `@pytest.mark.slow` live STF smoke test.
- `~/Scripts/claude-config/skills/port-strategy/SKILL.md` — **rewrite**. Mode-based orchestration prose.
- `~/Scripts/claude-config/skills/port-strategy/references/conventions.md` — **new**. Detailed conventions, keeping SKILL.md lean.

---

### Task 1: NT trade-log CSV parser

Parse the NT per-trade CSV into a normalized trades DataFrame. Pure function, synthetic-tested.

**Files:**
- Create: `strategy-platform-v2/scripts/parity_check.py`
- Test: `strategy-platform-v2/scripts/test_parity_check.py`

**Interfaces:**
- Produces:
  - `parse_nt_trade_log(path: str) -> pd.DataFrame` — columns: `entry_time` (datetime64), `exit_time` (datetime64), `direction` ('Long'/'Short'), `entry_price` (float), `exit_price` (float), `pnl` (float). Dates parsed day-first; `$`/`,` stripped from `Profit`.
  - `_money(s: str) -> float` — `'-$18.98'` → `-18.98`, `'$1,126.02'` → `1126.02`.

- [ ] **Step 1: Write the failing tests**

Create `strategy-platform-v2/scripts/test_parity_check.py`:

```python
import os
import textwrap
import pandas as pd
import pytest

import parity_check as pc


def test_money_parsing():
    assert pc._money("-$18.98") == -18.98
    assert pc._money("$1,126.02") == 1126.02
    assert pc._money("$0.00") == 0.0


def test_parse_nt_trade_log(tmp_path):
    csv = textwrap.dedent("""\
        Trade number,Instrument,Account,Strategy,Market pos.,Qty,Entry price,Exit price,Entry time,Exit time,Entry name,Exit name,Profit,Cum. net profit,Commission,Clearing Fee,Exchange Fee,IP Fee,NFA Fee,MAE,MFE,ETD,Bars
        1,NQ SEP26,Sim,,Long,1,30815.5,30814.75,16/06/2026 09:45:59,16/06/2026 09:46:00,STF_Long,Stop loss,-$18.98,-$18.98,$3.98,$0,$0,$0,$0,$15.00,$0.00,$18.98,1
        2,NQ SEP26,Sim,,Short,1,30818.25,30828.75,16/06/2026 10:00:13,16/06/2026 10:00:40,STF_Short,Stop loss,-$213.98,-$232.96,$3.98,$0,$0,$0,$0,$210.00,$355.00,$568.98,11
    """)
    p = tmp_path / "trades.csv"
    p.write_text(csv)
    df = pc.parse_nt_trade_log(str(p))
    assert list(df.columns) == ["entry_time", "exit_time", "direction",
                                "entry_price", "exit_price", "pnl"]
    assert len(df) == 2
    # day-first: 16/06 is 16 June, not an error
    assert df.iloc[0]["entry_time"] == pd.Timestamp("2026-06-16 09:45:59")
    assert df.iloc[0]["direction"] == "Long"
    assert df.iloc[0]["entry_price"] == 30815.5
    assert df.iloc[1]["pnl"] == -213.98
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ad/strategy-platform-v2/scripts && python3 -m pytest test_parity_check.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'parity_check'`.

- [ ] **Step 3: Write the parser**

Create `strategy-platform-v2/scripts/parity_check.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ad/strategy-platform-v2/scripts && python3 -m pytest test_parity_check.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
cd /home/ad/strategy-platform-v2
git add scripts/parity_check.py scripts/test_parity_check.py
git commit -m "feat(parity): NT trade-log CSV parser

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: NT native-export parsers (OHLC + tick) and tick→bar aggregator

Parse both native-export shapes to ET-naive frames; aggregate a tick frame to N-tick bars.

**Files:**
- Modify: `strategy-platform-v2/scripts/parity_check.py`
- Modify: `strategy-platform-v2/scripts/test_parity_check.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces:
  - `parse_nt_ohlc_export(path: str) -> pd.DataFrame` — DatetimeIndex (ET-naive), columns `open, high, low, close, volume`. Input UTC.
  - `parse_nt_tick_export(path: str) -> pd.DataFrame` — DatetimeIndex (ET-naive), column `price`, `volume`. Input UTC, subsecond timestamps.
  - `ticks_to_bars(ticks: pd.DataFrame, bar_size: int) -> pd.DataFrame` — OHLCV bars, one per `bar_size` ticks; index = each bar's first-tick time. Columns `open, high, low, close, volume`.

- [ ] **Step 1: Write the failing tests**

Add to `test_parity_check.py`:

```python
def test_parse_nt_ohlc_export(tmp_path):
    # 'YYYYMMDD HHMMSS;O;H;L;C;V', UTC -> ET-naive (UTC-4 in June DST)
    data = "20260616 140000;100.0;101.0;99.5;100.5;10\n20260616 140100;100.5;102.0;100.0;101.5;12\n"
    p = tmp_path / "ohlc.txt"
    p.write_text(data)
    df = pc.parse_nt_ohlc_export(str(p))
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    # 14:00 UTC in June = 10:00 ET
    assert df.index[0] == pd.Timestamp("2026-06-16 10:00:00")
    assert df.iloc[0]["high"] == 101.0


def test_parse_nt_tick_export(tmp_path):
    # 'YYYYMMDD HHMMSS<frac>;price;bid?;ask?;volume' — STF format: ts;price;...;vol
    # Real lines look like '20260616 040003 0780000;30832.75;30832.75;30833.25;1'
    data = ("20260616 140000 0000000;100.0;100.0;100.25;1\n"
            "20260616 140000 5000000;100.5;100.25;100.5;2\n")
    p = tmp_path / "ticks.txt"
    p.write_text(data)
    df = pc.parse_nt_tick_export(str(p))
    assert list(df.columns) == ["price", "volume"]
    assert df.iloc[0]["price"] == 100.0
    assert df.index[0] == pd.Timestamp("2026-06-16 10:00:00")  # 14:00 UTC -> 10:00 ET


def test_ticks_to_bars():
    idx = pd.to_datetime([
        "2026-06-16 10:00:00", "2026-06-16 10:00:01",
        "2026-06-16 10:00:02", "2026-06-16 10:00:03",
    ])
    ticks = pd.DataFrame({"price": [100, 101, 99, 102], "volume": [1, 1, 1, 1]}, index=idx)
    bars = pc.ticks_to_bars(ticks, bar_size=2)
    assert len(bars) == 2
    assert bars.iloc[0]["open"] == 100 and bars.iloc[0]["high"] == 101
    assert bars.iloc[0]["low"] == 100 and bars.iloc[0]["close"] == 101
    assert bars.iloc[1]["open"] == 99 and bars.iloc[1]["high"] == 102
    assert bars.iloc[0]["volume"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ad/strategy-platform-v2/scripts && python3 -m pytest test_parity_check.py -k "export or ticks_to_bars" -v`
Expected: FAIL — `AttributeError: module 'parity_check' has no attribute 'parse_nt_ohlc_export'`.

- [ ] **Step 3: Implement the parsers + aggregator**

Append to `parity_check.py`:

```python
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
            ts_field = parts[0]                       # 'YYYYMMDD HHMMSS NNNNNNN'
            ymd_hms = ts_field[:15]                   # 'YYYYMMDD HHMMSS'
            frac = ts_field[16:] if len(ts_field) > 15 else "0"
            base = pd.to_datetime(ymd_hms, format="%Y%m%d %H%M%S")
            ts = base + pd.to_timedelta(int(frac or 0) * 100, unit="ns")
            price = float(parts[1])
            vol = float(parts[-1]) if parts[-1] else 1.0
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ad/strategy-platform-v2/scripts && python3 -m pytest test_parity_check.py -k "export or ticks_to_bars" -v`
Expected: PASS (3 tests). All-file run should now be 5 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/ad/strategy-platform-v2
git add scripts/parity_check.py scripts/test_parity_check.py
git commit -m "feat(parity): NT native-export parsers + tick->bar aggregator

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Trade matcher + pre-flight guards

Match Python trades to NT trades; flag contract-series and date-coverage problems.

**Files:**
- Modify: `strategy-platform-v2/scripts/parity_check.py`
- Modify: `strategy-platform-v2/scripts/test_parity_check.py`

**Interfaces:**
- Consumes: normalized NT frame (Task 1) and a platform trades frame with `side`/`entry_time`/`entry_price` etc.
- Produces:
  - `match_trades(nt: pd.DataFrame, py: pd.DataFrame, timeframe_min: Optional[int], time_window_s: int, price_tol: float) -> dict` — returns `{matched: pd.DataFrame, nt_only: pd.DataFrame, py_only: pd.DataFrame}`. `matched` has paired columns `nt_entry_price`, `py_entry_price`, `nt_pnl_ticks`?, etc. When `timeframe_min` is set (time-bar), match key is `nt_entry_time == py_entry_time.ceil(Tmin) - Tmin` + same direction; else (tick-bar) nearest within `time_window_s` + same direction + entry price within `price_tol`.
  - `preflight_guards(matched: pd.DataFrame, nt: pd.DataFrame, py: pd.DataFrame) -> list[str]` — warning strings for (a) systematically large/near-constant matched entry-price deltas (contract-series), (b) one-sided trading days (coverage).
- Note: Python frame uses `side` ('Long'/'Short'); NT frame uses `direction` ('Long'/'Short'). The matcher normalizes both to a `direction` series.

- [ ] **Step 1: Write the failing tests**

Add to `test_parity_check.py`:

```python
def _py_frame(rows):
    # platform trades frame: side, entry_time, exit_time, entry_price, exit_price, pnl_ticks
    return pd.DataFrame(rows)


def test_match_trades_time_bar():
    # NT entry_time = bar OPEN (e.g. 10:00); Python sub-bar ts inside bar -> ceil(5min)-5min == 10:00
    nt = pd.DataFrame({
        "entry_time": [pd.Timestamp("2026-06-16 10:00:00")],
        "exit_time": [pd.Timestamp("2026-06-16 10:05:00")],
        "direction": ["Long"], "entry_price": [100.0],
        "exit_price": [101.0], "pnl": [50.0],
    })
    py = _py_frame({
        "side": ["Long"],
        "entry_time": [pd.Timestamp("2026-06-16 10:03:00")],  # inside 10:00-10:05 bar
        "exit_time": [pd.Timestamp("2026-06-16 10:05:00")],
        "entry_price": [100.0], "exit_price": [101.0], "pnl_ticks": [4.0],
    })
    res = pc.match_trades(nt, py, timeframe_min=5, time_window_s=0, price_tol=1.0)
    assert len(res["matched"]) == 1
    assert len(res["nt_only"]) == 0 and len(res["py_only"]) == 0


def test_match_trades_tick_bar_nearest():
    nt = pd.DataFrame({
        "entry_time": [pd.Timestamp("2026-06-16 10:00:00")],
        "exit_time": [pd.Timestamp("2026-06-16 10:00:05")],
        "direction": ["Short"], "entry_price": [200.0],
        "exit_price": [199.0], "pnl": [40.0],
    })
    py = _py_frame({
        "side": ["Short"],
        "entry_time": [pd.Timestamp("2026-06-16 10:00:02")],  # 2s away
        "exit_time": [pd.Timestamp("2026-06-16 10:00:06")],
        "entry_price": [200.25], "exit_price": [199.0], "pnl_ticks": [3.0],
    })
    res = pc.match_trades(nt, py, timeframe_min=None, time_window_s=5, price_tol=1.0)
    assert len(res["matched"]) == 1


def test_preflight_contract_series_warning():
    # all matched entry-price deltas ~ +500, near constant => contract series warning
    matched = pd.DataFrame({
        "nt_entry_price": [100.0, 110.0, 120.0],
        "py_entry_price": [600.0, 610.0, 620.0],
        "nt_entry_time": pd.to_datetime(["2026-06-16 10:00", "2026-06-16 11:00", "2026-06-17 10:00"]),
    })
    warns = pc.preflight_guards(matched,
                                nt=pd.DataFrame({"entry_time": matched["nt_entry_time"]}),
                                py=pd.DataFrame({"entry_time": matched["nt_entry_time"]}))
    assert any("contract" in w.lower() or "series" in w.lower() for w in warns)


def test_preflight_coverage_warning():
    matched = pd.DataFrame({"nt_entry_price": [100.0], "py_entry_price": [100.0],
                            "nt_entry_time": pd.to_datetime(["2026-06-16 10:00"])})
    nt = pd.DataFrame({"entry_time": pd.to_datetime(["2026-06-16 10:00", "2026-06-18 10:00"])})  # 18th NT-only
    py = pd.DataFrame({"entry_time": pd.to_datetime(["2026-06-16 10:00"])})
    warns = pc.preflight_guards(matched, nt=nt, py=py)
    assert any("coverage" in w.lower() or "only" in w.lower() for w in warns)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ad/strategy-platform-v2/scripts && python3 -m pytest test_parity_check.py -k "match_trades or preflight" -v`
Expected: FAIL — `AttributeError: ... 'match_trades'`.

- [ ] **Step 3: Implement matcher + guards**

Append to `parity_check.py`:

```python
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
    nt_only = []
    for i, ntr in nt.iterrows():
        hit = None
        for j, pyr in py.iterrows():
            if j in used_py or pyr["_dir"] != ntr["_dir"]:
                continue
            if timeframe_min is not None:
                if pyr["_key_time"] == ntr["entry_time"]:
                    hit = j
                    break
            else:
                dt = abs((pyr["entry_time"] - ntr["entry_time"]).total_seconds())
                if dt <= time_window_s and abs(pyr["entry_price"] - ntr["entry_price"]) <= price_tol:
                    hit = j
                    break
        if hit is None:
            nt_only.append(ntr)
        else:
            used_py.add(hit)
            pyr = py.loc[hit]
            matched_rows.append({
                "nt_entry_time": ntr["entry_time"], "py_entry_time": pyr["entry_time"],
                "direction": ntr["_dir"],
                "nt_entry_price": ntr["entry_price"], "py_entry_price": pyr["entry_price"],
                "nt_exit_price": ntr["exit_price"], "py_exit_price": pyr["exit_price"],
            })
    py_only = py.loc[[j for j in py.index if j not in used_py]]
    return {
        "matched": pd.DataFrame(matched_rows),
        "nt_only": pd.DataFrame(nt_only),
        "py_only": py_only.reset_index(drop=True),
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /home/ad/strategy-platform-v2/scripts && python3 -m pytest test_parity_check.py -k "match_trades or preflight" -v`
Expected: PASS (4 tests). Full-file run: 9 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/ad/strategy-platform-v2
git add scripts/parity_check.py scripts/test_parity_check.py
git commit -m "feat(parity): trade matcher + contract-series/coverage guards

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `parity()` orchestrator + report + CLI + live STF smoke test

Tie the pieces into the two-tier check, write the report, add a CLI, and verify against the real STF pair.

**Files:**
- Modify: `strategy-platform-v2/scripts/parity_check.py`
- Modify: `strategy-platform-v2/scripts/test_parity_check.py`

**Interfaces:**
- Consumes: Tasks 1–3 (`parse_nt_trade_log`, `parse_nt_ohlc_export`, `parse_nt_tick_export`, `ticks_to_bars`, `match_trades`, `preflight_guards`), plus `StrategyRegistry`, `loader`.
- Produces:
  - `parity(strategy_name, params, nt_trade_log, symbol, timeframe_min, start, end, nt_export_file=None, bar_size=None, tolerance=None, report_dir=None) -> dict` — returns `{tier1, tier2, warnings, verdict, report_path}`. `verdict` ∈ {`"pass"`, `"fail"`, `"data-blocked"`}.
  - `_run_python(strategy_name, params, bars) -> pd.DataFrame` — runs the registered strategy on a prepared bars frame and returns its trades frame.
  - `__main__` CLI: `--strategy --symbol --nt-log [--nt-export] [--timeframe-min] [--bar-size] --start --end`.

- [ ] **Step 1: Write the failing tests**

Add to `test_parity_check.py` (a synthetic orchestrator test + the slow live smoke test):

```python
def test_parity_verdict_pass_synthetic(monkeypatch, tmp_path):
    # NT log with one Long trade at a 5-min bar open
    csv = ("Trade number,Instrument,Account,Strategy,Market pos.,Qty,Entry price,Exit price,"
           "Entry time,Exit time,Entry name,Exit name,Profit,Cum. net profit,Commission,"
           "Clearing Fee,Exchange Fee,IP Fee,NFA Fee,MAE,MFE,ETD,Bars\n"
           "1,NQ,Sim,,Long,1,100.0,101.0,16/06/2026 10:00:00,16/06/2026 10:05:00,L,TP,$50.00,$50.00,$3.98,$0,$0,$0,$0,$0,$0,$0,1\n")
    log = tmp_path / "log.csv"; log.write_text(csv)

    bars = pd.DataFrame(
        {"open": [100.0], "high": [101.0], "low": [99.0], "close": [101.0], "volume": [10]},
        index=pd.to_datetime(["2026-06-16 10:00:00"]))

    # stub data load + strategy run so the test is hermetic (no DB)
    monkeypatch.setattr(pc, "_load_platform_bars", lambda *a, **k: bars)
    py_trades = pd.DataFrame({
        "side": ["Long"], "entry_time": [pd.Timestamp("2026-06-16 10:03:00")],
        "exit_time": [pd.Timestamp("2026-06-16 10:05:00")],
        "entry_price": [100.0], "exit_price": [101.0], "pnl_ticks": [4.0]})
    monkeypatch.setattr(pc, "_run_python", lambda *a, **k: py_trades)

    res = pc.parity(strategy_name="dummy", params={}, nt_trade_log=str(log),
                    symbol="NQ", timeframe_min=5, start="2026-06-16", end="2026-06-17",
                    nt_export_file=None, report_dir=str(tmp_path))
    assert res["verdict"] in ("pass", "data-blocked")
    assert res["tier1"]["matched"] == 1
    assert os.path.exists(res["report_path"])


@pytest.mark.slow
def test_parity_live_supertrendfractal(tmp_path):
    nt_log = "/home/ad/Scripts/Results/NinjaResults/STF_89Tick_Trades.csv"
    nt_export = "/home/ad/Scripts/Results/NinjaResults/NQ 09-26_16-24.Last.txt"
    if not (os.path.exists(nt_log) and os.path.exists(nt_export)):
        pytest.skip("STF reference files not present")
    # STF live config (from STF_HANDOFF.md), 89-tick NQ
    params = {"atr_multiplier": 3, "atr_period": 10, "fractal_length": 3,
              "exit_mode": "FixedTPTrailSL", "tp_ticks": 80,
              "session_filter": True, "tick_bar_size": 89}
    res = pc.parity(strategy_name="supertrendfractal", params=params,
                    nt_trade_log=nt_log, symbol="NQ", timeframe_min=None,
                    start="2026-06-16", end="2026-06-25",
                    nt_export_file=nt_export, bar_size=89,
                    tolerance={"price": 1.0, "time_window_s": 5},
                    report_dir=str(tmp_path))
    # Smoke only: assert it RAN end-to-end and produced a report + trade counts.
    # Do NOT assert verdict==pass (NQ-live vs MNQ-DB confound is documented).
    assert os.path.exists(res["report_path"])
    assert "tier2" in res and res["tier2"]["nt_trades"] > 0
    print("STF parity verdict:", res["verdict"], "warnings:", res["warnings"])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /home/ad/strategy-platform-v2/scripts && python3 -m pytest test_parity_check.py -k "verdict_pass_synthetic" -v`
Expected: FAIL — `AttributeError: ... 'parity'` (and `_load_platform_bars`/`_run_python` absent).

- [ ] **Step 3: Implement the orchestrator, runner, report, CLI**

Append to `parity_check.py`:

```python
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
```

- [ ] **Step 4: Run the synthetic orchestrator test**

Run: `cd /home/ad/strategy-platform-v2/scripts && python3 -m pytest test_parity_check.py -k "verdict_pass_synthetic" -v`
Expected: PASS. Then full non-slow run: `python3 -m pytest test_parity_check.py -v` → 10 passed.

- [ ] **Step 5: Run the live STF smoke test**

Run: `cd /home/ad/strategy-platform-v2/scripts && python3 -m pytest test_parity_check.py -k "live_supertrendfractal" -v -m slow -s`
Expected: PASS (it asserts it RAN and produced a report with `tier2.nt_trades > 0`; it prints the verdict + warnings). If it ERRORS on a real bug (e.g. the STF strategy needs a param the test omitted, or `supertrendfractal` rejects `params`), READ the error and fix the test's `params` to match the registered strategy's actual `param_grid` keys — inspect with `python3 -c "import strategy_platform.strategies; from strategy_platform.registry import StrategyRegistry as R; print(R.get('supertrendfractal')().param_grid.keys())"`. Do NOT loosen the harness to make it pass; the harness must run the real strategy.

- [ ] **Step 6: Commit**

```bash
cd /home/ad/strategy-platform-v2
git add scripts/parity_check.py scripts/test_parity_check.py
git commit -m "feat(parity): two-tier parity() orchestrator + report + CLI + STF smoke test

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Rewrite the skill (SKILL.md + conventions.md) and deploy

Replace the stale prompt with mode-based prose that calls the harness; deploy via apply.sh.

**Files:**
- Rewrite: `~/Scripts/claude-config/skills/port-strategy/SKILL.md`
- Create: `~/Scripts/claude-config/skills/port-strategy/references/conventions.md`

**Interfaces:**
- Consumes: `scripts/parity_check.py` CLI/`parity()` from Task 4.

- [ ] **Step 1: Write `references/conventions.md`**

Create `~/Scripts/claude-config/skills/port-strategy/references/conventions.md`:

```markdown
# Porting conventions (strategy-platform-v2)

## Database timezone
- `historical_data` is ET. `historical_data_1m` and `tick_data` are UTC.
- `tz_convert`/treat-as-UTC before ANY session-hour logic. (orb30_monti converts to America/New_York internally.)

## NinjaTrader native export
- Exported UTC regardless of chart timezone. Tools → Historical Data; "Get Data" first to fill gaps.
- 1-min OHLC: `YYYYMMDD HHMMSS;O;H;L;C;V` (semicolon).
- Tick: `YYYYMMDD HHMMSS<frac>;price;...;volume` (subsecond, one print/line).

## NT per-trade log (Trades.csv)
- Columns: `Market pos.` (Long/Short), `Entry price`, `Exit price`, `Entry time`, `Exit time`, `Profit` ($ string), `MAE`, `MFE`, `Bars`.
- Dates are **day-first** DD/MM/YYYY. Dollar fields: strip `$`/`,`.

## Tick data
- Load monthly-and-concat to avoid OOM (esp. MNQ).
- `bar_type='tick'` strategies must include `tick_bar_size` in `param_grid`.

## Commission
- Per-instrument round-trip from `Commisions.txt`; set `commission_rt` on the strategy.

## BaseStrategy contract
- Set `name`, `default_params`, `tick_size`, `tick_value`, `commission_rt`, `bar_type`.
- Implement `run_backtest(data, params) -> dict` returning at least `net_pnl`, `total_trades`, `win_rate`, `sharpe`, `max_drawdown`; trades frame uses columns `side`, `entry_time`, `exit_time`, `entry_price`, `exit_price`, `pnl`, `pnl_ticks`.
- Implement `param_grid` property. Decorate the class with `@register`.
- Expose EVERY NinjaTrader input as a platform param so all are optimizable.

## Fast iteration
- Use the local MCP `run_backtest` tool for Python-side backtests while debugging divergences.
- Use the local MCP `start_optimization` for sweeps (not in this skill's scope).
```

- [ ] **Step 2: Rewrite `SKILL.md`**

Overwrite `~/Scripts/claude-config/skills/port-strategy/SKILL.md`:

```markdown
---
name: port-strategy
description: Port or author a trading strategy as a matched NinjaTrader-C#/Python pair for strategy-platform-v2, gated on a two-tier trade-by-trade parity check. Use when porting a NinjaScript strategy to the platform, authoring a new strategy in both languages, generating the C# from a Python strategy, or debugging NT-vs-Python divergence.
---

You port and author strategies INTO the existing platform at
`/home/ad/strategy-platform-v2` (registry, loader, IS/MC/OOS pipeline, results
store, local MCP all already exist). You do NOT build a new pipeline. The
deliverable is a registered Python strategy (and/or generated NT8 C#) whose
trades MATCH its NT counterpart, proven by the parity harness.

Read `references/conventions.md` before writing any strategy code — it carries
the DB-timezone, NT-export, trade-log, tick, commission, and BaseStrategy rules
that cause parity bugs when missed.

## Step 0 — Detect the mode

- `.cs` file given, no Python yet → **port** (C#→Python).
- Prose description, neither side exists → **author** (generate C# + Python pair).
- Python strategy exists, no C# → **reverse** (generate NT8 C# for execution).
- Both exist, drift suspected → **fix-parity** (skip authoring; go to Step 3).

If ambiguous, ask which mode (use AskUserQuestion).

## Step 1 — Gather inputs (ask only what's missing)

- Source file path(s); instrument + bar type (time TF or N-tick); the NT trade
  log (`*_Trades.csv`) and, REQUIRED for the parity gate, the NT native export
  (`*.Last.txt`); the live config / param values used in the NT run; the date
  window of the NT run.

## Step 2 — Author / port the missing side

- Translate logic faithfully; expose EVERY NT input as a platform param.
- Python side: subclass `BaseStrategy` in
  `strategy_platform/strategies/<name>/strategy.py`, implement `run_backtest`
  and `param_grid`, set instrument metadata, decorate `@register`.
- Follow `references/conventions.md` exactly (timezones, tick handling).
- Auto-save as you go (large ports can hit token limits mid-run).

## Step 3 — Register & sanity-run

- Confirm the strategy appears in the registry.
- Run a quick Python backtest (the local MCP `run_backtest` tool or a small
  script) on the NT window to confirm it produces trades.

## Step 4 — Parity gate (the definition of done)

Run the harness:

```
cd /home/ad/strategy-platform-v2 && python3 scripts/parity_check.py \
  --strategy <name> --symbol <SYM> \
  --nt-log "<...Trades.csv>" --nt-export "<....Last.txt>" \
  [--timeframe-min N | --bar-size N] --start YYYY-MM-DD --end YYYY-MM-DD
```

- **Tier 1** (platform data) and **Tier 2** (NT's own export) both run.
- The port is DONE only when Tier 2 reports `pass` (matched trades agree, no
  unexplained one-sided trades).
- If the verdict is `data-blocked`, the harness flagged a CONTRACT-SERIES or
  COVERAGE problem — fix the DATA (same series, same window) before touching
  strategy logic. Do not "fix" logic to mask a data mismatch.
- If `fail`, debug the divergence trade-by-trade against `PARITY_REPORT_*.md`,
  iterating the Python (use the local MCP `run_backtest` for speed) until Tier 2
  passes.

## Key rules

- NinjaTrader stays execution-only; this skill stops at a verified, registered,
  optimizable Python port (and/or generated C#).
- Never declare a port done on metrics alone or on Tier 1 alone — Tier 2 is the gate.
- Do not guess missing details — ask.
```

- [ ] **Step 3: Deploy via apply.sh**

Run: `bash /home/ad/Scripts/claude-config/apply.sh 2>&1 | tail -5`
Then verify the deployed copy matches:
Run: `diff ~/Scripts/claude-config/skills/port-strategy/SKILL.md ~/.claude/skills/port-strategy/SKILL.md && echo "SKILL.md deployed OK"`
Run: `test -f ~/.claude/skills/port-strategy/references/conventions.md && echo "conventions.md deployed OK"`
Expected: both echo OK.

- [ ] **Step 4: Verify the description triggers correctly (manual check)**

Read the new `description` field aloud against the four modes: it must mention porting, authoring a pair, generating C# from Python, and debugging divergence. Confirm all four are present.
Expected: all four present (they are, in the frontmatter above).

- [ ] **Step 5: Commit**

```bash
cd /home/ad/Scripts/claude-config
git add skills/port-strategy/SKILL.md skills/port-strategy/references/conventions.md
git commit -m "feat(skill): rewrite port-strategy as mode-based, parity-gated workflow

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Four modes → Task 5 SKILL.md Step 0. ✓
- Two-tier parity (Tier 1 MySQL, Tier 2 native export, gate on Tier 2) → Task 4 `parity()`. ✓
- Reusable harness in `scripts/parity_check.py` → Tasks 1–4. ✓
- NT trade-log CSV parser (day-first dates, `$` strip, `Market pos.`) → Task 1. ✓
- Native-export parsers (OHLC + tick) + tick→bar aggregator → Task 2. ✓
- Trade matching key (`ceil(T)-T` time-bar; nearest-window tick-bar) → Task 3. ✓
- Pre-flight guards (contract-series, coverage) → Task 3. ✓
- Platform trade-frame columns (`side`/`entry_time`/`pnl_ticks`) → Task 3 matcher + Task 4 runner. ✓
- Conventions captured in `references/conventions.md` → Task 5 Step 1. ✓
- Edit-repo-then-apply.sh deployment → Task 5 Step 3. ✓
- Synthetic unit tests as hard gate + live STF smoke test → Tasks 1–4 tests. ✓
- STF live inputs + smoke-only caveat (NQ vs MNQ) → Task 4 Step 5 + slow test. ✓
- Out-of-scope (no multi-agent system, no auto-optimize, no live exec) → not built. ✓

**Placeholder scan:** none — every code/command step is concrete.

**Type consistency:** harness function names and signatures in Tasks 1–4 Produces blocks are used verbatim downstream (`parse_nt_trade_log`, `parse_nt_ohlc_export`, `parse_nt_tick_export`, `ticks_to_bars`, `match_trades`, `preflight_guards`, `parity`, `_load_platform_bars`, `_run_python`). Platform-side: trades frame `side`/`entry_time`/`entry_price`; NT frame `direction`/`entry_time` — matcher normalizes both via `_norm_direction`. `bar_type='tick'` → `bar_size`/`tick_bar_size` path is consistent across loader call, Tier-2 aggregation, and the STF test.

**Note on the STF live test:** it asserts the harness RAN and produced a report (smoke), not `verdict=="pass"` — deliberate, per the spec's documented NQ-live vs MNQ-DB confound. If the registered `supertrendfractal` rejects the test's `params`, the fix is to align the test params to its real `param_grid` keys (Task 4 Step 5 gives the inspection command), not to weaken the harness.
```
