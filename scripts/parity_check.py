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
