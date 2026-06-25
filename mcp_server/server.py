#!/usr/bin/env python3
"""
Local MCP server for strategy-platform-v2.

A thin, stdio-based MCP server that exposes Claude tools pointed at YOUR
engine and YOUR MySQL data. Everything runs in-process by importing the
real platform modules, so there is no logic duplication and nothing leaves
this machine.

Run (for Claude registration):
    python3 /home/ad/strategy-platform-v2/mcp_server/server.py

Register once with:
    claude mcp add --transport stdio --scope user strategy-platform -- \
        python3 /home/ad/strategy-platform-v2/mcp_server/server.py

Design notes
------------
- The server is deliberately "dumb": each tool validates args, calls a
  platform function, and JSON-serialises the result. No strategy logic here.
- Context discipline: bar/trade payloads are row-capped and summarised so a
  100k-row DataFrame never floods the conversation.
- Safety: read/run/save are free; delete requires confirm=True.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from typing import Any, Optional

# --- Make the platform importable regardless of CWD ----------------------
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Load .env (DB_HOST / DB_USER / DB_PASSWORD / DB_NAME) from the repo root,
# same as the dashboard does, so DB access works without manual env setup.
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
except Exception:
    pass

import pandas as pd  # noqa: E402

from mcp.server.fastmcp import FastMCP  # noqa: E402

# Importing the strategies package triggers @register decorators, populating
# the registry. Import it explicitly so list_strategies() is never empty.
import strategy_platform.strategies  # noqa: F401,E402
from strategy_platform.registry import StrategyRegistry  # noqa: E402
from strategy_platform.data import loader  # noqa: E402
from strategy_platform import results_store  # noqa: E402

mcp = FastMCP("strategy-platform")

# Cap on rows ever returned inline, to protect the conversation context.
_MAX_BAR_ROWS = 50
_MAX_TRADE_ROWS = 50


# =========================================================================
# Helpers
# =========================================================================
def _sym_safe(symbol: str) -> str:
    """Match results_store's on-disk symbol convention ('=' -> '_')."""
    return symbol.replace("=", "_")


def _now_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _jsonable(value: Any) -> Any:
    """Coerce pandas/numpy scalars to plain JSON-safe Python values."""
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):  # numpy scalar
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _summarise_metrics(result: dict) -> dict:
    """Pull the standard metrics out of a run_backtest() result dict.

    Drops bulky 'trades' / 'equity_curve' from the inline payload, replacing
    them with light summaries.
    """
    keys = ["net_pnl", "total_trades", "win_rate", "sharpe", "max_drawdown"]
    out = {k: _jsonable(result.get(k)) for k in keys if k in result}

    trades = result.get("trades")
    if isinstance(trades, pd.DataFrame):
        out["trades_count"] = int(len(trades))
        out["trades_sample"] = (
            trades.head(_MAX_TRADE_ROWS)
            .reset_index()
            .astype(str)
            .to_dict(orient="records")
        )
        if len(trades) > _MAX_TRADE_ROWS:
            out["trades_note"] = (
                f"showing first {_MAX_TRADE_ROWS} of {len(trades)} trades"
            )

    eq = result.get("equity_curve")
    if isinstance(eq, pd.Series) and len(eq):
        out["equity_start"] = _jsonable(eq.iloc[0])
        out["equity_end"] = _jsonable(eq.iloc[-1])
        out["equity_points"] = int(len(eq))

    # Surface any other scalar keys the strategy returned.
    for k, v in result.items():
        if k in out or k in ("trades", "equity_curve"):
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            out.setdefault(k, _jsonable(v))
    return out


# Timeframe spec: how to turn (symbol, timeframe) into a bar DataFrame.
# A timeframe is either "Nm" (minutes), or "Ntick" (tick bars).
def _load_bars(symbol: str, timeframe: str, start: Optional[str],
               end: Optional[str]) -> pd.DataFrame:
    tf = timeframe.strip().lower()

    if tf.endswith("tick"):
        bar_size = int(tf[:-4])
        return loader.load_tick_bars(symbol, bar_size, start=start, end=end)

    if tf.endswith("m"):
        minutes = int(tf[:-1])
    elif tf.endswith("min"):
        minutes = int(tf[:-3])
    else:
        raise ValueError(
            f"Unrecognised timeframe '{timeframe}'. "
            "Use e.g. '1m', '5m', '15m', '60m', or '1300tick'."
        )

    if minutes == 1:
        return loader.load_1m(symbol, start=start, end=end)
    if minutes == 5:
        return loader.load_5m(symbol, start=start, end=end)
    # Higher TF: base off 5m then resample (matches platform convention).
    base = loader.load_5m(symbol, start=start, end=end)
    return loader.resample_ohlcv(base, minutes)


# =========================================================================
# Discovery
# =========================================================================
@mcp.tool()
def list_strategies() -> dict:
    """List all registered strategy names available in the platform."""
    names = sorted(StrategyRegistry.list_strategies())
    return {"count": len(names), "strategies": names}


@mcp.tool()
def get_strategy_params(strategy: str) -> dict:
    """Return defaults, param grid, display names, and description for a strategy.

    Use this before run_backtest to learn what parameters exist and their
    valid values.
    """
    cls = StrategyRegistry.get(strategy)
    inst = cls()
    return {
        "strategy": strategy,
        "description": getattr(inst, "description", lambda: "")(),
        "default_params": {k: _jsonable(v) for k, v in inst.params().items()},
        "param_grid": {
            k: [_jsonable(x) for x in v] for k, v in inst.param_grid().items()
        },
        "display_names": inst.display_names(),
    }


# =========================================================================
# Market data
# =========================================================================
@mcp.tool()
def list_symbols() -> dict:
    """List all symbols known to the platform's data loader."""
    return {"symbols": list(loader.ALL_SYMBOLS)}


@mcp.tool()
def get_data_coverage(symbol: str) -> dict:
    """Return tick/commission metadata and known coverage for a symbol."""
    out: dict = {"symbol": symbol}
    try:
        out["meta"] = {k: _jsonable(v) for k, v in loader.get_meta(symbol).items()}
    except KeyError as e:
        out["meta_error"] = str(e)
    tick_cov = getattr(loader, "TICK_DATA_COVERAGE", {})
    if symbol in tick_cov:
        out["tick_coverage"] = tick_cov[symbol]
    return out


@mcp.tool()
def get_bars(symbol: str, timeframe: str = "5m",
             start: Optional[str] = None, end: Optional[str] = None,
             limit: int = _MAX_BAR_ROWS) -> dict:
    """Fetch a small OHLCV preview for a symbol/timeframe.

    timeframe: '1m', '5m', '15m', '60m', or 'Ntick' (e.g. '1300tick').
    start/end: ISO dates 'YYYY-MM-DD'. Returns at most `limit` rows (hard cap
    50) plus row count and date range, to keep context small.
    """
    limit = max(1, min(int(limit), _MAX_BAR_ROWS))
    df = _load_bars(symbol, timeframe, start, end)
    if df.empty:
        return {"symbol": symbol, "timeframe": timeframe, "rows": 0,
                "note": "no data for that range"}
    head = df.head(limit).copy()
    head.index = head.index.astype(str)
    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "rows": int(len(df)),
        "first_bar": str(df.index[0]),
        "last_bar": str(df.index[-1]),
        "sample": head.reset_index().astype(str).to_dict(orient="records"),
        "note": (f"showing first {limit} of {len(df)} bars"
                 if len(df) > limit else None),
    }


# =========================================================================
# Run
# =========================================================================
@mcp.tool()
def run_backtest(strategy: str, symbol: str, timeframe: str = "5m",
                 start: Optional[str] = None, end: Optional[str] = None,
                 params: Optional[dict] = None,
                 save: bool = False, label: Optional[str] = None) -> dict:
    """Run a single backtest on the platform's engine and return its metrics.

    strategy : registered strategy name (see list_strategies).
    symbol   : e.g. 'MNQ', 'MES', 'NQ=F'.
    timeframe: '1m', '5m', '15m', '60m', or 'Ntick'.
    start/end: ISO dates 'YYYY-MM-DD'.
    params   : parameter overrides (see get_strategy_params); defaults used
               for anything omitted.
    save     : if True, persist the result to the results_store DB so it
               appears in the dashboard / results browser.
    label    : optional human label when save=True.

    Returns the standard metrics dict (net_pnl, total_trades, win_rate,
    sharpe, max_drawdown) plus a capped trades sample.
    """
    params = params or {}
    cls = StrategyRegistry.get(strategy)
    inst = cls(params)
    df = _load_bars(symbol, timeframe, start, end)
    if df.empty:
        return {"error": f"no {timeframe} data for {symbol} in that range"}

    # Merge defaults with overrides so the engine gets a complete param set.
    full_params = {**inst.params(), **params}
    result = inst.run_backtest(df, full_params)
    summary = _summarise_metrics(result)
    summary["strategy"] = strategy
    summary["symbol"] = symbol
    summary["timeframe"] = timeframe
    summary["date_range"] = [str(df.index[0]), str(df.index[-1])]
    summary["params_used"] = {k: _jsonable(v) for k, v in full_params.items()}

    if save:
        bt_ts = _now_ts()
        payload = {
            "metrics": {k: _jsonable(result.get(k)) for k in
                        ("net_pnl", "total_trades", "win_rate", "sharpe",
                         "max_drawdown") if k in result},
            "params": {k: _jsonable(v) for k, v in full_params.items()},
            "symbol": symbol,
            "timeframe": timeframe,
            "date_range": [str(df.index[0]), str(df.index[-1])],
            "source": "mcp_server",
        }
        results_store.save_backtest(strategy, symbol, bt_ts, payload, label=label)
        summary["saved"] = {"bt_ts": bt_ts, "label": label}

    return summary


# =========================================================================
# Results store (read + full control; delete gated)
# =========================================================================
@mcp.tool()
def list_backtests(strategy: str, symbol: str) -> dict:
    """List saved backtest timestamps for a strategy/symbol."""
    ts = results_store.list_backtests(strategy, _sym_safe(symbol))
    return {"strategy": strategy, "symbol": symbol,
            "count": len(ts), "timestamps": ts}


@mcp.tool()
def load_backtest(strategy: str, symbol: str, bt_ts: str) -> dict:
    """Load a saved backtest payload by timestamp."""
    payload = results_store.load_backtest(strategy, _sym_safe(symbol), bt_ts)
    if payload is None:
        return {"error": "not found"}
    label = results_store.get_backtest_label(strategy, _sym_safe(symbol), bt_ts)
    return {"strategy": strategy, "symbol": symbol, "bt_ts": bt_ts,
            "label": label, "payload": payload}


@mcp.tool()
def list_optimizer_runs(strategy: str, symbol: str) -> dict:
    """List saved optimizer run timestamps for a strategy/symbol."""
    ts = results_store.list_optimizer_run_timestamps(strategy, _sym_safe(symbol))
    runs = []
    for t in ts:
        runs.append({
            "run_ts": t,
            "label": results_store.get_run_label(strategy, _sym_safe(symbol), t),
        })
    return {"strategy": strategy, "symbol": symbol,
            "count": len(runs), "runs": runs}


@mcp.tool()
def load_optimizer_run(strategy: str, symbol: str, run_ts: str,
                       stage: str = "a") -> dict:
    """Load one stage ('a' or 'b') of a saved optimizer run as a capped table."""
    df = results_store.load_optimizer_stage(
        strategy, _sym_safe(symbol), run_ts, stage)
    if df is None or df.empty:
        return {"error": "not found or empty"}
    settings = results_store.load_optimizer_run_settings(
        strategy, _sym_safe(symbol), run_ts)
    return {
        "strategy": strategy, "symbol": symbol, "run_ts": run_ts,
        "stage": stage, "rows": int(len(df)),
        "settings": settings,
        "sample": df.head(_MAX_TRADE_ROWS).astype(str).to_dict(orient="records"),
        "note": (f"showing first {_MAX_TRADE_ROWS} of {len(df)} rows"
                 if len(df) > _MAX_TRADE_ROWS else None),
    }


@mcp.tool()
def set_label(kind: str, strategy: str, symbol: str, ts: str,
              label: str) -> dict:
    """Set a human label on a saved result.

    kind: 'backtest' or 'optimizer'.
    ts  : the bt_ts or run_ts to label.
    """
    sym = _sym_safe(symbol)
    if kind == "backtest":
        results_store.set_backtest_label(strategy, sym, ts, label)
    elif kind == "optimizer":
        results_store.set_run_label(strategy, sym, ts, label)
    else:
        return {"error": "kind must be 'backtest' or 'optimizer'"}
    return {"ok": True, "kind": kind, "ts": ts, "label": label}


@mcp.tool()
def delete_result(kind: str, strategy: str, symbol: str, ts: str,
                  confirm: bool = False) -> dict:
    """Delete a saved result. REQUIRES confirm=True or it refuses.

    kind: 'backtest' or 'optimizer'.
    ts  : the bt_ts or run_ts to delete.
    """
    if not confirm:
        return {"refused": True,
                "message": f"Will delete {kind} {strategy}/{symbol}/{ts}. "
                           "Call again with confirm=True to proceed."}
    sym = _sym_safe(symbol)
    if kind == "backtest":
        results_store.delete_backtest(strategy, sym, ts)
    elif kind == "optimizer":
        results_store.delete_optimizer_run(strategy, sym, ts)
    else:
        return {"error": "kind must be 'backtest' or 'optimizer'"}
    return {"ok": True, "deleted": {"kind": kind, "strategy": strategy,
                                    "symbol": symbol, "ts": ts}}


if __name__ == "__main__":
    mcp.run()
