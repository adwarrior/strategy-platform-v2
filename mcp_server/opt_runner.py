#!/usr/bin/env python3
"""Detached CLI wrapper around pipeline.run_pipeline.

Spawned by the start_optimization MCP tool. Contains no business logic: it
marshals args and calls run_pipeline, which persists results to the DB.
stdout/stderr are redirected to the job log by the parent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_REPO_ROOT, ".env"))
except ImportError:
    pass
except Exception as e:
    sys.stderr.write(f"[opt_runner] warning: failed to load .env: {e}\n")

from strategy_platform.optimize.pipeline import run_pipeline


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", required=True)
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--run-ts", required=True)
    ap.add_argument("--timeframe-mins", type=int, default=5)
    ap.add_argument("--data-start", default=None)
    ap.add_argument("--data-end", default=None)
    ap.add_argument("--train-pct", type=float, default=0.70)
    ap.add_argument("--rank-by", default="sharpe")
    ap.add_argument("--min-trades", type=int, default=None)
    ap.add_argument("--grid-file", default=None,
                    help="Path to JSON file with a param_grid override dict.")
    args = ap.parse_args()

    grid = None
    if args.grid_file:
        try:
            with open(args.grid_file) as f:
                grid = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            sys.stderr.write(f"[opt_runner] error: could not read grid file {args.grid_file}: {e}\n")
            return 2

    print(f"[opt_runner] starting {args.strategy} {args.symbol} run_ts={args.run_ts}", flush=True)
    kwargs = dict(
        strategy_name=args.strategy,
        symbol=args.symbol,
        timeframe_mins=args.timeframe_mins,
        data_start=args.data_start,
        data_end=args.data_end,
        train_pct=args.train_pct,
        rank_by=args.rank_by,
        param_grid_override=grid,
        run_ts=args.run_ts,
    )
    if args.min_trades is not None:
        kwargs["min_trades"] = args.min_trades

    run_pipeline(**kwargs)  # returns stage DataFrames; we discard them — run_pipeline persists results to the DB itself
    print(f"[opt_runner] done run_ts={args.run_ts}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
