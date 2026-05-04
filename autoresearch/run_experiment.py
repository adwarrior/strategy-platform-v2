"""
run_experiment.py — thin IS-backtest runner for the autoresearch loop.

Loads the strategy's current default_params, runs the IS backtest,
and prints a compact JSON result to stdout.

Usage:
    python autoresearch/run_experiment.py --strategy wicktest5m --symbol NQ=F
    python autoresearch/run_experiment.py --strategy goldbot7   --symbol GC=F

    Optional overrides:
        --start 2024-01-01   IS start date (default: 2 years ago)
        --end   2025-01-01   IS end date   (default: 1 year ago, leaving OOS ahead)
        --train-pct 0.70     fraction of date range to use as IS (default 0.70)
        --refresh            force reload from MySQL (skip Parquet cache)

Output (JSON on stdout):
    {"sharpe": 1.23, "net_pnl": 4500.0, "trades": 87, "win_rate": 0.52,
     "max_drawdown": -1200.0, "profit_factor": 1.8, "is_start": "...", "is_end": "..."}

Exit code 0 = success, 1 = not enough trades or error.
"""

import argparse
import json
import sys
import os
from datetime import datetime, timedelta

# Make sure the package root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import strategy_platform  # noqa: F401 — triggers auto-registration
from strategy_platform.registry import StrategyRegistry
from strategy_platform.data.loader import load_5m, load_1m, is_oos_split

MIN_TRADES = 10  # reject runs with fewer trades


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--strategy',   required=True, help='Strategy name (e.g. wicktest5m)')
    p.add_argument('--symbol',     required=True, help='Symbol (e.g. NQ=F)')
    p.add_argument('--start',      default=None,  help='IS data start date YYYY-MM-DD')
    p.add_argument('--end',        default=None,  help='IS data end date YYYY-MM-DD')
    p.add_argument('--bar-type',   default='time', choices=['time', '1m', 'tick'],
                   help='Bar type: time=5M, 1m=1-minute, tick=tick bars')
    p.add_argument('--train-pct',  type=float, default=0.70)
    p.add_argument('--min-trades', type=int, default=MIN_TRADES, help='Minimum trades to accept a run')
    p.add_argument('--refresh',    action='store_true')
    return p.parse_args()


def main():
    args = parse_args()

    # Default date range: 2 years ending today
    today = datetime.today()
    start = args.start or (today - timedelta(days=730)).strftime('%Y-%m-%d')
    end   = args.end   or today.strftime('%Y-%m-%d')

    # Load strategy
    try:
        strategy_cls = StrategyRegistry.get(args.strategy)
    except KeyError as e:
        print(json.dumps({'error': str(e)}))
        sys.exit(1)

    strategy = strategy_cls()

    # Load data (uses db_host from strategy if set)
    db_host = getattr(strategy, 'db_host', None)
    try:
        if args.bar_type == '1m':
            df = load_1m(symbol=args.symbol, start=start, end=end, host=db_host)
        else:
            df = load_5m(
                symbol=args.symbol,
                start=start,
                end=end,
                host=db_host,
                refresh=args.refresh,
            )
    except Exception as e:
        print(json.dumps({'error': f'Data load failed: {e}'}))
        sys.exit(1)

    if df.empty:
        print(json.dumps({'error': 'No data returned'}))
        sys.exit(1)

    # IS slice
    df_is, _, _ = is_oos_split(df, train_pct=args.train_pct)

    # Run backtest with current default_params
    params = dict(strategy.default_params)
    try:
        prepared = strategy.prepare_data(df_is)
        result   = strategy.run_backtest_prepared(prepared, params)
    except Exception as e:
        print(json.dumps({'error': f'Backtest failed: {e}'}))
        sys.exit(1)

    n_trades = result.get('total_trades', result.get('trades', 0))
    if n_trades < args.min_trades:
        print(json.dumps({'error': f'Too few trades: {n_trades}'}))
        sys.exit(1)

    out = {
        'sharpe':        round(float(result.get('sharpe', 0)), 4),
        'net_pnl':       round(float(result.get('net_pnl', 0)), 2),
        'trades':        int(n_trades),
        'win_rate':      round(float(result.get('win_rate', 0)), 4),
        'max_drawdown':  round(float(result.get('max_drawdown', 0)), 2),
        'profit_factor': round(float(result.get('profit_factor', 0)), 4),
        'sortino':       round(float(result.get('sortino', 0)), 4),
        'is_start':      str(df_is.index[0].date()),
        'is_end':        str(df_is.index[-1].date()),
    }
    print(json.dumps(out))
    sys.exit(0)


if __name__ == '__main__':
    main()
