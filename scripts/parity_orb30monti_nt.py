"""Parity check: ORB30Monti Python port vs NT8 Market Replay using NT's
NATIVE EXPORT data (not MySQL 1m).

This removes the 'different data source' variable. We load NT's exported
1m bars (UTC timestamps), convert to ET, resample to 5m, and run the
strategy. Any remaining drift vs NT8's trade log = pure strategy-logic
difference, not data difference.

NT export: /home/ad/data/MNQ 06-26.Last.txt
Format: 'YYYYMMDD HHMMSS;Open;High;Low;Close;Volume'  (semicolon, UTC)

Usage:  cd /home/ad/strategy-platform-v2 && python scripts/parity_orb30monti_nt.py
"""

from __future__ import annotations

import json
import sys

import pandas as pd

sys.path.insert(0, '/home/ad/strategy-platform-v2')

from strategy_platform.data.loader import INSTRUMENT_META
from strategy_platform.strategies.orb30_monti.strategy import ORB30Monti


NT_FILE = "/home/ad/data/MNQ 06-26.Last.txt"


def load_nt_1m(path: str) -> pd.DataFrame:
    """Parse NT's native 1m export. Timestamps are UTC; convert to ET-naive
    (matching the convention the rest of the platform uses for emini DB data)."""
    df = pd.read_csv(
        path, sep=';', header=None,
        names=['ts', 'open', 'high', 'low', 'close', 'volume'],
        dtype={'ts': str},
    )
    # Format: '20260515 040100' → '2026-05-15 04:01:00'
    df.index = pd.to_datetime(df['ts'], format='%Y%m%d %H%M%S')
    df = df.drop(columns=['ts'])
    # UTC → ET, strip tz (matches DB convention from import_update_enter.py)
    df.index = df.index.tz_localize('UTC').tz_convert('America/New_York').tz_localize(None)
    df.index.name = 'datetime'
    return df


def resample_5m(df1: pd.DataFrame) -> pd.DataFrame:
    """Resample 1m → 5m, right-labelled right-closed (matches load_5m convention)."""
    return df1.resample('5min', label='right', closed='right').agg({
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum',
    }).dropna(subset=['open'])


def main() -> None:
    print("=" * 80)
    print("ORB30Monti — Python parity using NT's NATIVE export data")
    print(f"Source: {NT_FILE}")
    print("=" * 80)

    df1 = load_nt_1m(NT_FILE)
    print(f"\n1m bars: {len(df1):,} | range {df1.index.min()} → {df1.index.max()} (ET-naive)")

    df = resample_5m(df1)
    print(f"5m bars: {len(df):,}")
    print(f"\nFirst 3 5m bars:\n{df.head(3)}")
    print(f"\n5m bars on 2026-05-15 09:25–10:25 ET (should bracket NT trade-1 entry at 10:15):")
    print(df.loc['2026-05-15'].between_time('09:25', '10:25').to_string())

    # Strategy with MNQ meta
    s = ORB30Monti()
    m = INSTRUMENT_META['MNQ']
    s.tick_size, s.tick_value, s.commission_rt = m['tick_size'], m['tick_value'], m['commission']

    # NT8 settings: RangeWidth, $200 risk, RR=1, cap=50
    params = {
        'use_delta_filter':       False,
        'use_risk_sizing':        True,
        'risk_per_trade_dollars': 200.0,
        'max_contracts_cap':      50,
        'risk_reward_ratio':      1.0,
    }
    print(f"\nParams: {json.dumps(params, indent=2)}")

    result = s.run_backtest(df, params)

    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(json.dumps({k: v for k, v in result.items() if k != 'trades'}, indent=2, default=str))

    print("\n" + "-" * 80)
    print("TRADES")
    print("-" * 80)
    trades_df = result.get('trades')
    if isinstance(trades_df, pd.DataFrame) and not trades_df.empty:
        with pd.option_context('display.max_columns', None, 'display.width', 200,
                                'display.float_format', '{:,.4f}'.format):
            print(trades_df.to_string())
    else:
        print("(no trades)")


if __name__ == '__main__':
    main()
