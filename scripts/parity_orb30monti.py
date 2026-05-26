"""Parity check: ORB30Monti Python port vs NT8 market replay on MNQ 5m.

NT baseline: /home/ad/Scripts/Results/Ninja.txt — 5 trades, 15/05/26 → 22/05/26.
This script: same window, same params (delta off, $500 risk, 1:1 RR), MNQ 5m.

Usage:  cd /home/ad/strategy-platform-v2 && python scripts/parity_orb30monti.py
"""

from __future__ import annotations

import json
import sys

import pandas as pd

sys.path.insert(0, '/home/ad/strategy-platform-v2')

from strategy_platform.data.loader import load_5m, INSTRUMENT_META
from strategy_platform.strategies.orb30_monti.strategy import ORB30Monti


def main() -> None:
    print("=" * 80)
    print("ORB30Monti — Python parity run vs NT8 Market Replay")
    print("Window: 2026-05-14 → 2026-05-23 (NT covered 15/05–22/05)")
    print("=" * 80)

    # 1. Load 5m MNQ over parity window
    df = load_5m('MNQ', start='2026-05-14', end='2026-05-23')
    print(f"\nData: {len(df):,} 5m bars | {df.index.min()} -> {df.index.max()} | tz={df.index.tz}")
    print(f"First 3 bars:\n{df.head(3)}\n")

    # 2. Strategy with MNQ instrument meta
    s = ORB30Monti()
    m = INSTRUMENT_META['MNQ']
    s.tick_size, s.tick_value, s.commission_rt = m['tick_size'], m['tick_value'], m['commission']
    print(f"Instrument: tick_size={s.tick_size}, tick_value=${s.tick_value}, commission=${s.commission_rt}/RT")
    print(f"Point value: ${s.tick_value / s.tick_size}/pt\n")

    # 3. Run backtest with NT-matching params
    # NT8 settings used in market replay: RangeWidth sizing, $200 risk, RR=1, max cap 50.
    params = {
        'use_delta_filter':       False,
        'sizing_mode':            'range_width',
        'risk_per_trade_dollars': 200.0,
        'max_contracts_cap':      50,
        'risk_reward_ratio':      1.0,
    }
    print(f"Params: {json.dumps(params, indent=2)}\n")

    result = s.run_backtest(df, params)

    # 4. Summary
    print("-" * 80)
    print("SUMMARY")
    print("-" * 80)
    summary = {k: v for k, v in result.items() if k != 'trades'}
    print(json.dumps(summary, indent=2, default=str))

    # 5. Trade-by-trade
    print()
    print("-" * 80)
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
