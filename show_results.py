"""
show_results.py — summarise optimizer OOS results from the shared MySQL store.

Usage:
    python show_results.py                        # all strategies
    python show_results.py mobobands              # one strategy
    python show_results.py mobobands orb15m       # multiple
"""

import sys
import pandas as pd
from strategy_platform import results_store

OOS_COLS = [
    'oos_net_pnl', 'oos_sharpe', 'oos_win_rate', 'oos_max_drawdown',
    'oos_trades', 'oos_profit_factor',
]
MC_COLS  = ['mc_stability', 'mc_sharpe_p5', 'mc_pnl_p50']

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:,.2f}'.format)


def load_all_runs() -> pd.DataFrame:
    from sqlalchemy import text
    engine = results_store._results_engine()
    results_store.ensure_results_store()
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT r.id, r.strategy_name, r.symbol, r.label, r.run_ts, r.created_at,
                   s.csv_text
            FROM sp_optimizer_runs r
            JOIN sp_optimizer_run_stages s ON s.run_id = r.id AND s.stage = 'OOS'
            ORDER BY r.strategy_name, r.created_at DESC
        """)).fetchall()
    return rows


def main():
    filter_strategies = [s.lower() for s in sys.argv[1:]]

    rows = load_all_runs()
    if not rows:
        print("No OOS results found in the database.")
        return

    frames = []
    for run_id, strategy_name, symbol, label, run_ts, created_at, csv_text in rows:
        if filter_strategies and strategy_name.lower() not in filter_strategies:
            continue
        df = results_store._csv_text_to_df(csv_text)
        if df.empty:
            continue
        df['_strategy'] = strategy_name
        df['_symbol']   = symbol
        df['_label']    = label or ''
        df['_run_ts']   = run_ts
        df['_created']  = str(created_at)[:10]
        frames.append(df)

    if not frames:
        strategies = sorted({r[1] for r in rows})
        print(f"No matching strategies. Available: {', '.join(strategies)}")
        return

    all_df = pd.concat(frames, ignore_index=True)

    # Identify param columns (not meta, not stats)
    meta_cols = {'_strategy', '_symbol', '_label', '_run_ts', '_created'}
    stat_cols = set(OOS_COLS + MC_COLS)
    all_cols  = set(all_df.columns)
    param_cols = sorted(all_cols - meta_cols - stat_cols - {
        c for c in all_cols if c.startswith('oos_') or c.startswith('mc_') or
        c.startswith('bs_') or c in (
            'strategy_name', 'symbol', 'sym_safe', 'run_ts', 'label',
            '_is_start', '_is_end', '_oos_start', '_oos_end', 'param_keys',
        )
    })

    display_cols = (
        ['_strategy', '_symbol', '_label', '_run_ts', '_created'] +
        param_cols +
        [c for c in MC_COLS  if c in all_df.columns] +
        [c for c in OOS_COLS if c in all_df.columns]
    )
    display_cols = [c for c in display_cols if c in all_df.columns]

    for strategy_name, grp in all_df.groupby('_strategy'):
        print(f"\n{'='*80}")
        print(f"  STRATEGY: {strategy_name.upper()}")
        print(f"{'='*80}")

        for (symbol, run_ts, label), run_grp in grp.groupby(['_symbol', '_run_ts', '_label']):
            run_grp = run_grp.sort_values('oos_net_pnl', ascending=False) if 'oos_net_pnl' in run_grp.columns else run_grp
            print(f"\n  Run: {run_ts}  |  Symbol: {symbol}  |  Label: {label or '(none)'}")
            print(f"  {len(run_grp)} OOS combos\n")

            # Only show params that actually vary across combos in this run
            varying_params = [
                c for c in param_cols
                if c in run_grp.columns and run_grp[c].nunique(dropna=False) > 1
            ]

            run_display_cols = (
                varying_params +
                [c for c in MC_COLS  if c in run_grp.columns] +
                [c for c in OOS_COLS if c in run_grp.columns]
            )
            sub = run_grp[[c for c in run_display_cols if c in run_grp.columns]].copy()

            # Format money columns
            for col in ['oos_net_pnl', 'mc_pnl_p50', 'oos_max_drawdown']:
                if col in sub.columns:
                    sub[col] = sub[col].apply(lambda x: f'${x:,.0f}' if pd.notna(x) else 'n/a')
            for col in ['oos_win_rate', 'mc_stability']:
                if col in sub.columns:
                    sub[col] = sub[col].apply(lambda x: f'{x:.1%}' if pd.notna(x) else 'n/a')

            # Print fixed params separately so they don't clutter the table
            fixed_params = [
                c for c in param_cols
                if c in run_grp.columns and run_grp[c].nunique(dropna=False) == 1
            ]
            if fixed_params:
                fixed_vals = {c: run_grp[c].iloc[0] for c in fixed_params}
                print("  Fixed params: " + "  ".join(f"{k}={v}" for k, v in fixed_vals.items()))
                print()

            print(sub.to_string(index=False))

    print()


if __name__ == '__main__':
    main()
