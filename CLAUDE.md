# strategy-platform

Unified Python platform for backtesting and optimising NinjaTrader strategies.
Strategies are dropped in and the dashboard auto-adapts to each strategy's parameter set.

## Project layout

```
strategy-platform/
├── .env                                    # DB credentials (never commit)
├── CLAUDE.md                               # This file
├── requirements.txt
├── reports/                                # CSV + plot outputs
└── strategy_platform/                      # Main package
    ├── base_strategy.py                    # BaseStrategy ABC — all strategies inherit this
    ├── registry.py                         # @register decorator + StrategyRegistry
    ├── data/
    │   └── loader.py                       # Generic MySQL + Parquet data loader
    ├── optimize/
    │   ├── pipeline.py                     # 4-stage IS/MC/OOS/Bootstrap runner
    │   └── monte_carlo.py                  # Day-shuffle Monte Carlo
    ├── strategies/
    │   ├── wicktest5m/                     # WickTest5M_FTFCv3 (ported from WickTest-Optimizer)
    │   │   └── strategy.py
    │   └── goldbot7/                       # GoldBot7 (ported from GoldBot7-Optimizer)
    │       └── strategy.py
    └── dashboard/
        └── app.py                          # Auto-adaptive Streamlit dashboard
```

## How to add a new strategy

1. Create `strategy_platform/strategies/<your_strategy>/strategy.py`
2. Subclass `BaseStrategy`, set `name`, `default_params`, instrument metadata
3. Implement `run_backtest(data, params) -> dict`
4. Implement the `param_grid` property
5. Decorate the class with `@register`

The dashboard and optimization pipeline discover it automatically via the registry.

## Strategy contract

`run_backtest(data, params)` must return a dict with at least:
- `net_pnl` — total net P&L in dollars
- `total_trades` — int
- `win_rate` — float in [0, 1]
- `sharpe` — float
- `max_drawdown` — positive dollar value

## Database

- WickTest data: Host 172.23.48.1:3306, DB emini, table historical_data
- GoldBot7 data: Host 192.168.1.228:3306, DB emini, table historical_data
- Columns: id, symbol, datetime, open, high, low, close, volume
- Never hardcode DB credentials — always use `.env`

## IS/OOS split

- 70% in-sample / 30% out-of-sample, split by date (not row count)

## Coding conventions

- All data access goes through `strategy_platform/data/loader.py`
- Strategy logic lives only inside its own `strategies/<name>/strategy.py`
- Optimization entry point is `strategy_platform/optimize/pipeline.py`
- Reports saved to `reports/` as CSV and PNG
- Strategies self-register via `@register` — no manual wiring needed
