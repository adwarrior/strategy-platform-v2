"""
Abstract base class for all strategy-platform trading strategies.

Every strategy inherits from BaseStrategy and must implement:
  - run_backtest(data, params) -> dict   — the core simulation
  - param_grid property                 — the optimization search space

The platform's optimization pipeline calls run_backtest() directly,
passing in a slice of OHLCV data and a specific params dict.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import pandas as pd


class BaseStrategy(ABC):
    """
    Abstract base for all strategies in the unified platform.

    Class-level attributes to override in every subclass
    -------------------------------------------------------
    name : str
        Unique snake_case identifier used by the registry (e.g. "wicktest5m").
    default_params : dict
        Fallback values for every parameter the strategy uses.
    tick_size : float
        Minimum price increment for the instrument.
    tick_value : float
        Dollar value of one tick.
    commission_rt : float
        Round-trip commission per contract in dollars.
    """

    name: str = ""
    default_params: Dict[str, Any] = {}

    # Bar type — "time" for fixed-timeframe bars (default), "tick" for N-tick bars.
    # Tick-based strategies must include "tick_bar_size" in their param_grid.
    bar_type: str = "time"

    # Calculate mode — controls when signals are evaluated within each bar.
    # "on_bar_close"   : evaluate once per completed bar (default, fast)
    # "on_each_tick"   : re-evaluate on every tick as the bar builds (slow, matches NT OnEachTick)
    # "on_price_change": re-evaluate whenever price changes
    calculate_mode: str = "on_bar_close"

    # Valid calculate modes — used by dashboard to build selectbox
    CALCULATE_MODES: list = ["on_bar_close", "on_each_tick", "on_price_change"]

    # Instrument metadata — override in each strategy subclass
    tick_size: float = 0.25
    tick_value: float = 12.50
    commission_rt: float = 3.98

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        if not self.name:
            raise TypeError(
                f"{self.__class__.__name__} must define a non-empty class attribute 'name'."
            )
        self._params: Dict[str, Any] = {**self.default_params}
        if params:
            self._params.update(params)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def params(self) -> Dict[str, Any]:
        return dict(self._params)

    @property
    def param_grid(self) -> Dict[str, List[Any]]:
        """
        Return the parameter search space used by the optimization pipeline.

        Example::

            return {
                "profit": [30, 40, 50, 60],
                "min_bar_size_ticks": [3, 4, 5],
            }
        """
        return {}

    @property
    def description(self) -> str:
        return f"Strategy: {self.name}"

    @property
    def display_names(self) -> Dict[str, str]:
        """Optional human-readable labels for param keys. Override in subclasses."""
        return {}

    @property
    def param_dependencies(self) -> Dict[str, tuple]:
        """
        Map of dependent_param -> (controlling_param, required_value).

        When controlling_param's value != required_value, the dependent_param
        is irrelevant (produces identical results). The optimization pipeline
        collapses the dependent_param to its first grid value in those combos
        and deduplicates, eliminating wasted computation.

        Example::

            return {
                "htf_timeframe_mins": ("use_htf_confirmation", True),
                "ema_period":         ("use_ema_filter",       True),
            }

        Default: returns the strategy's param_conditional if defined, else empty dict.
        Strategies with `param_conditional` automatically benefit from deduplication
        without needing to override this property separately.
        """
        return getattr(self, 'param_conditional', {})

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def run_backtest(self, data: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single backtest on *data* using *params*.

        Parameters
        ----------
        data : pd.DataFrame
            OHLCV DataFrame with a DatetimeIndex and at minimum the columns:
            open, high, low, close, volume.
        params : dict
            Parameter set for this run (may be a grid-search candidate).

        Returns
        -------
        dict
            Must contain at least:
            - "net_pnl"      : float  — total net P&L in dollars
            - "total_trades" : int
            - "win_rate"     : float  — fraction in [0, 1]
            - "sharpe"       : float
            - "max_drawdown" : float  — as a positive dollar value
            Optionally also:
            - "trades"       : pd.DataFrame  — one row per trade
            - "equity_curve" : pd.Series
        """
        ...

    # ------------------------------------------------------------------
    # Optimization hooks (override for efficiency)
    # ------------------------------------------------------------------

    def prepare_data(self, df: pd.DataFrame) -> Any:
        """
        Pre-process OHLCV data once before the grid search begins.

        The result is passed to ``run_backtest_prepared`` and
        ``run_monte_carlo`` instead of the raw DataFrame, so expensive
        pre-processing (e.g. session computation) happens only once per
        IS/OOS slice rather than once per parameter combination.

        Default: returns *df* unchanged.  Override when your strategy has
        a costly pre-processing step (e.g. computing PDH/PDL sessions).
        """
        return df

    def run_backtest_prepared(self, prepared: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a backtest from pre-processed data.

        Default: assumes *prepared* is a pd.DataFrame and calls
        ``run_backtest``.  Override alongside ``prepare_data`` when
        your pre-processed form differs from a DataFrame.
        """
        return self.run_backtest(prepared, params)

    def run_monte_carlo(
        self,
        prepared: Any,
        params: Dict[str, Any],
        n_sims: int = 200,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Run a Monte Carlo stability test on pre-processed IS data.

        Default: returns NaN metrics (MC not implemented).  Override in
        strategies that support day-shuffle or other MC techniques.

        Returns a dict with keys:
          - mc_stability  : fraction of sims with positive net PnL
          - mc_sharpe_p5  : 5th-percentile Sharpe across sims
          - mc_pnl_p5     : 5th-percentile net PnL ($)
          - mc_pnl_p50    : median net PnL ($)
        """
        return {
            'mc_stability': float('nan'),
            'mc_sharpe_p5': float('nan'),
            'mc_pnl_p5':    float('nan'),
            'mc_pnl_p50':   float('nan'),
        }

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    def ticks_to_dollars(self, ticks: float) -> float:
        """Convert a tick count to a dollar P&L value."""
        return ticks * self.tick_value

    def dollars_to_ticks(self, dollars: float) -> float:
        """Convert a dollar amount to tick count."""
        return dollars / self.tick_value

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v!r}" for k, v in self._params.items())
        return f"{self.__class__.__name__}({params_str})"

    def __str__(self) -> str:
        return self.__repr__()
