"""
strategy_platform — unified backtesting and optimisation platform.

Importing this package auto-registers all bundled strategies.
"""

# Auto-register all bundled strategies by importing their modules.
# Each module applies @register to its strategy class on import.
import strategy_platform.strategies.goldbot7.strategy    # noqa: F401
import strategy_platform.strategies.goldbot6.strategy    # noqa: F401
import strategy_platform.strategies.wicktest5m.strategy  # noqa: F401
import strategy_platform.strategies.patscalp.strategy    # noqa: F401
import strategy_platform.strategies.orb15m.strategy      # noqa: F401
import strategy_platform.strategies.swingstrat.strategy  # noqa: F401
import strategy_platform.strategies.mobobands.strategy   # noqa: F401
import strategy_platform.strategies.waejurikpro.strategy # noqa: F401
import strategy_platform.strategies.nybreakout.strategy  # noqa: F401
import strategy_platform.strategies.cct.strategy             # noqa: F401
import strategy_platform.strategies.supertrendfractal.strategy  # noqa: F401
import strategy_platform.strategies.orb30_monti.strategy        # noqa: F401
import strategy_platform.strategies.atr_candle_breakout.strategy  # noqa: F401
import strategy_platform.strategies.aurora.strategy               # noqa: F401
import strategy_platform.strategies.magichour.strategy            # noqa: F401
