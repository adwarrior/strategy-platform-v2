"""
SwingStrat — Multi-timeframe SMC (Smart Money Concepts) strategy.

Swing leg detection on HTF (240min default) with FVG-based entry on LTF (15min default).
"""

from .strategy import SwingStrat15M, SwingStrat5M

__all__ = ['SwingStrat15M', 'SwingStrat5M']
