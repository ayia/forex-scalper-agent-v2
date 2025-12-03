"""Strategies module exports."""
from .base import BaseStrategy, Signal
from .trend_following import TrendFollowingStrategy
from .mean_reversion import MeanReversionStrategy
from .breakout import BreakoutStrategy, VolumeAnalyzer
from .improved import ImprovedTrendStrategy, ImprovedScalpingStrategy, create_improved_strategies

__all__ = [
    # Base
    'BaseStrategy',
    'Signal',
    # Classic strategies
    'TrendFollowingStrategy',
    'MeanReversionStrategy',
    'BreakoutStrategy',
    'VolumeAnalyzer',
    # Improved strategies
    'ImprovedTrendStrategy',
    'ImprovedScalpingStrategy',
    'create_improved_strategies',
]
