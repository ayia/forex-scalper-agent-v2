"""Improved strategies module - Backtest-validated strategies v2.3."""
from .trend import ImprovedTrendStrategy
from .scalping import ImprovedScalpingStrategy

__all__ = [
    'ImprovedTrendStrategy',
    'ImprovedScalpingStrategy',
]


def create_improved_strategies():
    """Create all improved strategies."""
    return [
        ImprovedTrendStrategy(),
        ImprovedScalpingStrategy()
    ]
