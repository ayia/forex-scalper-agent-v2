#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Strategy Selector
Activate/deactivate strategies based on market regime
"""

from typing import Dict, List
from market_regime_detector import MarketRegime


class StrategySelector:
    """
    Selects appropriate strategies based on market regime
    """
    
    def __init__(self):
        # Define which strategies work best in each regime
        self.regime_strategy_map = {
            MarketRegime.TRENDING_BULL: [
                ('trend_following', 1.2),  # (strategy_name, weight)
                ('breakout', 0.9)
            ],
            MarketRegime.TRENDING_BEAR: [
                ('trend_following', 1.2),
                ('breakout', 0.9)
            ],
            MarketRegime.RANGING: [
                ('mean_reversion', 1.3),
                ('breakout', 0.7)
            ],
            MarketRegime.VOLATILE: [
                ('breakout', 1.1),
                ('mean_reversion', 0.6)
            ],
            MarketRegime.CHOPPY: [
                ('mean_reversion', 0.8),
                ('trend_following', 0.3)
            ],
            MarketRegime.QUIET: [
                ('mean_reversion', 0.5),
                ('trend_following', 0.4),
                ('breakout', 0.3)
            ]
        }
    
    def select_strategies(self, regime: Dict) -> Dict:
        """
        Select and weight strategies for current regime
        
        Args:
            regime: Regime dict from RegimeDetector
            
        Returns:
            Dict with strategy names and their weights
        """
        regime_type = regime.get('regime', MarketRegime.RANGING)
        confidence = regime.get('confidence', 50) / 100  # Normalize to 0-1
        
        # Get strategies for this regime
        strategies = self.regime_strategy_map.get(regime_type, [])
        
        # Apply confidence multiplier
        weighted_strategies = {}
        for strategy, base_weight in strategies:
            # Higher confidence = stick closer to base weight
            # Lower confidence = reduce weight
            final_weight = base_weight * (0.5 + 0.5 * confidence)
            weighted_strategies[strategy] = round(final_weight, 2)
        
        return {
            'strategies': weighted_strategies,
            'regime': regime_type,
            'confidence': confidence
        }
    
    def should_trade(self, regime: Dict, min_confidence: int = 40) -> bool:
        """
        Determine if we should trade in current regime
        """
        regime_type = regime.get('regime', '')
        confidence = regime.get('confidence', 0)
        
        # Don't trade in very low confidence or choppy markets
        if confidence < min_confidence:
            return False
        
        if regime_type == MarketRegime.CHOPPY and confidence < 60:
            return False
        
        if regime_type == MarketRegime.QUIET and confidence < 50:
            return False
        
        return True


def get_active_strategies(regime: Dict) -> Dict:
    """
    Helper function to get active strategies for regime
    """
    selector = StrategySelector()
    return selector.select_strategies(regime)


if __name__ == "__main__":
    print("Adaptive Strategy Selector - Test Mode")
    print("Selects strategies based on market regime")
