#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Risk Manager
Dynamic position sizing, stop loss, and take profit based on:
- ATR volatility
- Market regime
- Trading session
- Spread conditions
- Pair characteristics
"""

import numpy as np
from typing import Dict, Tuple
import pandas as pd
from datetime import datetime

from risk.adaptive_thresholds import get_pair_profile, detect_session


class AdaptiveRiskManager:
    """
    Manages risk parameters adaptively based on market conditions
    """

    def __init__(self, base_risk_percent: float = 1.0):
        """
        Args:
            base_risk_percent: Base account risk per trade (default 1%)
        """
        self.base_risk_percent = base_risk_percent

        # Session risk multipliers
        self.session_multipliers = {
            'tokyo': 0.8,       # Lower risk during quieter session
            'london': 1.2,      # Higher risk during most liquid
            'ny': 1.2,          # Higher risk during US session
            'london_ny': 1.3,   # Highest during overlap
            'quiet': 0.6        # Minimal risk during quiet hours
        }

    def calculate_position_size(self, account_balance: float, pair: str,
                                atr: float, price: float,
                                regime: Dict = None,
                                spread: float = None) -> Dict:
        """
        Calculate adaptive position size

        Args:
            account_balance: Account balance in base currency
            pair: Trading pair
            atr: Current ATR value
            price: Current price
            regime: Market regime dict (from RegimeDetector)
            spread: Current spread in pips

        Returns:
            Dict with position_size, risk_amount, and metadata
        """
        # Get pair profile
        pair_profile = get_pair_profile(pair)

        # Get session
        session_info = detect_session()
        session = session_info['session']

        # Base risk amount
        base_risk = account_balance * (self.base_risk_percent / 100)

        # Apply session multiplier
        session_mult = self.session_multipliers.get(session, 1.0)

        # Apply volatility adjustment
        volatility_mult = self._get_volatility_multiplier(regime) if regime else 1.0

        # Apply spread penalty
        spread_mult = self._get_spread_multiplier(spread, pair_profile) if spread else 1.0

        # Apply pair-specific adjustment
        pair_mult = pair_profile.get('risk_multiplier', 1.0)

        # Calculate final risk amount
        adjusted_risk = base_risk * session_mult * volatility_mult * spread_mult * pair_mult

        # Calculate position size based on ATR stop loss
        sl_distance_pips = atr * pair_profile.get('atr_sl_multiplier', 1.5) * 10000
        sl_distance_price = sl_distance_pips / 10000

        # Position size = Risk / SL distance
        position_size = adjusted_risk / sl_distance_price

        return {
            'position_size': round(position_size, 2),
            'risk_amount': round(adjusted_risk, 2),
            'sl_distance_pips': round(sl_distance_pips, 1),
            'session_mult': session_mult,
            'volatility_mult': volatility_mult,
            'spread_mult': spread_mult,
            'pair_mult': pair_mult,
            'final_multiplier': round(session_mult * volatility_mult * spread_mult * pair_mult, 2)
        }

    def calculate_sl_tp(self, entry_price: float, direction: str,
                       pair: str, atr: float,
                       regime: Dict = None) -> Dict:
        """
        Calculate adaptive stop loss and take profit

        Args:
            entry_price: Entry price
            direction: 'long' or 'short'
            pair: Trading pair
            atr: Current ATR
            regime: Market regime dict

        Returns:
            Dict with sl, tp, and R:R ratio
        """
        pair_profile = get_pair_profile(pair)

        # Get ATR multipliers based on regime
        sl_multiplier = self._get_sl_multiplier(regime, pair_profile)
        tp_multiplier = self._get_tp_multiplier(regime, pair_profile)

        # Calculate SL distance
        sl_distance = atr * sl_multiplier

        # Calculate TP distance
        tp_distance = atr * tp_multiplier

        # Apply direction
        if direction.lower() in ['long', 'buy']:
            sl = entry_price - sl_distance
            tp = entry_price + tp_distance
        else:  # short/sell
            sl = entry_price + sl_distance
            tp = entry_price - tp_distance

        # Calculate R:R ratio
        risk = abs(entry_price - sl)
        reward = abs(tp - entry_price)
        rr_ratio = reward / risk if risk > 0 else 0

        return {
            'sl': round(sl, 5),
            'tp': round(tp, 5),
            'sl_distance': round(sl_distance, 5),
            'tp_distance': round(tp_distance, 5),
            'rr_ratio': round(rr_ratio, 2),
            'sl_pips': round(sl_distance * 10000, 1),
            'tp_pips': round(tp_distance * 10000, 1)
        }

    def _get_volatility_multiplier(self, regime: Dict) -> float:
        """
        Get risk multiplier based on volatility regime
        """
        if not regime:
            return 1.0

        volatility = regime.get('volatility_level', 'normal')
        regime_type = regime.get('regime', '')

        # Reduce risk in volatile/choppy markets
        if volatility == 'high' or regime_type == 'volatile':
            return 0.7
        elif volatility == 'low' or regime_type == 'quiet':
            return 0.8
        elif regime_type == 'choppy':
            return 0.6
        elif regime_type in ['trending_bull', 'trending_bear']:
            return 1.1  # Slightly increase in trending markets
        else:
            return 1.0

    def _get_spread_multiplier(self, spread: float, pair_profile: Dict) -> float:
        """
        Penalize high spreads
        """
        if spread is None:
            return 1.0

        # Get typical spread for this pair
        typical_spread = pair_profile.get('avg_spread_pips', 2.0)

        # If spread is more than 2x typical, reduce risk
        if spread > typical_spread * 2:
            return 0.5
        elif spread > typical_spread * 1.5:
            return 0.7
        elif spread > typical_spread:
            return 0.9
        else:
            return 1.0

    def _get_sl_multiplier(self, regime: Dict, pair_profile: Dict) -> float:
        """
        Get adaptive SL multiplier based on regime
        """
        base_multiplier = pair_profile.get('atr_sl_multiplier', 1.5)

        if not regime:
            return base_multiplier

        regime_type = regime.get('regime', '')
        volatility = regime.get('volatility_level', 'normal')

        # Wider stops in volatile markets
        if volatility == 'high' or regime_type == 'volatile':
            return base_multiplier * 1.3
        elif regime_type == 'choppy':
            return base_multiplier * 1.2
        elif regime_type in ['trending_bull', 'trending_bear']:
            return base_multiplier * 0.9  # Tighter stops in trends
        elif volatility == 'low':
            return base_multiplier * 0.8
        else:
            return base_multiplier

    def _get_tp_multiplier(self, regime: Dict, pair_profile: Dict) -> float:
        """
        Get adaptive TP multiplier based on regime
        """
        base_rr = pair_profile.get('min_rr', 1.5)
        base_sl_mult = pair_profile.get('atr_sl_multiplier', 1.5)

        # TP = SL * R:R ratio
        sl_mult = self._get_sl_multiplier(regime, pair_profile)

        if not regime:
            return sl_mult * base_rr

        regime_type = regime.get('regime', '')

        # Adjust R:R based on regime
        if regime_type in ['trending_bull', 'trending_bear']:
            # Higher targets in trending markets
            rr_ratio = base_rr * 1.3
        elif regime_type == 'ranging':
            # Lower targets in ranging markets
            rr_ratio = base_rr * 0.8
        elif regime_type == 'volatile':
            # Moderate targets in volatile markets
            rr_ratio = base_rr
        elif regime_type == 'choppy':
            # Quick profits in choppy markets
            rr_ratio = base_rr * 0.7
        else:
            rr_ratio = base_rr

        return sl_mult * rr_ratio


def get_adaptive_risk(entry_price: float, direction: str, pair: str,
                      atr: float, account_balance: float = 10000,
                      regime: Dict = None, spread: float = None) -> Dict:
    """
    Helper function to get complete risk parameters

    Args:
        entry_price: Entry price
        direction: 'long' or 'short'
        pair: Trading pair
        atr: Current ATR
        account_balance: Account balance
        regime: Market regime dict
        spread: Current spread in pips

    Returns:
        Dict with position_size, sl, tp, risk_amount, etc.
    """
    manager = AdaptiveRiskManager()

    # Calculate position size
    position = manager.calculate_position_size(
        account_balance=account_balance,
        pair=pair,
        atr=atr,
        price=entry_price,
        regime=regime,
        spread=spread
    )

    # Calculate SL/TP
    sl_tp = manager.calculate_sl_tp(
        entry_price=entry_price,
        direction=direction,
        pair=pair,
        atr=atr,
        regime=regime
    )

    # Combine results
    return {
        **position,
        **sl_tp,
        'entry_price': entry_price,
        'direction': direction,
        'pair': pair
    }
