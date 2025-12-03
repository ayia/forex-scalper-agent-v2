#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Correlation Manager
Manages correlation risk across trading pairs
"""

from typing import Dict, List
import pandas as pd
import numpy as np


class CorrelationManager:
    """
    Tracks correlations between pairs and adjusts risk
    """

    def __init__(self):
        # Pre-defined correlation groups (approximate)
        self.correlation_groups = {
            'eur_cluster': ['EURUSD', 'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD'],
            'usd_cluster': ['EURUSD', 'GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD', 'USDCHF', 'USDJPY'],
            'jpy_cluster': ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'CADJPY', 'CHFJPY'],
            'aud_nzd_cluster': ['AUDUSD', 'NZDUSD', 'AUDNZD', 'AUDJPY', 'NZDJPY'],
            'commodity_fx': ['AUDUSD', 'NZDUSD', 'CADUSD', 'USDCAD']
        }

        # Maximum number of correlated pairs to trade simultaneously
        self.max_correlated_trades = 2

    def check_correlation_risk(self, active_pairs: List[str],
                               new_pair: str) -> Dict:
        """
        Check if adding new_pair violates correlation limits

        Args:
            active_pairs: Currently open/active pairs
            new_pair: Pair being considered

        Returns:
            Dict with allow_trade, correlation_count, reason
        """
        # Count how many correlated pairs are already active
        correlated_count = 0
        correlated_with = []

        for group_name, group_pairs in self.correlation_groups.items():
            if new_pair in group_pairs:
                # Check how many from this group are active
                for active in active_pairs:
                    if active in group_pairs and active != new_pair:
                        correlated_count += 1
                        correlated_with.append(active)

        # Decide if we can trade
        allow_trade = correlated_count < self.max_correlated_trades

        return {
            'allow_trade': allow_trade,
            'correlated_count': correlated_count,
            'correlated_with': correlated_with,
            'reason': f"Already {correlated_count} correlated pairs" if not allow_trade else "OK"
        }

    def get_correlation_adjustment(self, pair: str, active_pairs: List[str]) -> float:
        """
        Get position size multiplier based on correlation exposure

        Returns:
            Float multiplier (0.5 - 1.0)
        """
        correlated_count = 0

        for group_pairs in self.correlation_groups.values():
            if pair in group_pairs:
                for active in active_pairs:
                    if active in group_pairs and active != pair:
                        correlated_count += 1

        # Reduce position size for each correlated pair
        if correlated_count == 0:
            return 1.0
        elif correlated_count == 1:
            return 0.8
        elif correlated_count == 2:
            return 0.6
        else:
            return 0.5


def check_pair_correlation(active_pairs: List[str], new_pair: str) -> Dict:
    """
    Helper function to check correlation
    """
    manager = CorrelationManager()
    return manager.check_correlation_risk(active_pairs, new_pair)
