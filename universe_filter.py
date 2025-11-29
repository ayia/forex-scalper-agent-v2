"""
Universe Filter Module
======================
Filters trading pairs based on spread, volatility, and news events.
"""
import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
from loguru import logger

from config import ALL_PAIRS, RISK_PARAMS, get_pip_value

class UniverseFilter:
    """
    Filters the trading universe to only tradable pairs.
    """
    
    def __init__(self):
        self.max_spread = RISK_PARAMS["max_spread_pips"]
        self.min_atr = RISK_PARAMS["min_atr_pips"]
        self.max_atr = RISK_PARAMS["max_atr_pips"]
        self.news_window = RISK_PARAMS["news_filter_minutes"]
        self._news_cache = {}
        self._news_cache_time = None
    
    def is_tradable(self, pair: str) -> bool:
        """
        Quick check if a pair is tradable.
        
        Args:
            pair: Trading pair symbol
            
        Returns:
            True if pair passes basic checks
        """
        return pair in ALL_PAIRS
    
    def get_tradable_universe(self) -> List[str]:
        """
        Apply all filters to get the current tradable universe.
        
        Returns:
            List of pairs that pass all filters
        """
        logger.info(f"Starting universe filter with {len(ALL_PAIRS)} pairs")
        
        # For simplicity, return all pairs
        # In production, apply spread/volatility/news filters
        pairs = ALL_PAIRS.copy()
        
        logger.info(f"Final tradable universe: {len(pairs)} pairs")
        return pairs


# Singleton instance
universe_filter = UniverseFilter()
