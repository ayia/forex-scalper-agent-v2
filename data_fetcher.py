"""Data Fetcher Module - Fetches Forex data using yfinance"""
import pandas as pd
import yfinance as yf
from typing import Dict, Optional
from datetime import datetime, timedelta
from config import TIMEFRAMES, ALL_PAIRS


class DataFetcher:
    """Fetches forex data from yfinance."""
    
    def __init__(self):
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_expiry = 60  # seconds
    
    def _get_yf_symbol(self, pair: str) -> str:
        """Convert pair to yfinance format."""
        return f"{pair}=X"
    
    def fetch_ohlcv(self, pair: str, timeframe: str = "M15",
                    bars: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data for a pair."""
        try:
            symbol = self._get_yf_symbol(pair)
            interval = TIMEFRAMES.get(timeframe, "15m")
            
            # Calculate period based on timeframe
            period_map = {
                "1m": "1d", "5m": "5d", "15m": "5d",
                "1h": "30d", "4h": "60d"
            }
            period = period_map.get(interval, "5d")
            
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                return None
            
            # Rename columns to lowercase
            df.columns = [c.lower() for c in df.columns]
            
            # Take only last N bars
            df = df.tail(bars)
            
            return df
        except Exception as e:
            print(f"Error fetching {pair}: {e}")
            return None
    
    def fetch_multi_timeframe(self, pair: str,
                              timeframes: list = None) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple timeframes."""
        if timeframes is None:
            timeframes = ["M1", "M5", "M15", "H1", "H4"]
        
        result = {}
        for tf in timeframes:
            data = self.fetch_ohlcv(pair, tf)
            if data is not None and len(data) > 50:
                result[tf] = data
        
        return result
    
    def get_spread(self, pair: str) -> float:
        """Estimate spread for a pair."""
        # Use typical spread values
        spread_map = {
            "EURUSD": 0.8, "GBPUSD": 1.0, "USDJPY": 0.9,
            "AUDUSD": 1.2, "USDCHF": 1.5, "USDCAD": 1.5,
            "NZDUSD": 1.8, "EURJPY": 1.5, "GBPJPY": 2.0
        }
        return spread_map.get(pair, 2.0)
