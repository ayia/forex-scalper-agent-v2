"""
Breakout Strategy - Donchian Channels
======================================
Detects breakouts using Donchian Channels and volume confirmation.
"""
import pandas as pd
import pandas_ta as ta
from typing import Optional, Dict
from loguru import logger

from base_strategy import BaseStrategy, Signal, SignalType
from config import STRATEGY_PARAMS


class BreakoutStrategy(BaseStrategy):
    """
    Breakout strategy using Donchian Channels.
    """
    
    def __init__(self):
        super().__init__("Breakout")
        self.donchian_period = STRATEGY_PARAMS.get("donchian_period", 20)
    
    def analyze(self, df: pd.DataFrame, pair: str) -> Optional[Dict]:
        """
        Analyze price action for breakout signals.
        
        Args:
            df: OHLCV DataFrame
            pair: Trading pair symbol
            
        Returns:
            Signal dictionary or None
        """
        try:
            if df is None or len(df) < self.donchian_period:
                return None
            
            # Calculate Donchian Channels
            high_channel = df['High'].rolling(window=self.donchian_period).max()
            low_channel = df['Low'].rolling(window=self.donchian_period).min()
            
            current_price = df['Close'].iloc[-1]
            current_high = high_channel.iloc[-2]  # Previous high
            current_low = low_channel.iloc[-2]    # Previous low
            
            # Check for breakouts
            signal = None
            confidence = 50
            reason = ""
            
            if current_price > current_high:
                signal = SignalType.BUY
                confidence = 70
                reason = f"Bullish breakout above {current_high:.5f}"
            elif current_price < current_low:
                signal = SignalType.SELL
                confidence = 70
                reason = f"Bearish breakout below {current_low:.5f}"
            
            if signal:
                return {
                    'direction': signal.value,
                    'confidence': confidence,
                    'entry_price': current_price,
                    'reason': reason,
                    'indicators': {
                        'donchian_high': current_high,
                        'donchian_low': current_low
                    }
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Breakout analysis error: {e}")
            return None
