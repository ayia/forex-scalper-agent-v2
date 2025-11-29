"""Base Strategy Module - Abstract base class for all trading strategies"""
from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Signal:
    """Trading signal data class."""
    pair: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    strategy: str
    timeframe: str
    timestamp: str


class BaseStrategy(ABC):
    """Abstract base class for trading strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def analyze(self, data: Dict[str, pd.DataFrame], pair: str) -> Optional[Signal]:
        """Analyze market data and return a signal if conditions are met."""
        pass
    
    def calculate_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Calculate EMA for a given period."""
        return df['close'].ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR (Average True Range)."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    def calculate_macd(self, df: pd.DataFrame,
                       fast: int = 12, slow: int = 26,
                       signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """Calculate MACD and Signal line."""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal
