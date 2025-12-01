"""Mean Reversion Strategy - Bollinger Bands + RSI"""
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from base_strategy import BaseStrategy, Signal
from config import STRATEGY_PARAMS, get_pip_value


class MeanReversionStrategy(BaseStrategy):
    """Mean reversion strategy using Bollinger Bands and RSI."""
    
    def __init__(self):
        super().__init__("MeanReversion")
        self.bb_period = STRATEGY_PARAMS["bb_period"]
        self.bb_std = STRATEGY_PARAMS["bb_std"]
        self.rsi_period = STRATEGY_PARAMS["rsi_period"]
        self.rsi_overbought = STRATEGY_PARAMS["rsi_overbought"]
        self.rsi_oversold = STRATEGY_PARAMS["rsi_oversold"]
        self.max_adx_for_reversion = 30  # Don't trade mean reversion in strong trends
    
    def calculate_bollinger_bands(self, df: pd.DataFrame):
        """Calculate Bollinger Bands."""
        middle = df['close'].rolling(window=self.bb_period).mean()
        std = df['close'].rolling(window=self.bb_period).std()
        upper = middle + (std * self.bb_std)
        lower = middle - (std * self.bb_std)
        return upper, middle, lower

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (Average Directional Index) to measure trend strength."""
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        # Set negative values to 0
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smooth the values
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx
    
    def analyze(self, data: Dict[str, pd.DataFrame], pair: str) -> Optional[Signal]:
        """Analyze for mean reversion signals."""
        if "M15" not in data:
            return None
        
        df = data["M15"]
        if len(df) < 50:
            return None
        
        # Calculate indicators
        upper, middle, lower = self.calculate_bollinger_bands(df)
        rsi = self.calculate_rsi(df, self.rsi_period)
        atr = self.calculate_atr(df)
        adx = self.calculate_adx(df)

        current_price = df['close'].iloc[-1]
        current_adx = adx.iloc[-1]
        pip_value = get_pip_value(pair)

        # CRITICAL FILTER: Don't trade mean reversion in strong trends (ADX > 30)
        if current_adx > self.max_adx_for_reversion:
            # Strong trend detected - mean reversion is dangerous here
            return None

        # Oversold + price at lower band = BUY
        bullish = (
            current_price <= lower.iloc[-1] and
            rsi.iloc[-1] < self.rsi_oversold
        )
        
        # Overbought + price at upper band = SELL
        bearish = (
            current_price >= upper.iloc[-1] and
            rsi.iloc[-1] > self.rsi_overbought
        )
        
        if not bullish and not bearish:
            return None
        
        direction = "BUY" if bullish else "SELL"
        atr_pips = atr.iloc[-1] / pip_value
        
        if direction == "BUY":
            stop_loss = current_price - (atr_pips * 1.5 * pip_value)
            take_profit = middle.iloc[-1]  # Target: middle band
        else:
            stop_loss = current_price + (atr_pips * 1.5 * pip_value)
            take_profit = middle.iloc[-1]
        
        return Signal(
            pair=pair,
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=70.0,
            strategy=self.name,
            timeframe="M15",
            timestamp=datetime.now().isoformat()
        )
