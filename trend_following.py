"""Trend Following Strategy - EMA Stack + MACD + SMC"""
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from base_strategy import BaseStrategy, Signal
from config import STRATEGY_PARAMS, get_pip_value


class TrendFollowingStrategy(BaseStrategy):
    """Trend following strategy using EMA stack and MACD."""
    
    def __init__(self):
        super().__init__("TrendFollowing")
        self.ema_fast = STRATEGY_PARAMS["ema_fast"]
        self.ema_medium = STRATEGY_PARAMS["ema_medium"]
        self.ema_slow = STRATEGY_PARAMS["ema_slow"]
    
    def analyze(self, data: Dict[str, pd.DataFrame], pair: str) -> Optional[Signal]:
        """Analyze for trend following signals."""
        if "M15" not in data or "H1" not in data:
            return None
        
        df_m15 = data["M15"]
        df_h1 = data["H1"]
        
        if len(df_m15) < 50 or len(df_h1) < 50:
            return None
        
        # Calculate EMAs on M15
        ema_fast = self.calculate_ema(df_m15, self.ema_fast)
        ema_medium = self.calculate_ema(df_m15, self.ema_medium)
        ema_slow = self.calculate_ema(df_m15, self.ema_slow)
        
        # Calculate MACD
        macd, signal = self.calculate_macd(df_m15)
        
        # Calculate ATR for SL/TP
        atr = self.calculate_atr(df_m15)
        
        current_price = df_m15['close'].iloc[-1]
        pip_value = get_pip_value(pair)
        
        # Check for bullish setup
        bullish = (
            ema_fast.iloc[-1] > ema_medium.iloc[-1] > ema_slow.iloc[-1] and
            macd.iloc[-1] > signal.iloc[-1] and
            macd.iloc[-2] <= signal.iloc[-2]  # MACD crossover
        )
        
        # Check for bearish setup
        bearish = (
            ema_fast.iloc[-1] < ema_medium.iloc[-1] < ema_slow.iloc[-1] and
            macd.iloc[-1] < signal.iloc[-1] and
            macd.iloc[-2] >= signal.iloc[-2]  # MACD crossunder
        )
        
        if not bullish and not bearish:
            return None
        
        direction = "BUY" if bullish else "SELL"
        atr_pips = atr.iloc[-1] / pip_value
        sl_pips = atr_pips * 1.5
        tp_pips = atr_pips * 3.0
        
        if direction == "BUY":
            stop_loss = current_price - (sl_pips * pip_value)
            take_profit = current_price + (tp_pips * pip_value)
        else:
            stop_loss = current_price + (sl_pips * pip_value)
            take_profit = current_price - (tp_pips * pip_value)
        
        return Signal(
            pair=pair,
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=75.0,
            strategy=self.name,
            timeframe="M15",
            timestamp=datetime.now().isoformat()
        )
