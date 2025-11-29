"""Risk Calculator Module - ATR-based SL/TP calculation"""
import pandas as pd
from typing import Dict, Tuple
from config import RISK_PARAMS, get_pip_value


class RiskCalculator:
    """Calculates risk parameters for trades."""
    
    def __init__(self):
        self.max_sl_pips = RISK_PARAMS["max_sl_pips"]
        self.sl_atr_multiplier = RISK_PARAMS["sl_atr_multiplier"]
        self.min_rr_ratio = RISK_PARAMS["min_rr_ratio"]
        self.target_rr_ratio = RISK_PARAMS["target_rr_ratio"]
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate current ATR value."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean().iloc[-1]
    
    def calculate_sl_tp(self, pair: str, direction: str,
                        entry_price: float, df: pd.DataFrame) -> Tuple[float, float]:
        """Calculate SL and TP based on ATR."""
        pip_value = get_pip_value(pair)
        atr = self.calculate_atr(df)
        atr_pips = atr / pip_value
        
        # Calculate SL in pips
        sl_pips = min(atr_pips * self.sl_atr_multiplier, self.max_sl_pips)
        tp_pips = sl_pips * self.target_rr_ratio
        
        if direction == "BUY":
            stop_loss = entry_price - (sl_pips * pip_value)
            take_profit = entry_price + (tp_pips * pip_value)
        else:
            stop_loss = entry_price + (sl_pips * pip_value)
            take_profit = entry_price - (tp_pips * pip_value)
        
        return stop_loss, take_profit
    
    def validate_risk_reward(self, entry: float, sl: float, tp: float) -> bool:
        """Validate if trade meets minimum RR ratio."""
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        
        if risk == 0:
            return False
        
        rr_ratio = reward / risk
        return rr_ratio >= self.min_rr_ratio
    
    def calculate_position_size(self, account_balance: float,
                                risk_percent: float,
                                sl_pips: float,
                                pip_value_per_lot: float = 10.0) -> float:
        """Calculate position size based on risk."""
        risk_amount = account_balance * (risk_percent / 100)
        position_size = risk_amount / (sl_pips * pip_value_per_lot)
        return round(position_size, 2)
