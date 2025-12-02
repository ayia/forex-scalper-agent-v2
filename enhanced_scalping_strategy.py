"""
Enhanced Scalping Strategy - Advanced Multi-Confirmation System
Combines best practices from DIY Custom Strategy Builder with our adaptive system.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from base_strategy import BaseStrategy, Signal
from config import STRATEGY_PARAMS, get_pip_value
from advanced_indicators import (
    RangeFilter, ChandelierExit, WaddahAttarExplosion,
    ChoppinessIndex, PVSRA, SupplyDemandZones, SessionManager,
    QQEMod, ConfirmationSystem, calculate_adaptive_sl_tp
)


@dataclass
class EnhancedSignal:
    """Enhanced signal with detailed metadata."""
    pair: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    strategy: str
    timeframe: str
    timestamp: str

    # Additional metadata
    leading_indicator: str
    confirmations_passed: List[str]
    confirmations_failed: List[str]
    session_info: Dict
    pvsra_analysis: Dict
    choppiness_value: float
    is_trending: bool
    supply_demand_zones: Dict


class EnhancedScalpingStrategy(BaseStrategy):
    """
    Enhanced scalping strategy with:
    - Leading indicator + multiple confirmations
    - Signal expiry system
    - PVSRA volume analysis
    - Supply/Demand zone-based SL/TP
    - Session-aware trading
    - Choppiness filter
    """

    def __init__(self, leading_indicator: str = 'RANGE_FILTER', signal_expiry: int = 3):
        """
        Initialize enhanced scalping strategy.

        Args:
            leading_indicator: Primary signal generator
                Options: 'RANGE_FILTER', 'CHANDELIER', 'WAE', 'QQE'
            signal_expiry: Number of candles to wait for confirmations
        """
        super().__init__("EnhancedScalping")

        self.leading_indicator = leading_indicator
        self.signal_expiry = signal_expiry

        # Initialize indicators
        self.range_filter = RangeFilter(period=100, multiplier=3.0, filter_type='DW')
        self.chandelier_exit = ChandelierExit(atr_period=22, multiplier=3.0)
        self.wae = WaddahAttarExplosion(sensitivity=150)
        self.choppiness = ChoppinessIndex(length=14, threshold=61.8)
        self.pvsra = PVSRA(volume_period=10)
        self.supply_demand = SupplyDemandZones(swing_length=10)
        self.session_manager = SessionManager()
        self.qqe = QQEMod(rsi_period=6, smoothing=5, qqe_factor=3.0)
        self.confirmation_system = ConfirmationSystem(signal_expiry=signal_expiry)

        # Track signal states
        self.leading_signal_candles = {}  # pair -> candles since leading signal

    def analyze(self, data: Dict[str, pd.DataFrame], pair: str) -> Optional[Signal]:
        """Analyze data and return signal if conditions are met."""

        # Get M15 data for primary analysis
        timeframe_key = None
        for key in ['M15', '15m', 'M5', '5m']:
            if key in data and data[key] is not None and len(data[key]) >= 100:
                timeframe_key = key
                break

        if timeframe_key is None:
            return None

        df = data[timeframe_key]

        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required):
            return None

        # Generate enhanced signal
        enhanced_signal = self._generate_enhanced_signal(df, pair, timeframe_key)

        if enhanced_signal is None:
            return None

        # Convert to base Signal format
        return Signal(
            pair=enhanced_signal.pair,
            direction=enhanced_signal.direction,
            entry_price=enhanced_signal.entry_price,
            stop_loss=enhanced_signal.stop_loss,
            take_profit=enhanced_signal.take_profit,
            confidence=enhanced_signal.confidence,
            strategy=enhanced_signal.strategy,
            timeframe=enhanced_signal.timeframe,
            timestamp=enhanced_signal.timestamp
        )

    def _generate_enhanced_signal(self, df: pd.DataFrame, pair: str,
                                   timeframe: str) -> Optional[EnhancedSignal]:
        """Generate enhanced signal with full analysis."""

        current_price = df['close'].iloc[-1]
        pip_value = get_pip_value(pair)

        # 1. Check Choppiness - Avoid ranging markets
        ci_value, is_trending = self.choppiness.calculate(df)
        if not is_trending:
            return None  # Skip choppy markets

        # 2. Get Session Info
        session_info = self.session_manager.get_current_session()
        session_score = self.session_manager.get_pair_session_score(pair)

        # Skip if pair not suitable for current session (unless overlap)
        if session_score < 50 and not session_info['is_overlap']:
            return None

        # 3. Get Leading Indicator Signal
        leading_result = self._get_leading_signal(df)
        if leading_result.signal == 'NEUTRAL':
            return None

        direction = leading_result.signal

        # 4. Get Confirmation Signals
        confirmations = self._get_confirmations(df, direction)
        confirmations_passed = [c for c in confirmations if c.signal == direction]
        confirmations_failed = [c for c in confirmations if c.signal != direction and c.signal != 'NEUTRAL']

        # Calculate confirmation rate
        total_confirmations = len(confirmations)
        passed_rate = len(confirmations_passed) / total_confirmations if total_confirmations > 0 else 0

        # Need at least 70% confirmation rate for strong signal
        if passed_rate < 0.6:
            return None

        # 5. PVSRA Volume Analysis
        pvsra_result = self.pvsra.calculate(df)

        # Check for volume confirmation
        volume_confirms = False
        if direction == 'BUY' and pvsra_result['is_bull']:
            volume_confirms = True
        elif direction == 'SELL' and not pvsra_result['is_bull']:
            volume_confirms = True

        # Climax volume is strong confirmation
        if pvsra_result['is_climax']:
            volume_confirms = True

        # 6. Supply/Demand Zone Analysis for SL/TP
        sd_zones = self.supply_demand.calculate(df)

        # Calculate ATR for SL/TP
        atr = self._calculate_atr(df)

        # Get adaptive SL/TP based on zones and session volatility
        stop_loss, take_profit = calculate_adaptive_sl_tp(
            entry=current_price,
            direction='BUY' if direction == 'LONG' else 'SELL',
            atr=atr,
            supply_demand=sd_zones,
            session_volatility=session_info['volatility']
        )

        # Validate SL/TP
        sl_pips = abs(current_price - stop_loss) / pip_value
        tp_pips = abs(take_profit - current_price) / pip_value

        # Risk/Reward check (minimum 1.5:1)
        if tp_pips / sl_pips < 1.5:
            return None

        # Maximum SL check (15 pips for scalping)
        if sl_pips > 15:
            return None

        # 7. Calculate Final Confidence
        confidence = self._calculate_confidence(
            leading_strength=leading_result.strength,
            confirmation_rate=passed_rate,
            session_score=session_score,
            volume_confirms=volume_confirms,
            is_trending=is_trending,
            ci_value=ci_value
        )

        # Minimum confidence threshold
        if confidence < 65:
            return None

        # Create enhanced signal
        return EnhancedSignal(
            pair=pair,
            direction='BUY' if direction == 'LONG' else 'SELL',
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            strategy=f"EnhancedScalping_{self.leading_indicator}",
            timeframe=timeframe,
            timestamp=datetime.now().isoformat(),
            leading_indicator=self.leading_indicator,
            confirmations_passed=[f"{c.signal}" for c in confirmations_passed],
            confirmations_failed=[f"{c.signal}" for c in confirmations_failed],
            session_info=session_info,
            pvsra_analysis=pvsra_result,
            choppiness_value=ci_value,
            is_trending=is_trending,
            supply_demand_zones=sd_zones
        )

    def _get_leading_signal(self, df: pd.DataFrame):
        """Get signal from the selected leading indicator."""
        if self.leading_indicator == 'RANGE_FILTER':
            return self.range_filter.calculate(df)
        elif self.leading_indicator == 'CHANDELIER':
            result, _, _ = self.chandelier_exit.calculate(df)
            return result
        elif self.leading_indicator == 'WAE':
            return self.wae.calculate(df)
        elif self.leading_indicator == 'QQE':
            return self.qqe.calculate(df)
        else:
            return self.range_filter.calculate(df)

    def _get_confirmations(self, df: pd.DataFrame, direction: str) -> list:
        """Get confirmation signals from multiple indicators."""
        confirmations = []

        # 1. Range Filter (if not leading)
        if self.leading_indicator != 'RANGE_FILTER':
            confirmations.append(self.range_filter.calculate(df))

        # 2. Chandelier Exit (if not leading)
        if self.leading_indicator != 'CHANDELIER':
            result, _, _ = self.chandelier_exit.calculate(df)
            confirmations.append(result)

        # 3. WAE (if not leading)
        if self.leading_indicator != 'WAE':
            confirmations.append(self.wae.calculate(df))

        # 4. QQE (if not leading)
        if self.leading_indicator != 'QQE':
            confirmations.append(self.qqe.calculate(df))

        # 5. EMA confirmation (price above/below EMA 50)
        ema_50 = df['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        current_price = df['close'].iloc[-1]

        from advanced_indicators import IndicatorResult
        if current_price > ema_50:
            confirmations.append(IndicatorResult('LONG', 60, ema_50))
        else:
            confirmations.append(IndicatorResult('SHORT', 60, ema_50))

        # 6. RSI confirmation (not overbought/oversold)
        rsi = self._calculate_rsi(df['close'].values, 14)
        if direction == 'LONG' and rsi < 70:
            confirmations.append(IndicatorResult('LONG', 50, rsi))
        elif direction == 'SHORT' and rsi > 30:
            confirmations.append(IndicatorResult('SHORT', 50, rsi))
        else:
            confirmations.append(IndicatorResult('NEUTRAL', 30, rsi))

        return confirmations

    def _calculate_confidence(self, leading_strength: float, confirmation_rate: float,
                             session_score: float, volume_confirms: bool,
                             is_trending: bool, ci_value: float) -> float:
        """Calculate final confidence score."""

        # Base confidence from leading indicator
        base = leading_strength * 0.3  # Max 30 points

        # Confirmation bonus
        conf_bonus = confirmation_rate * 25  # Max 25 points

        # Session bonus
        session_bonus = (session_score / 100) * 15  # Max 15 points

        # Volume bonus
        volume_bonus = 10 if volume_confirms else 0

        # Trend bonus (lower choppiness = stronger trend)
        trend_bonus = (61.8 - ci_value) / 61.8 * 20 if is_trending else 0

        total = base + conf_bonus + session_bonus + volume_bonus + trend_bonus

        return min(100, max(0, total))

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate ATR."""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
        tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))

        return pd.Series(tr).rolling(period).mean().iloc[-1]

    def _calculate_rsi(self, close: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).ewm(span=period, adjust=False).mean().iloc[-1]
        avg_loss = pd.Series(loss).ewm(span=period, adjust=False).mean().iloc[-1]

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))


class AdaptiveLeadingSelector:
    """
    Automatically selects the best leading indicator based on:
    - Current market regime
    - Session
    - Pair characteristics
    """

    REGIME_INDICATORS = {
        'TRENDING': ['RANGE_FILTER', 'CHANDELIER'],
        'VOLATILE': ['WAE', 'QQE'],
        'RANGING': ['QQE', 'RANGE_FILTER'],  # With tighter thresholds
    }

    SESSION_PREFERENCES = {
        'TOKYO': ['QQE', 'RANGE_FILTER'],  # Lower volatility sessions
        'LONDON': ['WAE', 'CHANDELIER'],  # High volatility
        'NEW_YORK': ['WAE', 'RANGE_FILTER'],  # High volatility
        'OVERLAP': ['WAE', 'CHANDELIER'],  # Maximum volatility
    }

    def select_leading_indicator(self, regime: str, session: str,
                                  pair: str) -> str:
        """Select optimal leading indicator for current conditions."""

        # High priority: session-specific selection during overlap
        if session == 'OVERLAP' or session in ['LONDON', 'NEW_YORK']:
            # High volatility - use momentum-based
            return 'WAE'

        # JPY pairs during Tokyo
        if 'JPY' in pair and session == 'TOKYO':
            return 'QQE'

        # Based on regime
        if regime == 'TRENDING':
            return 'RANGE_FILTER'
        elif regime == 'VOLATILE':
            return 'WAE'
        else:
            return 'QQE'


def create_enhanced_strategy(pair: str, regime: str = 'TRENDING') -> EnhancedScalpingStrategy:
    """
    Factory function to create an enhanced strategy
    with automatically selected parameters.
    """
    selector = AdaptiveLeadingSelector()
    session_manager = SessionManager()

    session_info = session_manager.get_current_session()
    active_session = session_info['active_sessions'][0] if session_info['active_sessions'] else 'LONDON'

    if session_info['is_overlap']:
        active_session = 'OVERLAP'

    leading = selector.select_leading_indicator(regime, active_session, pair)

    # Adjust signal expiry based on timeframe and volatility
    if session_info['volatility'] == 'HIGH':
        expiry = 2  # Faster signals during high volatility
    else:
        expiry = 3

    return EnhancedScalpingStrategy(leading_indicator=leading, signal_expiry=expiry)
