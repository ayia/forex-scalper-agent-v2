#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market Regime Detector
Detects market regimes: Trending, Ranging, Volatile, Choppy
Used for adaptive strategy selection and risk management
"""

import numpy as np
from typing import Dict, Tuple
import pandas as pd
from datetime import datetime


class MarketRegime:
    """Market regime classification"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    VOLATILE = "volatile"
    CHOPPY = "choppy"
    QUIET = "quiet"


class RegimeDetector:
    """
    Detects market regime using multiple indicators:
    - ADX for trend strength
    - ATR for volatility
    - Linear regression for direction
    - Bollinger Bands for range detection
    """

    def __init__(self):
        # Thresholds for regime classification
        self.adx_trending = 25  # ADX > 25 = trending
        self.adx_weak = 20      # ADX < 20 = ranging/choppy
        self.atr_high_multiplier = 1.5  # ATR > 1.5x average = volatile
        self.atr_low_multiplier = 0.6   # ATR < 0.6x average = quiet

    def detect_regime(self, df: pd.DataFrame, pair: str = None) -> Dict:
        """
        Detect market regime from OHLC data

        Args:
            df: DataFrame with columns: open, high, low, close, volume
            pair: Trading pair (for logging)

        Returns:
            Dict with regime, confidence, and characteristics
        """
        if len(df) < 50:
            return {
                'regime': MarketRegime.QUIET,
                'confidence': 0.5,
                'trend_strength': 0,
                'volatility_level': 'unknown',
                'direction': 'neutral'
            }

        # Calculate indicators
        adx = self._calculate_adx(df)
        atr = self._calculate_atr(df)
        atr_avg = atr.rolling(20).mean().iloc[-1] if len(atr) > 20 else atr.mean()
        current_atr = atr.iloc[-1]

        # Trend direction
        direction, slope = self._calculate_trend_direction(df)

        # Bollinger Bands for range detection
        bb_width = self._calculate_bb_width(df)

        # Regime classification
        regime = self._classify_regime(
            adx=adx,
            atr=current_atr,
            atr_avg=atr_avg,
            direction=direction,
            bb_width=bb_width
        )

        # Calculate confidence (0-100)
        confidence = self._calculate_confidence(
            adx=adx,
            atr=current_atr,
            atr_avg=atr_avg,
            regime=regime
        )

        # Volatility level
        if current_atr > atr_avg * self.atr_high_multiplier:
            volatility_level = 'high'
        elif current_atr < atr_avg * self.atr_low_multiplier:
            volatility_level = 'low'
        else:
            volatility_level = 'normal'

        return {
            'regime': regime,
            'confidence': confidence,
            'trend_strength': adx,
            'volatility_level': volatility_level,
            'direction': direction,
            'atr': current_atr,
            'atr_avg': atr_avg,
            'bb_width': bb_width
        }

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Average Directional Index"""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up = high - high.shift()
        down = low.shift() - low

        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)

        # Smoothed indicators
        atr = tr.rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(period).mean()
        return atr

    def _calculate_trend_direction(self, df: pd.DataFrame, period: int = 20) -> Tuple[str, float]:
        """Calculate trend direction using linear regression"""
        close = df['close'].tail(period)
        x = np.arange(len(close))

        # Linear regression
        slope, intercept = np.polyfit(x, close.values, 1)

        # Normalize slope
        avg_price = close.mean()
        normalized_slope = (slope / avg_price) * 100

        if normalized_slope > 0.1:
            direction = 'bullish'
        elif normalized_slope < -0.1:
            direction = 'bearish'
        else:
            direction = 'neutral'

        return direction, normalized_slope

    def _calculate_bb_width(self, df: pd.DataFrame, period: int = 20, std: int = 2) -> float:
        """Calculate Bollinger Bands width (normalized)"""
        close = df['close']
        sma = close.rolling(period).mean()
        bb_std = close.rolling(period).std()

        upper = sma + (bb_std * std)
        lower = sma - (bb_std * std)

        bb_width = ((upper - lower) / sma * 100).iloc[-1]
        return bb_width if not pd.isna(bb_width) else 0

    def _classify_regime(self, adx: float, atr: float, atr_avg: float,
                         direction: str, bb_width: float) -> str:
        """Classify market regime based on indicators"""

        # High volatility override
        if atr > atr_avg * self.atr_high_multiplier:
            return MarketRegime.VOLATILE

        # Very low volatility
        if atr < atr_avg * self.atr_low_multiplier:
            return MarketRegime.QUIET

        # Trending market
        if adx > self.adx_trending:
            if direction == 'bullish':
                return MarketRegime.TRENDING_BULL
            elif direction == 'bearish':
                return MarketRegime.TRENDING_BEAR
            else:
                # Strong ADX but no clear direction = choppy
                return MarketRegime.CHOPPY

        # Weak trend
        if adx < self.adx_weak:
            # Narrow Bollinger Bands = ranging
            if bb_width < 3:
                return MarketRegime.RANGING
            else:
                return MarketRegime.CHOPPY

        # Default: ranging
        return MarketRegime.RANGING

    def _calculate_confidence(self, adx: float, atr: float,
                             atr_avg: float, regime: str) -> int:
        """Calculate confidence in regime detection (0-100)"""
        confidence = 50  # Base confidence

        # Strong ADX increases confidence
        if adx > 30:
            confidence += 20
        elif adx > 25:
            confidence += 10
        elif adx < 15:
            confidence -= 10

        # Clear volatility signals increase confidence
        atr_ratio = atr / atr_avg if atr_avg > 0 else 1
        if atr_ratio > 2 or atr_ratio < 0.5:
            confidence += 15
        elif atr_ratio > 1.5 or atr_ratio < 0.7:
            confidence += 10

        # Regime-specific adjustments
        if regime == MarketRegime.VOLATILE and atr_ratio > 1.8:
            confidence += 10
        elif regime in [MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR]:
            if adx > 30:
                confidence += 10

        return max(0, min(100, confidence))


def get_regime(df: pd.DataFrame, pair: str = None) -> Dict:
    """
    Helper function to get market regime

    Args:
        df: OHLC DataFrame
        pair: Trading pair name

    Returns:
        Dict with regime information
    """
    detector = RegimeDetector()
    return detector.detect_regime(df, pair)
