"""
Breakout Strategy - Donchian Channels + Volume Confirmation
============================================================
Detects breakouts using Donchian Channels with advanced volume confirmation.

Volume Confirmation Features:
- Volume spike detection (current vs average)
- Volume trend analysis (increasing volume on breakout)
- Relative Volume Index (RVI)
- Volume-Price Trend (VPT) confirmation
- Accumulation/Distribution validation
"""
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Optional, Dict, Tuple
from loguru import logger

from base_strategy import BaseStrategy, Signal
from config import STRATEGY_PARAMS


class VolumeAnalyzer:
    """
    Advanced volume analysis for breakout confirmation.
    """

    def __init__(self):
        self.volume_ma_period = 20
        self.volume_spike_threshold = 1.5  # 150% of average
        self.strong_spike_threshold = 2.0   # 200% of average

    def analyze_volume(self, df: pd.DataFrame) -> Dict:
        """
        Comprehensive volume analysis.

        Returns:
            Dict with volume metrics and confirmation signals
        """
        if 'volume' not in df.columns and 'Volume' not in df.columns:
            return self._no_volume_result()

        # Normalize column name
        vol_col = 'volume' if 'volume' in df.columns else 'Volume'
        volume = df[vol_col].astype(float)

        if volume.sum() == 0:
            return self._no_volume_result()

        # 1. Volume Moving Average & Ratio
        volume_ma = volume.rolling(window=self.volume_ma_period).mean()
        current_volume = volume.iloc[-1]
        avg_volume = volume_ma.iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # 2. Volume Spike Detection
        is_spike = volume_ratio >= self.volume_spike_threshold
        is_strong_spike = volume_ratio >= self.strong_spike_threshold

        # 3. Volume Trend (last 5 candles)
        recent_volumes = volume.tail(5)
        volume_trend = self._calculate_volume_trend(recent_volumes)

        # 4. Relative Volume Index (RVI)
        rvi = self._calculate_rvi(df, vol_col)

        # 5. Volume-Price Trend (VPT)
        vpt, vpt_signal = self._calculate_vpt(df, vol_col)

        # 6. Accumulation/Distribution
        ad_line, ad_trend = self._calculate_ad(df, vol_col)

        # 7. On-Balance Volume (OBV) trend
        obv_trend = self._calculate_obv_trend(df, vol_col)

        # Calculate overall volume confirmation score (0-100)
        confirmation_score = self._calculate_confirmation_score(
            volume_ratio=volume_ratio,
            is_spike=is_spike,
            is_strong_spike=is_strong_spike,
            volume_trend=volume_trend,
            rvi=rvi,
            vpt_signal=vpt_signal,
            ad_trend=ad_trend,
            obv_trend=obv_trend
        )

        return {
            'has_volume_data': True,
            'current_volume': current_volume,
            'average_volume': avg_volume,
            'volume_ratio': round(volume_ratio, 2),
            'is_spike': is_spike,
            'is_strong_spike': is_strong_spike,
            'volume_trend': volume_trend,
            'rvi': round(rvi, 2),
            'vpt_signal': vpt_signal,
            'ad_trend': ad_trend,
            'obv_trend': obv_trend,
            'confirmation_score': confirmation_score,
            'is_confirmed': confirmation_score >= 60
        }

    def _no_volume_result(self) -> Dict:
        """Return neutral result when no volume data available."""
        return {
            'has_volume_data': False,
            'current_volume': 0,
            'average_volume': 0,
            'volume_ratio': 1.0,
            'is_spike': False,
            'is_strong_spike': False,
            'volume_trend': 'neutral',
            'rvi': 50,
            'vpt_signal': 'neutral',
            'ad_trend': 'neutral',
            'obv_trend': 'neutral',
            'confirmation_score': 50,
            'is_confirmed': True  # Don't penalize if no data
        }

    def _calculate_volume_trend(self, recent_volumes: pd.Series) -> str:
        """Determine if volume is increasing, decreasing, or neutral."""
        if len(recent_volumes) < 3:
            return 'neutral'

        # Linear regression slope
        x = np.arange(len(recent_volumes))
        slope = np.polyfit(x, recent_volumes.values, 1)[0]
        avg_vol = recent_volumes.mean()

        # Normalize slope
        normalized_slope = slope / avg_vol if avg_vol > 0 else 0

        if normalized_slope > 0.05:
            return 'increasing'
        elif normalized_slope < -0.05:
            return 'decreasing'
        return 'neutral'

    def _calculate_rvi(self, df: pd.DataFrame, vol_col: str) -> float:
        """
        Calculate Relative Volume Index.
        Compares up-volume to down-volume.
        Returns 0-100 (>50 bullish, <50 bearish)
        """
        close = df['Close'] if 'Close' in df.columns else df['close']
        volume = df[vol_col]

        # Calculate price change
        price_change = close.diff()

        # Separate up and down volume
        up_volume = volume.where(price_change > 0, 0)
        down_volume = volume.where(price_change < 0, 0)

        # Rolling sums
        up_sum = up_volume.rolling(14).sum()
        down_sum = down_volume.rolling(14).sum()

        # RVI calculation
        total = up_sum.iloc[-1] + down_sum.iloc[-1]
        if total == 0:
            return 50

        rvi = (up_sum.iloc[-1] / total) * 100
        return rvi

    def _calculate_vpt(self, df: pd.DataFrame, vol_col: str) -> Tuple[float, str]:
        """
        Calculate Volume-Price Trend.
        VPT = Previous VPT + Volume × (Current Close − Previous Close) / Previous Close
        """
        close = df['Close'] if 'Close' in df.columns else df['close']
        volume = df[vol_col]

        price_change_pct = close.pct_change()
        vpt = (volume * price_change_pct).cumsum()

        # VPT signal (compare to its MA)
        vpt_ma = vpt.rolling(10).mean()
        current_vpt = vpt.iloc[-1]
        current_vpt_ma = vpt_ma.iloc[-1]

        if current_vpt > current_vpt_ma * 1.02:
            signal = 'bullish'
        elif current_vpt < current_vpt_ma * 0.98:
            signal = 'bearish'
        else:
            signal = 'neutral'

        return current_vpt, signal

    def _calculate_ad(self, df: pd.DataFrame, vol_col: str) -> Tuple[float, str]:
        """
        Calculate Accumulation/Distribution Line.
        AD = ((Close - Low) - (High - Close)) / (High - Low) × Volume
        """
        high = df['High'] if 'High' in df.columns else df['high']
        low = df['Low'] if 'Low' in df.columns else df['low']
        close = df['Close'] if 'Close' in df.columns else df['close']
        volume = df[vol_col]

        # Money Flow Multiplier
        hl_range = high - low
        mf_multiplier = ((close - low) - (high - close)) / hl_range.replace(0, np.nan)
        mf_multiplier = mf_multiplier.fillna(0)

        # Money Flow Volume
        mf_volume = mf_multiplier * volume

        # AD Line
        ad_line = mf_volume.cumsum()

        # Trend (last 10 periods)
        ad_recent = ad_line.tail(10)
        if len(ad_recent) >= 3:
            slope = np.polyfit(np.arange(len(ad_recent)), ad_recent.values, 1)[0]
            if slope > 0:
                trend = 'accumulation'
            elif slope < 0:
                trend = 'distribution'
            else:
                trend = 'neutral'
        else:
            trend = 'neutral'

        return ad_line.iloc[-1], trend

    def _calculate_obv_trend(self, df: pd.DataFrame, vol_col: str) -> str:
        """
        Calculate On-Balance Volume trend.
        """
        close = df['Close'] if 'Close' in df.columns else df['close']
        volume = df[vol_col]

        # OBV calculation
        obv = (np.sign(close.diff()) * volume).cumsum()

        # OBV trend (last 10 periods)
        obv_ma = obv.rolling(10).mean()

        if obv.iloc[-1] > obv_ma.iloc[-1]:
            return 'bullish'
        elif obv.iloc[-1] < obv_ma.iloc[-1]:
            return 'bearish'
        return 'neutral'

    def _calculate_confirmation_score(
        self,
        volume_ratio: float,
        is_spike: bool,
        is_strong_spike: bool,
        volume_trend: str,
        rvi: float,
        vpt_signal: str,
        ad_trend: str,
        obv_trend: str
    ) -> int:
        """
        Calculate overall volume confirmation score (0-100).
        """
        score = 50  # Base score

        # Volume spike contribution (max 20 points)
        if is_strong_spike:
            score += 20
        elif is_spike:
            score += 12
        elif volume_ratio >= 1.2:
            score += 5
        elif volume_ratio < 0.8:
            score -= 10

        # Volume trend contribution (max 15 points)
        if volume_trend == 'increasing':
            score += 15
        elif volume_trend == 'decreasing':
            score -= 10

        # RVI contribution (max 10 points)
        if rvi > 60:
            score += 10
        elif rvi > 55:
            score += 5
        elif rvi < 40:
            score -= 5

        # VPT signal contribution (max 10 points)
        if vpt_signal == 'bullish':
            score += 10
        elif vpt_signal == 'bearish':
            score -= 5

        # A/D trend contribution (max 10 points)
        if ad_trend == 'accumulation':
            score += 10
        elif ad_trend == 'distribution':
            score -= 5

        # OBV trend contribution (max 5 points)
        if obv_trend == 'bullish':
            score += 5
        elif obv_trend == 'bearish':
            score -= 3

        return max(0, min(100, score))


class BreakoutStrategy(BaseStrategy):
    """
    Enhanced Breakout strategy using Donchian Channels with Volume Confirmation.

    Improvements over basic breakout:
    1. Volume spike confirmation
    2. Volume trend validation
    3. Multiple volume indicators (RVI, VPT, A/D, OBV)
    4. Confidence adjustment based on volume
    5. False breakout filtering
    """

    def __init__(self):
        super().__init__("Breakout")
        self.donchian_period = STRATEGY_PARAMS.get("donchian_period", 20)
        self.volume_analyzer = VolumeAnalyzer()

        # Breakout configuration
        self.require_volume_confirmation = True
        self.min_volume_score = 55  # Minimum volume confirmation score
        self.volume_confidence_boost = 15  # Max confidence boost from volume

    def generate_signal(self, df: pd.DataFrame, pair: str, timeframe: str) -> Optional[Signal]:
        """
        Generate trading signal for scanner_v2.py compatibility.

        Args:
            df: OHLCV DataFrame
            pair: Trading pair
            timeframe: Timeframe string

        Returns:
            Signal object or None
        """
        result = self.analyze(df, pair)
        if result:
            # Calculate basic SL/TP using ATR
            high_col = 'High' if 'High' in df.columns else 'high'
            low_col = 'Low' if 'Low' in df.columns else 'low'
            atr = df[high_col].rolling(14).max() - df[low_col].rolling(14).min()
            current_atr = atr.iloc[-1] if len(atr) > 0 else 0.001

            entry = result['entry_price']
            if result['direction'] == 'BUY':
                sl = entry - (current_atr * 1.5)
                tp = entry + (current_atr * 2.5)
            else:  # SELL
                sl = entry + (current_atr * 1.5)
                tp = entry - (current_atr * 2.5)

            from datetime import datetime
            return Signal(
                pair=pair,
                direction=result['direction'],
                entry_price=entry,
                stop_loss=sl,
                take_profit=tp,
                confidence=result['confidence'],
                strategy=self.name,
                timeframe=timeframe,
                timestamp=datetime.now().isoformat()
            )
        return None

    def analyze(self, df: pd.DataFrame, pair: str) -> Optional[Dict]:
        """
        Analyze price action for breakout signals with volume confirmation.

        Args:
            df: OHLCV DataFrame
            pair: Trading pair symbol

        Returns:
            Signal dictionary or None
        """
        try:
            if df is None or len(df) < self.donchian_period:
                return None

            # Normalize column names
            high_col = 'High' if 'High' in df.columns else 'high'
            low_col = 'Low' if 'Low' in df.columns else 'low'
            close_col = 'Close' if 'Close' in df.columns else 'close'

            # Calculate Donchian Channels
            high_channel = df[high_col].rolling(window=self.donchian_period).max()
            low_channel = df[low_col].rolling(window=self.donchian_period).min()

            current_price = df[close_col].iloc[-1]
            current_high = high_channel.iloc[-2]  # Previous high
            current_low = low_channel.iloc[-2]    # Previous low

            # Calculate breakout strength (how far price broke out)
            channel_width = current_high - current_low
            breakout_strength = 0

            # Check for breakouts
            direction = None
            base_confidence = 50
            reason = ""

            if current_price > current_high:
                direction = "BUY"
                base_confidence = 65
                breakout_strength = (current_price - current_high) / channel_width * 100
                reason = f"Bullish breakout above {current_high:.5f}"
            elif current_price < current_low:
                direction = "SELL"
                base_confidence = 65
                breakout_strength = (current_low - current_price) / channel_width * 100
                reason = f"Bearish breakout below {current_low:.5f}"

            if not direction:
                return None

            # Volume Analysis
            volume_analysis = self.volume_analyzer.analyze_volume(df)

            # Check volume confirmation if required
            if self.require_volume_confirmation and volume_analysis['has_volume_data']:
                if not volume_analysis['is_confirmed']:
                    logger.debug(
                        f"{pair}: Breakout rejected - insufficient volume "
                        f"(score: {volume_analysis['confirmation_score']})"
                    )
                    return None

                # Check direction alignment for directional volume indicators
                if direction == "BUY":
                    if volume_analysis['ad_trend'] == 'distribution':
                        logger.debug(f"{pair}: Bullish breakout with distribution - rejected")
                        return None
                    if volume_analysis['rvi'] < 40:
                        logger.debug(f"{pair}: Bullish breakout with bearish RVI - rejected")
                        return None
                else:  # SELL
                    if volume_analysis['ad_trend'] == 'accumulation':
                        logger.debug(f"{pair}: Bearish breakout with accumulation - rejected")
                        return None
                    if volume_analysis['rvi'] > 60:
                        logger.debug(f"{pair}: Bearish breakout with bullish RVI - rejected")
                        return None

            # Calculate final confidence with volume boost
            confidence = base_confidence

            if volume_analysis['has_volume_data']:
                # Volume score contribution (0-15 points)
                volume_boost = (volume_analysis['confirmation_score'] - 50) / 50 * self.volume_confidence_boost
                confidence += volume_boost

                # Strong volume spike bonus
                if volume_analysis['is_strong_spike']:
                    confidence += 5

                # Increasing volume bonus
                if volume_analysis['volume_trend'] == 'increasing':
                    confidence += 3

            # Breakout strength bonus (0-5 points)
            if breakout_strength > 10:
                confidence += min(5, breakout_strength / 2)

            confidence = max(50, min(95, confidence))

            return {
                'direction': direction,
                'confidence': round(confidence, 1),
                'entry_price': current_price,
                'reason': reason,
                'breakout_strength': round(breakout_strength, 2),
                'indicators': {
                    'donchian_high': current_high,
                    'donchian_low': current_low,
                    'channel_width': channel_width
                },
                'volume_analysis': volume_analysis
            }

        except Exception as e:
            logger.error(f"Breakout analysis error: {e}")
            return None

    def analyze_without_volume(self, df: pd.DataFrame, pair: str) -> Optional[Dict]:
        """
        Analyze breakout without volume confirmation (fallback method).
        """
        original_setting = self.require_volume_confirmation
        self.require_volume_confirmation = False
        result = self.analyze(df, pair)
        self.require_volume_confirmation = original_setting
        return result
