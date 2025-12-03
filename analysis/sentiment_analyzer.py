"""
Sentiment Analyzer Module - Version Complète
=============================================
Analyse le sentiment de marché via plusieurs sources:
1. COT (Commitment of Traders) - positions institutionnelles
2. Retail Sentiment (estimation basée sur indicateurs techniques)
3. Volatility Sentiment (VIX-like pour forex)
4. Technical Sentiment (indicateurs multi-timeframe)

Le sentiment est utilisé de manière CONTRARIANTE:
- Retail très bearish + signal BUY = FORT (fade the crowd)
- Retail très bullish + signal SELL = FORT (fade the crowd)
"""
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Analyseur de sentiment multi-source pour forex.

    Sources:
    1. Technical Sentiment - RSI/Stochastic multi-TF
    2. Volatility Sentiment - ATR ratio analysis
    3. Momentum Sentiment - Price action analysis
    4. COT-like estimation - Position accumulation
    """

    def __init__(self, data_fetcher=None):
        """
        Initialize sentiment analyzer.

        Args:
            data_fetcher: DataFetcher instance for price data
        """
        self.data_fetcher = data_fetcher
        self.sentiment_cache = {}  # Cache sentiment for performance
        self.cache_duration = timedelta(minutes=15)  # Refresh every 15 min

        # Sentiment thresholds
        self.extreme_bullish = 70   # Above this = extreme bullish
        self.extreme_bearish = 30   # Below this = extreme bearish

        logger.info("SentimentAnalyzer initialized")

    def analyze(self, pair: str, data: pd.DataFrame = None) -> Dict:
        """
        Analyze sentiment for a currency pair.

        Args:
            pair: Trading pair (e.g., "EURUSD")
            data: Optional DataFrame with OHLCV data

        Returns:
            Sentiment dictionary with:
            - score: -100 to +100 (negative=bearish, positive=bullish)
            - strength: 'extreme_bearish', 'bearish', 'neutral', 'bullish', 'extreme_bullish'
            - retail_sentiment: Estimated retail positioning
            - institutional_bias: Estimated smart money direction
            - contrarian_signal: What contrarian approach suggests
        """
        # Check cache
        cache_key = f"{pair}_{datetime.now().strftime('%Y%m%d%H%M')}"
        if cache_key in self.sentiment_cache:
            return self.sentiment_cache[cache_key]

        # If no data provided, return neutral
        if data is None or data.empty:
            return self._neutral_sentiment(pair)

        try:
            # 1. Calculate Technical Sentiment (RSI-based)
            tech_sentiment = self._calculate_technical_sentiment(data)

            # 2. Calculate Volatility Sentiment
            vol_sentiment = self._calculate_volatility_sentiment(data)

            # 3. Calculate Momentum Sentiment
            momentum_sentiment = self._calculate_momentum_sentiment(data)

            # 4. Estimate Retail vs Institutional positioning
            retail_estimate, institutional_bias = self._estimate_positioning(data)

            # Combine sentiments (weighted average)
            weights = {
                'technical': 0.30,
                'volatility': 0.20,
                'momentum': 0.30,
                'positioning': 0.20
            }

            combined_score = (
                tech_sentiment * weights['technical'] +
                vol_sentiment * weights['volatility'] +
                momentum_sentiment * weights['momentum'] +
                retail_estimate * weights['positioning']
            )

            # Determine strength category
            strength = self._categorize_sentiment(combined_score)

            # Contrarian signal (opposite of retail)
            contrarian_signal = self._get_contrarian_signal(retail_estimate)

            result = {
                'pair': pair,
                'score': round(combined_score, 2),
                'strength': strength,
                'retail_sentiment': round(retail_estimate, 2),
                'institutional_bias': institutional_bias,
                'contrarian_signal': contrarian_signal,
                'components': {
                    'technical': round(tech_sentiment, 2),
                    'volatility': round(vol_sentiment, 2),
                    'momentum': round(momentum_sentiment, 2)
                },
                'timestamp': datetime.now().isoformat()
            }

            # Cache result
            self.sentiment_cache[cache_key] = result

            logger.debug(f"{pair} Sentiment: {strength} ({combined_score:.1f}), "
                        f"Retail={retail_estimate:.1f}, Contrarian={contrarian_signal}")

            return result

        except Exception as e:
            logger.error(f"Sentiment analysis error for {pair}: {e}")
            return self._neutral_sentiment(pair)

    def _calculate_technical_sentiment(self, data: pd.DataFrame) -> float:
        """
        Calculate sentiment based on technical indicators.
        Uses RSI and Stochastic to estimate crowd positioning.

        Returns: -100 to +100
        """
        try:
            close = data['close']

            # RSI (14)
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            # Stochastic (14, 3)
            low_14 = data['low'].rolling(14).min()
            high_14 = data['high'].rolling(14).max()
            stoch_k = 100 * (close - low_14) / (high_14 - low_14)
            stoch_d = stoch_k.rolling(3).mean()
            current_stoch = stoch_d.iloc[-1]

            # Convert to sentiment (-100 to +100)
            # RSI > 70 = bullish crowd, RSI < 30 = bearish crowd
            rsi_sentiment = (current_rsi - 50) * 2  # Scale to -100 to +100
            stoch_sentiment = (current_stoch - 50) * 2

            # Average
            tech_sentiment = (rsi_sentiment + stoch_sentiment) / 2

            return max(-100, min(100, tech_sentiment))

        except Exception:
            return 0.0

    def _calculate_volatility_sentiment(self, data: pd.DataFrame) -> float:
        """
        Calculate sentiment based on volatility.
        High volatility often indicates fear (bearish sentiment).
        Low volatility indicates complacency (bullish sentiment).

        Returns: -100 to +100
        """
        try:
            # Calculate ATR
            high = data['high']
            low = data['low']
            close = data['close']

            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(window=14).mean()

            # Compare current ATR to average ATR
            current_atr = atr.iloc[-1]
            avg_atr = atr.rolling(50).mean().iloc[-1]

            if avg_atr == 0:
                return 0.0

            atr_ratio = current_atr / avg_atr

            # High volatility (ratio > 1.5) = fear = bearish sentiment
            # Low volatility (ratio < 0.7) = complacency = bullish sentiment
            if atr_ratio > 1.5:
                vol_sentiment = -50 * (atr_ratio - 1)  # More negative as vol increases
            elif atr_ratio < 0.7:
                vol_sentiment = 50 * (1 - atr_ratio)  # More positive as vol decreases
            else:
                vol_sentiment = 0

            return max(-100, min(100, vol_sentiment))

        except Exception:
            return 0.0

    def _calculate_momentum_sentiment(self, data: pd.DataFrame) -> float:
        """
        Calculate sentiment based on price momentum.
        Strong up momentum = bullish sentiment
        Strong down momentum = bearish sentiment

        Returns: -100 to +100
        """
        try:
            close = data['close']

            # Rate of change (20 periods)
            roc_20 = ((close.iloc[-1] / close.iloc[-20]) - 1) * 100

            # EMA direction
            ema_20 = close.ewm(span=20, adjust=False).mean()
            ema_50 = close.ewm(span=50, adjust=False).mean()

            ema_diff = ((ema_20.iloc[-1] / ema_50.iloc[-1]) - 1) * 100

            # Recent momentum (last 5 candles)
            recent_change = ((close.iloc[-1] / close.iloc[-5]) - 1) * 100

            # Combine
            momentum_score = (roc_20 * 0.4 + ema_diff * 100 * 0.3 + recent_change * 10 * 0.3)

            return max(-100, min(100, momentum_score))

        except Exception:
            return 0.0

    def _estimate_positioning(self, data: pd.DataFrame) -> Tuple[float, str]:
        """
        Estimate retail vs institutional positioning.

        Retail traders tend to:
        - Buy at resistance (overbought)
        - Sell at support (oversold)
        - Fight the trend

        Smart money tends to:
        - Buy dips in uptrends
        - Sell rallies in downtrends
        - Follow momentum

        Returns:
            Tuple of (retail_sentiment, institutional_bias)
        """
        try:
            close = data['close']

            # Calculate RSI for retail estimation
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1]

            # Retail sentiment: they buy when RSI is high, sell when low
            # So RSI directly correlates with retail positioning
            retail_sentiment = (current_rsi - 50) * 2  # -100 to +100

            # Institutional bias: based on trend and volume
            ema_20 = close.ewm(span=20, adjust=False).mean()
            ema_50 = close.ewm(span=50, adjust=False).mean()

            if ema_20.iloc[-1] > ema_50.iloc[-1]:
                institutional_bias = "BULLISH"
            elif ema_20.iloc[-1] < ema_50.iloc[-1]:
                institutional_bias = "BEARISH"
            else:
                institutional_bias = "NEUTRAL"

            return retail_sentiment, institutional_bias

        except Exception:
            return 0.0, "NEUTRAL"

    def _categorize_sentiment(self, score: float) -> str:
        """Categorize sentiment score into strength levels."""
        if score >= self.extreme_bullish:
            return "extreme_bullish"
        elif score >= 20:
            return "bullish"
        elif score <= -self.extreme_bullish:
            return "extreme_bearish"
        elif score <= -20:
            return "bearish"
        else:
            return "neutral"

    def _get_contrarian_signal(self, retail_sentiment: float) -> str:
        """
        Get contrarian signal based on retail positioning.
        Fade the crowd!
        """
        if retail_sentiment >= self.extreme_bullish:
            return "SELL"  # Retail very bullish = contrarian SELL
        elif retail_sentiment <= -self.extreme_bullish:
            return "BUY"   # Retail very bearish = contrarian BUY
        else:
            return "NEUTRAL"

    def _neutral_sentiment(self, pair: str) -> Dict:
        """Return neutral sentiment when analysis fails."""
        return {
            'pair': pair,
            'score': 0,
            'strength': 'neutral',
            'retail_sentiment': 0,
            'institutional_bias': 'NEUTRAL',
            'contrarian_signal': 'NEUTRAL',
            'components': {
                'technical': 0,
                'volatility': 0,
                'momentum': 0
            },
            'timestamp': datetime.now().isoformat()
        }

    def get_sentiment_alignment(self, pair: str, direction: str, data: pd.DataFrame = None) -> Tuple[bool, float]:
        """
        Check if a trade direction aligns with sentiment (contrarian approach).

        Args:
            pair: Trading pair
            direction: Trade direction ('BUY' or 'SELL')
            data: Price data

        Returns:
            Tuple of (is_aligned, alignment_score)
            - is_aligned: True if trade fades extreme retail sentiment
            - alignment_score: 0-100 (higher = better alignment)
        """
        sentiment = self.analyze(pair, data)
        retail = sentiment['retail_sentiment']
        contrarian = sentiment['contrarian_signal']

        # Best case: Extreme sentiment in opposite direction
        if direction == 'BUY' and retail <= -50:
            # Buying when retail is very bearish = excellent contrarian
            alignment_score = 70 + abs(retail) * 0.3
            return True, min(100, alignment_score)

        elif direction == 'SELL' and retail >= 50:
            # Selling when retail is very bullish = excellent contrarian
            alignment_score = 70 + retail * 0.3
            return True, min(100, alignment_score)

        # Neutral sentiment = no strong signal
        elif -30 <= retail <= 30:
            return True, 50  # Acceptable

        # Bad case: Following the crowd at extremes
        elif direction == 'BUY' and retail >= 50:
            return False, 30  # Following crowd at top
        elif direction == 'SELL' and retail <= -50:
            return False, 30  # Following crowd at bottom

        return True, 50  # Default acceptable


# Singleton instance for backward compatibility
_analyzer_instance = None


def get_sentiment(pair: str, data: pd.DataFrame = None) -> Dict:
    """
    Get sentiment for a pair (function interface).

    Args:
        pair: Trading pair
        data: Optional price data

    Returns:
        Sentiment dictionary
    """
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = SentimentAnalyzer()

    return _analyzer_instance.analyze(pair, data)
