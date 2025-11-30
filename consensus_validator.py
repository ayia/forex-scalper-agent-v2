"""Consensus Validator Module - Multi-Timeframe Signal Validation

This module validates trading signals by checking consensus across multiple
timeframes (M1, M5, M15, H1, H4) and integrates various validation techniques.

Part of Forex Scalper Agent V2 - Architecture Avancée
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from config import (
    STRATEGY_PARAMS, RISK_PARAMS,     LOG_CONFIG
)

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG['log_level']),
    format=LOG_CONFIG['format']
)
logger = logging.getLogger(__name__)


class ConsensusValidator:
    """
    Multi-Timeframe Consensus Validator.
    
    Validates trading signals by:
    - Checking H1 trend alignment
    - Verifying M15 structure (support/resistance)
    - Detecting RSI divergence
    - Analyzing retail sentiment
    - Calculating consensus score (0-100)
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConsensusValidator, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.timeframes = TRADING_PARAMS['timeframes']
        self.primary_tf = TRADING_PARAMS['primary_timeframe']
        self.min_score = TRADING_PARAMS['min_score']
        
        # SMC parameters for structure analysis
        self.swing_lookback = SMC_PARAMS['swing_lookback']
        self.liquidity_threshold = SMC_PARAMS['liquidity_threshold']
        
        # Indicator parameters
        self.rsi_period = INDICATORS['rsi_period']
        self.rsi_overbought = INDICATORS['rsi_overbought']
        self.rsi_oversold = INDICATORS['rsi_oversold']
        
        # Weight configuration for scoring
        self.weights = {
            'h1_alignment': 0.30,
            'm15_structure': 0.25,
            'divergence': 0.20,
            'sentiment': 0.15,
            'volume_confirm': 0.10
        }
        
        self._initialized = True
        logger.info("ConsensusValidator initialized")
    
    def validate_signal(
        self,
        signal_direction: str,
        multi_tf_data: Dict[str, pd.DataFrame],
        sentiment_score: Optional[float] = None
    ) -> Dict:
        """
        Validate a trading signal across multiple timeframes.
        
        Args:
            signal_direction: 'BUY' or 'SELL'
            multi_tf_data: Dict with timeframe keys and OHLCV DataFrames
            sentiment_score: Optional sentiment score from SentimentAnalyzer
            
        Returns:
            Dict with validation results and consensus score
        """
        try:
            validation_results = {
                'signal': signal_direction,
                'timestamp': datetime.utcnow().isoformat(),
                'validations': {},
                'consensus_score': 0,
                'is_valid': False,
                'confidence_level': 'LOW'
            }
            
            scores = {}
            
            # 1. H1 Trend Alignment
            if 'H1' in multi_tf_data:
                h1_result = self.check_h1_alignment(
                    multi_tf_data['H1'], 
                    signal_direction
                )
                validation_results['validations']['h1_alignment'] = h1_result
                scores['h1_alignment'] = h1_result['score']
            else:
                scores['h1_alignment'] = 50  # Neutral if no H1 data
            
            # 2. M15 Structure Check
            if 'M15' in multi_tf_data:
                m15_result = self.check_m15_structure(
                    multi_tf_data['M15'],
                    signal_direction
                )
                validation_results['validations']['m15_structure'] = m15_result
                scores['m15_structure'] = m15_result['score']
            else:
                scores['m15_structure'] = 50
            
            # 3. RSI Divergence Detection
            primary_data = multi_tf_data.get(self.primary_tf)
            if primary_data is not None and len(primary_data) >= self.rsi_period + 10:
                divergence_result = self.check_divergence(
                    primary_data,
                    signal_direction
                )
                validation_results['validations']['divergence'] = divergence_result
                scores['divergence'] = divergence_result['score']
            else:
                scores['divergence'] = 50
            
            # 4. Sentiment Analysis Integration
            if sentiment_score is not None:
                sentiment_result = self.validate_sentiment(
                    sentiment_score,
                    signal_direction
                )
                validation_results['validations']['sentiment'] = sentiment_result
                scores['sentiment'] = sentiment_result['score']
            else:
                scores['sentiment'] = 50
            
            # 5. Volume Confirmation
            if primary_data is not None and 'volume' in primary_data.columns:
                volume_result = self.check_volume_confirmation(primary_data)
                validation_results['validations']['volume_confirm'] = volume_result
                scores['volume_confirm'] = volume_result['score']
            else:
                scores['volume_confirm'] = 50
            
            # Calculate weighted consensus score
            consensus_score = self.calculate_consensus_score(scores)
            validation_results['consensus_score'] = consensus_score
            
            # Determine validity and confidence
            validation_results['is_valid'] = consensus_score >= self.min_score
            validation_results['confidence_level'] = self._get_confidence_level(
                consensus_score
            )
            
            logger.info(
                f"Signal validation complete: {signal_direction} | "
                f"Score: {consensus_score:.1f} | "
                f"Valid: {validation_results['is_valid']}"
            )
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating signal: {e}")
            return {
                'signal': signal_direction,
                'error': str(e),
                'consensus_score': 0,
                'is_valid': False
            }
    
    def check_h1_alignment(
        self,
        h1_data: pd.DataFrame,
        signal_direction: str
    ) -> Dict:
        """
        Check if H1 trend aligns with signal direction.
        Uses EMA crossover and price position relative to EMAs.
        """
        try:
            df = h1_data.copy()
            
            # Calculate EMAs
            ema_fast = INDICATORS['ema_fast']
            ema_slow = INDICATORS['ema_slow']
            
            df['ema_fast'] = df['close'].ewm(span=ema_fast, adjust=False).mean()
            df['ema_slow'] = df['close'].ewm(span=ema_slow, adjust=False).mean()
            
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            
            # Determine H1 trend
            ema_bullish = latest['ema_fast'] > latest['ema_slow']
            price_above_emas = (
                latest['close'] > latest['ema_fast'] and 
                latest['close'] > latest['ema_slow']
            )
            price_below_emas = (
                latest['close'] < latest['ema_fast'] and 
                latest['close'] < latest['ema_slow']
            )
            
            # Recent momentum
            momentum = (latest['close'] - prev['close']) / prev['close'] * 100
            
            # Score calculation
            score = 50  # Neutral baseline
            
            if signal_direction == 'BUY':
                if ema_bullish:
                    score += 25
                if price_above_emas:
                    score += 15
                if momentum > 0:
                    score += 10
            else:  # SELL
                if not ema_bullish:
                    score += 25
                if price_below_emas:
                    score += 15
                if momentum < 0:
                    score += 10
            
            score = min(100, max(0, score))
            
            trend = 'BULLISH' if ema_bullish else 'BEARISH'
            aligned = (
                (signal_direction == 'BUY' and ema_bullish) or
                (signal_direction == 'SELL' and not ema_bullish)
            )
            
            return {
                'trend': trend,
                'aligned': aligned,
                'ema_fast': round(latest['ema_fast'], 5),
                'ema_slow': round(latest['ema_slow'], 5),
                'momentum_pct': round(momentum, 4),
                'score': score
            }
            
        except Exception as e:
            logger.error(f"H1 alignment check error: {e}")
            return {'aligned': False, 'score': 50, 'error': str(e)}
    
    def check_m15_structure(
        self,
        m15_data: pd.DataFrame,
        signal_direction: str
    ) -> Dict:
        """
        Check M15 market structure for support/resistance levels.
        Identifies swing highs/lows and validates signal against structure.
        """
        try:
            df = m15_data.copy()
            lookback = self.swing_lookback
            
            # Identify swing highs and lows
            df['swing_high'] = df['high'].rolling(
                window=lookback, center=True
            ).max() == df['high']
            df['swing_low'] = df['low'].rolling(
                window=lookback, center=True
            ).min() == df['low']
            
            # Get recent swing points
            recent_highs = df[df['swing_high']]['high'].tail(3).tolist()
            recent_lows = df[df['swing_low']]['low'].tail(3).tolist()
            
            current_price = df.iloc[-1]['close']
            
            # Find nearest support and resistance
            resistance_levels = [h for h in recent_highs if h > current_price]
            support_levels = [l for l in recent_lows if l < current_price]
            
            nearest_resistance = min(resistance_levels) if resistance_levels else None
            nearest_support = max(support_levels) if support_levels else None
            
            # Calculate distance to levels (in pips for forex)
            pip_value = 0.0001  # Standard for most pairs
            
            score = 50  # Baseline
            structure_analysis = {
                'current_price': current_price,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance
            }
            
            if signal_direction == 'BUY':
                # Bullish: price near support is favorable
                if nearest_support:
                    distance_to_support = (current_price - nearest_support) / pip_value
                    if distance_to_support < 20:  # Very close to support
                        score += 30
                    elif distance_to_support < 50:
                        score += 15
                    structure_analysis['distance_to_support_pips'] = round(distance_to_support, 1)
                
                # Room to resistance
                if nearest_resistance:
                    room_to_resistance = (nearest_resistance - current_price) / pip_value
                    if room_to_resistance > 50:
                        score += 20
                    structure_analysis['room_to_resistance_pips'] = round(room_to_resistance, 1)
                        
            else:  # SELL
                # Bearish: price near resistance is favorable
                if nearest_resistance:
                    distance_to_resistance = (nearest_resistance - current_price) / pip_value
                    if distance_to_resistance < 20:
                        score += 30
                    elif distance_to_resistance < 50:
                        score += 15
                    structure_analysis['distance_to_resistance_pips'] = round(distance_to_resistance, 1)
                
                # Room to support
                if nearest_support:
                    room_to_support = (current_price - nearest_support) / pip_value
                    if room_to_support > 50:
                        score += 20
                    structure_analysis['room_to_support_pips'] = round(room_to_support, 1)
            
            score = min(100, max(0, score))
            structure_analysis['score'] = score
            
            return structure_analysis
            
        except Exception as e:
            logger.error(f"M15 structure check error: {e}")
            return {'score': 50, 'error': str(e)}
    
    def check_divergence(
        self,
        data: pd.DataFrame,
        signal_direction: str
    ) -> Dict:
        """
        Detect RSI divergence patterns.
        Bullish divergence: price lower low, RSI higher low
        Bearish divergence: price higher high, RSI lower high
        """
        try:
            df = data.copy()
            
            # Calculate RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = (-delta).where(delta < 0, 0)
            
            avg_gain = gain.rolling(window=self.rsi_period).mean()
            avg_loss = loss.rolling(window=self.rsi_period).mean()
            
            rs = avg_gain / avg_loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Look for divergence in last N candles
            lookback = 10
            recent = df.tail(lookback)
            
            # Find price extremes
            price_lows_idx = recent['low'].nsmallest(2).index.tolist()
            price_highs_idx = recent['high'].nlargest(2).index.tolist()
            
            bullish_divergence = False
            bearish_divergence = False
            divergence_strength = 0
            
            if len(price_lows_idx) >= 2:
                # Check for bullish divergence
                idx1, idx2 = price_lows_idx[0], price_lows_idx[1]
                if idx2 > idx1:  # Second low is more recent
                    price_lower = recent.loc[idx2, 'low'] < recent.loc[idx1, 'low']
                    rsi_higher = recent.loc[idx2, 'rsi'] > recent.loc[idx1, 'rsi']
                    if price_lower and rsi_higher:
                        bullish_divergence = True
                        divergence_strength = abs(
                            recent.loc[idx2, 'rsi'] - recent.loc[idx1, 'rsi']
                        )
            
            if len(price_highs_idx) >= 2:
                # Check for bearish divergence
                idx1, idx2 = price_highs_idx[0], price_highs_idx[1]
                if idx2 > idx1:
                    price_higher = recent.loc[idx2, 'high'] > recent.loc[idx1, 'high']
                    rsi_lower = recent.loc[idx2, 'rsi'] < recent.loc[idx1, 'rsi']
                    if price_higher and rsi_lower:
                        bearish_divergence = True
                        divergence_strength = abs(
                            recent.loc[idx1, 'rsi'] - recent.loc[idx2, 'rsi']
                        )
            
            # Score based on divergence alignment with signal
            score = 50
            
            if signal_direction == 'BUY' and bullish_divergence:
                score = 70 + min(30, divergence_strength)
            elif signal_direction == 'SELL' and bearish_divergence:
                score = 70 + min(30, divergence_strength)
            elif (signal_direction == 'BUY' and bearish_divergence) or \
                 (signal_direction == 'SELL' and bullish_divergence):
                score = 30  # Opposing divergence is bearish for signal
            
            current_rsi = df['rsi'].iloc[-1]
            
            return {
                'bullish_divergence': bullish_divergence,
                'bearish_divergence': bearish_divergence,
                'divergence_strength': round(divergence_strength, 2),
                'current_rsi': round(current_rsi, 2),
                'rsi_zone': self._get_rsi_zone(current_rsi),
                'score': min(100, max(0, score))
            }
            
        except Exception as e:
            logger.error(f"Divergence check error: {e}")
            return {'score': 50, 'error': str(e)}
    
    def _get_rsi_zone(self, rsi: float) -> str:
        """Categorize RSI value into zones."""
        if rsi >= self.rsi_overbought:
            return 'OVERBOUGHT'
        elif rsi <= self.rsi_oversold:
            return 'OVERSOLD'
        elif rsi >= 60:
            return 'BULLISH'
        elif rsi <= 40:
            return 'BEARISH'
        return 'NEUTRAL'
    
    def validate_sentiment(
        self,
        sentiment_score: float,
        signal_direction: str
    ) -> Dict:
        """
        Validate signal against market sentiment.
        Contrarian approach: fade extreme retail sentiment.
        """
        try:
            # Sentiment score expected: -1 (bearish) to +1 (bullish)
            # Contrarian logic: extreme retail bullish = fade (sell)
            
            score = 50
            sentiment_alignment = 'NEUTRAL'
            
            if signal_direction == 'BUY':
                if sentiment_score < -0.5:  # Retail very bearish
                    score = 80  # Contrarian bullish
                    sentiment_alignment = 'CONTRARIAN_BULLISH'
                elif sentiment_score < 0:
                    score = 65
                    sentiment_alignment = 'MILDLY_CONTRARIAN'
                elif sentiment_score > 0.5:  # Retail very bullish
                    score = 30  # Against crowd could be risky
                    sentiment_alignment = 'CROWD_FOLLOWING'
                else:
                    score = 50
                    
            else:  # SELL
                if sentiment_score > 0.5:  # Retail very bullish
                    score = 80  # Contrarian bearish
                    sentiment_alignment = 'CONTRARIAN_BEARISH'
                elif sentiment_score > 0:
                    score = 65
                    sentiment_alignment = 'MILDLY_CONTRARIAN'
                elif sentiment_score < -0.5:  # Retail very bearish
                    score = 30
                    sentiment_alignment = 'CROWD_FOLLOWING'
                else:
                    score = 50
            
            return {
                'sentiment_score': round(sentiment_score, 3),
                'alignment': sentiment_alignment,
                'interpretation': self._interpret_sentiment(sentiment_score),
                'score': score
            }
            
        except Exception as e:
            logger.error(f"Sentiment validation error: {e}")
            return {'score': 50, 'error': str(e)}
    
    def _interpret_sentiment(self, score: float) -> str:
        """Interpret sentiment score."""
        if score > 0.7:
            return 'EXTREME_BULLISH'
        elif score > 0.3:
            return 'BULLISH'
        elif score > -0.3:
            return 'NEUTRAL'
        elif score > -0.7:
            return 'BEARISH'
        return 'EXTREME_BEARISH'
    
    def check_volume_confirmation(
        self,
        data: pd.DataFrame
    ) -> Dict:
        """
        Check if volume confirms the price movement.
        Higher volume on trend direction is bullish for the signal.
        """
        try:
            df = data.copy()
            
            # Calculate volume metrics
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            latest = df.iloc[-1]
            prev_5 = df.tail(5)
            
            # Volume above average?
            volume_above_avg = latest['volume_ratio'] > 1.0
            avg_volume_ratio = prev_5['volume_ratio'].mean()
            
            # Price direction with volume
            price_up = latest['close'] > latest['open']
            volume_increasing = latest['volume'] > df['volume'].iloc[-2]
            
            score = 50
            
            if volume_above_avg:
                score += 20
            if avg_volume_ratio > 1.2:
                score += 15
            if (price_up and volume_increasing) or (not price_up and volume_increasing):
                score += 15  # Volume confirms price movement
            
            return {
                'current_volume_ratio': round(latest['volume_ratio'], 2),
                'avg_volume_ratio_5': round(avg_volume_ratio, 2),
                'volume_above_average': volume_above_avg,
                'volume_increasing': volume_increasing,
                'score': min(100, max(0, score))
            }
            
        except Exception as e:
            logger.error(f"Volume confirmation error: {e}")
            return {'score': 50, 'error': str(e)}
    
    def calculate_consensus_score(
        self,
        scores: Dict[str, float]
    ) -> float:
        """
        Calculate weighted consensus score from individual validations.
        
        Args:
            scores: Dict of validation scores (0-100)
            
        Returns:
            Weighted average score (0-100)
        """
        weighted_sum = 0
        total_weight = 0
        
        for key, score in scores.items():
            weight = self.weights.get(key, 0.1)
            weighted_sum += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 50.0
            
        consensus = weighted_sum / total_weight
        return round(consensus, 1)
    
    def _get_confidence_level(self, score: float) -> str:
        """Map consensus score to confidence level."""
        if score >= 85:
            return 'VERY_HIGH'
        elif score >= 75:
            return 'HIGH'
        elif score >= 65:
            return 'MEDIUM'
        elif score >= 50:
            return 'LOW'
        return 'VERY_LOW'
    
    def get_multi_tf_summary(
        self,
        multi_tf_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Generate a summary of conditions across all timeframes.
        """
        summary = {
            'timeframes_analyzed': [],
            'trend_alignment': {},
            'overall_bias': 'NEUTRAL'
        }
        
        bullish_count = 0
        bearish_count = 0
        
        for tf, data in multi_tf_data.items():
            if data is None or len(data) < 50:
                continue
                
            summary['timeframes_analyzed'].append(tf)
            
            # Calculate quick trend for each TF
            ema_fast = data['close'].ewm(span=8, adjust=False).mean()
            ema_slow = data['close'].ewm(span=21, adjust=False).mean()
            
            trend = 'BULLISH' if ema_fast.iloc[-1] > ema_slow.iloc[-1] else 'BEARISH'
            summary['trend_alignment'][tf] = trend
            
            if trend == 'BULLISH':
                bullish_count += 1
            else:
                bearish_count += 1
        
        # Determine overall bias
        if bullish_count > bearish_count:
            summary['overall_bias'] = 'BULLISH'
        elif bearish_count > bullish_count:
            summary['overall_bias'] = 'BEARISH'
        else:
            summary['overall_bias'] = 'MIXED'
        
        summary['bullish_tf_count'] = bullish_count
        summary['bearish_tf_count'] = bearish_count
        
        return summary


# Singleton instance
consensus_validator = ConsensusValidator()
