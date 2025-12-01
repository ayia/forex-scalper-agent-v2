"""
Multi-Timeframe Analyzer - Architecture Top-Down pour Scalping
===============================================================
Analyse hierarchique des timeframes:
- H4: Tendance principale (Bias directionnel)
- H1: Confirmation de tendance + Structure cle
- M15: Zone d'entree + Confluence d'indicateurs
- M5: Timing precis d'entree
- M1: Optimisation SL/TP pour scalping

Principe: On ne trade QUE dans la direction des HTF (H4/H1)
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from datetime import datetime

from config import get_pip_value, STRATEGY_PARAMS

logger = logging.getLogger(__name__)


@dataclass
class MTFAnalysis:
    """Result of multi-timeframe analysis."""
    pair: str
    htf_bias: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    htf_strength: float  # 0-100
    h4_trend: str
    h1_trend: str
    m15_signal: Optional[str]
    m5_entry: Optional[str]
    confluence_score: float  # 0-100
    entry_price: float
    optimal_sl: float
    optimal_tp: float
    structure_levels: Dict  # Key S/R levels
    timestamp: str


class MTFAnalyzer:
    """
    Analyseur Multi-Timeframe pour scalping optimise.

    Architecture Top-Down:
    1. H4 definit le BIAS (on ne trade que dans cette direction)
    2. H1 confirme la tendance et identifie la structure
    3. M15 donne les zones d'entree
    4. M5 optimise le timing
    5. M1 affine SL/TP
    """

    def __init__(self):
        # EMA periods for trend detection
        self.ema_fast = STRATEGY_PARAMS.get('ema_fast', 20)
        self.ema_medium = STRATEGY_PARAMS.get('ema_medium', 50)
        self.ema_slow = STRATEGY_PARAMS.get('ema_slow', 200)

        # RSI settings
        self.rsi_period = STRATEGY_PARAMS.get('rsi_period', 14)
        self.rsi_overbought = 70
        self.rsi_oversold = 30

        # Confluence thresholds
        self.min_confluence_score = 60  # Minimum score to generate signal

        logger.info("MTFAnalyzer initialized - Top-Down Architecture")

    def analyze(self, pair: str, mtf_data: Dict[str, pd.DataFrame]) -> Optional[MTFAnalysis]:
        """
        Perform complete multi-timeframe analysis.

        Args:
            pair: Currency pair
            mtf_data: Dict with timeframe -> DataFrame mapping
                      Required: H4, H1, M15, M5 (M1 optional)

        Returns:
            MTFAnalysis object or None if no valid setup
        """
        # Validate we have required timeframes
        required_tfs = ['H4', 'H1', 'M15', 'M5']

        # Also accept lowercase/yfinance format
        tf_mapping = {
            'H4': ['H4', '4h', '4H'],
            'H1': ['H1', '1h', '1H'],
            'M15': ['M15', '15m', '15M'],
            'M5': ['M5', '5m', '5M'],
            'M1': ['M1', '1m', '1M']
        }

        # Normalize timeframe keys
        normalized_data = {}
        for std_tf, variants in tf_mapping.items():
            for variant in variants:
                if variant in mtf_data and mtf_data[variant] is not None:
                    normalized_data[std_tf] = mtf_data[variant]
                    break

        for tf in required_tfs:
            if tf not in normalized_data or normalized_data[tf] is None or len(normalized_data[tf]) < 50:
                logger.debug(f"{pair}: Missing or insufficient data for {tf}")
                return None

        try:
            # 1. HIGHER TIMEFRAME ANALYSIS (H4 + H1)
            h4_analysis = self._analyze_htf(normalized_data['H4'], 'H4')
            h1_analysis = self._analyze_htf(normalized_data['H1'], 'H1')

            # Determine HTF bias
            htf_bias, htf_strength = self._determine_htf_bias(h4_analysis, h1_analysis)

            if htf_bias == 'NEUTRAL':
                logger.debug(f"{pair}: No clear HTF bias - skipping")
                return None

            # 2. LOWER TIMEFRAME ANALYSIS (M15 + M5)
            m15_signal = self._analyze_ltf_signal(normalized_data['M15'], htf_bias)
            m5_entry = self._analyze_entry_timing(normalized_data['M5'], htf_bias)

            # 3. Calculate confluence score
            confluence_score = self._calculate_confluence(
                h4_analysis, h1_analysis, m15_signal, m5_entry, htf_bias
            )

            if confluence_score < self.min_confluence_score:
                logger.debug(f"{pair}: Confluence score {confluence_score:.1f} < {self.min_confluence_score}")
                return None

            # 4. Calculate optimal SL/TP using M1 if available, otherwise M5
            entry_tf = normalized_data.get('M1', normalized_data['M5'])
            entry_price = entry_tf['close'].iloc[-1]
            optimal_sl, optimal_tp = self._calculate_scalping_levels(
                entry_tf, pair, htf_bias, h1_analysis
            )

            # 5. Get structure levels
            structure_levels = self._get_structure_levels(
                normalized_data['H1'], normalized_data['M15']
            )

            return MTFAnalysis(
                pair=pair,
                htf_bias=htf_bias,
                htf_strength=htf_strength,
                h4_trend=h4_analysis['trend'],
                h1_trend=h1_analysis['trend'],
                m15_signal=m15_signal,
                m5_entry=m5_entry,
                confluence_score=confluence_score,
                entry_price=entry_price,
                optimal_sl=optimal_sl,
                optimal_tp=optimal_tp,
                structure_levels=structure_levels,
                timestamp=datetime.now().isoformat()
            )

        except Exception as e:
            logger.error(f"MTF analysis error for {pair}: {e}")
            return None

    def _analyze_htf(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """
        Analyze higher timeframe for trend direction.

        Returns:
            Dict with trend info: trend, strength, ema_stack, rsi
        """
        close = df['close']

        # Calculate EMAs
        ema_20 = close.ewm(span=self.ema_fast, adjust=False).mean()
        ema_50 = close.ewm(span=self.ema_medium, adjust=False).mean()
        ema_200 = close.ewm(span=self.ema_slow, adjust=False).mean()

        # Calculate RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # Get current values
        current_price = close.iloc[-1]
        current_ema20 = ema_20.iloc[-1]
        current_ema50 = ema_50.iloc[-1]
        current_ema200 = ema_200.iloc[-1]

        # Determine trend
        bullish_stack = current_ema20 > current_ema50 > current_ema200
        bearish_stack = current_ema20 < current_ema50 < current_ema200
        price_above_emas = current_price > current_ema20 > current_ema50
        price_below_emas = current_price < current_ema20 < current_ema50

        # Calculate trend strength (0-100)
        if bullish_stack and price_above_emas:
            trend = 'BULLISH'
            # Strength based on EMA separation and RSI
            ema_sep = (current_ema20 - current_ema200) / current_ema200 * 100
            strength = min(100, 50 + ema_sep * 10 + (current_rsi - 50))
        elif bearish_stack and price_below_emas:
            trend = 'BEARISH'
            ema_sep = (current_ema200 - current_ema20) / current_ema200 * 100
            strength = min(100, 50 + ema_sep * 10 + (50 - current_rsi))
        elif bullish_stack:
            trend = 'BULLISH_WEAK'
            strength = 40
        elif bearish_stack:
            trend = 'BEARISH_WEAK'
            strength = 40
        else:
            trend = 'RANGING'
            strength = 20

        return {
            'timeframe': timeframe,
            'trend': trend,
            'strength': max(0, min(100, strength)),
            'rsi': current_rsi,
            'price': current_price,
            'ema_20': current_ema20,
            'ema_50': current_ema50,
            'ema_200': current_ema200,
            'bullish_stack': bullish_stack,
            'bearish_stack': bearish_stack
        }

    def _determine_htf_bias(self, h4: Dict, h1: Dict) -> Tuple[str, float]:
        """
        Determine overall HTF bias from H4 and H1 analysis.

        Returns:
            Tuple of (bias, strength)
        """
        # H4 has more weight (60%) than H1 (40%)
        h4_weight = 0.6
        h1_weight = 0.4

        # Convert trends to scores
        def trend_to_score(trend: str) -> float:
            scores = {
                'BULLISH': 100,
                'BULLISH_WEAK': 60,
                'RANGING': 50,
                'BEARISH_WEAK': 40,
                'BEARISH': 0
            }
            return scores.get(trend, 50)

        h4_score = trend_to_score(h4['trend'])
        h1_score = trend_to_score(h1['trend'])

        combined_score = h4_score * h4_weight + h1_score * h1_weight

        # Determine bias
        if combined_score >= 70:
            bias = 'BULLISH'
            strength = combined_score
        elif combined_score <= 30:
            bias = 'BEARISH'
            strength = 100 - combined_score
        else:
            bias = 'NEUTRAL'
            strength = 50

        # Additional check: both timeframes must agree for strong bias
        if h4['trend'].startswith('BULLISH') and h1['trend'].startswith('BEARISH'):
            bias = 'NEUTRAL'
            strength = 30
        elif h4['trend'].startswith('BEARISH') and h1['trend'].startswith('BULLISH'):
            bias = 'NEUTRAL'
            strength = 30

        return bias, strength

    def _analyze_ltf_signal(self, df: pd.DataFrame, htf_bias: str) -> Optional[str]:
        """
        Analyze M15 for entry signals aligned with HTF bias.

        Only generates signal if it aligns with HTF direction.
        """
        close = df['close']

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1]

        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()

        macd_cross_up = macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]
        macd_cross_down = macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]

        # Generate signals only in HTF direction
        if htf_bias == 'BULLISH':
            # Look for bullish signals: RSI from oversold + MACD cross up
            if current_rsi < 50 and macd_cross_up:
                return 'BUY_PULLBACK'
            elif current_rsi > 40 and macd.iloc[-1] > 0:
                return 'BUY_CONTINUATION'
            return None

        elif htf_bias == 'BEARISH':
            # Look for bearish signals: RSI from overbought + MACD cross down
            if current_rsi > 50 and macd_cross_down:
                return 'SELL_PULLBACK'
            elif current_rsi < 60 and macd.iloc[-1] < 0:
                return 'SELL_CONTINUATION'
            return None

        return None

    def _analyze_entry_timing(self, df: pd.DataFrame, htf_bias: str) -> Optional[str]:
        """
        Analyze M5 for precise entry timing.

        Looks for momentum confirmation in the direction of HTF bias.
        """
        close = df['close']

        # Short-term momentum (5-period ROC)
        roc = (close.iloc[-1] / close.iloc[-5] - 1) * 100

        # EMA cross on M5
        ema_8 = close.ewm(span=8, adjust=False).mean()
        ema_21 = close.ewm(span=21, adjust=False).mean()

        ema_bullish = ema_8.iloc[-1] > ema_21.iloc[-1]
        ema_bearish = ema_8.iloc[-1] < ema_21.iloc[-1]

        # Price action (last 3 candles)
        recent_bullish = close.iloc[-1] > close.iloc[-3]
        recent_bearish = close.iloc[-1] < close.iloc[-3]

        if htf_bias == 'BULLISH':
            if ema_bullish and roc > 0 and recent_bullish:
                return 'ENTRY_NOW'
            elif ema_bullish and roc > -0.1:
                return 'ENTRY_SOON'
            return None

        elif htf_bias == 'BEARISH':
            if ema_bearish and roc < 0 and recent_bearish:
                return 'ENTRY_NOW'
            elif ema_bearish and roc < 0.1:
                return 'ENTRY_SOON'
            return None

        return None

    def _calculate_confluence(self, h4: Dict, h1: Dict,
                             m15_signal: Optional[str],
                             m5_entry: Optional[str],
                             htf_bias: str) -> float:
        """
        Calculate overall confluence score (0-100).

        Weights:
        - H4 trend alignment: 25%
        - H1 trend alignment: 25%
        - M15 signal present: 25%
        - M5 entry timing: 15%
        - RSI alignment: 10%
        """
        score = 0

        # H4 alignment (25 points)
        if htf_bias == 'BULLISH' and h4['trend'] == 'BULLISH':
            score += 25
        elif htf_bias == 'BEARISH' and h4['trend'] == 'BEARISH':
            score += 25
        elif 'WEAK' in h4['trend']:
            score += 15

        # H1 alignment (25 points)
        if htf_bias == 'BULLISH' and h1['trend'].startswith('BULLISH'):
            score += 25
        elif htf_bias == 'BEARISH' and h1['trend'].startswith('BEARISH'):
            score += 25
        elif h1['trend'] == 'RANGING':
            score += 10

        # M15 signal (25 points)
        if m15_signal:
            if 'PULLBACK' in m15_signal:
                score += 25  # Pullback entries are best
            else:
                score += 20  # Continuation signals

        # M5 entry timing (15 points)
        if m5_entry == 'ENTRY_NOW':
            score += 15
        elif m5_entry == 'ENTRY_SOON':
            score += 10

        # RSI alignment (10 points)
        if htf_bias == 'BULLISH':
            if h1['rsi'] > 50 and h1['rsi'] < 70:
                score += 10
            elif h1['rsi'] > 40:
                score += 5
        elif htf_bias == 'BEARISH':
            if h1['rsi'] < 50 and h1['rsi'] > 30:
                score += 10
            elif h1['rsi'] < 60:
                score += 5

        return score

    def _calculate_scalping_levels(self, df: pd.DataFrame, pair: str,
                                   direction: str, h1_analysis: Dict) -> Tuple[float, float]:
        """
        Calculate optimal SL and TP for scalping using ATR.

        SL: Based on M5/M1 ATR (tighter for scalping)
        TP: Based on risk/reward and H1 structure
        """
        pip_value = get_pip_value(pair)
        current_price = df['close'].iloc[-1]

        # Calculate ATR on entry timeframe
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=14).mean().iloc[-1]

        # Recent swing high/low for structure-based SL
        swing_high = high.rolling(10).max().iloc[-1]
        swing_low = low.rolling(10).min().iloc[-1]

        if direction == 'BULLISH':
            # SL below recent swing low or 1.5x ATR
            sl_swing = swing_low - (2 * pip_value)  # 2 pips buffer
            sl_atr = current_price - (atr * 1.5)
            optimal_sl = max(sl_swing, sl_atr)  # Choose tighter SL

            # TP at 2.5x risk (scalping R:R)
            risk = current_price - optimal_sl
            optimal_tp = current_price + (risk * 2.5)

        else:  # BEARISH
            # SL above recent swing high or 1.5x ATR
            sl_swing = swing_high + (2 * pip_value)
            sl_atr = current_price + (atr * 1.5)
            optimal_sl = min(sl_swing, sl_atr)  # Choose tighter SL

            # TP at 2.5x risk
            risk = optimal_sl - current_price
            optimal_tp = current_price - (risk * 2.5)

        return optimal_sl, optimal_tp

    def _get_structure_levels(self, h1_df: pd.DataFrame,
                              m15_df: pd.DataFrame) -> Dict:
        """
        Identify key structure levels from H1 and M15.
        """
        # H1 levels
        h1_high = h1_df['high'].rolling(20).max().iloc[-1]
        h1_low = h1_df['low'].rolling(20).min().iloc[-1]

        # M15 levels (more recent)
        m15_high = m15_df['high'].rolling(10).max().iloc[-1]
        m15_low = m15_df['low'].rolling(10).min().iloc[-1]

        # Recent swing points (fractals)
        h1_swings = self._find_swing_points(h1_df)

        return {
            'h1_resistance': h1_high,
            'h1_support': h1_low,
            'm15_resistance': m15_high,
            'm15_support': m15_low,
            'swing_highs': h1_swings['highs'][-3:] if h1_swings['highs'] else [],
            'swing_lows': h1_swings['lows'][-3:] if h1_swings['lows'] else []
        }

    def _find_swing_points(self, df: pd.DataFrame, lookback: int = 5) -> Dict:
        """Find swing highs and lows."""
        highs = []
        lows = []

        for i in range(lookback, len(df) - lookback):
            # Swing high: higher than surrounding candles
            if df['high'].iloc[i] == df['high'].iloc[i-lookback:i+lookback+1].max():
                highs.append(df['high'].iloc[i])
            # Swing low: lower than surrounding candles
            if df['low'].iloc[i] == df['low'].iloc[i-lookback:i+lookback+1].min():
                lows.append(df['low'].iloc[i])

        return {'highs': highs, 'lows': lows}

    def get_trade_direction(self, analysis: MTFAnalysis) -> Optional[str]:
        """
        Get final trade direction from analysis.

        Returns 'BUY', 'SELL', or None
        """
        if analysis.htf_bias == 'BULLISH' and analysis.m15_signal and analysis.m5_entry:
            return 'BUY'
        elif analysis.htf_bias == 'BEARISH' and analysis.m15_signal and analysis.m5_entry:
            return 'SELL'
        return None


# Singleton instance
_mtf_analyzer = None


def get_mtf_analysis(pair: str, mtf_data: Dict[str, pd.DataFrame]) -> Optional[MTFAnalysis]:
    """
    Get MTF analysis for a pair (function interface).
    """
    global _mtf_analyzer
    if _mtf_analyzer is None:
        _mtf_analyzer = MTFAnalyzer()

    return _mtf_analyzer.analyze(pair, mtf_data)
