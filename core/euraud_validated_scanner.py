#!/usr/bin/env python3
"""
EUR/AUD Validated Strategy Scanner V1.0
========================================
Live scanner for validated EUR/AUD strategies.

Based on optimization results:
- Best Strategy: BB %B (Bollinger Band %B)
- Win Rate: 45.0%
- Profit Factor: 1.39
- Walk-Forward Efficiency: 125.2% (PASS)
- Monte Carlo: 100% positive simulations (PASS)

Strategy Logic:
- BUY when %B crosses below 0 (price breaks below lower band)
- SELL when %B crosses above 1 (price breaks above upper band)
- This is a mean reversion strategy at volatility extremes

EUR/AUD Characteristics:
- Commodity currency pair (AUD affected by Iron Ore, Gold, China)
- pip_value: 0.0001 (4 decimal)
- Average daily range: 80-120 pips
- Best sessions: London, Sydney/Tokyo overlap
- Inverse correlation with AUD/USD

Usage:
    python -m core.euraud_validated_scanner

Part of Forex Scalper Agent V2
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
import os
import sys

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.multi_source_fetcher import MultiSourceFetcher
except ImportError:
    from multi_source_fetcher import MultiSourceFetcher


# =============================================================================
# VALIDATED STRATEGY CONFIGURATIONS
# =============================================================================

EURAUD_CONFIG = {
    'pair': 'EURAUD',
    'pip_value': 0.0001,
    'avg_daily_range': 100,  # pips

    # Primary Strategy: BB %B
    'primary_strategy': {
        'name': 'BB_PCT_B',
        'params': {
            'bb_period': 20,
            'bb_std': 2.0,
            'rr': 1.5,
            'sl_atr_mult': 2.5,
        },
        'performance': {
            'profit_factor': 1.39,
            'win_rate': 45.0,
            'monthly_trades': 10,
            'wfe': 125.2,
        }
    },

    # Alternative Strategies
    'alternative_strategies': [
        {
            'name': 'BB_RSI',
            'profit_factor': 1.13,
            'win_rate': 34.7,
            'description': 'BB lower + RSI oversold'
        },
        {
            'name': 'ICHIMOKU_BASIC',
            'profit_factor': 1.12,
            'win_rate': 38.1,
            'description': 'Price above cloud + Tenkan > Kijun'
        },
        {
            'name': 'RSI_OVERSOLD',
            'profit_factor': 1.11,
            'win_rate': 34.4,
            'description': 'RSI crosses back above 30'
        },
    ],

    # Regime Recommendations
    'trade_regimes': ['TRENDING_UP', 'TRENDING_DOWN', 'STRONG_TREND_UP', 'RANGING'],
    'avoid_regimes': ['HIGH_VOLATILITY', 'STRONG_TREND_DOWN'],

    # Session Recommendations
    'best_sessions': {
        'LONDON': {'start': 7, 'end': 16, 'quality': 'BEST'},
        'SYDNEY_LONDON': {'start': 6, 'end': 8, 'quality': 'GOOD'},
    },
    'avoid_sessions': ['LATE_NY'],

    # Risk Management
    'risk': {
        'max_daily_loss': 500,  # USD
        'optimal_lot': 0.05,
        'max_concurrent': 2,
        'max_daily_trades': 3,
    }
}


# =============================================================================
# INDICATORS
# =============================================================================

class Indicators:
    """Technical indicators for EUR/AUD."""

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20,
                        std_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands with %B."""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_mult)
        lower = middle - (std * std_mult)
        pct_b = (series - lower) / (upper - lower + 1e-10)
        return upper, middle, lower, pct_b

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """RSI indicator."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """ADX with +DI and -DI."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr_smooth = tr.rolling(window=period).sum()
        plus_dm_smooth = plus_dm.rolling(window=period).sum()
        minus_dm_smooth = minus_dm.rolling(window=period).sum()

        plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
        minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(window=period).mean()

        return adx, plus_di, minus_di


# =============================================================================
# REGIME DETECTOR
# =============================================================================

class MarketRegime(Enum):
    """Market regime classification."""
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    STRONG_TREND_UP = "STRONG_TREND_UP"
    STRONG_TREND_DOWN = "STRONG_TREND_DOWN"
    RANGING = "RANGING"
    CONSOLIDATION = "CONSOLIDATION"
    NORMAL = "NORMAL"
    UNKNOWN = "UNKNOWN"


def detect_regime(df: pd.DataFrame) -> Tuple[MarketRegime, Dict]:
    """Detect current market regime."""
    if len(df) < 60:
        return MarketRegime.UNKNOWN, {}

    # Calculate indicators
    atr = Indicators.atr(df['high'], df['low'], df['close'])
    adx, plus_di, minus_di = Indicators.adx(df['high'], df['low'], df['close'])
    bb_upper, bb_middle, bb_lower, _ = Indicators.bollinger_bands(df['close'])
    bb_width = (bb_upper - bb_lower) / bb_middle * 100

    # Get current values
    current_atr = atr.iloc[-1]
    current_adx = adx.iloc[-1]
    current_bb_width = bb_width.iloc[-1]
    current_plus_di = plus_di.iloc[-1]
    current_minus_di = minus_di.iloc[-1]

    # Get averages
    atr_avg = atr.iloc[-60:].mean()
    bb_width_avg = bb_width.iloc[-60:].mean()

    details = {
        'atr': round(current_atr * 10000, 1),  # Convert to pips
        'atr_ratio': round(current_atr / atr_avg, 2) if atr_avg > 0 else 1,
        'adx': round(current_adx, 1),
        'plus_di': round(current_plus_di, 1),
        'minus_di': round(current_minus_di, 1),
        'bb_width': round(current_bb_width, 2)
    }

    # Classification
    if current_atr > atr_avg * 1.5:
        return MarketRegime.HIGH_VOLATILITY, details

    if current_atr < atr_avg * 0.5:
        return MarketRegime.LOW_VOLATILITY, details

    if current_adx > 40:
        if current_plus_di > current_minus_di:
            return MarketRegime.STRONG_TREND_UP, details
        else:
            return MarketRegime.STRONG_TREND_DOWN, details

    if current_adx > 25:
        if current_plus_di > current_minus_di:
            return MarketRegime.TRENDING_UP, details
        else:
            return MarketRegime.TRENDING_DOWN, details

    if current_bb_width < bb_width_avg * 0.6:
        return MarketRegime.CONSOLIDATION, details

    if current_adx < 20:
        return MarketRegime.RANGING, details

    return MarketRegime.NORMAL, details


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

@dataclass
class Signal:
    """Trading signal."""
    pair: str
    direction: str  # 'BUY' or 'SELL'
    strategy: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float
    regime: str
    timestamp: datetime
    details: Dict


def generate_bb_pct_b_signal(df: pd.DataFrame, config: Dict) -> Optional[Signal]:
    """Generate signal based on BB %B strategy."""
    params = config['primary_strategy']['params']

    # Calculate indicators
    bb_upper, bb_middle, bb_lower, pct_b = Indicators.bollinger_bands(
        df['close'],
        period=params['bb_period'],
        std_mult=params['bb_std']
    )
    atr = Indicators.atr(df['high'], df['low'], df['close'])

    # Get current and previous values
    current_pct_b = pct_b.iloc[-1]
    prev_pct_b = pct_b.iloc[-2]
    current_close = df['close'].iloc[-1]
    current_atr = atr.iloc[-1]

    # Detect regime
    regime, regime_details = detect_regime(df)
    regime_name = regime.value

    # Check if regime is tradeable
    is_tradeable_regime = regime_name in config['trade_regimes']

    signal_direction = None
    confidence = 0.0

    # BUY signal: %B crosses below 0 (now recovering)
    if prev_pct_b < 0 and current_pct_b >= 0:
        signal_direction = 'BUY'
        confidence = 0.7 if is_tradeable_regime else 0.5

    # SELL signal: %B crosses above 1 (now falling back)
    elif prev_pct_b > 1 and current_pct_b <= 1:
        signal_direction = 'SELL'
        confidence = 0.7 if is_tradeable_regime else 0.5

    if signal_direction is None:
        return None

    # Calculate SL and TP
    sl_distance = current_atr * params['sl_atr_mult']
    tp_distance = sl_distance * params['rr']

    if signal_direction == 'BUY':
        stop_loss = current_close - sl_distance
        take_profit = current_close + tp_distance
    else:
        stop_loss = current_close + sl_distance
        take_profit = current_close - tp_distance

    # Calculate pips
    sl_pips = sl_distance / config['pip_value']
    tp_pips = tp_distance / config['pip_value']

    return Signal(
        pair=config['pair'],
        direction=signal_direction,
        strategy='BB_PCT_B',
        entry_price=current_close,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence,
        regime=regime_name,
        timestamp=df.index[-1],
        details={
            'pct_b': round(current_pct_b, 3),
            'prev_pct_b': round(prev_pct_b, 3),
            'bb_upper': round(bb_upper.iloc[-1], 5),
            'bb_lower': round(bb_lower.iloc[-1], 5),
            'atr_pips': round(current_atr / config['pip_value'], 1),
            'sl_pips': round(sl_pips, 1),
            'tp_pips': round(tp_pips, 1),
            'regime_details': regime_details,
            'tradeable_regime': is_tradeable_regime
        }
    )


def generate_bb_rsi_signal(df: pd.DataFrame, config: Dict) -> Optional[Signal]:
    """Generate signal based on BB + RSI strategy (alternative)."""
    # Calculate indicators
    bb_upper, bb_middle, bb_lower, pct_b = Indicators.bollinger_bands(df['close'])
    rsi = Indicators.rsi(df['close'])
    atr = Indicators.atr(df['high'], df['low'], df['close'])

    current_close = df['close'].iloc[-1]
    current_rsi = rsi.iloc[-1]
    current_bb_lower = bb_lower.iloc[-1]
    current_bb_upper = bb_upper.iloc[-1]
    current_atr = atr.iloc[-1]

    regime, regime_details = detect_regime(df)
    regime_name = regime.value
    is_tradeable_regime = regime_name in config['trade_regimes']

    signal_direction = None
    confidence = 0.0

    # BUY: Price at/below lower BB AND RSI oversold
    if current_close <= current_bb_lower and current_rsi < 30:
        signal_direction = 'BUY'
        confidence = 0.6 if is_tradeable_regime else 0.4

    # SELL: Price at/above upper BB AND RSI overbought
    elif current_close >= current_bb_upper and current_rsi > 70:
        signal_direction = 'SELL'
        confidence = 0.6 if is_tradeable_regime else 0.4

    if signal_direction is None:
        return None

    # Calculate SL and TP with default params
    sl_distance = current_atr * 2.0
    tp_distance = sl_distance * 1.5

    if signal_direction == 'BUY':
        stop_loss = current_close - sl_distance
        take_profit = current_close + tp_distance
    else:
        stop_loss = current_close + sl_distance
        take_profit = current_close - tp_distance

    return Signal(
        pair=config['pair'],
        direction=signal_direction,
        strategy='BB_RSI',
        entry_price=current_close,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence,
        regime=regime_name,
        timestamp=df.index[-1],
        details={
            'rsi': round(current_rsi, 1),
            'bb_upper': round(current_bb_upper, 5),
            'bb_lower': round(current_bb_lower, 5),
            'atr_pips': round(current_atr / config['pip_value'], 1),
            'tradeable_regime': is_tradeable_regime
        }
    )


# =============================================================================
# SCANNER
# =============================================================================

class EURAUDScanner:
    """Live scanner for EUR/AUD validated strategies."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.config = EURAUD_CONFIG
        self.fetcher = MultiSourceFetcher(verbose=False)

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def get_session_quality(self) -> Tuple[str, str]:
        """Determine current session and quality."""
        now = datetime.utcnow()
        hour = now.hour

        # Check each session
        for session_name, session_info in self.config['best_sessions'].items():
            if session_info['start'] <= hour < session_info['end']:
                return session_name, session_info['quality']

        # Default
        if 12 <= hour < 21:
            return 'NEW_YORK', 'MODERATE'
        elif hour < 7:
            return 'ASIAN', 'LOW'
        else:
            return 'TRANSITION', 'LOW'

    def scan(self) -> Dict:
        """Run scanner and return analysis."""
        self.log("\n" + "=" * 70)
        self.log(f"EUR/AUD VALIDATED STRATEGY SCANNER")
        self.log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.log("=" * 70)

        # Get session info
        session, session_quality = self.get_session_quality()
        self.log(f"\nSession: {session} (Quality: {session_quality})")

        # Fetch data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

        self.log(f"Fetching data...")
        df = self.fetcher.fetch(self.config['pair'], start_date, end_date, '1h')

        if df is None or len(df) < 100:
            self.log("[ERROR] Insufficient data")
            return {'error': 'Insufficient data'}

        self.log(f"Got {len(df)} bars")

        # Detect regime
        regime, regime_details = detect_regime(df)
        regime_name = regime.value
        is_tradeable = regime_name in self.config['trade_regimes']

        self.log(f"\nMarket Regime: {regime_name}")
        self.log(f"  ADX: {regime_details.get('adx', 'N/A')}")
        self.log(f"  +DI: {regime_details.get('plus_di', 'N/A')} | -DI: {regime_details.get('minus_di', 'N/A')}")
        self.log(f"  ATR: {regime_details.get('atr', 'N/A')} pips")
        self.log(f"  Tradeable: {'YES' if is_tradeable else 'NO - AVOID'}")

        # Current price info
        current_close = df['close'].iloc[-1]
        self.log(f"\nCurrent Price: {current_close:.5f}")

        # Generate signals
        self.log(f"\n{'='*50}")
        self.log("SIGNAL ANALYSIS")
        self.log('='*50)

        signals = []

        # Primary strategy: BB %B
        bb_signal = generate_bb_pct_b_signal(df, self.config)
        if bb_signal:
            signals.append(bb_signal)
            self.log(f"\n[PRIMARY] BB %B Signal: {bb_signal.direction}")
            self.log(f"  Entry: {bb_signal.entry_price:.5f}")
            self.log(f"  SL: {bb_signal.stop_loss:.5f} ({bb_signal.details['sl_pips']:.1f} pips)")
            self.log(f"  TP: {bb_signal.take_profit:.5f} ({bb_signal.details['tp_pips']:.1f} pips)")
            self.log(f"  %B: {bb_signal.details['pct_b']:.3f}")
            self.log(f"  Confidence: {bb_signal.confidence*100:.0f}%")
        else:
            self.log(f"\n[PRIMARY] BB %B: No signal")

        # Alternative: BB + RSI
        bb_rsi_signal = generate_bb_rsi_signal(df, self.config)
        if bb_rsi_signal:
            signals.append(bb_rsi_signal)
            self.log(f"\n[ALT] BB+RSI Signal: {bb_rsi_signal.direction}")
            self.log(f"  Entry: {bb_rsi_signal.entry_price:.5f}")
            self.log(f"  RSI: {bb_rsi_signal.details['rsi']:.1f}")
            self.log(f"  Confidence: {bb_rsi_signal.confidence*100:.0f}%")
        else:
            self.log(f"\n[ALT] BB+RSI: No signal")

        # Summary
        self.log(f"\n{'='*50}")
        self.log("SUMMARY")
        self.log('='*50)

        if signals:
            best_signal = max(signals, key=lambda x: x.confidence)

            self.log(f"\nBest Signal: {best_signal.strategy} {best_signal.direction}")
            self.log(f"Confidence: {best_signal.confidence*100:.0f}%")

            if is_tradeable and session_quality in ['BEST', 'GOOD']:
                self.log(f"\n[OK] CONDITIONS FAVORABLE FOR TRADING")
                self.log(f"  - Regime: {regime_name} (tradeable)")
                self.log(f"  - Session: {session} ({session_quality})")
            else:
                self.log(f"\n[WARNING] CONDITIONS NOT OPTIMAL")
                if not is_tradeable:
                    self.log(f"  - Regime {regime_name} is not ideal")
                if session_quality not in ['BEST', 'GOOD']:
                    self.log(f"  - Session {session} has {session_quality} quality")
        else:
            self.log(f"\nNo active signals. Market conditions:")
            bb_upper, bb_middle, bb_lower, pct_b = Indicators.bollinger_bands(df['close'])
            rsi = Indicators.rsi(df['close'])

            self.log(f"  Price: {current_close:.5f}")
            self.log(f"  BB Upper: {bb_upper.iloc[-1]:.5f}")
            self.log(f"  BB Middle: {bb_middle.iloc[-1]:.5f}")
            self.log(f"  BB Lower: {bb_lower.iloc[-1]:.5f}")
            self.log(f"  %B: {pct_b.iloc[-1]:.3f}")
            self.log(f"  RSI: {rsi.iloc[-1]:.1f}")

            # Proximity analysis
            if pct_b.iloc[-1] < 0.2:
                self.log(f"\n  [INFO] Price near lower band - watch for BUY signal")
            elif pct_b.iloc[-1] > 0.8:
                self.log(f"\n  [INFO] Price near upper band - watch for SELL signal")
            else:
                self.log(f"\n  [INFO] Price in middle range - no immediate opportunities")

        return {
            'pair': self.config['pair'],
            'timestamp': datetime.now(),
            'session': session,
            'session_quality': session_quality,
            'regime': regime_name,
            'regime_details': regime_details,
            'is_tradeable_regime': is_tradeable,
            'current_price': current_close,
            'signals': signals,
            'best_signal': max(signals, key=lambda x: x.confidence) if signals else None
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run EUR/AUD scanner."""
    print("\n" + "=" * 70)
    print("EUR/AUD VALIDATED STRATEGY SCANNER V1.0")
    print("=" * 70)
    print("\nStrategy: BB %B (Bollinger Band Percent B)")
    print("Performance: PF=1.39, WR=45%, WFE=125%")
    print("\nOptimal Parameters:")
    print("  - R:R Ratio: 1.5")
    print("  - SL Multiplier: 2.5x ATR")
    print("  - BB Period: 20, Std: 2.0")

    scanner = EURAUDScanner(verbose=True)
    result = scanner.scan()

    if result.get('best_signal'):
        signal = result['best_signal']
        print("\n" + "=" * 70)
        print("TRADE RECOMMENDATION")
        print("=" * 70)
        print(f"\nPair: {signal.pair}")
        print(f"Direction: {signal.direction}")
        print(f"Strategy: {signal.strategy}")
        print(f"Entry: {signal.entry_price:.5f}")
        print(f"Stop Loss: {signal.stop_loss:.5f}")
        print(f"Take Profit: {signal.take_profit:.5f}")
        print(f"Confidence: {signal.confidence*100:.0f}%")
        print(f"Regime: {signal.regime}")

        # Risk calculation
        lot_size = EURAUD_CONFIG['risk']['optimal_lot']
        pip_value = 10 * lot_size  # ~$10 per pip at 0.1 lot for EUR/AUD
        sl_pips = signal.details['sl_pips']
        tp_pips = signal.details['tp_pips']

        print(f"\nRisk Management (at {lot_size} lots):")
        print(f"  Potential Loss: ${sl_pips * pip_value:.2f}")
        print(f"  Potential Profit: ${tp_pips * pip_value:.2f}")
        print(f"  R:R Ratio: 1:{tp_pips/sl_pips:.1f}")

    print("\n" + "=" * 70)
    print("STRATEGY RULES REMINDER")
    print("=" * 70)
    print("""
BB %B Strategy Rules:

  BUY Signal:
    - %B crosses from below 0 to above 0
    - Price was below lower Bollinger Band, now recovering
    - Best in TRENDING_UP or RANGING regimes

  SELL Signal:
    - %B crosses from above 1 to below 1
    - Price was above upper Bollinger Band, now falling
    - Best in TRENDING_DOWN or RANGING regimes

  Entry:
    - Wait for %B crossover confirmation
    - Enter at close of signal candle

  Stop Loss:
    - 2.5x ATR from entry

  Take Profit:
    - 1.5x (SL distance) = 3.75x ATR from entry

  Avoid:
    - HIGH_VOLATILITY regime
    - STRONG_TREND_DOWN regime
    - Late New York session
""")

    return result


if __name__ == "__main__":
    main()
