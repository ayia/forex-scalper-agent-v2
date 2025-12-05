#!/usr/bin/env python3
"""
EUR/JPY Validated Scanner V1.0
================================
Production-ready scanner for EUR/JPY with validated strategies and regime filtering.

Based on comprehensive backtest analysis:
- Best Strategy: Range Breakout (#40)
- Optimal R:R: 2.5
- SL: 1.5x ATR
- Profit Factor: 1.58 (optimized)

Alternative strategies included for confluence:
- Mean Reversion (#36): PF=1.22, best in RANGING/CONSOLIDATION
- Stochastic Cross (#14): PF=1.21, best in TRENDING_DOWN
- MACD+Stochastic (#31): PF=1.97, low trades but good for confirmation

Regime Analysis Results (from backtest):
- Range Breakout works best in: LONDON session, STRONG_TREND, NORMAL volatility
- Mean Reversion works best in: RANGING, CONSOLIDATION, any session
- Stochastic Cross works best in: TRENDING_DOWN, HIGH volatility

Usage:
    python -m core.eurjpy_validated_scanner
    python -m core.eurjpy_validated_scanner --json
    python -m core.eurjpy_validated_scanner --rules

Part of Forex Scalper Agent V2
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, Optional, List
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - Validated from backtest optimization
# =============================================================================

EURJPY_CONFIG = {
    'pair': 'EURJPY',
    'symbol': 'EURJPY=X',
    'pip_value': 0.01,

    # Primary strategy: Range Breakout
    'primary_strategy': 'RANGE_BREAKOUT',
    'rr': 2.5,
    'sl_mult': 1.5,

    # Signal filters
    'range_threshold': 0.7,   # Range must be < 70% of average
    'lookback_range': 8,      # 8 bars for range
    'lookback_avg': 20,       # 20 bars for average range

    # Backtest performance
    'backtest_pf': 1.58,
    'backtest_wr': 39.3,
    'monte_carlo_positive': 100.0,

    # Alternative confirmation thresholds
    'stoch_oversold': 30,
    'stoch_overbought': 70,
    'zscore_threshold': 2.0,
}

# =============================================================================
# REGIME RULES - Based on comprehensive analysis
# =============================================================================

REGIME_RULES = {
    # BEST REGIMES for Range Breakout
    'STRONG_TREND_UP': {
        'tradeable': True,
        'position_mult': 1.5,  # Increase size - PF=2.89
        'reason': 'Excellent PF (2.89), 48% win rate',
        'recommended_strategy': 'RANGE_BREAKOUT'
    },
    'STRONG_TREND_DOWN': {
        'tradeable': True,
        'position_mult': 1.3,  # PF=1.57
        'reason': 'Good PF (1.57), breakouts work well',
        'recommended_strategy': 'RANGE_BREAKOUT'
    },
    'TRENDING_DOWN': {
        'tradeable': True,
        'position_mult': 1.2,  # PF=1.56
        'reason': 'Good PF (1.56)',
        'recommended_strategy': 'RANGE_BREAKOUT'
    },
    'NORMAL': {
        'tradeable': True,
        'position_mult': 1.0,  # PF=1.32
        'reason': 'Acceptable PF (1.32)',
        'recommended_strategy': 'RANGE_BREAKOUT'
    },
    'TRENDING_UP': {
        'tradeable': True,
        'position_mult': 0.8,  # PF=1.19
        'reason': 'Marginal PF (1.19), use caution',
        'recommended_strategy': 'RANGE_BREAKOUT'
    },

    # BEST for Mean Reversion
    'CONSOLIDATION': {
        'tradeable': True,
        'position_mult': 1.3,  # Mean reversion PF=2.87
        'reason': 'Excellent for Mean Reversion (PF=2.87)',
        'recommended_strategy': 'MEAN_REVERSION'
    },
    'RANGING': {
        'tradeable': True,
        'position_mult': 1.3,  # Mean reversion PF=2.64
        'reason': 'Excellent for Mean Reversion (PF=2.64)',
        'recommended_strategy': 'MEAN_REVERSION'
    },

    # CAUTION
    'HIGH_VOLATILITY': {
        'tradeable': True,
        'position_mult': 0.5,  # Reduced size
        'reason': 'Mean Reversion works (PF=2.12), but reduce size',
        'recommended_strategy': 'MEAN_REVERSION'
    },

    # LOW_VOLATILITY - Rare
    'LOW_VOLATILITY': {
        'tradeable': True,
        'position_mult': 0.8,
        'reason': 'MACD+Stoch excellent (PF=6.61)',
        'recommended_strategy': 'MACD_STOCHASTIC'
    },

    # UNKNOWN
    'UNKNOWN': {
        'tradeable': False,
        'position_mult': 0,
        'reason': 'Insufficient data for classification'
    },
}

# Session rules based on analysis
SESSION_RULES = {
    'ASIAN': {
        'tradeable': True,
        'mult': 0.8,
        'reason': 'Mean Reversion best (PF=1.60), Range Breakout neutral (PF=1.00)'
    },
    'LONDON': {
        'tradeable': True,
        'mult': 1.3,
        'reason': 'Range Breakout excellent (PF=2.16), best session'
    },
    'NEW_YORK': {
        'tradeable': True,
        'mult': 1.0,
        'reason': 'Good for all strategies (PF~1.30)'
    },
}

# Volatility rules
VOLATILITY_RULES = {
    'VERY_LOW': {
        'tradeable': True,
        'mult': 0.5,
        'reason': 'Range Breakout poor (PF=0.11), use Mean Reversion'
    },
    'LOW': {
        'tradeable': True,
        'mult': 1.0,
        'reason': 'All strategies work well (PF~1.2-1.5)'
    },
    'NORMAL': {
        'tradeable': True,
        'mult': 1.2,
        'reason': 'Range Breakout good (PF=1.51)'
    },
    'HIGH': {
        'tradeable': True,
        'mult': 1.3,
        'reason': 'Range Breakout excellent (PF=1.85), Stoch Cross (PF=1.73)'
    },
    'VERY_HIGH': {
        'tradeable': True,
        'mult': 0.5,
        'reason': 'High risk, reduce position'
    },
}


class EURJPYValidatedScanner:
    """Production scanner for EUR/JPY with multi-strategy support."""

    def __init__(self):
        self.config = EURJPY_CONFIG
        self.pair = self.config['pair']
        self.symbol = self.config['symbol']

    def fetch_data(self, period: str = "5d", interval: str = "1h") -> Optional[pd.DataFrame]:
        """Fetch market data."""
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                return None
            return df
        except Exception:
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators."""
        df = df.copy()

        # EMAs
        df['ema8'] = df['Close'].ewm(span=8).mean()
        df['ema21'] = df['Close'].ewm(span=21).mean()
        df['ema50'] = df['Close'].ewm(span=50).mean()
        df['ema20'] = df['Close'].ewm(span=20).mean()

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Stochastic
        low_14 = df['Low'].rolling(14).min()
        high_14 = df['High'].rolling(14).max()
        df['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()

        # MACD
        ema12 = df['Close'].ewm(span=12).mean()
        ema26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ATR
        df['tr'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift(1)),
                abs(df['Low'] - df['Close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()

        # ADX
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr14 = df['tr'].rolling(14).mean()
        df['plus_di'] = 100 * (plus_dm.rolling(14).mean() / atr14)
        df['minus_di'] = 100 * (minus_dm.rolling(14).mean() / atr14)
        dx = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = dx.rolling(14).mean()

        # Bollinger Bands
        df['bb_mid'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * bb_std
        df['bb_lower'] = df['bb_mid'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100

        # Volatility ratio
        df['atr_sma'] = df['atr'].rolling(20).mean()
        df['volatility_ratio'] = df['atr'] / df['atr_sma']
        df['bb_width_sma'] = df['bb_width'].rolling(20).mean()

        # Z-Score for mean reversion
        df['zscore'] = (df['Close'] - df['Close'].rolling(20).mean()) / (df['Close'].rolling(20).std())

        # Range calculations for breakout strategy
        df['range_high'] = df['High'].rolling(self.config['lookback_range']).max()
        df['range_low'] = df['Low'].rolling(self.config['lookback_range']).min()
        df['range_size'] = df['range_high'] - df['range_low']
        df['avg_range'] = df['range_size'].rolling(self.config['lookback_avg']).mean()
        df['range_ratio'] = df['range_size'] / df['avg_range']

        return df

    def classify_regime(self, row) -> Dict:
        """Classify current market regime."""
        adx = row['adx']
        volatility_ratio = row['volatility_ratio']
        plus_di = row['plus_di']
        minus_di = row['minus_di']
        bb_width = row['bb_width']
        bb_width_sma = row['bb_width_sma']

        # Determine regime
        if volatility_ratio > 1.5 or (bb_width > bb_width_sma * 1.5):
            regime = 'HIGH_VOLATILITY'
            reason = f'ATR ratio={volatility_ratio:.2f} (>1.5)'
        elif volatility_ratio < 0.5 or (bb_width < bb_width_sma * 0.5):
            regime = 'LOW_VOLATILITY'
            reason = f'ATR ratio={volatility_ratio:.2f} (<0.5)'
        elif adx > 40:
            if plus_di > minus_di:
                regime = 'STRONG_TREND_UP'
                reason = f'ADX={adx:.1f} (>40), +DI>-DI'
            else:
                regime = 'STRONG_TREND_DOWN'
                reason = f'ADX={adx:.1f} (>40), -DI>+DI'
        elif adx > 25:
            if plus_di > minus_di:
                regime = 'TRENDING_UP'
                reason = f'ADX={adx:.1f} (25-40), +DI>-DI'
            else:
                regime = 'TRENDING_DOWN'
                reason = f'ADX={adx:.1f} (25-40), -DI>+DI'
        elif adx < 20:
            regime = 'RANGING'
            reason = f'ADX={adx:.1f} (<20)'
        elif bb_width < bb_width_sma * 0.6:
            regime = 'CONSOLIDATION'
            reason = f'BB squeeze, width={bb_width:.2f}'
        else:
            regime = 'NORMAL'
            reason = f'ADX={adx:.1f} (20-25)'

        rule = REGIME_RULES.get(regime, REGIME_RULES['NORMAL'])

        return {
            'regime': regime,
            'tradeable': rule['tradeable'],
            'position_mult': rule['position_mult'],
            'reason': rule['reason'],
            'recommended_strategy': rule.get('recommended_strategy', 'RANGE_BREAKOUT'),
            'classification_reason': reason
        }

    def classify_session(self, timestamp) -> Dict:
        """Classify trading session."""
        hour = timestamp.hour
        if 0 <= hour < 8:
            session = 'ASIAN'
        elif 8 <= hour < 16:
            session = 'LONDON'
        else:
            session = 'NEW_YORK'

        rule = SESSION_RULES[session]
        return {
            'session': session,
            'tradeable': rule['tradeable'],
            'mult': rule['mult'],
            'reason': rule['reason']
        }

    def classify_volatility(self, volatility_ratio: float) -> Dict:
        """Classify volatility level."""
        if volatility_ratio < 0.5:
            level = 'VERY_LOW'
        elif volatility_ratio < 0.85:
            level = 'LOW'
        elif volatility_ratio < 1.15:
            level = 'NORMAL'
        elif volatility_ratio < 1.5:
            level = 'HIGH'
        else:
            level = 'VERY_HIGH'

        rule = VOLATILITY_RULES[level]
        return {
            'level': level,
            'tradeable': rule['tradeable'],
            'mult': rule['mult'],
            'reason': rule['reason']
        }

    def check_range_breakout(self, current, prev) -> Optional[str]:
        """Check for Range Breakout signal."""
        range_ratio = current['range_ratio']
        range_high = prev['range_high']
        range_low = prev['range_low']
        close = current['Close']

        # Check if range is tight enough
        if range_ratio >= self.config['range_threshold']:
            return None  # Range too wide

        # Check for breakout
        if close > range_high:
            return 'BUY'
        elif close < range_low:
            return 'SELL'

        return None

    def check_mean_reversion(self, current) -> Optional[str]:
        """Check for Mean Reversion signal."""
        zscore = current['zscore']
        threshold = self.config['zscore_threshold']

        if zscore < -threshold:
            return 'BUY'
        elif zscore > threshold:
            return 'SELL'

        return None

    def check_stochastic_cross(self, current, prev) -> Optional[str]:
        """Check for Stochastic Cross signal."""
        oversold = self.config['stoch_oversold']
        overbought = self.config['stoch_overbought']

        k_crosses_up = (current['stoch_k'] > current['stoch_d']) and (prev['stoch_k'] <= prev['stoch_d'])
        k_crosses_down = (current['stoch_k'] < current['stoch_d']) and (prev['stoch_k'] >= prev['stoch_d'])

        if k_crosses_up and current['stoch_k'] < oversold + 10:
            return 'BUY'
        elif k_crosses_down and current['stoch_k'] > overbought - 10:
            return 'SELL'

        return None

    def check_macd_stochastic(self, current, prev) -> Optional[str]:
        """Check for MACD+Stochastic confirmation signal."""
        oversold = self.config['stoch_oversold']
        overbought = self.config['stoch_overbought']

        macd_cross_up = (current['macd'] > current['macd_signal']) and (prev['macd'] <= prev['macd_signal'])
        macd_cross_down = (current['macd'] < current['macd_signal']) and (prev['macd'] >= prev['macd_signal'])

        if macd_cross_up and current['stoch_k'] < oversold:
            return 'BUY'
        elif macd_cross_down and current['stoch_k'] > overbought:
            return 'SELL'

        return None

    def scan(self) -> Dict:
        """Perform full scan with multi-strategy analysis."""
        # Fetch data
        df = self.fetch_data()
        if df is None or len(df) < 50:
            return {'pair': self.pair, 'error': 'Insufficient data'}

        # Calculate indicators
        df = self.calculate_indicators(df)

        # Get current and previous bar
        current = df.iloc[-1]
        prev = df.iloc[-2]
        timestamp = df.index[-1]

        # Classify market conditions
        regime_info = self.classify_regime(current)
        session_info = self.classify_session(timestamp)
        volatility_info = self.classify_volatility(current['volatility_ratio'])

        # Check all strategies
        signals = {}
        signals['range_breakout'] = self.check_range_breakout(current, prev)
        signals['mean_reversion'] = self.check_mean_reversion(current)
        signals['stochastic_cross'] = self.check_stochastic_cross(current, prev)
        signals['macd_stochastic'] = self.check_macd_stochastic(current, prev)

        # Determine best strategy based on regime
        recommended = regime_info['recommended_strategy']
        strategy_map = {
            'RANGE_BREAKOUT': 'range_breakout',
            'MEAN_REVERSION': 'mean_reversion',
            'STOCHASTIC_CROSS': 'stochastic_cross',
            'MACD_STOCHASTIC': 'macd_stochastic'
        }
        primary_key = strategy_map.get(recommended, 'range_breakout')
        primary_signal = signals.get(primary_key)

        # Determine final direction
        final_direction = 'WATCH'
        block_reasons = []
        position_multiplier = 1.0

        active_signals = [(k, v) for k, v in signals.items() if v is not None]

        if active_signals:
            # Use primary signal if available, otherwise first active
            if primary_signal:
                raw_signal = primary_signal
                active_strategy = primary_key
            else:
                active_strategy, raw_signal = active_signals[0]

            # Apply regime filter
            if not regime_info['tradeable']:
                block_reasons.append(f"Regime {regime_info['regime']}: {regime_info['reason']}")
            else:
                position_multiplier *= regime_info['position_mult']

            # Apply session filter
            position_multiplier *= session_info['mult']

            # Apply volatility filter
            position_multiplier *= volatility_info['mult']

            # Final decision
            if block_reasons:
                final_direction = 'BLOCKED'
            else:
                final_direction = raw_signal
        else:
            raw_signal = None
            active_strategy = None

        # Calculate entry, SL, TP
        entry = round(current['Close'], 3)
        atr = current['atr']

        if final_direction == 'BUY':
            sl = round(entry - self.config['sl_mult'] * atr, 3)
            tp = round(entry + self.config['rr'] * self.config['sl_mult'] * atr, 3)
            sl_pips = round((entry - sl) / self.config['pip_value'], 1)
            tp_pips = round((tp - entry) / self.config['pip_value'], 1)
        elif final_direction == 'SELL':
            sl = round(entry + self.config['sl_mult'] * atr, 3)
            tp = round(entry - self.config['rr'] * self.config['sl_mult'] * atr, 3)
            sl_pips = round((sl - entry) / self.config['pip_value'], 1)
            tp_pips = round((entry - tp) / self.config['pip_value'], 1)
        else:
            sl = tp = sl_pips = tp_pips = None

        # Calculate confluence score
        confluence = 40  # Base
        if raw_signal:
            confluence += 15
        if regime_info['tradeable']:
            confluence += 15
        if session_info['mult'] >= 1.0:
            confluence += 10
        if volatility_info['mult'] >= 1.0:
            confluence += 10
        if len(active_signals) >= 2:
            confluence += 10  # Multiple strategies agree

        # Check strategy agreement
        agreement = len(set([v for k, v in signals.items() if v is not None]))
        strategies_agree = agreement == 1 if active_signals else False

        return {
            'pair': self.pair,
            'timestamp': str(timestamp),
            'direction': final_direction,
            'raw_signal': raw_signal,
            'active_strategy': active_strategy,
            'recommended_strategy': recommended,

            # All signals
            'signals': {
                'range_breakout': signals['range_breakout'],
                'mean_reversion': signals['mean_reversion'],
                'stochastic_cross': signals['stochastic_cross'],
                'macd_stochastic': signals['macd_stochastic'],
            },
            'strategies_agree': strategies_agree,

            # Prices
            'entry': entry,
            'stop_loss': sl,
            'take_profit': tp,
            'sl_pips': sl_pips,
            'tp_pips': tp_pips,
            'rr_ratio': self.config['rr'],

            # Indicators
            'stoch_k': round(current['stoch_k'], 1),
            'stoch_d': round(current['stoch_d'], 1),
            'macd': round(current['macd'], 4),
            'macd_signal': round(current['macd_signal'], 4),
            'zscore': round(current['zscore'], 2),
            'range_ratio': round(current['range_ratio'], 2),

            # Regime
            'regime': regime_info['regime'],
            'regime_tradeable': regime_info['tradeable'],
            'regime_reason': regime_info['reason'],
            'regime_details': regime_info,

            # Session
            'session': session_info['session'],
            'session_mult': session_info['mult'],
            'session_reason': session_info['reason'],

            # Volatility
            'volatility': round(current['volatility_ratio'] * 100, 1),
            'volatility_level': volatility_info['level'],
            'volatility_mult': volatility_info['mult'],

            # Other
            'adx': round(current['adx'], 1),
            'rsi': round(current['rsi'], 1),
            'atr': round(atr, 3),
            'plus_di': round(current['plus_di'], 1),
            'minus_di': round(current['minus_di'], 1),

            # Position
            'position_multiplier': round(position_multiplier, 2),
            'block_reasons': block_reasons,
            'confluence_score': confluence,

            # Backtest reference
            'backtest_pf': self.config['backtest_pf'],
            'backtest_wr': self.config['backtest_wr'],
        }

    def get_rules_summary(self) -> Dict:
        """Get summary of all trading rules."""
        return {
            'regime_rules': REGIME_RULES,
            'session_rules': SESSION_RULES,
            'volatility_rules': VOLATILITY_RULES,
            'strategies': {
                'range_breakout': 'Break of tight range (< 70% average) - Best for STRONG_TREND',
                'mean_reversion': 'Z-Score > 2 or < -2 - Best for RANGING/CONSOLIDATION',
                'stochastic_cross': '%K crosses %D in zones - Best for TRENDING_DOWN',
                'macd_stochastic': 'MACD cross + Stoch confirmation - High PF but low trades'
            },
            'optimal_params': {
                'rr': self.config['rr'],
                'sl_mult': self.config['sl_mult'],
                'backtest_pf': self.config['backtest_pf'],
            }
        }


def print_signal(signal: Dict):
    """Pretty print a signal."""
    if 'error' in signal:
        print(f"\n[X] {signal['pair']}: {signal['error']}")
        return

    print(f"\n{'='*65}")
    print(f"EUR/JPY VALIDATED SCANNER - {signal['timestamp'][:19]}")
    print(f"{'='*65}")

    # Direction
    direction = signal['direction']
    if direction == 'BUY':
        print(f"\n[+] SIGNAL: BUY ({signal['active_strategy']})")
    elif direction == 'SELL':
        print(f"\n[-] SIGNAL: SELL ({signal['active_strategy']})")
    elif direction == 'BLOCKED':
        print(f"\n[!] SIGNAL: BLOCKED")
        for reason in signal['block_reasons']:
            print(f"    -> {reason}")
    else:
        print(f"\n[~] SIGNAL: WATCH (No entry)")

    # Strategy signals
    print(f"\n[STRATEGY SIGNALS]")
    for strategy, sig in signal['signals'].items():
        status = "[OK]" if sig else "[--]"
        print(f"   {status} {strategy}: {sig if sig else 'None'}")

    if signal['strategies_agree']:
        print(f"   >> Multiple strategies agree!")

    print(f"   Recommended: {signal['recommended_strategy']}")

    # Prices
    if signal['entry']:
        print(f"\n[TRADE LEVELS]")
        print(f"   Entry:       {signal['entry']}")
        if signal['stop_loss']:
            print(f"   Stop Loss:   {signal['stop_loss']} ({signal['sl_pips']} pips)")
            print(f"   Take Profit: {signal['take_profit']} ({signal['tp_pips']} pips)")
            print(f"   R:R Ratio:   {signal['rr_ratio']}")

    # Regime
    regime_icon = "[OK]" if signal['regime_tradeable'] else "[X]"
    print(f"\n[REGIME]")
    print(f"   {regime_icon} {signal['regime']}")
    print(f"   {signal['regime_reason']}")

    # Session
    print(f"\n[SESSION]")
    sess_mult = signal['session_mult']
    sess_icon = "[OK]" if sess_mult >= 1.0 else "[--]"
    print(f"   {sess_icon} {signal['session']} (mult={sess_mult}x)")

    # Volatility
    print(f"\n[VOLATILITY]")
    vol_mult = signal['volatility_mult']
    vol_icon = "[OK]" if vol_mult >= 1.0 else "[--]"
    print(f"   {vol_icon} {signal['volatility_level']} ({signal['volatility']}%)")

    # Indicators
    print(f"\n[INDICATORS]")
    print(f"   ADX: {signal['adx']} | RSI: {signal['rsi']} | ATR: {signal['atr']}")
    print(f"   Stoch: %K={signal['stoch_k']} %D={signal['stoch_d']}")
    print(f"   Z-Score: {signal['zscore']} | Range Ratio: {signal['range_ratio']}")

    # Position sizing
    if signal['position_multiplier'] > 0:
        print(f"\n[POSITION SIZE]")
        print(f"   Multiplier: {signal['position_multiplier']*100:.0f}%")

    # Confluence
    print(f"\n[CONFLUENCE SCORE]: {signal['confluence_score']}/100")
    print(f"[BACKTEST PF]: {signal['backtest_pf']} | WR: {signal['backtest_wr']}%")

    print(f"\n{'='*65}")


if __name__ == "__main__":
    import json
    import sys

    scanner = EURJPYValidatedScanner()

    if '--json' in sys.argv:
        result = scanner.scan()
        print(json.dumps(result, indent=2, default=str))
    elif '--rules' in sys.argv:
        rules = scanner.get_rules_summary()
        print("\n" + "=" * 60)
        print("EUR/JPY TRADING RULES")
        print("=" * 60)

        print("\n[REGIME RULES]")
        for regime, rule in REGIME_RULES.items():
            icon = "[OK]" if rule['tradeable'] else "[X]"
            print(f"   {icon} {regime}: mult={rule['position_mult']}, "
                  f"use {rule.get('recommended_strategy', 'RANGE_BREAKOUT')}")

        print("\n[SESSION RULES]")
        for session, rule in SESSION_RULES.items():
            print(f"   {session}: mult={rule['mult']}")

        print("\n[VOLATILITY RULES]")
        for level, rule in VOLATILITY_RULES.items():
            print(f"   {level}: mult={rule['mult']}")

        print("\n[STRATEGIES]")
        for strat, desc in rules['strategies'].items():
            print(f"   {strat}: {desc}")

        print("\n[OPTIMAL PARAMS]")
        print(f"   R:R: {rules['optimal_params']['rr']}")
        print(f"   SL: {rules['optimal_params']['sl_mult']}x ATR")
        print(f"   Backtest PF: {rules['optimal_params']['backtest_pf']}")
    else:
        result = scanner.scan()
        print_signal(result)
