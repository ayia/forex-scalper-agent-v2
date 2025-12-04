#!/usr/bin/env python3
"""
EUR/JPY MACD+Stochastic Scanner V1.0
====================================
Dedicated scanner for EUR/JPY with MACD+Stochastic strategy and regime filtering.

Strategy: MACD Crossover + Stochastic Confirmation
- BUY: MACD crossover haussier + Stochastic < 30
- SELL: MACD crossover baissier + Stochastic > 70

Regime Filters (based on backtest analysis):
- TRADE: LOW_VOLATILITY, RANGING, STRONG_TREND_UP, TREND_DOWN
- BLOCK: TREND_UP, STRONG_TREND_DOWN, LONDON session, Normal volatility

Performance: PF=1.27 -> PF=3.07 (with trend filter)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, Optional

# =============================================================================
# CONFIGURATION - Optimized from backtest analysis
# =============================================================================
EURJPY_CONFIG = {
    'pair': 'EURJPY',
    'symbol': 'EURJPY=X',
    'strategy': 'MACD_STOCHASTIC',

    # Signal parameters
    'rr': 1.2,
    'sl_mult': 1.5,
    'stoch_oversold': 30,
    'stoch_overbought': 70,

    # Filters
    'adx_min': 15,
    'rsi_low': 25,
    'rsi_high': 75,

    # Performance from backtest
    'backtest_pf': 1.27,
    'backtest_pf_with_trend': 3.07,
}

# =============================================================================
# REGIME RULES - Based on backtest analysis
# =============================================================================
REGIME_RULES = {
    # TRADEABLE regimes with position multiplier
    'LOW_VOLATILITY': {
        'tradeable': True,
        'position_mult': 1.5,  # Increase size - PF=6.70
        'reason': 'Excellent PF (6.70), 75% win rate'
    },
    'RANGING': {
        'tradeable': True,
        'position_mult': 1.5,  # Increase size - PF=4.41
        'reason': 'Excellent PF (4.41), 75% win rate'
    },
    'STRONG_TREND_UP': {
        'tradeable': True,
        'position_mult': 1.0,  # Normal size - PF=1.81
        'reason': 'Good PF (1.81), most trades'
    },
    'TREND_DOWN': {
        'tradeable': True,
        'position_mult': 0.75,  # Reduced size - PF=1.32
        'reason': 'Acceptable PF (1.32)'
    },
    'NORMAL': {
        'tradeable': True,
        'position_mult': 0.5,  # Reduced size - PF=1.02
        'reason': 'Marginal PF (1.02), proceed with caution'
    },
    'HIGH_VOLATILITY': {
        'tradeable': True,
        'position_mult': 0.5,  # Reduced - limited data
        'reason': 'Limited data, proceed with caution'
    },

    # BLOCKED regimes
    'TREND_UP': {
        'tradeable': False,
        'position_mult': 0,
        'reason': 'PF=0.00, 0% win rate - systematic losses'
    },
    'STRONG_TREND_DOWN': {
        'tradeable': False,
        'position_mult': 0,
        'reason': 'PF=0.73, negative expectancy'
    },
}

# Session rules
SESSION_RULES = {
    'ASIAN': {'tradeable': True, 'mult': 1.2, 'reason': 'Best session PF=1.69'},
    'NEW_YORK': {'tradeable': True, 'mult': 1.0, 'reason': 'Good session PF=1.21'},
    'LONDON': {'tradeable': False, 'mult': 0, 'reason': 'Poor session PF=0.90'},
}

# Volatility rules
VOLATILITY_RULES = {
    'VERY_LOW': {'tradeable': True, 'mult': 1.3, 'reason': 'Excellent PF=3.53'},
    'LOW': {'tradeable': True, 'mult': 1.0, 'reason': 'Good PF=1.27'},
    'NORMAL': {'tradeable': False, 'mult': 0, 'reason': 'Poor PF=0.66'},
    'HIGH': {'tradeable': True, 'mult': 1.2, 'reason': 'Excellent PF=2.78'},
    'VERY_HIGH': {'tradeable': True, 'mult': 0.5, 'reason': 'Limited data'},
}


class EURJPYMACDStochScanner:
    """Scanner for EUR/JPY with MACD+Stochastic strategy and regime filtering."""

    def __init__(self):
        self.config = EURJPY_CONFIG
        self.pair = self.config['pair']
        self.symbol = self.config['symbol']

    def fetch_data(self, period: str = "5d", interval: str = "1h") -> Optional[pd.DataFrame]:
        """Fetch market data from yfinance."""
        try:
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period=period, interval=interval)
            if df.empty:
                return None
            return df
        except Exception as e:
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators."""
        df = df.copy()

        # EMAs for trend
        df['ema8'] = df['Close'].ewm(span=8).mean()
        df['ema21'] = df['Close'].ewm(span=21).mean()
        df['ema50'] = df['Close'].ewm(span=50).mean()

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

        # Bollinger Bands width for volatility
        df['bb_mid'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + 2 * bb_std
        df['bb_lower'] = df['bb_mid'] - 2 * bb_std
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100

        # Volatility ratio
        df['atr_sma'] = df['atr'].rolling(20).mean()
        df['volatility_ratio'] = df['atr'] / df['atr_sma']
        df['bb_width_sma'] = df['bb_width'].rolling(20).mean()

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
        elif volatility_ratio < 0.7 or (bb_width < bb_width_sma * 0.5):
            regime = 'LOW_VOLATILITY'
            reason = f'ATR ratio={volatility_ratio:.2f} (<0.7)'
        elif adx > 40:
            if plus_di > minus_di:
                regime = 'STRONG_TREND_UP'
                reason = f'ADX={adx:.1f} (>40), +DI>{"-"}DI'
            else:
                regime = 'STRONG_TREND_DOWN'
                reason = f'ADX={adx:.1f} (>40), -DI>+DI'
        elif adx > 25:
            if plus_di > minus_di:
                regime = 'TREND_UP'
                reason = f'ADX={adx:.1f} (25-40), +DI>{"-"}DI'
            else:
                regime = 'TREND_DOWN'
                reason = f'ADX={adx:.1f} (25-40), -DI>+DI'
        elif adx < 20:
            regime = 'RANGING'
            reason = f'ADX={adx:.1f} (<20)'
        else:
            regime = 'NORMAL'
            reason = f'ADX={adx:.1f} (20-25)'

        rule = REGIME_RULES.get(regime, REGIME_RULES['NORMAL'])

        return {
            'regime': regime,
            'tradeable': rule['tradeable'],
            'position_mult': rule['position_mult'],
            'reason': rule['reason'],
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
        if volatility_ratio < 0.7:
            level = 'VERY_LOW'
        elif volatility_ratio < 0.9:
            level = 'LOW'
        elif volatility_ratio < 1.1:
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

    def check_trend_alignment(self, signal_direction: str, plus_di: float, minus_di: float) -> Dict:
        """Check if signal aligns with trend direction."""
        trend_bullish = plus_di > minus_di

        if signal_direction == 'BUY':
            aligned = trend_bullish
        elif signal_direction == 'SELL':
            aligned = not trend_bullish
        else:
            aligned = True  # No signal, no alignment needed

        return {
            'aligned': aligned,
            'trend': 'BULLISH' if trend_bullish else 'BEARISH',
            'signal': signal_direction,
            'recommendation': 'TRADE' if aligned else 'CAUTION - Against trend'
        }

    def check_signal(self, current, prev) -> Optional[str]:
        """Check for MACD+Stochastic signal."""
        # MACD crossover
        macd_cross_up = prev['macd'] <= prev['macd_signal'] and current['macd'] > current['macd_signal']
        macd_cross_down = prev['macd'] >= prev['macd_signal'] and current['macd'] < current['macd_signal']

        # Stochastic conditions
        stoch_oversold = current['stoch_k'] < self.config['stoch_oversold']
        stoch_overbought = current['stoch_k'] > self.config['stoch_overbought']

        # RSI filter
        rsi_ok = self.config['rsi_low'] <= current['rsi'] <= self.config['rsi_high']

        # ADX filter
        adx_ok = current['adx'] >= self.config['adx_min']

        if macd_cross_up and stoch_oversold and rsi_ok and adx_ok:
            return 'BUY'
        elif macd_cross_down and stoch_overbought and rsi_ok and adx_ok:
            return 'SELL'

        return None

    def scan(self) -> Dict:
        """Perform full scan with regime filtering."""
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

        # Check for signal
        raw_signal = self.check_signal(current, prev)

        # Determine stochastic status
        stoch_k = current['stoch_k']
        if stoch_k < 20:
            stoch_status = 'OVERSOLD'
        elif stoch_k < 30:
            stoch_status = 'NEAR_OVERSOLD'
        elif stoch_k > 80:
            stoch_status = 'OVERBOUGHT'
        elif stoch_k > 70:
            stoch_status = 'NEAR_OVERBOUGHT'
        else:
            stoch_status = 'NEUTRAL'

        # Check MACD status
        macd_above_signal = current['macd'] > current['macd_signal']
        macd_status = 'BULLISH' if macd_above_signal else 'BEARISH'

        # Check trend alignment if we have a signal
        trend_info = self.check_trend_alignment(
            raw_signal or 'NONE',
            current['plus_di'],
            current['minus_di']
        )

        # Determine final signal with all filters
        final_direction = 'WATCH'
        block_reasons = []
        position_multiplier = 1.0

        if raw_signal:
            # Check regime filter
            if not regime_info['tradeable']:
                block_reasons.append(f"Regime {regime_info['regime']}: {regime_info['reason']}")
            else:
                position_multiplier *= regime_info['position_mult']

            # Check session filter
            if not session_info['tradeable']:
                block_reasons.append(f"Session {session_info['session']}: {session_info['reason']}")
            else:
                position_multiplier *= session_info['mult']

            # Check volatility filter
            if not volatility_info['tradeable']:
                block_reasons.append(f"Volatility {volatility_info['level']}: {volatility_info['reason']}")
            else:
                position_multiplier *= volatility_info['mult']

            # Check trend alignment (warning only, not blocking)
            if not trend_info['aligned']:
                # Reduce position size for counter-trend trades
                position_multiplier *= 0.5

            # Final decision
            if block_reasons:
                final_direction = 'BLOCKED'
            else:
                final_direction = raw_signal

        # Calculate entry, SL, TP
        entry = round(current['Close'], 3)
        atr = current['atr']

        if final_direction == 'BUY':
            sl = round(entry - self.config['sl_mult'] * atr, 3)
            tp = round(entry + self.config['rr'] * self.config['sl_mult'] * atr, 3)
        elif final_direction == 'SELL':
            sl = round(entry + self.config['sl_mult'] * atr, 3)
            tp = round(entry - self.config['rr'] * self.config['sl_mult'] * atr, 3)
        else:
            sl = None
            tp = None

        # Calculate confluence score
        confluence = 50  # Base score
        if raw_signal:
            confluence += 10
        if regime_info['tradeable']:
            confluence += 15
        if session_info['tradeable']:
            confluence += 10
        if volatility_info['tradeable']:
            confluence += 10
        if trend_info['aligned']:
            confluence += 15
        if regime_info['position_mult'] > 1:
            confluence += 10

        return {
            'pair': self.pair,
            'timestamp': str(timestamp),
            'direction': final_direction,
            'raw_signal': raw_signal,

            # Prices
            'entry': entry,
            'stop_loss': sl,
            'take_profit': tp,

            # MACD
            'macd': round(current['macd'], 5),
            'macd_signal': round(current['macd_signal'], 5),
            'macd_status': macd_status,

            # Stochastic
            'stoch_k': round(stoch_k, 1),
            'stoch_d': round(current['stoch_d'], 1),
            'stoch_status': stoch_status,

            # Trend
            'plus_di': round(current['plus_di'], 1),
            'minus_di': round(current['minus_di'], 1),
            'trend_direction': trend_info['trend'],
            'trend_aligned': trend_info['aligned'],

            # Regime
            'regime': regime_info['regime'],
            'regime_tradeable': regime_info['tradeable'],
            'regime_reason': regime_info['reason'],
            'regime_details': regime_info,

            # Session
            'session': session_info['session'],
            'session_tradeable': session_info['tradeable'],
            'session_reason': session_info['reason'],

            # Volatility
            'volatility': round(current['volatility_ratio'] * 100, 1),
            'volatility_level': volatility_info['level'],
            'volatility_tradeable': volatility_info['tradeable'],

            # Other indicators
            'adx': round(current['adx'], 1),
            'rsi': round(current['rsi'], 1),
            'atr': round(atr, 3),

            # Position sizing
            'position_multiplier': round(position_multiplier, 2),

            # Blocking info
            'block_reasons': block_reasons,

            # Confluence
            'confluence_score': confluence,

            # Config
            'config': {
                'rr': self.config['rr'],
                'sl_mult': self.config['sl_mult'],
                'stoch_oversold': self.config['stoch_oversold'],
                'stoch_overbought': self.config['stoch_overbought'],
                'backtest_pf': self.config['backtest_pf'],
            }
        }

    def get_regime_summary(self) -> Dict:
        """Get summary of regime rules."""
        summary = {
            'tradeable_regimes': [],
            'blocked_regimes': [],
            'session_rules': [],
            'volatility_rules': [],
        }

        for regime, rule in REGIME_RULES.items():
            entry = {
                'regime': regime,
                'position_mult': rule['position_mult'],
                'reason': rule['reason']
            }
            if rule['tradeable']:
                summary['tradeable_regimes'].append(entry)
            else:
                summary['blocked_regimes'].append(entry)

        for session, rule in SESSION_RULES.items():
            summary['session_rules'].append({
                'session': session,
                'tradeable': rule['tradeable'],
                'mult': rule['mult'],
                'reason': rule['reason']
            })

        for level, rule in VOLATILITY_RULES.items():
            summary['volatility_rules'].append({
                'level': level,
                'tradeable': rule['tradeable'],
                'mult': rule['mult'],
                'reason': rule['reason']
            })

        return summary


def print_signal(signal: Dict):
    """Pretty print a signal."""
    if 'error' in signal:
        print(f"\n[X] {signal['pair']}: {signal['error']}")
        return

    print(f"\n{'='*65}")
    print(f"EUR/JPY MACD+STOCHASTIC SCANNER - {signal['timestamp'][:19]}")
    print(f"{'='*65}")

    # Direction
    direction = signal['direction']
    if direction == 'BUY':
        print(f"[+] SIGNAL: BUY")
    elif direction == 'SELL':
        print(f"[-] SIGNAL: SELL")
    elif direction == 'BLOCKED':
        print(f"[!] SIGNAL: BLOCKED")
        for reason in signal['block_reasons']:
            print(f"    -> {reason}")
    else:
        print(f"[~] SIGNAL: WATCH (No entry)")

    # Prices
    print(f"\n[PRICES]")
    print(f"   Entry:       {signal['entry']}")
    if signal['stop_loss']:
        print(f"   Stop Loss:   {signal['stop_loss']}")
        print(f"   Take Profit: {signal['take_profit']}")

    # MACD
    print(f"\n[MACD]")
    print(f"   MACD: {signal['macd']} | Signal: {signal['macd_signal']}")
    print(f"   Status: {signal['macd_status']}")

    # Stochastic
    print(f"\n[STOCHASTIC]")
    print(f"   %K: {signal['stoch_k']} | %D: {signal['stoch_d']}")
    print(f"   Status: {signal['stoch_status']}")

    # Trend
    trend_icon = "[OK]" if signal['trend_aligned'] else "[!]"
    print(f"\n[TREND]")
    print(f"   {trend_icon} Direction: {signal['trend_direction']}")
    print(f"   +DI: {signal['plus_di']} | -DI: {signal['minus_di']}")
    print(f"   Aligned with signal: {'Yes' if signal['trend_aligned'] else 'No (reduced size)'}")

    # Regime
    regime_icon = "[OK]" if signal['regime_tradeable'] else "[X]"
    print(f"\n[MARKET REGIME]")
    print(f"   {regime_icon} {signal['regime']}")
    print(f"   Reason: {signal['regime_reason']}")

    # Session
    session_icon = "[OK]" if signal['session_tradeable'] else "[X]"
    print(f"\n[SESSION]")
    print(f"   {session_icon} {signal['session']}")
    print(f"   Reason: {signal['session_reason']}")

    # Volatility
    vol_icon = "[OK]" if signal['volatility_tradeable'] else "[X]"
    print(f"\n[VOLATILITY]")
    print(f"   {vol_icon} {signal['volatility_level']} ({signal['volatility']}%)")

    # Position sizing
    if signal['position_multiplier'] > 0:
        print(f"\n[POSITION SIZE]")
        print(f"   Multiplier: {signal['position_multiplier']*100:.0f}%")

    # Other indicators
    print(f"\n[OTHER INDICATORS]")
    print(f"   ADX: {signal['adx']} | RSI: {signal['rsi']} | ATR: {signal['atr']}")

    # Confluence
    print(f"\n[CONFLUENCE SCORE]")
    print(f"   {signal['confluence_score']}/100")

    print(f"{'='*65}")


if __name__ == "__main__":
    import json
    import sys

    scanner = EURJPYMACDStochScanner()

    if '--json' in sys.argv:
        result = scanner.scan()
        print(json.dumps(result, indent=2))
    elif '--rules' in sys.argv:
        rules = scanner.get_regime_summary()
        print("\n[OK] TRADEABLE REGIMES:")
        for r in rules['tradeable_regimes']:
            print(f"   + {r['regime']}: {r['reason']} (size={r['position_mult']*100:.0f}%)")
        print("\n[X] BLOCKED REGIMES:")
        for r in rules['blocked_regimes']:
            print(f"   - {r['regime']}: {r['reason']}")
        print("\n[SESSION RULES]:")
        for s in rules['session_rules']:
            icon = "[OK]" if s['tradeable'] else "[X]"
            print(f"   {icon} {s['session']}: {s['reason']}")
        print("\n[VOLATILITY RULES]:")
        for v in rules['volatility_rules']:
            icon = "[OK]" if v['tradeable'] else "[X]"
            print(f"   {icon} {v['level']}: {v['reason']}")
    else:
        result = scanner.scan()
        print_signal(result)
