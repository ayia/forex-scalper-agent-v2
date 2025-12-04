#!/usr/bin/env python3
"""
EURGBP Stochastic Scanner with Regime Detection V1.0
=====================================================
Specialized scanner for EUR/GBP using Stochastic Crossover strategy.
Includes automatic regime detection to filter trades.

Backtest-validated configuration:
- Strategy: Stochastic %K/%D Crossover
- R:R Ratio: 2.0
- Oversold: 20, Overbought: 80, Zone Buffer: 10
- SL: 1.5x ATR, TP: 3.0x ATR
- Win Rate: 35.9%, Profit Factor: 1.10

Regime Performance (2016-2024):
✅ TRADE: HIGH_VOLATILITY, RATE_DIVERGENCE, TRENDING_DOWN, RANGING, RECOVERY, RECENT
❌ AVOID: TRENDING_UP, LOW_VOLATILITY, UNCERTAINTY

Usage:
    python main.py --eurgbp-stochastic

Part of Forex Scalper Agent V2
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum


class MarketRegime(Enum):
    """Market regime classification."""
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    RECOVERY = "RECOVERY"
    RATE_DIVERGENCE = "RATE_DIVERGENCE"
    UNCERTAINTY = "UNCERTAINTY"
    RECENT = "RECENT"
    UNKNOWN = "UNKNOWN"


# =============================================================================
# REGIME TRADING RULES - Based on backtest results
# =============================================================================
REGIME_RULES = {
    # ✅ PROFITABLE REGIMES - Trade normally
    MarketRegime.HIGH_VOLATILITY: {
        'tradeable': True,
        'position_mult': 1.0,
        'reason': 'Best performer: PF=1.50, +$1,722'
    },
    MarketRegime.RATE_DIVERGENCE: {
        'tradeable': True,
        'position_mult': 1.0,
        'reason': 'Strong performer: PF=1.43, +$1,321'
    },
    MarketRegime.RANGING: {
        'tradeable': True,
        'position_mult': 1.0,
        'reason': 'Good in range: PF=1.29, +$678'
    },
    MarketRegime.TRENDING_DOWN: {
        'tradeable': True,
        'position_mult': 0.8,  # Slightly reduced
        'reason': 'Decent in downtrend: PF=1.22, +$791'
    },
    MarketRegime.RECENT: {
        'tradeable': True,
        'position_mult': 1.0,
        'reason': 'Validated on recent data: PF=1.14, +$1,360'
    },
    MarketRegime.RECOVERY: {
        'tradeable': True,
        'position_mult': 0.8,
        'reason': 'Moderate performer: PF=1.09, +$313'
    },

    # ❌ UNPROFITABLE REGIMES - Do not trade
    MarketRegime.TRENDING_UP: {
        'tradeable': False,
        'position_mult': 0.0,
        'reason': 'Stochastic stays overbought: PF=0.85, -$927'
    },
    MarketRegime.LOW_VOLATILITY: {
        'tradeable': False,
        'position_mult': 0.0,
        'reason': 'Insufficient movement: PF=0.74, -$274'
    },
    MarketRegime.UNCERTAINTY: {
        'tradeable': False,
        'position_mult': 0.0,
        'reason': 'Erratic movements: PF=0.37, -$1,482'
    },
    MarketRegime.UNKNOWN: {
        'tradeable': False,
        'position_mult': 0.0,
        'reason': 'Cannot determine regime'
    },
}

# =============================================================================
# OPTIMAL CONFIGURATION - From backtest optimization
# =============================================================================
EURGBP_CONFIG = {
    'pair': 'EURGBP',
    'strategy': 'STOCHASTIC_CROSSOVER',
    'rr': 2.0,                    # Risk-Reward ratio
    'stoch_period': 14,           # Stochastic period
    'stoch_smooth': 3,            # Stochastic smoothing
    'oversold': 20,               # Oversold threshold
    'overbought': 80,             # Overbought threshold
    'zone_buffer': 10,            # Buffer for entries (K<30 for BUY, K>70 for SELL)
    'sl_mult': 1.5,               # Stop Loss = ATR * 1.5
    'atr_period': 14,             # ATR calculation period
    'volatility_lookback': 20,    # Lookback for volatility regime
    'trend_lookback': 50,         # Lookback for trend detection

    # Performance metrics from backtest
    'backtest_pf': 1.10,
    'backtest_wr': 35.9,
    'backtest_roi': 35.0,
}


class EURGBPStochasticScanner:
    """
    Specialized scanner for EUR/GBP with Stochastic Crossover strategy.
    Includes automatic regime detection to filter unfavorable market conditions.
    """

    def __init__(self):
        self.config = EURGBP_CONFIG
        self.pair = 'EURGBP'
        self.regime_rules = REGIME_RULES

    def fetch_data(self, period: str = "60d", interval: str = "1h") -> Optional[pd.DataFrame]:
        """Fetch EURGBP data from yfinance."""
        try:
            symbol = "EURGBP=X"

            # Try yfinance Ticker first
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
            except Exception:
                df = None

            # Fallback to download
            if df is None or df.empty:
                df = yf.download(symbol, period=period, interval=interval, progress=False)

            if df is not None and not df.empty:
                # Normalize column names
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]
                return df
            return None
        except Exception as e:
            print(f"Error fetching EURGBP data: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators."""
        df = df.copy()

        # ATR
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=self.config['atr_period']).mean()

        # Stochastic %K and %D
        period = self.config['stoch_period']
        smooth = self.config['stoch_smooth']

        low_min = df['low'].rolling(window=period).min()
        high_max = df['high'].rolling(window=period).max()

        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 0.0001)
        df['stoch_d'] = df['stoch_k'].rolling(window=smooth).mean()

        # EMAs for trend detection
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

        # Volatility metrics
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * 100
        df['atr_pct'] = df['atr'] / df['close'] * 100

        # ADX for trend strength
        df['adx'] = self._calculate_adx(df)

        # Bollinger Band Width for range detection
        bb_mid = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_width'] = (2 * bb_std / bb_mid) * 100

        return df

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX indicator."""
        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff()
        minus_dm = low.diff().abs() * -1

        plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
        minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(window=period).mean()

        return adx

    def detect_regime(self, df: pd.DataFrame) -> Tuple[MarketRegime, Dict]:
        """
        Detect current market regime based on volatility, trend, and range metrics.

        Returns:
            Tuple of (MarketRegime, details_dict)
        """
        if len(df) < 60:
            return MarketRegime.UNKNOWN, {'reason': 'Insufficient data'}

        current = df.iloc[-1]

        # Get historical metrics for comparison
        volatility_mean = df['volatility'].iloc[-60:].mean()
        volatility_current = df['volatility'].iloc[-5:].mean()
        atr_pct_mean = df['atr_pct'].iloc[-60:].mean()
        atr_pct_current = current['atr_pct']

        # Trend detection
        ema_20 = current['ema_20']
        ema_50 = current['ema_50']
        ema_200 = current['ema_200'] if not pd.isna(current['ema_200']) else ema_50
        price = current['close']
        adx = current['adx']
        bb_width = current['bb_width']

        # Calculate trend slope (price change over last 20 bars)
        price_change_20 = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20] * 100

        details = {
            'volatility_ratio': round(volatility_current / volatility_mean, 2) if volatility_mean > 0 else 1.0,
            'atr_pct': round(atr_pct_current, 3),
            'adx': round(adx, 1),
            'bb_width': round(bb_width, 3),
            'price_change_20': round(price_change_20, 2),
            'ema_alignment': 'BULLISH' if ema_20 > ema_50 > ema_200 else ('BEARISH' if ema_20 < ema_50 < ema_200 else 'MIXED'),
        }

        # =================================================================
        # REGIME CLASSIFICATION LOGIC
        # =================================================================

        # 1. HIGH VOLATILITY - ATR/volatility significantly above average
        if volatility_current > volatility_mean * 1.5 or atr_pct_current > atr_pct_mean * 1.5:
            details['classification_reason'] = f'Volatility {details["volatility_ratio"]}x above average'
            return MarketRegime.HIGH_VOLATILITY, details

        # 2. LOW VOLATILITY - ATR/volatility significantly below average
        if volatility_current < volatility_mean * 0.5 and bb_width < 1.0:
            details['classification_reason'] = f'Volatility {details["volatility_ratio"]}x (low), BB Width {bb_width:.2f}% (narrow)'
            return MarketRegime.LOW_VOLATILITY, details

        # 3. STRONG UPTREND - Clear bullish alignment + strong ADX
        if ema_20 > ema_50 > ema_200 and adx > 25 and price_change_20 > 1.0:
            details['classification_reason'] = f'Strong uptrend: EMAs aligned, ADX={adx:.1f}, +{price_change_20:.1f}%'
            return MarketRegime.TRENDING_UP, details

        # 4. STRONG DOWNTREND - Clear bearish alignment + strong ADX
        if ema_20 < ema_50 < ema_200 and adx > 25 and price_change_20 < -1.0:
            details['classification_reason'] = f'Strong downtrend: EMAs aligned, ADX={adx:.1f}, {price_change_20:.1f}%'
            return MarketRegime.TRENDING_DOWN, details

        # 5. RANGING MARKET - Low ADX + narrow BB width + price between EMAs
        if adx < 20 and bb_width < 2.0 and abs(price_change_20) < 0.5:
            details['classification_reason'] = f'Range-bound: ADX={adx:.1f} (weak), BB={bb_width:.2f}% (narrow)'
            return MarketRegime.RANGING, details

        # 6. UNCERTAINTY - Mixed signals, EMAs crossed, high dispersion
        if details['ema_alignment'] == 'MIXED' and 20 < adx < 30:
            details['classification_reason'] = f'Mixed EMAs with moderate ADX={adx:.1f}'
            return MarketRegime.UNCERTAINTY, details

        # 7. RECOVERY - Coming out of downtrend, EMA20 crossing EMA50
        prev = df.iloc[-5]
        if prev['ema_20'] < prev['ema_50'] and ema_20 > ema_50 and price_change_20 > 0:
            details['classification_reason'] = 'EMA20 crossed above EMA50, recovery pattern'
            return MarketRegime.RECOVERY, details

        # 8. DEFAULT TO RECENT (normal conditions)
        details['classification_reason'] = 'Normal market conditions'
        return MarketRegime.RECENT, details

    def should_trade(self, regime: MarketRegime) -> Tuple[bool, str, float]:
        """
        Determine if we should trade based on current regime.

        Returns:
            Tuple of (should_trade, reason, position_multiplier)
        """
        rule = self.regime_rules.get(regime, self.regime_rules[MarketRegime.UNKNOWN])
        return rule['tradeable'], rule['reason'], rule['position_mult']

    def detect_signal(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Detect Stochastic crossover signal with regime filtering.

        Signal Rules:
        - BUY: %K crosses above %D, %K < 30 (oversold + buffer)
        - SELL: %K crosses below %D, %K > 70 (overbought - buffer)
        """
        if len(df) < 60:
            return None

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # 1. Detect regime
        regime, regime_details = self.detect_regime(df)
        can_trade, regime_reason, position_mult = self.should_trade(regime)

        # 2. Get Stochastic values
        k_current = current['stoch_k']
        d_current = current['stoch_d']
        k_prev = prev['stoch_k']
        d_prev = prev['stoch_d']

        oversold = self.config['oversold']
        overbought = self.config['overbought']
        buffer = self.config['zone_buffer']

        # Entry zones with buffer
        buy_zone = oversold + buffer      # K < 30 for BUY
        sell_zone = overbought - buffer   # K > 70 for SELL

        # 3. Check for crossover
        k_crossed_up = k_prev <= d_prev and k_current > d_current
        k_crossed_down = k_prev >= d_prev and k_current < d_current

        # 4. Determine signal type
        signal_type = None
        if k_crossed_up and k_current < buy_zone:
            signal_type = 'BUY'
        elif k_crossed_down and k_current > sell_zone:
            signal_type = 'SELL'

        # 5. Calculate SL/TP
        atr = current['atr']
        entry_price = current['close']
        rr = self.config['rr']
        sl_mult = self.config['sl_mult']

        if signal_type == 'BUY':
            sl = entry_price - (atr * sl_mult)
            tp = entry_price + (atr * sl_mult * rr)
        elif signal_type == 'SELL':
            sl = entry_price + (atr * sl_mult)
            tp = entry_price - (atr * sl_mult * rr)
        else:
            sl = tp = None

        # 6. Build result
        result = {
            'pair': self.pair,
            'timestamp': datetime.now().isoformat(),

            # Signal info
            'direction': signal_type if signal_type else 'WATCH',
            'entry': round(entry_price, 5),
            'stop_loss': round(sl, 5) if sl else None,
            'take_profit': round(tp, 5) if tp else None,

            # Stochastic values
            'stoch_k': round(k_current, 1),
            'stoch_d': round(d_current, 1),
            'stoch_status': 'OVERSOLD' if k_current < oversold else ('OVERBOUGHT' if k_current > overbought else 'NEUTRAL'),
            'crossover': 'UP' if k_crossed_up else ('DOWN' if k_crossed_down else 'NONE'),

            # Regime info
            'regime': regime.value,
            'regime_details': regime_details,
            'regime_tradeable': can_trade,
            'regime_reason': regime_reason,
            'position_multiplier': position_mult,

            # Other indicators
            'atr': round(atr, 5),
            'adx': round(current['adx'], 1),
            'volatility': round(current['volatility'], 3),

            # Config
            'config': {
                'rr': self.config['rr'],
                'sl_mult': self.config['sl_mult'],
                'oversold': oversold,
                'overbought': overbought,
                'zone_buffer': buffer,
                'backtest_pf': self.config['backtest_pf'],
                'backtest_wr': self.config['backtest_wr'],
            }
        }

        # 7. Apply regime filter
        if signal_type and not can_trade:
            result['direction'] = 'BLOCKED'
            result['block_reason'] = f"Regime {regime.value}: {regime_reason}"

        return result

    def scan(self) -> Optional[Dict]:
        """
        Main scan function - fetches data and returns current signal status.
        """
        df = self.fetch_data()
        if df is None or len(df) < 60:
            return {
                'pair': self.pair,
                'error': 'Insufficient data',
                'timestamp': datetime.now().isoformat()
            }

        df = self.calculate_indicators(df)
        return self.detect_signal(df)

    def get_regime_summary(self) -> Dict:
        """Get a summary of regime trading rules."""
        summary = {
            'tradeable_regimes': [],
            'blocked_regimes': [],
        }

        for regime, rule in self.regime_rules.items():
            entry = {
                'regime': regime.value,
                'reason': rule['reason'],
                'position_mult': rule['position_mult']
            }
            if rule['tradeable']:
                summary['tradeable_regimes'].append(entry)
            else:
                summary['blocked_regimes'].append(entry)

        return summary


def print_signal(signal: Dict):
    """Pretty print a signal."""
    if 'error' in signal:
        print(f"\n[X] {signal['pair']}: {signal['error']}")
        return

    print(f"\n{'='*60}")
    print(f"EUR/GBP STOCHASTIC SCANNER - {signal['timestamp'][:19]}")
    print(f"{'='*60}")

    # Direction with text indicator
    direction = signal['direction']
    if direction == 'BUY':
        print(f"[+] SIGNAL: BUY")
    elif direction == 'SELL':
        print(f"[-] SIGNAL: SELL")
    elif direction == 'BLOCKED':
        print(f"[!] SIGNAL: BLOCKED - {signal.get('block_reason', 'Regime filter')}")
    else:
        print(f"[~] SIGNAL: WATCH (No entry)")

    # Prices
    print(f"\n[PRICES]")
    print(f"   Entry:       {signal['entry']}")
    if signal['stop_loss']:
        print(f"   Stop Loss:   {signal['stop_loss']}")
        print(f"   Take Profit: {signal['take_profit']}")

    # Stochastic
    print(f"\n[STOCHASTIC]")
    print(f"   %K: {signal['stoch_k']} | %D: {signal['stoch_d']}")
    print(f"   Status: {signal['stoch_status']} | Crossover: {signal['crossover']}")

    # Regime
    regime_icon = "[OK]" if signal['regime_tradeable'] else "[X]"
    print(f"\n[MARKET REGIME]")
    print(f"   {regime_icon} {signal['regime']}")
    print(f"   Reason: {signal['regime_reason']}")
    if 'classification_reason' in signal['regime_details']:
        print(f"   Detection: {signal['regime_details']['classification_reason']}")

    # Position sizing
    if signal['position_multiplier'] > 0:
        print(f"   Position Size: {signal['position_multiplier']*100:.0f}%")

    # Other indicators
    print(f"\n[INDICATORS]")
    print(f"   ATR: {signal['atr']} | ADX: {signal['adx']} | Volatility: {signal['volatility']}%")

    # Config
    cfg = signal['config']
    print(f"\n[CONFIG]")
    print(f"   R:R={cfg['rr']} | SL={cfg['sl_mult']}xATR | Backtest PF={cfg['backtest_pf']}")
    print(f"   Oversold<{cfg['oversold']} | Overbought>{cfg['overbought']} | Buffer={cfg['zone_buffer']}")

    print(f"{'='*60}")


if __name__ == "__main__":
    import json
    import sys

    scanner = EURGBPStochasticScanner()

    # Check for --json flag
    if '--json' in sys.argv:
        result = scanner.scan()
        print(json.dumps(result, indent=2))
    elif '--rules' in sys.argv:
        rules = scanner.get_regime_summary()
        print("\n[OK] TRADEABLE REGIMES (Quand Trader):")
        for r in rules['tradeable_regimes']:
            print(f"   + {r['regime']}: {r['reason']} (size={r['position_mult']*100:.0f}%)")
        print("\n[X] BLOCKED REGIMES (Quand Eviter):")
        for r in rules['blocked_regimes']:
            print(f"   - {r['regime']}: {r['reason']}")
    else:
        result = scanner.scan()
        print_signal(result)
