#!/usr/bin/env python3
"""
EURCHF Mean Reversion Scanner with Regime Detection V1.0
=========================================================
Specialized scanner for EUR/CHF using Mean Reversion (Z-Score) strategy.
Includes automatic regime detection to filter trades.

Backtest-validated configuration (December 2024):
- Strategy: Mean Reversion (Z-Score based)
- R:R Ratio: 2.0
- ADX minimum: 20
- RSI range: 30-70
- SL: 1.5x ATR, TP: 3.0x ATR
- Win Rate: 50.8%, Profit Factor: 1.97

EUR/CHF Characteristics:
- Low volatility pair (Swiss franc safe-haven)
- Range-bound due to SNB interventions
- Ideal for mean reversion strategies

Regime Performance (2024 data):
TRADE: TRENDING_UP (PF=2.72), STRONG_TREND (PF=1.37), TRENDING_DOWN (PF=1.35), CONSOLIDATION (PF=1.28)
AVOID: HIGH_VOLATILITY, RANGING (insufficient data)

Monte Carlo Validation:
- 100% positive simulations
- Max DD 95th percentile: 1.6%
- Ruin probability: 0%

Usage:
    python main.py --pairs EURCHF
    python main.py --pairs EURCHF --active-only

Part of Forex Scalper Agent V3.3
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, Optional, Tuple
from enum import Enum

# Try to import multi-source fetcher
try:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.multi_source_fetcher import MultiSourceFetcher
    MULTI_SOURCE_AVAILABLE = True
except ImportError:
    MULTI_SOURCE_AVAILABLE = False


class MarketRegime(Enum):
    """Market regime classification."""
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    STRONG_TREND = "STRONG_TREND"
    RANGING = "RANGING"
    CONSOLIDATION = "CONSOLIDATION"
    NORMAL = "NORMAL"
    UNKNOWN = "UNKNOWN"


# =============================================================================
# REGIME TRADING RULES - Based on backtest results (December 2024)
# =============================================================================
REGIME_RULES = {
    # PROFITABLE REGIMES - Trade normally
    MarketRegime.TRENDING_UP: {
        'tradeable': True,
        'position_mult': 1.0,
        'reason': 'Best performer: PF=2.72, 13 trades'
    },
    MarketRegime.STRONG_TREND: {
        'tradeable': True,
        'position_mult': 1.0,
        'reason': 'Strong performer: PF=1.37, 24 trades'
    },
    MarketRegime.TRENDING_DOWN: {
        'tradeable': True,
        'position_mult': 1.0,
        'reason': 'Good performer: PF=1.35, 12 trades'
    },
    MarketRegime.CONSOLIDATION: {
        'tradeable': True,
        'position_mult': 1.0,
        'reason': 'Solid performer: PF=1.28, 23 trades'
    },
    MarketRegime.NORMAL: {
        'tradeable': True,
        'position_mult': 1.0,
        'reason': 'Normal market conditions'
    },

    # CAUTION REGIMES - Trade with reduced size
    MarketRegime.LOW_VOLATILITY: {
        'tradeable': True,
        'position_mult': 0.7,
        'reason': 'Reduced volatility - smaller moves'
    },

    # AVOID REGIMES - Do not trade
    MarketRegime.HIGH_VOLATILITY: {
        'tradeable': False,
        'position_mult': 0.0,
        'reason': 'SNB intervention risk - avoid trading'
    },
    MarketRegime.RANGING: {
        'tradeable': False,
        'position_mult': 0.0,
        'reason': 'Insufficient trend strength for mean reversion'
    },
    MarketRegime.UNKNOWN: {
        'tradeable': False,
        'position_mult': 0.0,
        'reason': 'Cannot determine regime'
    },
}

# =============================================================================
# OPTIMAL CONFIGURATION - From backtest optimization (December 2024)
# =============================================================================
EURCHF_CONFIG = {
    'pair': 'EURCHF',
    'strategy': 'MEAN_REVERSION_ZSCORE',

    # Z-Score parameters
    'zscore_period': 20,          # Lookback for mean/std calculation
    'zscore_buy_threshold': -2.0,  # Buy when Z-Score < -2
    'zscore_sell_threshold': 2.0,  # Sell when Z-Score > +2

    # Filters
    'adx_min': 20,                # ADX minimum for trend confirmation
    'rsi_low': 30,                # RSI lower bound filter
    'rsi_high': 70,               # RSI upper bound filter

    # Risk management
    'rr': 2.0,                    # Risk-Reward ratio
    'sl_mult': 1.5,               # Stop Loss = ATR * 1.5
    'atr_period': 14,             # ATR calculation period

    # Regime detection
    'volatility_lookback': 20,    # Lookback for volatility regime
    'trend_lookback': 50,         # Lookback for trend detection

    # Performance metrics from backtest
    'backtest_pf': 1.97,
    'backtest_wr': 50.8,
    'backtest_max_dd': 0.9,
    'backtest_trades': 65,
}


class EURCHFMeanReversionScanner:
    """
    Specialized scanner for EUR/CHF with Mean Reversion (Z-Score) strategy.
    Includes automatic regime detection to filter unfavorable market conditions.
    """

    def __init__(self):
        self.config = EURCHF_CONFIG
        self.pair = 'EURCHF'
        self.regime_rules = REGIME_RULES

    def fetch_data(self, period: str = "60d", interval: str = "1h") -> Optional[pd.DataFrame]:
        """Fetch EURCHF data from multiple sources."""
        try:
            # Try MultiSourceFetcher first (Twelve Data priority)
            if MULTI_SOURCE_AVAILABLE:
                try:
                    from datetime import timedelta
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=60)
                    fetcher = MultiSourceFetcher(verbose=False)
                    df = fetcher.fetch(
                        self.pair,
                        start_date.strftime('%Y-%m-%d'),
                        end_date.strftime('%Y-%m-%d'),
                        interval
                    )
                    if df is not None and len(df) > 0:
                        # Normalize column names
                        df.columns = [c.lower() for c in df.columns]
                        return df
                except Exception:
                    pass

            # Fallback to Yahoo Finance
            symbol = "EURCHF=X"
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
            except Exception:
                df = None

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
            print(f"Error fetching EURCHF data: {e}")
            return None

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all required indicators."""
        df = df.copy()

        close = df['close']
        high = df['high']
        low = df['low']

        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=self.config['atr_period']).mean()

        # Z-Score (Mean Reversion indicator)
        period = self.config['zscore_period']
        df['sma'] = close.rolling(window=period).mean()
        df['std'] = close.rolling(window=period).std()
        df['zscore'] = (close - df['sma']) / (df['std'] + 0.0001)

        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 0.0001)
        df['rsi'] = 100 - (100 / (1 + rs))

        # EMAs for trend detection
        df['ema_20'] = close.ewm(span=20, adjust=False).mean()
        df['ema_50'] = close.ewm(span=50, adjust=False).mean()
        df['ema_200'] = close.ewm(span=200, adjust=False).mean()

        # ADX for trend strength
        df['adx'], df['plus_di'], df['minus_di'] = self._calculate_adx(df)

        # Volatility metrics
        df['volatility'] = close.pct_change().rolling(window=20).std() * 100
        df['atr_pct'] = df['atr'] / close * 100

        # Bollinger Band Width for consolidation detection
        bb_mid = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        df['bb_width'] = (2 * bb_std / bb_mid) * 100

        return df

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX, +DI, -DI indicators."""
        high = df['high']
        low = df['low']
        close = df['close']

        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 0.0001))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 0.0001))

        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(window=period).mean()

        return adx.fillna(20), plus_di.fillna(0), minus_di.fillna(0)

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
        plus_di = current['plus_di']
        minus_di = current['minus_di']

        # Calculate trend slope
        price_change_20 = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20] * 100

        details = {
            'volatility_ratio': round(float(volatility_current / volatility_mean), 2) if volatility_mean > 0 else 1.0,
            'atr_pct': round(float(atr_pct_current), 4),
            'adx': round(float(adx), 1),
            'bb_width': round(float(bb_width), 3),
            'price_change_20': round(float(price_change_20), 2),
            'zscore': round(float(current['zscore']), 2),
            'ema_alignment': 'BULLISH' if ema_20 > ema_50 > ema_200 else ('BEARISH' if ema_20 < ema_50 < ema_200 else 'MIXED'),
        }

        # =================================================================
        # REGIME CLASSIFICATION LOGIC
        # =================================================================

        # 1. HIGH VOLATILITY - ATR/volatility significantly above average (SNB risk)
        if volatility_current > volatility_mean * 1.8 or atr_pct_current > atr_pct_mean * 1.8:
            details['classification_reason'] = f'High volatility {details["volatility_ratio"]}x above average - SNB risk'
            return MarketRegime.HIGH_VOLATILITY, details

        # 2. STRONG TREND - Very high ADX
        if adx > 40:
            details['classification_reason'] = f'Strong trend: ADX={adx:.1f}'
            return MarketRegime.STRONG_TREND, details

        # 3. TRENDING UP - Clear bullish alignment
        if adx > 25 and plus_di > minus_di and price_change_20 > 0.5:
            details['classification_reason'] = f'Uptrend: ADX={adx:.1f}, +DI>{minus_di:.0f}, +{price_change_20:.1f}%'
            return MarketRegime.TRENDING_UP, details

        # 4. TRENDING DOWN - Clear bearish alignment
        if adx > 25 and minus_di > plus_di and price_change_20 < -0.5:
            details['classification_reason'] = f'Downtrend: ADX={adx:.1f}, -DI>{plus_di:.0f}, {price_change_20:.1f}%'
            return MarketRegime.TRENDING_DOWN, details

        # 5. LOW VOLATILITY - Reduced movement
        if volatility_current < volatility_mean * 0.5 and bb_width < 0.8:
            details['classification_reason'] = f'Low volatility: {details["volatility_ratio"]}x, BB={bb_width:.2f}%'
            return MarketRegime.LOW_VOLATILITY, details

        # 6. CONSOLIDATION - Narrow BB width
        if bb_width < 1.2 and abs(price_change_20) < 0.3:
            details['classification_reason'] = f'Consolidation: BB={bb_width:.2f}%, flat price'
            return MarketRegime.CONSOLIDATION, details

        # 7. RANGING - Low ADX
        if adx < 15:
            details['classification_reason'] = f'Ranging: ADX={adx:.1f} (weak)'
            return MarketRegime.RANGING, details

        # 8. DEFAULT TO NORMAL
        details['classification_reason'] = 'Normal market conditions'
        return MarketRegime.NORMAL, details

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
        Detect Mean Reversion (Z-Score) signal with regime filtering.

        Signal Rules:
        - BUY: Z-Score < -2 (price 2 std below mean) + ADX >= 20 + RSI in range
        - SELL: Z-Score > +2 (price 2 std above mean) + ADX >= 20 + RSI in range
        """
        if len(df) < 60:
            return None

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # 1. Detect regime
        regime, regime_details = self.detect_regime(df)
        can_trade, regime_reason, position_mult = self.should_trade(regime)

        # 2. Get indicator values
        zscore = current['zscore']
        zscore_prev = prev['zscore']
        rsi = current['rsi']
        adx = current['adx']

        buy_threshold = self.config['zscore_buy_threshold']
        sell_threshold = self.config['zscore_sell_threshold']
        adx_min = self.config['adx_min']
        rsi_low = self.config['rsi_low']
        rsi_high = self.config['rsi_high']

        # 3. Check for signal conditions
        signal_type = None

        # Filters
        adx_ok = adx >= adx_min
        rsi_ok = rsi_low <= rsi <= rsi_high

        # BUY: Z-Score crosses above -2 from below (mean reversion from oversold)
        if zscore_prev < buy_threshold and zscore >= buy_threshold:
            if adx_ok and rsi_ok:
                signal_type = 'BUY'
        # Alternative: Currently at extreme oversold
        elif zscore < buy_threshold and adx_ok and rsi_ok:
            signal_type = 'BUY'

        # SELL: Z-Score crosses below +2 from above (mean reversion from overbought)
        if zscore_prev > sell_threshold and zscore <= sell_threshold:
            if adx_ok and rsi_ok:
                signal_type = 'SELL'
        # Alternative: Currently at extreme overbought
        elif zscore > sell_threshold and adx_ok and rsi_ok:
            signal_type = 'SELL'

        # 4. Calculate SL/TP
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

        # 5. Calculate confluence score
        confluence_score = 50  # Base score
        if signal_type:
            confluence_score += 25
            if abs(zscore) > 2.5:
                confluence_score += 10  # Strong deviation
            if adx > 25:
                confluence_score += 10  # Strong trend
            if can_trade:
                confluence_score += 5  # Favorable regime

        # 6. Build result
        result = {
            'pair': self.pair,
            'timestamp': datetime.now().isoformat(),

            # Signal info
            'direction': signal_type if signal_type else 'WATCH',
            'entry': round(float(entry_price), 5),
            'stop_loss': round(float(sl), 5) if sl else None,
            'take_profit': round(float(tp), 5) if tp else None,
            'confluence_score': confluence_score,

            # Z-Score values
            'zscore': round(float(zscore), 2),
            'zscore_prev': round(float(zscore_prev), 2),
            'zscore_status': 'OVERSOLD' if zscore < buy_threshold else ('OVERBOUGHT' if zscore > sell_threshold else 'NEUTRAL'),
            'mean_price': round(float(current['sma']), 5),

            # Filters status
            'rsi': round(float(rsi), 1),
            'adx': round(float(adx), 1),
            'filters_ok': bool(adx_ok and rsi_ok),

            # Regime info
            'regime': regime.value,
            'regime_details': regime_details,
            'regime_tradeable': bool(can_trade),
            'regime_reason': regime_reason,
            'position_multiplier': float(position_mult),

            # Other indicators
            'atr': round(float(atr), 5),
            'volatility': round(float(current['volatility']), 3),

            # Config
            'config': {
                'strategy': 'MEAN_REVERSION_ZSCORE',
                'rr': self.config['rr'],
                'sl_mult': self.config['sl_mult'],
                'zscore_buy': buy_threshold,
                'zscore_sell': sell_threshold,
                'adx_min': adx_min,
                'rsi_range': f'{rsi_low}-{rsi_high}',
                'backtest_pf': self.config['backtest_pf'],
                'backtest_wr': self.config['backtest_wr'],
            }
        }

        # 7. Apply regime filter
        if signal_type and not can_trade:
            result['direction'] = 'BLOCKED'
            result['block_reason'] = f"Regime {regime.value}: {regime_reason}"
            result['confluence_score'] = 25  # Reduce score for blocked signals

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
    print(f"EUR/CHF MEAN REVERSION SCANNER - {signal['timestamp'][:19]}")
    print(f"{'='*60}")

    # Direction with indicator
    direction = signal['direction']
    if direction == 'BUY':
        print(f"[+] SIGNAL: BUY (Mean Reversion from Oversold)")
    elif direction == 'SELL':
        print(f"[-] SIGNAL: SELL (Mean Reversion from Overbought)")
    elif direction == 'BLOCKED':
        print(f"[!] SIGNAL: BLOCKED - {signal.get('block_reason', 'Regime filter')}")
    else:
        print(f"[~] SIGNAL: WATCH (No entry)")

    # Prices
    print(f"\n[PRICES]")
    print(f"   Entry:       {signal['entry']}")
    print(f"   Mean Price:  {signal['mean_price']}")
    if signal['stop_loss']:
        print(f"   Stop Loss:   {signal['stop_loss']}")
        print(f"   Take Profit: {signal['take_profit']}")

    # Z-Score
    print(f"\n[Z-SCORE]")
    print(f"   Current: {signal['zscore']} | Previous: {signal['zscore_prev']}")
    print(f"   Status: {signal['zscore_status']}")
    print(f"   Thresholds: BUY < {signal['config']['zscore_buy']} | SELL > {signal['config']['zscore_sell']}")

    # Filters
    print(f"\n[FILTERS]")
    print(f"   RSI: {signal['rsi']} (range: {signal['config']['rsi_range']})")
    print(f"   ADX: {signal['adx']} (min: {signal['config']['adx_min']})")
    filters_status = "OK" if signal['filters_ok'] else "FAILED"
    print(f"   Status: [{filters_status}]")

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

    # Confluence
    print(f"\n[CONFLUENCE]")
    print(f"   Score: {signal['confluence_score']}/100")

    # Config
    cfg = signal['config']
    print(f"\n[CONFIG]")
    print(f"   Strategy: {cfg['strategy']}")
    print(f"   R:R={cfg['rr']} | SL={cfg['sl_mult']}xATR")
    print(f"   Backtest: PF={cfg['backtest_pf']} | WR={cfg['backtest_wr']}%")

    print(f"{'='*60}")


if __name__ == "__main__":
    import json
    import sys

    scanner = EURCHFMeanReversionScanner()

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
