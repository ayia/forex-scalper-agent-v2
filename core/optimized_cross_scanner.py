#!/usr/bin/env python3
"""
Optimized Cross Pairs Scanner
=============================
Scanner for 10 profitable cross pairs with pair-specific optimized configurations.

Configurations derived from 2-year backtest optimization (180 configs per pair).
Only includes pairs with Profit Factor >= 1.0

Usage:
    python main.py --optimized-cross

Part of Forex Scalper Agent V2
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# =============================================================================
# OPTIMIZED CONFIGURATIONS FROM BACKTEST (2 years, 180 configs tested per pair)
# =============================================================================
# Only profitable pairs (PF >= 1.0) are included
# Format: {'rr': R:R ratio, 'adx': ADX threshold, 'rsi': (oversold, overbought), 'score': min score}

OPTIMAL_CONFIGS = {
    # Top performers (PF > 1.05)
    'NZDJPY': {'rr': 1.2, 'adx': 12, 'rsi': (35, 65), 'score': 6, 'pf': 1.11, 'trades': 657},
    'CADJPY': {'rr': 2.5, 'adx': 25, 'rsi': (35, 65), 'score': 6, 'pf': 1.10, 'trades': 370},
    'AUDJPY': {'rr': 1.2, 'adx': 20, 'rsi': (30, 70), 'score': 6, 'pf': 1.07, 'trades': 788},
    'GBPCAD': {'rr': 2.0, 'adx': 25, 'rsi': (35, 65), 'score': 6, 'pf': 1.05, 'trades': 452},
    'CHFJPY': {'rr': 1.5, 'adx': 25, 'rsi': (25, 75), 'score': 4, 'pf': 1.05, 'trades': 1080},

    # Good performers (PF 1.01-1.04)
    'EURJPY': {'rr': 1.8, 'adx': 20, 'rsi': (25, 75), 'score': 5, 'pf': 1.04, 'trades': 947},
    'EURCAD': {'rr': 2.5, 'adx': 15, 'rsi': (35, 65), 'score': 6, 'pf': 1.03, 'trades': 380},
    'GBPAUD': {'rr': 2.5, 'adx': 12, 'rsi': (35, 65), 'score': 6, 'pf': 1.02, 'trades': 443},
    'EURAUD': {'rr': 2.5, 'adx': 25, 'rsi': (35, 65), 'score': 4, 'pf': 1.01, 'trades': 572},
    'GBPJPY': {'rr': 1.2, 'adx': 25, 'rsi': (30, 70), 'score': 6, 'pf': 1.01, 'trades': 686},
}

# Pairs sorted by Profit Factor (best first)
PROFITABLE_PAIRS = list(OPTIMAL_CONFIGS.keys())


class OptimizedCrossScanner:
    """
    Scanner that uses pair-specific optimized configurations.
    """

    def __init__(self):
        self.pairs = PROFITABLE_PAIRS
        self.configs = OPTIMAL_CONFIGS

        # EMA periods (fixed across all pairs)
        self.ema_fast = 8
        self.ema_slow = 21
        self.ema_trend = 50

        # ATR for SL calculation
        self.atr_period = 14
        self.sl_mult = 1.5

    def fetch_data(self, pair: str, period: str = "10d") -> Optional[pd.DataFrame]:
        """Fetch hourly data for a pair."""
        try:
            symbol = f"{pair}=X"

            # Try yfinance Ticker first
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval="1h")
            except Exception:
                df = None

            # Fallback to download
            if df is None or df.empty:
                df = yf.download(symbol, period=period, interval="1h", progress=False)

            if df is not None and not df.empty:
                # Normalize column names
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = [c[0].lower() for c in df.columns]
                else:
                    df.columns = [c.lower() for c in df.columns]
                return df
            return None
        except Exception:
            return None

    def calculate_indicators(self, df: pd.DataFrame, config: dict) -> pd.DataFrame:
        """Calculate all indicators with pair-specific config."""
        df = df.copy()

        # EMAs
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        df['ema_trend'] = df['close'].ewm(span=self.ema_trend, adjust=False).mean()

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # ADX
        df['adx'] = self._calculate_adx(df)

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
        df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

        # ATR
        high = df['high']
        low = df['low']
        close = df['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=self.atr_period).mean()

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

    def calculate_score_backtest(self, current: pd.Series, prev: pd.Series,
                                  is_bullish: bool, config: dict) -> int:
        """
        Calculate score EXACTLY like the backtest optimizer (0-8 scale).
        This ensures consistency between backtest results and live scanner.

        Score components (same as optimize_all_pairs_v2.py):
        - EMA crossover: 2 pts (required for signal)
        - Trend alignment: 2 pts (close vs EMA50 + slope)
        - RSI in range: 1 pt
        - ADX OK: 1 pt
        - MACD direction: 1 pt
        - Momentum (ROC): 1 pt
        Total max: 8 points
        """
        score = 0

        # EMA crossover (2 pts) - already checked before calling this
        ema_cross = (prev['ema_fast'] <= prev['ema_slow'] and current['ema_fast'] > current['ema_slow']) if is_bullish else \
                    (prev['ema_fast'] >= prev['ema_slow'] and current['ema_fast'] < current['ema_slow'])
        if ema_cross:
            score += 2

        # Trend alignment (2 pts) - close vs EMA50 + EMA slope
        ema_slope = (current['ema_trend'] - prev['ema_trend']) / prev['ema_trend'] * 100 if prev['ema_trend'] > 0 else 0
        if is_bullish:
            if current['close'] > current['ema_trend'] and ema_slope > 0:
                score += 2
        else:
            if current['close'] < current['ema_trend'] and ema_slope < 0:
                score += 2

        # RSI in range (1 pt)
        rsi_low, rsi_high = config['rsi']
        if rsi_low < current['rsi'] < rsi_high:
            score += 1

        # ADX OK (1 pt)
        if current['adx'] >= config['adx']:
            score += 1

        # MACD direction (1 pt)
        if is_bullish and current['macd_hist'] > 0:
            score += 1
        elif not is_bullish and current['macd_hist'] < 0:
            score += 1

        # Momentum ROC (1 pt) - use 5-bar price change
        roc = (current['close'] - prev['close']) / prev['close'] * 100 if prev['close'] > 0 else 0
        if is_bullish and roc > 0:
            score += 1
        elif not is_bullish and roc < 0:
            score += 1

        return score

    def calculate_score(self, current: pd.Series, prev: pd.Series,
                       is_bullish: bool, config: dict) -> int:
        """
        Calculate confluence score (0-100) for display purposes.
        This is for UI/JSON output, not for signal filtering.
        """
        # Get backtest score (0-8) and convert to 0-100 scale
        backtest_score = self.calculate_score_backtest(current, prev, is_bullish, config)
        return int(backtest_score * 12.5)  # 8 * 12.5 = 100

    def detect_signal(self, df: pd.DataFrame, pair: str, config: dict) -> Optional[Dict]:
        """Detect trading signal based on optimized config for this pair."""
        if len(df) < 60:
            return None

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # Get config params
        rr = config['rr']
        adx_min = config['adx']
        rsi_low, rsi_high = config['rsi']
        min_score = config['score']

        # Check RSI filter - if outside range, show as FILTERED but still return data
        rsi_in_range = rsi_low < float(current['rsi']) < rsi_high
        rsi_status = "OK" if rsi_in_range else ("OVERBOUGHT" if current['rsi'] > rsi_high else "OVERSOLD")

        # Check ADX filter (with MACD fallback)
        adx_ok = current['adx'] >= adx_min
        macd_strong = abs(current['macd_hist']) > abs(prev['macd_hist'])
        momentum_ok = adx_ok or macd_strong

        # EMA crossover detection
        ema_cross_up = prev['ema_fast'] <= prev['ema_slow'] and current['ema_fast'] > current['ema_slow']
        ema_cross_down = prev['ema_fast'] >= prev['ema_slow'] and current['ema_fast'] < current['ema_slow']

        # Trend alignment
        trend_up = current['close'] > current['ema_trend']
        trend_down = current['close'] < current['ema_trend']

        signal = None

        # BUY signal (only if RSI in range and momentum OK)
        if ema_cross_up and trend_up and rsi_in_range and momentum_ok:
            backtest_score = self.calculate_score_backtest(current, prev, True, config)
            if backtest_score >= min_score:  # Score is 0-8, min_score is 4-6
                sl = float(current['close']) - (float(current['atr']) * self.sl_mult)
                tp = float(current['close']) + (float(current['atr']) * self.sl_mult * rr)

                score_display = int(backtest_score * 12.5)  # Convert to 0-100 for display
                signal = {
                    'pair': pair,
                    'direction': 'BUY',
                    'entry': round(float(current['close']), 5),
                    'stop_loss': round(sl, 5),
                    'take_profit': round(tp, 5),
                    'confluence_score': score_display,
                    'backtest_score': backtest_score,
                    'rsi': round(float(current['rsi']), 1),
                    'rsi_status': rsi_status,
                    'adx': round(float(current['adx']), 1),
                    'config': {
                        'rr': rr,
                        'adx_min': adx_min,
                        'rsi_range': f"{rsi_low}-{rsi_high}",
                        'min_score': min_score,
                        'backtest_pf': config['pf']
                    }
                }

        # SELL signal (only if RSI in range and momentum OK)
        elif ema_cross_down and trend_down and rsi_in_range and momentum_ok:
            backtest_score = self.calculate_score_backtest(current, prev, False, config)
            if backtest_score >= min_score:
                sl = float(current['close']) + (float(current['atr']) * self.sl_mult)
                tp = float(current['close']) - (float(current['atr']) * self.sl_mult * rr)

                score_display = int(backtest_score * 12.5)  # Convert to 0-100 for display
                signal = {
                    'pair': pair,
                    'direction': 'SELL',
                    'entry': round(float(current['close']), 5),
                    'stop_loss': round(sl, 5),
                    'take_profit': round(tp, 5),
                    'confluence_score': score_display,
                    'backtest_score': backtest_score,
                    'rsi': round(float(current['rsi']), 1),
                    'rsi_status': rsi_status,
                    'adx': round(float(current['adx']), 1),
                    'config': {
                        'rr': rr,
                        'adx_min': adx_min,
                        'rsi_range': f"{rsi_low}-{rsi_high}",
                        'min_score': min_score,
                        'backtest_pf': config['pf']
                    }
                }

        # Add market context even if no crossover or filters not met
        if signal is None:
            # Return market analysis without active signal
            trend = 'BULLISH' if current['ema_fast'] > current['ema_slow'] else 'BEARISH'
            near_cross = bool(abs(current['ema_fast'] - current['ema_slow']) / current['close'] < 0.001)

            return {
                'pair': pair,
                'direction': 'WATCH',
                'trend': trend,
                'near_crossover': near_cross,
                'entry': round(float(current['close']), 5),
                'confluence_score': int(self.calculate_score(current, prev, trend == 'BULLISH', config)),
                'rsi': round(float(current['rsi']), 1),
                'rsi_status': rsi_status,
                'adx': round(float(current['adx']), 1),
                'config': {
                    'rr': rr,
                    'adx_min': adx_min,
                    'rsi_range': f"{rsi_low}-{rsi_high}",
                    'min_score': min_score,
                    'backtest_pf': config['pf']
                }
            }

        return signal

    def scan_pair(self, pair: str) -> Optional[Dict]:
        """Scan a single pair with its optimal config."""
        config = self.configs.get(pair)
        if not config:
            return None

        df = self.fetch_data(pair)
        if df is None or len(df) < 60:
            return None

        df = self.calculate_indicators(df, config)
        signal = self.detect_signal(df, pair, config)

        if signal:
            signal['timestamp'] = datetime.now().isoformat()

        return signal

    def scan_all(self) -> List[Dict]:
        """Scan all profitable pairs and return signals sorted by confluence."""
        signals = []

        for pair in self.pairs:
            result = self.scan_pair(pair)
            if result:
                signals.append(result)

        # Sort by: 1) Active signals first (BUY/SELL), 2) Confluence score desc, 3) Backtest PF desc
        def sort_key(s):
            is_active = 0 if s['direction'] in ['BUY', 'SELL'] else 1
            pf = s['config']['backtest_pf']
            return (is_active, -s['confluence_score'], -pf)

        signals.sort(key=sort_key)

        return signals


if __name__ == "__main__":
    import json
    scanner = OptimizedCrossScanner()
    results = scanner.scan_all()
    print(json.dumps(results, indent=2))
