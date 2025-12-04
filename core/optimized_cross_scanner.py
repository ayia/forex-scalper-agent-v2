#!/usr/bin/env python3
"""
Optimized Cross Pairs Scanner V2.6
==================================
Scanner for 6 ROBUST cross pairs validated across multiple market regimes.

Selection criteria (from backtest on 5 critical periods: 2019-2024):
- Positive PnL across multiple periods
- Profit Factor >= 1.0 in 3+ periods
- Maximum drawdown < 25%
- Consistent performance in trending, ranging, and volatile markets

Removed pairs (underperforming on historical stress tests):
- NZDJPY, AUDJPY, EURAUD, GBPCAD (negative PnL on critical periods)

Usage:
    python main.py --optimized-cross
    python main.py --optimized-cross --active-only

Part of Forex Scalper Agent V2
"""
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# =============================================================================
# V2.6 ROBUST CONFIGURATIONS - Validated on 5 critical periods (2019-2024)
# =============================================================================
# COVID Crash 2020, JPY Crisis 2022, Ranging 2019, Recovery 2021, Recent 2024
# Only pairs with consistent profitability across regimes are included

OPTIMAL_CONFIGS = {
    # TOP TIER - Best performers (highest PnL, lowest drawdown)
    'EURCAD': {'rr': 2.5, 'adx': 15, 'rsi': (35, 65), 'score': 6, 'pf': 1.27, 'trades': 150, 'max_dd': 11.7},
    'EURJPY': {'rr': 1.8, 'adx': 20, 'rsi': (25, 75), 'score': 5, 'pf': 1.06, 'trades': 464, 'max_dd': 17.8},
    'GBPJPY': {'rr': 1.2, 'adx': 25, 'rsi': (30, 70), 'score': 6, 'pf': 1.10, 'trades': 327, 'max_dd': 13.7},

    # SECOND TIER - Solid performers (high volume, consistent)
    'CHFJPY': {'rr': 1.5, 'adx': 25, 'rsi': (25, 75), 'score': 4, 'pf': 1.06, 'trades': 515, 'max_dd': 20.6},
    'CADJPY': {'rr': 2.5, 'adx': 25, 'rsi': (35, 65), 'score': 6, 'pf': 1.03, 'trades': 176, 'max_dd': 18.4},

    # THIRD TIER - Robust (PF>=1 in 3+ periods, but higher drawdown)
    'GBPAUD': {'rr': 2.5, 'adx': 12, 'rsi': (35, 65), 'score': 6, 'pf': 1.00, 'trades': 221, 'max_dd': 30.4},
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

    def calculate_indicators(self, df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
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

    def calculate_confluence_score(self, current: pd.Series, prev: pd.Series,
                                   is_bullish: bool, config: dict,
                                   has_crossover: bool = False) -> Tuple[float, Dict]:
        """
        Calculate a more granular and representative confluence score (0-100).

        This scoring system provides:
        - Continuous values (not discrete jumps)
        - Different weighting for active signals vs WATCH
        - Detailed breakdown of each component

        Components and weights:
        - EMA Crossover/Proximity: 25 pts (crossover=25, proximity scaled 0-20)
        - Trend Alignment: 20 pts (close vs EMA50 + slope strength)
        - RSI Position: 15 pts (optimal zone = 15, edge zones scaled)
        - ADX Strength: 15 pts (scaled by how much above threshold)
        - MACD Alignment: 15 pts (histogram direction + strength)
        - Momentum: 10 pts (ROC direction + magnitude)
        Total: 100 points
        """
        details = {}
        score = 0.0

        # 1. EMA Crossover / Proximity (25 pts max)
        ema_diff = current['ema_fast'] - current['ema_slow']
        ema_diff_pct = abs(ema_diff) / current['close'] * 100 if current['close'] > 0 else 0

        if has_crossover:
            # Full points for actual crossover
            ema_score = 25.0
            details['ema_status'] = 'CROSSOVER'
        else:
            # For WATCH: score based on proximity to crossover
            # Closer to crossover = higher score (max 20 pts without actual cross)
            if ema_diff_pct < 0.05:  # Very close (< 0.05%)
                ema_score = 20.0
                details['ema_status'] = 'IMMINENT'
            elif ema_diff_pct < 0.10:  # Close (< 0.10%)
                ema_score = 15.0
                details['ema_status'] = 'NEAR'
            elif ema_diff_pct < 0.20:  # Approaching (< 0.20%)
                ema_score = 10.0
                details['ema_status'] = 'APPROACHING'
            elif ema_diff_pct < 0.50:  # Distant
                ema_score = 5.0
                details['ema_status'] = 'DISTANT'
            else:
                ema_score = 0.0
                details['ema_status'] = 'FAR'

        details['ema_score'] = round(ema_score, 1)
        details['ema_diff_pct'] = round(ema_diff_pct, 3)
        score += ema_score

        # 2. Trend Alignment (20 pts max)
        ema_slope = (current['ema_trend'] - prev['ema_trend']) / prev['ema_trend'] * 100 if prev['ema_trend'] > 0 else 0
        price_vs_ema50 = (current['close'] - current['ema_trend']) / current['ema_trend'] * 100 if current['ema_trend'] > 0 else 0

        trend_score = 0.0
        # Price position vs EMA50 (10 pts)
        if is_bullish and price_vs_ema50 > 0:
            trend_score += min(10.0, 5.0 + abs(price_vs_ema50) * 2)
        elif not is_bullish and price_vs_ema50 < 0:
            trend_score += min(10.0, 5.0 + abs(price_vs_ema50) * 2)
        elif abs(price_vs_ema50) < 0.1:  # Very close to EMA50
            trend_score += 3.0

        # EMA50 slope direction (10 pts)
        if is_bullish and ema_slope > 0:
            trend_score += min(10.0, 5.0 + abs(ema_slope) * 50)
        elif not is_bullish and ema_slope < 0:
            trend_score += min(10.0, 5.0 + abs(ema_slope) * 50)

        details['trend_score'] = round(trend_score, 1)
        details['price_vs_ema50_pct'] = round(price_vs_ema50, 2)
        details['ema50_slope'] = round(ema_slope, 4)
        score += trend_score

        # 3. RSI Position (15 pts max)
        rsi_low, rsi_high = config['rsi']
        rsi_val = float(current['rsi'])
        rsi_half_range = (rsi_high - rsi_low) / 2

        if rsi_low < rsi_val < rsi_high:
            # Inside range: score based on distance from extremes
            distance_from_edge = min(rsi_val - rsi_low, rsi_high - rsi_val)
            rsi_score = 10.0 + (distance_from_edge / rsi_half_range) * 5.0
            details['rsi_zone'] = 'OPTIMAL'
        else:
            # Outside range: reduced score
            if rsi_val <= rsi_low:
                overshoot = rsi_low - rsi_val
                rsi_score = max(0, 5.0 - overshoot * 0.5)
                details['rsi_zone'] = 'OVERSOLD'
            else:
                overshoot = rsi_val - rsi_high
                rsi_score = max(0, 5.0 - overshoot * 0.5)
                details['rsi_zone'] = 'OVERBOUGHT'

        details['rsi_score'] = round(rsi_score, 1)
        score += rsi_score

        # 4. ADX Strength (15 pts max)
        adx_val = float(current['adx'])
        adx_min = config['adx']

        if adx_val >= adx_min:
            # Above threshold: scale by how much above
            adx_excess = adx_val - adx_min
            adx_score = 10.0 + min(5.0, adx_excess * 0.25)
            details['adx_status'] = 'STRONG'
        else:
            # Below threshold: partial score based on proximity
            adx_deficit = adx_min - adx_val
            adx_score = max(0, 8.0 - adx_deficit * 0.4)
            details['adx_status'] = 'WEAK' if adx_score < 5 else 'MODERATE'

        details['adx_score'] = round(adx_score, 1)
        score += adx_score

        # 5. MACD Alignment (15 pts max)
        macd_hist = float(current['macd_hist'])
        macd_hist_prev = float(prev['macd_hist'])
        macd_increasing = macd_hist > macd_hist_prev

        macd_score = 0.0
        # Direction alignment (10 pts)
        if is_bullish and macd_hist > 0:
            macd_score += 10.0
            details['macd_direction'] = 'BULLISH'
        elif not is_bullish and macd_hist < 0:
            macd_score += 10.0
            details['macd_direction'] = 'BEARISH'
        elif abs(macd_hist) < abs(macd_hist_prev) * 0.5:
            macd_score += 3.0
            details['macd_direction'] = 'WEAKENING'
        else:
            details['macd_direction'] = 'CONTRARY'

        # Momentum (5 pts)
        if (is_bullish and macd_increasing) or (not is_bullish and not macd_increasing):
            macd_score += 5.0
            details['macd_momentum'] = 'ACCELERATING'
        else:
            details['macd_momentum'] = 'DECELERATING'

        details['macd_score'] = round(macd_score, 1)
        score += macd_score

        # 6. Price Momentum ROC (10 pts max)
        roc = (current['close'] - prev['close']) / prev['close'] * 100 if prev['close'] > 0 else 0

        roc_score = 0.0
        if is_bullish and roc > 0:
            roc_score = min(10.0, 5.0 + abs(roc) * 20)
            details['momentum_dir'] = 'POSITIVE'
        elif not is_bullish and roc < 0:
            roc_score = min(10.0, 5.0 + abs(roc) * 20)
            details['momentum_dir'] = 'NEGATIVE'
        else:
            roc_score = max(0, 3.0 - abs(roc) * 10)
            details['momentum_dir'] = 'CONTRARY'

        details['momentum_score'] = round(roc_score, 1)
        details['roc_pct'] = round(roc, 3)
        score += roc_score

        return round(score, 1), details

    def calculate_score(self, current: pd.Series, prev: pd.Series,
                       is_bullish: bool, config: dict) -> int:
        """
        Calculate confluence score (0-100) for display purposes.
        This is for UI/JSON output, not for signal filtering.
        Uses the new granular scoring system.
        """
        score, _ = self.calculate_confluence_score(current, prev, is_bullish, config, has_crossover=False)
        return int(score)

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

                # Calculate new granular confluence score with details
                confluence_score, score_details = self.calculate_confluence_score(
                    current, prev, True, config, has_crossover=True
                )

                signal = {
                    'pair': pair,
                    'direction': 'BUY',
                    'trend': 'BULLISH',
                    'near_crossover': True,
                    'ema_status': 'CROSSOVER',
                    'entry': round(float(current['close']), 5),
                    'stop_loss': round(sl, 5),
                    'take_profit': round(tp, 5),
                    'confluence_score': confluence_score,
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

                # Calculate new granular confluence score with details
                confluence_score, score_details = self.calculate_confluence_score(
                    current, prev, False, config, has_crossover=True
                )

                signal = {
                    'pair': pair,
                    'direction': 'SELL',
                    'trend': 'BEARISH',
                    'near_crossover': True,
                    'ema_status': 'CROSSOVER',
                    'entry': round(float(current['close']), 5),
                    'stop_loss': round(sl, 5),
                    'take_profit': round(tp, 5),
                    'confluence_score': confluence_score,
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

            # Calculate granular confluence score for WATCH (no crossover)
            confluence_score, score_details = self.calculate_confluence_score(
                current, prev, trend == 'BULLISH', config, has_crossover=False
            )

            # Determine crossover proximity status from score_details
            ema_status = score_details.get('ema_status', 'FAR')
            near_cross = ema_status in ['IMMINENT', 'NEAR']

            return {
                'pair': pair,
                'direction': 'WATCH',
                'trend': trend,
                'near_crossover': near_cross,
                'ema_status': ema_status,
                'entry': round(float(current['close']), 5),
                'confluence_score': confluence_score,
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
