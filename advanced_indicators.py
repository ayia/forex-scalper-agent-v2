"""
Advanced Indicators Module - Inspired by DIY Custom Strategy Builder [ZP]
Implements sophisticated indicators for scalping with adaptive parameters.
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime, time
import math


@dataclass
class IndicatorResult:
    """Result from an indicator calculation."""
    signal: str  # 'LONG', 'SHORT', 'NEUTRAL'
    strength: float  # 0-100
    value: float
    metadata: Dict = None


class RangeFilter:
    """
    Range Filter - Adaptive noise filtering indicator.
    Two types: Default (standard) and DW (Donchian Width based).
    """

    def __init__(self, period: int = 100, multiplier: float = 3.0, filter_type: str = 'DW'):
        self.period = period
        self.multiplier = multiplier
        self.filter_type = filter_type  # 'Default' or 'DW'

    def calculate(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate Range Filter signal."""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        if self.filter_type == 'DW':
            return self._calculate_dw(close, high, low)
        else:
            return self._calculate_default(close, high, low)

    def _calculate_default(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> IndicatorResult:
        """Standard Range Filter calculation."""
        # Calculate smoothed range
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
        tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
        tr[0] = high[0] - low[0]

        # EMA of true range
        smooth_range = self._ema(tr, self.period) * self.multiplier

        # Calculate filter
        filt = np.zeros(len(close))
        filt[0] = close[0]

        upward = np.zeros(len(close))
        downward = np.zeros(len(close))

        for i in range(1, len(close)):
            if close[i] > filt[i-1]:
                filt[i] = max(filt[i-1], close[i] - smooth_range[i])
            elif close[i] < filt[i-1]:
                filt[i] = min(filt[i-1], close[i] + smooth_range[i])
            else:
                filt[i] = filt[i-1]

            if filt[i] > filt[i-1]:
                upward[i] = upward[i-1] + 1
                downward[i] = 0
            elif filt[i] < filt[i-1]:
                downward[i] = downward[i-1] + 1
                upward[i] = 0
            else:
                upward[i] = upward[i-1]
                downward[i] = downward[i-1]

        # Generate signal
        is_long = close[-1] > filt[-1] and upward[-1] > 0
        is_short = close[-1] < filt[-1] and downward[-1] > 0

        if is_long:
            strength = min(100, upward[-1] * 10)
            return IndicatorResult('LONG', strength, filt[-1], {'filter': filt[-1], 'upward': upward[-1]})
        elif is_short:
            strength = min(100, downward[-1] * 10)
            return IndicatorResult('SHORT', strength, filt[-1], {'filter': filt[-1], 'downward': downward[-1]})

        return IndicatorResult('NEUTRAL', 0, filt[-1], {'filter': filt[-1]})

    def _calculate_dw(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> IndicatorResult:
        """DW (Donchian Width) Range Filter - better for scalping."""
        # Conditional sampling
        src = close

        # Calculate range size based on Donchian Channel width
        hh = pd.Series(high).rolling(self.period).max().values
        ll = pd.Series(low).rolling(self.period).min().values
        rng_size = (hh - ll) / self.period * self.multiplier

        # Apply Range Filter Type 2 logic
        filt = np.zeros(len(close))
        filt[0] = close[0]

        for i in range(1, len(close)):
            if np.isnan(rng_size[i]):
                filt[i] = filt[i-1]
                continue

            if src[i] > filt[i-1]:
                filt[i] = max(filt[i-1], src[i] - rng_size[i])
            elif src[i] < filt[i-1]:
                filt[i] = min(filt[i-1], src[i] + rng_size[i])
            else:
                filt[i] = filt[i-1]

        # Trend detection
        is_upward = filt[-1] > filt[-2] if len(filt) > 1 else False
        is_downward = filt[-1] < filt[-2] if len(filt) > 1 else False

        if is_upward and close[-1] > filt[-1]:
            return IndicatorResult('LONG', 75, filt[-1], {'trend': 'up'})
        elif is_downward and close[-1] < filt[-1]:
            return IndicatorResult('SHORT', 75, filt[-1], {'trend': 'down'})

        return IndicatorResult('NEUTRAL', 0, filt[-1])

    def _ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA."""
        return pd.Series(data).ewm(span=period, adjust=False).mean().values


class ChandelierExit:
    """
    Chandelier Exit - ATR-based trailing stop indicator.
    Excellent for scalping SL placement.
    """

    def __init__(self, atr_period: int = 22, multiplier: float = 3.0, use_close: bool = True):
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.use_close = use_close

    def calculate(self, df: pd.DataFrame) -> Tuple[IndicatorResult, float, float]:
        """
        Calculate Chandelier Exit.
        Returns: (signal, long_stop, short_stop)
        """
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # Calculate ATR
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
        tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).rolling(self.atr_period).mean().values * self.multiplier

        # Calculate stops
        if self.use_close:
            highest = pd.Series(close).rolling(self.atr_period).max().values
            lowest = pd.Series(close).rolling(self.atr_period).min().values
        else:
            highest = pd.Series(high).rolling(self.atr_period).max().values
            lowest = pd.Series(low).rolling(self.atr_period).min().values

        long_stop = np.zeros(len(close))
        short_stop = np.zeros(len(close))
        direction = np.zeros(len(close))

        for i in range(self.atr_period, len(close)):
            # Long stop (trailing stop for longs)
            new_long_stop = highest[i] - atr[i]
            if close[i-1] > long_stop[i-1]:
                long_stop[i] = max(long_stop[i-1], new_long_stop)
            else:
                long_stop[i] = new_long_stop

            # Short stop (trailing stop for shorts)
            new_short_stop = lowest[i] + atr[i]
            if close[i-1] < short_stop[i-1]:
                short_stop[i] = min(short_stop[i-1], new_short_stop)
            else:
                short_stop[i] = new_short_stop

            # Direction
            if close[i] > short_stop[i-1]:
                direction[i] = 1
            elif close[i] < long_stop[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]

        current_dir = direction[-1]
        if current_dir == 1:
            return (IndicatorResult('LONG', 70, long_stop[-1]), long_stop[-1], short_stop[-1])
        elif current_dir == -1:
            return (IndicatorResult('SHORT', 70, short_stop[-1]), long_stop[-1], short_stop[-1])

        return (IndicatorResult('NEUTRAL', 0, close[-1]), long_stop[-1], short_stop[-1])


class WaddahAttarExplosion:
    """
    Waddah Attar Explosion (WAE) - Momentum/Volatility indicator.
    Excellent for detecting explosive moves in scalping.
    """

    def __init__(self, sensitivity: int = 150, fast_length: int = 20,
                 slow_length: int = 40, channel_length: int = 20, bb_mult: float = 2.0):
        self.sensitivity = sensitivity
        self.fast_length = fast_length
        self.slow_length = slow_length
        self.channel_length = channel_length
        self.bb_mult = bb_mult

    def calculate(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate WAE signal."""
        close = df['close'].values

        # Calculate dead zone (based on ATR)
        tr = np.maximum(
            df['high'].values - df['low'].values,
            np.abs(df['high'].values - np.roll(close, 1))
        )
        tr = np.maximum(tr, np.abs(df['low'].values - np.roll(close, 1)))
        deadzone = pd.Series(tr).rolling(100).mean().values[-1] * 3.7

        # MACD calculation
        fast_ma = pd.Series(close).ewm(span=self.fast_length, adjust=False).mean().values
        slow_ma = pd.Series(close).ewm(span=self.slow_length, adjust=False).mean().values
        macd = fast_ma - slow_ma
        macd_prev = np.roll(macd, 1)

        # MACD change * sensitivity
        t1 = (macd - macd_prev) * self.sensitivity

        # Bollinger Band width (explosion line)
        basis = pd.Series(close).rolling(self.channel_length).mean().values
        std = pd.Series(close).rolling(self.channel_length).std().values
        bb_upper = basis + self.bb_mult * std
        bb_lower = basis - self.bb_mult * std
        e1 = bb_upper - bb_lower

        # Current values
        trend_up = max(0, t1[-1])
        trend_down = max(0, -t1[-1])
        explosion = e1[-1]

        # Signal conditions
        is_long = trend_up > 0 and trend_up > explosion and explosion > deadzone and trend_up > deadzone
        is_short = trend_down > 0 and trend_down > explosion and explosion > deadzone and trend_down > deadzone

        if is_long:
            strength = min(100, (trend_up / deadzone) * 30)
            return IndicatorResult('LONG', strength, trend_up, {
                'trend_up': trend_up, 'explosion': explosion, 'deadzone': deadzone
            })
        elif is_short:
            strength = min(100, (trend_down / deadzone) * 30)
            return IndicatorResult('SHORT', strength, trend_down, {
                'trend_down': trend_down, 'explosion': explosion, 'deadzone': deadzone
            })

        return IndicatorResult('NEUTRAL', 0, 0, {'deadzone': deadzone})


class ChoppinessIndex:
    """
    Choppiness Index - Detects ranging vs trending markets.
    Critical for scalping: avoid choppy markets.
    """

    def __init__(self, length: int = 14, threshold: float = 61.8):
        self.length = length
        self.threshold = threshold  # Above this = choppy, below = trending

    def calculate(self, df: pd.DataFrame) -> Tuple[float, bool]:
        """
        Calculate Choppiness Index.
        Returns: (ci_value, is_trending)
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # ATR sum
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
        tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
        tr[0] = high[0] - low[0]

        atr_sum = pd.Series(tr).rolling(self.length).sum().values

        # Highest high - Lowest low
        hh = pd.Series(high).rolling(self.length).max().values
        ll = pd.Series(low).rolling(self.length).min().values

        # Choppiness Index formula
        with np.errstate(divide='ignore', invalid='ignore'):
            ci = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(self.length)

        ci_value = ci[-1] if not np.isnan(ci[-1]) else 50
        is_trending = ci_value < self.threshold

        return ci_value, is_trending


class PVSRA:
    """
    Price Volume Spread Analysis (PVSRA).
    Identifies significant volume candles for smart money detection.
    """

    def __init__(self, volume_period: int = 10):
        self.volume_period = volume_period

    def calculate(self, df: pd.DataFrame) -> Dict:
        """
        Analyze PVSRA signals.
        Returns classification of current candle.
        """
        if 'volume' not in df.columns or df['volume'].sum() == 0:
            return {'type': 'NORMAL', 'is_climax': False, 'is_rising': False}

        close = df['close'].values
        open_price = df['open'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values

        # Average volume
        avg_vol = pd.Series(volume).rolling(self.volume_period).mean().values

        # Value calculation (volume * spread)
        value = volume * (high - low)
        highest_value = pd.Series(value).rolling(self.volume_period).max().values

        # Current candle analysis
        is_bull = close[-1] > open_price[-1]

        # Volume classification
        va = 0
        if volume[-1] >= avg_vol[-1] * 2 or value[-1] >= highest_value[-1]:
            va = 1  # Climax volume (200%+)
        elif volume[-1] >= avg_vol[-1] * 1.5:
            va = 2  # Rising volume (150%+)

        # Result
        if va == 1:
            candle_type = 'CLIMAX_BULL' if is_bull else 'CLIMAX_BEAR'
            is_climax = True
        elif va == 2:
            candle_type = 'RISING_BULL' if is_bull else 'RISING_BEAR'
            is_climax = False
        else:
            candle_type = 'BULL' if is_bull else 'BEAR'
            is_climax = False

        return {
            'type': candle_type,
            'is_climax': va == 1,
            'is_rising': va == 2,
            'is_bull': is_bull,
            'volume_ratio': volume[-1] / avg_vol[-1] if avg_vol[-1] > 0 else 1
        }


class SupplyDemandZones:
    """
    Supply and Demand Zone Detection.
    Used for intelligent SL/TP placement.
    """

    def __init__(self, swing_length: int = 10, zone_width_atr: float = 2.5):
        self.swing_length = swing_length
        self.zone_width_atr = zone_width_atr

    def calculate(self, df: pd.DataFrame) -> Dict:
        """
        Detect supply and demand zones.
        Returns nearest zones for SL/TP placement.
        """
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        # Calculate ATR for zone width
        tr = np.maximum(high - low, np.abs(high - np.roll(close, 1)))
        tr = np.maximum(tr, np.abs(low - np.roll(close, 1)))
        atr = pd.Series(tr).rolling(50).mean().values[-1]
        zone_buffer = atr * (self.zone_width_atr / 10)

        # Find swing highs and lows
        swing_highs = []
        swing_lows = []

        for i in range(self.swing_length, len(high) - self.swing_length):
            # Swing high
            if high[i] == max(high[i-self.swing_length:i+self.swing_length+1]):
                swing_highs.append({
                    'price': high[i],
                    'index': i,
                    'zone_top': high[i],
                    'zone_bottom': high[i] - zone_buffer
                })

            # Swing low
            if low[i] == min(low[i-self.swing_length:i+self.swing_length+1]):
                swing_lows.append({
                    'price': low[i],
                    'index': i,
                    'zone_top': low[i] + zone_buffer,
                    'zone_bottom': low[i]
                })

        current_price = close[-1]

        # Find nearest supply (above price)
        supply_zones = [z for z in swing_highs if z['zone_bottom'] > current_price]
        supply_zones.sort(key=lambda x: x['price'])
        nearest_supply = supply_zones[0] if supply_zones else None

        # Find nearest demand (below price)
        demand_zones = [z for z in swing_lows if z['zone_top'] < current_price]
        demand_zones.sort(key=lambda x: -x['price'])
        nearest_demand = demand_zones[0] if demand_zones else None

        return {
            'nearest_supply': nearest_supply,
            'nearest_demand': nearest_demand,
            'all_supply_zones': swing_highs[-10:],  # Last 10 zones
            'all_demand_zones': swing_lows[-10:],
            'atr': atr
        }


class SessionManager:
    """
    Trading Session Manager with automatic DST handling.
    Critical for scalping to trade during high-liquidity periods.
    """

    # Session times in UTC (will be adjusted for DST)
    SESSIONS = {
        'TOKYO': {'start': time(0, 0), 'end': time(9, 0)},
        'LONDON': {'start': time(8, 0), 'end': time(16, 0)},
        'NEW_YORK': {'start': time(13, 0), 'end': time(21, 0)},
        'SYDNEY': {'start': time(22, 0), 'end': time(7, 0)},
    }

    # Best pairs for each session
    SESSION_PAIRS = {
        'TOKYO': ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'NZDJPY', 'AUDUSD'],
        'LONDON': ['EURUSD', 'GBPUSD', 'EURGBP', 'EURCHF', 'GBPCHF', 'EURJPY'],
        'NEW_YORK': ['EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY', 'USDCHF'],
        'SYDNEY': ['AUDUSD', 'NZDUSD', 'AUDJPY', 'NZDJPY'],
    }

    # Overlap periods (highest volatility)
    OVERLAPS = {
        'LONDON_NEWYORK': {'start': time(13, 0), 'end': time(16, 0)},  # Best time for majors
        'TOKYO_LONDON': {'start': time(8, 0), 'end': time(9, 0)},
    }

    def __init__(self):
        self.current_session = None
        self.is_overlap = False

    def get_current_session(self, timestamp: datetime = None) -> Dict:
        """
        Get current trading session information.
        Returns session name, optimal pairs, and volatility expectation.
        """
        if timestamp is None:
            timestamp = datetime.utcnow()

        current_time = timestamp.time()
        active_sessions = []

        for session_name, times in self.SESSIONS.items():
            if self._is_time_in_session(current_time, times['start'], times['end']):
                active_sessions.append(session_name)

        # Check overlaps
        in_overlap = False
        overlap_name = None
        for overlap_name, times in self.OVERLAPS.items():
            if self._is_time_in_session(current_time, times['start'], times['end']):
                in_overlap = True
                break

        # Determine volatility expectation
        if in_overlap:
            volatility = 'HIGH'
            optimal_pairs = list(set(
                self.SESSION_PAIRS.get('LONDON', []) +
                self.SESSION_PAIRS.get('NEW_YORK', [])
            ))
        elif active_sessions:
            if 'LONDON' in active_sessions or 'NEW_YORK' in active_sessions:
                volatility = 'MEDIUM_HIGH'
            elif 'TOKYO' in active_sessions:
                volatility = 'MEDIUM'
            else:
                volatility = 'LOW'

            optimal_pairs = []
            for session in active_sessions:
                optimal_pairs.extend(self.SESSION_PAIRS.get(session, []))
            optimal_pairs = list(set(optimal_pairs))
        else:
            volatility = 'LOW'
            optimal_pairs = []

        return {
            'active_sessions': active_sessions,
            'is_overlap': in_overlap,
            'overlap_name': overlap_name if in_overlap else None,
            'volatility': volatility,
            'optimal_pairs': optimal_pairs,
            'timestamp': timestamp
        }

    def get_pair_session_score(self, pair: str, timestamp: datetime = None) -> float:
        """
        Get a score (0-100) for how suitable a pair is for the current session.
        """
        session_info = self.get_current_session(timestamp)

        if pair in session_info['optimal_pairs']:
            if session_info['is_overlap']:
                return 100.0
            elif session_info['volatility'] == 'MEDIUM_HIGH':
                return 85.0
            elif session_info['volatility'] == 'MEDIUM':
                return 70.0
            else:
                return 50.0
        else:
            # Pair not optimal for current session
            if session_info['volatility'] == 'HIGH':
                return 60.0  # Still tradeable during overlap
            else:
                return 30.0  # Low priority

    def _is_time_in_session(self, current: time, start: time, end: time) -> bool:
        """Check if current time is within session."""
        if start <= end:
            return start <= current <= end
        else:  # Session crosses midnight
            return current >= start or current <= end


class QQEMod:
    """
    QQE Mod - Enhanced RSI-based indicator.
    Very effective for scalping entries.
    """

    def __init__(self, rsi_period: int = 6, smoothing: int = 5,
                 qqe_factor: float = 3.0, threshold: float = 3.0):
        self.rsi_period = rsi_period
        self.smoothing = smoothing
        self.qqe_factor = qqe_factor
        self.threshold = threshold

    def calculate(self, df: pd.DataFrame) -> IndicatorResult:
        """Calculate QQE Mod signal."""
        close = df['close'].values

        # Calculate RSI
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = pd.Series(gain).ewm(span=self.rsi_period, adjust=False).mean().values
        avg_loss = pd.Series(loss).ewm(span=self.rsi_period, adjust=False).mean().values

        with np.errstate(divide='ignore', invalid='ignore'):
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

        rsi = np.nan_to_num(rsi, nan=50)

        # Smooth RSI
        rsi_ma = pd.Series(rsi).ewm(span=self.smoothing, adjust=False).mean().values

        # ATR of RSI
        wilders_period = self.rsi_period * 2 - 1
        atr_rsi = np.abs(rsi_ma - np.roll(rsi_ma, 1))
        ma_atr_rsi = pd.Series(atr_rsi).ewm(span=wilders_period, adjust=False).mean().values
        dar = pd.Series(ma_atr_rsi).ewm(span=wilders_period, adjust=False).mean().values * self.qqe_factor

        # Calculate bands
        longband = np.zeros(len(close))
        shortband = np.zeros(len(close))
        trend = np.zeros(len(close))

        for i in range(1, len(close)):
            new_longband = rsi_ma[i] - dar[i]
            new_shortband = rsi_ma[i] + dar[i]

            if rsi_ma[i-1] > longband[i-1] and rsi_ma[i] > longband[i-1]:
                longband[i] = max(longband[i-1], new_longband)
            else:
                longband[i] = new_longband

            if rsi_ma[i-1] < shortband[i-1] and rsi_ma[i] < shortband[i-1]:
                shortband[i] = min(shortband[i-1], new_shortband)
            else:
                shortband[i] = new_shortband

            if rsi_ma[i] > shortband[i-1]:
                trend[i] = 1
            elif rsi_ma[i] < longband[i-1]:
                trend[i] = -1
            else:
                trend[i] = trend[i-1]

        # Signal based on RSI level and trend
        rsi_level = rsi_ma[-1] - 50

        greenbar = rsi_level > self.threshold
        redbar = rsi_level < -self.threshold

        if greenbar and trend[-1] == 1:
            strength = min(100, abs(rsi_level) * 3)
            return IndicatorResult('LONG', strength, rsi_level, {'rsi_ma': rsi_ma[-1], 'trend': 1})
        elif redbar and trend[-1] == -1:
            strength = min(100, abs(rsi_level) * 3)
            return IndicatorResult('SHORT', strength, rsi_level, {'rsi_ma': rsi_ma[-1], 'trend': -1})

        return IndicatorResult('NEUTRAL', 0, rsi_level, {'rsi_ma': rsi_ma[-1], 'trend': trend[-1]})


class ConfirmationSystem:
    """
    Multi-indicator confirmation system with signal expiry.
    Inspired by DIY Custom Strategy Builder's approach.
    """

    def __init__(self, signal_expiry: int = 3):
        self.signal_expiry = signal_expiry
        self.indicators = {}
        self.leading_signal_count = {'LONG': 0, 'SHORT': 0}

    def add_leading_signal(self, signal: str, timestamp: int):
        """Track leading indicator signal for expiry calculation."""
        if signal in ['LONG', 'SHORT']:
            self.leading_signal_count[signal] = timestamp

    def check_confirmation(self, leading_signal: str, confirmations: List[IndicatorResult],
                          candles_since_leading: int) -> Tuple[bool, float, List[str]]:
        """
        Check if confirmations align with leading signal within expiry window.
        Returns: (is_confirmed, confidence, failed_confirmations)
        """
        if candles_since_leading > self.signal_expiry:
            return False, 0, ['Signal expired']

        confirmed_count = 0
        failed_list = []

        for conf in confirmations:
            if conf.signal == leading_signal or conf.signal == 'NEUTRAL':
                confirmed_count += 1
            else:
                failed_list.append(f"{conf.signal} conflict")

        total = len(confirmations)
        if total == 0:
            return True, 70, []  # No confirmations = default confidence

        confirmation_rate = confirmed_count / total
        is_confirmed = confirmation_rate >= 0.7  # 70% of confirmations must align

        # Calculate confidence
        base_confidence = 60
        confirmation_bonus = confirmation_rate * 30
        expiry_penalty = (candles_since_leading / self.signal_expiry) * 10

        confidence = base_confidence + confirmation_bonus - expiry_penalty

        return is_confirmed, confidence, failed_list


# Utility functions
def calculate_adaptive_sl_tp(entry: float, direction: str, atr: float,
                             supply_demand: Dict, session_volatility: str) -> Tuple[float, float]:
    """
    Calculate adaptive SL/TP based on:
    - Supply/Demand zones
    - Session volatility
    - ATR
    """
    # Base multipliers
    if session_volatility == 'HIGH':
        sl_mult = 1.8
        tp_mult = 3.0
    elif session_volatility == 'MEDIUM_HIGH':
        sl_mult = 1.5
        tp_mult = 2.5
    elif session_volatility == 'MEDIUM':
        sl_mult = 1.3
        tp_mult = 2.0
    else:
        sl_mult = 1.0
        tp_mult = 1.5

    if direction == 'BUY':
        # SL below entry, consider demand zone
        default_sl = entry - (atr * sl_mult)
        if supply_demand.get('nearest_demand'):
            demand_sl = supply_demand['nearest_demand']['zone_bottom']
            sl = min(default_sl, demand_sl)  # Use tighter of the two
        else:
            sl = default_sl

        # TP above entry, consider supply zone
        default_tp = entry + (atr * tp_mult)
        if supply_demand.get('nearest_supply'):
            supply_tp = supply_demand['nearest_supply']['zone_bottom']
            tp = min(default_tp, supply_tp)  # Don't exceed supply zone
        else:
            tp = default_tp

    else:  # SELL
        # SL above entry, consider supply zone
        default_sl = entry + (atr * sl_mult)
        if supply_demand.get('nearest_supply'):
            supply_sl = supply_demand['nearest_supply']['zone_top']
            sl = max(default_sl, supply_sl)
        else:
            sl = default_sl

        # TP below entry, consider demand zone
        default_tp = entry - (atr * tp_mult)
        if supply_demand.get('nearest_demand'):
            demand_tp = supply_demand['nearest_demand']['zone_top']
            tp = max(default_tp, demand_tp)
        else:
            tp = default_tp

    return sl, tp
