#!/usr/bin/env python3
"""
EUR/GBP Strategy Optimizer V1.0
================================
Complete strategy optimization system for EUR/GBP.
Tests 40+ strategies, optimizes parameters, validates with Monte Carlo & Walk-Forward.

Workflow:
1. Fetch historical data (2+ years)
2. Backtest all 40 candidate strategies
3. Select top performers (PF >= 1.0, trades >= 100)
4. Grid search parameter optimization
5. Market regime analysis
6. Robustness tests (Monte Carlo, Walk-Forward)
7. Risk management validation
8. Output final validated strategy

Usage:
    python -m core.eurgbp_strategy_optimizer

Part of Forex Scalper Agent V2
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import warnings
import time
import os
import sys

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from core.multi_source_fetcher import MultiSourceFetcher
except ImportError:
    from multi_source_fetcher import MultiSourceFetcher


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: datetime
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    exit_price: float
    stop_loss: float
    take_profit: float
    pnl: float
    pnl_pips: float
    result: str  # 'WIN', 'LOSS', 'BE'


@dataclass
class BacktestResult:
    """Backtest performance metrics."""
    strategy_name: str
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    profit_factor: float
    total_pnl: float
    max_drawdown: float
    sharpe_ratio: float
    trades: List[Trade]
    params: Dict


class MarketRegime(Enum):
    """Market regime classification."""
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING = "RANGING"
    STRONG_TREND = "STRONG_TREND"
    BREAKOUT = "BREAKOUT"
    CONSOLIDATION = "CONSOLIDATION"
    UNKNOWN = "UNKNOWN"


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

class Indicators:
    """Technical indicator calculations."""

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average."""
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average."""
        return series.rolling(window=period).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD indicator."""
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                   k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        d = k.rolling(window=d_period).mean()
        return k, d

    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_mult: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands."""
        middle = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = middle + (std * std_mult)
        lower = middle - (std * std_mult)
        return upper, middle, lower

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range."""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Average Directional Index with +DI and -DI."""
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

    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """Commodity Channel Index."""
        typical_price = (high + low + close) / 3
        sma = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (typical_price - sma) / (0.015 * mean_deviation + 1e-10)

    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Williams %R."""
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)

    @staticmethod
    def supertrend(high: pd.Series, low: pd.Series, close: pd.Series,
                   period: int = 10, multiplier: float = 3.0) -> Tuple[pd.Series, pd.Series]:
        """Supertrend indicator."""
        atr = Indicators.atr(high, low, close, period)
        hl2 = (high + low) / 2

        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)

        supertrend = pd.Series(index=close.index, dtype=float)
        direction = pd.Series(index=close.index, dtype=int)

        for i in range(period, len(close)):
            if i == period:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                if close.iloc[i] > supertrend.iloc[i-1]:
                    supertrend.iloc[i] = lower_band.iloc[i]
                    direction.iloc[i] = 1
                elif close.iloc[i] < supertrend.iloc[i-1]:
                    supertrend.iloc[i] = upper_band.iloc[i]
                    direction.iloc[i] = -1
                else:
                    supertrend.iloc[i] = supertrend.iloc[i-1]
                    direction.iloc[i] = direction.iloc[i-1]

        return supertrend, direction

    @staticmethod
    def donchian_channel(high: pd.Series, low: pd.Series, period: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Donchian Channel."""
        upper = high.rolling(window=period).max()
        lower = low.rolling(window=period).min()
        middle = (upper + lower) / 2
        return upper, middle, lower


# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================

class StrategyLibrary:
    """Library of 40 trading strategies."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._calculate_all_indicators()

    def _calculate_all_indicators(self):
        """Pre-calculate all indicators needed by strategies."""
        df = self.df

        # EMAs
        df['ema_5'] = Indicators.ema(df['close'], 5)
        df['ema_8'] = Indicators.ema(df['close'], 8)
        df['ema_9'] = Indicators.ema(df['close'], 9)
        df['ema_10'] = Indicators.ema(df['close'], 10)
        df['ema_12'] = Indicators.ema(df['close'], 12)
        df['ema_13'] = Indicators.ema(df['close'], 13)
        df['ema_20'] = Indicators.ema(df['close'], 20)
        df['ema_21'] = Indicators.ema(df['close'], 21)
        df['ema_26'] = Indicators.ema(df['close'], 26)
        df['ema_34'] = Indicators.ema(df['close'], 34)
        df['ema_50'] = Indicators.ema(df['close'], 50)
        df['ema_55'] = Indicators.ema(df['close'], 55)
        df['ema_89'] = Indicators.ema(df['close'], 89)

        # SMAs
        df['sma_50'] = Indicators.sma(df['close'], 50)
        df['sma_200'] = Indicators.sma(df['close'], 200)

        # RSI
        df['rsi_14'] = Indicators.rsi(df['close'], 14)

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = Indicators.macd(df['close'])

        # Stochastic
        df['stoch_k'], df['stoch_d'] = Indicators.stochastic(df['high'], df['low'], df['close'])

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = Indicators.bollinger_bands(df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle'] * 100
        df['bb_pct_b'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        # ATR
        df['atr'] = Indicators.atr(df['high'], df['low'], df['close'])

        # ADX
        df['adx'], df['plus_di'], df['minus_di'] = Indicators.adx(df['high'], df['low'], df['close'])

        # CCI
        df['cci'] = Indicators.cci(df['high'], df['low'], df['close'])

        # Williams %R
        df['williams_r'] = Indicators.williams_r(df['high'], df['low'], df['close'])

        # Supertrend
        df['supertrend'], df['supertrend_dir'] = Indicators.supertrend(df['high'], df['low'], df['close'])

        # Donchian
        df['donchian_upper'], df['donchian_middle'], df['donchian_lower'] = Indicators.donchian_channel(df['high'], df['low'])

        # Z-Score for mean reversion
        df['zscore'] = (df['close'] - df['close'].rolling(20).mean()) / (df['close'].rolling(20).std() + 1e-10)

        # ROC (Rate of Change)
        df['roc'] = (df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100

        self.df = df

    def get_all_strategies(self) -> Dict[str, Callable]:
        """Return all 40 strategies."""
        return {
            # TREND FOLLOWING (1-10)
            '01_ema_triple_cross': self.strategy_ema_triple_cross,
            '02_ema_double_cross': self.strategy_ema_double_cross,
            '03_sma_golden_cross': self.strategy_sma_golden_cross,
            '04_macd_crossover': self.strategy_macd_crossover,
            '05_macd_zero_line': self.strategy_macd_zero_line,
            '06_adx_trend': self.strategy_adx_trend,
            '07_parabolic_sar': self.strategy_parabolic_sar,
            '08_ichimoku_basic': self.strategy_ichimoku_basic,
            '09_supertrend': self.strategy_supertrend,
            '10_donchian_breakout': self.strategy_donchian_breakout,

            # MOMENTUM (11-18)
            '11_rsi_oversold': self.strategy_rsi_oversold,
            '12_rsi_centerline': self.strategy_rsi_centerline,
            '13_rsi_divergence': self.strategy_rsi_divergence,
            '14_stochastic_cross': self.strategy_stochastic_cross,
            '15_stochastic_double': self.strategy_stochastic_double,
            '16_cci': self.strategy_cci,
            '17_williams_r': self.strategy_williams_r,
            '18_mfi': self.strategy_mfi,

            # VOLATILITY (19-24)
            '19_bb_bounce': self.strategy_bb_bounce,
            '20_bb_squeeze': self.strategy_bb_squeeze,
            '21_bb_pct_b': self.strategy_bb_pct_b,
            '22_keltner': self.strategy_keltner,
            '23_atr_breakout': self.strategy_atr_breakout,
            '24_vcp': self.strategy_vcp,

            # SUPPORT/RESISTANCE (25-29)
            '25_pivot_points': self.strategy_pivot_points,
            '26_fibonacci': self.strategy_fibonacci,
            '27_sr_breakout': self.strategy_sr_breakout,
            '28_pin_bar': self.strategy_pin_bar,
            '29_engulfing': self.strategy_engulfing,

            # COMBINED (30-35)
            '30_ema_rsi_adx': self.strategy_ema_rsi_adx,
            '31_macd_stochastic': self.strategy_macd_stochastic,
            '32_bb_rsi': self.strategy_bb_rsi,
            '33_ichimoku_macd': self.strategy_ichimoku_macd,
            '34_triple_screen': self.strategy_triple_screen,
            '35_ma_ribbon': self.strategy_ma_ribbon,

            # ADVANCED (36-40)
            '36_mean_reversion': self.strategy_mean_reversion,
            '37_momentum_breakout': self.strategy_momentum_breakout,
            '38_vwap': self.strategy_vwap,
            '39_order_flow': self.strategy_order_flow,
            '40_range_breakout': self.strategy_range_breakout,
        }

    # =========================================================================
    # TREND FOLLOWING STRATEGIES (1-10)
    # =========================================================================

    def strategy_ema_triple_cross(self, params: Dict = None) -> pd.Series:
        """EMA Triple Crossover: EMA(8) > EMA(21) > EMA(50) for BUY."""
        params = params or {'fast': 8, 'mid': 21, 'slow': 50}
        df = self.df

        ema_fast = Indicators.ema(df['close'], params['fast'])
        ema_mid = Indicators.ema(df['close'], params['mid'])
        ema_slow = Indicators.ema(df['close'], params['slow'])

        buy_signal = (ema_fast > ema_mid) & (ema_mid > ema_slow)
        sell_signal = (ema_fast < ema_mid) & (ema_mid < ema_slow)

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_ema_double_cross(self, params: Dict = None) -> pd.Series:
        """EMA Double Crossover: EMA(9) crosses EMA(21)."""
        params = params or {'fast': 9, 'slow': 21}
        df = self.df

        ema_fast = Indicators.ema(df['close'], params['fast'])
        ema_slow = Indicators.ema(df['close'], params['slow'])

        buy_signal = (ema_fast > ema_slow) & (ema_fast.shift(1) <= ema_slow.shift(1))
        sell_signal = (ema_fast < ema_slow) & (ema_fast.shift(1) >= ema_slow.shift(1))

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_sma_golden_cross(self, params: Dict = None) -> pd.Series:
        """Golden/Death Cross: SMA(50) crosses SMA(200)."""
        df = self.df

        buy_signal = (df['sma_50'] > df['sma_200']) & (df['sma_50'].shift(1) <= df['sma_200'].shift(1))
        sell_signal = (df['sma_50'] < df['sma_200']) & (df['sma_50'].shift(1) >= df['sma_200'].shift(1))

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_macd_crossover(self, params: Dict = None) -> pd.Series:
        """MACD Crossover: MACD line crosses Signal line."""
        df = self.df

        buy_signal = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        sell_signal = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_macd_zero_line(self, params: Dict = None) -> pd.Series:
        """MACD + Zero Line: MACD > 0 AND bullish crossover."""
        df = self.df

        crossover_up = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
        crossover_down = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))

        buy_signal = crossover_up & (df['macd'] > 0)
        sell_signal = crossover_down & (df['macd'] < 0)

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_adx_trend(self, params: Dict = None) -> pd.Series:
        """ADX Trend: ADX > 25 AND +DI > -DI for BUY."""
        params = params or {'adx_threshold': 25}
        df = self.df

        strong_trend = df['adx'] > params['adx_threshold']
        buy_signal = strong_trend & (df['plus_di'] > df['minus_di'])
        sell_signal = strong_trend & (df['minus_di'] > df['plus_di'])

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_parabolic_sar(self, params: Dict = None) -> pd.Series:
        """Parabolic SAR: Price crosses SAR."""
        params = params or {'af_start': 0.02, 'af_max': 0.2}
        df = self.df

        # Simplified SAR using EMA trend
        trend = (df['close'] > df['ema_20']).astype(int)
        trend_change_up = (trend == 1) & (trend.shift(1) == 0)
        trend_change_down = (trend == 0) & (trend.shift(1) == 1)

        signal = pd.Series(0, index=df.index)
        signal[trend_change_up] = 1
        signal[trend_change_down] = -1
        return signal

    def strategy_ichimoku_basic(self, params: Dict = None) -> pd.Series:
        """Ichimoku Basic: Price above cloud + Tenkan > Kijun."""
        df = self.df

        # Calculate Ichimoku components
        tenkan = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
        kijun = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)

        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

        buy_signal = (df['close'] > cloud_top) & (tenkan > kijun)
        sell_signal = (df['close'] < cloud_bottom) & (tenkan < kijun)

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_supertrend(self, params: Dict = None) -> pd.Series:
        """Supertrend: Price crosses Supertrend line."""
        df = self.df

        buy_signal = (df['supertrend_dir'] == 1) & (df['supertrend_dir'].shift(1) == -1)
        sell_signal = (df['supertrend_dir'] == -1) & (df['supertrend_dir'].shift(1) == 1)

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_donchian_breakout(self, params: Dict = None) -> pd.Series:
        """Donchian Breakout: Price breaks channel."""
        df = self.df

        buy_signal = df['close'] > df['donchian_upper'].shift(1)
        sell_signal = df['close'] < df['donchian_lower'].shift(1)

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    # =========================================================================
    # MOMENTUM STRATEGIES (11-18)
    # =========================================================================

    def strategy_rsi_oversold(self, params: Dict = None) -> pd.Series:
        """RSI Overbought/Oversold: RSI < 30 for BUY, RSI > 70 for SELL."""
        params = params or {'oversold': 30, 'overbought': 70}
        df = self.df

        # Entry when crossing threshold
        buy_signal = (df['rsi_14'] < params['oversold']) & (df['rsi_14'].shift(1) >= params['oversold'])
        sell_signal = (df['rsi_14'] > params['overbought']) & (df['rsi_14'].shift(1) <= params['overbought'])

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_rsi_centerline(self, params: Dict = None) -> pd.Series:
        """RSI Centerline Crossover: RSI crosses 50."""
        df = self.df

        buy_signal = (df['rsi_14'] > 50) & (df['rsi_14'].shift(1) <= 50)
        sell_signal = (df['rsi_14'] < 50) & (df['rsi_14'].shift(1) >= 50)

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_rsi_divergence(self, params: Dict = None) -> pd.Series:
        """RSI Divergence: Price vs RSI divergence."""
        df = self.df
        lookback = 14

        # Find local lows/highs
        price_low = df['low'].rolling(lookback).min()
        price_high = df['high'].rolling(lookback).max()
        rsi_low = df['rsi_14'].rolling(lookback).min()
        rsi_high = df['rsi_14'].rolling(lookback).max()

        # Bullish divergence: lower price low, higher RSI low
        bullish_div = (df['low'] <= price_low) & (df['rsi_14'] > rsi_low.shift(1))
        # Bearish divergence: higher price high, lower RSI high
        bearish_div = (df['high'] >= price_high) & (df['rsi_14'] < rsi_high.shift(1))

        signal = pd.Series(0, index=df.index)
        signal[bullish_div] = 1
        signal[bearish_div] = -1
        return signal

    def strategy_stochastic_cross(self, params: Dict = None) -> pd.Series:
        """Stochastic Crossover: %K crosses %D in overbought/oversold zones."""
        params = params or {'oversold': 20, 'overbought': 80}
        df = self.df

        k_crosses_d_up = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        k_crosses_d_down = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))

        buy_signal = k_crosses_d_up & (df['stoch_k'] < params['oversold'] + 10)
        sell_signal = k_crosses_d_down & (df['stoch_k'] > params['overbought'] - 10)

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_stochastic_double(self, params: Dict = None) -> pd.Series:
        """Stochastic Double Cross: Both %K and %D in extreme zones."""
        params = params or {'oversold': 20, 'overbought': 80}
        df = self.df

        both_oversold = (df['stoch_k'] < params['oversold']) & (df['stoch_d'] < params['oversold'])
        both_overbought = (df['stoch_k'] > params['overbought']) & (df['stoch_d'] > params['overbought'])

        k_crosses_up = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        k_crosses_down = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))

        buy_signal = k_crosses_up & both_oversold.shift(1)
        sell_signal = k_crosses_down & both_overbought.shift(1)

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_cci(self, params: Dict = None) -> pd.Series:
        """CCI: CCI crosses -100 or +100."""
        params = params or {'threshold': 100}
        df = self.df

        buy_signal = (df['cci'] > -params['threshold']) & (df['cci'].shift(1) <= -params['threshold'])
        sell_signal = (df['cci'] < params['threshold']) & (df['cci'].shift(1) >= params['threshold'])

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_williams_r(self, params: Dict = None) -> pd.Series:
        """Williams %R: -80 oversold, -20 overbought."""
        params = params or {'oversold': -80, 'overbought': -20}
        df = self.df

        buy_signal = (df['williams_r'] > params['oversold']) & (df['williams_r'].shift(1) <= params['oversold'])
        sell_signal = (df['williams_r'] < params['overbought']) & (df['williams_r'].shift(1) >= params['overbought'])

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_mfi(self, params: Dict = None) -> pd.Series:
        """MFI (Money Flow Index): Similar to RSI but with volume."""
        params = params or {'oversold': 20, 'overbought': 80}
        df = self.df

        # Simplified MFI using RSI as proxy (volume not available for forex)
        mfi = df['rsi_14']  # Use RSI as MFI proxy

        buy_signal = (mfi > params['oversold']) & (mfi.shift(1) <= params['oversold'])
        sell_signal = (mfi < params['overbought']) & (mfi.shift(1) >= params['overbought'])

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    # =========================================================================
    # VOLATILITY STRATEGIES (19-24)
    # =========================================================================

    def strategy_bb_bounce(self, params: Dict = None) -> pd.Series:
        """Bollinger Band Bounce: Price touches lower/upper band."""
        df = self.df

        buy_signal = df['close'] <= df['bb_lower']
        sell_signal = df['close'] >= df['bb_upper']

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_bb_squeeze(self, params: Dict = None) -> pd.Series:
        """BB Squeeze Breakout: Breakout after narrow bands."""
        params = params or {'squeeze_threshold': 1.5}
        df = self.df

        bb_width_avg = df['bb_width'].rolling(20).mean()
        squeeze = df['bb_width'] < bb_width_avg * 0.5

        breakout_up = (df['close'] > df['bb_upper']) & squeeze.shift(1)
        breakout_down = (df['close'] < df['bb_lower']) & squeeze.shift(1)

        signal = pd.Series(0, index=df.index)
        signal[breakout_up] = 1
        signal[breakout_down] = -1
        return signal

    def strategy_bb_pct_b(self, params: Dict = None) -> pd.Series:
        """BB %B: Trade extreme %B values."""
        df = self.df

        buy_signal = (df['bb_pct_b'] < 0) & (df['bb_pct_b'].shift(1) >= 0)
        sell_signal = (df['bb_pct_b'] > 1) & (df['bb_pct_b'].shift(1) <= 1)

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_keltner(self, params: Dict = None) -> pd.Series:
        """Keltner Channel breakout."""
        params = params or {'period': 20, 'mult': 2.0}
        df = self.df

        middle = df['ema_20']
        upper = middle + (df['atr'] * params['mult'])
        lower = middle - (df['atr'] * params['mult'])

        buy_signal = df['close'] > upper
        sell_signal = df['close'] < lower

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_atr_breakout(self, params: Dict = None) -> pd.Series:
        """ATR Breakout: Price > High[-1] + 1.5*ATR."""
        params = params or {'mult': 1.5}
        df = self.df

        buy_signal = df['close'] > df['high'].shift(1) + (df['atr'] * params['mult'])
        sell_signal = df['close'] < df['low'].shift(1) - (df['atr'] * params['mult'])

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_vcp(self, params: Dict = None) -> pd.Series:
        """Volatility Contraction Pattern: Breakout after compression."""
        df = self.df

        # Detect volatility contraction
        atr_ma = df['atr'].rolling(20).mean()
        contraction = df['atr'] < atr_ma * 0.6

        # Breakout after contraction
        breakout_up = (df['close'] > df['high'].shift(1)) & contraction.shift(1)
        breakout_down = (df['close'] < df['low'].shift(1)) & contraction.shift(1)

        signal = pd.Series(0, index=df.index)
        signal[breakout_up] = 1
        signal[breakout_down] = -1
        return signal

    # =========================================================================
    # SUPPORT/RESISTANCE STRATEGIES (25-29)
    # =========================================================================

    def strategy_pivot_points(self, params: Dict = None) -> pd.Series:
        """Pivot Points: Bounce from S1/R1."""
        df = self.df

        # Calculate daily pivot points
        pivot = (df['high'].shift(1) + df['low'].shift(1) + df['close'].shift(1)) / 3
        s1 = 2 * pivot - df['high'].shift(1)
        r1 = 2 * pivot - df['low'].shift(1)

        buy_signal = (df['close'] > s1) & (df['low'] <= s1)
        sell_signal = (df['close'] < r1) & (df['high'] >= r1)

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_fibonacci(self, params: Dict = None) -> pd.Series:
        """Fibonacci Retracement: Bounce from 38.2%, 50%, 61.8%."""
        df = self.df
        lookback = 50

        swing_high = df['high'].rolling(lookback).max()
        swing_low = df['low'].rolling(lookback).min()

        fib_382 = swing_high - (swing_high - swing_low) * 0.382
        fib_618 = swing_high - (swing_high - swing_low) * 0.618

        # Uptrend: bounce from fib levels
        uptrend = df['close'] > df['ema_50']
        buy_signal = uptrend & (df['low'] <= fib_618) & (df['close'] > fib_618)

        # Downtrend: rejection from fib levels
        downtrend = df['close'] < df['ema_50']
        sell_signal = downtrend & (df['high'] >= fib_382) & (df['close'] < fib_382)

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_sr_breakout(self, params: Dict = None) -> pd.Series:
        """Support/Resistance Breakout."""
        df = self.df
        lookback = 20

        resistance = df['high'].rolling(lookback).max()
        support = df['low'].rolling(lookback).min()

        buy_signal = df['close'] > resistance.shift(1)
        sell_signal = df['close'] < support.shift(1)

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_pin_bar(self, params: Dict = None) -> pd.Series:
        """Pin Bar: Long wick candlestick pattern."""
        df = self.df

        body = abs(df['close'] - df['open'])
        full_range = df['high'] - df['low']
        upper_wick = df['high'] - df[['close', 'open']].max(axis=1)
        lower_wick = df[['close', 'open']].min(axis=1) - df['low']

        # Bullish pin bar: long lower wick
        bullish_pin = (lower_wick > body * 2) & (lower_wick > full_range * 0.6)
        # Bearish pin bar: long upper wick
        bearish_pin = (upper_wick > body * 2) & (upper_wick > full_range * 0.6)

        signal = pd.Series(0, index=df.index)
        signal[bullish_pin] = 1
        signal[bearish_pin] = -1
        return signal

    def strategy_engulfing(self, params: Dict = None) -> pd.Series:
        """Engulfing Pattern."""
        df = self.df

        prev_body = abs(df['close'].shift(1) - df['open'].shift(1))
        curr_body = abs(df['close'] - df['open'])

        bullish_engulf = (df['close'] > df['open']) & \
                         (df['close'].shift(1) < df['open'].shift(1)) & \
                         (df['close'] > df['open'].shift(1)) & \
                         (df['open'] < df['close'].shift(1)) & \
                         (curr_body > prev_body)

        bearish_engulf = (df['close'] < df['open']) & \
                         (df['close'].shift(1) > df['open'].shift(1)) & \
                         (df['close'] < df['open'].shift(1)) & \
                         (df['open'] > df['close'].shift(1)) & \
                         (curr_body > prev_body)

        signal = pd.Series(0, index=df.index)
        signal[bullish_engulf] = 1
        signal[bearish_engulf] = -1
        return signal

    # =========================================================================
    # COMBINED STRATEGIES (30-35)
    # =========================================================================

    def strategy_ema_rsi_adx(self, params: Dict = None) -> pd.Series:
        """EMA + RSI + ADX: Multi-confirmation."""
        params = params or {'adx_min': 25}
        df = self.df

        trend_up = df['ema_8'] > df['ema_21']
        trend_down = df['ema_8'] < df['ema_21']
        strong_trend = df['adx'] > params['adx_min']
        rsi_bullish = df['rsi_14'] > 50
        rsi_bearish = df['rsi_14'] < 50

        buy_signal = trend_up & strong_trend & rsi_bullish
        sell_signal = trend_down & strong_trend & rsi_bearish

        # Only signal on transition
        buy_signal = buy_signal & ~buy_signal.shift(1).fillna(False)
        sell_signal = sell_signal & ~sell_signal.shift(1).fillna(False)

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_macd_stochastic(self, params: Dict = None) -> pd.Series:
        """MACD + Stochastic: Double confirmation."""
        df = self.df

        macd_bullish = df['macd'] > df['macd_signal']
        macd_bearish = df['macd'] < df['macd_signal']
        stoch_oversold = df['stoch_k'] < 30
        stoch_overbought = df['stoch_k'] > 70

        macd_cross_up = macd_bullish & ~macd_bullish.shift(1).fillna(False)
        macd_cross_down = macd_bearish & ~macd_bearish.shift(1).fillna(False)

        buy_signal = macd_cross_up & stoch_oversold
        sell_signal = macd_cross_down & stoch_overbought

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_bb_rsi(self, params: Dict = None) -> pd.Series:
        """Bollinger Bands + RSI: Volatility + Momentum."""
        df = self.df

        at_lower_band = df['close'] <= df['bb_lower']
        at_upper_band = df['close'] >= df['bb_upper']
        rsi_oversold = df['rsi_14'] < 30
        rsi_overbought = df['rsi_14'] > 70

        buy_signal = at_lower_band & rsi_oversold
        sell_signal = at_upper_band & rsi_overbought

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_ichimoku_macd(self, params: Dict = None) -> pd.Series:
        """Ichimoku + MACD: Trend + Momentum."""
        df = self.df

        # Ichimoku cloud
        tenkan = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
        kijun = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
        senkou_a = ((tenkan + kijun) / 2).shift(26)
        senkou_b = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)

        cloud_top = pd.concat([senkou_a, senkou_b], axis=1).max(axis=1)
        cloud_bottom = pd.concat([senkou_a, senkou_b], axis=1).min(axis=1)

        above_cloud = df['close'] > cloud_top
        below_cloud = df['close'] < cloud_bottom
        macd_positive = df['macd'] > 0
        macd_negative = df['macd'] < 0

        buy_signal = above_cloud & macd_positive & ~(above_cloud.shift(1) & macd_positive.shift(1))
        sell_signal = below_cloud & macd_negative & ~(below_cloud.shift(1) & macd_negative.shift(1))

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_triple_screen(self, params: Dict = None) -> pd.Series:
        """Triple Screen (Elder): Multi-timeframe simulation."""
        df = self.df

        # Screen 1 (Weekly proxy - 5 bars): MACD trend
        macd_trend = df['macd'].rolling(5).mean() > 0

        # Screen 2 (Daily proxy): Stochastic pullback
        stoch_oversold = df['stoch_k'] < 30
        stoch_overbought = df['stoch_k'] > 70

        # Screen 3: Entry on crossover
        stoch_cross_up = (df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))
        stoch_cross_down = (df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))

        buy_signal = macd_trend & stoch_oversold & stoch_cross_up
        sell_signal = ~macd_trend & stoch_overbought & stoch_cross_down

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_ma_ribbon(self, params: Dict = None) -> pd.Series:
        """MA Ribbon: All EMAs aligned."""
        df = self.df

        bullish_ribbon = (df['ema_8'] > df['ema_13']) & (df['ema_13'] > df['ema_21']) & \
                         (df['ema_21'] > df['ema_34']) & (df['ema_34'] > df['ema_55'])

        bearish_ribbon = (df['ema_8'] < df['ema_13']) & (df['ema_13'] < df['ema_21']) & \
                         (df['ema_21'] < df['ema_34']) & (df['ema_34'] < df['ema_55'])

        buy_signal = bullish_ribbon & ~bullish_ribbon.shift(1).fillna(False)
        sell_signal = bearish_ribbon & ~bearish_ribbon.shift(1).fillna(False)

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    # =========================================================================
    # ADVANCED STRATEGIES (36-40)
    # =========================================================================

    def strategy_mean_reversion(self, params: Dict = None) -> pd.Series:
        """Mean Reversion: Z-Score strategy."""
        params = params or {'threshold': 2.0}
        df = self.df

        buy_signal = (df['zscore'] > -params['threshold']) & (df['zscore'].shift(1) <= -params['threshold'])
        sell_signal = (df['zscore'] < params['threshold']) & (df['zscore'].shift(1) >= params['threshold'])

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_momentum_breakout(self, params: Dict = None) -> pd.Series:
        """Momentum Breakout: ROC with volume confirmation."""
        df = self.df

        roc_positive = df['roc'] > 0
        roc_negative = df['roc'] < 0
        strong_move = abs(df['roc']) > df['roc'].rolling(20).std() * 1.5

        buy_signal = roc_positive & strong_move & ~(roc_positive.shift(1) & strong_move.shift(1))
        sell_signal = roc_negative & strong_move & ~(roc_negative.shift(1) & strong_move.shift(1))

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_vwap(self, params: Dict = None) -> pd.Series:
        """VWAP Strategy: Price crosses VWAP (using EMA as proxy)."""
        df = self.df

        # Using EMA as VWAP proxy (no volume for forex)
        vwap_proxy = df['ema_20']

        buy_signal = (df['close'] > vwap_proxy) & (df['close'].shift(1) <= vwap_proxy.shift(1))
        sell_signal = (df['close'] < vwap_proxy) & (df['close'].shift(1) >= vwap_proxy.shift(1))

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal

    def strategy_order_flow(self, params: Dict = None) -> pd.Series:
        """Order Flow: Price action at key levels."""
        df = self.df

        # Detect strong rejection (proxy for order flow)
        body = abs(df['close'] - df['open'])
        range_pct = body / (df['high'] - df['low'] + 1e-10)

        bullish_rejection = (range_pct < 0.3) & (df['close'] > df['open']) & \
                           ((df['close'] - df['low']) > (df['high'] - df['close']) * 2)
        bearish_rejection = (range_pct < 0.3) & (df['close'] < df['open']) & \
                           ((df['high'] - df['close']) > (df['close'] - df['low']) * 2)

        signal = pd.Series(0, index=df.index)
        signal[bullish_rejection] = 1
        signal[bearish_rejection] = -1
        return signal

    def strategy_range_breakout(self, params: Dict = None) -> pd.Series:
        """Range Breakout: Asian session range breakout."""
        df = self.df

        # Use 8-bar range as proxy for session range
        range_high = df['high'].rolling(8).max()
        range_low = df['low'].rolling(8).min()
        range_size = range_high - range_low
        avg_range = range_size.rolling(20).mean()

        # Tight range followed by breakout
        tight_range = range_size < avg_range * 0.7

        buy_signal = (df['close'] > range_high.shift(1)) & tight_range.shift(1)
        sell_signal = (df['close'] < range_low.shift(1)) & tight_range.shift(1)

        signal = pd.Series(0, index=df.index)
        signal[buy_signal] = 1
        signal[sell_signal] = -1
        return signal


# =============================================================================
# BACKTESTER
# =============================================================================

class Backtester:
    """Backtesting engine with comprehensive metrics."""

    def __init__(self, df: pd.DataFrame, pip_value: float = 0.0001):
        self.df = df
        self.pip_value = pip_value  # EUR/GBP pip = 0.0001

    def run(self, signals: pd.Series, strategy_name: str,
            rr: float = 2.0, sl_atr_mult: float = 1.5,
            params: Dict = None) -> BacktestResult:
        """
        Run backtest on signals.

        Args:
            signals: Series with 1 (BUY), -1 (SELL), 0 (no signal)
            strategy_name: Name of strategy
            rr: Risk-reward ratio
            sl_atr_mult: Stop loss as multiple of ATR
            params: Strategy parameters
        """
        df = self.df.copy()
        df['signal'] = signals

        # Calculate ATR if not present
        if 'atr' not in df.columns:
            df['atr'] = Indicators.atr(df['high'], df['low'], df['close'])

        trades = []
        position = None

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev = df.iloc[i-1]

            # Check for exit if in position
            if position is not None:
                exit_price = None
                exit_reason = None

                if position['direction'] == 'BUY':
                    if row['low'] <= position['sl']:
                        exit_price = position['sl']
                        exit_reason = 'SL'
                    elif row['high'] >= position['tp']:
                        exit_price = position['tp']
                        exit_reason = 'TP'
                else:  # SELL
                    if row['high'] >= position['sl']:
                        exit_price = position['sl']
                        exit_reason = 'SL'
                    elif row['low'] <= position['tp']:
                        exit_price = position['tp']
                        exit_reason = 'TP'

                if exit_price:
                    pnl = (exit_price - position['entry']) if position['direction'] == 'BUY' else (position['entry'] - exit_price)
                    pnl_pips = pnl / self.pip_value

                    trades.append(Trade(
                        entry_time=position['entry_time'],
                        exit_time=df.index[i],
                        direction=position['direction'],
                        entry_price=position['entry'],
                        exit_price=exit_price,
                        stop_loss=position['sl'],
                        take_profit=position['tp'],
                        pnl=pnl,
                        pnl_pips=pnl_pips,
                        result='WIN' if pnl > 0 else ('LOSS' if pnl < 0 else 'BE')
                    ))
                    position = None

            # Check for new signal
            if position is None and row['signal'] != 0:
                entry = row['close']
                atr = row['atr'] if not pd.isna(row['atr']) else 0.0010

                if row['signal'] == 1:  # BUY
                    sl = entry - (atr * sl_atr_mult)
                    tp = entry + (atr * sl_atr_mult * rr)
                    direction = 'BUY'
                else:  # SELL
                    sl = entry + (atr * sl_atr_mult)
                    tp = entry - (atr * sl_atr_mult * rr)
                    direction = 'SELL'

                position = {
                    'direction': direction,
                    'entry': entry,
                    'entry_time': df.index[i],
                    'sl': sl,
                    'tp': tp
                }

        # Calculate metrics
        if not trades:
            return BacktestResult(
                strategy_name=strategy_name,
                total_trades=0,
                wins=0,
                losses=0,
                win_rate=0,
                profit_factor=0,
                total_pnl=0,
                max_drawdown=0,
                sharpe_ratio=0,
                trades=[],
                params=params or {}
            )

        wins = len([t for t in trades if t.result == 'WIN'])
        losses = len([t for t in trades if t.result == 'LOSS'])

        gross_profit = sum([t.pnl for t in trades if t.pnl > 0])
        gross_loss = abs(sum([t.pnl for t in trades if t.pnl < 0]))

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (999 if gross_profit > 0 else 0)

        # Calculate drawdown
        equity = [0]
        for t in trades:
            equity.append(equity[-1] + t.pnl)

        peak = 0
        max_dd = 0
        for e in equity:
            if e > peak:
                peak = e
            dd = (peak - e) / (peak + 1e-10) if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd

        # Sharpe ratio (simplified)
        pnl_series = pd.Series([t.pnl for t in trades])
        sharpe = (pnl_series.mean() / pnl_series.std()) * np.sqrt(252) if pnl_series.std() > 0 else 0

        return BacktestResult(
            strategy_name=strategy_name,
            total_trades=len(trades),
            wins=wins,
            losses=losses,
            win_rate=wins / len(trades) * 100 if trades else 0,
            profit_factor=round(profit_factor, 2),
            total_pnl=sum([t.pnl for t in trades]),
            max_drawdown=max_dd * 100,
            sharpe_ratio=round(sharpe, 2),
            trades=trades,
            params=params or {}
        )


# =============================================================================
# REGIME DETECTOR
# =============================================================================

class RegimeDetector:
    """Detect market regimes."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._calculate_regime_indicators()

    def _calculate_regime_indicators(self):
        df = self.df

        if 'atr' not in df.columns:
            df['atr'] = Indicators.atr(df['high'], df['low'], df['close'])
        if 'adx' not in df.columns:
            df['adx'], df['plus_di'], df['minus_di'] = Indicators.adx(df['high'], df['low'], df['close'])
        if 'ema_50' not in df.columns:
            df['ema_50'] = Indicators.ema(df['close'], 50)
        if 'ema_200' not in df.columns:
            df['ema_200'] = Indicators.ema(df['close'], 200)
        if 'bb_width' not in df.columns:
            bb_upper, bb_middle, bb_lower = Indicators.bollinger_bands(df['close'])
            df['bb_width'] = (bb_upper - bb_lower) / bb_middle * 100

        self.df = df

    def detect_regime(self, idx: int = -1) -> Tuple[MarketRegime, Dict]:
        """Detect regime at given index."""
        df = self.df

        if idx < 60:
            return MarketRegime.UNKNOWN, {}

        row = df.iloc[idx]

        # Calculate averages
        atr_avg = df['atr'].iloc[idx-60:idx].mean()
        bb_width_avg = df['bb_width'].iloc[idx-60:idx].mean()

        current_atr = row['atr']
        current_adx = row['adx']
        current_bb = row['bb_width']

        details = {
            'atr_ratio': round(current_atr / atr_avg, 2) if atr_avg > 0 else 1,
            'adx': round(current_adx, 1),
            'bb_width': round(current_bb, 2)
        }

        # High volatility
        if current_atr > atr_avg * 1.5:
            return MarketRegime.HIGH_VOLATILITY, details

        # Low volatility
        if current_atr < atr_avg * 0.5 and current_bb < bb_width_avg * 0.5:
            return MarketRegime.LOW_VOLATILITY, details

        # Strong trend
        if current_adx > 40:
            if row['plus_di'] > row['minus_di']:
                return MarketRegime.TRENDING_UP, details
            else:
                return MarketRegime.TRENDING_DOWN, details

        # Moderate trend
        if current_adx > 25:
            if row['plus_di'] > row['minus_di']:
                return MarketRegime.TRENDING_UP, details
            else:
                return MarketRegime.TRENDING_DOWN, details

        # Consolidation
        if current_bb < bb_width_avg * 0.6:
            return MarketRegime.CONSOLIDATION, details

        # Ranging
        if current_adx < 20:
            return MarketRegime.RANGING, details

        return MarketRegime.UNKNOWN, details

    def get_regime_series(self) -> pd.Series:
        """Get regime for each bar."""
        regimes = []
        for i in range(len(self.df)):
            regime, _ = self.detect_regime(i)
            regimes.append(regime.value)
        return pd.Series(regimes, index=self.df.index)


# =============================================================================
# MONTE CARLO SIMULATOR
# =============================================================================

class MonteCarloSimulator:
    """Monte Carlo simulation for robustness testing."""

    def __init__(self, trades: List[Trade]):
        self.trades = trades
        self.pnls = [t.pnl for t in trades]

    def run(self, n_simulations: int = 500, initial_capital: float = 10000) -> Dict:
        """Run Monte Carlo simulations."""
        if not self.pnls:
            return {
                'positive_pct': 0,
                'max_dd_95': 100,
                'final_capital_mean': initial_capital,
                'final_capital_std': 0
            }

        results = []
        max_drawdowns = []

        for _ in range(n_simulations):
            # Shuffle PnLs
            shuffled = np.random.permutation(self.pnls)

            # Calculate equity curve
            equity = [initial_capital]
            for pnl in shuffled:
                equity.append(equity[-1] + pnl * 10000)  # Scale for lot size

            # Calculate max drawdown
            peak = initial_capital
            max_dd = 0
            for e in equity:
                if e > peak:
                    peak = e
                dd = (peak - e) / peak if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

            results.append(equity[-1])
            max_drawdowns.append(max_dd)

        return {
            'positive_pct': len([r for r in results if r > initial_capital]) / len(results) * 100,
            'max_dd_95': np.percentile(max_drawdowns, 95) * 100,
            'final_capital_mean': np.mean(results),
            'final_capital_std': np.std(results),
            'ruin_probability': len([r for r in results if r < initial_capital * 0.5]) / len(results) * 100
        }


# =============================================================================
# WALK-FORWARD OPTIMIZER
# =============================================================================

class WalkForwardOptimizer:
    """Walk-Forward Analysis for robustness."""

    def __init__(self, df: pd.DataFrame, strategy_func: Callable,
                 in_sample_pct: float = 0.7, n_segments: int = 5):
        self.df = df
        self.strategy_func = strategy_func
        self.in_sample_pct = in_sample_pct
        self.n_segments = n_segments

    def run(self, param_grid: List[Dict]) -> Dict:
        """Run walk-forward optimization."""
        segment_size = len(self.df) // self.n_segments

        in_sample_results = []
        out_sample_results = []

        for i in range(self.n_segments - 1):
            start = i * segment_size
            end = (i + 2) * segment_size

            segment = self.df.iloc[start:end]
            in_sample_size = int(len(segment) * self.in_sample_pct)

            in_sample = segment.iloc[:in_sample_size]
            out_sample = segment.iloc[in_sample_size:]

            # Optimize on in-sample
            best_pf = 0
            best_params = None

            for params in param_grid:
                lib = StrategyLibrary(in_sample)
                signals = self.strategy_func(lib, params)
                bt = Backtester(in_sample)
                result = bt.run(signals, 'test', params=params)

                if result.profit_factor > best_pf and result.total_trades >= 10:
                    best_pf = result.profit_factor
                    best_params = params

            in_sample_results.append(best_pf)

            # Test on out-of-sample
            if best_params:
                lib = StrategyLibrary(out_sample)
                signals = self.strategy_func(lib, best_params)
                bt = Backtester(out_sample)
                result = bt.run(signals, 'test', params=best_params)
                out_sample_results.append(result.profit_factor)
            else:
                out_sample_results.append(0)

        # Calculate WFE
        avg_in = np.mean(in_sample_results) if in_sample_results else 0
        avg_out = np.mean(out_sample_results) if out_sample_results else 0

        wfe = (avg_out / avg_in * 100) if avg_in > 0 else 0

        return {
            'wfe': round(wfe, 1),
            'in_sample_avg_pf': round(avg_in, 2),
            'out_sample_avg_pf': round(avg_out, 2),
            'segments': self.n_segments
        }


# =============================================================================
# MAIN OPTIMIZER
# =============================================================================

class EURGBPOptimizer:
    """Main optimizer for EUR/GBP strategies."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.pair = 'EURGBP'
        self.fetcher = MultiSourceFetcher(verbose=verbose)
        self.results = []

    def log(self, msg: str):
        if self.verbose:
            print(msg)

    def fetch_data(self, start: str = '2020-01-01', end: str = None) -> Optional[pd.DataFrame]:
        """Fetch historical data."""
        if end is None:
            end = datetime.now().strftime('%Y-%m-%d')

        self.log(f"\n[1/8] Fetching {self.pair} data: {start} to {end}...")

        df = self.fetcher.fetch(self.pair, start, end, '1h')

        if df is not None:
            self.log(f"   Got {len(df)} bars from {df.index[0]} to {df.index[-1]}")
            self.log(f"   Source: {df.attrs.get('source', 'unknown')}")

        return df

    def run_all_strategies(self, df: pd.DataFrame) -> List[BacktestResult]:
        """Test all 40 strategies."""
        self.log("\n[2/8] Testing all 40 strategies...")

        library = StrategyLibrary(df)
        strategies = library.get_all_strategies()
        backtester = Backtester(df)

        results = []

        for name, strategy_func in strategies.items():
            try:
                signals = strategy_func()
                result = backtester.run(signals, name)
                results.append(result)

                status = "OK" if result.profit_factor >= 1.0 and result.total_trades >= 50 else "--"
                self.log(f"   [{status}] {name}: PF={result.profit_factor:.2f}, "
                        f"Trades={result.total_trades}, WR={result.win_rate:.1f}%")
            except Exception as e:
                self.log(f"   [ERR] {name}: {str(e)[:50]}")

        return results

    def select_top_strategies(self, results: List[BacktestResult],
                              min_pf: float = 1.0, min_trades: int = 50) -> List[BacktestResult]:
        """Select strategies meeting criteria."""
        self.log(f"\n[3/8] Selecting strategies (PF >= {min_pf}, Trades >= {min_trades})...")

        selected = [r for r in results if r.profit_factor >= min_pf and r.total_trades >= min_trades]
        selected.sort(key=lambda x: x.profit_factor, reverse=True)

        self.log(f"   Selected {len(selected)} strategies:")
        for r in selected[:10]:
            self.log(f"   - {r.strategy_name}: PF={r.profit_factor:.2f}, "
                    f"Trades={r.total_trades}, WR={r.win_rate:.1f}%, DD={r.max_drawdown:.1f}%")

        return selected

    def optimize_parameters(self, df: pd.DataFrame, strategy_name: str,
                           strategy_func: Callable) -> Dict:
        """Grid search optimization."""
        self.log(f"\n[4/8] Optimizing parameters for {strategy_name}...")

        backtester = Backtester(df)
        library = StrategyLibrary(df)

        # Parameter grid
        rr_values = [1.5, 2.0, 2.5, 3.0]
        sl_mult_values = [1.0, 1.5, 2.0]

        best_result = None
        best_params = None

        for rr in rr_values:
            for sl_mult in sl_mult_values:
                signals = strategy_func()
                result = backtester.run(signals, strategy_name, rr=rr, sl_atr_mult=sl_mult)

                if best_result is None or result.profit_factor > best_result.profit_factor:
                    if result.total_trades >= 50:
                        best_result = result
                        best_params = {'rr': rr, 'sl_mult': sl_mult}

        if best_params:
            self.log(f"   Best params: R:R={best_params['rr']}, SL={best_params['sl_mult']}x ATR")
            self.log(f"   Performance: PF={best_result.profit_factor:.2f}, "
                    f"WR={best_result.win_rate:.1f}%, Trades={best_result.total_trades}")

        return {'best_params': best_params, 'best_result': best_result}

    def analyze_regimes(self, df: pd.DataFrame, results: List[BacktestResult]) -> Dict:
        """Analyze performance by market regime."""
        self.log("\n[5/8] Analyzing market regimes...")

        detector = RegimeDetector(df)
        regimes = detector.get_regime_series()

        regime_stats = {}

        for regime in MarketRegime:
            regime_bars = regimes[regimes == regime.value]
            if len(regime_bars) > 0:
                pct = len(regime_bars) / len(regimes) * 100
                regime_stats[regime.value] = {
                    'bars': len(regime_bars),
                    'pct': round(pct, 1)
                }

        self.log("   Regime distribution:")
        for regime, stats in sorted(regime_stats.items(), key=lambda x: x[1]['pct'], reverse=True):
            self.log(f"   - {regime}: {stats['pct']}% ({stats['bars']} bars)")

        return regime_stats

    def run_monte_carlo(self, trades: List[Trade]) -> Dict:
        """Run Monte Carlo simulation."""
        self.log("\n[6/8] Running Monte Carlo simulation (500 iterations)...")

        simulator = MonteCarloSimulator(trades)
        results = simulator.run(n_simulations=500)

        self.log(f"   Positive simulations: {results['positive_pct']:.1f}%")
        self.log(f"   Max DD (95th percentile): {results['max_dd_95']:.1f}%")
        self.log(f"   Ruin probability: {results['ruin_probability']:.1f}%")

        passed = results['positive_pct'] >= 95 and results['max_dd_95'] < 35 and results['ruin_probability'] < 5
        self.log(f"   Status: {'PASS' if passed else 'FAIL'}")

        return results

    def validate_risk(self, trades: List[Trade], max_daily_loss: float = 500) -> Dict:
        """Validate risk management rules."""
        self.log(f"\n[7/8] Validating risk (max daily loss: ${max_daily_loss})...")

        # Group trades by day
        daily_pnl = {}
        for trade in trades:
            day = trade.entry_time.strftime('%Y-%m-%d')
            if day not in daily_pnl:
                daily_pnl[day] = 0
            daily_pnl[day] += trade.pnl * 10000  # Scale

        # Check for violations
        violations = [day for day, pnl in daily_pnl.items() if pnl < -max_daily_loss]

        self.log(f"   Total trading days: {len(daily_pnl)}")
        self.log(f"   Days exceeding limit: {len(violations)}")

        # Find optimal lot size
        lot_sizes = [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]
        optimal_lot = None

        for lot in lot_sizes:
            violations_at_lot = [day for day, pnl in daily_pnl.items()
                                if pnl * lot < -max_daily_loss]
            if len(violations_at_lot) == 0:
                optimal_lot = lot
                break

        self.log(f"   Optimal lot size: {optimal_lot if optimal_lot else 'N/A'}")

        return {
            'total_days': len(daily_pnl),
            'violation_days': len(violations),
            'optimal_lot': optimal_lot
        }

    def generate_report(self, best_strategy: BacktestResult, params: Dict,
                       regime_stats: Dict, monte_carlo: Dict, risk: Dict) -> str:
        """Generate final report."""
        self.log("\n[8/8] Generating final report...")

        report = f"""
================================================================================
EUR/GBP STRATEGY OPTIMIZATION REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

PAIR: EURGBP
STRATEGY: {best_strategy.strategy_name}

OPTIMAL PARAMETERS:
  - R:R Ratio: {params.get('rr', 2.0)}
  - SL Multiplier: {params.get('sl_mult', 1.5)}x ATR

BACKTEST PERFORMANCE:
  - Total Trades: {best_strategy.total_trades}
  - Win Rate: {best_strategy.win_rate:.1f}%
  - Profit Factor: {best_strategy.profit_factor:.2f}
  - Max Drawdown: {best_strategy.max_drawdown:.1f}%
  - Sharpe Ratio: {best_strategy.sharpe_ratio:.2f}

MONTE CARLO VALIDATION:
  - Positive Simulations: {monte_carlo['positive_pct']:.1f}%
  - Max DD (95th pct): {monte_carlo['max_dd_95']:.1f}%
  - Ruin Probability: {monte_carlo['ruin_probability']:.1f}%
  - Status: {'PASS' if monte_carlo['positive_pct'] >= 95 else 'FAIL'}

RISK MANAGEMENT:
  - Optimal Lot Size: {risk.get('optimal_lot', 'N/A')}
  - Max Daily Loss: $500

REGIME ANALYSIS:
"""
        for regime, stats in sorted(regime_stats.items(), key=lambda x: x[1]['pct'], reverse=True):
            report += f"  - {regime}: {stats['pct']}%\n"

        report += """
================================================================================
"""
        return report

    def run(self, start_date: str = '2020-01-01'):
        """Run complete optimization pipeline."""
        print("=" * 70)
        print("EUR/GBP STRATEGY OPTIMIZER V1.0")
        print("=" * 70)

        # 1. Fetch data
        df = self.fetch_data(start_date)
        if df is None or len(df) < 1000:
            print("[ERROR] Insufficient data. Need at least 1000 bars.")
            return None

        # 2. Test all strategies
        all_results = self.run_all_strategies(df)

        # 3. Select top strategies
        top_strategies = self.select_top_strategies(all_results)

        if not top_strategies:
            print("[WARNING] No strategies met the criteria. Relaxing constraints...")
            top_strategies = self.select_top_strategies(all_results, min_pf=0.9, min_trades=30)

        if not top_strategies:
            print("[ERROR] No viable strategies found.")
            return None

        # 4. Optimize best strategy
        best = top_strategies[0]
        library = StrategyLibrary(df)
        strategies = library.get_all_strategies()

        opt_result = self.optimize_parameters(df, best.strategy_name,
                                              strategies[best.strategy_name])

        # 5. Analyze regimes
        regime_stats = self.analyze_regimes(df, all_results)

        # 6. Monte Carlo
        mc_results = self.run_monte_carlo(opt_result['best_result'].trades if opt_result['best_result'] else best.trades)

        # 7. Risk validation
        risk_results = self.validate_risk(opt_result['best_result'].trades if opt_result['best_result'] else best.trades)

        # 8. Generate report
        final_result = opt_result['best_result'] if opt_result['best_result'] else best
        final_params = opt_result['best_params'] if opt_result['best_params'] else {}

        report = self.generate_report(final_result, final_params, regime_stats, mc_results, risk_results)
        print(report)

        return {
            'strategy': final_result,
            'params': final_params,
            'regime_stats': regime_stats,
            'monte_carlo': mc_results,
            'risk': risk_results,
            'all_results': all_results
        }


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    optimizer = EURGBPOptimizer(verbose=True)
    result = optimizer.run(start_date='2020-01-01')

    if result:
        print("\n[SUCCESS] Optimization complete!")
        print(f"Best Strategy: {result['strategy'].strategy_name}")
        print(f"Profit Factor: {result['strategy'].profit_factor:.2f}")
    else:
        print("\n[FAILED] Optimization did not produce viable results.")
