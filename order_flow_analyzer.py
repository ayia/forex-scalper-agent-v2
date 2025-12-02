"""
Order Flow Analyzer - Advanced Price Action Analysis
=====================================================
Analyzes order flow and market microstructure for better entry timing.

Features:
- Delta analysis (buying vs selling pressure)
- Absorption detection
- Imbalance zones
- Liquidity pool detection
- Stop hunt identification
- Institutional footprint detection
- Volume profile analysis
- POC (Point of Control) identification
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class OrderFlowSignal(Enum):
    """Order flow signal types."""
    STRONG_BUYING = "STRONG_BUYING"
    MODERATE_BUYING = "MODERATE_BUYING"
    NEUTRAL = "NEUTRAL"
    MODERATE_SELLING = "MODERATE_SELLING"
    STRONG_SELLING = "STRONG_SELLING"
    ABSORPTION_BUY = "ABSORPTION_BUY"      # Selling absorbed, bullish
    ABSORPTION_SELL = "ABSORPTION_SELL"    # Buying absorbed, bearish
    STOP_HUNT_LONG = "STOP_HUNT_LONG"      # Stops swept, reversal up
    STOP_HUNT_SHORT = "STOP_HUNT_SHORT"    # Stops swept, reversal down
    IMBALANCE_BUY = "IMBALANCE_BUY"        # Bullish imbalance
    IMBALANCE_SELL = "IMBALANCE_SELL"      # Bearish imbalance


@dataclass
class ImbalanceZone:
    """Represents a price imbalance zone."""
    high: float
    low: float
    direction: str  # 'bullish' or 'bearish'
    strength: float  # 0-100
    timestamp: datetime
    filled: bool = False


@dataclass
class LiquidityPool:
    """Represents a liquidity pool (cluster of stops)."""
    price_level: float
    estimated_size: float
    type: str  # 'above' or 'below'
    touched: bool = False
    swept: bool = False


@dataclass
class OrderFlowAnalysis:
    """Complete order flow analysis result."""
    pair: str
    timeframe: str
    timestamp: datetime

    # Delta analysis
    delta: float                    # Buy volume - Sell volume
    cumulative_delta: float         # Running total
    delta_divergence: bool          # Price vs delta divergence

    # Pressure analysis
    buying_pressure: float          # 0-100
    selling_pressure: float         # 0-100
    net_pressure: float             # -100 to +100

    # Signals
    primary_signal: OrderFlowSignal
    signal_strength: float          # 0-100

    # Zones
    imbalance_zones: List[ImbalanceZone]
    liquidity_pools: List[LiquidityPool]

    # Institutional activity
    institutional_buying: bool
    institutional_selling: bool
    large_player_activity: float    # 0-100

    # Volume profile
    poc: float                      # Point of Control
    value_area_high: float
    value_area_low: float
    volume_node_type: str           # 'high_volume' or 'low_volume'

    # Absorption
    absorption_detected: bool
    absorption_direction: Optional[str]

    # Stop hunt
    stop_hunt_detected: bool
    stop_hunt_direction: Optional[str]

    # Entry recommendation
    entry_quality: float            # 0-100
    recommended_action: str


class OrderFlowAnalyzer:
    """
    Advanced order flow analyzer for forex scalping.

    Uses price action and volume to infer order flow without
    actual Level 2 data (which isn't available in forex).
    """

    def __init__(self):
        # Configuration
        self.imbalance_threshold = 2.0    # Min ratio for imbalance
        self.absorption_threshold = 1.5   # Volume spike for absorption
        self.stop_hunt_threshold = 0.7    # % of ATR for stop hunt
        self.large_player_volume = 2.0    # Multiple of avg for institutional

        # Volume profile settings
        self.value_area_pct = 0.70        # 70% of volume
        self.num_price_bins = 50          # Bins for volume profile

        logger.info("OrderFlowAnalyzer initialized")

    def analyze(
        self,
        df: pd.DataFrame,
        pair: str,
        timeframe: str = "M15"
    ) -> OrderFlowAnalysis:
        """
        Perform complete order flow analysis.

        Args:
            df: OHLCV DataFrame
            pair: Trading pair
            timeframe: Timeframe string

        Returns:
            OrderFlowAnalysis with all metrics
        """
        if df is None or len(df) < 50:
            return self._neutral_analysis(pair, timeframe)

        # Normalize column names
        df = self._normalize_columns(df)

        try:
            # 1. Calculate Delta
            delta_analysis = self._analyze_delta(df)

            # 2. Calculate Pressure
            pressure_analysis = self._analyze_pressure(df)

            # 3. Detect Imbalances
            imbalance_zones = self._detect_imbalances(df)

            # 4. Detect Liquidity Pools
            liquidity_pools = self._detect_liquidity_pools(df)

            # 5. Detect Absorption
            absorption = self._detect_absorption(df)

            # 6. Detect Stop Hunts
            stop_hunt = self._detect_stop_hunt(df)

            # 7. Detect Institutional Activity
            institutional = self._detect_institutional_activity(df)

            # 8. Build Volume Profile
            volume_profile = self._build_volume_profile(df)

            # 9. Determine Primary Signal
            primary_signal, signal_strength = self._determine_signal(
                delta_analysis,
                pressure_analysis,
                absorption,
                stop_hunt,
                institutional
            )

            # 10. Calculate Entry Quality
            entry_quality = self._calculate_entry_quality(
                delta_analysis,
                pressure_analysis,
                absorption,
                stop_hunt,
                imbalance_zones,
                signal_strength
            )

            # 11. Determine Recommended Action
            recommended_action = self._get_recommendation(
                primary_signal,
                entry_quality,
                delta_analysis
            )

            return OrderFlowAnalysis(
                pair=pair,
                timeframe=timeframe,
                timestamp=datetime.now(),

                delta=delta_analysis['delta'],
                cumulative_delta=delta_analysis['cumulative_delta'],
                delta_divergence=delta_analysis['divergence'],

                buying_pressure=pressure_analysis['buying'],
                selling_pressure=pressure_analysis['selling'],
                net_pressure=pressure_analysis['net'],

                primary_signal=primary_signal,
                signal_strength=signal_strength,

                imbalance_zones=imbalance_zones,
                liquidity_pools=liquidity_pools,

                institutional_buying=institutional['buying'],
                institutional_selling=institutional['selling'],
                large_player_activity=institutional['activity_score'],

                poc=volume_profile['poc'],
                value_area_high=volume_profile['vah'],
                value_area_low=volume_profile['val'],
                volume_node_type=volume_profile['node_type'],

                absorption_detected=absorption['detected'],
                absorption_direction=absorption['direction'],

                stop_hunt_detected=stop_hunt['detected'],
                stop_hunt_direction=stop_hunt['direction'],

                entry_quality=entry_quality,
                recommended_action=recommended_action
            )

        except Exception as e:
            logger.error(f"Order flow analysis error: {e}")
            return self._neutral_analysis(pair, timeframe)

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        return df

    def _analyze_delta(self, df: pd.DataFrame) -> Dict:
        """
        Analyze volume delta (buying vs selling pressure).

        In forex, we estimate delta using:
        - Candle direction
        - Close position within candle
        - Volume
        """
        close = df['close']
        open_ = df['open']
        high = df['high']
        low = df['low']
        volume = df['volume'] if 'volume' in df.columns else pd.Series([1] * len(df))

        # Estimate buy/sell volume per candle
        candle_range = high - low
        candle_range = candle_range.replace(0, 0.0001)

        # Close position within range (0 = low, 1 = high)
        close_position = (close - low) / candle_range

        # Estimate delta: close near high = more buying, close near low = more selling
        buy_volume = volume * close_position
        sell_volume = volume * (1 - close_position)

        delta = buy_volume - sell_volume
        cumulative_delta = delta.cumsum()

        # Current values
        current_delta = delta.iloc[-1]
        current_cum_delta = cumulative_delta.iloc[-1]

        # Delta divergence: price makes new high but delta doesn't (or vice versa)
        recent_price = close.tail(10)
        recent_delta = cumulative_delta.tail(10)

        price_higher_high = recent_price.iloc[-1] > recent_price.iloc[:-1].max()
        delta_higher_high = recent_delta.iloc[-1] > recent_delta.iloc[:-1].max()

        price_lower_low = recent_price.iloc[-1] < recent_price.iloc[:-1].min()
        delta_lower_low = recent_delta.iloc[-1] < recent_delta.iloc[:-1].min()

        divergence = (price_higher_high and not delta_higher_high) or \
                     (price_lower_low and not delta_lower_low)

        # Delta trend
        delta_ma = delta.rolling(5).mean()
        delta_trend = 'bullish' if delta_ma.iloc[-1] > 0 else 'bearish'

        return {
            'delta': current_delta,
            'cumulative_delta': current_cum_delta,
            'divergence': divergence,
            'trend': delta_trend,
            'delta_series': delta,
            'cumulative_series': cumulative_delta
        }

    def _analyze_pressure(self, df: pd.DataFrame) -> Dict:
        """
        Analyze buying and selling pressure.

        Uses multiple indicators:
        - Candle body analysis
        - Wick analysis
        - Volume weighting
        """
        close = df['close']
        open_ = df['open']
        high = df['high']
        low = df['low']

        # Last N candles for analysis
        n = 10
        recent = df.tail(n)

        # 1. Candle body analysis
        bodies = recent['close'] - recent['open']
        bullish_bodies = bodies[bodies > 0].sum()
        bearish_bodies = abs(bodies[bodies < 0].sum())
        total_body = bullish_bodies + bearish_bodies

        body_buy_pressure = (bullish_bodies / total_body * 100) if total_body > 0 else 50
        body_sell_pressure = (bearish_bodies / total_body * 100) if total_body > 0 else 50

        # 2. Wick analysis (rejection)
        upper_wicks = recent['high'] - recent[['close', 'open']].max(axis=1)
        lower_wicks = recent[['close', 'open']].min(axis=1) - recent['low']

        total_wicks = upper_wicks.sum() + lower_wicks.sum()
        # Upper wicks = selling pressure, lower wicks = buying pressure
        wick_buy_pressure = (lower_wicks.sum() / total_wicks * 100) if total_wicks > 0 else 50
        wick_sell_pressure = (upper_wicks.sum() / total_wicks * 100) if total_wicks > 0 else 50

        # 3. Recent momentum
        momentum = (close.iloc[-1] - close.iloc[-n]) / close.iloc[-n] * 100
        momentum_pressure = 50 + momentum * 10  # Scale

        # Combine pressures
        buying_pressure = (body_buy_pressure * 0.4 + wick_buy_pressure * 0.3 +
                          min(100, max(0, momentum_pressure)) * 0.3)
        selling_pressure = (body_sell_pressure * 0.4 + wick_sell_pressure * 0.3 +
                           min(100, max(0, 100 - momentum_pressure)) * 0.3)

        # Normalize
        buying_pressure = min(100, max(0, buying_pressure))
        selling_pressure = min(100, max(0, selling_pressure))

        net_pressure = buying_pressure - selling_pressure

        return {
            'buying': round(buying_pressure, 1),
            'selling': round(selling_pressure, 1),
            'net': round(net_pressure, 1)
        }

    def _detect_imbalances(self, df: pd.DataFrame) -> List[ImbalanceZone]:
        """
        Detect price imbalance zones (Fair Value Gaps).

        Imbalance = gap between candle highs/lows where price moved
        so fast that there was no trading on the other side.
        """
        imbalances = []

        for i in range(2, len(df)):
            # Bullish imbalance: current low > previous candle's high
            # (gap between candle 1 high and candle 3 low)
            candle_1_high = df['high'].iloc[i-2]
            candle_3_low = df['low'].iloc[i]

            if candle_3_low > candle_1_high:
                # Bullish imbalance (gap up)
                gap_size = candle_3_low - candle_1_high
                avg_range = (df['high'] - df['low']).tail(20).mean()
                strength = min(100, (gap_size / avg_range) * 50)

                if strength > 20:  # Minimum strength threshold
                    imbalances.append(ImbalanceZone(
                        high=candle_3_low,
                        low=candle_1_high,
                        direction='bullish',
                        strength=strength,
                        timestamp=df.index[i] if hasattr(df.index[i], 'isoformat') else datetime.now()
                    ))

            # Bearish imbalance: current high < previous candle's low
            candle_1_low = df['low'].iloc[i-2]
            candle_3_high = df['high'].iloc[i]

            if candle_3_high < candle_1_low:
                # Bearish imbalance (gap down)
                gap_size = candle_1_low - candle_3_high
                avg_range = (df['high'] - df['low']).tail(20).mean()
                strength = min(100, (gap_size / avg_range) * 50)

                if strength > 20:
                    imbalances.append(ImbalanceZone(
                        high=candle_1_low,
                        low=candle_3_high,
                        direction='bearish',
                        strength=strength,
                        timestamp=df.index[i] if hasattr(df.index[i], 'isoformat') else datetime.now()
                    ))

        # Return most recent imbalances
        return imbalances[-5:] if imbalances else []

    def _detect_liquidity_pools(self, df: pd.DataFrame) -> List[LiquidityPool]:
        """
        Detect liquidity pools (clusters of likely stop losses).

        Stops are typically placed:
        - Below recent swing lows (for longs)
        - Above recent swing highs (for shorts)
        - At round numbers
        """
        pools = []
        lookback = 20

        high = df['high']
        low = df['low']
        close = df['close']

        current_price = close.iloc[-1]

        # Find swing highs and lows
        for i in range(lookback, len(df) - lookback):
            # Swing low detection
            if low.iloc[i] == low.iloc[i-lookback:i+lookback+1].min():
                swing_low = low.iloc[i]
                # Estimate pool size based on how many times price tested this level
                tests = sum(1 for j in range(max(0, i-50), min(len(df), i+50))
                           if abs(low.iloc[j] - swing_low) / swing_low < 0.001)

                if swing_low < current_price:  # Pool below current price
                    pools.append(LiquidityPool(
                        price_level=swing_low,
                        estimated_size=tests * 10,  # Arbitrary scale
                        type='below'
                    ))

            # Swing high detection
            if high.iloc[i] == high.iloc[i-lookback:i+lookback+1].max():
                swing_high = high.iloc[i]
                tests = sum(1 for j in range(max(0, i-50), min(len(df), i+50))
                           if abs(high.iloc[j] - swing_high) / swing_high < 0.001)

                if swing_high > current_price:  # Pool above current price
                    pools.append(LiquidityPool(
                        price_level=swing_high,
                        estimated_size=tests * 10,
                        type='above'
                    ))

        # Sort by proximity to current price and return top pools
        pools.sort(key=lambda x: abs(x.price_level - current_price))
        return pools[:6]  # 3 above, 3 below approximately

    def _detect_absorption(self, df: pd.DataFrame) -> Dict:
        """
        Detect volume absorption patterns.

        Absorption = high volume with small price movement
        Indicates large players absorbing retail orders.
        """
        volume = df['volume'] if 'volume' in df.columns else pd.Series([1] * len(df))
        close = df['close']
        candle_range = df['high'] - df['low']

        # Average metrics
        avg_volume = volume.rolling(20).mean()
        avg_range = candle_range.rolling(20).mean()

        # Current candle analysis
        current_volume = volume.iloc[-1]
        current_range = candle_range.iloc[-1]
        current_avg_vol = avg_volume.iloc[-1]
        current_avg_range = avg_range.iloc[-1]

        if current_avg_vol == 0 or current_avg_range == 0:
            return {'detected': False, 'direction': None, 'strength': 0}

        # Absorption criteria:
        # - Volume > 1.5x average
        # - Range < 0.7x average (small movement despite high volume)
        volume_ratio = current_volume / current_avg_vol
        range_ratio = current_range / current_avg_range

        absorption_detected = volume_ratio > self.absorption_threshold and range_ratio < 0.7

        if absorption_detected:
            # Determine direction based on close position
            close_position = (close.iloc[-1] - df['low'].iloc[-1]) / current_range \
                            if current_range > 0 else 0.5

            if close_position > 0.6:
                # Close near high with absorption = buying absorbed, likely reversal down
                direction = 'sell'
            elif close_position < 0.4:
                # Close near low with absorption = selling absorbed, likely reversal up
                direction = 'buy'
            else:
                direction = None

            strength = volume_ratio * (1 - range_ratio) * 50

            return {
                'detected': True,
                'direction': direction,
                'strength': min(100, strength)
            }

        return {'detected': False, 'direction': None, 'strength': 0}

    def _detect_stop_hunt(self, df: pd.DataFrame) -> Dict:
        """
        Detect stop hunt patterns.

        Stop hunt = price briefly breaks a key level (taking out stops)
        then quickly reverses back.
        """
        high = df['high']
        low = df['low']
        close = df['close']
        open_ = df['open']

        # Calculate ATR for threshold
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]

        # Recent swing levels
        recent_high = high.tail(20).max()
        recent_low = low.tail(20).min()

        # Current candle analysis
        current_high = high.iloc[-1]
        current_low = low.iloc[-1]
        current_close = close.iloc[-1]
        current_open = open_.iloc[-1]

        stop_hunt_detected = False
        direction = None

        # Bear trap (stop hunt long): price breaks below recent low but closes above
        if current_low < recent_low and current_close > recent_low:
            wick_below = recent_low - current_low
            if wick_below > atr * self.stop_hunt_threshold:
                stop_hunt_detected = True
                direction = 'long'

        # Bull trap (stop hunt short): price breaks above recent high but closes below
        if current_high > recent_high and current_close < recent_high:
            wick_above = current_high - recent_high
            if wick_above > atr * self.stop_hunt_threshold:
                stop_hunt_detected = True
                direction = 'short'

        return {
            'detected': stop_hunt_detected,
            'direction': direction
        }

    def _detect_institutional_activity(self, df: pd.DataFrame) -> Dict:
        """
        Detect signs of institutional/large player activity.

        Indicators:
        - Unusual volume spikes
        - Large candles with follow-through
        - Systematic accumulation/distribution patterns
        """
        volume = df['volume'] if 'volume' in df.columns else pd.Series([1] * len(df))
        close = df['close']
        candle_range = df['high'] - df['low']

        # Volume analysis
        avg_volume = volume.rolling(20).mean()
        volume_std = volume.rolling(20).std()

        current_volume = volume.iloc[-1]
        current_avg = avg_volume.iloc[-1]
        current_std = volume_std.iloc[-1]

        if current_avg == 0:
            return {'buying': False, 'selling': False, 'activity_score': 0}

        # Z-score of current volume
        volume_zscore = (current_volume - current_avg) / current_std if current_std > 0 else 0

        # Large volume = institutional activity
        institutional_volume = volume_zscore > 2

        # Determine direction
        price_change = close.iloc[-1] - close.iloc[-5]

        buying = institutional_volume and price_change > 0
        selling = institutional_volume and price_change < 0

        # Activity score
        activity_score = min(100, abs(volume_zscore) * 25) if volume_zscore > 1 else 0

        return {
            'buying': buying,
            'selling': selling,
            'activity_score': round(activity_score, 1)
        }

    def _build_volume_profile(self, df: pd.DataFrame) -> Dict:
        """
        Build volume profile and find POC (Point of Control).

        Volume profile shows where most trading occurred.
        """
        volume = df['volume'] if 'volume' in df.columns else pd.Series([1] * len(df))
        high = df['high']
        low = df['low']
        close = df['close']

        # Price range
        price_high = high.max()
        price_low = low.min()
        price_range = price_high - price_low

        if price_range == 0:
            return {
                'poc': close.iloc[-1],
                'vah': close.iloc[-1],
                'val': close.iloc[-1],
                'node_type': 'neutral'
            }

        # Create price bins
        bin_size = price_range / self.num_price_bins
        bins = np.arange(price_low, price_high + bin_size, bin_size)

        # Distribute volume across bins
        volume_at_price = np.zeros(len(bins) - 1)

        for i in range(len(df)):
            candle_low = low.iloc[i]
            candle_high = high.iloc[i]
            candle_volume = volume.iloc[i]

            # Distribute volume evenly across price range of candle
            for j in range(len(bins) - 1):
                bin_low = bins[j]
                bin_high = bins[j + 1]

                # Check overlap
                overlap_low = max(candle_low, bin_low)
                overlap_high = min(candle_high, bin_high)

                if overlap_high > overlap_low:
                    overlap_pct = (overlap_high - overlap_low) / (candle_high - candle_low) \
                                  if candle_high > candle_low else 0
                    volume_at_price[j] += candle_volume * overlap_pct

        # Find POC (highest volume price)
        poc_index = np.argmax(volume_at_price)
        poc = (bins[poc_index] + bins[poc_index + 1]) / 2

        # Calculate Value Area (70% of volume)
        total_volume = volume_at_price.sum()
        target_volume = total_volume * self.value_area_pct

        # Start from POC and expand
        current_volume = volume_at_price[poc_index]
        val_index = poc_index
        vah_index = poc_index

        while current_volume < target_volume:
            # Expand to the side with more volume
            below_vol = volume_at_price[val_index - 1] if val_index > 0 else 0
            above_vol = volume_at_price[vah_index + 1] if vah_index < len(volume_at_price) - 1 else 0

            if above_vol >= below_vol and vah_index < len(volume_at_price) - 1:
                vah_index += 1
                current_volume += above_vol
            elif val_index > 0:
                val_index -= 1
                current_volume += below_vol
            else:
                break

        val = bins[val_index]
        vah = bins[vah_index + 1]

        # Determine if current price is at high or low volume node
        current_price = close.iloc[-1]
        current_bin = int((current_price - price_low) / bin_size)
        current_bin = min(current_bin, len(volume_at_price) - 1)

        avg_volume_at_price = volume_at_price.mean()
        node_type = 'high_volume' if volume_at_price[current_bin] > avg_volume_at_price else 'low_volume'

        return {
            'poc': poc,
            'vah': vah,
            'val': val,
            'node_type': node_type,
            'profile': volume_at_price,
            'bins': bins
        }

    def _determine_signal(
        self,
        delta: Dict,
        pressure: Dict,
        absorption: Dict,
        stop_hunt: Dict,
        institutional: Dict
    ) -> Tuple[OrderFlowSignal, float]:
        """Determine primary order flow signal and strength."""

        strength = 50  # Base strength

        # Stop hunt has highest priority
        if stop_hunt['detected']:
            if stop_hunt['direction'] == 'long':
                return OrderFlowSignal.STOP_HUNT_LONG, 85
            else:
                return OrderFlowSignal.STOP_HUNT_SHORT, 85

        # Absorption second priority
        if absorption['detected'] and absorption['direction']:
            strength = 70 + absorption['strength'] * 0.2
            if absorption['direction'] == 'buy':
                return OrderFlowSignal.ABSORPTION_BUY, min(90, strength)
            else:
                return OrderFlowSignal.ABSORPTION_SELL, min(90, strength)

        # Institutional activity
        if institutional['activity_score'] > 50:
            strength = 65 + institutional['activity_score'] * 0.25
            if institutional['buying']:
                return OrderFlowSignal.STRONG_BUYING, min(85, strength)
            elif institutional['selling']:
                return OrderFlowSignal.STRONG_SELLING, min(85, strength)

        # Delta and pressure based signals
        net_pressure = pressure['net']

        if net_pressure > 30:
            if delta['delta'] > 0:
                strength = 60 + net_pressure * 0.3
                return OrderFlowSignal.STRONG_BUYING, min(80, strength)
            else:
                strength = 55 + net_pressure * 0.2
                return OrderFlowSignal.MODERATE_BUYING, min(70, strength)

        elif net_pressure < -30:
            if delta['delta'] < 0:
                strength = 60 + abs(net_pressure) * 0.3
                return OrderFlowSignal.STRONG_SELLING, min(80, strength)
            else:
                strength = 55 + abs(net_pressure) * 0.2
                return OrderFlowSignal.MODERATE_SELLING, min(70, strength)

        return OrderFlowSignal.NEUTRAL, 50

    def _calculate_entry_quality(
        self,
        delta: Dict,
        pressure: Dict,
        absorption: Dict,
        stop_hunt: Dict,
        imbalances: List[ImbalanceZone],
        signal_strength: float
    ) -> float:
        """Calculate overall entry quality score (0-100)."""

        quality = signal_strength * 0.4  # Base from signal strength

        # Delta confirmation
        if abs(delta['delta']) > 0:
            quality += 10

        # No divergence is good
        if not delta['divergence']:
            quality += 10
        else:
            quality -= 10

        # Clear pressure direction
        if abs(pressure['net']) > 40:
            quality += 10

        # Absorption or stop hunt = high quality
        if absorption['detected']:
            quality += 15
        if stop_hunt['detected']:
            quality += 20

        # Near unfilled imbalance = good entry
        if any(not z.filled for z in imbalances):
            quality += 10

        return max(0, min(100, quality))

    def _get_recommendation(
        self,
        signal: OrderFlowSignal,
        entry_quality: float,
        delta: Dict
    ) -> str:
        """Generate trade recommendation based on analysis."""

        if entry_quality < 50:
            return "WAIT"

        if signal in [OrderFlowSignal.STOP_HUNT_LONG, OrderFlowSignal.ABSORPTION_BUY,
                     OrderFlowSignal.STRONG_BUYING, OrderFlowSignal.IMBALANCE_BUY]:
            if entry_quality >= 70:
                return "STRONG_BUY"
            else:
                return "BUY"

        if signal in [OrderFlowSignal.STOP_HUNT_SHORT, OrderFlowSignal.ABSORPTION_SELL,
                     OrderFlowSignal.STRONG_SELLING, OrderFlowSignal.IMBALANCE_SELL]:
            if entry_quality >= 70:
                return "STRONG_SELL"
            else:
                return "SELL"

        if signal == OrderFlowSignal.MODERATE_BUYING:
            return "LEAN_BUY"
        if signal == OrderFlowSignal.MODERATE_SELLING:
            return "LEAN_SELL"

        return "WAIT"

    def _neutral_analysis(self, pair: str, timeframe: str) -> OrderFlowAnalysis:
        """Return neutral analysis when data is insufficient."""
        return OrderFlowAnalysis(
            pair=pair,
            timeframe=timeframe,
            timestamp=datetime.now(),
            delta=0,
            cumulative_delta=0,
            delta_divergence=False,
            buying_pressure=50,
            selling_pressure=50,
            net_pressure=0,
            primary_signal=OrderFlowSignal.NEUTRAL,
            signal_strength=50,
            imbalance_zones=[],
            liquidity_pools=[],
            institutional_buying=False,
            institutional_selling=False,
            large_player_activity=0,
            poc=0,
            value_area_high=0,
            value_area_low=0,
            volume_node_type='neutral',
            absorption_detected=False,
            absorption_direction=None,
            stop_hunt_detected=False,
            stop_hunt_direction=None,
            entry_quality=50,
            recommended_action="WAIT"
        )

    def get_entry_timing(
        self,
        analysis: OrderFlowAnalysis,
        direction: str
    ) -> Dict:
        """
        Get specific entry timing recommendations.

        Args:
            analysis: OrderFlowAnalysis result
            direction: Intended trade direction ('BUY' or 'SELL')

        Returns:
            Dict with timing recommendation and confidence
        """
        confidence = 50
        timing = "WAIT"
        reason = []

        if direction == "BUY":
            # Best buy entry conditions
            if analysis.stop_hunt_direction == 'long':
                confidence += 25
                timing = "ENTER_NOW"
                reason.append("Stop hunt reversal detected")

            if analysis.absorption_direction == 'buy':
                confidence += 20
                timing = "ENTER_NOW"
                reason.append("Selling absorbed")

            if analysis.buying_pressure > 60:
                confidence += 15
                reason.append(f"Strong buying pressure ({analysis.buying_pressure}%)")

            if analysis.delta > 0:
                confidence += 10
                reason.append("Positive delta")

            if analysis.net_pressure > 20:
                timing = "ENTER_SOON" if timing == "WAIT" else timing
                confidence += 5

        else:  # SELL
            if analysis.stop_hunt_direction == 'short':
                confidence += 25
                timing = "ENTER_NOW"
                reason.append("Stop hunt reversal detected")

            if analysis.absorption_direction == 'sell':
                confidence += 20
                timing = "ENTER_NOW"
                reason.append("Buying absorbed")

            if analysis.selling_pressure > 60:
                confidence += 15
                reason.append(f"Strong selling pressure ({analysis.selling_pressure}%)")

            if analysis.delta < 0:
                confidence += 10
                reason.append("Negative delta")

            if analysis.net_pressure < -20:
                timing = "ENTER_SOON" if timing == "WAIT" else timing
                confidence += 5

        return {
            'timing': timing,
            'confidence': min(100, confidence),
            'reasons': reason
        }


# Convenience function
def analyze_order_flow(df: pd.DataFrame, pair: str, timeframe: str = "M15") -> OrderFlowAnalysis:
    """
    Analyze order flow for a trading pair.

    Args:
        df: OHLCV DataFrame
        pair: Trading pair
        timeframe: Timeframe string

    Returns:
        OrderFlowAnalysis result
    """
    analyzer = OrderFlowAnalyzer()
    return analyzer.analyze(df, pair, timeframe)


if __name__ == "__main__":
    print("Order Flow Analyzer - Forex Scalper Agent V2")
    print("=" * 50)
    print("\nFeatures:")
    print("  - Delta analysis (buying vs selling pressure)")
    print("  - Absorption detection")
    print("  - Imbalance zone identification")
    print("  - Liquidity pool detection")
    print("  - Stop hunt identification")
    print("  - Institutional footprint detection")
    print("  - Volume profile analysis")
    print("\nUsage:")
    print("  analyzer = OrderFlowAnalyzer()")
    print("  result = analyzer.analyze(df, 'EURUSD', 'M15')")
    print("  print(result.recommended_action)")
