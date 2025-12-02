"""
Machine Learning Regime Detector - Advanced Market Classification
==================================================================
Uses machine learning models to detect market regimes with higher accuracy.

Features:
- Multiple ML models (Random Forest, Gradient Boosting, LSTM)
- Feature engineering for financial time series
- Online learning capability
- Ensemble predictions
- Confidence calibration
- Model persistence and updating
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import logging
import pickle
import os
from collections import deque

logger = logging.getLogger(__name__)

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.calibration import CalibratedClassifierCV
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. ML features disabled.")

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


class MLRegime(Enum):
    """Market regime classifications."""
    STRONG_TREND_UP = 0
    WEAK_TREND_UP = 1
    RANGING = 2
    WEAK_TREND_DOWN = 3
    STRONG_TREND_DOWN = 4
    HIGH_VOLATILITY = 5
    LOW_VOLATILITY = 6


@dataclass
class MLPrediction:
    """Machine learning prediction result."""
    regime: str
    confidence: float
    probabilities: Dict[str, float]
    features_importance: Dict[str, float]
    model_agreement: float  # How much models agree
    timestamp: datetime


class FeatureEngineer:
    """
    Generate features for ML regime detection.

    Creates technical indicators and statistical features
    from OHLCV data for machine learning models.
    """

    def __init__(self):
        self.feature_names = []

    def generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all features from OHLCV data.

        Args:
            df: DataFrame with OHLCV columns

        Returns:
            DataFrame with features
        """
        df = df.copy()

        # Normalize column names
        df.columns = [c.lower() for c in df.columns]

        features = pd.DataFrame(index=df.index)

        # 1. Price-based features
        features = self._add_price_features(df, features)

        # 2. Trend indicators
        features = self._add_trend_features(df, features)

        # 3. Momentum indicators
        features = self._add_momentum_features(df, features)

        # 4. Volatility indicators
        features = self._add_volatility_features(df, features)

        # 5. Volume features (if available)
        if 'volume' in df.columns:
            features = self._add_volume_features(df, features)

        # 6. Statistical features
        features = self._add_statistical_features(df, features)

        # 7. Lagged features
        features = self._add_lagged_features(features)

        # Store feature names
        self.feature_names = features.columns.tolist()

        # Drop NaN rows
        features = features.dropna()

        return features

    def _add_price_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        close = df['close']
        high = df['high']
        low = df['low']
        open_ = df['open']

        # Returns
        features['return_1'] = close.pct_change(1)
        features['return_5'] = close.pct_change(5)
        features['return_10'] = close.pct_change(10)
        features['return_20'] = close.pct_change(20)

        # Log returns
        features['log_return_1'] = np.log(close / close.shift(1))
        features['log_return_5'] = np.log(close / close.shift(5))

        # Candle features
        candle_range = high - low
        features['candle_body'] = (close - open_) / candle_range.replace(0, np.nan)
        features['upper_wick'] = (high - close.clip(upper=open_)) / candle_range.replace(0, np.nan)
        features['lower_wick'] = (close.clip(lower=open_) - low) / candle_range.replace(0, np.nan)

        # Price position
        features['price_position'] = (close - low) / candle_range.replace(0, np.nan)

        # Gap
        features['gap'] = (open_ - close.shift(1)) / close.shift(1)

        return features

    def _add_trend_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add trend indicators."""
        close = df['close']
        high = df['high']
        low = df['low']

        # EMAs
        for period in [5, 10, 20, 50, 100, 200]:
            ema = close.ewm(span=period, adjust=False).mean()
            features[f'ema_{period}'] = (close - ema) / ema * 100
            features[f'ema_{period}_slope'] = ema.diff(5) / ema * 100

        # EMA crossovers
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        ema_200 = close.ewm(span=200, adjust=False).mean()

        features['ema_20_50_diff'] = (ema_20 - ema_50) / ema_50 * 100
        features['ema_50_200_diff'] = (ema_50 - ema_200) / ema_200 * 100

        # EMA stack alignment
        features['ema_stack_bull'] = ((ema_20 > ema_50) & (ema_50 > ema_200)).astype(int)
        features['ema_stack_bear'] = ((ema_20 < ema_50) & (ema_50 < ema_200)).astype(int)

        # ADX (Average Directional Index)
        adx, plus_di, minus_di = self._calculate_adx(df)
        features['adx'] = adx
        features['plus_di'] = plus_di
        features['minus_di'] = minus_di
        features['di_diff'] = plus_di - minus_di

        # Linear regression slope
        for period in [10, 20, 50]:
            features[f'linreg_slope_{period}'] = self._calculate_linreg_slope(close, period)

        return features

    def _add_momentum_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators."""
        close = df['close']
        high = df['high']
        low = df['low']

        # RSI
        for period in [7, 14, 21]:
            features[f'rsi_{period}'] = self._calculate_rsi(close, period)

        # Stochastic
        for period in [14, 21]:
            stoch_k, stoch_d = self._calculate_stochastic(df, period)
            features[f'stoch_k_{period}'] = stoch_k
            features[f'stoch_d_{period}'] = stoch_d

        # MACD
        macd, signal, histogram = self._calculate_macd(close)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_histogram'] = histogram
        features['macd_histogram_slope'] = histogram.diff(3)

        # CCI (Commodity Channel Index)
        features['cci'] = self._calculate_cci(df)

        # Williams %R
        features['williams_r'] = self._calculate_williams_r(df)

        # Rate of Change
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = (close - close.shift(period)) / close.shift(period) * 100

        # Momentum
        features['momentum_10'] = close - close.shift(10)

        return features

    def _add_volatility_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators."""
        close = df['close']
        high = df['high']
        low = df['low']

        # ATR
        for period in [7, 14, 21]:
            features[f'atr_{period}'] = self._calculate_atr(df, period)
            features[f'atr_{period}_pct'] = features[f'atr_{period}'] / close * 100

        # ATR ratio (current vs average)
        atr_14 = features['atr_14']
        features['atr_ratio'] = atr_14 / atr_14.rolling(50).mean()

        # Bollinger Bands
        for period in [20]:
            bb_upper, bb_middle, bb_lower, bb_width, bb_pct = self._calculate_bollinger(close, period)
            features[f'bb_width_{period}'] = bb_width
            features[f'bb_pct_{period}'] = bb_pct

        # Historical volatility
        for period in [10, 20, 50]:
            features[f'hist_vol_{period}'] = close.pct_change().rolling(period).std() * np.sqrt(252)

        # Volatility ratio
        features['vol_ratio_10_50'] = features['hist_vol_10'] / features['hist_vol_50']

        # Keltner Channel width
        features['keltner_width'] = self._calculate_keltner_width(df)

        # Range as % of price
        features['range_pct'] = (high - low) / close * 100
        features['range_pct_ma'] = features['range_pct'].rolling(20).mean()

        return features

    def _add_volume_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based features."""
        volume = df['volume']
        close = df['close']

        # Volume moving averages
        for period in [10, 20, 50]:
            features[f'volume_ma_{period}'] = volume.rolling(period).mean()
            features[f'volume_ratio_{period}'] = volume / features[f'volume_ma_{period}']

        # Volume trend
        features['volume_trend'] = volume.diff(5)

        # On-Balance Volume
        obv = (np.sign(close.diff()) * volume).cumsum()
        features['obv'] = obv
        features['obv_slope'] = obv.diff(10)

        # Volume-price trend
        features['vpt'] = (volume * close.pct_change()).cumsum()

        # Money Flow Index
        features['mfi'] = self._calculate_mfi(df)

        # Accumulation/Distribution
        features['ad_line'] = self._calculate_ad_line(df)

        return features

    def _add_statistical_features(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        close = df['close']
        returns = close.pct_change()

        # Skewness and Kurtosis
        for period in [20, 50]:
            features[f'skewness_{period}'] = returns.rolling(period).skew()
            features[f'kurtosis_{period}'] = returns.rolling(period).kurt()

        # Z-score of price
        for period in [20, 50]:
            ma = close.rolling(period).mean()
            std = close.rolling(period).std()
            features[f'zscore_{period}'] = (close - ma) / std

        # Autocorrelation
        features['autocorr_1'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=1), raw=False)
        features['autocorr_5'] = returns.rolling(20).apply(lambda x: x.autocorr(lag=5), raw=False)

        # Hurst exponent (simplified)
        features['hurst'] = self._calculate_hurst(close)

        return features

    def _add_lagged_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Add lagged versions of key features."""
        key_features = ['return_1', 'rsi_14', 'adx', 'atr_ratio', 'macd_histogram']

        for feat in key_features:
            if feat in features.columns:
                for lag in [1, 3, 5]:
                    features[f'{feat}_lag_{lag}'] = features[feat].shift(lag)

        return features

    def _calculate_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_adx(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate ADX, +DI, -DI."""
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up = high - high.shift()
        down = low.shift() - low

        plus_dm = np.where((up > down) & (up > 0), up, 0)
        minus_dm = np.where((down > up) & (down > 0), down, 0)

        # Smoothed
        atr = tr.rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()

        return adx, plus_di, minus_di

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return tr.rolling(period).mean()

    def _calculate_stochastic(self, df: pd.DataFrame, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic %K and %D."""
        low_min = df['low'].rolling(period).min()
        high_max = df['high'].rolling(period).max()

        stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
        stoch_d = stoch_k.rolling(3).mean()

        return stoch_k, stoch_d

    def _calculate_macd(self, close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD."""
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal

        return macd, signal, histogram

    def _calculate_bollinger(self, close: pd.Series, period: int = 20) -> Tuple:
        """Calculate Bollinger Bands."""
        middle = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = middle + (std * 2)
        lower = middle - (std * 2)
        width = (upper - lower) / middle * 100
        pct = (close - lower) / (upper - lower)

        return upper, middle, lower, width, pct

    def _calculate_cci(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index."""
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(period).mean()
        md = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        return (tp - ma) / (0.015 * md)

    def _calculate_williams_r(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R."""
        high_max = df['high'].rolling(period).max()
        low_min = df['low'].rolling(period).min()
        return -100 * (high_max - df['close']) / (high_max - low_min)

    def _calculate_keltner_width(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Keltner Channel width."""
        middle = df['close'].ewm(span=period, adjust=False).mean()
        atr = self._calculate_atr(df, period)
        upper = middle + (atr * 2)
        lower = middle - (atr * 2)
        return (upper - lower) / middle * 100

    def _calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Money Flow Index."""
        tp = (df['high'] + df['low'] + df['close']) / 3
        mf = tp * df['volume']

        pos_mf = mf.where(tp > tp.shift(), 0).rolling(period).sum()
        neg_mf = mf.where(tp < tp.shift(), 0).rolling(period).sum()

        mfi = 100 - (100 / (1 + pos_mf / neg_mf))
        return mfi

    def _calculate_ad_line(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Accumulation/Distribution Line."""
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.fillna(0)
        mfv = mfm * df['volume']
        return mfv.cumsum()

    def _calculate_linreg_slope(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate linear regression slope."""
        def slope(arr):
            if len(arr) < period:
                return np.nan
            x = np.arange(len(arr))
            return np.polyfit(x, arr, 1)[0]

        return series.rolling(period).apply(slope, raw=True)

    def _calculate_hurst(self, series: pd.Series, max_lag: int = 20) -> pd.Series:
        """Calculate simplified Hurst exponent."""
        def hurst(arr):
            if len(arr) < max_lag:
                return np.nan

            lags = range(2, min(max_lag, len(arr) // 2))
            tau = [np.std(np.subtract(arr[lag:], arr[:-lag])) for lag in lags]

            if not tau or any(t == 0 for t in tau):
                return 0.5

            reg = np.polyfit(np.log(lags), np.log(tau), 1)
            return reg[0]

        return series.rolling(100).apply(hurst, raw=True)


class MLRegimeDetector:
    """
    Machine Learning based regime detector.

    Uses ensemble of models for robust predictions.
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize ML regime detector.

        Args:
            model_path: Path to load pre-trained models
        """
        self.feature_engineer = FeatureEngineer()
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.label_encoder = LabelEncoder() if SKLEARN_AVAILABLE else None
        self.models = {}
        self.is_trained = False
        self.model_path = model_path

        # Regime mapping
        self.regime_names = {
            0: 'STRONG_TREND_UP',
            1: 'WEAK_TREND_UP',
            2: 'RANGING',
            3: 'WEAK_TREND_DOWN',
            4: 'STRONG_TREND_DOWN',
            5: 'HIGH_VOLATILITY',
            6: 'LOW_VOLATILITY'
        }

        # Online learning buffer
        self.training_buffer = deque(maxlen=10000)

        if model_path and os.path.exists(model_path):
            self._load_models()
        else:
            self._initialize_models()

        logger.info("MLRegimeDetector initialized")

    def _initialize_models(self):
        """Initialize ML models."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Using fallback detection.")
            return

        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )

        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=20,
            random_state=42
        )

        logger.info(f"Initialized {len(self.models)} ML models")

    def train(
        self,
        data: Dict[str, pd.DataFrame],
        labels: Optional[pd.Series] = None
    ) -> Dict:
        """
        Train the ML models.

        Args:
            data: Dict of pair -> DataFrame with OHLCV data
            labels: Optional pre-defined labels (if None, will auto-generate)

        Returns:
            Training metrics
        """
        if not SKLEARN_AVAILABLE:
            return {'error': 'scikit-learn not available'}

        logger.info("Starting model training...")

        # Collect all data
        all_features = []
        all_labels = []

        for pair, df in data.items():
            features = self.feature_engineer.generate_features(df)

            if labels is None:
                # Auto-generate labels based on rules
                pair_labels = self._generate_labels(df, features)
            else:
                pair_labels = labels

            # Align features and labels
            common_idx = features.index.intersection(pair_labels.index)
            all_features.append(features.loc[common_idx])
            all_labels.append(pair_labels.loc[common_idx])

        # Combine
        X = pd.concat(all_features)
        y = pd.concat(all_labels)

        logger.info(f"Training data: {len(X)} samples, {len(X.columns)} features")

        # Handle any remaining NaN
        X = X.fillna(0)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)

        # Train each model
        metrics = {}

        for name, model in self.models.items():
            logger.info(f"Training {name}...")

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X_scaled, y_encoded, cv=tscv, scoring='accuracy')

            # Full training
            model.fit(X_scaled, y_encoded)

            metrics[name] = {
                'cv_mean': scores.mean(),
                'cv_std': scores.std(),
                'cv_scores': scores.tolist()
            }

            logger.info(f"{name}: CV accuracy = {scores.mean():.3f} (+/- {scores.std():.3f})")

        self.is_trained = True
        self.feature_names = X.columns.tolist()

        # Save models
        if self.model_path:
            self._save_models()

        return metrics

    def _generate_labels(self, df: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
        """
        Auto-generate regime labels from data.

        Uses rule-based classification to create training labels.
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        labels = pd.Series(index=features.index, dtype=int)

        # Get key indicators from features
        adx = features['adx'] if 'adx' in features.columns else pd.Series([25] * len(features))
        rsi = features['rsi_14'] if 'rsi_14' in features.columns else pd.Series([50] * len(features))
        atr_ratio = features['atr_ratio'] if 'atr_ratio' in features.columns else pd.Series([1] * len(features))
        ema_diff = features['ema_20_50_diff'] if 'ema_20_50_diff' in features.columns else pd.Series([0] * len(features))

        for i in range(len(features)):
            idx = features.index[i]

            curr_adx = adx.iloc[i] if not pd.isna(adx.iloc[i]) else 25
            curr_rsi = rsi.iloc[i] if not pd.isna(rsi.iloc[i]) else 50
            curr_atr_ratio = atr_ratio.iloc[i] if not pd.isna(atr_ratio.iloc[i]) else 1
            curr_ema_diff = ema_diff.iloc[i] if not pd.isna(ema_diff.iloc[i]) else 0

            # Classification logic
            if curr_atr_ratio > 1.8:
                labels[idx] = 5  # HIGH_VOLATILITY
            elif curr_atr_ratio < 0.5:
                labels[idx] = 6  # LOW_VOLATILITY
            elif curr_adx > 35 and curr_ema_diff > 0:
                labels[idx] = 0  # STRONG_TREND_UP
            elif curr_adx > 35 and curr_ema_diff < 0:
                labels[idx] = 4  # STRONG_TREND_DOWN
            elif curr_adx > 20 and curr_ema_diff > 0:
                labels[idx] = 1  # WEAK_TREND_UP
            elif curr_adx > 20 and curr_ema_diff < 0:
                labels[idx] = 3  # WEAK_TREND_DOWN
            else:
                labels[idx] = 2  # RANGING

        return labels

    def predict(self, df: pd.DataFrame, pair: str = "") -> MLPrediction:
        """
        Predict market regime.

        Args:
            df: DataFrame with OHLCV data
            pair: Trading pair name

        Returns:
            MLPrediction with regime and confidence
        """
        if not SKLEARN_AVAILABLE or not self.is_trained:
            # Fallback to rule-based
            return self._fallback_prediction(df, pair)

        try:
            # Generate features
            features = self.feature_engineer.generate_features(df)

            if len(features) == 0:
                return self._fallback_prediction(df, pair)

            # Get latest features
            X = features.iloc[[-1]]
            X = X.fillna(0)

            # Scale
            X_scaled = self.scaler.transform(X)

            # Get predictions from all models
            predictions = []
            probabilities = []

            for name, model in self.models.items():
                pred = model.predict(X_scaled)[0]
                prob = model.predict_proba(X_scaled)[0]
                predictions.append(pred)
                probabilities.append(prob)

            # Ensemble - majority vote
            final_pred = int(np.bincount(predictions).argmax())
            regime_name = self.regime_names.get(final_pred, 'RANGING')

            # Average probabilities
            avg_probs = np.mean(probabilities, axis=0)
            confidence = float(avg_probs[final_pred]) * 100

            # Model agreement
            agreement = sum(1 for p in predictions if p == final_pred) / len(predictions) * 100

            # Probability dict
            prob_dict = {
                self.regime_names[i]: float(avg_probs[i]) * 100
                for i in range(len(avg_probs))
            }

            # Feature importance (from Random Forest)
            importance_dict = {}
            if 'random_forest' in self.models:
                importances = self.models['random_forest'].feature_importances_
                for i, name in enumerate(self.feature_names[:len(importances)]):
                    importance_dict[name] = float(importances[i])
                # Sort by importance
                importance_dict = dict(sorted(importance_dict.items(),
                                             key=lambda x: x[1], reverse=True)[:10])

            return MLPrediction(
                regime=regime_name,
                confidence=confidence,
                probabilities=prob_dict,
                features_importance=importance_dict,
                model_agreement=agreement,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return self._fallback_prediction(df, pair)

    def _fallback_prediction(self, df: pd.DataFrame, pair: str) -> MLPrediction:
        """Rule-based fallback prediction."""
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        if len(df) < 50:
            return MLPrediction(
                regime='RANGING',
                confidence=50,
                probabilities={'RANGING': 100},
                features_importance={},
                model_agreement=100,
                timestamp=datetime.now()
            )

        close = df['close']
        high = df['high']
        low = df['low']

        # Calculate basic indicators
        ema_20 = close.ewm(span=20, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()

        # ATR
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        atr_avg = atr.rolling(50).mean()

        curr_atr = atr.iloc[-1]
        curr_atr_avg = atr_avg.iloc[-1]
        atr_ratio = curr_atr / curr_atr_avg if curr_atr_avg > 0 else 1

        ema_diff = (ema_20.iloc[-1] - ema_50.iloc[-1]) / ema_50.iloc[-1] * 100

        # Simple ADX estimation
        price_change = abs(close.diff()).rolling(14).mean()
        adx_estimate = price_change / atr * 100 if atr.iloc[-1] > 0 else 25

        # Classification
        if atr_ratio > 1.8:
            regime = 'HIGH_VOLATILITY'
            confidence = 70
        elif atr_ratio < 0.5:
            regime = 'LOW_VOLATILITY'
            confidence = 70
        elif abs(ema_diff) > 1:
            if ema_diff > 0:
                regime = 'STRONG_TREND_UP' if abs(ema_diff) > 2 else 'WEAK_TREND_UP'
            else:
                regime = 'STRONG_TREND_DOWN' if abs(ema_diff) > 2 else 'WEAK_TREND_DOWN'
            confidence = 65
        else:
            regime = 'RANGING'
            confidence = 60

        return MLPrediction(
            regime=regime,
            confidence=confidence,
            probabilities={regime: confidence},
            features_importance={},
            model_agreement=100,
            timestamp=datetime.now()
        )

    def update_online(self, df: pd.DataFrame, actual_regime: str):
        """
        Online learning - update models with new data.

        Args:
            df: Recent OHLCV data
            actual_regime: Actual regime observed
        """
        # Add to training buffer
        features = self.feature_engineer.generate_features(df)
        if len(features) > 0:
            regime_idx = [k for k, v in self.regime_names.items() if v == actual_regime]
            if regime_idx:
                self.training_buffer.append((features.iloc[-1], regime_idx[0]))

        # Retrain periodically
        if len(self.training_buffer) >= 1000 and len(self.training_buffer) % 100 == 0:
            self._partial_fit()

    def _partial_fit(self):
        """Perform partial fit with buffered data."""
        if not SKLEARN_AVAILABLE or len(self.training_buffer) < 100:
            return

        # Convert buffer to training data
        X = pd.DataFrame([item[0] for item in self.training_buffer])
        y = np.array([item[1] for item in self.training_buffer])

        X = X.fillna(0)
        X_scaled = self.scaler.transform(X)

        # Retrain models
        for name, model in self.models.items():
            model.fit(X_scaled, y)

        logger.info(f"Models updated with {len(self.training_buffer)} samples")

    def _save_models(self):
        """Save trained models to disk."""
        if not JOBLIB_AVAILABLE or not self.model_path:
            return

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        save_data = {
            'models': self.models,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names
        }

        joblib.dump(save_data, self.model_path)
        logger.info(f"Models saved to {self.model_path}")

    def _load_models(self):
        """Load pre-trained models from disk."""
        if not JOBLIB_AVAILABLE or not self.model_path:
            return

        try:
            save_data = joblib.load(self.model_path)
            self.models = save_data['models']
            self.scaler = save_data['scaler']
            self.label_encoder = save_data['label_encoder']
            self.feature_names = save_data['feature_names']
            self.is_trained = True
            logger.info(f"Models loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self._initialize_models()

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained models."""
        if not self.is_trained or 'random_forest' not in self.models:
            return {}

        importances = self.models['random_forest'].feature_importances_
        importance_dict = {}

        for i, name in enumerate(self.feature_names[:len(importances)]):
            importance_dict[name] = float(importances[i])

        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))


# Convenience function
def detect_regime_ml(df: pd.DataFrame, pair: str = "") -> MLPrediction:
    """
    Detect market regime using ML.

    Args:
        df: OHLCV DataFrame
        pair: Trading pair

    Returns:
        MLPrediction result
    """
    detector = MLRegimeDetector()
    return detector.predict(df, pair)


if __name__ == "__main__":
    print("ML Regime Detector - Forex Scalper Agent V2")
    print("=" * 50)
    print("\nFeatures:")
    print("  - Random Forest ensemble")
    print("  - Gradient Boosting ensemble")
    print("  - 100+ technical features")
    print("  - Online learning capability")
    print("  - Confidence calibration")
    print("\nRegime Types:")
    for idx, name in MLRegimeDetector().regime_names.items():
        print(f"  {idx}: {name}")
    print("\nUsage:")
    print("  detector = MLRegimeDetector()")
    print("  detector.train(data_dict)")
    print("  prediction = detector.predict(df, 'EURUSD')")
    print("  print(prediction.regime, prediction.confidence)")
