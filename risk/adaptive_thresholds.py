"""Adaptive Thresholds System - Dynamic parameter adjustment based on pair, session, and market conditions"""
from datetime import datetime, timezone
from typing import Dict, Tuple, Optional
import pandas as pd

# ============================================================================
# PART 1: PAIR-SPECIFIC PROFILES
# ============================================================================

PAIR_PROFILES = {
    # MAJOR PAIRS - Low volatility, tight spreads, high liquidity
    "EURUSD": {
        "volatility_class": "low",
        "base_confidence": 65,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "min_rr_ratio": 1.5,
        "typical_atr_m15": 0.0008,
        "typical_spread": 0.5,
        "atr_sl_multiplier": 1.5,
        "avg_spread_pips": 0.8,
        "risk_multiplier": 1.0
    },
    "USDCHF": {
        "volatility_class": "low",
        "base_confidence": 65,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "min_rr_ratio": 1.5,
        "typical_atr_m15": 0.0009,
        "typical_spread": 0.8,
        "atr_sl_multiplier": 1.5,
        "avg_spread_pips": 1.2,
        "risk_multiplier": 1.0
    },
    "USDJPY": {
        "volatility_class": "low",
        "base_confidence": 63,
        "rsi_overbought": 71,
        "rsi_oversold": 29,
        "min_rr_ratio": 1.6,
        "typical_atr_m15": 0.08,
        "typical_spread": 0.5,
        "atr_sl_multiplier": 1.5,
        "avg_spread_pips": 1.0,
        "risk_multiplier": 1.0
    },
    # MINOR MAJORS - Moderate volatility
    "GBPUSD": {
        "volatility_class": "medium",
        "base_confidence": 60,
        "rsi_overbought": 72,
        "rsi_oversold": 28,
        "min_rr_ratio": 1.8,
        "typical_atr_m15": 0.0012,
        "typical_spread": 0.8,
        "atr_sl_multiplier": 1.6,
        "avg_spread_pips": 1.5,
        "risk_multiplier": 0.9
    },
    "AUDUSD": {
        "volatility_class": "medium",
        "base_confidence": 60,
        "rsi_overbought": 72,
        "rsi_oversold": 28,
        "min_rr_ratio": 1.7,
        "typical_atr_m15": 0.0010,
        "typical_spread": 0.7,
        "atr_sl_multiplier": 1.5,
        "avg_spread_pips": 1.2,
        "risk_multiplier": 0.9
    },
    "USDCAD": {
        "volatility_class": "medium",
        "base_confidence": 62,
        "rsi_overbought": 71,
        "rsi_oversold": 29,
        "min_rr_ratio": 1.6,
        "typical_atr_m15": 0.0010,
        "typical_spread": 0.9,
        "atr_sl_multiplier": 1.5,
        "avg_spread_pips": 1.5,
        "risk_multiplier": 1.0
    },
    "NZDUSD": {
        "volatility_class": "medium",
        "base_confidence": 60,
        "rsi_overbought": 72,
        "rsi_oversold": 28,
        "min_rr_ratio": 1.7,
        "typical_atr_m15": 0.0009,
        "typical_spread": 1.0,
        "atr_sl_multiplier": 1.5,
        "avg_spread_pips": 1.8,
        "risk_multiplier": 0.9
    },
    # EUR CROSSES
    "EURJPY": {
        "volatility_class": "medium-high",
        "base_confidence": 58,
        "rsi_overbought": 73,
        "rsi_oversold": 27,
        "min_rr_ratio": 1.9,
        "typical_atr_m15": 0.12,
        "typical_spread": 1.2,
        "atr_sl_multiplier": 1.7,
        "avg_spread_pips": 1.5,
        "risk_multiplier": 0.85
    },
    "EURGBP": {
        "volatility_class": "low",
        "base_confidence": 66,
        "rsi_overbought": 69,
        "rsi_oversold": 31,
        "min_rr_ratio": 1.4,
        "typical_atr_m15": 0.0006,
        "typical_spread": 1.0,
        "atr_sl_multiplier": 1.4,
        "avg_spread_pips": 1.2,
        "risk_multiplier": 1.0
    },
    "EURCHF": {
        "volatility_class": "low",
        "base_confidence": 68,
        "rsi_overbought": 68,
        "rsi_oversold": 32,
        "min_rr_ratio": 1.3,
        "typical_atr_m15": 0.0005,
        "typical_spread": 1.5,
        "atr_sl_multiplier": 1.4,
        "avg_spread_pips": 2.0,
        "risk_multiplier": 0.95
    },
    "EURAUD": {
        "volatility_class": "medium",
        "base_confidence": 59,
        "rsi_overbought": 72,
        "rsi_oversold": 28,
        "min_rr_ratio": 1.8,
        "typical_atr_m15": 0.0014,
        "typical_spread": 1.5,
        "atr_sl_multiplier": 1.6,
        "avg_spread_pips": 2.5,
        "risk_multiplier": 0.85
    },
    "EURCAD": {
        "volatility_class": "medium",
        "base_confidence": 61,
        "rsi_overbought": 71,
        "rsi_oversold": 29,
        "min_rr_ratio": 1.7,
        "typical_atr_m15": 0.0012,
        "typical_spread": 1.8,
        "atr_sl_multiplier": 1.6,
        "avg_spread_pips": 2.5,
        "risk_multiplier": 0.85
    },
    # GBP CROSSES - High volatility
    "GBPJPY": {
        "volatility_class": "high",
        "base_confidence": 55,
        "rsi_overbought": 75,
        "rsi_oversold": 25,
        "min_rr_ratio": 2.0,
        "typical_atr_m15": 0.18,
        "typical_spread": 2.0,
        "atr_sl_multiplier": 1.8,
        "avg_spread_pips": 2.5,
        "risk_multiplier": 0.75
    },
    "GBPAUD": {
        "volatility_class": "high",
        "base_confidence": 56,
        "rsi_overbought": 74,
        "rsi_oversold": 26,
        "min_rr_ratio": 2.0,
        "typical_atr_m15": 0.0016,
        "typical_spread": 2.5,
        "atr_sl_multiplier": 1.8,
        "avg_spread_pips": 3.0,
        "risk_multiplier": 0.75
    },
    "GBPCAD": {
        "volatility_class": "high",
        "base_confidence": 57,
        "rsi_overbought": 73,
        "rsi_oversold": 27,
        "min_rr_ratio": 1.9,
        "typical_atr_m15": 0.0015,
        "typical_spread": 2.2,
        "atr_sl_multiplier": 1.7,
        "avg_spread_pips": 3.0,
        "risk_multiplier": 0.8
    },
    "GBPCHF": {
        "volatility_class": "high",
        "base_confidence": 57,
        "rsi_overbought": 73,
        "rsi_oversold": 27,
        "min_rr_ratio": 1.9,
        "typical_atr_m15": 0.0015,
        "typical_spread": 2.5,
        "atr_sl_multiplier": 1.7,
        "avg_spread_pips": 3.0,
        "risk_multiplier": 0.8
    },
    # JPY CROSSES
    "AUDJPY": {
        "volatility_class": "medium-high",
        "base_confidence": 58,
        "rsi_overbought": 73,
        "rsi_oversold": 27,
        "min_rr_ratio": 1.9,
        "typical_atr_m15": 0.11,
        "typical_spread": 1.5,
        "atr_sl_multiplier": 1.7,
        "avg_spread_pips": 2.0,
        "risk_multiplier": 0.85
    },
    "CADJPY": {
        "volatility_class": "medium-high",
        "base_confidence": 59,
        "rsi_overbought": 73,
        "rsi_oversold": 27,
        "min_rr_ratio": 1.8,
        "typical_atr_m15": 0.10,
        "typical_spread": 1.8,
        "atr_sl_multiplier": 1.6,
        "avg_spread_pips": 2.0,
        "risk_multiplier": 0.85
    },
    "CHFJPY": {
        "volatility_class": "medium-high",
        "base_confidence": 59,
        "rsi_overbought": 72,
        "rsi_oversold": 28,
        "min_rr_ratio": 1.8,
        "typical_atr_m15": 0.11,
        "typical_spread": 2.0,
        "atr_sl_multiplier": 1.6,
        "avg_spread_pips": 2.5,
        "risk_multiplier": 0.85
    },
    "NZDJPY": {
        "volatility_class": "medium-high",
        "base_confidence": 58,
        "rsi_overbought": 73,
        "rsi_oversold": 27,
        "min_rr_ratio": 1.9,
        "typical_atr_m15": 0.09,
        "typical_spread": 2.0,
        "atr_sl_multiplier": 1.6,
        "avg_spread_pips": 2.5,
        "risk_multiplier": 0.85
    },
}

# ============================================================================
# PART 2: SESSION DETECTION & PROFILES
# ============================================================================

SESSION_PROFILES = {
    "tokyo": {
        "hours": (0, 9),
        "volatility_mult": 0.7,
        "confidence_adj": +5,
        "session": "tokyo",
        "description": "Tokyo Session - Low volume"
    },
    "london": {
        "hours": (8, 16),
        "volatility_mult": 1.3,
        "confidence_adj": -5,
        "session": "london",
        "description": "London Session - High volume"
    },
    "newyork": {
        "hours": (13, 22),
        "volatility_mult": 1.2,
        "confidence_adj": -3,
        "session": "ny",
        "description": "New York Session"
    },
    "london_ny_overlap": {
        "hours": (13, 16),
        "volatility_mult": 1.5,
        "confidence_adj": -8,
        "session": "london_ny",
        "description": "London/NY Overlap - Maximum volume"
    },
    "quiet": {
        "hours": (22, 24),
        "volatility_mult": 0.5,
        "confidence_adj": +10,
        "session": "quiet",
        "description": "Quiet hours"
    }
}


def get_current_session() -> Tuple[str, Dict]:
    """Detect current trading session based on UTC time."""
    now_utc = datetime.now(timezone.utc)
    current_hour = now_utc.hour

    # Check London/NY overlap first (priority)
    if 13 <= current_hour < 16:
        return "london_ny_overlap", SESSION_PROFILES["london_ny_overlap"]

    # Check other sessions
    for session_name, profile in SESSION_PROFILES.items():
        start, end = profile["hours"]
        if start <= current_hour < end:
            return session_name, profile

    # Default to quiet hours
    return "quiet", SESSION_PROFILES["quiet"]


def get_pair_profile(pair: str) -> Dict:
    """
    Get the profile configuration for a specific trading pair.

    Args:
        pair: Trading pair symbol (e.g., "EURUSD")

    Returns:
        Dict containing pair-specific configuration parameters
    """
    default_profile = {
        "volatility_class": "medium",
        "base_confidence": 60,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "min_rr_ratio": 1.5,
        "typical_atr_m15": 0.0010,
        "typical_spread": 1.0,
        "atr_sl_multiplier": 1.5,
        "avg_spread_pips": 2.0,
        "risk_multiplier": 1.0
    }
    return PAIR_PROFILES.get(pair, default_profile)


def detect_session() -> Dict:
    """
    Detect current trading session based on UTC time.

    Returns:
        Dict with session info
    """
    session_name, profile = get_current_session()
    return {
        'session': profile.get('session', session_name),
        'name': session_name,
        'description': profile.get('description', ''),
        'volatility_mult': profile.get('volatility_mult', 1.0),
        'confidence_adj': profile.get('confidence_adj', 0)
    }


# ============================================================================
# PART 3: VOLATILITY REGIME DETECTION
# ============================================================================

def detect_volatility_regime(df: pd.DataFrame, pair: str) -> Dict:
    """
    Analyze current volatility vs historical average.
    Returns volatility multiplier and regime classification.
    """
    if 'ATR' not in df.columns or len(df) < 50:
        return {"regime": "unknown", "multiplier": 1.0, "current_atr": None}

    current_atr = df['ATR'].iloc[-1]
    avg_atr_50 = df['ATR'].rolling(50).mean().iloc[-1]

    if pd.isna(current_atr) or pd.isna(avg_atr_50) or avg_atr_50 == 0:
        return {"regime": "unknown", "multiplier": 1.0, "current_atr": current_atr}

    volatility_ratio = current_atr / avg_atr_50

    # Classify regime
    if volatility_ratio > 1.5:
        regime = "high_volatility"
        confidence_adj = -10
    elif volatility_ratio > 1.2:
        regime = "elevated_volatility"
        confidence_adj = -5
    elif volatility_ratio < 0.6:
        regime = "low_volatility"
        confidence_adj = +10
    elif volatility_ratio < 0.8:
        regime = "reduced_volatility"
        confidence_adj = +5
    else:
        regime = "normal_volatility"
        confidence_adj = 0

    return {
        "regime": regime,
        "multiplier": volatility_ratio,
        "confidence_adj": confidence_adj,
        "current_atr": current_atr,
        "avg_atr": avg_atr_50
    }


# ============================================================================
# PART 4: ADAPTIVE THRESHOLDS CALCULATOR
# ============================================================================

class AdaptiveThresholds:
    """
    Main class for calculating adaptive thresholds based on:
    - Pair characteristics
    - Current session
    - Volatility regime
    - Spread conditions
    """

    def __init__(self):
        self.pair_profiles = PAIR_PROFILES
        self.session_profiles = SESSION_PROFILES

    def get_thresholds(self, pair: str, df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Calculate adaptive thresholds for a specific pair.

        Args:
            pair: Pair symbol (e.g., 'EURUSD')
            df: DataFrame with price data (optional, for volatility analysis)

        Returns:
            Dict with adjusted thresholds
        """
        pair_clean = pair.replace('=X', '')

        # Get base profile for this pair
        if pair_clean not in self.pair_profiles:
            base_profile = self.pair_profiles["EURUSD"].copy()
        else:
            base_profile = self.pair_profiles[pair_clean].copy()

        # Start with base values
        confidence = base_profile["base_confidence"]
        rsi_ob = base_profile["rsi_overbought"]
        rsi_os = base_profile["rsi_oversold"]
        min_rr = base_profile["min_rr_ratio"]

        # ADJUSTMENT 1: Session-based
        session_name, session_profile = get_current_session()
        confidence += session_profile["confidence_adj"]

        # ADJUSTMENT 2: Volatility regime (if data available)
        volatility_info = {}
        if df is not None:
            volatility_info = detect_volatility_regime(df, pair_clean)
            confidence += volatility_info.get("confidence_adj", 0)

            # Adjust RSI thresholds based on volatility
            if volatility_info["regime"] == "high_volatility":
                rsi_ob += 3
                rsi_os -= 3
            elif volatility_info["regime"] == "low_volatility":
                rsi_ob -= 3
                rsi_os += 3

        # ADJUSTMENT 3: Clamp values to reasonable ranges
        confidence = max(40, min(80, confidence))
        rsi_ob = max(65, min(80, rsi_ob))
        rsi_os = max(20, min(35, rsi_os))
        min_rr = max(1.0, min(3.0, min_rr))

        return {
            "pair": pair_clean,
            "confidence_threshold": confidence,
            "rsi_overbought": rsi_ob,
            "rsi_oversold": rsi_os,
            "min_rr_ratio": min_rr,
            "volatility_class": base_profile["volatility_class"],
            "session": session_name,
            "session_desc": session_profile["description"],
            "volatility_regime": volatility_info.get("regime", "unknown"),
            "volatility_mult": volatility_info.get("multiplier", 1.0),
            "adjustments_applied": {
                "base": base_profile["base_confidence"],
                "session_adj": session_profile["confidence_adj"],
                "volatility_adj": volatility_info.get("confidence_adj", 0),
                "final": confidence
            }
        }


def get_adaptive_thresholds(pair: str, df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Quick function to get adaptive thresholds for a pair.

    Usage:
        thresholds = get_adaptive_thresholds('EURUSD=X', df)
        confidence = thresholds['confidence_threshold']
    """
    calculator = AdaptiveThresholds()
    return calculator.get_thresholds(pair, df)
