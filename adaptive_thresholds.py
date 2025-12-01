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
        "typical_atr_m15": 0.0008,  # ~8 pips
        "typical_spread": 0.5
    },
    "USDCHF": {
        "volatility_class": "low",
        "base_confidence": 65,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "min_rr_ratio": 1.5,
        "typical_atr_m15": 0.0009,
        "typical_spread": 0.8
    },
    "USDJPY": {
        "volatility_class": "low",
        "base_confidence": 63,
        "rsi_overbought": 71,
        "rsi_oversold": 29,
        "min_rr_ratio": 1.6,
        "typical_atr_m15": 0.08,  # JPY pairs
        "typical_spread": 0.5
    },
    
    # MINOR MAJORS - Moderate volatility
    "GBPUSD": {
        "volatility_class": "medium",
        "base_confidence": 60,
        "rsi_overbought": 72,
        "rsi_oversold": 28,
        "min_rr_ratio": 1.8,
        "typical_atr_m15": 0.0012,  # ~12 pips
        "typical_spread": 0.8
    },
    "AUDUSD": {
        "volatility_class": "medium",
        "base_confidence": 60,
        "rsi_overbought": 72,
        "rsi_oversold": 28,
        "min_rr_ratio": 1.7,
        "typical_atr_m15": 0.0010,
        "typical_spread": 0.7
    },
    "USDCAD": {
        "volatility_class": "medium",
        "base_confidence": 62,
        "rsi_overbought": 71,
        "rsi_oversold": 29,
        "min_rr_ratio": 1.6,
        "typical_atr_m15": 0.0010,
        "typical_spread": 0.9
    },
    "NZDUSD": {
        "volatility_class": "medium",
        "base_confidence": 60,
        "rsi_overbought": 72,
        "rsi_oversold": 28,
        "min_rr_ratio": 1.7,
        "typical_atr_m15": 0.0009,
        "typical_spread": 1.0
    },
    
    # EUR CROSSES - Moderate to high
    "EURJPY": {
        "volatility_class": "medium-high",
        "base_confidence": 58,
        "rsi_overbought": 73,
        "rsi_oversold": 27,
        "min_rr_ratio": 1.9,
        "typical_atr_m15": 0.12,
        "typical_spread": 1.2
    },
    "EURGBP": {
        "volatility_class": "low",
        "base_confidence": 66,
        "rsi_overbought": 69,
        "rsi_oversold": 31,
        "min_rr_ratio": 1.4,
        "typical_atr_m15": 0.0006,
        "typical_spread": 1.0
    },
    "EURCHF": {
        "volatility_class": "low",
        "base_confidence": 68,
        "rsi_overbought": 68,
        "rsi_oversold": 32,
        "min_rr_ratio": 1.3,
        "typical_atr_m15": 0.0005,
        "typical_spread": 1.5
    },
    "EURAUD": {
        "volatility_class": "medium",
        "base_confidence": 59,
        "rsi_overbought": 72,
        "rsi_oversold": 28,
        "min_rr_ratio": 1.8,
        "typical_atr_m15": 0.0014,
        "typical_spread": 1.5
    },
    "EURCAD": {
        "volatility_class": "medium",
        "base_confidence": 61,
        "rsi_overbought": 71,
        "rsi_oversold": 29,
        "min_rr_ratio": 1.7,
        "typical_atr_m15": 0.0012,
        "typical_spread": 1.8
    },
    
    # GBP CROSSES - High volatility
    "GBPJPY": {
        "volatility_class": "high",
        "base_confidence": 55,
        "rsi_overbought": 75,
        "rsi_oversold": 25,
        "min_rr_ratio": 2.0,
        "typical_atr_m15": 0.18,
        "typical_spread": 2.0
    },
    "GBPAUD": {
        "volatility_class": "high",
        "base_confidence": 56,
        "rsi_overbought": 74,
        "rsi_oversold": 26,
        "min_rr_ratio": 2.0,
        "typical_atr_m15": 0.0016,
        "typical_spread": 2.5
    },
    "GBPCAD": {
        "volatility_class": "high",
        "base_confidence": 57,
        "rsi_overbought": 73,
        "rsi_oversold": 27,
        "min_rr_ratio": 1.9,
        "typical_atr_m15": 0.0015,
        "typical_spread": 2.2
    },
    "GBPCHF": {
        "volatility_class": "high",
        "base_confidence": 57,
        "rsi_overbought": 73,
        "rsi_oversold": 27,
        "min_rr_ratio": 1.9,
        "typical_atr_m15": 0.0015,
        "typical_spread": 2.5
    },
    
    # JPY CROSSES - High volatility
    "AUDJPY": {
        "volatility_class": "medium-high",
        "base_confidence": 58,
        "rsi_overbought": 73,
        "rsi_oversold": 27,
        "min_rr_ratio": 1.9,
        "typical_atr_m15": 0.11,
        "typical_spread": 1.5
    },
    "CADJPY": {
        "volatility_class": "medium-high",
        "base_confidence": 59,
        "rsi_overbought": 73,
        "rsi_oversold": 27,
        "min_rr_ratio": 1.8,
        "typical_atr_m15": 0.10,
        "typical_spread": 1.8
    },
    "CHFJPY": {
        "volatility_class": "medium-high",
        "base_confidence": 59,
        "rsi_overbought": 72,
        "rsi_oversold": 28,
        "min_rr_ratio": 1.8,
        "typical_atr_m15": 0.11,
        "typical_spread": 2.0
    },
    "NZDJPY": {
        "volatility_class": "medium-high",
        "base_confidence": 58,
        "rsi_overbought": 73,
        "rsi_oversold": 27,
        "min_rr_ratio": 1.9,
        "typical_atr_m15": 0.09,
        "typical_spread": 2.0
    },
}

# ============================================================================
# PART 2: SESSION DETECTION & PROFILES
# ============================================================================

SESSION_PROFILES = {
    "tokyo": {
        "hours": (0, 9),  # 00:00-09:00 UTC
        "volatility_mult": 0.7,
        "confidence_adj": +5,  # Plus strict (moins de mouvements)
        "description": "Tokyo Session - Low volume"
    },
    "london": {
        "hours": (8, 16),  # 08:00-16:00 UTC
        "volatility_mult": 1.3,
        "confidence_adj": -5,  # Plus permissif (haute volatilité)
        "description": "London Session - High volume"
    },
    "newyork": {
        "hours": (13, 22),  # 13:00-22:00 UTC
        "volatility_mult": 1.2,
        "confidence_adj": -3,
        "description": "New York Session"
    },
    "london_ny_overlap": {
        "hours": (13, 16),  # 13:00-16:00 UTC
        "volatility_mult": 1.5,
        "confidence_adj": -8,  # Très permissif (max volatilité)
        "description": "London/NY Overlap - Maximum volume"
    },
    "quiet": {
        "hours": (22, 24),  # Late NY/Pre-Tokyo
        "volatility_mult": 0.5,
        "confidence_adj": +10,  # Très strict (marché mort)
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
    # Return pair profile or default values if pair not found
    default_profile = {
        "volatility_class": "medium",
        "base_confidence": 60,
        "rsi_overbought": 70,
        "rsi_oversold": 30,
        "min_rr_ratio": 1.5,
        "typical_atr_m15": 0.0010,
        "typical_spread": 1.0
    }
    return PAIR_PROFILES.get(pair, default_profile)

def detect_session() -> Tuple[str, Dict]:
    """
    Alias for get_current_session() for backward compatibility.
    Detect current trading session based on UTC time.

    Returns:
        Tuple of (session_name, session_profile)
    """
    return get_current_session()

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
            # Default to EURUSD profile if pair unknown
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
                rsi_ob += 3  # Wider bands
                rsi_os -= 3
            elif volatility_info["regime"] == "low_volatility":
                rsi_ob -= 3  # Tighter bands
                rsi_os += 3
        
        # ADJUSTMENT 3: Clamp values to reasonable ranges
        confidence = max(40, min(80, confidence))  # 40-80%
        rsi_ob = max(65, min(80, rsi_ob))  # 65-80
        rsi_os = max(20, min(35, rsi_os))  # 20-35
        min_rr = max(1.0, min(3.0, min_rr))  # 1.0-3.0
        
        return {
            "pair": pair_clean,
            "confidence_threshold": confidence,
            "rsi_overbought": rsi_ob,
            "rsi_oversold": rsi_os,
            "min_rr_ratio": min_rr,
            # Metadata
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
    
    def get_all_pairs_thresholds(self) -> Dict[str, Dict]:
        """Get current thresholds for all configured pairs."""
        thresholds = {}
        for pair in self.pair_profiles.keys():
            thresholds[pair] = self.get_thresholds(pair)
        return thresholds
    
    def print_current_thresholds(self, pair: Optional[str] = None):
        """Print formatted thresholds for debugging."""
        if pair:
            pairs_to_show = [pair.replace('=X', '')]
        else:
            pairs_to_show = list(self.pair_profiles.keys())[:5]  # Show first 5
        
        print("\n" + "="*80)
        print("ADAPTIVE THRESHOLDS - Current Settings")
        print("="*80)
        
        session_name, session_profile = get_current_session()
        print(f"Current Session: {session_name.upper()} - {session_profile['description']}")
        print(f"UTC Time: {datetime.now(timezone.utc).strftime('%H:%M:%S')}")
        print("-"*80)
        
        for p in pairs_to_show:
            thresholds = self.get_thresholds(p)
            print(f"\n{p:10} [Vol: {thresholds['volatility_class']:12}]")
            print(f"  Confidence: {thresholds['confidence_threshold']:5.1f}%  "
                  f"RSI: {thresholds['rsi_oversold']}-{thresholds['rsi_overbought']}  "
                  f"Min R:R: {thresholds['min_rr_ratio']:.1f}")
            print(f"  Session Adj: {thresholds['adjustments_applied']['session_adj']:+3d}  "
                  f"Volatility: {thresholds['volatility_regime']}")
        
        print("\n" + "="*80)

# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def get_adaptive_thresholds(pair: str, df: Optional[pd.DataFrame] = None) -> Dict:
    """
    Quick function to get adaptive thresholds for a pair.
    
    Usage:
        thresholds = get_adaptive_thresholds('EURUSD=X', df)
        confidence = thresholds['confidence_threshold']
    """
    calculator = AdaptiveThresholds()
    return calculator.get_thresholds(pair, df)

if __name__ == "__main__":
    # Demo usage
    print("\n🎯 ADAPTIVE THRESHOLDS SYSTEM - Demo")
    print("="*80)
    
    calculator = AdaptiveThresholds()
    
    # Show thresholds for different pair types
    print("\n📊 Sample Pairs (Current Session Adjustments):")
    calculator.print_current_thresholds()
    
    print("\n\n💡 Usage in Scanner:")
    print("""    from adaptive_thresholds import get_adaptive_thresholds
    
    # In your strategy/scanner:
    thresholds = get_adaptive_thresholds(pair, df)
    
    if signal_confidence >= thresholds['confidence_threshold']:
        # Accept signal
        pass
    """)
