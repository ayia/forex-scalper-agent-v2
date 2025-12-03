"""Forex Scalper Agent V2 - Configuration principale."""
from typing import Dict
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# RISK MANAGEMENT PARAMETERS
RISK_PARAMS: Dict[str, float] = {
    "max_spread_pips": 2.0,
    "min_atr_pips": 5.0,
    "max_atr_pips": 30.0,
    "max_sl_pips": 15.0,
    "sl_atr_multiplier": 1.5,
    "min_rr_ratio": 1.5,
    "target_rr_ratio": 2.0,
    "correlation_threshold": 0.85,
    "confidence_threshold": 60,
    "news_filter_minutes": 30
}

# STRATEGY INDICATOR PARAMETERS
STRATEGY_PARAMS: Dict[str, any] = {
    "ema_fast": 20,
    "ema_medium": 50,
    "ema_slow": 200,
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "bb_period": 20,
    "bb_std": 2.5,
    "atr_period": 14,
    "donchian_period": 20,
    "fractal_lookback": 10,
    "sweep_threshold_pips": 3.0
}

# SCORING WEIGHTS
SCORING_WEIGHTS: Dict[str, int] = {
    "strategy_signal": 40,
    "h1_alignment": 30,
    "m15_structure": 20,
    "sentiment_bonus": 10,
    "divergence_penalty": -20
}

# SMC (Smart Money Concepts) PARAMETERS
SMC_PARAMS: Dict[str, any] = {
    "swing_lookback": 10,
    "liquidity_threshold": 0.5,
}

# INDICATORS CONFIGURATION
INDICATORS: Dict[str, any] = {
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
}

# API CONFIGURATION
API_CONFIG: Dict[str, str] = {
    "data_provider": os.getenv("DATA_PROVIDER", "yfinance"),
    "news_api_key": os.getenv("NEWS_API_KEY", ""),
    "forex_api_key": os.getenv("FOREX_API_KEY", ""),
}

# LOGGING & OUTPUT
LOG_CONFIG: Dict[str, str] = {
    "log_level": "INFO",
    "signals_csv": "logs/signals.csv",
    "performance_csv": "logs/performance.csv",
    "cache_dir": "data/cache/"
}
