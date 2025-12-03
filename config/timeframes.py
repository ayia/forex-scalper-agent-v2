"""Configuration des timeframes."""
from typing import Dict, List

# TIMEFRAMES CONFIGURATION
TIMEFRAMES: Dict[str, str] = {
    "M1": "1m", "M5": "5m", "M15": "15m",
    "H1": "1h", "H4": "4h"
}

SCALPING_TIMEFRAMES: List[str] = ["M1", "M5", "M15"]
TREND_TIMEFRAMES: List[str] = ["H1", "H4"]
