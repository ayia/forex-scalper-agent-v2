"""Configuration des paires Forex."""
from typing import Dict, List

# FOREX PAIRS UNIVERSE
MAJOR_PAIRS: List[str] = [
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD",
    "USDCHF", "USDCAD", "NZDUSD"
]

CROSS_PAIRS: List[str] = [
    "EURJPY", "EURGBP", "EURCHF", "EURAUD", "EURCAD",
    "GBPJPY", "GBPAUD", "GBPCAD", "GBPCHF",
    "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY"
]

ALL_PAIRS: List[str] = MAJOR_PAIRS + CROSS_PAIRS

# PIP VALUES
PIP_VALUES: Dict[str, float] = {
    "USDJPY": 0.01, "EURJPY": 0.01, "GBPJPY": 0.01,
    "AUDJPY": 0.01, "CADJPY": 0.01, "CHFJPY": 0.01, "NZDJPY": 0.01,
    "DEFAULT": 0.0001
}


def get_pip_value(pair: str) -> float:
    """Get the pip value for a given pair."""
    return PIP_VALUES.get(pair, PIP_VALUES["DEFAULT"])
