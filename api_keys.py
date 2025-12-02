"""
API Keys Configuration - Forex Scalper Agent V2
================================================
Centralized API keys for data sources.

IMPORTANT:
- Ne jamais commit ce fichier sur Git public!
- Ajoutez 'api_keys.py' a votre .gitignore
"""

from enum import Enum


class APIKeys(Enum):
    """API Keys for forex data sources (100% gratuit)."""

    # Twelve Data - 800 req/jour, ~1 min delai (RECOMMANDE)
    # https://twelvedata.com
    TWELVE_DATA = "0d071977c5cf4da9981b55d7c26f59a3"

    # Finnhub - 60 req/min, temps reel (EXCELLENT BACKUP)
    # https://finnhub.io
    FINNHUB = "cv9l6ghr01qpd9s7rhj0cv9l6ghr01qpd9s7rhjg"

    # Alpha Vantage - 5 req/min (optionnel)
    # https://alphavantage.co
    ALPHA_VANTAGE = ""


# Helper functions for easy access
def get_twelve_data_key() -> str:
    """Get Twelve Data API key."""
    return APIKeys.TWELVE_DATA.value


def get_finnhub_key() -> str:
    """Get Finnhub API key."""
    return APIKeys.FINNHUB.value


def get_alpha_vantage_key() -> str:
    """Get Alpha Vantage API key."""
    return APIKeys.ALPHA_VANTAGE.value


# Dict format for DataFetcher
API_KEYS_DICT = {
    "twelve_data": APIKeys.TWELVE_DATA.value,
    "finnhub": APIKeys.FINNHUB.value,
    "alpha_vantage": APIKeys.ALPHA_VANTAGE.value,
}
