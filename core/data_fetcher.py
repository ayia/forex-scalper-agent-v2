"""
Data Fetcher Module - Multi-Source Forex Data (100% Gratuit)
=============================================================
Sources de données gratuites avec fallback automatique:

PRIORITÉ POUR SCALPING (délai minimal):
1. Twelve Data - 800 req/jour, ~1 min délai (API key gratuite)
2. Finnhub - 60 req/min, temps réel (API key gratuite)
3. Alpha Vantage - 5 req/min, temps réel (API key gratuite)
4. yfinance - Illimité, délai 15-20 min (FALLBACK UNIQUEMENT)

Configuration des API keys (GRATUIT - sans carte bancaire):
- TWELVE_DATA_API_KEY: https://twelvedata.com (recommandé)
- FINNHUB_API_KEY: https://finnhub.io (excellent backup)
- ALPHA_VANTAGE_API_KEY: https://alphavantage.co

Le système bascule automatiquement entre les sources si une échoue.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import time
import os

from config import TIMEFRAMES, ALL_PAIRS

# Try to import API keys from local config (recommended)
try:
    from config.api_keys import get_twelve_data_key, get_finnhub_key, get_alpha_vantage_key
    _API_KEYS_FROM_FILE = True
except ImportError:
    _API_KEYS_FROM_FILE = False
    def get_twelve_data_key(): return ""
    def get_finnhub_key(): return ""
    def get_alpha_vantage_key(): return ""

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Abstract base class for data sources."""

    @abstractmethod
    def fetch_ohlcv(self, pair: str, timeframe: str, bars: int) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get source name."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if source is available."""
        pass


class YFinanceSource(DataSource):
    """yfinance data source - Gratuit illimité."""

    def __init__(self):
        self.name = "yfinance"
        self.last_request = 0
        self.min_interval = 1.0  # 1 seconde entre requêtes

    def get_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        return True  # Toujours disponible

    def _get_symbol(self, pair: str) -> str:
        """Convert pair to yfinance format."""
        return f"{pair}=X"

    def fetch_ohlcv(self, pair: str, timeframe: str, bars: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from yfinance."""
        try:
            # Rate limiting
            elapsed = time.time() - self.last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            symbol = self._get_symbol(pair)
            interval = TIMEFRAMES.get(timeframe, "15m")

            # Calculate period based on timeframe
            period_map = {
                "1m": "1d", "5m": "5d", "15m": "5d",
                "1h": "30d", "4h": "60d"
            }
            period = period_map.get(interval, "5d")

            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval=interval)

            self.last_request = time.time()

            if df.empty:
                logger.warning(f"[yfinance] No data for {pair} {timeframe}")
                return None

            # Normalize columns
            df.columns = [c.lower() for c in df.columns]
            df = df.tail(bars)

            logger.debug(f"[yfinance] Fetched {len(df)} bars for {pair} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"[yfinance] Error fetching {pair}: {e}")
            return None


class AlphaVantageSource(DataSource):
    """Alpha Vantage data source - Gratuit avec API key (5 req/min)."""

    def __init__(self, api_key: str = None):
        self.name = "alpha_vantage"
        # Priority: parameter > api_keys.py > environment variable
        self.api_key = api_key or get_alpha_vantage_key() or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        self.base_url = "https://www.alphavantage.co/query"
        self.last_request = 0
        self.min_interval = 12.0  # 5 req/min = 12 sec entre requêtes
        self.requests_today = 0
        self.max_daily = 500  # Limite quotidienne gratuite

    def get_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        return bool(self.api_key) and self.requests_today < self.max_daily

    def _get_interval(self, timeframe: str) -> str:
        """Convert timeframe to Alpha Vantage format."""
        interval_map = {
            "M1": "1min", "M5": "5min", "M15": "15min",
            "M30": "30min", "H1": "60min", "H4": "60min"  # H4 not supported, use H1
        }
        return interval_map.get(timeframe, "15min")

    def fetch_ohlcv(self, pair: str, timeframe: str, bars: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Alpha Vantage."""
        if not self.is_available():
            logger.warning(f"[alpha_vantage] Not available (no API key or limit reached)")
            return None

        try:
            # Rate limiting
            elapsed = time.time() - self.last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            # Parse pair
            from_currency = pair[:3]
            to_currency = pair[3:]
            interval = self._get_interval(timeframe)

            params = {
                "function": "FX_INTRADAY",
                "from_symbol": from_currency,
                "to_symbol": to_currency,
                "interval": interval,
                "outputsize": "compact",  # Last 100 data points
                "apikey": self.api_key
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            self.last_request = time.time()
            self.requests_today += 1

            if response.status_code != 200:
                logger.error(f"[alpha_vantage] HTTP {response.status_code}")
                return None

            data = response.json()

            # Check for errors
            if "Error Message" in data:
                logger.error(f"[alpha_vantage] API Error: {data['Error Message']}")
                return None

            if "Note" in data:  # Rate limit message
                logger.warning(f"[alpha_vantage] Rate limit: {data['Note']}")
                return None

            # Parse time series data
            time_series_key = f"Time Series FX (Intraday)"
            # Try different key formats
            for key in data.keys():
                if "Time Series" in key:
                    time_series_key = key
                    break

            if time_series_key not in data:
                logger.warning(f"[alpha_vantage] No time series data in response")
                return None

            time_series = data[time_series_key]

            # Convert to DataFrame
            records = []
            for timestamp, values in time_series.items():
                records.append({
                    'datetime': pd.to_datetime(timestamp),
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': 0  # FX doesn't have volume
                })

            df = pd.DataFrame(records)
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            df = df.tail(bars)

            logger.debug(f"[alpha_vantage] Fetched {len(df)} bars for {pair} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"[alpha_vantage] Error fetching {pair}: {e}")
            return None


class TwelveDataSource(DataSource):
    """Twelve Data source - Gratuit avec API key (800 req/jour, ~1 min délai)."""

    def __init__(self, api_key: str = None):
        self.name = "twelve_data"
        # Priority: parameter > api_keys.py > environment variable
        self.api_key = api_key or get_twelve_data_key() or os.environ.get("TWELVE_DATA_API_KEY", "")
        self.base_url = "https://api.twelvedata.com/time_series"
        self.last_request = 0
        self.min_interval = 0.5  # 8 credits/min = ~7.5 sec, mais on peut aller plus vite
        self.requests_today = 0
        self.max_daily = 800
        self.requests_per_minute = 0
        self.minute_start = time.time()

    def get_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        return bool(self.api_key) and self.requests_today < self.max_daily

    def _get_interval(self, timeframe: str) -> str:
        """Convert timeframe to Twelve Data format."""
        interval_map = {
            "M1": "1min", "M5": "5min", "M15": "15min",
            "M30": "30min", "H1": "1h", "H4": "4h"
        }
        return interval_map.get(timeframe, "15min")

    def _get_symbol(self, pair: str) -> str:
        """Convert pair to Twelve Data format."""
        return f"{pair[:3]}/{pair[3:]}"

    def fetch_ohlcv(self, pair: str, timeframe: str, bars: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Twelve Data."""
        if not self.is_available():
            logger.warning(f"[twelve_data] Not available (no API key or limit reached)")
            return None

        try:
            # Rate limiting
            elapsed = time.time() - self.last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)

            symbol = self._get_symbol(pair)
            interval = self._get_interval(timeframe)

            params = {
                "symbol": symbol,
                "interval": interval,
                "outputsize": bars,
                "apikey": self.api_key
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            self.last_request = time.time()
            self.requests_today += 1

            if response.status_code != 200:
                logger.error(f"[twelve_data] HTTP {response.status_code}")
                return None

            data = response.json()

            # Check for errors
            if "status" in data and data["status"] == "error":
                logger.error(f"[twelve_data] API Error: {data.get('message', 'Unknown')}")
                return None

            if "values" not in data:
                logger.warning(f"[twelve_data] No values in response")
                return None

            # Convert to DataFrame
            records = []
            for candle in data["values"]:
                records.append({
                    'datetime': pd.to_datetime(candle['datetime']),
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close']),
                    'volume': 0
                })

            df = pd.DataFrame(records)
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)

            logger.debug(f"[twelve_data] Fetched {len(df)} bars for {pair} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"[twelve_data] Error fetching {pair}: {e}")
            return None


class FinnhubSource(DataSource):
    """Finnhub data source - Gratuit avec API key (60 req/min, temps réel)."""

    def __init__(self, api_key: str = None):
        self.name = "finnhub"
        # Priority: parameter > api_keys.py > environment variable
        self.api_key = api_key or get_finnhub_key() or os.environ.get("FINNHUB_API_KEY", "")
        self.base_url = "https://finnhub.io/api/v1/forex/candle"
        self.last_request = 0
        self.min_interval = 1.0  # 60 req/min = 1 sec entre requêtes
        self.requests_per_minute = 0
        self.minute_start = time.time()

    def get_name(self) -> str:
        return self.name

    def is_available(self) -> bool:
        return bool(self.api_key)

    def _get_resolution(self, timeframe: str) -> str:
        """Convert timeframe to Finnhub resolution."""
        resolution_map = {
            "M1": "1", "M5": "5", "M15": "15",
            "M30": "30", "H1": "60", "H4": "240", "D": "D"
        }
        return resolution_map.get(timeframe, "15")

    def _get_symbol(self, pair: str) -> str:
        """Convert pair to Finnhub format (OANDA broker)."""
        # Finnhub forex uses format: OANDA:EUR_USD
        return f"OANDA:{pair[:3]}_{pair[3:]}"

    def _rate_limit(self):
        """Ensure we don't exceed 60 requests per minute."""
        current_time = time.time()

        # Reset counter every minute
        if current_time - self.minute_start >= 60:
            self.requests_per_minute = 0
            self.minute_start = current_time

        # If we've hit the limit, wait
        if self.requests_per_minute >= 55:  # Leave some buffer
            wait_time = 60 - (current_time - self.minute_start)
            if wait_time > 0:
                logger.debug(f"[finnhub] Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                self.requests_per_minute = 0
                self.minute_start = time.time()

        # Basic interval between requests
        elapsed = current_time - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def fetch_ohlcv(self, pair: str, timeframe: str, bars: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Finnhub."""
        if not self.is_available():
            logger.warning(f"[finnhub] Not available (no API key)")
            return None

        try:
            self._rate_limit()

            symbol = self._get_symbol(pair)
            resolution = self._get_resolution(timeframe)

            # Calculate time range based on timeframe and bars needed
            now = int(time.time())
            timeframe_seconds = {
                "M1": 60, "M5": 300, "M15": 900,
                "M30": 1800, "H1": 3600, "H4": 14400, "D": 86400
            }
            seconds_per_bar = timeframe_seconds.get(timeframe, 900)
            from_time = now - (bars * seconds_per_bar * 2)  # Extra buffer

            params = {
                "symbol": symbol,
                "resolution": resolution,
                "from": from_time,
                "to": now,
                "token": self.api_key
            }

            response = requests.get(self.base_url, params=params, timeout=10)
            self.last_request = time.time()
            self.requests_per_minute += 1

            if response.status_code != 200:
                logger.error(f"[finnhub] HTTP {response.status_code}")
                return None

            data = response.json()

            # Check for errors
            if data.get("s") == "no_data":
                logger.warning(f"[finnhub] No data for {pair} {timeframe}")
                return None

            if "c" not in data or not data["c"]:
                logger.warning(f"[finnhub] Empty response for {pair}")
                return None

            # Convert to DataFrame
            df = pd.DataFrame({
                'datetime': pd.to_datetime(data['t'], unit='s'),
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'close': data['c'],
                'volume': data.get('v', [0] * len(data['c']))
            })

            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            df = df.tail(bars)

            logger.debug(f"[finnhub] Fetched {len(df)} bars for {pair} {timeframe}")
            return df

        except Exception as e:
            logger.error(f"[finnhub] Error fetching {pair}: {e}")
            return None


class DataFetcher:
    """
    Multi-Source Data Fetcher avec fallback automatique.

    Ordre de priorité (optimisé pour scalping):
    1. Twelve Data - ~1 min délai, 800 req/jour (RECOMMANDÉ)
    2. Finnhub - temps réel, 60 req/min (EXCELLENT BACKUP)
    3. Alpha Vantage - temps réel, 5 req/min (limité)
    4. yfinance - 15-20 min délai (FALLBACK UNIQUEMENT)

    Configuration des API keys (GRATUIT - sans carte bancaire):
    - TWELVE_DATA_API_KEY: https://twelvedata.com
    - FINNHUB_API_KEY: https://finnhub.io
    - ALPHA_VANTAGE_API_KEY: https://alphavantage.co
    """

    def __init__(self,
                 alpha_vantage_key: str = None,
                 twelve_data_key: str = None,
                 finnhub_key: str = None,
                 preferred_source: str = "auto"):
        """
        Initialize multi-source data fetcher.

        Args:
            alpha_vantage_key: Alpha Vantage API key (gratuit sur alphavantage.co)
            twelve_data_key: Twelve Data API key (gratuit sur twelvedata.com)
            finnhub_key: Finnhub API key (gratuit sur finnhub.io)
            preferred_source: Source préférée ou "auto" pour sélection intelligente
        """
        # Initialize all sources
        self.sources: Dict[str, DataSource] = {
            "twelve_data": TwelveDataSource(twelve_data_key),
            "finnhub": FinnhubSource(finnhub_key),
            "alpha_vantage": AlphaVantageSource(alpha_vantage_key),
            "yfinance": YFinanceSource()  # Fallback - toujours dernier
        }

        # Priorité pour le scalping: sources temps réel d'abord
        self.source_priority = ["twelve_data", "finnhub", "alpha_vantage", "yfinance"]

        # Auto-detect best available source
        if preferred_source == "auto":
            self.preferred_source = self._detect_best_source()
        else:
            self.preferred_source = preferred_source

        self.cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self.cache_duration = timedelta(seconds=30)  # Réduit pour scalping

        # Statistics
        self.fetch_count = {name: 0 for name in self.sources}
        self.error_count = {name: 0 for name in self.sources}

        # Log available sources
        available = [name for name, src in self.sources.items() if src.is_available()]
        logger.info(f"DataFetcher initialized with sources: {', '.join(available)}")
        logger.info(f"Preferred source: {self.preferred_source}")

        # Warn if using yfinance only
        if self.preferred_source == "yfinance" and len(available) == 1:
            logger.warning("[!] ATTENTION: Seul yfinance est disponible (delai 15-20 min)")
            logger.warning("Pour du scalping, configurez TWELVE_DATA_API_KEY ou FINNHUB_API_KEY")
            logger.warning("APIs gratuites: https://twelvedata.com | https://finnhub.io")

    def _detect_best_source(self) -> str:
        """Auto-detect the best available source based on priority."""
        for source_name in self.source_priority:
            if source_name in self.sources and self.sources[source_name].is_available():
                logger.info(f"Auto-selected best source: {source_name}")
                return source_name
        return "yfinance"  # Ultimate fallback

    def _get_source_order(self) -> List[str]:
        """Get ordered list of sources to try (optimized for scalping)."""
        # Use priority order, with preferred source first
        order = []

        # Add preferred source first if available
        if self.preferred_source in self.sources:
            order.append(self.preferred_source)

        # Then add rest in priority order
        for name in self.source_priority:
            if name not in order:
                order.append(name)

        return order

    def fetch_ohlcv(self, pair: str, timeframe: str = "M15",
                    bars: int = 100) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV data with automatic fallback.

        Args:
            pair: Currency pair (e.g., "EURUSD")
            timeframe: Timeframe (M1, M5, M15, H1, H4)
            bars: Number of bars to fetch

        Returns:
            DataFrame with OHLCV data or None
        """
        # Check cache first
        cache_key = f"{pair}_{timeframe}"
        if cache_key in self.cache:
            df, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < self.cache_duration:
                logger.debug(f"Cache hit for {pair} {timeframe}")
                return df.copy()

        # Try each source in order
        source_order = self._get_source_order()

        for source_name in source_order:
            source = self.sources[source_name]

            if not source.is_available():
                continue

            df = source.fetch_ohlcv(pair, timeframe, bars)

            if df is not None and not df.empty:
                # Success! Cache and return
                self.cache[cache_key] = (df, datetime.now())
                self.fetch_count[source_name] += 1
                logger.debug(f"Fetched {pair} {timeframe} from {source_name}")
                return df
            else:
                self.error_count[source_name] += 1
                logger.warning(f"Failed to fetch from {source_name}, trying next...")

        logger.error(f"All sources failed for {pair} {timeframe}")
        return None

    def fetch_multi_timeframe(self, pair: str,
                              timeframes: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes.

        Args:
            pair: Currency pair
            timeframes: List of timeframes (default: M1, M5, M15, H1, H4)

        Returns:
            Dict mapping timeframe to DataFrame
        """
        if timeframes is None:
            timeframes = ["M1", "M5", "M15", "H1", "H4"]

        result = {}
        for tf in timeframes:
            data = self.fetch_ohlcv(pair, tf)
            if data is not None and len(data) > 50:
                result[tf] = data

        return result

    def get_spread(self, pair: str) -> float:
        """
        Get estimated spread for a pair.

        Note: Ces valeurs sont des estimations typiques.
        Pour des spreads réels, utilisez une API broker.
        """
        spread_map = {
            # Majors (spreads serrés)
            "EURUSD": 0.8, "GBPUSD": 1.0, "USDJPY": 0.9,
            "AUDUSD": 1.2, "USDCHF": 1.5, "USDCAD": 1.5, "NZDUSD": 1.8,
            # Crosses EUR
            "EURJPY": 1.5, "EURGBP": 1.2, "EURCHF": 2.0,
            "EURAUD": 2.5, "EURCAD": 2.5,
            # Crosses GBP
            "GBPJPY": 2.5, "GBPAUD": 3.0, "GBPCAD": 3.0, "GBPCHF": 3.0,
            # Crosses JPY
            "AUDJPY": 2.0, "CADJPY": 2.0, "CHFJPY": 2.5, "NZDJPY": 2.5
        }
        return spread_map.get(pair, 2.5)

    def get_statistics(self) -> Dict:
        """Get fetcher statistics."""
        total_fetches = sum(self.fetch_count.values())
        total_errors = sum(self.error_count.values())

        return {
            "total_fetches": total_fetches,
            "total_errors": total_errors,
            "success_rate": (total_fetches / (total_fetches + total_errors) * 100)
                           if (total_fetches + total_errors) > 0 else 100,
            "by_source": {
                name: {
                    "fetches": self.fetch_count[name],
                    "errors": self.error_count[name],
                    "available": self.sources[name].is_available()
                }
                for name in self.sources
            },
            "cache_size": len(self.cache)
        }

    def set_preferred_source(self, source: str):
        """Change preferred data source."""
        if source in self.sources:
            self.preferred_source = source
            logger.info(f"Preferred source changed to: {source}")
        else:
            logger.warning(f"Unknown source: {source}")

    def clear_cache(self):
        """Clear data cache."""
        self.cache.clear()
        logger.info("Cache cleared")


# For backward compatibility
def create_data_fetcher(alpha_vantage_key: str = None,
                        twelve_data_key: str = None,
                        finnhub_key: str = None) -> DataFetcher:
    """Factory function to create a DataFetcher instance."""
    return DataFetcher(
        alpha_vantage_key=alpha_vantage_key,
        twelve_data_key=twelve_data_key,
        finnhub_key=finnhub_key
    )


def check_api_keys() -> Dict[str, bool]:
    """
    Check which API keys are configured (from api_keys.py or environment).

    Returns:
        Dict mapping source name to availability status
    """
    return {
        "twelve_data": bool(get_twelve_data_key() or os.environ.get("TWELVE_DATA_API_KEY")),
        "finnhub": bool(get_finnhub_key() or os.environ.get("FINNHUB_API_KEY")),
        "alpha_vantage": bool(get_alpha_vantage_key() or os.environ.get("ALPHA_VANTAGE_API_KEY")),
        "yfinance": True  # Always available
    }
