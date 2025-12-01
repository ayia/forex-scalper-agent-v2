"""
Data Fetcher Module - Multi-Source Forex Data (100% Gratuit)
=============================================================
Sources de données gratuites avec fallback automatique:
1. yfinance (par défaut) - Illimité, délai 15-20 min
2. Alpha Vantage - 5 req/min, temps réel (API key gratuite)
3. Twelve Data - 800 req/jour, temps réel (API key gratuite)

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
        self.api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
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
    """Twelve Data source - Gratuit avec API key (800 req/jour)."""

    def __init__(self, api_key: str = None):
        self.name = "twelve_data"
        self.api_key = api_key or os.environ.get("TWELVE_DATA_API_KEY", "")
        self.base_url = "https://api.twelvedata.com/time_series"
        self.last_request = 0
        self.min_interval = 1.0  # Pas de limite stricte par minute
        self.requests_today = 0
        self.max_daily = 800

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


class DataFetcher:
    """
    Multi-Source Data Fetcher avec fallback automatique.

    Ordre de priorité:
    1. yfinance (toujours disponible)
    2. Alpha Vantage (si API key configurée)
    3. Twelve Data (si API key configurée)

    Configuration des API keys:
    - Variables d'environnement: ALPHA_VANTAGE_API_KEY, TWELVE_DATA_API_KEY
    - Ou directement dans le constructeur
    """

    def __init__(self,
                 alpha_vantage_key: str = None,
                 twelve_data_key: str = None,
                 preferred_source: str = "yfinance"):
        """
        Initialize multi-source data fetcher.

        Args:
            alpha_vantage_key: Alpha Vantage API key (gratuit sur alphavantage.co)
            twelve_data_key: Twelve Data API key (gratuit sur twelvedata.com)
            preferred_source: Source préférée ("yfinance", "alpha_vantage", "twelve_data")
        """
        # Initialize all sources
        self.sources: Dict[str, DataSource] = {
            "yfinance": YFinanceSource(),
            "alpha_vantage": AlphaVantageSource(alpha_vantage_key),
            "twelve_data": TwelveDataSource(twelve_data_key)
        }

        self.preferred_source = preferred_source
        self.cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        self.cache_duration = timedelta(seconds=60)

        # Statistics
        self.fetch_count = {"yfinance": 0, "alpha_vantage": 0, "twelve_data": 0}
        self.error_count = {"yfinance": 0, "alpha_vantage": 0, "twelve_data": 0}

        # Log available sources
        available = [name for name, src in self.sources.items() if src.is_available()]
        logger.info(f"DataFetcher initialized with sources: {', '.join(available)}")

    def _get_source_order(self) -> List[str]:
        """Get ordered list of sources to try."""
        # Start with preferred, then others
        order = [self.preferred_source]
        for name in self.sources.keys():
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
                        twelve_data_key: str = None) -> DataFetcher:
    """Factory function to create a DataFetcher instance."""
    return DataFetcher(
        alpha_vantage_key=alpha_vantage_key,
        twelve_data_key=twelve_data_key
    )
