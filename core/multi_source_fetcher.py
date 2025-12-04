#!/usr/bin/env python3
"""
Multi-Source Data Fetcher V1.0
==============================
Fetches forex data from multiple APIs with automatic fallback.

Priority order:
1. Twelve Data (800 req/day, historical data available)
2. Finnhub (60 req/min, real-time)
3. Yahoo Finance (fallback, limited to 730 days for hourly)

Features:
- Automatic source rotation on failure
- Rate limiting handling
- Unified DataFrame output
- Historical data support (back to 2017)

Usage:
    from core.multi_source_fetcher import MultiSourceFetcher
    fetcher = MultiSourceFetcher()
    df = fetcher.fetch('CADJPY', '2020-01-01', '2024-12-01', '1h')
"""

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Import API keys
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from api_keys import get_twelve_data_key, get_finnhub_key
except ImportError:
    def get_twelve_data_key(): return ""
    def get_finnhub_key(): return ""


class MultiSourceFetcher:
    """
    Multi-source forex data fetcher with automatic fallback.

    Sources:
    - Twelve Data: Best for historical data (free tier: 800 req/day)
    - Finnhub: Best for real-time (free tier: 60 req/min)
    - Yahoo Finance: Fallback (730-day limit for hourly)
    """

    # Interval mapping for each API
    TWELVE_DATA_INTERVALS = {
        '1h': '1h',
        '1H': '1h',
        '4h': '4h',
        '4H': '4h',
        '1d': '1day',
        '1D': '1day',
        '1day': '1day',
        'daily': '1day',
    }

    FINNHUB_RESOLUTIONS = {
        '1h': '60',
        '1H': '60',
        '4h': '240',
        '4H': '240',
        '1d': 'D',
        '1D': 'D',
        '1day': 'D',
        'daily': 'D',
    }

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.twelve_data_key = get_twelve_data_key()
        self.finnhub_key = get_finnhub_key()

        # Rate limiting tracking
        self.twelve_data_calls = 0
        self.twelve_data_reset = datetime.now()
        self.finnhub_calls = 0
        self.finnhub_reset = datetime.now()

    def _log(self, msg: str):
        if self.verbose:
            print(f"[MultiSourceFetcher] {msg}")

    def _convert_pair_twelve_data(self, pair: str) -> str:
        """Convert pair to Twelve Data format (e.g., CADJPY -> CAD/JPY)"""
        if len(pair) == 6:
            return f"{pair[:3]}/{pair[3:]}"
        return pair

    def _convert_pair_finnhub(self, pair: str) -> str:
        """Convert pair to Finnhub format (e.g., CADJPY -> OANDA:CAD_JPY)"""
        if len(pair) == 6:
            return f"OANDA:{pair[:3]}_{pair[3:]}"
        return pair

    def _convert_pair_yfinance(self, pair: str) -> str:
        """Convert pair to Yahoo Finance format (e.g., CADJPY -> CADJPY=X)"""
        return f"{pair}=X"

    def fetch_twelve_data(self, pair: str, start: str, end: str,
                          interval: str = '1h') -> Optional[pd.DataFrame]:
        """
        Fetch data from Twelve Data API.

        Supports historical data back to ~2010 for forex.
        Free tier: 800 API calls/day, 8 calls/minute.
        """
        if not self.twelve_data_key:
            self._log("No Twelve Data API key configured")
            return None

        try:
            symbol = self._convert_pair_twelve_data(pair)
            td_interval = self.TWELVE_DATA_INTERVALS.get(interval, '1h')

            # Calculate output size (max 5000 per request)
            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)

            # For hourly data, estimate bars needed
            if 'h' in td_interval:
                hours_diff = (end_dt - start_dt).total_seconds() / 3600
                # Forex: ~24h * 5 days/week = ~120 bars/week
                estimated_bars = int(hours_diff * 0.7)  # Account for weekends
            else:
                estimated_bars = (end_dt - start_dt).days

            output_size = min(5000, max(100, estimated_bars))

            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': symbol,
                'interval': td_interval,
                'start_date': start,
                'end_date': end,
                'outputsize': output_size,
                'apikey': self.twelve_data_key,
                'format': 'JSON',
                'timezone': 'UTC'
            }

            self._log(f"Fetching {pair} from Twelve Data ({td_interval}, {start} to {end})")

            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if data.get('status') != 'ok':
                self._log(f"Twelve Data error: {data.get('message', 'Unknown error')}")
                return None

            if 'values' not in data or not data['values']:
                self._log("No data returned from Twelve Data")
                return None

            # Convert to DataFrame
            df = pd.DataFrame(data['values'])
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)

            # Convert columns to float
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # Add volume if not present (forex doesn't have volume)
            if 'volume' not in df.columns:
                df['volume'] = 0

            # Sort by date ascending
            df = df.sort_index()

            self._log(f"Twelve Data: Got {len(df)} bars")
            return df

        except Exception as e:
            self._log(f"Twelve Data error: {e}")
            return None

    def fetch_finnhub(self, pair: str, start: str, end: str,
                      interval: str = '1h') -> Optional[pd.DataFrame]:
        """
        Fetch data from Finnhub API.

        Better for recent/real-time data.
        Free tier: 60 API calls/minute.
        Note: Limited historical data on free tier.
        """
        if not self.finnhub_key:
            self._log("No Finnhub API key configured")
            return None

        try:
            symbol = self._convert_pair_finnhub(pair)
            resolution = self.FINNHUB_RESOLUTIONS.get(interval, '60')

            start_ts = int(pd.to_datetime(start).timestamp())
            end_ts = int(pd.to_datetime(end).timestamp())

            url = "https://finnhub.io/api/v1/forex/candle"
            params = {
                'symbol': symbol,
                'resolution': resolution,
                'from': start_ts,
                'to': end_ts,
                'token': self.finnhub_key
            }

            self._log(f"Fetching {pair} from Finnhub ({resolution}, {start} to {end})")

            response = requests.get(url, params=params, timeout=30)
            data = response.json()

            if data.get('s') == 'no_data' or 'c' not in data:
                self._log("No data returned from Finnhub")
                return None

            # Convert to DataFrame
            df = pd.DataFrame({
                'open': data['o'],
                'high': data['h'],
                'low': data['l'],
                'close': data['c'],
                'volume': data.get('v', [0] * len(data['c'])),
            })

            df.index = pd.to_datetime(data['t'], unit='s')
            df = df.sort_index()

            self._log(f"Finnhub: Got {len(df)} bars")
            return df

        except Exception as e:
            self._log(f"Finnhub error: {e}")
            return None

    def fetch_yahoo(self, pair: str, start: str, end: str,
                    interval: str = '1h') -> Optional[pd.DataFrame]:
        """
        Fetch data from Yahoo Finance (fallback).

        Limitations:
        - Hourly data limited to last 730 days
        - Daily data available for longer periods
        """
        try:
            symbol = self._convert_pair_yfinance(pair)

            # Check if we need to force daily due to YF limitations
            start_date = pd.to_datetime(start)
            today = pd.to_datetime(datetime.now().strftime('%Y-%m-%d'))
            days_ago = (today - start_date).days

            yf_interval = interval
            if days_ago > 700 and interval in ['1h', '1H']:
                yf_interval = '1d'
                self._log(f"Yahoo: Forcing daily interval (data too old for hourly)")

            self._log(f"Fetching {pair} from Yahoo Finance ({yf_interval}, {start} to {end})")

            df = yf.download(symbol, start=start, end=end, interval=yf_interval, progress=False)

            if df.empty:
                self._log("No data returned from Yahoo Finance")
                return None

            # Normalize columns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            # Ensure standard columns
            df = df.rename(columns={'adj close': 'close'})

            self._log(f"Yahoo Finance: Got {len(df)} bars")
            return df

        except Exception as e:
            self._log(f"Yahoo Finance error: {e}")
            return None

    def fetch(self, pair: str, start: str, end: str,
              interval: str = '1h',
              source_priority: List[str] = None) -> Optional[pd.DataFrame]:
        """
        Fetch data with automatic fallback between sources.

        Args:
            pair: Currency pair (e.g., 'CADJPY')
            start: Start date (e.g., '2020-01-01')
            end: End date (e.g., '2024-12-01')
            interval: Data interval ('1h', '4h', '1d')
            source_priority: List of sources to try in order
                            Default: ['twelve_data', 'finnhub', 'yahoo']

        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        if source_priority is None:
            # Default priority: Twelve Data first (best historical), then Finnhub, then Yahoo
            source_priority = ['twelve_data', 'finnhub', 'yahoo']

        fetchers = {
            'twelve_data': self.fetch_twelve_data,
            'finnhub': self.fetch_finnhub,
            'yahoo': self.fetch_yahoo,
        }

        for source in source_priority:
            if source not in fetchers:
                continue

            df = fetchers[source](pair, start, end, interval)

            if df is not None and len(df) > 0:
                # Add source metadata
                df.attrs['source'] = source
                df.attrs['pair'] = pair
                df.attrs['interval'] = interval
                return df

            # Small delay before trying next source
            time.sleep(0.5)

        self._log(f"All sources failed for {pair}")
        return None

    def fetch_multiple_periods(self, pair: str, periods: List[Dict],
                                interval: str = '1h') -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple time periods.

        Args:
            pair: Currency pair
            periods: List of {'name': str, 'start': str, 'end': str}
            interval: Data interval

        Returns:
            Dict mapping period name to DataFrame
        """
        results = {}

        for period in periods:
            name = period['name']
            self._log(f"Fetching period: {name}")

            df = self.fetch(pair, period['start'], period['end'], interval)

            if df is not None:
                results[name] = df
            else:
                results[name] = None

            # Rate limiting: wait between periods
            time.sleep(1)

        return results


# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Source Data Fetcher Test")
    print("=" * 60)

    fetcher = MultiSourceFetcher(verbose=True)

    # Test 1: Recent data (should use Twelve Data or Finnhub)
    print("\n[Test 1] Recent 1H data (last 30 days)")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

    df = fetcher.fetch('CADJPY', start_date, end_date, '1h')
    if df is not None:
        print(f"  Source: {df.attrs.get('source', 'unknown')}")
        print(f"  Bars: {len(df)}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    # Test 2: Historical data (COVID period)
    print("\n[Test 2] Historical data (COVID - March 2020)")
    df = fetcher.fetch('CADJPY', '2020-03-01', '2020-04-30', '1d')
    if df is not None:
        print(f"  Source: {df.attrs.get('source', 'unknown')}")
        print(f"  Bars: {len(df)}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")

    # Test 3: Multiple periods
    print("\n[Test 3] Multiple periods")
    periods = [
        {'name': 'COVID_Crash', 'start': '2020-02-15', 'end': '2020-04-15'},
        {'name': 'Recent', 'start': '2024-01-01', 'end': '2024-06-01'},
    ]

    results = fetcher.fetch_multiple_periods('EURGBP', periods, '1d')
    for name, data in results.items():
        if data is not None:
            print(f"  {name}: {len(data)} bars from {data.attrs.get('source', 'unknown')}")
        else:
            print(f"  {name}: FAILED")

    print("\n" + "=" * 60)
    print("Test complete!")
