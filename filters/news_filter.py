"""
News Filter Module - Economic Calendar
=======================================
Filters trades around major economic events:
- NFP (Non-Farm Payrolls)
- FOMC (Federal Reserve)
- ECB (European Central Bank)
- BOE (Bank of England)
- BOJ (Bank of Japan)
- GDP, CPI, PMI releases
- etc.

Rules:
1. No new positions 30 min before high-impact news
2. Close positions or widen stops before major news
3. Avoid trading during releases
"""
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class NewsImpact(Enum):
    """Impact level of economic news."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"  # Central bank decisions, NFP


@dataclass
class EconomicEvent:
    """Represents an economic event."""
    timestamp: datetime
    currency: str
    event_name: str
    impact: NewsImpact
    previous: Optional[str] = None
    forecast: Optional[str] = None
    actual: Optional[str] = None


class NewsFilter:
    """
    Filters trades based on economic calendar.

    Features:
    1. Built-in schedule of recurring events (FOMC, NFP, etc.)
    2. Pre-trade filter (blocks trades before high-impact news)
    3. Position management recommendations around news
    4. Multi-currency awareness (filters based on pair currencies)
    """

    def __init__(self):
        """Initialize news filter with default settings."""
        # Time buffers around news events (in minutes)
        self.critical_buffer_before = 60   # 1 hour before critical news
        self.critical_buffer_after = 30    # 30 min after
        self.high_buffer_before = 30       # 30 min before high impact
        self.high_buffer_after = 15        # 15 min after
        self.medium_buffer_before = 15     # 15 min before medium
        self.medium_buffer_after = 10      # 10 min after

        # Load recurring events schedule
        self.recurring_events = self._load_recurring_events()

        # Cache for today's events
        self._today_events_cache = {}
        self._cache_date = None

        logger.info("NewsFilter initialized with economic calendar")

    def _load_recurring_events(self) -> List[Dict]:
        """
        Load schedule of recurring economic events.
        This is a simplified static list - in production, use a live calendar API.
        """
        return [
            # USD Events
            {
                'currency': 'USD',
                'event_name': 'Non-Farm Payrolls',
                'impact': NewsImpact.CRITICAL,
                'day_of_month': 'first_friday',
                'time_utc': '13:30'
            },
            {
                'currency': 'USD',
                'event_name': 'FOMC Rate Decision',
                'impact': NewsImpact.CRITICAL,
                'frequency': 'fomc_schedule',
                'time_utc': '19:00'
            },
            {
                'currency': 'USD',
                'event_name': 'CPI (Inflation)',
                'impact': NewsImpact.HIGH,
                'day_of_month': 'mid_month',
                'time_utc': '13:30'
            },
            {
                'currency': 'USD',
                'event_name': 'Initial Jobless Claims',
                'impact': NewsImpact.MEDIUM,
                'day_of_week': 'thursday',
                'time_utc': '13:30'
            },
            {
                'currency': 'USD',
                'event_name': 'GDP',
                'impact': NewsImpact.HIGH,
                'frequency': 'quarterly',
                'time_utc': '13:30'
            },
            {
                'currency': 'USD',
                'event_name': 'ISM Manufacturing PMI',
                'impact': NewsImpact.HIGH,
                'day_of_month': 'first_business_day',
                'time_utc': '15:00'
            },
            {
                'currency': 'USD',
                'event_name': 'Retail Sales',
                'impact': NewsImpact.MEDIUM,
                'day_of_month': 'mid_month',
                'time_utc': '13:30'
            },

            # EUR Events
            {
                'currency': 'EUR',
                'event_name': 'ECB Rate Decision',
                'impact': NewsImpact.CRITICAL,
                'frequency': 'monthly',
                'time_utc': '13:15'
            },
            {
                'currency': 'EUR',
                'event_name': 'ECB Press Conference',
                'impact': NewsImpact.CRITICAL,
                'frequency': 'after_ecb',
                'time_utc': '13:45'
            },
            {
                'currency': 'EUR',
                'event_name': 'German CPI',
                'impact': NewsImpact.HIGH,
                'frequency': 'monthly',
                'time_utc': '13:00'
            },
            {
                'currency': 'EUR',
                'event_name': 'Eurozone CPI',
                'impact': NewsImpact.HIGH,
                'frequency': 'monthly',
                'time_utc': '10:00'
            },
            {
                'currency': 'EUR',
                'event_name': 'German Manufacturing PMI',
                'impact': NewsImpact.MEDIUM,
                'day_of_month': 'first_business_day',
                'time_utc': '08:55'
            },

            # GBP Events
            {
                'currency': 'GBP',
                'event_name': 'BOE Rate Decision',
                'impact': NewsImpact.CRITICAL,
                'frequency': 'monthly',
                'time_utc': '12:00'
            },
            {
                'currency': 'GBP',
                'event_name': 'UK CPI',
                'impact': NewsImpact.HIGH,
                'frequency': 'monthly',
                'time_utc': '07:00'
            },
            {
                'currency': 'GBP',
                'event_name': 'UK GDP',
                'impact': NewsImpact.HIGH,
                'frequency': 'quarterly',
                'time_utc': '07:00'
            },
            {
                'currency': 'GBP',
                'event_name': 'UK Employment',
                'impact': NewsImpact.HIGH,
                'frequency': 'monthly',
                'time_utc': '07:00'
            },

            # JPY Events
            {
                'currency': 'JPY',
                'event_name': 'BOJ Rate Decision',
                'impact': NewsImpact.CRITICAL,
                'frequency': 'monthly',
                'time_utc': '03:00'
            },
            {
                'currency': 'JPY',
                'event_name': 'BOJ Press Conference',
                'impact': NewsImpact.CRITICAL,
                'frequency': 'after_boj',
                'time_utc': '06:00'
            },
            {
                'currency': 'JPY',
                'event_name': 'Japan CPI',
                'impact': NewsImpact.MEDIUM,
                'frequency': 'monthly',
                'time_utc': '00:30'
            },

            # AUD Events
            {
                'currency': 'AUD',
                'event_name': 'RBA Rate Decision',
                'impact': NewsImpact.CRITICAL,
                'frequency': 'monthly',
                'time_utc': '04:30'
            },
            {
                'currency': 'AUD',
                'event_name': 'Australia Employment',
                'impact': NewsImpact.HIGH,
                'frequency': 'monthly',
                'time_utc': '00:30'
            },

            # NZD Events
            {
                'currency': 'NZD',
                'event_name': 'RBNZ Rate Decision',
                'impact': NewsImpact.CRITICAL,
                'frequency': 'quarterly',
                'time_utc': '02:00'
            },

            # CAD Events
            {
                'currency': 'CAD',
                'event_name': 'BOC Rate Decision',
                'impact': NewsImpact.CRITICAL,
                'frequency': 'monthly',
                'time_utc': '15:00'
            },
            {
                'currency': 'CAD',
                'event_name': 'Canada Employment',
                'impact': NewsImpact.HIGH,
                'frequency': 'monthly',
                'time_utc': '13:30'
            },

            # CHF Events
            {
                'currency': 'CHF',
                'event_name': 'SNB Rate Decision',
                'impact': NewsImpact.CRITICAL,
                'frequency': 'quarterly',
                'time_utc': '08:30'
            },
        ]

    def get_upcoming_events(self, hours_ahead: int = 24) -> List[EconomicEvent]:
        """
        Get upcoming economic events within the specified time window.

        Args:
            hours_ahead: How many hours to look ahead

        Returns:
            List of upcoming EconomicEvent objects
        """
        now = datetime.utcnow()
        end_time = now + timedelta(hours=hours_ahead)

        upcoming = []
        today = now.date()
        day_of_week = now.strftime('%A').lower()

        for event_template in self.recurring_events:
            event_time = None

            if event_template.get('day_of_week') == day_of_week:
                time_parts = event_template['time_utc'].split(':')
                event_time = now.replace(
                    hour=int(time_parts[0]),
                    minute=int(time_parts[1]),
                    second=0,
                    microsecond=0
                )

            elif event_template.get('day_of_month') == 'first_friday':
                if day_of_week == 'friday' and today.day <= 7:
                    time_parts = event_template['time_utc'].split(':')
                    event_time = now.replace(
                        hour=int(time_parts[0]),
                        minute=int(time_parts[1]),
                        second=0,
                        microsecond=0
                    )

            elif event_template.get('day_of_month') == 'first_business_day':
                if today.day == 1 or (today.day <= 3 and day_of_week in ['monday', 'tuesday']):
                    time_parts = event_template['time_utc'].split(':')
                    event_time = now.replace(
                        hour=int(time_parts[0]),
                        minute=int(time_parts[1]),
                        second=0,
                        microsecond=0
                    )

            elif event_template.get('day_of_month') == 'mid_month':
                if 10 <= today.day <= 15:
                    time_parts = event_template['time_utc'].split(':')
                    event_time = now.replace(
                        hour=int(time_parts[0]),
                        minute=int(time_parts[1]),
                        second=0,
                        microsecond=0
                    )

            if event_time and now <= event_time <= end_time:
                upcoming.append(EconomicEvent(
                    timestamp=event_time,
                    currency=event_template['currency'],
                    event_name=event_template['event_name'],
                    impact=event_template['impact']
                ))

        upcoming.sort(key=lambda x: x.timestamp)
        return upcoming

    def should_trade(self, pair: str) -> Tuple[bool, str, Optional[EconomicEvent]]:
        """
        Check if trading is allowed for a pair based on upcoming news.

        Args:
            pair: Trading pair (e.g., "EURUSD")

        Returns:
            Tuple of (can_trade, reason, blocking_event)
        """
        currencies = self._get_currencies_from_pair(pair)
        upcoming = self.get_upcoming_events(hours_ahead=2)

        now = datetime.utcnow()

        for event in upcoming:
            if event.currency not in currencies:
                continue

            time_to_event = (event.timestamp - now).total_seconds() / 60

            if event.impact == NewsImpact.CRITICAL:
                if -self.critical_buffer_after <= time_to_event <= self.critical_buffer_before:
                    return False, f"CRITICAL news in {time_to_event:.0f}min: {event.event_name}", event

            elif event.impact == NewsImpact.HIGH:
                if -self.high_buffer_after <= time_to_event <= self.high_buffer_before:
                    return False, f"HIGH impact news in {time_to_event:.0f}min: {event.event_name}", event

            elif event.impact == NewsImpact.MEDIUM:
                if -self.medium_buffer_after <= time_to_event <= self.medium_buffer_before:
                    return False, f"MEDIUM impact news in {time_to_event:.0f}min: {event.event_name}", event

        return True, "No significant news events", None

    def get_risk_adjustment(self, pair: str) -> float:
        """
        Get risk adjustment multiplier based on upcoming news.

        Args:
            pair: Trading pair

        Returns:
            Risk multiplier (0.0 to 1.0)
        """
        can_trade, reason, event = self.should_trade(pair)

        if not can_trade:
            return 0.0

        currencies = self._get_currencies_from_pair(pair)
        upcoming = self.get_upcoming_events(hours_ahead=4)

        now = datetime.utcnow()
        min_risk = 1.0

        for event in upcoming:
            if event.currency not in currencies:
                continue

            time_to_event = (event.timestamp - now).total_seconds() / 60

            if event.impact == NewsImpact.CRITICAL:
                if time_to_event <= 120:
                    risk = 0.3 + (time_to_event / 120) * 0.7
                    min_risk = min(min_risk, risk)

            elif event.impact == NewsImpact.HIGH:
                if time_to_event <= 60:
                    risk = 0.5 + (time_to_event / 60) * 0.5
                    min_risk = min(min_risk, risk)

        return min_risk

    def get_stop_adjustment(self, pair: str) -> float:
        """
        Get stop loss adjustment multiplier for news volatility.

        Args:
            pair: Trading pair

        Returns:
            SL multiplier (1.0 = normal, >1.0 = wider stops)
        """
        currencies = self._get_currencies_from_pair(pair)
        upcoming = self.get_upcoming_events(hours_ahead=2)

        now = datetime.utcnow()
        max_adjustment = 1.0

        for event in upcoming:
            if event.currency not in currencies:
                continue

            time_to_event = (event.timestamp - now).total_seconds() / 60

            if event.impact == NewsImpact.CRITICAL and time_to_event <= 60:
                max_adjustment = max(max_adjustment, 1.5)
            elif event.impact == NewsImpact.HIGH and time_to_event <= 30:
                max_adjustment = max(max_adjustment, 1.3)

        return max_adjustment

    def _get_currencies_from_pair(self, pair: str) -> List[str]:
        """Extract base and quote currencies from pair."""
        pair = pair.upper().replace('=X', '')
        if len(pair) == 6:
            return [pair[:3], pair[3:]]
        return [pair]

    def get_news_summary(self, pair: str = None) -> Dict:
        """
        Get a summary of upcoming news for dashboard display.

        Args:
            pair: Optional pair to filter events

        Returns:
            Summary dictionary
        """
        upcoming = self.get_upcoming_events(hours_ahead=24)

        if pair:
            currencies = self._get_currencies_from_pair(pair)
            upcoming = [e for e in upcoming if e.currency in currencies]

        critical_events = [e for e in upcoming if e.impact == NewsImpact.CRITICAL]
        high_events = [e for e in upcoming if e.impact == NewsImpact.HIGH]
        medium_events = [e for e in upcoming if e.impact == NewsImpact.MEDIUM]

        return {
            'total_events': len(upcoming),
            'critical': len(critical_events),
            'high': len(high_events),
            'medium': len(medium_events),
            'next_critical': critical_events[0] if critical_events else None,
            'next_high': high_events[0] if high_events else None,
            'events': [
                {
                    'time': e.timestamp.isoformat(),
                    'currency': e.currency,
                    'event': e.event_name,
                    'impact': e.impact.value
                }
                for e in upcoming[:10]
            ]
        }


# Singleton instance
_news_filter_instance = None


def get_news_filter() -> NewsFilter:
    """Get singleton NewsFilter instance."""
    global _news_filter_instance
    if _news_filter_instance is None:
        _news_filter_instance = NewsFilter()
    return _news_filter_instance


def should_trade_news(pair: str) -> Tuple[bool, str]:
    """
    Quick function to check if trading is allowed.

    Args:
        pair: Trading pair

    Returns:
        Tuple of (can_trade, reason)
    """
    filter_instance = get_news_filter()
    can_trade, reason, _ = filter_instance.should_trade(pair)
    return can_trade, reason


def get_news_risk_adjustment(pair: str) -> float:
    """
    Quick function to get risk adjustment.

    Args:
        pair: Trading pair

    Returns:
        Risk multiplier (0.0 to 1.0)
    """
    return get_news_filter().get_risk_adjustment(pair)
