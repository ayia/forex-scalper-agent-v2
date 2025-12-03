"""
Session Manager Module
======================
Manages trading sessions and provides session-aware filtering.

Trading Sessions (UTC):
- Tokyo: 00:00 - 09:00
- London: 07:00 - 16:00
- New York: 12:00 - 21:00
- London/NY Overlap: 12:00 - 16:00

Session characteristics affect:
- Volatility expectations
- Pair selection
- Risk parameters
"""
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, time
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class TradingSession(Enum):
    """Trading session identifiers."""
    TOKYO = "tokyo"
    LONDON = "london"
    NEW_YORK = "ny"
    LONDON_NY_OVERLAP = "london_ny"
    QUIET = "quiet"


@dataclass
class SessionInfo:
    """Information about current trading session."""
    session: TradingSession
    name: str
    is_active: bool
    is_overlap: bool
    hours_remaining: float
    volatility_level: str
    recommended_pairs: List[str]
    risk_multiplier: float


# Session definitions with characteristics
SESSION_CONFIG = {
    TradingSession.TOKYO: {
        'name': 'Tokyo/Asian',
        'start': time(0, 0),
        'end': time(9, 0),
        'volatility': 'low',
        'risk_multiplier': 0.8,
        'recommended_pairs': ['USDJPY', 'EURJPY', 'GBPJPY', 'AUDJPY', 'AUDUSD', 'NZDUSD'],
        'description': 'Lower volatility, JPY and AUD pairs most active'
    },
    TradingSession.LONDON: {
        'name': 'London/European',
        'start': time(7, 0),
        'end': time(16, 0),
        'volatility': 'high',
        'risk_multiplier': 1.2,
        'recommended_pairs': ['EURUSD', 'GBPUSD', 'EURGBP', 'EURJPY', 'GBPJPY', 'USDCHF'],
        'description': 'Highest liquidity, EUR and GBP pairs most active'
    },
    TradingSession.NEW_YORK: {
        'name': 'New York/US',
        'start': time(12, 0),
        'end': time(21, 0),
        'volatility': 'high',
        'risk_multiplier': 1.2,
        'recommended_pairs': ['EURUSD', 'GBPUSD', 'USDCAD', 'USDJPY', 'USDCHF'],
        'description': 'High volatility, USD pairs most active'
    },
    TradingSession.LONDON_NY_OVERLAP: {
        'name': 'London/NY Overlap',
        'start': time(12, 0),
        'end': time(16, 0),
        'volatility': 'very_high',
        'risk_multiplier': 1.3,
        'recommended_pairs': ['EURUSD', 'GBPUSD', 'USDCHF', 'USDJPY'],
        'description': 'Highest volatility and liquidity of the day'
    },
    TradingSession.QUIET: {
        'name': 'Quiet Hours',
        'start': time(21, 0),
        'end': time(0, 0),
        'volatility': 'very_low',
        'risk_multiplier': 0.6,
        'recommended_pairs': [],
        'description': 'Low liquidity, avoid trading if possible'
    }
}


class SessionManager:
    """
    Manages trading sessions and provides session-aware filtering.

    Features:
    1. Current session detection
    2. Session-specific pair recommendations
    3. Risk multiplier adjustments
    4. Overlap detection
    """

    def __init__(self):
        """Initialize session manager."""
        logger.info("SessionManager initialized")

    def get_current_session(self, utc_time: datetime = None) -> SessionInfo:
        """
        Get current trading session information.

        Args:
            utc_time: Optional UTC time (defaults to now)

        Returns:
            SessionInfo with current session details
        """
        if utc_time is None:
            utc_time = datetime.utcnow()

        current_time = utc_time.time()
        hour = utc_time.hour

        # Check for London/NY overlap first (highest priority)
        if 12 <= hour < 16:
            session = TradingSession.LONDON_NY_OVERLAP
            is_overlap = True
        # Check London session
        elif 7 <= hour < 16:
            session = TradingSession.LONDON
            is_overlap = False
        # Check NY session (after overlap)
        elif 16 <= hour < 21:
            session = TradingSession.NEW_YORK
            is_overlap = False
        # Check Tokyo session
        elif 0 <= hour < 9:
            session = TradingSession.TOKYO
            is_overlap = False
        # Quiet hours
        else:
            session = TradingSession.QUIET
            is_overlap = False

        config = SESSION_CONFIG[session]

        # Calculate hours remaining
        if session == TradingSession.QUIET:
            # Hours until Tokyo opens
            hours_remaining = 24 - hour if hour >= 21 else 0 - hour
            if hours_remaining < 0:
                hours_remaining += 24
        else:
            end_hour = config['end'].hour
            hours_remaining = end_hour - hour
            if hours_remaining < 0:
                hours_remaining += 24

        return SessionInfo(
            session=session,
            name=config['name'],
            is_active=session != TradingSession.QUIET,
            is_overlap=is_overlap,
            hours_remaining=hours_remaining,
            volatility_level=config['volatility'],
            recommended_pairs=config['recommended_pairs'],
            risk_multiplier=config['risk_multiplier']
        )

    def is_pair_recommended(self, pair: str, session: TradingSession = None) -> Tuple[bool, str]:
        """
        Check if a pair is recommended for the current/specified session.

        Args:
            pair: Trading pair
            session: Optional session (defaults to current)

        Returns:
            Tuple of (is_recommended, reason)
        """
        if session is None:
            session_info = self.get_current_session()
            session = session_info.session

        config = SESSION_CONFIG[session]
        recommended = config['recommended_pairs']

        pair_upper = pair.upper().replace('=X', '')

        if not recommended:
            return False, f"Quiet hours - no pairs recommended"

        if pair_upper in recommended:
            return True, f"{pair_upper} is active during {config['name']}"
        else:
            return False, f"{pair_upper} may have lower liquidity during {config['name']}"

    def should_trade_session(self, pair: str = None) -> Tuple[bool, str, float]:
        """
        Check if trading is recommended for current session.

        Args:
            pair: Optional pair to check recommendation

        Returns:
            Tuple of (should_trade, reason, risk_multiplier)
        """
        session_info = self.get_current_session()

        if session_info.session == TradingSession.QUIET:
            return False, "Quiet hours - low liquidity", 0.0

        if session_info.hours_remaining < 0.5:
            return False, f"Session ending in {session_info.hours_remaining*60:.0f} minutes", 0.5

        if pair:
            is_recommended, reason = self.is_pair_recommended(pair)
            if not is_recommended:
                return True, reason, session_info.risk_multiplier * 0.8

        return True, f"Active session: {session_info.name}", session_info.risk_multiplier

    def get_session_pairs(self, session: TradingSession = None) -> List[str]:
        """
        Get recommended pairs for a session.

        Args:
            session: Trading session (defaults to current)

        Returns:
            List of recommended pairs
        """
        if session is None:
            session_info = self.get_current_session()
            session = session_info.session

        return SESSION_CONFIG[session]['recommended_pairs']

    def get_risk_multiplier(self, pair: str = None) -> float:
        """
        Get risk multiplier for current session.

        Args:
            pair: Optional pair for pair-specific adjustment

        Returns:
            Risk multiplier (0.6 - 1.3)
        """
        session_info = self.get_current_session()
        multiplier = session_info.risk_multiplier

        if pair:
            is_recommended, _ = self.is_pair_recommended(pair)
            if not is_recommended:
                multiplier *= 0.8

        return multiplier

    def get_session_summary(self) -> Dict:
        """
        Get summary of all sessions for dashboard.

        Returns:
            Dictionary with session information
        """
        current = self.get_current_session()

        sessions_status = {}
        for session, config in SESSION_CONFIG.items():
            is_current = session == current.session
            sessions_status[session.value] = {
                'name': config['name'],
                'is_active': is_current,
                'volatility': config['volatility'],
                'pairs': config['recommended_pairs']
            }

        return {
            'current_session': current.session.value,
            'current_name': current.name,
            'is_overlap': current.is_overlap,
            'hours_remaining': current.hours_remaining,
            'volatility': current.volatility_level,
            'risk_multiplier': current.risk_multiplier,
            'recommended_pairs': current.recommended_pairs,
            'all_sessions': sessions_status
        }


# Singleton instance
_session_manager_instance = None


def get_session_manager() -> SessionManager:
    """Get singleton SessionManager instance."""
    global _session_manager_instance
    if _session_manager_instance is None:
        _session_manager_instance = SessionManager()
    return _session_manager_instance


def get_current_session() -> SessionInfo:
    """Get current session info."""
    return get_session_manager().get_current_session()


def should_trade_session(pair: str = None) -> Tuple[bool, str, float]:
    """Check if trading is allowed in current session."""
    return get_session_manager().should_trade_session(pair)


def get_session_risk_multiplier(pair: str = None) -> float:
    """Get session-based risk multiplier."""
    return get_session_manager().get_risk_multiplier(pair)
