"""Filters module exports."""
from .news_filter import (
    NewsFilter,
    NewsImpact,
    EconomicEvent,
    get_news_filter,
    should_trade_news,
    get_news_risk_adjustment
)
from .session_manager import (
    SessionManager,
    SessionInfo,
    TradingSession,
    SESSION_CONFIG,
    get_session_manager,
    get_current_session,
    should_trade_session,
    get_session_risk_multiplier
)

__all__ = [
    # News Filter
    'NewsFilter',
    'NewsImpact',
    'EconomicEvent',
    'get_news_filter',
    'should_trade_news',
    'get_news_risk_adjustment',
    # Session Manager
    'SessionManager',
    'SessionInfo',
    'TradingSession',
    'SESSION_CONFIG',
    'get_session_manager',
    'get_current_session',
    'should_trade_session',
    'get_session_risk_multiplier',
]
