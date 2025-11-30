#!/usr/bin/env python3
"""
Forex Scalper Agent V2 - Scanner Principal
Ce scanner orchestre TOUS les modules du système:
- DataFetcher pour récupérer les données via yfinance
- UniverseFilter pour filtrer les paires tradables
- TrendFollowing, MeanReversion, Breakout strategies
- ConsensusValidator pour validation multi-timeframe
- RiskCalculator pour SL/TP dynamiques
- SentimentAnalyzer pour l'analyse de sentiment
- TradeLogger pour l'enregistrement des signaux

Part of Forex Scalper Agent V2 - Architecture Complète
"""
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional
import logging

# Import de TOUS nos modules existants
from config import (
    STRATEGY_PARAMS, RISK_PARAMS,
    LOG_CONFIG, ALL_PAIRS, TIMEFRAMES
)
from data_fetcher import DataFetcher
from universe_filter import UniverseFilter
from base_strategy import BaseStrategy
from trend_following import TrendFollowingStrategy
from mean_reversion import MeanReversionStrategy
from breakout import BreakoutStrategy
from risk_calculator import RiskCalculator
from consensus_validator import ConsensusValidator
from sentiment_analyzer import SentimentAnalyzer
from trade_logger import TradeLogger
from adaptive_thresholds import get_adaptive_thresholds

# Configuration du logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG['log_level']),
)
     from  market_regime_detector import get_regime
 from adaptive_risk_manager import get_adaptive_risk
 from adaptive_strategy_selector import get_active_strategies, StrategySelector
 from correlation_manager import check_pair_correlation, CorrelationManager
logger = logging.getLogger(__name__)
