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
    from market_regime_detector import get_regime
from adaptive_risk_manager import get_adaptive_risk
from adaptive_strategy_selector import get_active_strategies, StrategySelector
from correlation_manager import check_pair_correlation, CorrelationManager
)
logger = logging.getLogger(__name__)


class ForexScannerV2:
    """
    Scanner principal qui orchestre tous les agents du système.
    Utilise TOUS les modules créés dans notre architecture V2.
    """
    
    def __init__(self):
        """Initialise le scanner avec tous les modules."""
        logger.info("=" * 60)
        logger.info("FOREX SCALPER AGENT V2 - INITIALISATION")
        logger.info("=" * 60)
        
        # Initialisation de tous les composants
        self.data_fetcher = DataFetcher()
        self.universe_filter = UniverseFilter()
        self.risk_calculator = RiskCalculator()
        self.consensus_validator = ConsensusValidator()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.trade_logger = TradeLogger()
        
        # Chargement des stratégies
        self.strategies: List[BaseStrategy] = [
            TrendFollowingStrategy(),

                    # NEW: Initialize adaptive modules
        self.strategy_selector = StrategySelector()
        self.correlation_manager = CorrelationManager()
        self.active_pairs = []  # Track currently active/open pairs
            MeanReversionStrategy(),
            BreakoutStrategy()
        ]
        
        # Configuration depuis config.py
        self.pairs = [f"{p}=X" for p in ALL_PAIRS]
        self.timeframes = TIMEFRAMES
        logger.info(f"Paires configurées: {len(self.pairs)}")
        logger.info(f"Timeframes: {self.timeframes}")
        logger.info(f"Stratégies actives: {[s.name for s in self.strategies]}")
        logger.info("Initialisation complète!")
    
    def scan_pair(self, pair: str) -> List[Dict]:
        """
        Analyse complète d'une paire avec tous les modules.
        
        Args:
            pair: Symbole de la paire (ex: 'EURUSD=X')
        
        Returns:
            Liste des signaux détectés
        """
        signals = []
        pair_name = pair.replace('=X', '')
        
        try:
            # Get adaptive thresholds for this pair (dynamic based on session, volatility, pair)
            adaptive_th = get_adaptive_thresholds(pair)
            
            # 1. Vérification avec UniverseFilter
            if not self.universe_filter.is_tradable(pair):
                logger.debug(f"{pair_name}: Filtré par UniverseFilter")

                        # NEW: Detect market regime
        regime = None
        if multi_tf_data.get('1h'):  # Use 1H for regime detection
            regime = get_regime(multi_tf_data['1h'], pair_name)
            logger.debug(f"{pair_name} Regime: {regime['regime']} (conf: {regime['confidence']}%)")
        
        # NEW: Check if we should trade this regime
        if regime and not self.strategy_selector.should_trade(regime):
            logger.debug(f"{pair_name}: Regime not suitable for trading")
            return signals
        
        # NEW: Get active strategies for this regime
        if regime:
            active_strats = get_active_strategies(regime)
            strategy_weights = active_strats['strategies']
        else:
            strategy_weights = {s.name.lower().replace('strategy', '').strip(): 1.0 
                               for s in self.strategies}
        
        # NEW: Check correlation risk
        corr_check = check_pair_correlation(self.active_pairs, pair_name)
        if not corr_check['allow_trade']:
            logger.debug(f"{pair_name}: Correlation limit reached ({corr_check['reason']})")
            return signals
                return signals
            
            # 2. Récupération des données multi-timeframe via DataFetcher
            multi_tf_data = {}
            for tf in self.timeframes:
                df = self.data_fetcher.fetch(pair, tf)
                if df is not None and len(df) > 50:
                    multi_tf_data[tf] = df
            
            if not multi_tf_data:
                logger.warning(f"{pair_name}: Pas de données disponibles")
                return signals
            
            # 3. Analyse de sentiment via SentimentAnalyzer
            sentiment = self.sentiment_analyzer.analyze(pair_name)
            
            # 4. Analyse avec chaque stratégie
            for strategy in self.strategies:
                for tf, df in multi_tf_data.items():
                    try:
                        signal = strategy.analyze(df, pair)

                                    # NEW: Check strategy weight
            strategy_name_key = strategy.name.lower().replace('strategy', '').strip()
            weight = strategy_weights.get(strategy_name_key, 0.5)
            
            # Skip if weight too low
            if weight < 0.4:
                continue
            
                        if signal and signal.get('direction'):
                            # 5. Validation multi-TF via ConsensusValidator
                            validation = self.consensus_validator.validate(
                                signal, multi_tf_data
                            )
                            
                            if validation.get('is_valid', False):
                                # 6. Calcul du risque via RiskCalculator
                                current_price = df['Close'].iloc[-1]
                                # NEW: Use adaptive risk manager
                                risk_params = get_adaptive_risk(
                                    entry_price=current_price,
                                    direction=signal['direction'],
                                    pair=pair_name,
                                    atr=df['ATR'].iloc[-1] if 'ATR' in df else 0.001,
                                    regime=regime,
                                    spread=None  # Can add real spread here
                                )
                                
                                # Construction du signal complet
                                complete_signal = {
                                    'timestamp': datetime.now().isoformat(),
                                    'pair': pair_name,
                                    'timeframe': tf,
                                    'strategy': strategy.name,
                                    'direction': signal['direction'],
                                    'entry_price': current_price,
                                    'confidence': signal.get('confidence', 0),
                                    'validation_score': validation.get('score', 0),
                                    'sentiment': sentiment,
                                                                    'regime': regime['regime'] if regime else 'unknown',
                                'regime_confidence': regime['confidence'] if regime else 0,
                                'confidence': signal.get('confidence', 50) * weight,  # Apply regime weight
