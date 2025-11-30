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
                        
                        if signal and signal.get('direction'):
                            # 5. Validation multi-TF via ConsensusValidator
                            validation = self.consensus_validator.validate(
                                signal, multi_tf_data
                            )
                            
                            if validation.get('is_valid', False):
                                # 6. Calcul du risque via RiskCalculator
                                current_price = df['Close'].iloc[-1]
                                risk_params = self.risk_calculator.calculate(
                                    pair=pair,
                                    direction=signal['direction'],
                                    entry_price=current_price,
                                    df=df
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
                                    'risk': risk_params,
                                    'stop_loss': risk_params.get('stop_loss'),
                                    'take_profit': risk_params.get('take_profit'),
                                    'risk_reward': risk_params.get('risk_reward', 0)
                                }
                                
                                # Filtrer par confidence minimum (adaptive!)
                                if complete_signal['confidence'] >= adaptive_th['confidence_threshold']:
                                    signals.append(complete_signal)
                                    
                                    # 7. Log via TradeLogger
                                    self.trade_logger.log_signal(complete_signal)
                                    
                                    logger.info(
                                        f"SIGNAL: {pair_name} {tf} "
                                        f"{signal['direction']} "
                                        f"[{strategy.name}] "
                                        f"Conf: {complete_signal['confidence']:.1f}%"
                                    )
                    
                    except Exception as e:
                        logger.debug(f"Erreur {strategy.name}/{tf}: {e}")
                        continue
        
        except Exception as e:
            logger.error(f"Erreur scan {pair_name}: {e}")
        
        return signals
    
    def scan_all(self) -> List[Dict]:
        """
        Scanne toutes les paires configurées.
        
        Returns:
            Liste de tous les signaux détectés
        """
        logger.info("\n" + "=" * 60)
        logger.info(f"SCAN COMPLET - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        all_signals = []
        
        for i, pair in enumerate(self.pairs, 1):
            logger.info(f"Scanning {pair.replace('=X', '')} ({i}/{len(self.pairs)})...")
            signals = self.scan_pair(pair)
            all_signals.extend(signals)
            
            # Pause pour éviter rate limiting yfinance
            if i < len(self.pairs):
                time.sleep(0.5)
        
        # Tri par confidence décroissante
        all_signals.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        logger.info(f"\nScan terminé: {len(all_signals)} signaux détectés")
        
        return all_signals
    
    def run_continuous(self, interval_minutes: int = 5):
        """
        Mode de scanning continu.
        
        Args:
            interval_minutes: Intervalle entre les scans
        """
        logger.info(f"Mode continu activé (intervalle: {interval_minutes} min)")
        logger.info("Appuyez sur Ctrl+C pour arrêter")
        
        try:
            while True:
                signals = self.scan_all()
                
                if signals:
                    print("\n" + "=" * 60)
                    print("SIGNAUX DÉTECTÉS:")
                    print("=" * 60)
                    for sig in signals[:10]:  # Top 10
                        print(
                            f"  {sig['pair']:8} {sig['timeframe']:4} "
                            f"{sig['direction']:5} {sig['strategy']:20} "
                            f"Conf: {sig['confidence']:.1f}% "
                            f"R:R {sig['risk_reward']:.2f}"
                        )
                
                logger.info(f"Prochain scan dans {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)
        
        except KeyboardInterrupt:
            logger.info("\nArrêt du scanner...")
    
    def get_top_signals(self, n: int = 5) -> List[Dict]:
        """
        Retourne les N meilleurs signaux actuels.
        
        Args:
            n: Nombre de signaux à retourner
        
        Returns:
            Liste des meilleurs signaux
        """
        signals = self.scan_all()
        return signals[:n]


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description='Forex Scalper Agent V2 - Scanner Multi-Stratégie'
    )
    parser.add_argument(
        '--once', 
        action='store_true',
        help='Exécuter un seul scan puis quitter'
    )
    parser.add_argument(
        '--interval', 
        type=int, 
        default=5,
        help='Intervalle de scan en minutes (défaut: 5)'
    )
    parser.add_argument(
        '--pairs',
        nargs='+',
        help='Paires spécifiques à scanner (ex: EURUSD GBPUSD)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Sortie en format JSON'
    )
    
    args = parser.parse_args()
    
    # Initialisation du scanner
    scanner = ForexScannerV2()
    
    # Override des paires si spécifiées
    if args.pairs:
        scanner.pairs = [f"{p}=X" for p in args.pairs]
    
    if args.once:
        # Mode single scan
        signals = scanner.scan_all()
        
        if args.json:
            print(json.dumps(signals, indent=2, default=str))
        else:
            print(f"\n{'='*60}")
            print(f"RÉSULTATS DU SCAN - {len(signals)} signaux")
            print(f"{'='*60}")
            
            for sig in signals:
                print(f"\n[{sig['pair']}] {sig['timeframe']} - {sig['direction'].upper()}")
                print(f"  Stratégie: {sig['strategy']}")
                print(f"  Confidence: {sig['confidence']:.1f}%")
                print(f"  Entry: {sig['entry_price']:.5f}")
                print(f"  SL: {sig['stop_loss']:.5f}")
                print(f"  TP: {sig['take_profit']:.5f}")
                print(f"  R:R: {sig['risk_reward']:.2f}")
    else:
        # Mode continu
        scanner.run_continuous(args.interval)


if __name__ == '__main__':
    main()
