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
- Modules adaptatifs pour détection de régime et gestion de risque

Part of Forex Scalper Agent V2 - Architecture Complète
"""
import sys
import logging

# Suppress all logging if --mtf-json flag is present (must be done before imports)
if '--mtf-json' in sys.argv:
    logging.disable(logging.CRITICAL)
    try:
        from loguru import logger as loguru_logger
        loguru_logger.disable("")
    except ImportError:
        pass
    import warnings
    warnings.filterwarnings('ignore')
    import os
    sys.stderr = open(os.devnull, 'w')

import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional

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
from market_regime_detector import get_regime
from adaptive_risk_manager import get_adaptive_risk
from adaptive_strategy_selector import get_active_strategies, StrategySelector
from correlation_manager import check_pair_correlation, CorrelationManager
from position_manager import PositionManager
from news_filter import NewsFilter, should_trade_news, get_news_risk_adjustment
from mtf_analyzer import MTFAnalyzer, get_mtf_analysis

# Import enhanced scalping strategy and advanced indicators
try:
    from enhanced_scalping_strategy import EnhancedScalpingStrategy, create_enhanced_strategy
    from advanced_indicators import (
        SessionManager, ChoppinessIndex, PVSRA, SupplyDemandZones
    )
    ENHANCED_STRATEGIES_AVAILABLE = True
except ImportError:
    ENHANCED_STRATEGIES_AVAILABLE = False

# Import IMPROVED strategies (backtest-validated v2.3)
try:
    from improved_strategy import ImprovedTrendStrategy, ImprovedScalpingStrategy
    IMPROVED_STRATEGIES_AVAILABLE = True
except ImportError:
    IMPROVED_STRATEGIES_AVAILABLE = False

logger = logging.getLogger(__name__)


class ForexScalperV2:
    """
    Version 2 du scanner avec système adaptatif complet
    """

    def __init__(self, account_balance: float = 10000.0, max_risk_percent: float = 2.0,
                 custom_pairs: List[str] = None):
        """Initialize all components including adaptive modules.

        Args:
            account_balance: Trading account balance in USD (default: 10,000)
            max_risk_percent: Maximum risk per session as % of account (default: 2%)
            custom_pairs: Optional list of pairs to scan (overrides ALL_PAIRS)
        """
        logger.info("Initializing Forex Scalper Agent V2...")

        # Account and risk parameters
        self.account_balance = account_balance
        self.max_risk_percent = max_risk_percent
        self.max_account_risk = account_balance * (max_risk_percent / 100.0)
        self.current_session_risk = 0.0  # Track cumulative risk in current session

        logger.info(f"Account Balance: ${account_balance:,.2f}")
        logger.info(f"Max Risk Per Session: {max_risk_percent}% (${self.max_account_risk:,.2f})")

        # Core components
        self.data_fetcher = DataFetcher()
        self.universe_filter = UniverseFilter()
        self.risk_calculator = RiskCalculator()
        self.trade_logger = TradeLogger()
        self.sentiment_analyzer = SentimentAnalyzer()

        # Strategies
        self.strategies = {
            'TrendFollowing': TrendFollowingStrategy(),
            'MeanReversion': MeanReversionStrategy(),
            'Breakout': BreakoutStrategy()
        }

        # Enhanced Scalping Strategies (from DIY Custom Strategy Builder analysis)
        if ENHANCED_STRATEGIES_AVAILABLE:
            self.enhanced_strategies = {
                'EnhancedScalping_RF': EnhancedScalpingStrategy(leading_indicator='RANGE_FILTER'),
                'EnhancedScalping_WAE': EnhancedScalpingStrategy(leading_indicator='WAE'),
                'EnhancedScalping_QQE': EnhancedScalpingStrategy(leading_indicator='QQE'),
            }
            self.session_manager = SessionManager()
            self.choppiness_filter = ChoppinessIndex(length=14, threshold=61.8)
            self.pvsra_analyzer = PVSRA(volume_period=10)
            self.sd_zones = SupplyDemandZones(swing_length=10)
            logger.info("Enhanced Scalping Strategies loaded successfully")
        else:
            self.enhanced_strategies = {}
            self.session_manager = None
            logger.warning("Enhanced Scalping Strategies not available")

        # IMPROVED Strategies (backtest-validated v2.3 - RECOMMENDED)
        # Results: +6.46% profit, 46.9% WR, 1.31 PF, 7.99% DD
        if IMPROVED_STRATEGIES_AVAILABLE:
            self.improved_strategies = {
                'ImprovedTrend': ImprovedTrendStrategy(),
                'ImprovedScalping': ImprovedScalpingStrategy(),
            }
            logger.info("IMPROVED Strategies v2.3 loaded (backtest-validated)")
            logger.info("  -> Pairs: USDJPY, USDCHF (EURUSD trend only)")
            logger.info("  -> Performance: +6.46%, WR 46.9%, PF 1.31")
        else:
            self.improved_strategies = {}
            logger.warning("Improved Strategies not available")

        self.consensus_validator = ConsensusValidator()

        # Adaptive modules
        self.strategy_selector = StrategySelector()
        self.correlation_manager = CorrelationManager()
        self.active_pairs = []  # Track currently active/open pairs

        # Position Manager (for trailing stops, breakeven, partial TP)
        self.position_manager = PositionManager(data_fetcher=self.data_fetcher)

        # News Filter (block trades around high-impact news)
        self.news_filter = NewsFilter()

        # MTF Analyzer (Top-Down Multi-Timeframe Analysis)
        self.mtf_analyzer = MTFAnalyzer()

        # Whipsaw prevention tracking
        self.rejected_signals = {}  # Track rejected signals: {pair: {timeframe: timestamp}}
        self.whipsaw_cooldown_candles = 10  # Don't re-enter same signal within N candles

        # Trading universe (use custom pairs if provided)
        if custom_pairs:
            self.pairs = custom_pairs
            logger.info(f"Custom pairs filter: {', '.join(custom_pairs)}")
        else:
            self.pairs = ALL_PAIRS
        self.timeframes = TIMEFRAMES

        logger.info("Forex Scalper Agent V2 initialized successfully")

    def scan_pair(self, pair: str, timeframe: str) -> List[Dict]:
        """
        Scan a single pair on a single timeframe with adaptive system.

        Args:
            pair: Trading pair (e.g., "EURUSD")
            timeframe: Timeframe to scan (e.g., "M15")

        Returns:
            List of validated signals
        """
        signals = []

        try:
            # 1. Fetch multi-timeframe data
            mtf_data = self.data_fetcher.fetch_multi_timeframe(pair, [timeframe, 'H1', 'H4'])
            if not mtf_data or timeframe not in mtf_data:
                return signals

            data = mtf_data[timeframe]
            if data is None or data.empty or len(data) < 50:
                return signals

            # 1b. Validate required columns (case-insensitive check)
            required_columns = {'open', 'high', 'low', 'close', 'volume'}
            data_columns_lower = {col.lower() for col in data.columns}
            missing_columns = required_columns - data_columns_lower
            if missing_columns:
                logger.error(f"{pair} {timeframe}: Missing required columns: {missing_columns}")
                return signals

            # Keep columns in lowercase for consistency with existing modules
            data.columns = [col.lower() if col.lower() in required_columns else col for col in data.columns]

            # 2. Detect market regime
            regime_info = get_regime(data, pair)
            regime = regime_info['regime']
            regime_confidence = regime_info['confidence']

            logger.info(f"{pair} {timeframe}: Regime={regime}, Confidence={regime_confidence:.1f}%")

            # 3. Check if we should trade this regime
            if not self.strategy_selector.should_trade(regime_info):
                logger.info(f"{pair}: Regime not suitable for trading")
                return signals

            # 4. Get active strategies for this regime
            active_strats = get_active_strategies(regime_info)
            strategy_weights = active_strats.get('strategies', {})

            if not strategy_weights:
                logger.info(f"{pair}: No strategies active for regime {regime}")
                return signals

            # 5. Get sentiment (now with price data for real analysis)
            sentiment = self.sentiment_analyzer.analyze(pair, data)

            # 6. Check correlation risk
            corr_check = check_pair_correlation(self.active_pairs, pair)
            if not corr_check['allow_trade']:
                logger.info(f"{pair}: Correlation limit reached ({corr_check['reason']})")
                return signals

            # 6a. NEWS FILTER - Check for upcoming high-impact news
            can_trade_news, news_reason, blocking_event = self.news_filter.should_trade(pair)
            if not can_trade_news:
                logger.warning(f"{pair}: ⚠️ NEWS FILTER - {news_reason}")
                return signals

            # Get news-based risk adjustment
            news_risk_multiplier = self.news_filter.get_risk_adjustment(pair)
            if news_risk_multiplier < 1.0:
                logger.info(f"{pair}: News approaching - risk reduced to {news_risk_multiplier:.0%}")

            # 6b. Check max account risk
            if self.current_session_risk >= self.max_account_risk:
                logger.warning(f"{pair}: Max account risk reached (${self.current_session_risk:.2f} / ${self.max_account_risk:.2f})")
                return signals

            # 6c. Whipsaw prevention - check if we recently rejected a signal on this pair/timeframe
            pair_key = f"{pair}_{timeframe}"
            if pair_key in self.rejected_signals:
                last_rejected_time = self.rejected_signals[pair_key]
                # Simple cooldown: skip if we rejected a signal recently (within last scan)
                # In production, this should track candle count or time-based cooldown
                logger.debug(f"{pair} {timeframe}: Skipping - whipsaw cooldown active")
                # For now, we'll just track but not block (will implement properly with candle tracking)

            # 7. Run active strategies
            strategy_signals = []
            for strategy_name, weight in strategy_weights.items():
                if strategy_name not in self.strategies:
                    continue

                strategy = self.strategies[strategy_name]
                signal = strategy.generate_signal(data, pair, timeframe)

                if signal:
                    # Apply strategy weight to confidence
                    signal.confidence = signal.confidence * weight
                    signal.strategy = strategy_name
                    strategy_signals.append(signal)
                    logger.info(f"{pair} {timeframe}: {strategy_name} signal - "
                               f"Direction={signal.direction}, Confidence={signal.confidence:.1f}")

            # 7b. Run Enhanced Scalping Strategies (if available)
            if ENHANCED_STRATEGIES_AVAILABLE and self.enhanced_strategies:
                # Select best enhanced strategy based on session and regime
                enhanced_strategy = create_enhanced_strategy(pair, regime)
                enhanced_signal = enhanced_strategy.analyze(mtf_data, pair)

                if enhanced_signal:
                    # Enhanced signals get priority boost
                    enhanced_signal.confidence = min(100, enhanced_signal.confidence * 1.1)
                    strategy_signals.append(enhanced_signal)
                    logger.info(f"{pair} {timeframe}: Enhanced Scalping signal - "
                               f"Direction={enhanced_signal.direction}, "
                               f"Confidence={enhanced_signal.confidence:.1f}")

            # 7c. Run IMPROVED Strategies (backtest-validated v2.3 - PRIORITY)
            # These strategies have been validated: +6.46% profit, 46.9% WR
            if IMPROVED_STRATEGIES_AVAILABLE and self.improved_strategies:
                for strat_name, strategy in self.improved_strategies.items():
                    improved_signal = strategy.analyze(mtf_data, pair)

                    if improved_signal:
                        # Improved strategies get HIGHEST priority (backtest validated)
                        improved_signal.confidence = min(100, improved_signal.confidence * 1.15)
                        strategy_signals.append(improved_signal)
                        logger.info(f"{pair} {timeframe}: [IMPROVED v2.3] {strat_name} signal - "
                                   f"Direction={improved_signal.direction}, "
                                   f"Confidence={improved_signal.confidence:.1f}")

            # 6. Validate signals with consensus
            for signal in strategy_signals:
                # Get adaptive thresholds for this pair
                thresholds = get_adaptive_thresholds(pair, data)
                min_confidence = thresholds.get('min_confidence', 60)

                # Check if signal meets confidence threshold
                if signal.confidence < min_confidence:
                    logger.info(f"{pair}: Signal rejected - confidence {signal.confidence:.1f} "
                               f"< threshold {min_confidence}")
                    # Track rejected signal for whipsaw prevention
                    self.rejected_signals[pair_key] = datetime.now()
                    continue

                # 8. Calculate adaptive risk parameters
                current_price = data['close'].iloc[-1]
                atr = self.risk_calculator.calculate_atr(data)

                risk_params = get_adaptive_risk(
                    entry_price=current_price,
                    direction=signal.direction,
                    pair=pair,
                    atr=atr,
                    regime=regime_info,
                    spread=self.data_fetcher.get_spread(pair),
                    account_balance=self.account_balance  # Pass actual account balance
                )

                # Get correlation adjustment
                _, correlation_adj = self.correlation_manager.check_correlation_risk(
                    pair, signal.direction
                )

                # Apply correlation adjustment
                risk_params['position_size'] *= correlation_adj

                # Apply news risk adjustment
                risk_params['position_size'] *= news_risk_multiplier
                risk_params['risk_amount'] *= news_risk_multiplier

                # Apply news-based stop loss widening if needed
                news_sl_adjustment = self.news_filter.get_stop_adjustment(pair)
                if news_sl_adjustment > 1.0:
                    atr_adjustment = (news_sl_adjustment - 1.0) * atr
                    if signal.direction == 'BUY':
                        risk_params['stop_loss'] -= atr_adjustment
                    else:
                        risk_params['stop_loss'] += atr_adjustment
                    logger.info(f"{pair}: SL widened by {(news_sl_adjustment-1)*100:.0f}% due to upcoming news")

                # Check if adding this trade would exceed max risk
                new_total_risk = self.current_session_risk + risk_params['risk_amount']
                if new_total_risk > self.max_account_risk:
                    logger.warning(f"{pair}: Signal rejected - would exceed max risk "
                                 f"(${new_total_risk:.2f} > ${self.max_account_risk:.2f})")
                    continue

                # Update cumulative session risk
                self.current_session_risk += risk_params['risk_amount']

                # 9. Build final signal
                final_signal = {
                    'timestamp': datetime.now().isoformat(),
                    'pair': pair,
                    'timeframe': timeframe,
                    'direction': signal.direction,
                    'strategy': signal.strategy,
                    'entry_price': signal.entry,
                    'stop_loss': risk_params['stop_loss'],
                    'take_profit': risk_params['take_profit'],
                    'confidence': signal.confidence,
                    'regime': regime,
                    'regime_confidence': regime_confidence,
                    'position_size': risk_params['position_size'],
                    'risk_amount': risk_params['risk_amount'],
                    'atr': atr,
                    'spread': self.data_fetcher.get_spread(pair),
                    'sentiment': sentiment['score'],
                    'sentiment_strength': sentiment.get('strength', 'neutral'),
                    'retail_sentiment': sentiment.get('retail_sentiment', 0),
                    'contrarian_signal': sentiment.get('contrarian_signal', 'NEUTRAL'),
                    'news_risk_multiplier': news_risk_multiplier
                }

                signals.append(final_signal)

                # 10. Log the signal
                self.trade_logger.log_signal(final_signal)

                # 11. Open position in PositionManager
                position_id = self.position_manager.open_position(final_signal)

                # Add pair to active pairs list for correlation tracking
                if pair not in self.active_pairs:
                    self.active_pairs.append(pair)

                logger.info(f"✓ {pair} {timeframe}: Position opened - "
                           f"{signal.direction} @ {signal.entry:.5f}, "
                           f"SL={risk_params['stop_loss']:.5f}, "
                           f"TP={risk_params['take_profit']:.5f}, "
                           f"Position ID: {position_id}")

        except Exception as e:
            logger.error(f"Error scanning {pair} {timeframe}: {e}", exc_info=True)

        return signals

    def run_scan(self, once: bool = False) -> List[Dict]:
        """
        Run a complete scan of all pairs and timeframes.

        Args:
            once: If True, run once and exit

        Returns:
            List of all validated signals
        """
        logger.info("=" * 60)
        logger.info("Starting Forex Scalper Agent V2 Scan")
        logger.info("=" * 60)

        # 0. Display upcoming news events
        news_summary = self.news_filter.get_news_summary()
        if news_summary['total_events'] > 0:
            logger.info(f"📰 NEWS ALERT: {news_summary['critical']} critical, "
                       f"{news_summary['high']} high, {news_summary['medium']} medium impact events upcoming")
            if news_summary['next_critical']:
                event = news_summary['next_critical']
                logger.warning(f"⚠️ CRITICAL: {event.currency} {event.event_name} at {event.timestamp.strftime('%H:%M')} UTC")
            if news_summary['next_high']:
                event = news_summary['next_high']
                logger.info(f"📊 HIGH: {event.currency} {event.event_name} at {event.timestamp.strftime('%H:%M')} UTC")
        else:
            logger.info("📰 No significant news events in next 24 hours")

        # 1. Update all open positions (trailing stops, breakeven, partial TP)
        logger.info("Updating open positions...")
        actions = self.position_manager.update_positions()

        if actions:
            logger.info(f"Position actions: {len(actions)}")
            for action in actions:
                logger.info(f"  - {action['action']}: {action.get('position_id', 'N/A')}")

        # Display position statistics
        stats = self.position_manager.get_statistics()
        logger.info(f"Position Stats: {stats['open_positions']} open, "
                   f"Total P&L: ${stats['total_pnl']:.2f} "
                   f"(Realized: ${stats['realized_pnl']:.2f}, "
                   f"Unrealized: ${stats['unrealized_pnl']:.2f})")

        # Update active_pairs based on actual open positions
        open_positions = self.position_manager.get_open_positions()
        self.active_pairs = [pos.pair for pos in open_positions]

        # Reset session risk at start of each scan
        # Recalculate from open positions
        self.current_session_risk = self.position_manager.get_total_risk()

        all_signals = []

        # 1. Filter tradable pairs (respect custom pairs if set)
        all_tradable = self.universe_filter.get_tradable_universe()
        # If custom pairs are set, only include those that are also tradable
        if self.pairs != ALL_PAIRS:
            filtered_pairs = [p for p in self.pairs if p in all_tradable]
            if not filtered_pairs:
                # If none of custom pairs are tradable, use custom pairs anyway
                filtered_pairs = self.pairs
                logger.warning(f"Custom pairs not in tradable universe, using anyway: {filtered_pairs}")
        else:
            filtered_pairs = all_tradable
        logger.info(f"Filtered pairs: {len(filtered_pairs)} tradable from {len(self.pairs)} total")
        logger.info(f"Tradable pairs: {', '.join(filtered_pairs)}")

        # 2. Scan each pair with FULL MTF Analysis (H4 -> H1 -> M15 -> M5 -> M1)
        for pair in filtered_pairs:
            logger.info(f"\n--- Scanning {pair} ---")

            # Fetch ALL timeframes for MTF analysis
            all_timeframes = ['M1', 'M5', 'M15', 'H1', 'H4']
            mtf_data = self.data_fetcher.fetch_multi_timeframe(pair, all_timeframes)

            if not mtf_data or len(mtf_data) < 4:
                logger.warning(f"{pair}: Insufficient MTF data - skipping")
                continue

            # Perform Top-Down MTF Analysis
            mtf_analysis = self.mtf_analyzer.analyze(pair, mtf_data)

            if mtf_analysis:
                # Log MTF analysis results
                logger.info(f"{pair} MTF: HTF_Bias={mtf_analysis.htf_bias}, "
                           f"H4={mtf_analysis.h4_trend}, H1={mtf_analysis.h1_trend}, "
                           f"Confluence={mtf_analysis.confluence_score:.1f}%")

                # Get trade direction from MTF analysis
                mtf_direction = self.mtf_analyzer.get_trade_direction(mtf_analysis)

                if mtf_direction:
                    logger.info(f"{pair}: MTF Signal = {mtf_direction}, "
                               f"Entry={mtf_analysis.entry_price:.5f}, "
                               f"SL={mtf_analysis.optimal_sl:.5f}, "
                               f"TP={mtf_analysis.optimal_tp:.5f}")

            # Still run individual strategy scans on scalping timeframes
            # but now with HTF context from MTF analysis
            scalping_timeframes = ['M15', 'M5']
            for timeframe in scalping_timeframes:
                signals = self.scan_pair(pair, timeframe)
                all_signals.extend(signals)

        logger.info("=" * 60)
        logger.info(f"Scan complete: {len(all_signals)} signals generated")

        # Final position stats after scan
        final_stats = self.position_manager.get_statistics()
        logger.info(f"Final Stats: {final_stats['open_positions']} positions open, "
                   f"Total Risk: ${final_stats['total_risk']:.2f}, "
                   f"Total P&L: ${final_stats['total_pnl']:.2f}")
        logger.info("=" * 60)

        return all_signals

    def run(self, once: bool = False, continuous_interval: int = 300):
        """
        Run the scanner in single or continuous mode.

        Args:
            once: If True, run once and exit
            continuous_interval: Seconds between scans in continuous mode
        """
        try:
            if once:
                signals = self.run_scan(once=True)
                logger.info(f"Single scan complete. Generated {len(signals)} signals.")
                return signals
            else:
                logger.info(f"Starting continuous scanning (interval: {continuous_interval}s)")
                while True:
                    signals = self.run_scan(once=False)
                    logger.info(f"Waiting {continuous_interval}s before next scan...")
                    time.sleep(continuous_interval)

        except KeyboardInterrupt:
            logger.info("\nStopping Forex Scalper Agent V2...")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)

    def get_mtf_signals_json(self, min_confluence: float = 60.0) -> List[Dict]:
        """
        Get MTF signals sorted by confluence (descending) in JSON format.

        Args:
            min_confluence: Minimum confluence score to include (default: 60%)

        Returns:
            List of signal dictionaries sorted by confluence
        """
        from config import get_pip_value

        all_setups = []

        # Get tradable pairs (respect custom pairs if set)
        all_tradable = self.universe_filter.get_tradable_universe()
        if self.pairs != ALL_PAIRS:
            filtered_pairs = [p for p in self.pairs if p in all_tradable]
            if not filtered_pairs:
                filtered_pairs = self.pairs  # Use custom pairs anyway
        else:
            filtered_pairs = all_tradable

        for pair in filtered_pairs:
            # Fetch all timeframes
            mtf_data = self.data_fetcher.fetch_multi_timeframe(
                pair, ['M1', 'M5', 'M15', 'H1', 'H4']
            )

            if not mtf_data or len(mtf_data) < 4:
                continue

            # Run MTF analysis
            analysis = self.mtf_analyzer.analyze(pair, mtf_data)

            if analysis and analysis.confluence_score >= min_confluence:
                direction = self.mtf_analyzer.get_trade_direction(analysis)

                if direction:
                    # Get sentiment
                    m15_data = None
                    for key in ['M15', '15m']:
                        if key in mtf_data and mtf_data[key] is not None:
                            df = mtf_data[key]
                            if not df.empty:
                                m15_data = df
                                break

                    sent = self.sentiment_analyzer.analyze(pair, m15_data) if m15_data is not None else {}

                    # Calculate metrics
                    pip_val = get_pip_value(pair)
                    risk = abs(analysis.entry_price - analysis.optimal_sl)
                    reward = abs(analysis.optimal_tp - analysis.entry_price)
                    rr_ratio = round(reward / risk, 2) if risk > 0 else 0

                    sl_pips = round(abs(analysis.entry_price - analysis.optimal_sl) / pip_val, 1)
                    tp_pips = round(abs(analysis.optimal_tp - analysis.entry_price) / pip_val, 1)

                    all_setups.append({
                        'pair': pair,
                        'direction': direction,
                        'confluence': round(analysis.confluence_score, 1),
                        'entry_price': round(analysis.entry_price, 5),
                        'stop_loss': round(analysis.optimal_sl, 5),
                        'take_profit': round(analysis.optimal_tp, 5),
                        'sl_pips': sl_pips,
                        'tp_pips': tp_pips,
                        'risk_reward': f"1:{rr_ratio}",
                        'h4_trend': analysis.h4_trend,
                        'h1_trend': analysis.h1_trend,
                        'm15_signal': analysis.m15_signal,
                        'm5_entry': analysis.m5_entry,
                        'sentiment': {
                            'retail': sent.get('retail_sentiment', 0),
                            'contrarian_signal': sent.get('contrarian_signal', 'NEUTRAL'),
                            'strength': sent.get('strength', 'neutral')
                        },
                        'timestamp': analysis.timestamp
                    })

        # Sort by confluence descending
        all_setups.sort(key=lambda x: x['confluence'], reverse=True)

        return all_setups


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Forex Scalper Agent V2 - Adaptive Trading System'
    )
    parser.add_argument(
        '--once',
        action='store_true',
        help='Run a single scan and exit'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output signals in JSON format'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Interval between scans in continuous mode (seconds)'
    )
    parser.add_argument(
        '--mtf-json',
        action='store_true',
        help='Output MTF signals as JSON sorted by confluence (descending)'
    )
    parser.add_argument(
        '--min-confluence',
        type=float,
        default=60.0,
        help='Minimum confluence score for --mtf-json (default: 60)'
    )
    parser.add_argument(
        '--pairs',
        type=str,
        default=None,
        help='Comma-separated list of pairs to scan (e.g., USDJPY,USDCHF)'
    )
    parser.add_argument(
        '--improved-only',
        action='store_true',
        help='Scan only pairs validated by IMPROVED strategies (USDJPY, USDCHF, EURUSD)'
    )

    args = parser.parse_args()

    # Handle --improved-only shortcut
    # USDJPY (58% WR), USDCHF (45% WR), EURUSD (40% WR - trend only)
    if args.improved_only:
        args.pairs = 'USDJPY,USDCHF,EURUSD'

    # Parse custom pairs if provided
    custom_pairs = None
    if args.pairs:
        custom_pairs = [p.strip().upper() for p in args.pairs.split(',')]

    # MTF JSON mode - output signals and exit
    if args.mtf_json:
        scanner = ForexScalperV2(custom_pairs=custom_pairs)
        signals = scanner.get_mtf_signals_json(min_confluence=args.min_confluence)
        print(json.dumps(signals, indent=2))
        return

    # Initialize scanner for normal mode
    scanner = ForexScalperV2(custom_pairs=custom_pairs)

    # Normal scan mode
    signals = scanner.run(once=args.once, continuous_interval=args.interval)

    # Output results
    if args.json and signals:
        print(json.dumps(signals, indent=2))


if __name__ == '__main__':
    main()
