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
from market_regime_detector import get_regime
from adaptive_risk_manager import get_adaptive_risk
from adaptive_strategy_selector import get_active_strategies, StrategySelector
from correlation_manager import check_pair_correlation, CorrelationManager
from position_manager import PositionManager

logger = logging.getLogger(__name__)


class ForexScalperV2:
    """
    Version 2 du scanner avec système adaptatif complet
    """

    def __init__(self, account_balance: float = 10000.0, max_risk_percent: float = 2.0):
        """Initialize all components including adaptive modules.

        Args:
            account_balance: Trading account balance in USD (default: 10,000)
            max_risk_percent: Maximum risk per session as % of account (default: 2%)
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

        self.consensus_validator = ConsensusValidator()

        # Adaptive modules
        self.strategy_selector = StrategySelector()
        self.correlation_manager = CorrelationManager()
        self.active_pairs = []  # Track currently active/open pairs

        # Position Manager (for trailing stops, breakeven, partial TP)
        self.position_manager = PositionManager(data_fetcher=self.data_fetcher)

        # Whipsaw prevention tracking
        self.rejected_signals = {}  # Track rejected signals: {pair: {timeframe: timestamp}}
        self.whipsaw_cooldown_candles = 10  # Don't re-enter same signal within N candles

        # Trading universe
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

            # 5. Get sentiment
            sentiment = self.sentiment_analyzer.analyze(pair)

            # 6. Check correlation risk
            corr_check = check_pair_correlation(self.active_pairs, pair)
            if not corr_check['allow_trade']:
                logger.info(f"{pair}: Correlation limit reached ({corr_check['reason']})")
                return signals

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
                    'sentiment': sentiment['score']
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

        # 1. Filter tradable pairs
        filtered_pairs = self.universe_filter.get_tradable_universe()
        logger.info(f"Filtered pairs: {len(filtered_pairs)} tradable from {len(self.pairs)} total")
        logger.info(f"Tradable pairs: {', '.join(filtered_pairs)}")

        # 2. Scan each pair
        for pair in filtered_pairs:
            logger.info(f"\n--- Scanning {pair} ---")

            # Scan on primary scalping timeframes
            primary_timeframes = ['M15', 'M5']  # Focus on these for scalping

            for timeframe in primary_timeframes:
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

    args = parser.parse_args()

    # Initialize and run scanner
    scanner = ForexScalperV2()
    signals = scanner.run(once=args.once, continuous_interval=args.interval)

    # Output results
    if args.json and signals:
        print(json.dumps(signals, indent=2))


if __name__ == '__main__':
    main()
