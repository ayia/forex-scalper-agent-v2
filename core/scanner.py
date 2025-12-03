#!/usr/bin/env python3
"""
Forex Scalper Agent V2 - Core Scanner
=====================================
Main scanner that orchestrates ALL system modules:
- DataFetcher for fetching data via yfinance
- UniverseFilter for filtering tradable pairs
- TrendFollowing, MeanReversion, Breakout strategies
- ConsensusValidator for multi-timeframe validation
- RiskCalculator for dynamic SL/TP
- SentimentAnalyzer for sentiment analysis
- TradeLogger for signal logging
- Adaptive modules for regime detection and risk management

Part of Forex Scalper Agent V2 - Complete Architecture
"""
import sys
import logging
import time
import json
from datetime import datetime
from typing import Dict, List, Optional

# Import from reorganized modules
from config import (
    STRATEGY_PARAMS, RISK_PARAMS,
    LOG_CONFIG, ALL_PAIRS, TIMEFRAMES,
    get_pip_value
)
from core.data_fetcher import DataFetcher
from core.universe_filter import UniverseFilter

from strategies import (
    TrendFollowingStrategy,
    MeanReversionStrategy,
    BreakoutStrategy
)

from analysis import (
    get_regime,
    MTFAnalyzer,
    get_mtf_analysis,
    SentimentAnalyzer,
    ConsensusValidator
)

from risk import (
    get_adaptive_thresholds,
    get_adaptive_risk,
    check_pair_correlation,
    CorrelationManager
)

from filters import (
    NewsFilter,
    should_trade_news,
    get_news_risk_adjustment,
    get_current_session
)

# Try to import improved strategies
try:
    from strategies.improved import ImprovedTrendStrategy, ImprovedScalpingStrategy
    IMPROVED_STRATEGIES_AVAILABLE = True
except ImportError:
    IMPROVED_STRATEGIES_AVAILABLE = False

# Legacy imports for backward compatibility with original modules
try:
    from risk_calculator import RiskCalculator
except ImportError:
    RiskCalculator = None

try:
    from trade_logger import TradeLogger
except ImportError:
    TradeLogger = None

try:
    from position_manager import PositionManager
except ImportError:
    PositionManager = None

try:
    from adaptive_strategy_selector import get_active_strategies, StrategySelector
except ImportError:
    get_active_strategies = None
    StrategySelector = None

logger = logging.getLogger(__name__)


class ForexScalperV2:
    """
    Version 2 of the scanner with complete adaptive system
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
        self.current_session_risk = 0.0

        logger.info(f"Account Balance: ${account_balance:,.2f}")
        logger.info(f"Max Risk Per Session: {max_risk_percent}% (${self.max_account_risk:,.2f})")

        # Core components
        self.data_fetcher = DataFetcher()
        self.universe_filter = UniverseFilter()
        self.risk_calculator = RiskCalculator() if RiskCalculator else None
        self.trade_logger = TradeLogger() if TradeLogger else None
        self.sentiment_analyzer = SentimentAnalyzer()

        # Strategies
        self.strategies = {
            'TrendFollowing': TrendFollowingStrategy(),
            'MeanReversion': MeanReversionStrategy(),
            'Breakout': BreakoutStrategy()
        }

        # IMPROVED Strategies (backtest-validated v2.3)
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
        self.strategy_selector = StrategySelector() if StrategySelector else None
        self.correlation_manager = CorrelationManager()
        self.active_pairs = []

        # Position Manager
        self.position_manager = PositionManager(data_fetcher=self.data_fetcher) if PositionManager else None

        # News Filter
        self.news_filter = NewsFilter()

        # MTF Analyzer
        self.mtf_analyzer = MTFAnalyzer()

        # Whipsaw prevention
        self.rejected_signals = {}
        self.whipsaw_cooldown_candles = 10

        # Trading universe
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

            # 1b. Validate required columns
            required_columns = {'open', 'high', 'low', 'close', 'volume'}
            data_columns_lower = {col.lower() for col in data.columns}
            missing_columns = required_columns - data_columns_lower
            if missing_columns:
                logger.error(f"{pair} {timeframe}: Missing required columns: {missing_columns}")
                return signals

            data.columns = [col.lower() if col.lower() in required_columns else col for col in data.columns]

            # 2. Detect market regime
            regime_info = get_regime(data, pair)
            regime = regime_info['regime']
            regime_confidence = regime_info['confidence']

            logger.info(f"{pair} {timeframe}: Regime={regime}, Confidence={regime_confidence:.1f}%")

            # 3. Check if we should trade this regime
            if self.strategy_selector and not self.strategy_selector.should_trade(regime_info):
                logger.info(f"{pair}: Regime not suitable for trading")
                return signals

            # 4. Get active strategies for this regime
            if get_active_strategies:
                active_strats = get_active_strategies(regime_info)
                strategy_weights = active_strats.get('strategies', {})
            else:
                strategy_weights = {'TrendFollowing': 1.0, 'MeanReversion': 0.5, 'Breakout': 0.5}

            if not strategy_weights:
                logger.info(f"{pair}: No strategies active for regime {regime}")
                return signals

            # 5. Get sentiment
            sentiment = self.sentiment_analyzer.analyze(pair, data)

            # 6. Check correlation risk
            corr_check = check_pair_correlation(self.active_pairs, pair)
            if not corr_check['allow_trade']:
                logger.info(f"{pair}: Correlation limit reached ({corr_check['reason']})")
                return signals

            # 6a. NEWS FILTER
            can_trade_news, news_reason, blocking_event = self.news_filter.should_trade(pair)
            if not can_trade_news:
                logger.warning(f"{pair}: NEWS FILTER - {news_reason}")
                return signals

            news_risk_multiplier = self.news_filter.get_risk_adjustment(pair)
            if news_risk_multiplier < 1.0:
                logger.info(f"{pair}: News approaching - risk reduced to {news_risk_multiplier:.0%}")

            # 6b. Check max account risk
            if self.current_session_risk >= self.max_account_risk:
                logger.warning(f"{pair}: Max account risk reached")
                return signals

            # 6c. Whipsaw prevention
            pair_key = f"{pair}_{timeframe}"

            # 7. Run active strategies
            strategy_signals = []
            for strategy_name, weight in strategy_weights.items():
                if strategy_name not in self.strategies:
                    continue

                strategy = self.strategies[strategy_name]
                signal = strategy.generate_signal(data, pair, timeframe)

                if signal:
                    signal.confidence = signal.confidence * weight
                    signal.strategy = strategy_name
                    strategy_signals.append(signal)
                    logger.info(f"{pair} {timeframe}: {strategy_name} signal - "
                               f"Direction={signal.direction}, Confidence={signal.confidence:.1f}")

            # 7c. Run IMPROVED Strategies
            if IMPROVED_STRATEGIES_AVAILABLE and self.improved_strategies:
                for strat_name, strategy in self.improved_strategies.items():
                    improved_signal = strategy.analyze(mtf_data, pair)

                    if improved_signal:
                        improved_signal.confidence = min(100, improved_signal.confidence * 1.15)
                        strategy_signals.append(improved_signal)
                        logger.info(f"{pair} {timeframe}: [IMPROVED v2.3] {strat_name} signal - "
                                   f"Direction={improved_signal.direction}, "
                                   f"Confidence={improved_signal.confidence:.1f}")

            # 8. Validate signals with consensus
            for signal in strategy_signals:
                thresholds = get_adaptive_thresholds(pair, data)
                min_confidence = thresholds.get('min_confidence', 60)

                if signal.confidence < min_confidence:
                    logger.info(f"{pair}: Signal rejected - confidence {signal.confidence:.1f} "
                               f"< threshold {min_confidence}")
                    self.rejected_signals[pair_key] = datetime.now()
                    continue

                # 9. Calculate adaptive risk parameters
                current_price = data['close'].iloc[-1]
                atr = self._calculate_atr(data)

                risk_params = get_adaptive_risk(
                    entry_price=current_price,
                    direction=signal.direction,
                    pair=pair,
                    atr=atr,
                    regime=regime_info,
                    spread=self.data_fetcher.get_spread(pair) if hasattr(self.data_fetcher, 'get_spread') else None,
                    account_balance=self.account_balance
                )

                # Get correlation adjustment
                corr_adj = self.correlation_manager.get_correlation_adjustment(pair, self.active_pairs)

                # Apply adjustments
                risk_params['position_size'] *= corr_adj
                risk_params['position_size'] *= news_risk_multiplier
                risk_params['risk_amount'] *= news_risk_multiplier

                # News-based stop loss widening
                news_sl_adjustment = self.news_filter.get_stop_adjustment(pair)
                if news_sl_adjustment > 1.0:
                    atr_adjustment = (news_sl_adjustment - 1.0) * atr
                    if signal.direction == 'BUY':
                        risk_params['sl'] -= atr_adjustment
                    else:
                        risk_params['sl'] += atr_adjustment
                    logger.info(f"{pair}: SL widened by {(news_sl_adjustment-1)*100:.0f}% due to upcoming news")

                # Check risk limits
                new_total_risk = self.current_session_risk + risk_params['risk_amount']
                if new_total_risk > self.max_account_risk:
                    logger.warning(f"{pair}: Signal rejected - would exceed max risk")
                    continue

                self.current_session_risk += risk_params['risk_amount']

                # 10. Build final signal
                final_signal = {
                    'timestamp': datetime.now().isoformat(),
                    'pair': pair,
                    'timeframe': timeframe,
                    'direction': signal.direction,
                    'strategy': signal.strategy,
                    'entry_price': signal.entry,
                    'stop_loss': risk_params['sl'],
                    'take_profit': risk_params['tp'],
                    'confidence': signal.confidence,
                    'regime': regime,
                    'regime_confidence': regime_confidence,
                    'position_size': risk_params['position_size'],
                    'risk_amount': risk_params['risk_amount'],
                    'atr': atr,
                    'spread': self.data_fetcher.get_spread(pair) if hasattr(self.data_fetcher, 'get_spread') else None,
                    'sentiment': sentiment['score'],
                    'sentiment_strength': sentiment.get('strength', 'neutral'),
                    'retail_sentiment': sentiment.get('retail_sentiment', 0),
                    'contrarian_signal': sentiment.get('contrarian_signal', 'NEUTRAL'),
                    'news_risk_multiplier': news_risk_multiplier
                }

                signals.append(final_signal)

                # Log the signal
                if self.trade_logger:
                    self.trade_logger.log_signal(final_signal)

                # Open position
                if self.position_manager:
                    position_id = self.position_manager.open_position(final_signal)
                else:
                    position_id = "N/A"

                if pair not in self.active_pairs:
                    self.active_pairs.append(pair)

                logger.info(f"✓ {pair} {timeframe}: Position opened - "
                           f"{signal.direction} @ {signal.entry:.5f}, "
                           f"SL={risk_params['sl']:.5f}, "
                           f"TP={risk_params['tp']:.5f}, "
                           f"Position ID: {position_id}")

        except Exception as e:
            logger.error(f"Error scanning {pair} {timeframe}: {e}", exc_info=True)

        return signals

    def _calculate_atr(self, data, period: int = 14) -> float:
        """Calculate ATR."""
        high = data['high']
        low = data['low']
        close = data['close']

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        import pandas as pd
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr.iloc[-1] if not atr.empty else 0.0

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

        # Display session info
        session_info = get_current_session()
        logger.info(f"Current Session: {session_info.name} (Risk Mult: {session_info.risk_multiplier})")

        # Display upcoming news events
        news_summary = self.news_filter.get_news_summary()
        if news_summary['total_events'] > 0:
            logger.info(f"NEWS ALERT: {news_summary['critical']} critical, "
                       f"{news_summary['high']} high, {news_summary['medium']} medium impact events")
            if news_summary['next_critical']:
                event = news_summary['next_critical']
                logger.warning(f"CRITICAL: {event.currency} {event.event_name} at {event.timestamp.strftime('%H:%M')} UTC")
        else:
            logger.info("No significant news events in next 24 hours")

        # Update positions
        if self.position_manager:
            logger.info("Updating open positions...")
            actions = self.position_manager.update_positions()

            if actions:
                logger.info(f"Position actions: {len(actions)}")
                for action in actions:
                    logger.info(f"  - {action['action']}: {action.get('position_id', 'N/A')}")

            stats = self.position_manager.get_statistics()
            logger.info(f"Position Stats: {stats['open_positions']} open, "
                       f"Total P&L: ${stats['total_pnl']:.2f}")

            open_positions = self.position_manager.get_open_positions()
            self.active_pairs = [pos.pair for pos in open_positions]
            self.current_session_risk = self.position_manager.get_total_risk()

        all_signals = []

        # Filter tradable pairs
        all_tradable = self.universe_filter.get_tradable_universe()
        if self.pairs != ALL_PAIRS:
            filtered_pairs = [p for p in self.pairs if p in all_tradable]
            if not filtered_pairs:
                filtered_pairs = self.pairs
                logger.warning(f"Custom pairs not in tradable universe, using anyway: {filtered_pairs}")
        else:
            filtered_pairs = all_tradable

        logger.info(f"Filtered pairs: {len(filtered_pairs)} tradable from {len(self.pairs)} total")
        logger.info(f"Tradable pairs: {', '.join(filtered_pairs)}")

        # Scan each pair with MTF Analysis
        for pair in filtered_pairs:
            logger.info(f"\n--- Scanning {pair} ---")

            all_timeframes = ['M1', 'M5', 'M15', 'H1', 'H4']
            mtf_data = self.data_fetcher.fetch_multi_timeframe(pair, all_timeframes)

            if not mtf_data or len(mtf_data) < 4:
                logger.warning(f"{pair}: Insufficient MTF data - skipping")
                continue

            # MTF Analysis
            mtf_analysis = self.mtf_analyzer.analyze(pair, mtf_data)

            if mtf_analysis:
                logger.info(f"{pair} MTF: HTF_Bias={mtf_analysis.htf_bias}, "
                           f"H4={mtf_analysis.h4_trend}, H1={mtf_analysis.h1_trend}, "
                           f"Confluence={mtf_analysis.confluence_score:.1f}%")

                mtf_direction = self.mtf_analyzer.get_trade_direction(mtf_analysis)

                if mtf_direction:
                    logger.info(f"{pair}: MTF Signal = {mtf_direction}, "
                               f"Entry={mtf_analysis.entry_price:.5f}, "
                               f"SL={mtf_analysis.optimal_sl:.5f}, "
                               f"TP={mtf_analysis.optimal_tp:.5f}")

            # Scan scalping timeframes
            scalping_timeframes = ['M15', 'M5']
            for timeframe in scalping_timeframes:
                signals = self.scan_pair(pair, timeframe)
                all_signals.extend(signals)

        logger.info("=" * 60)
        logger.info(f"Scan complete: {len(all_signals)} signals generated")

        if self.position_manager:
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
        all_setups = []

        # Get tradable pairs
        all_tradable = self.universe_filter.get_tradable_universe()
        if self.pairs != ALL_PAIRS:
            filtered_pairs = [p for p in self.pairs if p in all_tradable]
            if not filtered_pairs:
                filtered_pairs = self.pairs
        else:
            filtered_pairs = all_tradable

        for pair in filtered_pairs:
            mtf_data = self.data_fetcher.fetch_multi_timeframe(
                pair, ['M1', 'M5', 'M15', 'H1', 'H4']
            )

            if not mtf_data or len(mtf_data) < 4:
                continue

            analysis = self.mtf_analyzer.analyze(pair, mtf_data)

            if analysis and analysis.confluence_score >= min_confluence:
                direction = self.mtf_analyzer.get_trade_direction(analysis)

                if direction:
                    m15_data = None
                    for key in ['M15', '15m']:
                        if key in mtf_data and mtf_data[key] is not None:
                            df = mtf_data[key]
                            if not df.empty:
                                m15_data = df
                                break

                    sent = self.sentiment_analyzer.analyze(pair, m15_data) if m15_data is not None else {}

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

        all_setups.sort(key=lambda x: x['confluence'], reverse=True)

        return all_setups
