"""
Forex Scalper Agent - Main Entry Point
This module provides the main ForexScalperAgent class for running the trading system.
"""

import sys
from typing import Dict, List
from config import STRATEGY_PARAMS, RISK_PARAMS, ALL_PAIRS, TIMEFRAMES
from data_fetcher import DataFetcher
from universe_filter import UniverseFilter
from breakout import BreakoutStrategy
from trend_following import TrendFollowingStrategy
from mean_reversion import MeanReversionStrategy
from consensus_validator import ConsensusValidator
from risk_calculator import RiskCalculator
from trade_logger import TradeLogger
from sentiment_analyzer import SentimentAnalyzer


class ForexScalperAgent:
    """
    Main agent for coordinating forex scalping operations.
    """
    
    def __init__(self):
        """Initialize the ForexScalperAgent with all necessary components."""
        self.data_fetcher = DataFetcher()
        self.universe_filter = UniverseFilter()
        self.risk_calculator = RiskCalculator()
        self.trade_logger = TradeLogger()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Initialize strategies
        self.strategies = [
            BreakoutStrategy(),
            TrendFollowingStrategy(),
            MeanReversionStrategy()
        ]
        
        self.consensus_validator = ConsensusValidator(self.strategies)
        
        # Trading pairs
        self.pairs = [f"{p}=X" for p in ALL_PAIRS]
        self.timeframes = TIMEFRAMES
    
    def run_scan(self, once: bool = False) -> List[Dict]:
        """
        Run a scan of all trading pairs.
        
        Args:
            once: If True, run once and exit. If False, run continuously.
        
        Returns:
            List of trade signals found during the scan.
        """
        print("Starting forex scalper scan...")
        signals = []
        
        # Filter universe
        filtered_pairs = self.universe_filter.filter_universe(self.pairs)
        print(f"Filtered pairs: {len(filtered_pairs)} from {len(self.pairs)}")
        
        # Scan each pair
        for pair in filtered_pairs:
            for timeframe in self.timeframes:
                try:
                    # Fetch data
                    data = self.data_fetcher.fetch_data(pair, timeframe)
                    if data is None or data.empty:
                        continue
                    
                    # Get sentiment
                    sentiment = self.sentiment_analyzer.analyze(pair)
                    
                    # Run strategies
                    strategy_signals = []
                    for strategy in self.strategies:
                        signal = strategy.generate_signal(data, pair, timeframe)
                        if signal:
                            strategy_signals.append(signal)
                    
                    # Validate with consensus
                    if strategy_signals:
                        validated = self.consensus_validator.validate(strategy_signals)
                        if validated:
                            # Calculate risk
                            risk = self.risk_calculator.calculate_position_size(
                                pair, validated['entry_price'], validated['stop_loss']
                            )
                            validated['position_size'] = risk
                            
                            # Log trade
                            self.trade_logger.log_signal(validated)
                            signals.append(validated)
                            
                            print(f"Signal found: {validated}")
                
                except Exception as e:
                    print(f"Error scanning {pair} {timeframe}: {e}")
                    continue
        
        print(f"Scan complete. Found {len(signals)} signals.")
        return signals
    
    def run(self, once: bool = False):
        """
        Run the agent.
        
        Args:
            once: If True, run once and exit. If False, run continuously.
        """
        try:
            signals = self.run_scan(once=once)
            
            if once:
                print(f"Single scan complete. Found {len(signals)} signals.")
                return signals
            else:
                # For continuous mode, would implement loop here
                print("Continuous mode not yet implemented.")
                return signals
        
        except KeyboardInterrupt:
            print("\nStopping forex scalper agent...")
            sys.exit(0)
        except Exception as e:
            print(f"Error running agent: {e}")
            sys.exit(1)


def main():
    """Main entry point for the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Forex Scalper Agent')
    parser.add_argument('--once', action='store_true', help='Run once and exit')
    parser.add_argument('--json', action='store_true', help='Output in JSON format')
    
    args = parser.parse_args()
    
    agent = ForexScalperAgent()
    signals = agent.run(once=args.once)
    
    if args.json:
        import json
        print(json.dumps(signals, indent=2))


if __name__ == '__main__':
    main()
