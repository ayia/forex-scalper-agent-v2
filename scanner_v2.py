"""Forex Scanner V2 - Main Orchestrator Module
Integrates all modules for multi-pair, multi-timeframe analysis.
"""
import time
from typing import List, Dict, Optional
from datetime import datetime

# Import all modules
from config import ALL_PAIRS, TIMEFRAMES, RISK_PARAMS
from data_fetcher import DataFetcher
from base_strategy import BaseStrategy, Signal
from trend_following import TrendFollowingStrategy
from mean_reversion import MeanReversionStrategy
from risk_calculator import RiskCalculator
from trade_logger import TradeLogger


class ForexScannerV2:
    """Main scanner that orchestrates all modules."""
    
    def __init__(self):
        # Initialize all modules
        self.data_fetcher = DataFetcher()
        self.risk_calculator = RiskCalculator()
        self.trade_logger = TradeLogger()
        
        # Load strategies
        self.strategies: List[BaseStrategy] = [
            TrendFollowingStrategy(),
            MeanReversionStrategy(),
        ]
        
        self.signals: List[Signal] = []
    
    def scan_pair(self, pair: str) -> List[Signal]:
        """Scan a single pair with all strategies."""
        signals = []
        
        # Fetch multi-timeframe data
        data = self.data_fetcher.fetch_multi_timeframe(pair)
        if not data:
            return signals
        
        # Run all strategies
        for strategy in self.strategies:
            signal = strategy.analyze(data, pair)
            if signal:
                # Validate risk/reward
                if self.risk_calculator.validate_risk_reward(
                    signal.entry_price, signal.stop_loss, signal.take_profit
                ):
                    signals.append(signal)
                    self.trade_logger.log_signal(signal)
        
        return signals
    
    def scan_all(self) -> List[Signal]:
        """Scan all configured pairs."""
        all_signals = []
        
        for pair in ALL_PAIRS:
            print(f"Scanning {pair}...")
            signals = self.scan_pair(pair)
            all_signals.extend(signals)
            time.sleep(0.5)  # Rate limiting
        
        self.signals = all_signals
        return all_signals
    
    def get_top_signals(self, n: int = 5) -> List[Signal]:
        """Get top N signals by confidence."""
        sorted_signals = sorted(
            self.signals, key=lambda x: x.confidence, reverse=True
        )
        return sorted_signals[:n]


def main():
    """Main entry point."""
    print("=" * 50)
    print("Forex Scalper Agent V2")
    print("Multi-Pair, Multi-Timeframe, Multi-Strategy Scanner")
    print("=" * 50)
    
    scanner = ForexScannerV2()
    
    print(f"\nScanning {len(ALL_PAIRS)} pairs...")
    signals = scanner.scan_all()
    
    print(f"\n{'='*50}")
    print(f"Found {len(signals)} signals")
    
    if signals:
        print("\nTop Signals:")
        for i, sig in enumerate(scanner.get_top_signals(5), 1):
            print(f"{i}. {sig.pair} {sig.direction} @ {sig.entry_price:.5f}")
            print(f"   SL: {sig.stop_loss:.5f} TP: {sig.take_profit:.5f}")
            print(f"   Confidence: {sig.confidence}% Strategy: {sig.strategy}")


if __name__ == "__main__":
    main()
