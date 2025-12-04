#!/usr/bin/env python3
"""
Forex Scalper Agent V2 - Main Entry Point
==========================================
Lightweight CLI entry point that delegates to the core scanner.

Usage:
    python main.py --pairs CADJPY            # Scan CADJPY with optimized strategy (JSON output)
    python main.py --pairs CADJPY,EURJPY     # Scan multiple optimized pairs
    python main.py --pairs CADJPY --active-only  # Only BUY/SELL signals
    python main.py --once                    # Single scan (legacy)
    python main.py --json                    # JSON output (legacy)
    python main.py --mtf-json                # MTF signals as JSON
    python main.py --optimized-cross         # All 6 profitable cross pairs
    python main.py --interval 300            # Continuous mode

Part of Forex Scalper Agent V2 - Complete Architecture
"""
import sys
import logging
import argparse
import json

# Suppress all logging if --mtf-json, --optimized-cross, or --pairs flag is present
if '--mtf-json' in sys.argv or '--optimized-cross' in sys.argv or '--pairs' in sys.argv:
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

from config import LOG_CONFIG

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_CONFIG.get('log_level', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Forex Scalper Agent V2 - Adaptive Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --pairs CADJPY            Scan CADJPY with optimized strategy (JSON)
  python main.py --pairs CADJPY,EURJPY     Scan multiple optimized pairs
  python main.py --pairs CADJPY --active-only  Only BUY/SELL signals
  python main.py --optimized-cross         All 6 profitable cross pairs
  python main.py --once                    Run a single scan (legacy mode)
  python main.py --mtf-json                Get MTF signals as JSON
  python main.py --interval 300            Continuous mode (5 min interval)

Optimized Pairs (validated 2-year backtest):
  - CADJPY: Best performer, PF=1.13, +46% ROI/2 years
  - EURCAD, EURJPY, GBPJPY, CHFJPY, GBPAUD

Strategy: EMA Crossover (8/21/50) with pair-specific configs
        """
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
        help='Interval between scans in continuous mode (seconds, default: 300)'
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
    parser.add_argument(
        '--optimized-cross',
        action='store_true',
        help='Scan 10 profitable cross pairs with optimized configs (JSON output)'
    )
    parser.add_argument(
        '--active-only',
        action='store_true',
        help='With --optimized-cross: show only active signals (BUY/SELL), exclude WATCH'
    )
    parser.add_argument(
        '--balance',
        type=float,
        default=10000.0,
        help='Account balance in USD (default: 10000)'
    )
    parser.add_argument(
        '--max-risk',
        type=float,
        default=2.0,
        help='Maximum risk per session as %% of account (default: 2.0)'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='Forex Scalper Agent V2.6'
    )

    args = parser.parse_args()

    # Handle --improved-only shortcut
    if args.improved_only:
        args.pairs = 'USDJPY,USDCHF,EURUSD'

    # Parse custom pairs if provided
    custom_pairs = None
    if args.pairs:
        custom_pairs = [p.strip().upper() for p in args.pairs.split(',')]

    # Import optimized cross scanner for pairs mode
    from core.optimized_cross_scanner import OptimizedCrossScanner, OPTIMAL_CONFIGS

    # List of pairs that have optimized configs
    OPTIMIZED_PAIRS = set(OPTIMAL_CONFIGS.keys())

    # --pairs mode: Use optimized strategy for validated pairs (JSON output)
    if args.pairs and custom_pairs:
        # Check which pairs have optimized configs
        optimized_request = [p for p in custom_pairs if p in OPTIMIZED_PAIRS]
        non_optimized = [p for p in custom_pairs if p not in OPTIMIZED_PAIRS]

        if non_optimized:
            # Warn about pairs without optimized configs
            print(json.dumps({
                "error": f"Pairs without optimized config: {non_optimized}",
                "available_pairs": list(OPTIMIZED_PAIRS),
                "hint": "Use one of the available optimized pairs for best results"
            }, indent=2))
            return

        # Scan only the requested pairs with optimized configs
        scanner = OptimizedCrossScanner()
        scanner.pairs = optimized_request  # Override default pairs

        signals = []
        for pair in optimized_request:
            result = scanner.scan_pair(pair)
            if result:
                signals.append(result)

        # Sort by confluence score descending
        signals.sort(key=lambda s: (-s['confluence_score'],))

        # Filter to active signals only if --active-only is set
        if args.active_only:
            signals = [s for s in signals if s['direction'] in ['BUY', 'SELL']]

        print(json.dumps(signals, indent=2))
        return

    # Import scanner for legacy modes (after logging setup)
    from core.scanner import ForexScalperV2

    # MTF JSON mode - output signals and exit
    if args.mtf_json:
        scanner = ForexScalperV2(
            account_balance=args.balance,
            max_risk_percent=args.max_risk,
            custom_pairs=custom_pairs
        )
        signals = scanner.get_mtf_signals_json(min_confluence=args.min_confluence)
        print(json.dumps(signals, indent=2))
        return

    # Optimized Cross Pairs mode - scan all 6 profitable pairs with optimal configs
    if args.optimized_cross:
        scanner = OptimizedCrossScanner()
        signals = scanner.scan_all()

        # Filter to active signals only if --active-only is set
        if args.active_only:
            signals = [s for s in signals if s['direction'] in ['BUY', 'SELL']]

        print(json.dumps(signals, indent=2))
        return

    # Initialize scanner
    scanner = ForexScalperV2(
        account_balance=args.balance,
        max_risk_percent=args.max_risk,
        custom_pairs=custom_pairs
    )

    # Run scanner
    signals = scanner.run(once=args.once, continuous_interval=args.interval)

    # Output results
    if args.json and signals:
        print(json.dumps(signals, indent=2))


if __name__ == '__main__':
    main()
