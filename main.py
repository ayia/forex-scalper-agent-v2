#!/usr/bin/env python3
"""
Forex Scalper Agent V2 - Main Entry Point
==========================================
Lightweight CLI entry point that delegates to the core scanner.

Usage:
    python main.py --once                    # Single scan
    python main.py --json                    # JSON output
    python main.py --mtf-json                # MTF signals as JSON
    python main.py --pairs USDJPY,USDCHF     # Custom pairs
    python main.py --improved-only           # Validated pairs only
    python main.py --interval 300            # Continuous mode

Part of Forex Scalper Agent V2 - Complete Architecture
"""
import sys
import logging
import argparse
import json

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
  python main.py --once                    Run a single scan
  python main.py --json --once             Single scan with JSON output
  python main.py --mtf-json                Get MTF signals as JSON
  python main.py --pairs USDJPY,USDCHF     Scan specific pairs
  python main.py --improved-only           Scan backtest-validated pairs
  python main.py --interval 300            Continuous mode (5 min interval)

Strategies:
  - TrendFollowing: EMA stack + MACD crossover
  - MeanReversion: Bollinger Bands + RSI extremes
  - Breakout: Donchian Channels + Volume
  - ImprovedTrend v2.3: Backtest-validated (+6.46% profit)
  - ImprovedScalping v2.3: For USDJPY, USDCHF
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
        version='Forex Scalper Agent V2.3'
    )

    args = parser.parse_args()

    # Handle --improved-only shortcut
    if args.improved_only:
        args.pairs = 'USDJPY,USDCHF,EURUSD'

    # Parse custom pairs if provided
    custom_pairs = None
    if args.pairs:
        custom_pairs = [p.strip().upper() for p in args.pairs.split(',')]

    # Import scanner (after logging setup to respect --mtf-json)
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
