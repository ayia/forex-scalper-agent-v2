#!/usr/bin/env python3
"""
Forex Scalper Agent V3.1 - Main Entry Point
============================================
Lightweight CLI entry point for VALIDATED strategies only.

Validated Pairs:
    - CADJPY: EMA Crossover (8/21/50), PF=1.10, R:R=2.5
    - EURGBP: Stochastic Crossover, PF=1.10, R:R=2.0 (with regime filter)
    - EURJPY: MACD+Stochastic, PF=1.27, R:R=1.2 (with regime/session/volatility filter)

Usage:
    python main.py --pairs CADJPY            # Scan CADJPY (EMA strategy)
    python main.py --pairs EURGBP            # Scan EURGBP (Stochastic strategy)
    python main.py --pairs EURJPY            # Scan EURJPY (MACD+Stochastic strategy)
    python main.py --pairs CADJPY,EURGBP,EURJPY  # Scan all validated pairs
    python main.py --pairs CADJPY --active-only  # Only BUY/SELL signals

Part of Forex Scalper Agent V3.1
"""
import sys
import logging
import argparse
import json

# Suppress all logging if --mtf-json, --optimized-cross, --eurgbp-stochastic, or --pairs flag is present
if '--mtf-json' in sys.argv or '--optimized-cross' in sys.argv or '--pairs' in sys.argv or '--eurgbp-stochastic' in sys.argv:
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
  python main.py --pairs CADJPY            Scan CADJPY (EMA Crossover)
  python main.py --pairs EURGBP            Scan EURGBP (Stochastic Crossover)
  python main.py --pairs EURJPY            Scan EURJPY (MACD+Stochastic)
  python main.py --pairs CADJPY,EURGBP,EURJPY  Scan all validated pairs
  python main.py --pairs CADJPY --active-only  Only BUY/SELL signals

Validated Pairs (backtest-proven):
  - CADJPY: EMA Crossover (8/21/50), PF=1.10, R:R=2.5
  - EURGBP: Stochastic Crossover, PF=1.10, R:R=2.0 (with regime filter)
  - EURJPY: MACD+Stochastic, PF=1.27, R:R=1.2 (with regime/session filter)
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
        '--eurgbp-stochastic',
        action='store_true',
        help='Scan EUR/GBP with Stochastic Crossover strategy and regime filter (JSON output)'
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
        version='Forex Scalper Agent V3.1'
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
        # Import dedicated scanners
        from core.eurgbp_stochastic_scanner import EURGBPStochasticScanner
        from core.eurjpy_macd_stoch_scanner import EURJPYMACDStochScanner

        # Pairs with dedicated scanners
        STOCHASTIC_PAIRS = {'EURGBP'}
        MACD_STOCH_PAIRS = {'EURJPY'}

        # All available optimized pairs (EMA + Stochastic + MACD)
        ALL_AVAILABLE_PAIRS = OPTIMIZED_PAIRS | STOCHASTIC_PAIRS | MACD_STOCH_PAIRS

        # Separate pairs by strategy type
        eurgbp_request = [p for p in custom_pairs if p in STOCHASTIC_PAIRS]
        eurjpy_request = [p for p in custom_pairs if p in MACD_STOCH_PAIRS]
        ema_request = [p for p in custom_pairs if p in OPTIMIZED_PAIRS]
        non_optimized = [p for p in custom_pairs if p not in ALL_AVAILABLE_PAIRS]

        if non_optimized:
            # Warn about pairs without optimized configs
            print(json.dumps({
                "error": f"Pairs without optimized config: {non_optimized}",
                "available_pairs": sorted(list(ALL_AVAILABLE_PAIRS)),
                "hint": "Use one of the available optimized pairs for best results"
            }, indent=2))
            return

        signals = []

        # Scan EURGBP with Stochastic scanner
        if eurgbp_request:
            eurgbp_scanner = EURGBPStochasticScanner()
            result = eurgbp_scanner.scan()
            if result and 'error' not in result:
                # Add confluence_score for sorting compatibility
                result['confluence_score'] = 75 if result['regime_tradeable'] else 25
                signals.append(result)

        # Scan EURJPY with MACD+Stochastic scanner
        if eurjpy_request:
            eurjpy_scanner = EURJPYMACDStochScanner()
            result = eurjpy_scanner.scan()
            if result and 'error' not in result:
                # confluence_score already included in scanner
                signals.append(result)

        # Scan other pairs with EMA CrossOver scanner
        if ema_request:
            scanner = OptimizedCrossScanner()
            scanner.pairs = ema_request
            for pair in ema_request:
                result = scanner.scan_pair(pair)
                if result:
                    signals.append(result)

        # Sort by confluence score descending
        signals.sort(key=lambda s: (-s.get('confluence_score', 0),))

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

    # EUR/GBP Stochastic mode - scan with regime filter
    if args.eurgbp_stochastic:
        from core.eurgbp_stochastic_scanner import EURGBPStochasticScanner
        scanner = EURGBPStochasticScanner()
        result = scanner.scan()

        # Filter blocked signals if --active-only
        if args.active_only and result.get('direction') in ['WATCH', 'BLOCKED']:
            print(json.dumps([], indent=2))
        else:
            print(json.dumps(result, indent=2))
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
