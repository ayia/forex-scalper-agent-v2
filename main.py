#!/usr/bin/env python3
"""
Forex Scalper Agent V4.0 - Main Entry Point
============================================
Lightweight CLI entry point for VALIDATED strategies only.

VALIDATION STATUS (as of 2024-12):
    Data Source: Twelve Data API (1H data back to 2017)
    Anti-overfitting tests: Monte Carlo, Walk-Forward, Parameter Jitter
    Periods tested: COVID, Ukraine War, Banking Crisis, Fed Hiking, etc.

    VALIDATED (4 pairs):
    - CADJPY: EMA Crossover (8/21/50), 7/9 periods, WFE=309%
    - EURCHF: Mean Reversion Z-Score, PF=1.97, WR=50.8%, Monte Carlo 100%
    - EURGBP: RSI Divergence + Stochastic Double, PF=1.10-1.31, Score=85.5/100
    - EURJPY: Range Breakout + Mean Reversion, PF=1.58, WR=39.3%, Monte Carlo 100% (NEW!)

Usage:
    python main.py --pairs CADJPY            # Scan CADJPY (validated)
    python main.py --pairs EURCHF            # Scan EURCHF (validated)
    python main.py --pairs EURGBP            # Scan EURGBP (validated)
    python main.py --pairs EURJPY            # Scan EURJPY (validated - NEW!)
    python main.py --pairs CADJPY,EURCHF,EURGBP,EURJPY  # Scan all validated pairs
    python main.py --pairs CADJPY --active-only  # Only BUY/SELL signals

Part of Forex Scalper Agent V4.0
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
        description='Forex Scalper Agent V4.0 - Adaptive Trading System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --pairs CADJPY              Scan CADJPY (VALIDATED - EMA Crossover)
  python main.py --pairs EURCHF              Scan EURCHF (VALIDATED - Mean Reversion)
  python main.py --pairs EURGBP              Scan EURGBP (VALIDATED - RSI Divergence + Stochastic)
  python main.py --pairs EURJPY              Scan EURJPY (VALIDATED - Range Breakout + Mean Reversion)
  python main.py --pairs CADJPY,EURCHF,EURGBP,EURJPY  Scan all validated pairs
  python main.py --pairs CADJPY --active-only  Only BUY/SELL signals

Validated Pairs (tested on 8 historical periods including COVID, Ukraine War):
  - CADJPY: EMA Crossover (8/21/50), 7/9 periods pass, WFE=309%
  - EURCHF: Mean Reversion Z-Score, PF=1.97, WR=50.8%, Monte Carlo 100%
  - EURGBP: RSI Divergence + Stochastic Double, PF=1.10-1.31, Score=85.5/100
  - EURJPY: Range Breakout + Mean Reversion, PF=1.58, WR=39.3%, Monte Carlo 100%
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
        version='Forex Scalper Agent V4.0 (CADJPY + EURCHF + EURGBP + EURJPY validated)'
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
        from core.eurchf_mean_reversion_scanner import EURCHFMeanReversionScanner
        from core.eurgbp_validated_scanner import EURGBPValidatedScanner
        from core.eurjpy_validated_scanner import EURJPYValidatedScanner

        # Pairs with dedicated scanners
        MEAN_REVERSION_PAIRS = {'EURCHF'}
        EURGBP_PAIRS = {'EURGBP'}
        EURJPY_PAIRS = {'EURJPY'}

        # All available optimized pairs (EMA + Mean Reversion + EURGBP + EURJPY)
        ALL_AVAILABLE_PAIRS = OPTIMIZED_PAIRS | MEAN_REVERSION_PAIRS | EURGBP_PAIRS | EURJPY_PAIRS

        # Separate pairs by strategy type
        eurchf_request = [p for p in custom_pairs if p in MEAN_REVERSION_PAIRS]
        eurgbp_request = [p for p in custom_pairs if p in EURGBP_PAIRS]
        eurjpy_request = [p for p in custom_pairs if p in EURJPY_PAIRS]
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

        # Scan EURCHF with Mean Reversion scanner
        if eurchf_request:
            eurchf_scanner = EURCHFMeanReversionScanner()
            result = eurchf_scanner.scan()
            if result and 'error' not in result:
                # confluence_score already included in scanner
                signals.append(result)

        # Scan EURGBP with validated scanner (RSI Divergence + Stochastic Double)
        if eurgbp_request:
            eurgbp_scanner = EURGBPValidatedScanner()
            result = eurgbp_scanner.scan()
            if result and 'error' not in result:
                # Add confluence_score based on signal quality
                if result.get('direction') in ['BUY', 'SELL']:
                    # Active signal: base 70 + adjustments
                    base_score = 70
                    if result.get('regime_tradeable', False):
                        base_score += 10
                    if result.get('session_tradeable', False):
                        base_score += 10
                    # Adjust by position multiplier
                    base_score *= result.get('position_multiplier', 1.0)
                    result['confluence_score'] = min(100, base_score)
                else:
                    # WATCH or BLOCKED
                    result['confluence_score'] = 40 if result.get('direction') == 'BLOCKED' else 30
                signals.append(result)

        # Scan EURJPY with validated scanner (Range Breakout + Mean Reversion)
        if eurjpy_request:
            eurjpy_scanner = EURJPYValidatedScanner()
            result = eurjpy_scanner.scan()
            if result and 'error' not in result:
                # Add confluence_score based on signal quality
                if result.get('direction') in ['BUY', 'SELL']:
                    # Active signal: base 70 + adjustments
                    base_score = 70
                    if result.get('regime_tradeable', False):
                        base_score += 10
                    if result.get('session_tradeable', False):
                        base_score += 10
                    # Adjust by position multiplier
                    base_score *= result.get('position_multiplier', 1.0)
                    result['confluence_score'] = min(100, base_score)
                else:
                    # WATCH or BLOCKED
                    result['confluence_score'] = 40 if result.get('direction') == 'BLOCKED' else 30
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
