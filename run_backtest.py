#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Backtest - Script de validation de strategie
=================================================
Execute un backtest complet sur vos strategies et genere un rapport detaille.

Usage:
    python run_backtest.py
    python run_backtest.py --pairs EURUSD,GBPUSD --days 30
    python run_backtest.py --strategy enhanced
"""

import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import local modules
from data_fetcher import DataFetcher
from backtester import BacktestEngine, BacktestResult, MonteCarloSimulator
from config import ALL_PAIRS, MAJOR_PAIRS


def fetch_historical_data(
    pairs: List[str],
    timeframes: List[str],
    days: int = 30
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Recupere les donnees historiques pour le backtest.

    Args:
        pairs: Liste des paires a tester
        timeframes: Liste des timeframes
        days: Nombre de jours d'historique

    Returns:
        Dict structure {pair: {timeframe: DataFrame}}
    """
    import time as time_module

    print(f"\n[DATA] Recuperation des donnees historiques ({days} jours)...")

    fetcher = DataFetcher()
    data = {}

    # Calcul du nombre de barres necessaires par timeframe
    bars_needed = {
        'M1': days * 24 * 60,
        'M5': days * 24 * 12,
        'M15': days * 24 * 4,
        'M30': days * 24 * 2,
        'H1': days * 24,
        'H4': days * 6,
        'D': days
    }

    request_count = 0
    for pair in pairs:
        print(f"   Chargement {pair}...", end=" ", flush=True)
        data[pair] = {}

        for tf in timeframes:
            # Rate limiting: pause every 8 requests (Twelve Data limit)
            if request_count > 0 and request_count % 7 == 0:
                print("(pause API)...", end=" ", flush=True)
                time_module.sleep(62)  # Wait for rate limit reset

            bars = min(bars_needed.get(tf, 500), 500)  # API limit
            df = fetcher.fetch_ohlcv(pair, tf, bars)
            request_count += 1

            if df is not None and len(df) > 50:
                # Remove timezone info to avoid comparison issues
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                data[pair][tf] = df

            time_module.sleep(0.5)  # Small delay between requests

        loaded_tfs = list(data[pair].keys())
        print(f"OK ({', '.join(loaded_tfs)})")

    return data


# Import des vraies strategies du projet
try:
    from trend_following import TrendFollowingStrategy
    from mean_reversion import MeanReversionStrategy
    REAL_STRATEGIES_AVAILABLE = True
except ImportError:
    REAL_STRATEGIES_AVAILABLE = False

# Import des strategies ameliorees
try:
    from improved_strategy import ImprovedTrendStrategy, ImprovedScalpingStrategy
    IMPROVED_STRATEGIES_AVAILABLE = True
except ImportError:
    IMPROVED_STRATEGIES_AVAILABLE = False


class SimpleScalpingStrategy:
    """
    Strategie de scalping simplifiee pour le backtest.
    Basee sur EMA crossover + RSI confirmation.
    """

    def __init__(self, name: str = "SimpleScalping"):
        self.name = name
        self.ema_fast = 8
        self.ema_slow = 21
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70

    def generate_signal(self, df: pd.DataFrame, pair: str, timeframe: str) -> Optional[Dict]:
        """Genere un signal de trading."""
        if len(df) < 50:
            return None

        # Normaliser les colonnes
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        # Calculer EMA
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()

        # Calculer RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).ewm(span=self.rsi_period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=self.rsi_period, adjust=False).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Dernieres valeurs
        current = df.iloc[-1]
        previous = df.iloc[-2]

        # ATR pour SL/TP
        high = df['high']
        low = df['low']
        close = df['close']
        tr = pd.concat([high - low, abs(high - close.shift()), abs(low - close.shift())], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]

        pip_value = 0.01 if 'JPY' in pair else 0.0001

        signal = None

        # Signal BUY: EMA cross up + RSI pas surbought
        if (previous['ema_fast'] <= previous['ema_slow'] and
            current['ema_fast'] > current['ema_slow'] and
            current['rsi'] < self.rsi_overbought and
            current['rsi'] > self.rsi_oversold):

            entry = current['close']
            sl = entry - (atr * 1.5)
            tp = entry + (atr * 2.5)

            # Verifier SL max 15 pips
            sl_pips = abs(entry - sl) / pip_value
            if sl_pips <= 15:
                signal = {
                    'direction': 'BUY',
                    'entry_price': entry,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'confidence': 70
                }

        # Signal SELL: EMA cross down + RSI pas survendu
        elif (previous['ema_fast'] >= previous['ema_slow'] and
              current['ema_fast'] < current['ema_slow'] and
              current['rsi'] > self.rsi_oversold and
              current['rsi'] < self.rsi_overbought):

            entry = current['close']
            sl = entry + (atr * 1.5)
            tp = entry - (atr * 2.5)

            sl_pips = abs(sl - entry) / pip_value
            if sl_pips <= 15:
                signal = {
                    'direction': 'SELL',
                    'entry_price': entry,
                    'stop_loss': sl,
                    'take_profit': tp,
                    'confidence': 70
                }

        return signal


def calculate_strategy_grade(result: BacktestResult) -> tuple:
    """
    Calcule une note sur 10 pour la strategie basee sur les metriques.

    Criteres:
    - Profit Factor (25%)
    - Win Rate (20%)
    - Sharpe Ratio (20%)
    - Max Drawdown (15%)
    - Expectancy (10%)
    - Consistency (10%)

    Returns:
        (note, details)
    """
    scores = {}

    # Profit Factor (25 points max)
    if result.profit_factor >= 2.0:
        scores['profit_factor'] = 25
    elif result.profit_factor >= 1.5:
        scores['profit_factor'] = 20
    elif result.profit_factor >= 1.2:
        scores['profit_factor'] = 15
    elif result.profit_factor >= 1.0:
        scores['profit_factor'] = 10
    else:
        scores['profit_factor'] = 0

    # Win Rate (20 points max)
    if result.win_rate >= 60:
        scores['win_rate'] = 20
    elif result.win_rate >= 50:
        scores['win_rate'] = 15
    elif result.win_rate >= 40:
        scores['win_rate'] = 10
    elif result.win_rate >= 30:
        scores['win_rate'] = 5
    else:
        scores['win_rate'] = 0

    # Sharpe Ratio (20 points max)
    if result.sharpe_ratio >= 2.0:
        scores['sharpe'] = 20
    elif result.sharpe_ratio >= 1.5:
        scores['sharpe'] = 16
    elif result.sharpe_ratio >= 1.0:
        scores['sharpe'] = 12
    elif result.sharpe_ratio >= 0.5:
        scores['sharpe'] = 8
    else:
        scores['sharpe'] = 0

    # Max Drawdown (15 points max) - Lower is better
    if result.max_drawdown_pct <= 5:
        scores['drawdown'] = 15
    elif result.max_drawdown_pct <= 10:
        scores['drawdown'] = 12
    elif result.max_drawdown_pct <= 15:
        scores['drawdown'] = 9
    elif result.max_drawdown_pct <= 20:
        scores['drawdown'] = 6
    else:
        scores['drawdown'] = 0

    # Expectancy (10 points max)
    if result.expectancy > 20:
        scores['expectancy'] = 10
    elif result.expectancy > 10:
        scores['expectancy'] = 8
    elif result.expectancy > 5:
        scores['expectancy'] = 6
    elif result.expectancy > 0:
        scores['expectancy'] = 4
    else:
        scores['expectancy'] = 0

    # Consistency - Based on profit factor and consecutive losses (10 points max)
    if result.consecutive_losses <= 3 and result.profit_factor >= 1.2:
        scores['consistency'] = 10
    elif result.consecutive_losses <= 5 and result.profit_factor >= 1.0:
        scores['consistency'] = 7
    elif result.consecutive_losses <= 7:
        scores['consistency'] = 4
    else:
        scores['consistency'] = 0

    total = sum(scores.values())
    grade = total / 10  # Convert to /10 scale

    return grade, scores


def print_backtest_report(result: BacktestResult, grade: float, grade_details: dict):
    """Affiche un rapport de backtest formaté."""

    print("\n" + "=" * 70)
    print("                    RAPPORT DE BACKTEST")
    print("=" * 70)

    # Periode
    print(f"\nPeriode: {result.start_date.strftime('%Y-%m-%d')} -> {result.end_date.strftime('%Y-%m-%d')}")
    print(f"Balance initiale: ${result.initial_balance:,.2f}")
    print(f"Balance finale:   ${result.final_balance:,.2f}")

    # Performance
    print("\n" + "-" * 70)
    print("PERFORMANCE")
    print("-" * 70)
    pnl_pct = (result.net_profit / result.initial_balance) * 100
    print(f"Profit Net:     ${result.net_profit:>10,.2f} ({pnl_pct:>+.2f}%)")
    print(f"Profit Total:   ${result.total_profit:>10,.2f}")
    print(f"Perte Totale:   ${result.total_loss:>10,.2f}")
    print(f"Profit Factor:  {result.profit_factor:>10.2f}")

    # Statistiques des trades
    print("\n" + "-" * 70)
    print("STATISTIQUES DES TRADES")
    print("-" * 70)
    print(f"Total Trades:   {result.total_trades:>10}")
    print(f"Trades Gagnants:{result.winning_trades:>10} ({result.win_rate:.1f}%)")
    print(f"Trades Perdants:{result.losing_trades:>10}")
    print(f"Gain Moyen:     ${result.avg_win:>10.2f}")
    print(f"Perte Moyenne:  ${result.avg_loss:>10.2f}")
    print(f"Plus Grand Gain:${result.largest_win:>10.2f}")
    print(f"Plus Grande Perte:${result.largest_loss:>9.2f}")

    # Metriques de risque
    print("\n" + "-" * 70)
    print("METRIQUES DE RISQUE")
    print("-" * 70)
    print(f"Max Drawdown:   ${result.max_drawdown:>10.2f} ({result.max_drawdown_pct:.2f}%)")
    print(f"Sharpe Ratio:   {result.sharpe_ratio:>10.2f}")
    print(f"Sortino Ratio:  {result.sortino_ratio:>10.2f}")
    print(f"Calmar Ratio:   {result.calmar_ratio:>10.2f}")
    print(f"Ratio R/R Moyen:{result.avg_rr_ratio:>10.2f}")
    print(f"Expectancy:     ${result.expectancy:>10.2f}")
    print(f"Pertes Consecutives Max: {result.consecutive_losses}")

    # Note finale
    print("\n" + "=" * 70)
    print("                         NOTE FINALE")
    print("=" * 70)

    # Afficher la note avec style
    if grade >= 8:
        rating = "EXCELLENT"
        bar = "[##########]"
    elif grade >= 6:
        rating = "BON"
        bar = "[########  ]"
    elif grade >= 5:
        rating = "MOYEN"
        bar = "[######    ]"
    elif grade >= 3:
        rating = "FAIBLE"
        bar = "[####      ]"
    else:
        rating = "MAUVAIS"
        bar = "[##        ]"

    print(f"\n   NOTE: {grade:.1f} / 10  {bar}  {rating}")

    print("\n   Details du scoring:")
    print(f"   - Profit Factor:  {grade_details['profit_factor']:>3}/25")
    print(f"   - Win Rate:       {grade_details['win_rate']:>3}/20")
    print(f"   - Sharpe Ratio:   {grade_details['sharpe']:>3}/20")
    print(f"   - Max Drawdown:   {grade_details['drawdown']:>3}/15")
    print(f"   - Expectancy:     {grade_details['expectancy']:>3}/10")
    print(f"   - Consistency:    {grade_details['consistency']:>3}/10")
    print(f"   -------------------------")
    print(f"   TOTAL:            {sum(grade_details.values()):>3}/100")

    # Recommandations
    print("\n" + "-" * 70)
    print("RECOMMANDATIONS")
    print("-" * 70)

    recommendations = []

    if result.profit_factor < 1.0:
        recommendations.append("[CRITIQUE] Strategie non profitable - NE PAS utiliser en live!")
    elif result.profit_factor < 1.2:
        recommendations.append("[ATTENTION] Profit factor faible - Ameliorer les filtres d'entree")

    if result.win_rate < 40:
        recommendations.append("[ATTENTION] Win rate faible - Revoir les criteres d'entree")

    if result.max_drawdown_pct > 20:
        recommendations.append("[ATTENTION] Drawdown eleve - Reduire le risque par trade")

    if result.sharpe_ratio < 1.0:
        recommendations.append("[INFO] Sharpe ratio < 1 - Rendement ajuste au risque faible")

    if result.consecutive_losses > 5:
        recommendations.append("[INFO] Trop de pertes consecutives - Ajouter des filtres de marche")

    if result.total_trades < 30:
        recommendations.append("[INFO] Echantillon faible - Augmenter la periode de test")

    if not recommendations:
        recommendations.append("[OK] Strategie solide - Valider avec paper trading avant live")

    for rec in recommendations:
        print(f"   {rec}")

    print("\n" + "=" * 70)


def run_monte_carlo(result: BacktestResult, simulations: int = 1000):
    """Execute une simulation Monte Carlo."""

    if not result.trades:
        print("\n[!] Pas assez de trades pour Monte Carlo")
        return

    print(f"\n[MONTE CARLO] Simulation avec {simulations} iterations...")

    simulator = MonteCarloSimulator(result.trades, result.initial_balance)
    mc_result = simulator.run(simulations)

    print("\n" + "-" * 70)
    print("SIMULATION MONTE CARLO")
    print("-" * 70)
    print(f"Balance finale moyenne:  ${mc_result['mean_final_balance']:>10,.2f}")
    print(f"Balance finale mediane:  ${mc_result['median_final_balance']:>10,.2f}")
    print(f"Ecart-type:              ${mc_result['std_final_balance']:>10,.2f}")
    print(f"Percentile 5% (pire):    ${mc_result['percentile_5']:>10,.2f}")
    print(f"Percentile 95% (meilleur):${mc_result['percentile_95']:>9,.2f}")
    print(f"Drawdown max moyen:      {mc_result['mean_max_drawdown']:>10.2f}%")
    print(f"Pire drawdown possible:  {mc_result['worst_case_drawdown']:>10.2f}%")
    print(f"Probabilite de perte:    {mc_result['probability_of_loss']:>10.1f}%")


def main():
    """Point d'entree principal."""

    parser = argparse.ArgumentParser(description='Run Backtest - Validation de strategie')
    parser.add_argument('--pairs', type=str, default='USDJPY,USDCHF,EURUSD',
                       help='Paires optimales validees par backtest (WR > 40%%)')
    parser.add_argument('--days', type=int, default=30,
                       help='Nombre de jours d\'historique')
    parser.add_argument('--balance', type=float, default=10000,
                       help='Balance initiale')
    parser.add_argument('--risk', type=float, default=0.01,
                       help='Risque par trade (0.01 = 1%)')
    parser.add_argument('--monte-carlo', action='store_true',
                       help='Executer simulation Monte Carlo')
    parser.add_argument('--strategy', type=str, default='all',
                       choices=['simple', 'trend', 'improved', 'all'],
                       help='Strategie: simple, trend, improved (amelioree), ou all')
    parser.add_argument('--improved', action='store_true',
                       help='Utiliser les strategies ameliorees (recommande)')

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("          FOREX SCALPER AGENT V2 - BACKTEST ENGINE")
    print("=" * 70)

    # Parser les paires
    pairs = [p.strip().upper() for p in args.pairs.split(',')]
    timeframes = ['M15', 'H1']  # Scalping timeframes

    print(f"\nConfiguration:")
    print(f"   Paires:    {', '.join(pairs)}")
    print(f"   Timeframes: {', '.join(timeframes)}")
    print(f"   Historique: {args.days} jours")
    print(f"   Balance:    ${args.balance:,.2f}")
    print(f"   Risque:     {args.risk * 100:.1f}%")

    # Recuperer les donnees
    try:
        data = fetch_historical_data(pairs, timeframes, args.days)
    except Exception as e:
        print(f"\n[ERREUR] Impossible de recuperer les donnees: {e}")
        return 1

    # Verifier qu'on a des donnees
    total_bars = sum(len(df) for pair_data in data.values() for df in pair_data.values())
    if total_bars < 100:
        print("\n[ERREUR] Pas assez de donnees pour le backtest")
        return 1

    print(f"\n[OK] {total_bars} bougies chargees au total")

    # Creer le backtester
    print("\n[BACKTEST] Demarrage de la simulation...")

    engine = BacktestEngine(
        initial_balance=args.balance,
        risk_per_trade=args.risk,
        spread_pips=1.0,
        commission_per_trade=0,
        max_positions=3
    )

    # Charger les donnees
    engine.load_data(data)

    # Ajouter les strategies selon le choix
    strategies_added = []

    # Mode ameliore (--improved ou --strategy improved)
    use_improved = args.improved or args.strategy == 'improved'

    if use_improved and IMPROVED_STRATEGIES_AVAILABLE:
        print("   [MODE AMELIORE] Utilisation des strategies optimisees")
        try:
            strategy_improved_trend = ImprovedTrendStrategy()
            engine.add_strategy(strategy_improved_trend)
            strategies_added.append("ImprovedTrend")

            strategy_improved_scalp = ImprovedScalpingStrategy()
            engine.add_strategy(strategy_improved_scalp)
            strategies_added.append("ImprovedScalping")
        except Exception as e:
            print(f"   [!] Strategies ameliorees non disponibles: {e}")
    else:
        # Strategies originales
        if args.strategy in ['simple', 'all']:
            strategy_simple = SimpleScalpingStrategy("EMA_RSI_Scalping")
            engine.add_strategy(strategy_simple)
            strategies_added.append("EMA_RSI_Scalping")

        if args.strategy in ['trend', 'all'] and REAL_STRATEGIES_AVAILABLE:
            try:
                strategy_trend = TrendFollowingStrategy()
                engine.add_strategy(strategy_trend)
                strategies_added.append("TrendFollowing")
            except Exception as e:
                print(f"   [!] TrendFollowing non disponible: {e}")

    if not strategies_added:
        print("\n[ERREUR] Aucune strategie disponible")
        return 1

    print(f"   Strategies: {', '.join(strategies_added)}")

    # Executer le backtest
    result = engine.run(pairs=pairs, timeframes=timeframes)

    # Calculer la note
    grade, grade_details = calculate_strategy_grade(result)

    # Afficher le rapport
    print_backtest_report(result, grade, grade_details)

    # Monte Carlo optionnel
    if args.monte_carlo and result.total_trades >= 20:
        run_monte_carlo(result)

    # Exporter les trades
    if result.trades:
        export_path = f"backtest_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        engine.export_trades(export_path)
        print(f"\n[EXPORT] Trades exportes vers: {export_path}")

    print("\n[FIN] Backtest termine.")

    return 0 if grade >= 5 else 1


if __name__ == "__main__":
    sys.exit(main())
