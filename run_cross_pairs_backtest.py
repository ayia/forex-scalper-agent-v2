#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross Pairs Backtest - Analyse Complete sur 2 ans
==================================================
Backtest de 15 paires cross avec rapport detaille.

Paires testees:
EUR/GBP, EUR/JPY, EUR/CHF, EUR/AUD, EUR/CAD
GBP/JPY, GBP/CHF, GBP/AUD, GBP/CAD
AUD/JPY, CAD/JPY, CHF/JPY
AUD/CAD, AUD/NZD, NZD/JPY
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import time as time_module
import yfinance as yf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paires cross a tester
CROSS_PAIRS = [
    'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD',
    'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD',
    'AUDJPY', 'CADJPY', 'CHFJPY',
    'AUDCAD', 'AUDNZD', 'NZDJPY'
]

# Mapping vers symboles yfinance
YFINANCE_SYMBOLS = {
    'EURGBP': 'EURGBP=X', 'EURJPY': 'EURJPY=X', 'EURCHF': 'EURCHF=X',
    'EURAUD': 'EURAUD=X', 'EURCAD': 'EURCAD=X', 'GBPJPY': 'GBPJPY=X',
    'GBPCHF': 'GBPCHF=X', 'GBPAUD': 'GBPAUD=X', 'GBPCAD': 'GBPCAD=X',
    'AUDJPY': 'AUDJPY=X', 'CADJPY': 'CADJPY=X', 'CHFJPY': 'CHFJPY=X',
    'AUDCAD': 'AUDCAD=X', 'AUDNZD': 'AUDNZD=X', 'NZDJPY': 'NZDJPY=X'
}


class BacktestStrategy:
    """Strategie de backtest combinant EMA + RSI + MACD."""

    def __init__(self):
        self.ema_fast = 8
        self.ema_slow = 21
        self.ema_trend = 50
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule tous les indicateurs."""
        df = df.copy()

        # EMAs
        df['ema_fast'] = df['Close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['Close'].ewm(span=self.ema_slow, adjust=False).mean()
        df['ema_trend'] = df['Close'].ewm(span=self.ema_trend, adjust=False).mean()

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(span=self.rsi_period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=self.rsi_period, adjust=False).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ATR
        high = df['High']
        low = df['Low']
        close = df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()

        # Bollinger Bands
        df['bb_mid'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
        df['bb_lower'] = df['bb_mid'] - (bb_std * 2)

        return df

    def generate_signals(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Genere les signaux de trading."""
        df = self.calculate_indicators(df)
        df['signal'] = 0
        df['sl'] = 0.0
        df['tp'] = 0.0

        pip_value = 0.01 if 'JPY' in pair else 0.0001

        for i in range(2, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(current['atr']) or current['atr'] == 0:
                continue

            # BUY Signal
            buy_conditions = (
                prev['ema_fast'] <= prev['ema_slow'] and
                current['ema_fast'] > current['ema_slow'] and
                current['Close'] > current['ema_trend'] and
                current['rsi'] > 30 and current['rsi'] < 70 and
                current['macd_hist'] > 0
            )

            # SELL Signal
            sell_conditions = (
                prev['ema_fast'] >= prev['ema_slow'] and
                current['ema_fast'] < current['ema_slow'] and
                current['Close'] < current['ema_trend'] and
                current['rsi'] > 30 and current['rsi'] < 70 and
                current['macd_hist'] < 0
            )

            if buy_conditions:
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('sl')] = current['Close'] - (current['atr'] * 1.5)
                df.iloc[i, df.columns.get_loc('tp')] = current['Close'] + (current['atr'] * 2.5)
            elif sell_conditions:
                df.iloc[i, df.columns.get_loc('signal')] = -1
                df.iloc[i, df.columns.get_loc('sl')] = current['Close'] + (current['atr'] * 1.5)
                df.iloc[i, df.columns.get_loc('tp')] = current['Close'] - (current['atr'] * 2.5)

        return df


class CrossPairsBacktester:
    """Backtester pour les paires cross."""

    def __init__(self, initial_balance: float = 10000, risk_per_trade: float = 0.01):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.strategy = BacktestStrategy()
        self.results = {}
        self.all_trades = []

    def fetch_data(self, pair: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Recupere les donnees via yfinance."""
        symbol = YFINANCE_SYMBOLS.get(pair, f"{pair}=X")
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=period, interval="1h")
            if df.empty:
                return None
            return df
        except Exception as e:
            logger.error(f"Erreur fetch {pair}: {e}")
            return None

    def run_backtest_pair(self, pair: str, df: pd.DataFrame) -> Dict:
        """Execute le backtest pour une paire."""
        df = self.strategy.generate_signals(df, pair)

        trades = []
        balance = self.initial_balance
        position = None
        equity_curve = [balance]

        pip_value = 0.01 if 'JPY' in pair else 0.0001

        for i in range(len(df)):
            row = df.iloc[i]

            # Gerer position ouverte
            if position is not None:
                # Check SL
                if position['direction'] == 'BUY':
                    if row['Low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) / pip_value
                        balance += pnl * position['size']
                        trades.append({
                            'pair': pair,
                            'direction': 'BUY',
                            'entry': position['entry'],
                            'exit': position['sl'],
                            'pnl': pnl * position['size'],
                            'pips': pnl,
                            'result': 'LOSS',
                            'entry_time': position['time'],
                            'exit_time': row.name
                        })
                        position = None
                    # Check TP
                    elif row['High'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) / pip_value
                        balance += pnl * position['size']
                        trades.append({
                            'pair': pair,
                            'direction': 'BUY',
                            'entry': position['entry'],
                            'exit': position['tp'],
                            'pnl': pnl * position['size'],
                            'pips': pnl,
                            'result': 'WIN',
                            'entry_time': position['time'],
                            'exit_time': row.name
                        })
                        position = None
                else:  # SELL
                    if row['High'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) / pip_value
                        balance += pnl * position['size']
                        trades.append({
                            'pair': pair,
                            'direction': 'SELL',
                            'entry': position['entry'],
                            'exit': position['sl'],
                            'pnl': pnl * position['size'],
                            'pips': pnl,
                            'result': 'LOSS',
                            'entry_time': position['time'],
                            'exit_time': row.name
                        })
                        position = None
                    elif row['Low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) / pip_value
                        balance += pnl * position['size']
                        trades.append({
                            'pair': pair,
                            'direction': 'SELL',
                            'entry': position['entry'],
                            'exit': position['tp'],
                            'pnl': pnl * position['size'],
                            'pips': pnl,
                            'result': 'WIN',
                            'entry_time': position['time'],
                            'exit_time': row.name
                        })
                        position = None

            # Ouvrir nouvelle position
            if position is None and row['signal'] != 0:
                risk_amount = balance * self.risk_per_trade
                sl_distance = abs(row['Close'] - row['sl'])
                if sl_distance > 0:
                    position_size = risk_amount / (sl_distance / pip_value)
                    position = {
                        'direction': 'BUY' if row['signal'] == 1 else 'SELL',
                        'entry': row['Close'],
                        'sl': row['sl'],
                        'tp': row['tp'],
                        'size': position_size,
                        'time': row.name
                    }

            equity_curve.append(balance)

        # Calculer les metriques
        if not trades:
            return {
                'pair': pair,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'largest_win': 0,
                'largest_loss': 0,
                'trades': [],
                'equity_curve': equity_curve
            }

        winning = [t for t in trades if t['result'] == 'WIN']
        losing = [t for t in trades if t['result'] == 'LOSS']

        total_profit = sum(t['pnl'] for t in winning) if winning else 0
        total_loss = abs(sum(t['pnl'] for t in losing)) if losing else 0

        # Max Drawdown
        peak = self.initial_balance
        max_dd = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100
            if dd > max_dd:
                max_dd = dd

        return {
            'pair': pair,
            'total_trades': len(trades),
            'winning_trades': len(winning),
            'losing_trades': len(losing),
            'win_rate': len(winning) / len(trades) * 100 if trades else 0,
            'total_pnl': balance - self.initial_balance,
            'total_pnl_pct': (balance - self.initial_balance) / self.initial_balance * 100,
            'final_balance': balance,
            'profit_factor': total_profit / total_loss if total_loss > 0 else 0,
            'max_drawdown': max_dd,
            'avg_win': sum(t['pnl'] for t in winning) / len(winning) if winning else 0,
            'avg_loss': sum(t['pnl'] for t in losing) / len(losing) if losing else 0,
            'largest_win': max(t['pnl'] for t in winning) if winning else 0,
            'largest_loss': min(t['pnl'] for t in losing) if losing else 0,
            'avg_pips_win': sum(t['pips'] for t in winning) / len(winning) if winning else 0,
            'avg_pips_loss': sum(t['pips'] for t in losing) / len(losing) if losing else 0,
            'trades': trades,
            'equity_curve': equity_curve
        }

    def run_full_backtest(self, pairs: List[str] = None, period: str = "2y"):
        """Execute le backtest complet."""
        if pairs is None:
            pairs = CROSS_PAIRS

        print("\n" + "=" * 70)
        print("       BACKTEST CROSS PAIRS - 2 ANS D'HISTORIQUE")
        print("=" * 70)
        print(f"\nPaires: {len(pairs)}")
        print(f"Periode: {period}")
        print(f"Balance initiale: ${self.initial_balance:,.2f}")
        print(f"Risque par trade: {self.risk_per_trade * 100:.1f}%")
        print("\n" + "-" * 70)
        print("Chargement des donnees...")
        print("-" * 70)

        for pair in pairs:
            print(f"  {pair}...", end=" ", flush=True)
            df = self.fetch_data(pair, period)

            if df is None or len(df) < 100:
                print("SKIP (donnees insuffisantes)")
                continue

            result = self.run_backtest_pair(pair, df)
            self.results[pair] = result
            self.all_trades.extend(result['trades'])

            print(f"OK ({result['total_trades']} trades, WR: {result['win_rate']:.1f}%)")
            time_module.sleep(0.5)  # Rate limiting

        return self.results

    def generate_report(self) -> str:
        """Genere le rapport complet."""
        report = []

        report.append("\n" + "=" * 80)
        report.append("                    RAPPORT DE BACKTEST COMPLET")
        report.append("                    15 PAIRES CROSS - 2 ANS")
        report.append("=" * 80)

        # Stats globales
        total_trades = sum(r['total_trades'] for r in self.results.values())
        total_wins = sum(r['winning_trades'] for r in self.results.values())
        total_losses = sum(r['losing_trades'] for r in self.results.values())
        total_pnl = sum(r['total_pnl'] for r in self.results.values())

        report.append("\n" + "-" * 80)
        report.append("STATISTIQUES GLOBALES")
        report.append("-" * 80)
        report.append(f"Paires analysees:      {len(self.results)}")
        report.append(f"Total trades:          {total_trades}")
        report.append(f"Trades gagnants:       {total_wins} ({total_wins/total_trades*100:.1f}%)" if total_trades > 0 else "Trades gagnants:       0")
        report.append(f"Trades perdants:       {total_losses}")
        report.append(f"PnL Total:             ${total_pnl:,.2f} ({total_pnl/self.initial_balance*100:+.2f}%)")

        # Calcul profit factor global
        all_wins = sum(r['avg_win'] * r['winning_trades'] for r in self.results.values() if r['winning_trades'] > 0)
        all_losses = abs(sum(r['avg_loss'] * r['losing_trades'] for r in self.results.values() if r['losing_trades'] > 0))
        global_pf = all_wins / all_losses if all_losses > 0 else 0
        report.append(f"Profit Factor Global:  {global_pf:.2f}")

        # Max Drawdown global
        max_dd = max(r['max_drawdown'] for r in self.results.values()) if self.results else 0
        report.append(f"Max Drawdown:          {max_dd:.2f}%")

        # Tableau par paire
        report.append("\n" + "-" * 80)
        report.append("RESULTATS PAR PAIRE")
        report.append("-" * 80)
        report.append(f"{'Paire':<10} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'PnL':>12} {'PF':>6} {'MaxDD':>7} {'Grade':>7}")
        report.append("-" * 80)

        # Trier par win rate
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['win_rate'], reverse=True)

        for pair, r in sorted_results:
            # Grade
            if r['win_rate'] >= 50 and r['profit_factor'] >= 1.5:
                grade = "A"
            elif r['win_rate'] >= 45 and r['profit_factor'] >= 1.2:
                grade = "B"
            elif r['win_rate'] >= 40 and r['profit_factor'] >= 1.0:
                grade = "C"
            elif r['win_rate'] >= 35:
                grade = "D"
            else:
                grade = "F"

            report.append(
                f"{pair:<10} {r['total_trades']:>7} {r['winning_trades']:>6} "
                f"{r['win_rate']:>6.1f}% ${r['total_pnl']:>10,.2f} "
                f"{r['profit_factor']:>5.2f} {r['max_drawdown']:>6.2f}% {grade:>7}"
            )

        # Top 5 paires
        report.append("\n" + "-" * 80)
        report.append("TOP 5 PAIRES (par Win Rate)")
        report.append("-" * 80)

        for i, (pair, r) in enumerate(sorted_results[:5], 1):
            report.append(f"  {i}. {pair}")
            report.append(f"     Win Rate: {r['win_rate']:.1f}%")
            report.append(f"     Profit Factor: {r['profit_factor']:.2f}")
            report.append(f"     PnL: ${r['total_pnl']:,.2f} ({r['total_pnl_pct']:+.2f}%)")
            report.append(f"     Trades: {r['total_trades']} (W:{r['winning_trades']}/L:{r['losing_trades']})")
            report.append(f"     Avg Win: ${r['avg_win']:.2f} | Avg Loss: ${r['avg_loss']:.2f}")
            report.append(f"     Max Drawdown: {r['max_drawdown']:.2f}%")
            report.append("")

        # Paires a eviter
        report.append("-" * 80)
        report.append("PAIRES A EVITER (WR < 40% ou PF < 1.0)")
        report.append("-" * 80)

        avoid_pairs = [(p, r) for p, r in sorted_results if r['win_rate'] < 40 or r['profit_factor'] < 1.0]
        if avoid_pairs:
            for pair, r in avoid_pairs:
                report.append(f"  - {pair}: WR={r['win_rate']:.1f}%, PF={r['profit_factor']:.2f}")
        else:
            report.append("  Aucune paire a eviter!")

        # Analyse par devise de base
        report.append("\n" + "-" * 80)
        report.append("ANALYSE PAR DEVISE DE BASE")
        report.append("-" * 80)

        base_currencies = {}
        for pair, r in self.results.items():
            base = pair[:3]
            if base not in base_currencies:
                base_currencies[base] = {'trades': 0, 'wins': 0, 'pnl': 0}
            base_currencies[base]['trades'] += r['total_trades']
            base_currencies[base]['wins'] += r['winning_trades']
            base_currencies[base]['pnl'] += r['total_pnl']

        for base, stats in sorted(base_currencies.items(), key=lambda x: x[1]['wins']/x[1]['trades'] if x[1]['trades'] > 0 else 0, reverse=True):
            wr = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
            report.append(f"  {base}: {stats['trades']} trades, WR={wr:.1f}%, PnL=${stats['pnl']:,.2f}")

        # Analyse par devise de cotation
        report.append("\n" + "-" * 80)
        report.append("ANALYSE PAR DEVISE DE COTATION")
        report.append("-" * 80)

        quote_currencies = {}
        for pair, r in self.results.items():
            quote = pair[3:]
            if quote not in quote_currencies:
                quote_currencies[quote] = {'trades': 0, 'wins': 0, 'pnl': 0}
            quote_currencies[quote]['trades'] += r['total_trades']
            quote_currencies[quote]['wins'] += r['winning_trades']
            quote_currencies[quote]['pnl'] += r['total_pnl']

        for quote, stats in sorted(quote_currencies.items(), key=lambda x: x[1]['wins']/x[1]['trades'] if x[1]['trades'] > 0 else 0, reverse=True):
            wr = stats['wins'] / stats['trades'] * 100 if stats['trades'] > 0 else 0
            report.append(f"  {quote}: {stats['trades']} trades, WR={wr:.1f}%, PnL=${stats['pnl']:,.2f}")

        # Recommandations finales
        report.append("\n" + "=" * 80)
        report.append("RECOMMANDATIONS")
        report.append("=" * 80)

        # Paires recommandees (WR >= 45% et PF >= 1.2)
        recommended = [(p, r) for p, r in sorted_results if r['win_rate'] >= 45 and r['profit_factor'] >= 1.2 and r['total_trades'] >= 10]

        if recommended:
            report.append("\nPAIRES RECOMMANDEES POUR LE TRADING LIVE:")
            for pair, r in recommended:
                report.append(f"  [OK] {pair} - WR:{r['win_rate']:.1f}%, PF:{r['profit_factor']:.2f}, MaxDD:{r['max_drawdown']:.1f}%")
        else:
            report.append("\n[!] Aucune paire ne repond aux criteres stricts (WR>=45%, PF>=1.2)")
            report.append("    Considerez les paires avec WR>=40% pour du paper trading")

        # Paires potentielles (WR >= 40%)
        potential = [(p, r) for p, r in sorted_results if 40 <= r['win_rate'] < 45 and r['profit_factor'] >= 1.0 and r['total_trades'] >= 10]
        if potential:
            report.append("\nPAIRES POTENTIELLES (a surveiller):")
            for pair, r in potential:
                report.append(f"  [~] {pair} - WR:{r['win_rate']:.1f}%, PF:{r['profit_factor']:.2f}")

        report.append("\n" + "=" * 80)
        report.append("FIN DU RAPPORT")
        report.append("=" * 80 + "\n")

        return "\n".join(report)


def main():
    """Point d'entree principal."""
    print("\n" + "=" * 70)
    print("  FOREX SCALPER AGENT V2 - CROSS PAIRS BACKTEST")
    print("  15 paires | 2 ans | Strategies EMA+RSI+MACD")
    print("=" * 70)

    # Creer le backtester
    backtester = CrossPairsBacktester(
        initial_balance=10000,
        risk_per_trade=0.01  # 1% risque par trade
    )

    # Executer le backtest
    results = backtester.run_full_backtest(CROSS_PAIRS, period="2y")

    # Generer le rapport
    report = backtester.generate_report()
    print(report)

    # Sauvegarder le rapport
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"backtest_cross_pairs_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n[EXPORT] Rapport sauvegarde: {report_file}")

    # Exporter les trades en CSV
    if backtester.all_trades:
        trades_df = pd.DataFrame(backtester.all_trades)
        trades_file = f"backtest_trades_cross_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"[EXPORT] Trades exportes: {trades_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
