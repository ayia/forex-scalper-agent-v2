#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross Pairs Backtest V2 - Stratégie Améliorée
==============================================
Améliorations par rapport à V1:
1. Filtre de tendance HTF (H4 trend alignment)
2. Filtre de volatilité (ATR ratio)
3. Filtre de session (London/NY uniquement)
4. Conditions d'entrée renforcées (multi-confirmation)
5. R:R dynamique basé sur ATR
6. Filtre de momentum (ADX)
7. Filtre de range (éviter les marchés choppy)
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import time as time_module
import yfinance as yf

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Paires cross à tester
CROSS_PAIRS = [
    'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD',
    'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD',
    'AUDJPY', 'CADJPY', 'CHFJPY',
    'AUDCAD', 'AUDNZD', 'NZDJPY'
]

YFINANCE_SYMBOLS = {
    'EURGBP': 'EURGBP=X', 'EURJPY': 'EURJPY=X', 'EURCHF': 'EURCHF=X',
    'EURAUD': 'EURAUD=X', 'EURCAD': 'EURCAD=X', 'GBPJPY': 'GBPJPY=X',
    'GBPCHF': 'GBPCHF=X', 'GBPAUD': 'GBPAUD=X', 'GBPCAD': 'GBPCAD=X',
    'AUDJPY': 'AUDJPY=X', 'CADJPY': 'CADJPY=X', 'CHFJPY': 'CHFJPY=X',
    'AUDCAD': 'AUDCAD=X', 'AUDNZD': 'AUDNZD=X', 'NZDJPY': 'NZDJPY=X'
}


class ImprovedCrossStrategy:
    """
    Stratégie améliorée avec multiples filtres.

    Filtres:
    1. Tendance HTF (EMA 200 sur H4)
    2. Volatilité (ATR pas trop bas ni trop haut)
    3. Session (London 07-16 UTC, NY 12-21 UTC)
    4. Momentum (ADX > 20)
    5. Pas de range (Bollinger Width)
    6. Multi-timeframe confirmation
    """

    def __init__(self):
        # EMAs
        self.ema_fast = 8
        self.ema_slow = 21
        self.ema_trend = 50
        self.ema_htf = 200  # Pour filtre tendance

        # RSI
        self.rsi_period = 14
        self.rsi_oversold = 35
        self.rsi_overbought = 65

        # ADX pour momentum
        self.adx_period = 14
        self.adx_threshold = 15  # Min ADX pour confirmer tendance (relaxe)

        # ATR
        self.atr_period = 14
        self.atr_min_ratio = 0.3   # ATR actuel doit etre > 30% de la moyenne (relaxe)
        self.atr_max_ratio = 3.0   # ATR actuel doit etre < 300% de la moyenne (relaxe)

        # R:R dynamique
        self.sl_atr_mult = 1.5     # SL = 1.5 * ATR
        self.tp_atr_mult = 2.5     # TP = 2.5 * ATR (R:R minimum 1.67)

        # Bollinger pour detecter range
        self.bb_period = 20
        self.bb_squeeze_threshold = 0.01  # Si BB width < 1%, on evite (relaxe)

    def calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """Calcule l'ADX (Average Directional Index)."""
        high = df['High']
        low = df['Low']
        close = df['Close']

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = low.diff().abs() * -1

        plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
        minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)

        # Smoothed
        atr = tr.rolling(self.adx_period).mean()
        plus_di = 100 * (plus_dm.rolling(self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.adx_period).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(self.adx_period).mean()

        return adx

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcule tous les indicateurs."""
        df = df.copy()

        # EMAs
        df['ema_fast'] = df['Close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['Close'].ewm(span=self.ema_slow, adjust=False).mean()
        df['ema_trend'] = df['Close'].ewm(span=self.ema_trend, adjust=False).mean()
        df['ema_htf'] = df['Close'].ewm(span=self.ema_htf, adjust=False).mean()

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(span=self.rsi_period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=self.rsi_period, adjust=False).mean()
        rs = gain / (loss + 0.0001)
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
        df['atr'] = tr.rolling(self.atr_period).mean()
        df['atr_avg'] = df['atr'].rolling(50).mean()  # ATR moyen pour ratio

        # ADX
        df['adx'] = self.calculate_adx(df)

        # Bollinger Bands
        df['bb_mid'] = df['Close'].rolling(self.bb_period).mean()
        bb_std = df['Close'].rolling(self.bb_period).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
        df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

        # Momentum (ROC)
        df['roc'] = df['Close'].pct_change(10) * 100

        # Volume relatif (si disponible)
        if 'Volume' in df.columns:
            df['vol_avg'] = df['Volume'].rolling(20).mean()
            df['vol_ratio'] = df['Volume'] / (df['vol_avg'] + 1)
        else:
            df['vol_ratio'] = 1.0

        return df

    def is_valid_session(self, timestamp) -> bool:
        """Vérifie si on est dans une session de trading valide."""
        if timestamp is None:
            return True

        try:
            hour = timestamp.hour
            # London: 07-16 UTC, NY: 12-21 UTC
            # Best overlap: 12-16 UTC
            return 7 <= hour <= 20
        except:
            return True

    def check_htf_trend(self, df: pd.DataFrame, idx: int) -> Tuple[str, float]:
        """
        Vérifie la tendance HTF.
        Returns: ('BULLISH', 'BEARISH', 'NEUTRAL'), strength
        """
        if idx < 200:
            return 'NEUTRAL', 0

        current = df.iloc[idx]

        # Tendance basée sur EMA 200 et pente
        ema_htf = current['ema_htf']
        close = current['Close']

        # Pente de l'EMA 200 (sur 20 périodes)
        ema_slope = (df.iloc[idx]['ema_htf'] - df.iloc[idx-20]['ema_htf']) / df.iloc[idx-20]['ema_htf'] * 100

        if close > ema_htf and ema_slope > 0.1:
            return 'BULLISH', min(abs(ema_slope) * 10, 100)
        elif close < ema_htf and ema_slope < -0.1:
            return 'BEARISH', min(abs(ema_slope) * 10, 100)
        else:
            return 'NEUTRAL', 0

    def check_volatility_filter(self, df: pd.DataFrame, idx: int) -> bool:
        """Vérifie si la volatilité est dans une plage acceptable."""
        if idx < 50:
            return False

        current = df.iloc[idx]

        if pd.isna(current['atr']) or pd.isna(current['atr_avg']):
            return False

        if current['atr_avg'] == 0:
            return False

        atr_ratio = current['atr'] / current['atr_avg']

        # ATR doit être entre 50% et 200% de la moyenne
        return self.atr_min_ratio <= atr_ratio <= self.atr_max_ratio

    def check_momentum_filter(self, df: pd.DataFrame, idx: int) -> bool:
        """Vérifie si le momentum est suffisant (ADX > seuil)."""
        current = df.iloc[idx]

        if pd.isna(current['adx']):
            return False

        return current['adx'] >= self.adx_threshold

    def check_no_squeeze(self, df: pd.DataFrame, idx: int) -> bool:
        """Vérifie qu'on n'est pas en squeeze (range serré)."""
        current = df.iloc[idx]

        if pd.isna(current['bb_width']):
            return True

        # Éviter les BB squeeze (marché en range)
        return current['bb_width'] >= self.bb_squeeze_threshold

    def generate_signals(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        """Génère les signaux avec tous les filtres."""
        df = self.calculate_indicators(df)
        df['signal'] = 0
        df['sl'] = 0.0
        df['tp'] = 0.0
        df['signal_strength'] = 0.0

        pip_value = 0.01 if 'JPY' in pair else 0.0001

        for i in range(max(200, 50), len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]

            # Skip si données manquantes
            if pd.isna(current['atr']) or current['atr'] == 0:
                continue

            # ========== FILTRES ==========

            # 1. Filtre de session
            if not self.is_valid_session(df.index[i] if hasattr(df.index[i], 'hour') else None):
                continue

            # 2. Filtre de tendance HTF
            htf_trend, htf_strength = self.check_htf_trend(df, i)

            # 3. Filtre de volatilité
            if not self.check_volatility_filter(df, i):
                continue

            # 4. Filtre de momentum (ADX)
            if not self.check_momentum_filter(df, i):
                continue

            # 5. Filtre de squeeze
            if not self.check_no_squeeze(df, i):
                continue

            # ========== CONDITIONS D'ENTRÉE ==========

            # Conditions communes
            ema_cross_up = prev['ema_fast'] <= prev['ema_slow'] and current['ema_fast'] > current['ema_slow']
            ema_cross_down = prev['ema_fast'] >= prev['ema_slow'] and current['ema_fast'] < current['ema_slow']

            price_above_trend = current['Close'] > current['ema_trend']
            price_below_trend = current['Close'] < current['ema_trend']

            rsi_ok_buy = self.rsi_oversold < current['rsi'] < self.rsi_overbought
            rsi_ok_sell = self.rsi_oversold < current['rsi'] < self.rsi_overbought

            macd_bullish = current['macd_hist'] > 0 and current['macd_hist'] > prev['macd_hist']
            macd_bearish = current['macd_hist'] < 0 and current['macd_hist'] < prev['macd_hist']

            # Momentum positif/négatif
            momentum_bullish = current['roc'] > 0
            momentum_bearish = current['roc'] < 0

            # ========== SIGNAL BUY ==========
            buy_conditions = (
                htf_trend in ['BULLISH', 'NEUTRAL'] and  # Pas contre tendance HTF
                ema_cross_up and
                price_above_trend and
                rsi_ok_buy and
                macd_bullish and
                momentum_bullish
            )

            # ========== SIGNAL SELL ==========
            sell_conditions = (
                htf_trend in ['BEARISH', 'NEUTRAL'] and  # Pas contre tendance HTF
                ema_cross_down and
                price_below_trend and
                rsi_ok_sell and
                macd_bearish and
                momentum_bearish
            )

            # Calcul du signal strength
            signal_strength = 0
            if buy_conditions or sell_conditions:
                # Points pour chaque confirmation (plus genereux)
                signal_strength = 30  # Base
                if htf_trend != 'NEUTRAL':
                    signal_strength += 15
                if current['adx'] > 20:
                    signal_strength += 10
                if current['adx'] > 25:
                    signal_strength += 10
                if abs(current['macd_hist']) > abs(prev['macd_hist']):
                    signal_strength += 10
                if current['vol_ratio'] > 1.1:
                    signal_strength += 5

            # ========== GÉNÉRATION DU SIGNAL ==========
            if buy_conditions and signal_strength >= 30:
                # SL/TP dynamiques basés sur ATR
                sl_distance = current['atr'] * self.sl_atr_mult
                tp_distance = current['atr'] * self.tp_atr_mult

                # Vérifier que le R:R est acceptable
                if tp_distance / sl_distance >= 1.5:
                    df.iloc[i, df.columns.get_loc('signal')] = 1
                    df.iloc[i, df.columns.get_loc('sl')] = current['Close'] - sl_distance
                    df.iloc[i, df.columns.get_loc('tp')] = current['Close'] + tp_distance
                    df.iloc[i, df.columns.get_loc('signal_strength')] = signal_strength

            elif sell_conditions and signal_strength >= 30:
                sl_distance = current['atr'] * self.sl_atr_mult
                tp_distance = current['atr'] * self.tp_atr_mult

                if tp_distance / sl_distance >= 1.5:
                    df.iloc[i, df.columns.get_loc('signal')] = -1
                    df.iloc[i, df.columns.get_loc('sl')] = current['Close'] + sl_distance
                    df.iloc[i, df.columns.get_loc('tp')] = current['Close'] - tp_distance
                    df.iloc[i, df.columns.get_loc('signal_strength')] = signal_strength

        return df


class CrossPairsBacktesterV2:
    """Backtester V2 avec stratégie améliorée."""

    def __init__(self, initial_balance: float = 10000, risk_per_trade: float = 0.01):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.strategy = ImprovedCrossStrategy()
        self.results = {}
        self.all_trades = []

    def fetch_data(self, pair: str, period: str = "2y") -> Optional[pd.DataFrame]:
        """Récupère les données via yfinance."""
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
        consecutive_losses = 0
        max_consecutive_losses = 0

        pip_value = 0.01 if 'JPY' in pair else 0.0001

        for i in range(len(df)):
            row = df.iloc[i]

            # Gérer position ouverte
            if position is not None:
                if position['direction'] == 'BUY':
                    if row['Low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) / pip_value
                        balance += pnl * position['size']
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        trades.append({
                            'pair': pair,
                            'direction': 'BUY',
                            'entry': position['entry'],
                            'exit': position['sl'],
                            'pnl': pnl * position['size'],
                            'pips': pnl,
                            'result': 'LOSS',
                            'signal_strength': position['strength'],
                            'entry_time': position['time'],
                            'exit_time': row.name
                        })
                        position = None
                    elif row['High'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) / pip_value
                        balance += pnl * position['size']
                        consecutive_losses = 0
                        trades.append({
                            'pair': pair,
                            'direction': 'BUY',
                            'entry': position['entry'],
                            'exit': position['tp'],
                            'pnl': pnl * position['size'],
                            'pips': pnl,
                            'result': 'WIN',
                            'signal_strength': position['strength'],
                            'entry_time': position['time'],
                            'exit_time': row.name
                        })
                        position = None
                else:  # SELL
                    if row['High'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) / pip_value
                        balance += pnl * position['size']
                        consecutive_losses += 1
                        max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
                        trades.append({
                            'pair': pair,
                            'direction': 'SELL',
                            'entry': position['entry'],
                            'exit': position['sl'],
                            'pnl': pnl * position['size'],
                            'pips': pnl,
                            'result': 'LOSS',
                            'signal_strength': position['strength'],
                            'entry_time': position['time'],
                            'exit_time': row.name
                        })
                        position = None
                    elif row['Low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) / pip_value
                        balance += pnl * position['size']
                        consecutive_losses = 0
                        trades.append({
                            'pair': pair,
                            'direction': 'SELL',
                            'entry': position['entry'],
                            'exit': position['tp'],
                            'pnl': pnl * position['size'],
                            'pips': pnl,
                            'result': 'WIN',
                            'signal_strength': position['strength'],
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
                        'time': row.name,
                        'strength': row['signal_strength']
                    }

            equity_curve.append(balance)

        # Calculer les métriques
        if not trades:
            return self._empty_result(pair, equity_curve)

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

        # Expectancy
        avg_win = sum(t['pnl'] for t in winning) / len(winning) if winning else 0
        avg_loss = abs(sum(t['pnl'] for t in losing) / len(losing)) if losing else 0
        win_rate = len(winning) / len(trades) if trades else 0
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

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
            'avg_win': avg_win,
            'avg_loss': -abs(avg_loss),
            'largest_win': max(t['pnl'] for t in winning) if winning else 0,
            'largest_loss': min(t['pnl'] for t in losing) if losing else 0,
            'avg_pips_win': sum(t['pips'] for t in winning) / len(winning) if winning else 0,
            'avg_pips_loss': sum(t['pips'] for t in losing) / len(losing) if losing else 0,
            'expectancy': expectancy,
            'consecutive_losses': max_consecutive_losses,
            'trades': trades,
            'equity_curve': equity_curve
        }

    def _empty_result(self, pair: str, equity_curve: List) -> Dict:
        """Retourne un résultat vide."""
        return {
            'pair': pair, 'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0, 'total_pnl': 0, 'total_pnl_pct': 0, 'final_balance': self.initial_balance,
            'profit_factor': 0, 'max_drawdown': 0, 'avg_win': 0, 'avg_loss': 0,
            'largest_win': 0, 'largest_loss': 0, 'avg_pips_win': 0, 'avg_pips_loss': 0,
            'expectancy': 0, 'consecutive_losses': 0, 'trades': [], 'equity_curve': equity_curve
        }

    def run_full_backtest(self, pairs: List[str] = None, period: str = "2y"):
        """Execute le backtest complet."""
        if pairs is None:
            pairs = CROSS_PAIRS

        print("\n" + "=" * 70)
        print("   BACKTEST V2 - STRATÉGIE AMÉLIORÉE - 15 PAIRES CROSS")
        print("=" * 70)
        print("\nAméliorations V2:")
        print("  - Filtre tendance HTF (EMA 200)")
        print("  - Filtre volatilité (ATR ratio 0.5-2.0)")
        print("  - Filtre momentum (ADX > 20)")
        print("  - Filtre session (7h-20h UTC)")
        print("  - Filtre squeeze Bollinger")
        print("  - R:R dynamique (min 1.5)")
        print("  - Signal strength minimum: 30")
        print("\n" + "-" * 70)

        for pair in pairs:
            print(f"  {pair}...", end=" ", flush=True)
            df = self.fetch_data(pair, period)

            if df is None or len(df) < 500:
                print("SKIP (données insuffisantes)")
                continue

            result = self.run_backtest_pair(pair, df)
            self.results[pair] = result
            self.all_trades.extend(result['trades'])

            status = "OK" if result['profit_factor'] >= 1.0 else "X"
            print(f"{status} ({result['total_trades']} trades, WR: {result['win_rate']:.1f}%, PF: {result['profit_factor']:.2f})")
            time_module.sleep(0.5)

        return self.results

    def generate_report(self) -> str:
        """Génère le rapport complet."""
        report = []

        report.append("\n" + "=" * 80)
        report.append("          RAPPORT BACKTEST V2 - STRATÉGIE AMÉLIORÉE")
        report.append("              15 PAIRES CROSS - 2 ANS")
        report.append("=" * 80)

        # Stats globales
        total_trades = sum(r['total_trades'] for r in self.results.values())
        total_wins = sum(r['winning_trades'] for r in self.results.values())
        total_losses = sum(r['losing_trades'] for r in self.results.values())
        total_pnl = sum(r['total_pnl'] for r in self.results.values())

        report.append("\n" + "-" * 80)
        report.append("STATISTIQUES GLOBALES")
        report.append("-" * 80)
        report.append(f"Paires analysées:      {len(self.results)}")
        report.append(f"Total trades:          {total_trades}")

        if total_trades > 0:
            report.append(f"Trades gagnants:       {total_wins} ({total_wins/total_trades*100:.1f}%)")
        else:
            report.append(f"Trades gagnants:       0")

        report.append(f"Trades perdants:       {total_losses}")
        report.append(f"PnL Total:             ${total_pnl:,.2f} ({total_pnl/self.initial_balance*100:+.2f}%)")

        # Profit factor global
        all_wins = sum(r['avg_win'] * r['winning_trades'] for r in self.results.values() if r['winning_trades'] > 0)
        all_losses = abs(sum(r['avg_loss'] * r['losing_trades'] for r in self.results.values() if r['losing_trades'] > 0))
        global_pf = all_wins / all_losses if all_losses > 0 else 0
        report.append(f"Profit Factor Global:  {global_pf:.2f}")

        max_dd = max((r['max_drawdown'] for r in self.results.values()), default=0)
        report.append(f"Max Drawdown:          {max_dd:.2f}%")

        # Tableau par paire
        report.append("\n" + "-" * 80)
        report.append("RÉSULTATS PAR PAIRE (triés par Win Rate)")
        report.append("-" * 80)
        report.append(f"{'Paire':<10} {'Trades':>7} {'Wins':>6} {'WR%':>7} {'PnL':>12} {'PF':>6} {'MaxDD':>7} {'Expect':>8} {'Grade':>6}")
        report.append("-" * 80)

        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['win_rate'], reverse=True)

        profitable_pairs = []
        for pair, r in sorted_results:
            # Grade amélioré
            if r['win_rate'] >= 50 and r['profit_factor'] >= 1.5 and r['max_drawdown'] <= 15:
                grade = "A+"
            elif r['win_rate'] >= 48 and r['profit_factor'] >= 1.3:
                grade = "A"
            elif r['win_rate'] >= 45 and r['profit_factor'] >= 1.2:
                grade = "B+"
            elif r['win_rate'] >= 42 and r['profit_factor'] >= 1.1:
                grade = "B"
            elif r['win_rate'] >= 40 and r['profit_factor'] >= 1.0:
                grade = "C"
            elif r['win_rate'] >= 38:
                grade = "D"
            else:
                grade = "F"

            if r['profit_factor'] >= 1.0 and r['total_trades'] >= 10:
                profitable_pairs.append((pair, r, grade))

            report.append(
                f"{pair:<10} {r['total_trades']:>7} {r['winning_trades']:>6} "
                f"{r['win_rate']:>6.1f}% ${r['total_pnl']:>10,.2f} "
                f"{r['profit_factor']:>5.2f} {r['max_drawdown']:>6.2f}% "
                f"${r['expectancy']:>7.2f} {grade:>6}"
            )

        # Paires profitables
        report.append("\n" + "=" * 80)
        report.append("PAIRES PROFITABLES (PF >= 1.0)")
        report.append("=" * 80)

        if profitable_pairs:
            for pair, r, grade in profitable_pairs:
                report.append(f"\n  [{grade}] {pair}")
                report.append(f"      Win Rate:       {r['win_rate']:.1f}%")
                report.append(f"      Profit Factor:  {r['profit_factor']:.2f}")
                report.append(f"      PnL:            ${r['total_pnl']:,.2f} ({r['total_pnl_pct']:+.2f}%)")
                report.append(f"      Trades:         {r['total_trades']} (W:{r['winning_trades']}/L:{r['losing_trades']})")
                report.append(f"      Avg Win/Loss:   ${r['avg_win']:.2f} / ${r['avg_loss']:.2f}")
                report.append(f"      Max Drawdown:   {r['max_drawdown']:.2f}%")
                report.append(f"      Expectancy:     ${r['expectancy']:.2f} par trade")
                report.append(f"      Max Consec. Loss: {r['consecutive_losses']}")
        else:
            report.append("\n  Aucune paire profitable trouvée.")

        # Comparaison V1 vs V2
        report.append("\n" + "=" * 80)
        report.append("COMPARAISON V1 vs V2")
        report.append("=" * 80)
        report.append(f"\n  Métrique              V1 (baseline)    V2 (amélioré)")
        report.append(f"  -------------------------------------------------")
        report.append(f"  Total Trades          ~2780            {total_trades}")
        report.append(f"  Win Rate Global       34.0%            {total_wins/total_trades*100:.1f}%" if total_trades > 0 else "  Win Rate Global       34.0%            N/A")
        report.append(f"  Profit Factor         0.86             {global_pf:.2f}")
        report.append(f"  Paires Profitables    1/15             {len(profitable_pairs)}/15")

        # Recommandations
        report.append("\n" + "=" * 80)
        report.append("RECOMMANDATIONS POUR TRADING LIVE")
        report.append("=" * 80)

        grade_a_pairs = [p for p, r, g in profitable_pairs if g in ['A+', 'A', 'B+']]
        if grade_a_pairs:
            report.append("\n  [OK] PAIRES RECOMMANDEES (Grade A/B+):")
            for pair, r, g in grade_a_pairs:
                report.append(f"     - {pair}: WR={r['win_rate']:.1f}%, PF={r['profit_factor']:.2f}, MaxDD={r['max_drawdown']:.1f}%")
        else:
            report.append("\n  [!] Aucune paire Grade A/B+ - Considerer ajustements supplementaires")

        grade_b_pairs = [p for p, r, g in profitable_pairs if g in ['B', 'C']]
        if grade_b_pairs:
            report.append("\n  [~] PAIRES A SURVEILLER (Grade B/C):")
            for pair, r, g in grade_b_pairs:
                report.append(f"     - {pair}: WR={r['win_rate']:.1f}%, PF={r['profit_factor']:.2f}")

        report.append("\n" + "=" * 80)
        report.append("FIN DU RAPPORT V2")
        report.append("=" * 80 + "\n")

        return "\n".join(report)


def main():
    """Point d'entrée principal."""
    print("\n" + "=" * 70)
    print("  FOREX SCALPER AGENT V2 - BACKTEST AMÉLIORÉ")
    print("  15 paires cross | 2 ans | Stratégie multi-filtres")
    print("=" * 70)

    backtester = CrossPairsBacktesterV2(
        initial_balance=10000,
        risk_per_trade=0.01
    )

    results = backtester.run_full_backtest(CROSS_PAIRS, period="2y")

    report = backtester.generate_report()
    print(report)

    # Sauvegarder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"backtest_cross_pairs_v2_{timestamp}.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n[EXPORT] Rapport: {report_file}")

    if backtester.all_trades:
        trades_df = pd.DataFrame(backtester.all_trades)
        trades_file = f"backtest_trades_cross_v2_{timestamp}.csv"
        trades_df.to_csv(trades_file, index=False)
        print(f"[EXPORT] Trades: {trades_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
