#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross Pairs Backtest V4 - Strategie Optimisee JPY + Trailing Stop
==================================================================
Optimisations basees sur les resultats V3:
1. Focus sur paires JPY (meilleures performances)
2. Filtre session Tokyo (00-09 UTC) et London (07-16 UTC)
3. R:R optimise 1:2.2 pour JPY
4. Trailing Stop pour securiser les profits
5. Breakeven a +1R
6. Filtre de volatilite adapte au JPY
7. Score de confluence renforce
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import time as time_module
import yfinance as yf

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Toutes les paires cross
ALL_CROSS_PAIRS = [
    'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD',
    'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD',
    'AUDJPY', 'CADJPY', 'CHFJPY',
    'AUDCAD', 'AUDNZD', 'NZDJPY'
]

# Paires JPY (focus principal)
JPY_PAIRS = ['EURJPY', 'GBPJPY', 'AUDJPY', 'CADJPY', 'CHFJPY', 'NZDJPY']

YFINANCE_SYMBOLS = {p: f"{p}=X" for p in ALL_CROSS_PAIRS}


class JPYOptimizedStrategy:
    """
    Strategie optimisee pour les paires JPY.

    Caracteristiques:
    - Session filter: Tokyo (00-09) + London (07-16)
    - EMA adapte: 8/21/50
    - RSI: 25-75 (plus large pour JPY volatile)
    - ADX > 18 (momentum confirme)
    - R:R: 1:2.2
    - Trailing stop: active a +1.5R
    - Breakeven: a +1R
    """

    def __init__(self, pair_type='JPY'):
        self.pair_type = pair_type

        # EMAs
        self.ema_fast = 8
        self.ema_slow = 21
        self.ema_trend = 50

        # RSI - plus large pour JPY
        self.rsi_period = 14
        if pair_type == 'JPY':
            self.rsi_oversold = 25
            self.rsi_overbought = 75
        else:
            self.rsi_oversold = 30
            self.rsi_overbought = 70

        # ADX - relaxe pour plus de trades
        self.adx_period = 14
        self.adx_min = 15 if pair_type == 'JPY' else 12

        # ATR & R:R - plus conservateur pour meilleur WR
        self.atr_period = 14
        if pair_type == 'JPY':
            self.sl_mult = 1.5   # SL standard
            self.tp_mult = 2.25  # R:R = 1.5
        else:
            self.sl_mult = 1.5
            self.tp_mult = 2.25  # R:R = 1.5

        # Trailing Stop - active plus tot
        self.trailing_start = 1.0  # Active trailing a +1R
        self.trailing_step = 0.3   # Trail par pas de 0.3R
        self.breakeven_at = 0.7    # Breakeven a +0.7R

    def is_valid_session(self, timestamp, pair) -> Tuple[bool, str]:
        """
        Verifie si on est dans une session optimale.
        Elargi pour capturer plus de trades.
        """
        if timestamp is None:
            return True, "unknown"

        try:
            hour = timestamp.hour

            if 'JPY' in pair:
                # JPY: Sessions principales elargies
                # Tokyo: 00-09 UTC
                if 0 <= hour <= 9:
                    return True, "tokyo"
                # London: 07-17 UTC
                elif 7 <= hour <= 17:
                    return True, "london"
                # NY matin (overlap): 12-16 UTC
                elif 12 <= hour <= 16:
                    return True, "ny_overlap"
                else:
                    return False, "closed"
            else:
                # Autres paires: London + NY complet
                if 6 <= hour <= 21:
                    return True, "london_ny"
                else:
                    return False, "closed"
        except:
            return True, "unknown"

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # EMAs
        df['ema_fast'] = df['Close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['Close'].ewm(span=self.ema_slow, adjust=False).mean()
        df['ema_trend'] = df['Close'].ewm(span=self.ema_trend, adjust=False).mean()

        # RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).ewm(span=self.rsi_period, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(span=self.rsi_period, adjust=False).mean()
        df['rsi'] = 100 - (100 / (1 + gain / (loss + 0.0001)))

        # MACD
        ema12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # ATR
        tr = pd.concat([
            df['High'] - df['Low'],
            abs(df['High'] - df['Close'].shift()),
            abs(df['Low'] - df['Close'].shift())
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.atr_period).mean()

        # ADX
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr14 = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 0.0001)
        minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 0.0001)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        df['adx'] = dx.rolling(14).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di

        # Momentum
        df['roc'] = df['Close'].pct_change(5) * 100
        df['roc_10'] = df['Close'].pct_change(10) * 100

        # Tendance EMA (pente)
        df['ema_slope'] = (df['ema_trend'] - df['ema_trend'].shift(10)) / df['ema_trend'].shift(10) * 100

        # Volatilite relative
        df['atr_ratio'] = df['atr'] / df['atr'].rolling(50).mean()

        # Bollinger Width pour detecter squeeze
        bb_mid = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['bb_width'] = (bb_std * 4) / bb_mid

        return df

    def calculate_confluence_score(self, row, prev, trend_aligned, is_buy) -> int:
        """Calcule un score de confluence (0-100)."""
        score = 0

        # 1. Tendance alignee (+25)
        if trend_aligned:
            score += 25

        # 2. ADX fort (+20)
        if row['adx'] >= 25:
            score += 20
        elif row['adx'] >= 20:
            score += 15
        elif row['adx'] >= self.adx_min:
            score += 10

        # 3. MACD confirme (+20)
        if is_buy and row['macd_hist'] > 0 and row['macd_hist'] > prev['macd_hist']:
            score += 20
        elif not is_buy and row['macd_hist'] < 0 and row['macd_hist'] < prev['macd_hist']:
            score += 20

        # 4. RSI zone favorable (+15)
        if is_buy and 30 <= row['rsi'] <= 60:
            score += 15
        elif not is_buy and 40 <= row['rsi'] <= 70:
            score += 15

        # 5. Momentum aligne (+10)
        if is_buy and row['roc'] > 0 and row['roc_10'] > 0:
            score += 10
        elif not is_buy and row['roc'] < 0 and row['roc_10'] < 0:
            score += 10

        # 6. DI+ vs DI- (+10)
        if is_buy and row['plus_di'] > row['minus_di']:
            score += 10
        elif not is_buy and row['minus_di'] > row['plus_di']:
            score += 10

        return score

    def generate_signals(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        # Determiner le type de paire
        self.pair_type = 'JPY' if 'JPY' in pair else 'OTHER'
        self.__init__(self.pair_type)

        df = self.calculate_indicators(df)
        df['signal'] = 0
        df['sl'] = 0.0
        df['tp'] = 0.0
        df['confluence'] = 0

        for i in range(60, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(current['atr']) or current['atr'] == 0:
                continue
            if pd.isna(current['adx']):
                continue

            # Filtre de session
            valid_session, session = self.is_valid_session(
                df.index[i] if hasattr(df.index[i], 'hour') else None,
                pair
            )
            if not valid_session:
                continue

            # Filtre volatilite (eviter extremes) - relaxe
            if current['atr_ratio'] < 0.3 or current['atr_ratio'] > 3.0:
                continue

            # Filtre squeeze (eviter range serre) - relaxe
            if current['bb_width'] < 0.005:
                continue

            # Conditions de base
            ema_cross_up = prev['ema_fast'] <= prev['ema_slow'] and current['ema_fast'] > current['ema_slow']
            ema_cross_down = prev['ema_fast'] >= prev['ema_slow'] and current['ema_fast'] < current['ema_slow']

            # Tendance alignee (relaxee - juste au dessus/dessous EMA trend)
            trend_up = current['Close'] > current['ema_trend']
            trend_down = current['Close'] < current['ema_trend']

            # RSI pas en extremes
            rsi_ok_buy = self.rsi_oversold < current['rsi'] < self.rsi_overbought
            rsi_ok_sell = self.rsi_oversold < current['rsi'] < self.rsi_overbought

            adx_ok = current['adx'] >= self.adx_min

            # Conditions BUY - EMA cross + tendance OR confirmation forte
            macd_bullish = current['macd_hist'] > 0 and current['macd_hist'] > prev['macd_hist']
            buy_signal = ema_cross_up and trend_up and rsi_ok_buy and (adx_ok or macd_bullish)

            # Conditions SELL
            macd_bearish = current['macd_hist'] < 0 and current['macd_hist'] < prev['macd_hist']
            sell_signal = ema_cross_down and trend_down and rsi_ok_sell and (adx_ok or macd_bearish)

            if buy_signal:
                confluence = self.calculate_confluence_score(current, prev, trend_up, True)
                if confluence >= 35:  # Seuil de confluence reduit
                    df.iloc[i, df.columns.get_loc('signal')] = 1
                    df.iloc[i, df.columns.get_loc('sl')] = current['Close'] - (current['atr'] * self.sl_mult)
                    df.iloc[i, df.columns.get_loc('tp')] = current['Close'] + (current['atr'] * self.tp_mult)
                    df.iloc[i, df.columns.get_loc('confluence')] = confluence

            elif sell_signal:
                confluence = self.calculate_confluence_score(current, prev, trend_down, False)
                if confluence >= 35:
                    df.iloc[i, df.columns.get_loc('signal')] = -1
                    df.iloc[i, df.columns.get_loc('sl')] = current['Close'] + (current['atr'] * self.sl_mult)
                    df.iloc[i, df.columns.get_loc('tp')] = current['Close'] - (current['atr'] * self.tp_mult)
                    df.iloc[i, df.columns.get_loc('confluence')] = confluence

        return df


class BacktesterV4:
    """Backtester V4 avec trailing stop et breakeven."""

    def __init__(self, initial_balance=10000, risk_per_trade=0.01):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.strategy = JPYOptimizedStrategy()
        self.results = {}
        self.all_trades = []

    def fetch_data(self, pair, period="2y"):
        try:
            ticker = yf.Ticker(YFINANCE_SYMBOLS.get(pair, f"{pair}=X"))
            df = ticker.history(period=period, interval="1h")
            return df if not df.empty else None
        except:
            return None

    def run_pair(self, pair, df):
        df = self.strategy.generate_signals(df, pair)

        trades = []
        balance = self.initial_balance
        position = None
        equity = [balance]
        consec_loss = 0
        max_consec = 0

        pip = 0.01 if 'JPY' in pair else 0.0001

        for i in range(len(df)):
            row = df.iloc[i]

            # Gerer position ouverte avec trailing stop
            if position:
                current_price = row['Close']
                entry = position['entry']
                sl = position['sl']
                tp = position['tp']
                initial_risk = position['risk']

                if position['dir'] == 'BUY':
                    current_r = (current_price - entry) / initial_risk

                    # Breakeven a +1R
                    if current_r >= 1.0 and not position.get('breakeven_set'):
                        position['sl'] = entry + (pip * 2)  # +2 pips
                        position['breakeven_set'] = True

                    # Trailing stop a +1.5R
                    if current_r >= 1.5:
                        new_sl = current_price - (initial_risk * 0.8)
                        if new_sl > position['sl']:
                            position['sl'] = new_sl

                    sl = position['sl']

                    # Check SL
                    if row['Low'] <= sl:
                        pnl = (sl - entry) / pip * position['size']
                        balance += pnl
                        result = 'WIN' if pnl > 0 else 'LOSS'
                        if result == 'LOSS':
                            consec_loss += 1
                            max_consec = max(max_consec, consec_loss)
                        else:
                            consec_loss = 0
                        trades.append({
                            'pair': pair, 'dir': 'BUY', 'pnl': pnl,
                            'result': result, 'exit_type': 'SL',
                            'confluence': position['confluence']
                        })
                        position = None
                    # Check TP
                    elif row['High'] >= tp:
                        pnl = (tp - entry) / pip * position['size']
                        balance += pnl
                        consec_loss = 0
                        trades.append({
                            'pair': pair, 'dir': 'BUY', 'pnl': pnl,
                            'result': 'WIN', 'exit_type': 'TP',
                            'confluence': position['confluence']
                        })
                        position = None

                else:  # SELL
                    current_r = (entry - current_price) / initial_risk

                    # Breakeven
                    if current_r >= 1.0 and not position.get('breakeven_set'):
                        position['sl'] = entry - (pip * 2)
                        position['breakeven_set'] = True

                    # Trailing
                    if current_r >= 1.5:
                        new_sl = current_price + (initial_risk * 0.8)
                        if new_sl < position['sl']:
                            position['sl'] = new_sl

                    sl = position['sl']

                    if row['High'] >= sl:
                        pnl = (entry - sl) / pip * position['size']
                        balance += pnl
                        result = 'WIN' if pnl > 0 else 'LOSS'
                        if result == 'LOSS':
                            consec_loss += 1
                            max_consec = max(max_consec, consec_loss)
                        else:
                            consec_loss = 0
                        trades.append({
                            'pair': pair, 'dir': 'SELL', 'pnl': pnl,
                            'result': result, 'exit_type': 'SL',
                            'confluence': position['confluence']
                        })
                        position = None
                    elif row['Low'] <= tp:
                        pnl = (entry - tp) / pip * position['size']
                        balance += pnl
                        consec_loss = 0
                        trades.append({
                            'pair': pair, 'dir': 'SELL', 'pnl': pnl,
                            'result': 'WIN', 'exit_type': 'TP',
                            'confluence': position['confluence']
                        })
                        position = None

            # Nouvelle position
            if position is None and row['signal'] != 0:
                risk = balance * self.risk_per_trade
                sl_dist = abs(row['Close'] - row['sl'])
                if sl_dist > 0:
                    size = risk / (sl_dist / pip)
                    position = {
                        'dir': 'BUY' if row['signal'] == 1 else 'SELL',
                        'entry': row['Close'],
                        'sl': row['sl'],
                        'tp': row['tp'],
                        'size': size,
                        'risk': sl_dist,
                        'confluence': row['confluence'],
                        'breakeven_set': False
                    }

            equity.append(balance)

        if not trades:
            return self._empty(pair)

        wins = [t for t in trades if t['result'] == 'WIN']
        losses = [t for t in trades if t['result'] == 'LOSS']

        total_win = sum(t['pnl'] for t in wins) if wins else 0
        total_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0

        peak = self.initial_balance
        max_dd = 0
        for e in equity:
            peak = max(peak, e)
            dd = (peak - e) / peak * 100
            max_dd = max(max_dd, dd)

        # Analyse par type de sortie
        tp_exits = len([t for t in trades if t.get('exit_type') == 'TP'])
        sl_exits = len([t for t in trades if t.get('exit_type') == 'SL'])
        be_wins = len([t for t in trades if t['result'] == 'WIN' and t.get('exit_type') == 'SL'])

        return {
            'pair': pair,
            'trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'wr': len(wins) / len(trades) * 100 if trades else 0,
            'pnl': balance - self.initial_balance,
            'pnl_pct': (balance - self.initial_balance) / self.initial_balance * 100,
            'pf': total_win / total_loss if total_loss > 0 else 0,
            'max_dd': max_dd,
            'avg_win': total_win / len(wins) if wins else 0,
            'avg_loss': total_loss / len(losses) if losses else 0,
            'consec_loss': max_consec,
            'tp_exits': tp_exits,
            'be_wins': be_wins,
            'trade_list': trades
        }

    def _empty(self, pair):
        return {'pair': pair, 'trades': 0, 'wins': 0, 'losses': 0, 'wr': 0,
                'pnl': 0, 'pnl_pct': 0, 'pf': 0, 'max_dd': 0, 'avg_win': 0,
                'avg_loss': 0, 'consec_loss': 0, 'tp_exits': 0, 'be_wins': 0,
                'trade_list': []}

    def run_all(self, pairs=None, period="2y"):
        if pairs is None:
            pairs = ALL_CROSS_PAIRS

        print("\n" + "=" * 75)
        print("   BACKTEST V4 - STRATEGIE JPY OPTIMISEE + TRAILING STOP")
        print("=" * 75)
        print("\nOptimisations V4.2:")
        print("  - Sessions elargies (Tokyo/London/NY)")
        print("  - R:R conservateur: 1:1.5")
        print("  - Trailing stop active a +1R")
        print("  - Breakeven a +0.7R")
        print("  - Score confluence minimum: 35/100")
        print("  - ADX ou MACD confirmation")
        print("-" * 75)

        for pair in pairs:
            print(f"  {pair}...", end=" ", flush=True)
            df = self.fetch_data(pair, period)

            if df is None or len(df) < 200:
                print("SKIP")
                continue

            r = self.run_pair(pair, df)
            self.results[pair] = r
            self.all_trades.extend(r['trade_list'])

            is_jpy = 'JPY' in pair
            marker = "[JPY]" if is_jpy else "[---]"
            status = "OK" if r['pf'] >= 1.0 and r['trades'] >= 10 else "X "
            print(f"{marker} {status} {r['trades']:>4} trades, WR: {r['wr']:>5.1f}%, PF: {r['pf']:.2f}, PnL: ${r['pnl']:>7.0f}")
            time_module.sleep(0.3)

        return self.results

    def report(self):
        lines = []
        lines.append("\n" + "=" * 85)
        lines.append("              RAPPORT BACKTEST V4 - STRATEGIE JPY OPTIMISEE")
        lines.append("                      15 PAIRES CROSS - 2 ANS")
        lines.append("=" * 85)

        # Stats globales
        total_trades = sum(r['trades'] for r in self.results.values())
        total_wins = sum(r['wins'] for r in self.results.values())
        total_pnl = sum(r['pnl'] for r in self.results.values())

        # Stats JPY vs autres
        jpy_results = {k: v for k, v in self.results.items() if 'JPY' in k}
        other_results = {k: v for k, v in self.results.items() if 'JPY' not in k}

        jpy_trades = sum(r['trades'] for r in jpy_results.values())
        jpy_wins = sum(r['wins'] for r in jpy_results.values())
        jpy_pnl = sum(r['pnl'] for r in jpy_results.values())

        other_trades = sum(r['trades'] for r in other_results.values())
        other_wins = sum(r['wins'] for r in other_results.values())
        other_pnl = sum(r['pnl'] for r in other_results.values())

        lines.append("\n" + "-" * 85)
        lines.append("STATISTIQUES GLOBALES")
        lines.append("-" * 85)
        lines.append(f"{'Metrique':<30} {'Global':>15} {'Paires JPY':>15} {'Autres':>15}")
        lines.append("-" * 85)
        lines.append(f"{'Trades':<30} {total_trades:>15} {jpy_trades:>15} {other_trades:>15}")

        wr_global = total_wins/total_trades*100 if total_trades > 0 else 0
        wr_jpy = jpy_wins/jpy_trades*100 if jpy_trades > 0 else 0
        wr_other = other_wins/other_trades*100 if other_trades > 0 else 0
        lines.append(f"{'Win Rate':<30} {wr_global:>14.1f}% {wr_jpy:>14.1f}% {wr_other:>14.1f}%")

        lines.append(f"{'PnL':<30} ${total_pnl:>13,.0f} ${jpy_pnl:>13,.0f} ${other_pnl:>13,.0f}")

        # Profit Factor
        all_wins = sum(r['avg_win'] * r['wins'] for r in self.results.values() if r['wins'] > 0)
        all_losses = sum(r['avg_loss'] * r['losses'] for r in self.results.values() if r['losses'] > 0)
        global_pf = all_wins / all_losses if all_losses > 0 else 0

        jpy_wins_amt = sum(r['avg_win'] * r['wins'] for r in jpy_results.values() if r['wins'] > 0)
        jpy_losses_amt = sum(r['avg_loss'] * r['losses'] for r in jpy_results.values() if r['losses'] > 0)
        jpy_pf = jpy_wins_amt / jpy_losses_amt if jpy_losses_amt > 0 else 0

        other_wins_amt = sum(r['avg_win'] * r['wins'] for r in other_results.values() if r['wins'] > 0)
        other_losses_amt = sum(r['avg_loss'] * r['losses'] for r in other_results.values() if r['losses'] > 0)
        other_pf = other_wins_amt / other_losses_amt if other_losses_amt > 0 else 0

        lines.append(f"{'Profit Factor':<30} {global_pf:>15.2f} {jpy_pf:>15.2f} {other_pf:>15.2f}")

        # Tableau par paire
        lines.append("\n" + "-" * 85)
        lines.append("RESULTATS PAR PAIRE (tries par Profit Factor)")
        lines.append("-" * 85)
        lines.append(f"{'Paire':<10} {'Trades':>7} {'WR%':>7} {'PnL':>12} {'PF':>6} {'MaxDD':>7} {'TP%':>6} {'BE%':>6} {'Grade':>6}")
        lines.append("-" * 85)

        sorted_r = sorted(self.results.items(), key=lambda x: x[1]['pf'], reverse=True)
        profitable = []

        for pair, r in sorted_r:
            if r['trades'] < 10:
                grade = "N/A"
            elif r['wr'] >= 50 and r['pf'] >= 1.5:
                grade = "A+"
            elif r['wr'] >= 45 and r['pf'] >= 1.3:
                grade = "A"
            elif r['wr'] >= 42 and r['pf'] >= 1.2:
                grade = "B+"
            elif r['wr'] >= 40 and r['pf'] >= 1.1:
                grade = "B"
            elif r['pf'] >= 1.0:
                grade = "C"
            else:
                grade = "F"

            if r['pf'] >= 1.0 and r['trades'] >= 10:
                profitable.append((pair, r, grade))

            tp_pct = r['tp_exits'] / r['trades'] * 100 if r['trades'] > 0 else 0
            be_pct = r['be_wins'] / r['trades'] * 100 if r['trades'] > 0 else 0

            jpy_marker = "*" if 'JPY' in pair else " "
            lines.append(f"{jpy_marker}{pair:<9} {r['trades']:>7} {r['wr']:>6.1f}% ${r['pnl']:>10,.0f} {r['pf']:>5.2f} {r['max_dd']:>6.1f}% {tp_pct:>5.1f}% {be_pct:>5.1f}% {grade:>6}")

        lines.append("\n* = paire JPY")

        # Paires profitables
        lines.append("\n" + "=" * 85)
        lines.append("PAIRES PROFITABLES (PF >= 1.0, Trades >= 10)")
        lines.append("=" * 85)

        if profitable:
            for pair, r, grade in profitable:
                jpy = "[JPY]" if 'JPY' in pair else ""
                lines.append(f"\n  [{grade}] {pair} {jpy}")
                lines.append(f"      Trades: {r['trades']} | Win Rate: {r['wr']:.1f}%")
                lines.append(f"      PnL: ${r['pnl']:,.2f} ({r['pnl_pct']:+.1f}%)")
                lines.append(f"      Profit Factor: {r['pf']:.2f}")
                lines.append(f"      Max Drawdown: {r['max_dd']:.1f}%")
                lines.append(f"      Avg Win: ${r['avg_win']:.2f} | Avg Loss: ${r['avg_loss']:.2f}")
                lines.append(f"      TP Exits: {r['tp_exits']} | BE Wins: {r['be_wins']}")
        else:
            lines.append("\n  Aucune paire profitable avec criteres stricts.")

        # Comparaison V1 vs V3 vs V4
        lines.append("\n" + "=" * 85)
        lines.append("EVOLUTION DES PERFORMANCES")
        lines.append("=" * 85)
        lines.append(f"\n{'Metrique':<25} {'V1':>12} {'V3':>12} {'V4':>12} {'Evol V1->V4':>15}")
        lines.append("-" * 85)
        lines.append(f"{'Total Trades':<25} {'2,780':>12} {'12,459':>12} {total_trades:>12,}")
        lines.append(f"{'Win Rate':<25} {'34.0%':>12} {'34.0%':>12} {wr_global:>11.1f}%")
        lines.append(f"{'Profit Factor':<25} {'0.86':>12} {'0.94':>12} {global_pf:>11.2f} {'+' if global_pf > 0.86 else ''}{((global_pf-0.86)/0.86*100):+.1f}%")
        lines.append(f"{'Paires Profitables':<25} {'1/15':>12} {'3/15':>12} {f'{len(profitable)}/15':>12}")
        lines.append(f"{'PnL Total':<25} {'$-23,388':>12} {'$-46,588':>12} ${total_pnl:>10,.0f}")

        # Recommandations
        lines.append("\n" + "=" * 85)
        lines.append("RECOMMANDATIONS FINALES")
        lines.append("=" * 85)

        grade_a = [p for p in profitable if p[2] in ['A+', 'A']]
        grade_b = [p for p in profitable if p[2] in ['B+', 'B']]
        grade_c = [p for p in profitable if p[2] == 'C']

        if grade_a:
            lines.append("\n  [RECOMMANDE] PAIRES GRADE A (Trading Live):")
            for pair, r, g in grade_a:
                lines.append(f"     >> {pair}: WR={r['wr']:.1f}%, PF={r['pf']:.2f}, MaxDD={r['max_dd']:.1f}%")

        if grade_b:
            lines.append("\n  [ACCEPTABLE] PAIRES GRADE B (Paper Trading):")
            for pair, r, g in grade_b:
                lines.append(f"     -> {pair}: WR={r['wr']:.1f}%, PF={r['pf']:.2f}")

        if grade_c:
            lines.append("\n  [A SURVEILLER] PAIRES GRADE C:")
            for pair, r, g in grade_c:
                lines.append(f"     ~  {pair}: WR={r['wr']:.1f}%, PF={r['pf']:.2f}")

        # Conclusion
        lines.append("\n" + "-" * 85)
        if jpy_pf > other_pf:
            lines.append(f"  CONCLUSION: Les paires JPY surperforment (PF {jpy_pf:.2f} vs {other_pf:.2f})")
            lines.append(f"              Focus recommande sur: {', '.join([p for p, r, g in profitable if 'JPY' in p])}")

        lines.append("\n" + "=" * 85)

        return "\n".join(lines)


def main():
    bt = BacktesterV4(initial_balance=10000, risk_per_trade=0.01)
    bt.run_all(ALL_CROSS_PAIRS, "2y")

    report = bt.report()
    print(report)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f"backtest_v4_{ts}.txt", 'w') as f:
        f.write(report)
    print(f"\n[EXPORT] backtest_v4_{ts}.txt")

    if bt.all_trades:
        pd.DataFrame(bt.all_trades).to_csv(f"trades_v4_{ts}.csv", index=False)
        print(f"[EXPORT] trades_v4_{ts}.csv")


if __name__ == "__main__":
    main()
