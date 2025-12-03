#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross Pairs Backtest V3 - Strategie Optimisee
==============================================
Version equilibree avec filtres ajustes pour plus de trades
tout en maintenant la qualite des signaux.

Ameliorations:
1. EMA trend 50 (au lieu de 200)
2. ADX seuil reduit a 15
3. Conditions RSI elargies
4. Multi-confirmation simplifiee
5. R:R dynamique 1:1.8 minimum
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

CROSS_PAIRS = [
    'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD',
    'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD',
    'AUDJPY', 'CADJPY', 'CHFJPY',
    'AUDCAD', 'AUDNZD', 'NZDJPY'
]

YFINANCE_SYMBOLS = {p: f"{p}=X" for p in CROSS_PAIRS}


class OptimizedStrategy:
    """Strategie optimisee pour cross pairs."""

    def __init__(self):
        # EMAs
        self.ema_fast = 8
        self.ema_slow = 21
        self.ema_trend = 50

        # RSI
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70

        # ADX
        self.adx_period = 14
        self.adx_min = 15

        # ATR
        self.atr_period = 14
        self.sl_mult = 1.5
        self.tp_mult = 2.7  # R:R = 1.8

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

        # ADX simplifie
        plus_dm = df['High'].diff()
        minus_dm = -df['Low'].diff()
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        atr14 = tr.rolling(14).mean()
        plus_di = 100 * plus_dm.rolling(14).mean() / (atr14 + 0.0001)
        minus_di = 100 * minus_dm.rolling(14).mean() / (atr14 + 0.0001)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        df['adx'] = dx.rolling(14).mean()

        # Momentum
        df['roc'] = df['Close'].pct_change(5) * 100

        # Tendance EMA (pente)
        df['ema_slope'] = (df['ema_trend'] - df['ema_trend'].shift(10)) / df['ema_trend'].shift(10) * 100

        return df

    def generate_signals(self, df: pd.DataFrame, pair: str) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df['signal'] = 0
        df['sl'] = 0.0
        df['tp'] = 0.0

        pip_value = 0.01 if 'JPY' in pair else 0.0001

        for i in range(60, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(current['atr']) or current['atr'] == 0:
                continue
            if pd.isna(current['adx']):
                continue

            # Conditions de base
            ema_cross_up = prev['ema_fast'] <= prev['ema_slow'] and current['ema_fast'] > current['ema_slow']
            ema_cross_down = prev['ema_fast'] >= prev['ema_slow'] and current['ema_fast'] < current['ema_slow']

            trend_up = current['Close'] > current['ema_trend'] and current['ema_slope'] > 0
            trend_down = current['Close'] < current['ema_trend'] and current['ema_slope'] < 0

            rsi_ok = self.rsi_oversold < current['rsi'] < self.rsi_overbought
            adx_ok = current['adx'] >= self.adx_min

            macd_bull = current['macd_hist'] > 0
            macd_bear = current['macd_hist'] < 0

            mom_bull = current['roc'] > 0
            mom_bear = current['roc'] < 0

            # Compter les confirmations
            buy_score = sum([
                ema_cross_up * 2,
                trend_up * 2,
                rsi_ok,
                adx_ok,
                macd_bull,
                mom_bull
            ])

            sell_score = sum([
                ema_cross_down * 2,
                trend_down * 2,
                rsi_ok,
                adx_ok,
                macd_bear,
                mom_bear
            ])

            # Signal si score >= 5
            if buy_score >= 5:
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('sl')] = current['Close'] - (current['atr'] * self.sl_mult)
                df.iloc[i, df.columns.get_loc('tp')] = current['Close'] + (current['atr'] * self.tp_mult)
            elif sell_score >= 5:
                df.iloc[i, df.columns.get_loc('signal')] = -1
                df.iloc[i, df.columns.get_loc('sl')] = current['Close'] + (current['atr'] * self.sl_mult)
                df.iloc[i, df.columns.get_loc('tp')] = current['Close'] - (current['atr'] * self.tp_mult)

        return df


class BacktesterV3:
    def __init__(self, initial_balance=10000, risk_per_trade=0.01):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.strategy = OptimizedStrategy()
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

            if position:
                if position['dir'] == 'BUY':
                    if row['Low'] <= position['sl']:
                        pnl = (position['sl'] - position['entry']) / pip * position['size']
                        balance += pnl
                        consec_loss += 1
                        max_consec = max(max_consec, consec_loss)
                        trades.append({'pair': pair, 'dir': 'BUY', 'pnl': pnl, 'result': 'LOSS'})
                        position = None
                    elif row['High'] >= position['tp']:
                        pnl = (position['tp'] - position['entry']) / pip * position['size']
                        balance += pnl
                        consec_loss = 0
                        trades.append({'pair': pair, 'dir': 'BUY', 'pnl': pnl, 'result': 'WIN'})
                        position = None
                else:
                    if row['High'] >= position['sl']:
                        pnl = (position['entry'] - position['sl']) / pip * position['size']
                        balance += pnl
                        consec_loss += 1
                        max_consec = max(max_consec, consec_loss)
                        trades.append({'pair': pair, 'dir': 'SELL', 'pnl': pnl, 'result': 'LOSS'})
                        position = None
                    elif row['Low'] <= position['tp']:
                        pnl = (position['entry'] - position['tp']) / pip * position['size']
                        balance += pnl
                        consec_loss = 0
                        trades.append({'pair': pair, 'dir': 'SELL', 'pnl': pnl, 'result': 'WIN'})
                        position = None

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
                        'size': size
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

        wr = len(wins) / len(trades) * 100
        pf = total_win / total_loss if total_loss > 0 else 0
        avg_win = total_win / len(wins) if wins else 0
        avg_loss = total_loss / len(losses) if losses else 0

        return {
            'pair': pair,
            'trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'wr': wr,
            'pnl': balance - self.initial_balance,
            'pnl_pct': (balance - self.initial_balance) / self.initial_balance * 100,
            'pf': pf,
            'max_dd': max_dd,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'consec_loss': max_consec,
            'trade_list': trades
        }

    def _empty(self, pair):
        return {'pair': pair, 'trades': 0, 'wins': 0, 'losses': 0, 'wr': 0,
                'pnl': 0, 'pnl_pct': 0, 'pf': 0, 'max_dd': 0, 'avg_win': 0,
                'avg_loss': 0, 'consec_loss': 0, 'trade_list': []}

    def run_all(self, pairs=None, period="2y"):
        if pairs is None:
            pairs = CROSS_PAIRS

        print("\n" + "=" * 70)
        print("   BACKTEST V3 - STRATEGIE OPTIMISEE - 15 PAIRES CROSS")
        print("=" * 70)
        print("\nParametres:")
        print("  - EMA Cross: 8/21, Trend: 50")
        print("  - RSI: 30-70, ADX > 15")
        print("  - R:R dynamique: 1:1.8")
        print("  - Score minimum: 5/8 confirmations")
        print("-" * 70)

        for pair in pairs:
            print(f"  {pair}...", end=" ", flush=True)
            df = self.fetch_data(pair, period)

            if df is None or len(df) < 200:
                print("SKIP")
                continue

            r = self.run_pair(pair, df)
            self.results[pair] = r
            self.all_trades.extend(r['trade_list'])

            status = "[OK]" if r['pf'] >= 1.0 and r['trades'] >= 10 else "[--]"
            print(f"{status} {r['trades']} trades, WR: {r['wr']:.1f}%, PF: {r['pf']:.2f}, PnL: ${r['pnl']:.0f}")
            time_module.sleep(0.3)

        return self.results

    def report(self):
        lines = []
        lines.append("\n" + "=" * 80)
        lines.append("            RAPPORT BACKTEST V3 - STRATEGIE OPTIMISEE")
        lines.append("                  15 PAIRES CROSS - 2 ANS")
        lines.append("=" * 80)

        # Stats globales
        total_trades = sum(r['trades'] for r in self.results.values())
        total_wins = sum(r['wins'] for r in self.results.values())
        total_pnl = sum(r['pnl'] for r in self.results.values())

        lines.append("\n" + "-" * 80)
        lines.append("STATISTIQUES GLOBALES")
        lines.append("-" * 80)
        lines.append(f"Paires analysees:      {len(self.results)}")
        lines.append(f"Total trades:          {total_trades}")
        lines.append(f"Win Rate Global:       {total_wins/total_trades*100:.1f}%" if total_trades > 0 else "Win Rate: N/A")
        lines.append(f"PnL Total:             ${total_pnl:,.2f}")

        all_wins = sum(r['avg_win'] * r['wins'] for r in self.results.values() if r['wins'] > 0)
        all_losses = sum(r['avg_loss'] * r['losses'] for r in self.results.values() if r['losses'] > 0)
        global_pf = all_wins / all_losses if all_losses > 0 else 0
        lines.append(f"Profit Factor:         {global_pf:.2f}")

        # Tableau
        lines.append("\n" + "-" * 80)
        lines.append("RESULTATS PAR PAIRE")
        lines.append("-" * 80)
        lines.append(f"{'Paire':<10} {'Trades':>7} {'WR%':>7} {'PnL':>12} {'PF':>6} {'MaxDD':>7} {'Grade':>6}")
        lines.append("-" * 80)

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

            lines.append(f"{pair:<10} {r['trades']:>7} {r['wr']:>6.1f}% ${r['pnl']:>10,.2f} {r['pf']:>5.2f} {r['max_dd']:>6.1f}% {grade:>6}")

        # Paires profitables
        lines.append("\n" + "=" * 80)
        lines.append("PAIRES PROFITABLES (PF >= 1.0, Trades >= 10)")
        lines.append("=" * 80)

        if profitable:
            for pair, r, grade in profitable:
                lines.append(f"\n  [{grade}] {pair}")
                lines.append(f"      Trades: {r['trades']} | Win Rate: {r['wr']:.1f}%")
                lines.append(f"      PnL: ${r['pnl']:,.2f} ({r['pnl_pct']:+.1f}%)")
                lines.append(f"      Profit Factor: {r['pf']:.2f}")
                lines.append(f"      Max Drawdown: {r['max_dd']:.1f}%")
                lines.append(f"      Avg Win: ${r['avg_win']:.2f} | Avg Loss: ${r['avg_loss']:.2f}")
        else:
            lines.append("\n  Aucune paire profitable avec criteres stricts.")
            lines.append("  Paires proches du seuil:")
            near_profitable = [(p, r) for p, r in sorted_r if 0.9 <= r['pf'] < 1.0 and r['trades'] >= 10]
            for pair, r in near_profitable[:5]:
                lines.append(f"    - {pair}: WR={r['wr']:.1f}%, PF={r['pf']:.2f}")

        # Comparaison
        lines.append("\n" + "=" * 80)
        lines.append("COMPARAISON AVEC V1 (baseline)")
        lines.append("=" * 80)
        lines.append(f"\n{'Metrique':<25} {'V1':>15} {'V3':>15} {'Change':>15}")
        lines.append("-" * 70)
        lines.append(f"{'Total Trades':<25} {'2780':>15} {total_trades:>15}")
        lines.append(f"{'Win Rate':<25} {'34.0%':>15} {total_wins/total_trades*100:>14.1f}%" if total_trades > 0 else "")
        lines.append(f"{'Profit Factor':<25} {'0.86':>15} {global_pf:>14.2f}")
        lines.append(f"{'Paires Profitables':<25} {'1/15':>15} {f'{len(profitable)}/15':>15}")

        # Recommandations
        lines.append("\n" + "=" * 80)
        lines.append("RECOMMANDATIONS")
        lines.append("=" * 80)

        if profitable:
            lines.append("\n[OK] PAIRES RECOMMANDEES:")
            for pair, r, grade in [p for p in profitable if p[2] in ['A+', 'A', 'B+', 'B']]:
                lines.append(f"     * {pair} (Grade {grade}): WR={r['wr']:.1f}%, PF={r['pf']:.2f}")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)


def main():
    bt = BacktesterV3(initial_balance=10000, risk_per_trade=0.01)
    bt.run_all(CROSS_PAIRS, "2y")

    report = bt.report()
    print(report)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f"backtest_v3_{ts}.txt", 'w') as f:
        f.write(report)
    print(f"\n[EXPORT] backtest_v3_{ts}.txt")

    if bt.all_trades:
        pd.DataFrame(bt.all_trades).to_csv(f"trades_v3_{ts}.csv", index=False)
        print(f"[EXPORT] trades_v3_{ts}.csv")


if __name__ == "__main__":
    main()
