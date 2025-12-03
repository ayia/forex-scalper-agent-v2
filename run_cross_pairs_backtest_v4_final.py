#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross Pairs Backtest V4 FINAL - V3 + Trailing Stop
===================================================
Base sur V3 (3 paires profitables) avec ajout de:
1. Trailing Stop active a +1R
2. Breakeven a +0.7R
3. Sessions optimales (optionnel)
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


class OptimizedStrategyV4:
    """Strategie V3 + Trailing Stop."""

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

        # ATR & R:R
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
        df['score'] = 0

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

            # Score de confirmation
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
                df.iloc[i, df.columns.get_loc('score')] = buy_score
            elif sell_score >= 5:
                df.iloc[i, df.columns.get_loc('signal')] = -1
                df.iloc[i, df.columns.get_loc('sl')] = current['Close'] + (current['atr'] * self.sl_mult)
                df.iloc[i, df.columns.get_loc('tp')] = current['Close'] - (current['atr'] * self.tp_mult)
                df.iloc[i, df.columns.get_loc('score')] = sell_score

        return df


class BacktesterV4Final:
    """Backtester V4 avec trailing stop et breakeven."""

    def __init__(self, initial_balance=10000, risk_per_trade=0.01, use_trailing=True):
        self.initial_balance = initial_balance
        self.risk_per_trade = risk_per_trade
        self.use_trailing = use_trailing
        self.strategy = OptimizedStrategyV4()
        self.results = {}
        self.all_trades = []

        # Trailing parameters
        self.trailing_start = 1.0   # Active a +1R
        self.breakeven_at = 0.7     # BE a +0.7R

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

            # Gerer position ouverte
            if position:
                current_price = row['Close']
                entry = position['entry']
                initial_risk = position['risk']
                sl = position['sl']
                tp = position['tp']

                if position['dir'] == 'BUY':
                    current_r = (current_price - entry) / initial_risk if initial_risk > 0 else 0

                    # Trailing stop (si active)
                    if self.use_trailing:
                        # Breakeven
                        if current_r >= self.breakeven_at and not position.get('be_set'):
                            position['sl'] = entry + (pip * 2)
                            position['be_set'] = True

                        # Trailing
                        if current_r >= self.trailing_start:
                            new_sl = current_price - (initial_risk * 0.7)
                            if new_sl > position['sl']:
                                position['sl'] = new_sl

                    sl = position['sl']

                    # Check SL
                    if row['Low'] <= sl:
                        pnl = (sl - entry) / pip * position['size']
                        balance += pnl
                        result = 'WIN' if pnl > 0 else 'LOSS'
                        exit_type = 'BE' if position.get('be_set') and pnl >= 0 else 'SL'
                        if result == 'LOSS':
                            consec_loss += 1
                            max_consec = max(max_consec, consec_loss)
                        else:
                            consec_loss = 0
                        trades.append({
                            'pair': pair, 'dir': 'BUY', 'pnl': pnl,
                            'result': result, 'exit_type': exit_type,
                            'score': position['score']
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
                            'score': position['score']
                        })
                        position = None

                else:  # SELL
                    current_r = (entry - current_price) / initial_risk if initial_risk > 0 else 0

                    if self.use_trailing:
                        if current_r >= self.breakeven_at and not position.get('be_set'):
                            position['sl'] = entry - (pip * 2)
                            position['be_set'] = True

                        if current_r >= self.trailing_start:
                            new_sl = current_price + (initial_risk * 0.7)
                            if new_sl < position['sl']:
                                position['sl'] = new_sl

                    sl = position['sl']

                    if row['High'] >= sl:
                        pnl = (entry - sl) / pip * position['size']
                        balance += pnl
                        result = 'WIN' if pnl > 0 else 'LOSS'
                        exit_type = 'BE' if position.get('be_set') and pnl >= 0 else 'SL'
                        if result == 'LOSS':
                            consec_loss += 1
                            max_consec = max(max_consec, consec_loss)
                        else:
                            consec_loss = 0
                        trades.append({
                            'pair': pair, 'dir': 'SELL', 'pnl': pnl,
                            'result': result, 'exit_type': exit_type,
                            'score': position['score']
                        })
                        position = None

                    elif row['Low'] <= tp:
                        pnl = (entry - tp) / pip * position['size']
                        balance += pnl
                        consec_loss = 0
                        trades.append({
                            'pair': pair, 'dir': 'SELL', 'pnl': pnl,
                            'result': 'WIN', 'exit_type': 'TP',
                            'score': position['score']
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
                        'score': row['score'],
                        'be_set': False
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

        tp_exits = len([t for t in trades if t.get('exit_type') == 'TP'])
        be_exits = len([t for t in trades if t.get('exit_type') == 'BE'])

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
            'be_exits': be_exits,
            'trade_list': trades
        }

    def _empty(self, pair):
        return {'pair': pair, 'trades': 0, 'wins': 0, 'losses': 0, 'wr': 0,
                'pnl': 0, 'pnl_pct': 0, 'pf': 0, 'max_dd': 0, 'avg_win': 0,
                'avg_loss': 0, 'consec_loss': 0, 'tp_exits': 0, 'be_exits': 0,
                'trade_list': []}

    def run_all(self, pairs=None, period="2y"):
        if pairs is None:
            pairs = CROSS_PAIRS

        mode = "V3 + Trailing Stop" if self.use_trailing else "V3 Sans Trailing"

        print("\n" + "=" * 75)
        print(f"   BACKTEST V4 FINAL - {mode}")
        print("=" * 75)
        print("\nParametres:")
        print("  - Strategie: V3 (score >= 5/8)")
        print("  - R:R: 1:1.8")
        if self.use_trailing:
            print(f"  - Trailing Stop: Active a +{self.trailing_start}R")
            print(f"  - Breakeven: a +{self.breakeven_at}R")
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
        mode = "V3 + Trailing Stop" if self.use_trailing else "V3 Sans Trailing"

        lines.append("\n" + "=" * 85)
        lines.append(f"              RAPPORT BACKTEST V4 FINAL - {mode}")
        lines.append("                      15 PAIRES CROSS - 2 ANS")
        lines.append("=" * 85)

        # Stats globales
        total_trades = sum(r['trades'] for r in self.results.values())
        total_wins = sum(r['wins'] for r in self.results.values())
        total_pnl = sum(r['pnl'] for r in self.results.values())

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
        lines.append(f"{'Trades':<30} {total_trades:>15,} {jpy_trades:>15,} {other_trades:>15,}")

        wr_global = total_wins/total_trades*100 if total_trades > 0 else 0
        wr_jpy = jpy_wins/jpy_trades*100 if jpy_trades > 0 else 0
        wr_other = other_wins/other_trades*100 if other_trades > 0 else 0
        lines.append(f"{'Win Rate':<30} {wr_global:>14.1f}% {wr_jpy:>14.1f}% {wr_other:>14.1f}%")
        lines.append(f"{'PnL':<30} ${total_pnl:>13,.0f} ${jpy_pnl:>13,.0f} ${other_pnl:>13,.0f}")

        # Profit Factor
        all_wins = sum(r['avg_win'] * r['wins'] for r in self.results.values() if r['wins'] > 0)
        all_losses = sum(r['avg_loss'] * r['losses'] for r in self.results.values() if r['losses'] > 0)
        global_pf = all_wins / all_losses if all_losses > 0 else 0
        lines.append(f"{'Profit Factor':<30} {global_pf:>15.2f}")

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
            be_pct = r['be_exits'] / r['trades'] * 100 if r['trades'] > 0 else 0

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
                lines.append(f"      TP Exits: {r['tp_exits']} ({r['tp_exits']/r['trades']*100:.1f}%)")
                lines.append(f"      BE Exits: {r['be_exits']} ({r['be_exits']/r['trades']*100:.1f}%)")
        else:
            lines.append("\n  Aucune paire profitable avec criteres stricts.")

        lines.append("\n" + "=" * 85)

        return "\n".join(lines)


def main():
    print("\n" + "=" * 75)
    print("   COMPARAISON: V3 vs V3+Trailing")
    print("=" * 75)

    # Test sans trailing d'abord
    print("\n>>> TEST SANS TRAILING STOP <<<")
    bt_no_trail = BacktesterV4Final(use_trailing=False)
    bt_no_trail.run_all(CROSS_PAIRS, "2y")

    # Test avec trailing
    print("\n\n>>> TEST AVEC TRAILING STOP <<<")
    bt_trail = BacktesterV4Final(use_trailing=True)
    bt_trail.run_all(CROSS_PAIRS, "2y")

    # Rapports
    report_no_trail = bt_no_trail.report()
    report_trail = bt_trail.report()

    print(report_no_trail)
    print(report_trail)

    # Comparaison
    print("\n" + "=" * 85)
    print("              COMPARAISON FINALE: Sans Trailing vs Avec Trailing")
    print("=" * 85)

    pf_no = sum(r['avg_win'] * r['wins'] for r in bt_no_trail.results.values() if r['wins'] > 0)
    pl_no = sum(r['avg_loss'] * r['losses'] for r in bt_no_trail.results.values() if r['losses'] > 0)
    pf_no = pf_no / pl_no if pl_no > 0 else 0

    pf_tr = sum(r['avg_win'] * r['wins'] for r in bt_trail.results.values() if r['wins'] > 0)
    pl_tr = sum(r['avg_loss'] * r['losses'] for r in bt_trail.results.values() if r['losses'] > 0)
    pf_tr = pf_tr / pl_tr if pl_tr > 0 else 0

    trades_no = sum(r['trades'] for r in bt_no_trail.results.values())
    trades_tr = sum(r['trades'] for r in bt_trail.results.values())

    wins_no = sum(r['wins'] for r in bt_no_trail.results.values())
    wins_tr = sum(r['wins'] for r in bt_trail.results.values())

    pnl_no = sum(r['pnl'] for r in bt_no_trail.results.values())
    pnl_tr = sum(r['pnl'] for r in bt_trail.results.values())

    prof_no = len([r for r in bt_no_trail.results.values() if r['pf'] >= 1.0 and r['trades'] >= 10])
    prof_tr = len([r for r in bt_trail.results.values() if r['pf'] >= 1.0 and r['trades'] >= 10])

    print(f"\n{'Metrique':<25} {'Sans Trailing':>15} {'Avec Trailing':>15} {'Delta':>15}")
    print("-" * 70)
    print(f"{'Trades':<25} {trades_no:>15,} {trades_tr:>15,}")
    print(f"{'Win Rate':<25} {wins_no/trades_no*100:>14.1f}% {wins_tr/trades_tr*100:>14.1f}% {(wins_tr/trades_tr-wins_no/trades_no)*100:>+14.1f}%")
    print(f"{'Profit Factor':<25} {pf_no:>15.2f} {pf_tr:>15.2f} {pf_tr-pf_no:>+15.2f}")
    print(f"{'PnL Total':<25} ${pnl_no:>14,.0f} ${pnl_tr:>14,.0f} ${pnl_tr-pnl_no:>+14,.0f}")
    print(f"{'Paires Profitables':<25} {prof_no:>15}/15 {prof_tr:>15}/15")

    # Conclusion
    print("\n" + "-" * 70)
    if pf_tr > pf_no:
        print(f"  RESULTAT: Le trailing stop AMELIORE les performances (PF +{(pf_tr-pf_no)/pf_no*100:.1f}%)")
    elif pf_tr < pf_no:
        print(f"  RESULTAT: Le trailing stop DEGRADE les performances (PF {(pf_tr-pf_no)/pf_no*100:.1f}%)")
    else:
        print("  RESULTAT: Pas de difference significative")

    print("\n" + "=" * 85)

    # Export
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    with open(f"backtest_v4_final_{ts}.txt", 'w') as f:
        f.write(report_no_trail)
        f.write("\n\n" + "="*85 + "\n")
        f.write(report_trail)

    print(f"\n[EXPORT] backtest_v4_final_{ts}.txt")


if __name__ == "__main__":
    main()
