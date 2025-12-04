#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest 2 Ans - 0.25 Lot - Timeframe H1
========================================
Backtest complet sur 2 ans avec:
- Timeframe: H1 (1 heure) pour detection des signaux
- Capital: 10,000 USD
- Taille: 0.25 lot (25,000 unites)
- Limite perte journaliere: $500
- 6 paires robustes validees
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')

# Configuration
OPTIMAL_CONFIGS = {
    'EURCAD': {'rr': 2.5, 'adx': 15, 'rsi': (35, 65), 'score': 6, 'pf': 1.27},
    'EURJPY': {'rr': 1.8, 'adx': 20, 'rsi': (25, 75), 'score': 5, 'pf': 1.06},
    'GBPJPY': {'rr': 1.2, 'adx': 25, 'rsi': (30, 70), 'score': 6, 'pf': 1.10},
    'CHFJPY': {'rr': 1.5, 'adx': 25, 'rsi': (25, 75), 'score': 4, 'pf': 1.06},
    'CADJPY': {'rr': 2.5, 'adx': 25, 'rsi': (35, 65), 'score': 6, 'pf': 1.03},
    'GBPAUD': {'rr': 2.5, 'adx': 12, 'rsi': (35, 65), 'score': 6, 'pf': 1.00},
}

# Parametres de trading
CAPITAL = 10000
LOT_SIZE = 0.25  # 0.25 lot
UNITS = LOT_SIZE * 100000  # 25,000 unites
DAILY_LOSS_LIMIT = -500
PIP_VALUE = 2.50  # ~$2.50 par pip pour 0.25 lot

PIP_VALUES = {
    'EURCAD': 0.0001,
    'EURJPY': 0.01,
    'GBPJPY': 0.01,
    'CHFJPY': 0.01,
    'CADJPY': 0.01,
    'GBPAUD': 0.0001,
}


def fetch_data(pair: str, period: str = "730d") -> Optional[pd.DataFrame]:
    """Telecharge les donnees H1 sur 2 ans (max 730 jours pour H1)."""
    symbol = f"{pair}=X"
    try:
        df = yf.download(symbol, period=period, interval='1h', progress=False)
        if df.empty or len(df) < 500:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        if 'adj_close' in df.columns:
            df['close'] = df['adj_close']
        return df
    except Exception as e:
        print(f"Erreur fetch {pair}: {e}")
        return None


def calculate_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calcule tous les indicateurs."""
    df = df.copy()

    # EMAs (8/21/50)
    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    # RSI (14)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ADX (14)
    high, low, close = df['high'], df['low'], df['close']
    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1
    plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
    minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14).mean()
    df['atr'] = atr
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    df['adx'] = dx.rolling(window=14).mean()

    # MACD (12/26/9)
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Momentum (ROC 10)
    df['momentum'] = df['close'].pct_change(periods=10) * 100

    return df


def calculate_score(row, config, prev_row=None):
    """Calcule le score de confluence (0-8)."""
    score = 0
    direction = None
    adx_min = config['adx']
    rsi_low, rsi_high = config['rsi']
    ema_8, ema_21 = row['ema_8'], row['ema_21']

    if prev_row is not None:
        prev_ema_8, prev_ema_21 = prev_row['ema_8'], prev_row['ema_21']
        if prev_ema_8 <= prev_ema_21 and ema_8 > ema_21:
            score += 2
            direction = 'BUY'
        elif prev_ema_8 >= prev_ema_21 and ema_8 < ema_21:
            score += 2
            direction = 'SELL'

    if direction is None:
        return 0, None

    ema_50, price = row['ema_50'], row['close']
    if direction == 'BUY' and price > ema_50:
        score += 2
    elif direction == 'SELL' and price < ema_50:
        score += 2

    rsi = row['rsi']
    if rsi_low <= rsi <= rsi_high:
        score += 1

    if row['adx'] >= adx_min:
        score += 1

    macd_hist = row['macd_hist']
    if direction == 'BUY' and macd_hist > 0:
        score += 1
    elif direction == 'SELL' and macd_hist < 0:
        score += 1

    momentum = row['momentum']
    if direction == 'BUY' and momentum > 0:
        score += 1
    elif direction == 'SELL' and momentum < 0:
        score += 1

    return score, direction


def run_backtest(df: pd.DataFrame, pair: str, config: dict) -> List[dict]:
    """Execute le backtest et retourne la liste des trades."""
    df = df.copy()
    rr = config['rr']
    min_score = config['score']
    sl_mult = 1.5
    tp_mult = sl_mult * rr
    trades = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        score, direction = calculate_score(row, config, prev_row)

        if score >= min_score and direction is not None:
            entry_price = row['close']
            atr = row['atr']
            if pd.isna(atr) or atr <= 0:
                continue

            sl_dist = atr * sl_mult
            tp_dist = atr * tp_mult

            if direction == 'BUY':
                sl_price = entry_price - sl_dist
                tp_price = entry_price + tp_dist
            else:
                sl_price = entry_price + sl_dist
                tp_price = entry_price - tp_dist

            entry_time = df.index[i]
            result = None
            exit_price = None
            exit_time = None

            for j in range(i+1, min(i+100, len(df))):
                future_bar = df.iloc[j]
                if direction == 'BUY':
                    if future_bar['low'] <= sl_price:
                        result = 'LOSS'
                        exit_price = sl_price
                        exit_time = df.index[j]
                        break
                    if future_bar['high'] >= tp_price:
                        result = 'WIN'
                        exit_price = tp_price
                        exit_time = df.index[j]
                        break
                else:
                    if future_bar['high'] >= sl_price:
                        result = 'LOSS'
                        exit_price = sl_price
                        exit_time = df.index[j]
                        break
                    if future_bar['low'] <= tp_price:
                        result = 'WIN'
                        exit_price = tp_price
                        exit_time = df.index[j]
                        break

            if result is None:
                continue

            # Calcul du P&L avec 0.25 lot
            if direction == 'BUY':
                pips = (exit_price - entry_price) / PIP_VALUES.get(pair, 0.0001)
            else:
                pips = (entry_price - exit_price) / PIP_VALUES.get(pair, 0.0001)

            pnl = pips * PIP_VALUE  # $2.50 par pip pour 0.25 lot

            # Date du trade
            if hasattr(exit_time, 'date'):
                trade_date = exit_time.date()
            else:
                trade_date = pd.Timestamp(exit_time).date()

            trades.append({
                'pair': pair,
                'date': trade_date,
                'entry_time': entry_time,
                'exit_time': exit_time,
                'direction': direction,
                'entry': entry_price,
                'exit': exit_price,
                'result': result,
                'pips': pips,
                'pnl': pnl,
                'score': score
            })

    return trades


def analyze_daily_losses(all_trades: List[dict]) -> dict:
    """Analyse les pertes journalieres."""
    daily_pnl = defaultdict(float)
    daily_trades = defaultdict(list)

    for trade in all_trades:
        date = trade['date']
        daily_pnl[date] += trade['pnl']
        daily_trades[date].append(trade)

    days_over_limit = []
    worst_day = None
    worst_day_pnl = 0
    best_day = None
    best_day_pnl = 0

    for date, pnl in daily_pnl.items():
        if pnl < DAILY_LOSS_LIMIT:
            days_over_limit.append({
                'date': date,
                'pnl': pnl,
                'trades': daily_trades[date]
            })
        if pnl < worst_day_pnl:
            worst_day = date
            worst_day_pnl = pnl
        if pnl > best_day_pnl:
            best_day = date
            best_day_pnl = pnl

    return {
        'daily_pnl': dict(daily_pnl),
        'daily_trades': dict(daily_trades),
        'days_over_limit': days_over_limit,
        'worst_day': worst_day,
        'worst_day_pnl': worst_day_pnl,
        'best_day': best_day,
        'best_day_pnl': best_day_pnl,
        'total_days': len(daily_pnl),
        'losing_days': sum(1 for p in daily_pnl.values() if p < 0),
        'winning_days': sum(1 for p in daily_pnl.values() if p > 0)
    }


def main():
    print("=" * 90)
    print("   BACKTEST 2 ANS - 0.25 LOT - TIMEFRAME H1")
    print("=" * 90)
    print(f"\nCapital: ${CAPITAL:,.2f}")
    print(f"Taille: {LOT_SIZE} lot ({UNITS:,.0f} unites)")
    print(f"Valeur pip: ~${PIP_VALUE:.2f}")
    print(f"Limite perte/jour: ${abs(DAILY_LOSS_LIMIT)}")
    print(f"Timeframe: H1 (detection signaux)")
    print(f"Periode: ~2 ans (730 jours max pour H1)")
    print(f"Paires: {', '.join(OPTIMAL_CONFIGS.keys())}")
    print("\nTelechargement des donnees H1...")

    all_trades = []
    pair_stats = {}

    for pair in OPTIMAL_CONFIGS.keys():
        config = OPTIMAL_CONFIGS[pair]
        print(f"\n  {pair}...", end=" ", flush=True)

        df = fetch_data(pair, period="730d")
        if df is None or len(df) < 500:
            print("SKIP (donnees insuffisantes)")
            continue

        print(f"{len(df)} barres H1...", end=" ", flush=True)

        df = calculate_indicators(df, config)
        trades = run_backtest(df, pair, config)
        all_trades.extend(trades)

        # Stats par paire
        if trades:
            wins = [t for t in trades if t['result'] == 'WIN']
            losses = [t for t in trades if t['result'] == 'LOSS']
            total_pnl = sum(t['pnl'] for t in trades)
            total_win = sum(t['pnl'] for t in wins)
            total_loss = abs(sum(t['pnl'] for t in losses))
            pf = total_win / total_loss if total_loss > 0 else float('inf')
            wr = len(wins) / len(trades) * 100 if trades else 0

            pair_stats[pair] = {
                'trades': len(trades),
                'wins': len(wins),
                'losses': len(losses),
                'wr': wr,
                'pf': pf,
                'pnl': total_pnl,
                'avg_win': total_win / len(wins) if wins else 0,
                'avg_loss': total_loss / len(losses) if losses else 0
            }

            status = "OK" if total_pnl > 0 else "XX"
            print(f"[{status}] {len(trades)} trades, WR={wr:.1f}%, PF={pf:.2f}, P&L=${total_pnl:+,.2f}")
        else:
            print("0 trades")

    # Analyse des pertes journalieres
    daily_analysis = analyze_daily_losses(all_trades)

    # Calculs globaux
    total_trades = len(all_trades)
    total_wins = sum(1 for t in all_trades if t['result'] == 'WIN')
    total_losses = sum(1 for t in all_trades if t['result'] == 'LOSS')
    total_pnl = sum(t['pnl'] for t in all_trades)
    total_win_amount = sum(t['pnl'] for t in all_trades if t['result'] == 'WIN')
    total_loss_amount = abs(sum(t['pnl'] for t in all_trades if t['result'] == 'LOSS'))
    global_pf = total_win_amount / total_loss_amount if total_loss_amount > 0 else 0
    global_wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    # Calculer les dates de debut et fin
    if all_trades:
        start_date = min(t['entry_time'] for t in all_trades)
        end_date = max(t['exit_time'] for t in all_trades)
        days_traded = (end_date - start_date).days
    else:
        days_traded = 0

    # RAPPORT
    print("\n")
    print("=" * 90)
    print("              RAPPORT BACKTEST 2 ANS - 0.25 LOT")
    print("=" * 90)
    print(f"\nDate du rapport: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Periode testee: {days_traded} jours (~{days_traded/365:.1f} ans)")
    print(f"Capital initial: ${CAPITAL:,.2f}")
    print(f"Taille position: {LOT_SIZE} lot")
    print("=" * 90)

    # Resume global
    print("\n" + "-" * 90)
    print("RESUME GLOBAL")
    print("-" * 90)
    print(f"Total trades:       {total_trades}")
    print(f"Trades gagnants:    {total_wins} ({global_wr:.1f}%)")
    print(f"Trades perdants:    {total_losses}")
    print(f"Profit Factor:      {global_pf:.2f}")
    print(f"P&L Total:          ${total_pnl:+,.2f}")
    print(f"Capital final:      ${CAPITAL + total_pnl:,.2f}")
    print(f"ROI sur {days_traded/365:.1f} ans:    {total_pnl/CAPITAL*100:+.2f}%")
    print(f"ROI annualise:      {(total_pnl/CAPITAL*100)/(days_traded/365):+.2f}%/an")

    # Performance par paire
    print("\n" + "-" * 90)
    print("PERFORMANCE PAR PAIRE")
    print("-" * 90)
    print(f"{'Paire':<10} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'PF':>8} {'P&L':>14} {'Statut':>8}")
    print("-" * 90)

    sorted_pairs = sorted(pair_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)
    for pair, stats in sorted_pairs:
        status = "[OK]" if stats['pnl'] > 0 else "[XX]"
        print(f"{pair:<10} {stats['trades']:>8} {stats['wins']:>6} {stats['wr']:>7.1f}% "
              f"{stats['pf']:>7.2f} ${stats['pnl']:>+12,.2f} {status:>8}")

    print("-" * 90)
    print(f"{'TOTAL':<10} {total_trades:>8} {total_wins:>6} {global_wr:>7.1f}% "
          f"{global_pf:>7.2f} ${total_pnl:>+12,.2f}")

    # Analyse des pertes journalieres
    print("\n" + "-" * 90)
    print("ANALYSE PERTES JOURNALIERES (Limite: -$500/jour)")
    print("-" * 90)
    print(f"Total jours de trading:    {daily_analysis['total_days']}")
    print(f"Jours gagnants:            {daily_analysis['winning_days']} ({daily_analysis['winning_days']/daily_analysis['total_days']*100:.1f}%)")
    print(f"Jours perdants:            {daily_analysis['losing_days']} ({daily_analysis['losing_days']/daily_analysis['total_days']*100:.1f}%)")
    print(f"Meilleur jour:             {daily_analysis['best_day']} -> ${daily_analysis['best_day_pnl']:+,.2f}")
    print(f"Pire jour:                 {daily_analysis['worst_day']} -> ${daily_analysis['worst_day_pnl']:+,.2f}")

    print("\n" + "=" * 90)
    if daily_analysis['days_over_limit']:
        print(f"[XX] ATTENTION: {len(daily_analysis['days_over_limit'])} JOUR(S) ONT DEPASSE -$500")
        print("=" * 90)
        print(f"\n{'Date':<12} {'Perte':>12} {'Depassement':>12} {'Nb Trades':>10}")
        print("-" * 50)
        for day in sorted(daily_analysis['days_over_limit'], key=lambda x: x['pnl'])[:10]:
            depassement = day['pnl'] - DAILY_LOSS_LIMIT
            print(f"{str(day['date']):<12} ${day['pnl']:>+10,.2f} ${depassement:>+10,.2f} {len(day['trades']):>10}")
    else:
        print("[OK] AUCUN JOUR N'A DEPASSE LA LIMITE DE -$500")
        print("=" * 90)
        print(f"\nLa strategie respecte la limite de perte journaliere.")
        print(f"Pire jour: ${daily_analysis['worst_day_pnl']:,.2f}")
        print(f"Marge de securite: ${DAILY_LOSS_LIMIT - daily_analysis['worst_day_pnl']:.2f}")

    # Analyse par mois
    print("\n" + "-" * 90)
    print("PERFORMANCE MENSUELLE")
    print("-" * 90)

    monthly_pnl = defaultdict(float)
    for trade in all_trades:
        month_key = trade['date'].strftime('%Y-%m') if hasattr(trade['date'], 'strftime') else str(trade['date'])[:7]
        monthly_pnl[month_key] += trade['pnl']

    print(f"{'Mois':<10} {'P&L':>14} {'Cumul':>14} {'Statut':>10}")
    print("-" * 50)

    cumul = 0
    profitable_months = 0
    losing_months = 0
    for month in sorted(monthly_pnl.keys()):
        pnl = monthly_pnl[month]
        cumul += pnl
        status = "[OK]" if pnl > 0 else "[XX]"
        if pnl > 0:
            profitable_months += 1
        else:
            losing_months += 1
        print(f"{month:<10} ${pnl:>+12,.2f} ${cumul:>+12,.2f} {status:>10}")

    print("-" * 50)
    print(f"Mois profitables: {profitable_months}/{len(monthly_pnl)} ({profitable_months/len(monthly_pnl)*100:.1f}%)")

    # Score final
    print("\n" + "=" * 90)
    if total_pnl > 5000:
        grade = "A+ - EXCELLENT"
    elif total_pnl > 2000:
        grade = "A - TRES BON"
    elif total_pnl > 0:
        grade = "B - PROFITABLE"
    elif total_pnl > -2000:
        grade = "C - FAIBLE"
    else:
        grade = "F - A REVOIR"

    limit_respected = len(daily_analysis['days_over_limit']) == 0

    print(f"SCORE GLOBAL: {grade}")
    print(f"ROI Total: {total_pnl/CAPITAL*100:+.2f}% sur ~2 ans")
    print(f"ROI Annualise: {(total_pnl/CAPITAL*100)/(days_traded/365):+.2f}%/an")
    print(f"Limite -$500/jour: {'RESPECTEE' if limit_respected else 'NON RESPECTEE'}")
    print(f"Capital: ${CAPITAL:,.2f} -> ${CAPITAL + total_pnl:,.2f}")
    print("=" * 90)

    return {
        'total_pnl': total_pnl,
        'total_trades': total_trades,
        'global_pf': global_pf,
        'global_wr': global_wr,
        'daily_analysis': daily_analysis,
        'pair_stats': pair_stats
    }


if __name__ == "__main__":
    main()
