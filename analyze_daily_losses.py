#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyse des Pertes Journalieres
===============================
Verifie si la limite de -$500/jour a ete respectee pendant le backtest.
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

# Configuration identique
OPTIMAL_CONFIGS = {
    'EURCAD': {'rr': 2.5, 'adx': 15, 'rsi': (35, 65), 'score': 6, 'pf': 1.27},
    'EURJPY': {'rr': 1.8, 'adx': 20, 'rsi': (25, 75), 'score': 5, 'pf': 1.06},
    'GBPJPY': {'rr': 1.2, 'adx': 25, 'rsi': (30, 70), 'score': 6, 'pf': 1.10},
    'CHFJPY': {'rr': 1.5, 'adx': 25, 'rsi': (25, 75), 'score': 4, 'pf': 1.06},
    'CADJPY': {'rr': 2.5, 'adx': 25, 'rsi': (35, 65), 'score': 6, 'pf': 1.03},
    'GBPAUD': {'rr': 2.5, 'adx': 12, 'rsi': (35, 65), 'score': 6, 'pf': 1.00},
}

CAPITAL = 10000
LOT_SIZE = 1.0
UNITS = LOT_SIZE * 100000
DAILY_LOSS_LIMIT = -500  # Limite de perte journaliere

PIP_VALUES = {
    'EURCAD': 0.0001,
    'EURJPY': 0.01,
    'GBPJPY': 0.01,
    'CHFJPY': 0.01,
    'CADJPY': 0.01,
    'GBPAUD': 0.0001,
}


def fetch_data(pair: str, days: int = 35) -> Optional[pd.DataFrame]:
    symbol = f"{pair}=X"
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'), interval='1h', progress=False)
        if df.empty or len(df) < 100:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        if 'adj_close' in df.columns:
            df['close'] = df['adj_close']
        return df
    except:
        return None


def calculate_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()
    df['ema_8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

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

    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['momentum'] = df['close'].pct_change(periods=10) * 100
    return df


def calculate_score(row, config, prev_row=None):
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


def run_backtest_with_daily_tracking(df: pd.DataFrame, pair: str, config: dict) -> List[dict]:
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

            pip_value = 10.0
            if direction == 'BUY':
                pips = (exit_price - entry_price) / PIP_VALUES.get(pair, 0.0001)
            else:
                pips = (entry_price - exit_price) / PIP_VALUES.get(pair, 0.0001)

            pnl = pips * pip_value

            # Extraire la date du trade (pour regroupement journalier)
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
                'result': result,
                'pnl': pnl
            })

    return trades


def main():
    print("=" * 80)
    print("   ANALYSE DES PERTES JOURNALIERES - LIMITE $500/JOUR")
    print("=" * 80)
    print(f"\nCapital: ${CAPITAL:,.2f}")
    print(f"Taille: {LOT_SIZE} lot")
    print(f"Limite de perte journaliere: ${abs(DAILY_LOSS_LIMIT)}")
    print("\nCollecte des trades...")

    all_trades = []

    for pair in OPTIMAL_CONFIGS.keys():
        config = OPTIMAL_CONFIGS[pair]
        print(f"  {pair}...", end=" ", flush=True)

        df = fetch_data(pair, days=35)
        if df is None or len(df) < 100:
            print("SKIP")
            continue

        df = calculate_indicators(df, config)
        trades = run_backtest_with_daily_tracking(df, pair, config)
        all_trades.extend(trades)
        print(f"{len(trades)} trades")

    # Regrouper par jour
    daily_pnl = defaultdict(float)
    daily_trades = defaultdict(list)

    for trade in all_trades:
        date = trade['date']
        daily_pnl[date] += trade['pnl']
        daily_trades[date].append(trade)

    # Analyser les jours
    print("\n" + "=" * 80)
    print("RESUME PAR JOUR")
    print("=" * 80)
    print(f"{'Date':<12} {'Trades':>8} {'Wins':>6} {'Losses':>8} {'P&L':>14} {'Status':>12}")
    print("-" * 80)

    days_over_limit = []
    worst_day = None
    worst_day_pnl = 0
    best_day = None
    best_day_pnl = 0

    for date in sorted(daily_pnl.keys()):
        pnl = daily_pnl[date]
        trades = daily_trades[date]
        wins = sum(1 for t in trades if t['result'] == 'WIN')
        losses = sum(1 for t in trades if t['result'] == 'LOSS')

        if pnl < DAILY_LOSS_LIMIT:
            status = "[DEPASSE]"
            days_over_limit.append((date, pnl, trades))
        elif pnl < 0:
            status = "[PERTE]"
        else:
            status = "[OK]"

        if pnl < worst_day_pnl:
            worst_day = date
            worst_day_pnl = pnl
        if pnl > best_day_pnl:
            best_day = date
            best_day_pnl = pnl

        print(f"{str(date):<12} {len(trades):>8} {wins:>6} {losses:>8} ${pnl:>+12,.2f} {status:>12}")

    print("-" * 80)

    # Resume
    total_days = len(daily_pnl)
    losing_days = sum(1 for p in daily_pnl.values() if p < 0)
    winning_days = sum(1 for p in daily_pnl.values() if p > 0)

    print("\n" + "=" * 80)
    print("ANALYSE DE LA LIMITE JOURNALIERE")
    print("=" * 80)

    print(f"\nTotal jours de trading: {total_days}")
    print(f"Jours gagnants: {winning_days} ({winning_days/total_days*100:.1f}%)")
    print(f"Jours perdants: {losing_days} ({losing_days/total_days*100:.1f}%)")
    print(f"\nMeilleur jour: {best_day} -> ${best_day_pnl:+,.2f}")
    print(f"Pire jour:     {worst_day} -> ${worst_day_pnl:+,.2f}")

    print(f"\n{'='*80}")
    if days_over_limit:
        print(f"[XX] ATTENTION: {len(days_over_limit)} JOUR(S) ONT DEPASSE LA LIMITE DE -$500")
        print("="*80)
        for date, pnl, trades in days_over_limit:
            print(f"\n  Date: {date}")
            print(f"  Perte totale: ${pnl:,.2f}")
            print(f"  Depassement: ${pnl - DAILY_LOSS_LIMIT:,.2f}")
            print(f"  Trades ce jour:")
            for t in trades:
                print(f"    - {t['pair']} {t['direction']}: ${t['pnl']:+,.2f} ({t['result']})")
    else:
        print("[OK] AUCUN JOUR N'A DEPASSE LA LIMITE DE -$500")
        print("="*80)
        print(f"\nLa strategie respecte la limite de perte journaliere de ${abs(DAILY_LOSS_LIMIT)}.")
        print(f"Pire jour: ${worst_day_pnl:,.2f} (marge de ${DAILY_LOSS_LIMIT - worst_day_pnl:,.2f})")

    # Recommandation
    print("\n" + "=" * 80)
    print("RECOMMANDATION")
    print("=" * 80)

    if days_over_limit:
        print("\nPour respecter la limite de -$500/jour, vous pouvez:")
        print("  1. Reduire la taille des positions (ex: 0.5 lot au lieu de 1 lot)")
        print("  2. Limiter le nombre de trades par jour")
        print("  3. Implementer un stop journalier automatique")
    else:
        print(f"\nAvec 1 lot, la limite de -$500/jour est respectee.")
        print(f"Marge de securite moyenne: ${abs(DAILY_LOSS_LIMIT) - abs(worst_day_pnl):.2f}")


if __name__ == "__main__":
    main()
