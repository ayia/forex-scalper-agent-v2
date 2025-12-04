#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest Dernier Mois - Strategie Optimized Cross
=================================================
Backtest detaille sur le dernier mois avec:
- Timeframe: H1 (1 heure)
- Capital: 10,000 USD
- Taille: 1.0 lot standard (100,000 unites)
- 6 paires robustes validees

Ce backtest utilise EXACTEMENT la meme logique de detection
de signaux que le scanner --optimized-cross.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION IDENTIQUE AU SCANNER --optimized-cross
# =============================================================================
OPTIMAL_CONFIGS = {
    'EURCAD': {'rr': 2.5, 'adx': 15, 'rsi': (35, 65), 'score': 6, 'pf': 1.27},
    'EURJPY': {'rr': 1.8, 'adx': 20, 'rsi': (25, 75), 'score': 5, 'pf': 1.06},
    'GBPJPY': {'rr': 1.2, 'adx': 25, 'rsi': (30, 70), 'score': 6, 'pf': 1.10},
    'CHFJPY': {'rr': 1.5, 'adx': 25, 'rsi': (25, 75), 'score': 4, 'pf': 1.06},
    'CADJPY': {'rr': 2.5, 'adx': 25, 'rsi': (35, 65), 'score': 6, 'pf': 1.03},
    'GBPAUD': {'rr': 2.5, 'adx': 12, 'rsi': (35, 65), 'score': 6, 'pf': 1.00},
}

# Parametres de trading
CAPITAL = 10000  # USD
LOT_SIZE = 1.0   # 1 lot standard = 100,000 unites
UNITS = LOT_SIZE * 100000  # 100,000 unites

# Valeur du pip par paire (pour 1 lot standard)
# Pour les paires XXX/JPY: 1 pip = 0.01, valeur = (0.01 / rate) * 100000
# Pour les autres paires: 1 pip = 0.0001, valeur = (0.0001 / rate) * 100000
PIP_VALUES = {
    'EURCAD': 0.0001,
    'EURJPY': 0.01,
    'GBPJPY': 0.01,
    'CHFJPY': 0.01,
    'CADJPY': 0.01,
    'GBPAUD': 0.0001,
}


def fetch_data(pair: str, days: int = 35) -> Optional[pd.DataFrame]:
    """Telecharge les donnees H1 pour le dernier mois."""
    symbol = f"{pair}=X"

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        df = yf.download(
            symbol,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1h',
            progress=False
        )

        if df.empty or len(df) < 100:
            return None

        # Normaliser les colonnes
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
    """Calcule tous les indicateurs - IDENTIQUE au scanner."""
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
    high = df['high']
    low = df['low']
    close = df['close']

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


def calculate_score(row: pd.Series, config: dict, prev_row: pd.Series = None) -> tuple:
    """
    Calcule le score de confluence (0-8) - IDENTIQUE au scanner.
    Retourne (score, direction, details)
    """
    score = 0
    direction = None
    details = []

    adx_min = config['adx']
    rsi_low, rsi_high = config['rsi']

    # 1. EMA Cross (2 points) - MUST HAVE
    ema_8 = row['ema_8']
    ema_21 = row['ema_21']

    bullish_cross = False
    bearish_cross = False

    if prev_row is not None:
        prev_ema_8 = prev_row['ema_8']
        prev_ema_21 = prev_row['ema_21']

        # Bullish: EMA8 crosses above EMA21
        if prev_ema_8 <= prev_ema_21 and ema_8 > ema_21:
            bullish_cross = True
            score += 2
            direction = 'BUY'
            details.append("EMA Cross UP (+2)")
        # Bearish: EMA8 crosses below EMA21
        elif prev_ema_8 >= prev_ema_21 and ema_8 < ema_21:
            bearish_cross = True
            score += 2
            direction = 'SELL'
            details.append("EMA Cross DOWN (+2)")

    if not bullish_cross and not bearish_cross:
        return 0, None, []

    # 2. Trend alignment avec EMA50 (2 points)
    ema_50 = row['ema_50']
    price = row['close']

    if direction == 'BUY' and price > ema_50:
        score += 2
        details.append("Price > EMA50 (+2)")
    elif direction == 'SELL' and price < ema_50:
        score += 2
        details.append("Price < EMA50 (+2)")

    # 3. RSI in range (1 point)
    rsi = row['rsi']
    if rsi_low <= rsi <= rsi_high:
        score += 1
        details.append(f"RSI {rsi:.1f} in range (+1)")

    # 4. ADX above threshold (1 point)
    adx = row['adx']
    if adx >= adx_min:
        score += 1
        details.append(f"ADX {adx:.1f} >= {adx_min} (+1)")

    # 5. MACD alignment (1 point)
    macd_hist = row['macd_hist']
    if direction == 'BUY' and macd_hist > 0:
        score += 1
        details.append("MACD Bullish (+1)")
    elif direction == 'SELL' and macd_hist < 0:
        score += 1
        details.append("MACD Bearish (+1)")

    # 6. Momentum (1 point)
    momentum = row['momentum']
    if direction == 'BUY' and momentum > 0:
        score += 1
        details.append("Momentum UP (+1)")
    elif direction == 'SELL' and momentum < 0:
        score += 1
        details.append("Momentum DOWN (+1)")

    return score, direction, details


def calculate_pip_value(pair: str, rate: float) -> float:
    """Calcule la valeur d'un pip en USD pour 1 lot."""
    pip_size = PIP_VALUES.get(pair, 0.0001)

    if pair.endswith('JPY'):
        # Pour XXX/JPY: (0.01 / rate) * 100000
        pip_value = (pip_size / rate) * UNITS
    elif pair.startswith('EUR') or pair.startswith('GBP'):
        # Pour EUR/XXX ou GBP/XXX:
        # Approximation: on utilise un taux moyen
        pip_value = pip_size * UNITS / rate
    else:
        pip_value = pip_size * UNITS / rate

    # Convertir en USD (approximation)
    # Pour simplifier, on utilise une valeur fixe typique
    return 10.0  # ~$10 par pip pour 1 lot standard (approximation)


def run_backtest(df: pd.DataFrame, pair: str, config: dict) -> dict:
    """Execute le backtest sur une paire."""
    df = df.copy()

    rr = config['rr']
    min_score = config['score']
    sl_mult = 1.5
    tp_mult = sl_mult * rr

    trades = []
    equity = CAPITAL
    peak_equity = CAPITAL
    max_drawdown = 0

    # Parcourir les donnees
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]

        score, direction, details = calculate_score(row, config, prev_row)

        # Signal valide si score >= min_score
        if score >= min_score and direction is not None:
            entry_price = row['close']
            atr = row['atr']

            if pd.isna(atr) or atr <= 0:
                continue

            # Calcul SL/TP - IDENTIQUE au scanner
            sl_dist = atr * sl_mult
            tp_dist = atr * tp_mult

            if direction == 'BUY':
                sl_price = entry_price - sl_dist
                tp_price = entry_price + tp_dist
            else:
                sl_price = entry_price + sl_dist
                tp_price = entry_price - tp_dist

            entry_time = df.index[i]

            # Simuler le trade (chercher SL ou TP dans les barres suivantes)
            result = None
            exit_price = None
            exit_time = None
            bars_held = 0

            for j in range(i+1, min(i+100, len(df))):  # Max 100 barres
                future_bar = df.iloc[j]
                bars_held = j - i

                if direction == 'BUY':
                    # Check SL hit (low touches SL)
                    if future_bar['low'] <= sl_price:
                        result = 'LOSS'
                        exit_price = sl_price
                        exit_time = df.index[j]
                        break
                    # Check TP hit (high touches TP)
                    if future_bar['high'] >= tp_price:
                        result = 'WIN'
                        exit_price = tp_price
                        exit_time = df.index[j]
                        break
                else:  # SELL
                    # Check SL hit (high touches SL)
                    if future_bar['high'] >= sl_price:
                        result = 'LOSS'
                        exit_price = sl_price
                        exit_time = df.index[j]
                        break
                    # Check TP hit (low touches TP)
                    if future_bar['low'] <= tp_price:
                        result = 'WIN'
                        exit_price = tp_price
                        exit_time = df.index[j]
                        break

            # Si pas de resultat, trade encore ouvert (on l'ignore pour ce backtest)
            if result is None:
                continue

            # Calcul du P&L
            pip_value = 10.0  # $10 par pip pour 1 lot standard

            if direction == 'BUY':
                pips = (exit_price - entry_price) / PIP_VALUES.get(pair, 0.0001)
            else:
                pips = (entry_price - exit_price) / PIP_VALUES.get(pair, 0.0001)

            pnl = pips * pip_value

            # Update equity
            equity += pnl

            # Track drawdown
            if equity > peak_equity:
                peak_equity = equity
            drawdown = (peak_equity - equity) / peak_equity * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            trades.append({
                'entry_time': entry_time,
                'exit_time': exit_time,
                'direction': direction,
                'entry': entry_price,
                'sl': sl_price,
                'tp': tp_price,
                'exit': exit_price,
                'result': result,
                'pips': pips,
                'pnl': pnl,
                'equity': equity,
                'score': score,
                'bars_held': bars_held,
                'details': details
            })

    # Calculer les statistiques
    if not trades:
        return {
            'pair': pair,
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'wr': 0,
            'pf': 0,
            'total_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_dd': 0,
            'final_equity': CAPITAL,
            'roi': 0,
            'avg_bars': 0,
            'trades_list': []
        }

    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']

    total_win = sum(t['pnl'] for t in wins)
    total_loss = abs(sum(t['pnl'] for t in losses))

    pf = total_win / total_loss if total_loss > 0 else float('inf')
    wr = len(wins) / len(trades) * 100 if trades else 0

    avg_win = total_win / len(wins) if wins else 0
    avg_loss = total_loss / len(losses) if losses else 0

    avg_bars = sum(t['bars_held'] for t in trades) / len(trades) if trades else 0

    return {
        'pair': pair,
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'wr': wr,
        'pf': pf,
        'total_pnl': sum(t['pnl'] for t in trades),
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'max_dd': max_drawdown,
        'final_equity': equity,
        'roi': (equity - CAPITAL) / CAPITAL * 100,
        'avg_bars': avg_bars,
        'trades_list': trades
    }


def generate_detailed_report(results: Dict[str, dict]):
    """Genere le rapport detaille."""

    print("\n")
    print("=" * 90)
    print("       RAPPORT DE BACKTEST DETAILLE - STRATEGIE OPTIMIZED CROSS")
    print("=" * 90)
    print(f"\nDate du rapport: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Periode: Dernier mois (30 jours)")
    print(f"Timeframe: H1 (1 heure)")
    print(f"Capital initial: ${CAPITAL:,.2f}")
    print(f"Taille position: {LOT_SIZE} lot ({UNITS:,.0f} unites)")
    print(f"Strategie: EMA Crossover (8/21/50) avec parametres optimises")
    print("=" * 90)

    # Resume global
    total_trades = sum(r['trades'] for r in results.values())
    total_wins = sum(r['wins'] for r in results.values())
    total_losses = sum(r['losses'] for r in results.values())
    total_pnl = sum(r['total_pnl'] for r in results.values())
    final_equity = CAPITAL + total_pnl

    print("\n" + "-" * 90)
    print("RESUME GLOBAL")
    print("-" * 90)
    print(f"Total trades:      {total_trades}")
    print(f"Trades gagnants:   {total_wins} ({total_wins/total_trades*100:.1f}%)" if total_trades > 0 else "")
    print(f"Trades perdants:   {total_losses} ({total_losses/total_trades*100:.1f}%)" if total_trades > 0 else "")
    print(f"P&L Total:         ${total_pnl:+,.2f}")
    print(f"Capital final:     ${final_equity:,.2f}")
    print(f"ROI:               {(final_equity - CAPITAL) / CAPITAL * 100:+.2f}%")

    # Tableau par paire
    print("\n" + "-" * 90)
    print("PERFORMANCE PAR PAIRE")
    print("-" * 90)
    print(f"{'Paire':<10} {'Trades':>8} {'Wins':>6} {'Losses':>8} {'WR%':>8} {'PF':>8} {'P&L':>14} {'MaxDD%':>8} {'ROI%':>8}")
    print("-" * 90)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['total_pnl'], reverse=True)

    for pair, r in sorted_results:
        status = "[OK]" if r['total_pnl'] > 0 else "[XX]"
        print(f"{pair:<10} {r['trades']:>8} {r['wins']:>6} {r['losses']:>8} "
              f"{r['wr']:>7.1f}% {r['pf']:>7.2f} ${r['total_pnl']:>+12,.2f} "
              f"{r['max_dd']:>7.1f}% {r['roi']:>+7.2f}% {status}")

    print("-" * 90)
    global_wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    max_dd = max(r['max_dd'] for r in results.values()) if results else 0
    global_roi = (final_equity - CAPITAL) / CAPITAL * 100
    print(f"{'TOTAL':<10} {total_trades:>8} {total_wins:>6} {total_losses:>8} "
          f"{global_wr:>7.1f}% {'---':>8} ${total_pnl:>+12,.2f} "
          f"{max_dd:>7.1f}% {global_roi:>+7.2f}%")

    # Detail par paire
    for pair, r in sorted_results:
        if r['trades'] == 0:
            continue

        config = OPTIMAL_CONFIGS[pair]

        print("\n" + "=" * 90)
        print(f"DETAIL: {pair}")
        print("=" * 90)
        print(f"Configuration: R:R={config['rr']}, ADX>={config['adx']}, RSI={config['rsi']}, Score>={config['score']}")
        print(f"Backtest PF attendu: {config['pf']}")
        print("-" * 90)

        print(f"\nStatistiques:")
        print(f"  - Nombre de trades:    {r['trades']}")
        print(f"  - Trades gagnants:     {r['wins']} ({r['wr']:.1f}%)")
        print(f"  - Trades perdants:     {r['losses']}")
        print(f"  - Profit Factor:       {r['pf']:.2f}")
        print(f"  - Gain moyen:          ${r['avg_win']:+,.2f}")
        print(f"  - Perte moyenne:       ${r['avg_loss']:,.2f}")
        print(f"  - Duree moyenne:       {r['avg_bars']:.1f} barres ({r['avg_bars']:.1f} heures)")
        print(f"  - Max Drawdown:        {r['max_dd']:.1f}%")
        print(f"  - P&L Total:           ${r['total_pnl']:+,.2f}")
        print(f"  - ROI:                 {r['roi']:+.2f}%")

        # Liste des trades
        print(f"\nHistorique des trades:")
        print("-" * 90)
        print(f"{'#':>3} {'Date/Heure':<18} {'Dir':>5} {'Entry':>10} {'Exit':>10} "
              f"{'Pips':>8} {'P&L':>12} {'Res':>6} {'Score':>5} {'Bars':>5}")
        print("-" * 90)

        for i, t in enumerate(r['trades_list'], 1):
            entry_str = t['entry_time'].strftime('%Y-%m-%d %H:%M') if hasattr(t['entry_time'], 'strftime') else str(t['entry_time'])[:16]
            print(f"{i:>3} {entry_str:<18} {t['direction']:>5} {t['entry']:>10.5f} {t['exit']:>10.5f} "
                  f"{t['pips']:>+8.1f} ${t['pnl']:>+10,.2f} {t['result']:>6} {t['score']:>5} {t['bars_held']:>5}")

        print("-" * 90)

    # Analyse et recommandations
    print("\n" + "=" * 90)
    print("ANALYSE ET RECOMMANDATIONS")
    print("=" * 90)

    profitable_pairs = [p for p, r in sorted_results if r['total_pnl'] > 0]
    losing_pairs = [p for p, r in sorted_results if r['total_pnl'] <= 0]

    print(f"\nPaires profitables: {', '.join(profitable_pairs) if profitable_pairs else 'Aucune'}")
    print(f"Paires en perte: {', '.join(losing_pairs) if losing_pairs else 'Aucune'}")

    # Meilleur et pire trade
    all_trades = []
    for pair, r in results.items():
        for t in r['trades_list']:
            t['pair'] = pair
            all_trades.append(t)

    if all_trades:
        best_trade = max(all_trades, key=lambda x: x['pnl'])
        worst_trade = min(all_trades, key=lambda x: x['pnl'])

        print(f"\nMeilleur trade: {best_trade['pair']} {best_trade['direction']} "
              f"le {str(best_trade['entry_time'])[:16]} -> ${best_trade['pnl']:+,.2f}")
        print(f"Pire trade:     {worst_trade['pair']} {worst_trade['direction']} "
              f"le {str(worst_trade['entry_time'])[:16]} -> ${worst_trade['pnl']:+,.2f}")

    # Score final
    if total_pnl > 1000:
        grade = "A - EXCELLENT"
    elif total_pnl > 500:
        grade = "B - TRES BON"
    elif total_pnl > 0:
        grade = "C - PROFITABLE"
    elif total_pnl > -500:
        grade = "D - FAIBLE"
    else:
        grade = "F - A REVOIR"

    print(f"\n{'='*90}")
    print(f"SCORE GLOBAL: {grade}")
    print(f"ROI sur 1 mois: {global_roi:+.2f}%")
    print(f"Capital: ${CAPITAL:,.2f} -> ${final_equity:,.2f} ({'+' if total_pnl >= 0 else ''}{total_pnl:,.2f})")
    print(f"{'='*90}")


def main():
    print("=" * 90)
    print("   BACKTEST DERNIER MOIS - STRATEGIE OPTIMIZED CROSS")
    print("=" * 90)
    print(f"\nCapital: ${CAPITAL:,.2f}")
    print(f"Taille: {LOT_SIZE} lot ({UNITS:,.0f} unites)")
    print(f"Timeframe: H1")
    print(f"Paires: {', '.join(OPTIMAL_CONFIGS.keys())}")
    print("\nTelechargement des donnees et execution du backtest...")

    results = {}

    for pair in OPTIMAL_CONFIGS.keys():
        config = OPTIMAL_CONFIGS[pair]
        print(f"\n  {pair}...", end=" ", flush=True)

        df = fetch_data(pair, days=35)
        if df is None or len(df) < 100:
            print("SKIP (donnees insuffisantes)")
            results[pair] = {
                'pair': pair, 'trades': 0, 'wins': 0, 'losses': 0,
                'wr': 0, 'pf': 0, 'total_pnl': 0, 'avg_win': 0,
                'avg_loss': 0, 'max_dd': 0, 'final_equity': CAPITAL,
                'roi': 0, 'avg_bars': 0, 'trades_list': []
            }
            continue

        df = calculate_indicators(df, config)
        result = run_backtest(df, pair, config)
        results[pair] = result

        status = "OK" if result['total_pnl'] > 0 else "XX"
        print(f"[{status}] {result['trades']} trades, WR={result['wr']:.1f}%, "
              f"PF={result['pf']:.2f}, P&L=${result['total_pnl']:+,.2f}")

    # Generer le rapport
    generate_detailed_report(results)

    return results


if __name__ == "__main__":
    main()
