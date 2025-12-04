#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backtest sur Periodes Critiques
===============================
Teste la strategie EMA Crossover optimisee sur des periodes de marche significatives.

NOTE: yfinance limite H1 aux 730 derniers jours.
      Pour les periodes anciennes, on utilise le timeframe Daily.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import yfinance as yf

warnings.filterwarnings('ignore')

# Periodes critiques a tester
# Note: Les periodes > 730 jours utilisent Daily au lieu de H1
CRITICAL_PERIODS = {
    'COVID_CRASH': {
        'name': 'COVID Crash & Recovery',
        'start': '2020-01-01',
        'end': '2020-06-30',
        'description': 'Volatilite extreme, spreads elargis, mouvements de 500+ pips/jour',
        'interval': '1d'  # Daily car > 730 jours
    },
    'JPY_CRISIS_2022': {
        'name': 'JPY Crisis & Fed Pivot',
        'start': '2022-10-01',
        'end': '2023-03-31',
        'description': 'USD/JPY de 150 a 127, forte tendance, intervention BOJ',
        'interval': '1d'  # Daily car > 730 jours
    },
    'RANGING_2019': {
        'name': 'Marche Ranging 2019',
        'start': '2019-01-01',
        'end': '2019-12-31',
        'description': 'Marche calme, oscillations sans tendance claire',
        'interval': '1d'  # Daily car > 730 jours
    },
    'POST_COVID_2021': {
        'name': 'Recovery 2021',
        'start': '2021-01-01',
        'end': '2021-12-31',
        'description': 'Reprise post-COVID, tendances claires',
        'interval': '1d'  # Daily car > 730 jours
    },
    'RECENT_2024': {
        'name': 'Recent 2024',
        'start': '2024-01-01',
        'end': '2024-11-30',
        'description': 'Conditions de marche actuelles',
        'interval': '1h'  # H1 possible car < 730 jours
    }
}

# Configurations optimisees - 6 paires ROBUSTES selectionnees apres backtest periodes critiques
# Criteres: PnL positif sur toutes periodes, PF >= 1.0 dans 3+ periodes, faible drawdown
OPTIMAL_CONFIGS = {
    # TOP PERFORMERS (PnL positif sur toutes periodes)
    'EURCAD': {'rr': 2.5, 'adx': 15, 'rsi': (35, 65), 'score': 6, 'pf': 1.27},  # Best PF, lowest DD
    'EURJPY': {'rr': 1.8, 'adx': 20, 'rsi': (25, 75), 'score': 5, 'pf': 1.06},  # High volume, consistent
    'GBPJPY': {'rr': 1.2, 'adx': 25, 'rsi': (30, 70), 'score': 6, 'pf': 1.10},  # Good WR 48%
    'CHFJPY': {'rr': 1.5, 'adx': 25, 'rsi': (25, 75), 'score': 4, 'pf': 1.06},  # Most trades
    'CADJPY': {'rr': 2.5, 'adx': 25, 'rsi': (35, 65), 'score': 6, 'pf': 1.03},  # Stable
    'GBPAUD': {'rr': 2.5, 'adx': 12, 'rsi': (35, 65), 'score': 6, 'pf': 1.00},  # Robust (PF>=1 in 3+ periods)
}


def fetch_period_data(pair: str, start: str, end: str, interval: str = "1d") -> Optional[pd.DataFrame]:
    """Telecharge les donnees pour une periode specifique."""
    symbol = f"{pair}=X"

    try:
        df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)

        min_bars = 50 if interval == "1d" else 100

        if df.empty or len(df) < min_bars:
            return None

        # Normaliser les colonnes
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [c.replace(' ', '_') for c in df.columns]
        if 'Adj_Close' in df.columns:
            df['Close'] = df['Adj_Close']

        return df

    except Exception as e:
        return None


def calculate_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calcule tous les indicateurs techniques."""
    df = df.copy()
    adx_min = config['adx']
    rsi_range = config['rsi']

    # EMAs (8/21/50)
    df['ema_fast'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['ema_slow'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['ema_trend'] = df['Close'].ewm(span=50, adjust=False).mean()

    # EMA cross
    df['ema_fast_prev'] = df['ema_fast'].shift(1)
    df['ema_slow_prev'] = df['ema_slow'].shift(1)
    df['ema_cross_up'] = (df['ema_fast_prev'] <= df['ema_slow_prev']) & (df['ema_fast'] > df['ema_slow'])
    df['ema_cross_down'] = (df['ema_fast_prev'] >= df['ema_slow_prev']) & (df['ema_fast'] < df['ema_slow'])

    # EMA slope
    df['ema_slope'] = (df['ema_trend'] - df['ema_trend'].shift(10)) / df['ema_trend'].shift(10) * 100

    # Trend
    df['trend_up'] = (df['Close'] > df['ema_trend']) & (df['ema_slope'] > 0)
    df['trend_down'] = (df['Close'] < df['ema_trend']) & (df['ema_slope'] < 0)

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 0.0001)))
    df['rsi_ok'] = (df['rsi'] > rsi_range[0]) & (df['rsi'] < rsi_range[1])

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    df['macd_bull'] = df['macd_hist'] > 0
    df['macd_bear'] = df['macd_hist'] < 0

    # Momentum (ROC)
    df['roc'] = df['Close'].pct_change(5) * 100
    df['mom_bull'] = df['roc'] > 0
    df['mom_bear'] = df['roc'] < 0

    # ATR
    tr = pd.concat([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift()),
        abs(df['Low'] - df['Close'].shift())
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()

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
    df['adx_ok'] = df['adx'] >= adx_min

    return df


def run_backtest(df: pd.DataFrame, pair: str, config: dict) -> dict:
    """Exécute le backtest sur les données."""
    rr = config['rr']
    min_score = config['score']
    sl_mult = 1.5
    tp_mult = 1.5 * rr

    # Calcul des scores
    df['buy_score'] = (
        df['ema_cross_up'].astype(int) * 2 +
        df['trend_up'].astype(int) * 2 +
        df['rsi_ok'].astype(int) +
        df['adx_ok'].astype(int) +
        df['macd_bull'].astype(int) +
        df['mom_bull'].astype(int)
    )

    df['sell_score'] = (
        df['ema_cross_down'].astype(int) * 2 +
        df['trend_down'].astype(int) * 2 +
        df['rsi_ok'].astype(int) +
        df['adx_ok'].astype(int) +
        df['macd_bear'].astype(int) +
        df['mom_bear'].astype(int)
    )

    # Signaux
    df['signal'] = 0
    df.loc[df['buy_score'] >= min_score, 'signal'] = 1
    df.loc[df['sell_score'] >= min_score, 'signal'] = -1

    # SL/TP
    df['sl_dist'] = df['atr'] * sl_mult
    df['tp_dist'] = df['atr'] * tp_mult

    # Simulation
    trades = []
    balance = 10000
    initial_balance = 10000
    position = None
    pip = 0.01 if 'JPY' in pair else 0.0001
    max_balance = balance
    max_drawdown = 0

    df_valid = df.iloc[60:].dropna(subset=['atr', 'adx'])

    for i in range(len(df_valid)):
        row = df_valid.iloc[i]

        if position:
            if position['dir'] == 'BUY':
                if row['Low'] <= position['sl']:
                    pnl = (position['sl'] - position['entry']) / pip * position['size']
                    balance += pnl
                    trades.append({
                        'result': 'LOSS',
                        'pnl': pnl,
                        'dir': 'BUY',
                        'entry': position['entry'],
                        'exit': position['sl']
                    })
                    position = None
                elif row['High'] >= position['tp']:
                    pnl = (position['tp'] - position['entry']) / pip * position['size']
                    balance += pnl
                    trades.append({
                        'result': 'WIN',
                        'pnl': pnl,
                        'dir': 'BUY',
                        'entry': position['entry'],
                        'exit': position['tp']
                    })
                    position = None
            else:  # SELL
                if row['High'] >= position['sl']:
                    pnl = (position['entry'] - position['sl']) / pip * position['size']
                    balance += pnl
                    trades.append({
                        'result': 'LOSS',
                        'pnl': pnl,
                        'dir': 'SELL',
                        'entry': position['entry'],
                        'exit': position['sl']
                    })
                    position = None
                elif row['Low'] <= position['tp']:
                    pnl = (position['entry'] - position['tp']) / pip * position['size']
                    balance += pnl
                    trades.append({
                        'result': 'WIN',
                        'pnl': pnl,
                        'dir': 'SELL',
                        'entry': position['entry'],
                        'exit': position['tp']
                    })
                    position = None

        # Track drawdown
        if balance > max_balance:
            max_balance = balance
        current_dd = (max_balance - balance) / max_balance * 100
        if current_dd > max_drawdown:
            max_drawdown = current_dd

        if position is None and row['signal'] != 0:
            risk = balance * 0.01
            sl_dist = row['sl_dist']
            if sl_dist > 0:
                size = risk / (sl_dist / pip)
                entry = row['Close']
                if row['signal'] == 1:
                    position = {
                        'dir': 'BUY',
                        'entry': entry,
                        'sl': entry - sl_dist,
                        'tp': entry + row['tp_dist'],
                        'size': size
                    }
                else:
                    position = {
                        'dir': 'SELL',
                        'entry': entry,
                        'sl': entry + sl_dist,
                        'tp': entry - row['tp_dist'],
                        'size': size
                    }

    if not trades:
        return {
            'trades': 0, 'wins': 0, 'losses': 0,
            'wr': 0, 'pf': 0, 'pnl': 0, 'pnl_pct': 0,
            'max_dd': 0, 'buy_trades': 0, 'sell_trades': 0,
            'buy_wr': 0, 'sell_wr': 0
        }

    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']
    buy_trades = [t for t in trades if t['dir'] == 'BUY']
    sell_trades = [t for t in trades if t['dir'] == 'SELL']
    buy_wins = [t for t in buy_trades if t['result'] == 'WIN']
    sell_wins = [t for t in sell_trades if t['result'] == 'WIN']

    total_win = sum(t['pnl'] for t in wins) if wins else 0
    total_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.0001

    return {
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'wr': len(wins) / len(trades) * 100 if trades else 0,
        'pf': total_win / total_loss if total_loss > 0 else 0,
        'pnl': balance - initial_balance,
        'pnl_pct': (balance - initial_balance) / initial_balance * 100,
        'max_dd': max_drawdown,
        'buy_trades': len(buy_trades),
        'sell_trades': len(sell_trades),
        'buy_wr': len(buy_wins) / len(buy_trades) * 100 if buy_trades else 0,
        'sell_wr': len(sell_wins) / len(sell_trades) * 100 if sell_trades else 0,
        'avg_win': total_win / len(wins) if wins else 0,
        'avg_loss': total_loss / len(losses) if losses else 0
    }


def test_period(period_key: str, period_info: dict) -> dict:
    """Teste toutes les paires sur une periode."""
    interval = period_info.get('interval', '1d')

    print(f"\n{'='*70}")
    print(f"PERIODE: {period_info['name']} [{interval.upper()}]")
    print(f"Du {period_info['start']} au {period_info['end']}")
    print(f"Description: {period_info['description']}")
    print('='*70)

    results = {}
    total_trades = 0
    total_wins = 0
    total_pnl = 0

    min_bars = 50 if interval == "1d" else 100

    for pair, config in OPTIMAL_CONFIGS.items():
        print(f"  {pair}...", end=" ", flush=True)

        df = fetch_period_data(pair, period_info['start'], period_info['end'], interval)
        if df is None or len(df) < min_bars:
            print("SKIP (donnees insuffisantes)")
            continue

        df = calculate_indicators(df, config)
        result = run_backtest(df, pair, config)

        results[pair] = result
        total_trades += result['trades']
        total_wins += result['wins']
        total_pnl += result['pnl']

        status = "OK" if result['pf'] >= 1.0 else "XX"
        print(f"[{status}] {result['trades']} trades, WR={result['wr']:.1f}%, PF={result['pf']:.2f}, PnL={result['pnl']:+.2f}$")

    # Résumé de la période
    period_summary = {
        'name': period_info['name'],
        'start': period_info['start'],
        'end': period_info['end'],
        'pairs_tested': len(results),
        'total_trades': total_trades,
        'total_wins': total_wins,
        'total_wr': total_wins / total_trades * 100 if total_trades > 0 else 0,
        'total_pnl': total_pnl,
        'profitable_pairs': sum(1 for r in results.values() if r['pf'] >= 1.0),
        'results_by_pair': results
    }

    return period_summary


def generate_report(all_results: dict):
    """Genere le rapport detaille."""
    print("\n")
    print("=" * 80)
    print("              RAPPORT DETAILLE - BACKTEST PERIODES CRITIQUES")
    print("=" * 80)
    print(f"Date du rapport: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Strategie: EMA Crossover (8/21/50) avec parametres optimises par paire")
    print(f"Timeframe: Daily (periodes anciennes) / H1 (periodes recentes)")
    print(f"Risk par trade: 1%")
    print("=" * 80)

    # Tableau comparatif des periodes
    print("\n" + "-" * 80)
    print("COMPARAISON PAR PERIODE")
    print("-" * 80)
    print(f"{'Periode':<30} {'Trades':>8} {'WR%':>8} {'PnL':>12} {'Pairs OK':>10}")
    print("-" * 80)

    total_all_trades = 0
    total_all_wins = 0
    total_all_pnl = 0

    for period_key, summary in all_results.items():
        print(f"{summary['name']:<30} {summary['total_trades']:>8} "
              f"{summary['total_wr']:>7.1f}% {summary['total_pnl']:>+11.2f}$ "
              f"{summary['profitable_pairs']:>5}/{summary['pairs_tested']}")
        total_all_trades += summary['total_trades']
        total_all_wins += summary['total_wins']
        total_all_pnl += summary['total_pnl']

    print("-" * 80)
    total_wr = total_all_wins / total_all_trades * 100 if total_all_trades > 0 else 0
    print(f"{'TOTAL':<30} {total_all_trades:>8} {total_wr:>7.1f}% {total_all_pnl:>+11.2f}$")

    # Detail par paire (agrege sur toutes les periodes)
    print("\n" + "-" * 80)
    print("PERFORMANCE PAR PAIRE (TOUTES PERIODES CONFONDUES)")
    print("-" * 80)
    print(f"{'Paire':<10} {'Trades':>8} {'Wins':>6} {'WR%':>8} {'PF':>8} {'PnL':>12} {'MaxDD%':>8}")
    print("-" * 80)

    pair_aggregates = {}
    for pair in OPTIMAL_CONFIGS.keys():
        pair_aggregates[pair] = {
            'trades': 0, 'wins': 0, 'pnl': 0, 'max_dd': 0,
            'total_win_amount': 0, 'total_loss_amount': 0
        }

    for period_key, summary in all_results.items():
        for pair, result in summary['results_by_pair'].items():
            pair_aggregates[pair]['trades'] += result['trades']
            pair_aggregates[pair]['wins'] += result['wins']
            pair_aggregates[pair]['pnl'] += result['pnl']
            if result['max_dd'] > pair_aggregates[pair]['max_dd']:
                pair_aggregates[pair]['max_dd'] = result['max_dd']
            pair_aggregates[pair]['total_win_amount'] += result.get('avg_win', 0) * result['wins']
            pair_aggregates[pair]['total_loss_amount'] += result.get('avg_loss', 0) * result['losses']

    # Trier par PnL
    sorted_pairs = sorted(pair_aggregates.items(), key=lambda x: x[1]['pnl'], reverse=True)

    for pair, agg in sorted_pairs:
        wr = agg['wins'] / agg['trades'] * 100 if agg['trades'] > 0 else 0
        pf = agg['total_win_amount'] / agg['total_loss_amount'] if agg['total_loss_amount'] > 0 else 0
        print(f"{pair:<10} {agg['trades']:>8} {agg['wins']:>6} {wr:>7.1f}% {pf:>7.2f} "
              f"{agg['pnl']:>+11.2f}$ {agg['max_dd']:>7.1f}%")

    # Analyse par regime de marche
    print("\n" + "-" * 80)
    print("ANALYSE PAR REGIME DE MARCHE")
    print("-" * 80)

    regimes = {
        'HIGH_VOLATILITY': ['COVID_CRASH'],
        'TRENDING': ['JPY_CRISIS_2022', 'POST_COVID_2021'],
        'RANGING': ['RANGING_2019'],
        'RECENT': ['RECENT_2024']
    }

    for regime_name, period_keys in regimes.items():
        regime_trades = 0
        regime_wins = 0
        regime_pnl = 0

        for pk in period_keys:
            if pk in all_results:
                regime_trades += all_results[pk]['total_trades']
                regime_wins += all_results[pk]['total_wins']
                regime_pnl += all_results[pk]['total_pnl']

        regime_wr = regime_wins / regime_trades * 100 if regime_trades > 0 else 0
        status = "[OK] PROFITABLE" if regime_pnl > 0 else "[XX] PERTE"
        print(f"{regime_name:<20}: {regime_trades:>5} trades, WR={regime_wr:.1f}%, PnL={regime_pnl:>+.2f}$ {status}")

    # Recommandations
    print("\n" + "-" * 80)
    print("CONCLUSIONS & RECOMMANDATIONS")
    print("-" * 80)

    # Identifier les meilleures paires
    best_pairs = [p for p, a in sorted_pairs if a['pnl'] > 0][:5]
    worst_pairs = [p for p, a in sorted_pairs if a['pnl'] <= 0]

    print(f"\n[+] Meilleures paires (toutes periodes): {', '.join(best_pairs) if best_pairs else 'Aucune'}")
    if worst_pairs:
        print(f"[-] Paires a eviter/revoir: {', '.join(worst_pairs)}")

    # Verifier la robustesse
    robust_pairs = []
    for pair in OPTIMAL_CONFIGS.keys():
        profitable_periods = 0
        for period_key, summary in all_results.items():
            if pair in summary['results_by_pair']:
                if summary['results_by_pair'][pair]['pf'] >= 1.0:
                    profitable_periods += 1
        if profitable_periods >= 3:  # Profitable dans au moins 3 periodes sur 5
            robust_pairs.append(pair)

    print(f"\n[*] Paires ROBUSTES (PF>=1 dans 3+ periodes): {', '.join(robust_pairs) if robust_pairs else 'Aucune'}")

    # Score global
    global_score = (total_all_pnl / 10000) * 100  # Score base sur ROI
    if global_score > 20:
        grade = "A - EXCELLENT"
    elif global_score > 10:
        grade = "B - BON"
    elif global_score > 0:
        grade = "C - ACCEPTABLE"
    elif global_score > -10:
        grade = "D - FAIBLE"
    else:
        grade = "F - A REVOIR"

    print(f"\n{'='*80}")
    print(f"SCORE GLOBAL: {grade}")
    print(f"ROI Total: {global_score:+.2f}% sur {total_all_trades} trades")
    print(f"{'='*80}")


def main():
    print("=" * 80)
    print("   BACKTEST STRATEGIE EMA CROSSOVER - PERIODES CRITIQUES")
    print("=" * 80)
    print("\nCe test evalue la robustesse de la strategie sur differents regimes de marche:")
    print("- Haute volatilite (COVID 2020) [Daily]")
    print("- Forte tendance (JPY Crisis 2022) [Daily]")
    print("- Marche ranging (2019) [Daily]")
    print("- Recovery (2021) [Daily]")
    print("- Conditions recentes (2024) [H1]")

    all_results = {}

    for period_key, period_info in CRITICAL_PERIODS.items():
        try:
            result = test_period(period_key, period_info)
            all_results[period_key] = result
        except Exception as e:
            print(f"\nErreur sur période {period_key}: {e}")
            continue

    if all_results:
        generate_report(all_results)
    else:
        print("\nAucun résultat à afficher.")


if __name__ == "__main__":
    main()
