#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimiseur de Parametres par Paire V2
=====================================
Version optimisee avec cache et parallelisation.
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging
import warnings
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.ERROR)
warnings.filterwarnings('ignore')

ALL_PAIRS = [
    'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD',
    'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD',
    'AUDJPY', 'CADJPY', 'CHFJPY',
    'AUDCAD', 'AUDNZD', 'NZDJPY'
]

# Configurations a tester (reduit pour performance)
RR_OPTIONS = [1.2, 1.5, 1.8, 2.0, 2.5]
ADX_OPTIONS = [12, 15, 20, 25]
RSI_OPTIONS = [(25, 75), (30, 70), (35, 65)]
SCORE_OPTIONS = [4, 5, 6]


def fetch_data_safe(pair: str, period: str = "2y") -> Optional[pd.DataFrame]:
    """Telecharge les donnees avec gestion d'erreur robuste."""
    symbol = f"{pair}=X"

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval="1h")

        if df.empty or len(df) < 500:
            # Essayer avec un autre format
            df = yf.download(symbol, period=period, interval="1h", progress=False)

        if df.empty or len(df) < 500:
            return None

        # Normaliser les colonnes
        df.columns = [c.replace(' ', '_') for c in df.columns]
        if 'Adj_Close' in df.columns:
            df['Close'] = df['Adj_Close']

        return df

    except Exception as e:
        return None


def calculate_indicators_vectorized(df: pd.DataFrame, adx_min: int, rsi_range: tuple) -> pd.DataFrame:
    """Calcule tous les indicateurs de maniere vectorisee."""
    df = df.copy()

    # EMAs
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


def backtest_fast(df: pd.DataFrame, pair: str, rr: float, min_score: int) -> dict:
    """Backtest rapide vectorise."""
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
    position = None
    pip = 0.01 if 'JPY' in pair else 0.0001

    df_valid = df.iloc[60:].dropna(subset=['atr', 'adx'])

    for i in range(len(df_valid)):
        row = df_valid.iloc[i]

        if position:
            if position['dir'] == 'BUY':
                if row['Low'] <= position['sl']:
                    pnl = (position['sl'] - position['entry']) / pip * position['size']
                    balance += pnl
                    trades.append({'result': 'LOSS', 'pnl': pnl})
                    position = None
                elif row['High'] >= position['tp']:
                    pnl = (position['tp'] - position['entry']) / pip * position['size']
                    balance += pnl
                    trades.append({'result': 'WIN', 'pnl': pnl})
                    position = None
            else:
                if row['High'] >= position['sl']:
                    pnl = (position['entry'] - position['sl']) / pip * position['size']
                    balance += pnl
                    trades.append({'result': 'LOSS', 'pnl': pnl})
                    position = None
                elif row['Low'] <= position['tp']:
                    pnl = (position['entry'] - position['tp']) / pip * position['size']
                    balance += pnl
                    trades.append({'result': 'WIN', 'pnl': pnl})
                    position = None

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
        return {'trades': 0, 'wr': 0, 'pf': 0, 'pnl': 0}

    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']

    total_win = sum(t['pnl'] for t in wins) if wins else 0
    total_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0.0001

    return {
        'trades': len(trades),
        'wr': len(wins) / len(trades) * 100 if trades else 0,
        'pf': total_win / total_loss if total_loss > 0 else 0,
        'pnl': balance - 10000
    }


def optimize_pair(pair: str) -> Tuple[str, Optional[dict], List[dict]]:
    """Optimise une paire avec toutes les configurations."""
    print(f"[{pair}] Telechargement...", end=" ", flush=True)

    df_raw = fetch_data_safe(pair)
    if df_raw is None or len(df_raw) < 500:
        print("SKIP (donnees insuffisantes)")
        return pair, None, []

    print(f"OK ({len(df_raw)} barres) - Optimisation...", end=" ", flush=True)

    all_results = []

    for adx in ADX_OPTIONS:
        for rsi in RSI_OPTIONS:
            # Calculer les indicateurs une seule fois pour chaque combo ADX/RSI
            df = calculate_indicators_vectorized(df_raw, adx, rsi)

            for rr in RR_OPTIONS:
                for score in SCORE_OPTIONS:
                    result = backtest_fast(df.copy(), pair, rr, score)
                    result['config'] = {
                        'rr': rr,
                        'adx': adx,
                        'rsi': rsi,
                        'score': score
                    }
                    all_results.append(result)

    # Filtrer et trier
    valid_results = [r for r in all_results if r['trades'] >= 20]
    if not valid_results:
        valid_results = [r for r in all_results if r['trades'] >= 10]

    if not valid_results:
        print("AUCUN TRADE")
        return pair, None, all_results

    # Meilleur: profitable avec le plus de trades
    profitable = [r for r in valid_results if r['pf'] >= 1.0]

    if profitable:
        best = max(profitable, key=lambda x: (x['pf'], x['trades']))
        status = "PROFITABLE"
    else:
        best = max(valid_results, key=lambda x: x['pf'])
        status = f"NON-PROFITABLE (PF={best['pf']:.2f})"

    print(status)
    cfg = best['config']
    print(f"        Config: R:R={cfg['rr']}, ADX>={cfg['adx']}, RSI={cfg['rsi']}, Score>={cfg['score']}")
    print(f"        Stats: {best['trades']} trades, WR={best['wr']:.1f}%, PF={best['pf']:.2f}, PnL=${best['pnl']:.0f}")

    return pair, best, all_results


def main():
    print("\n" + "=" * 80)
    print("   OPTIMISATION DES PARAMETRES PAR PAIRE V2")
    print("=" * 80)
    print(f"\nConfigurations testees:")
    print(f"  - R:R: {RR_OPTIONS}")
    print(f"  - ADX: {ADX_OPTIONS}")
    print(f"  - RSI: {RSI_OPTIONS}")
    print(f"  - Score min: {SCORE_OPTIONS}")
    total_configs = len(RR_OPTIONS) * len(ADX_OPTIONS) * len(RSI_OPTIONS) * len(SCORE_OPTIONS)
    print(f"  - Total: {total_configs} configurations par paire")
    print("-" * 80)

    results = {}

    for pair in ALL_PAIRS:
        pair_name, best, _ = optimize_pair(pair)
        if best:
            results[pair_name] = best

    # Rapport final
    print("\n" + "=" * 80)
    print("                    RAPPORT D'OPTIMISATION FINAL")
    print("=" * 80)

    # Paires profitables
    profitable_pairs = [(p, r) for p, r in results.items() if r['pf'] >= 1.0]
    profitable_pairs.sort(key=lambda x: x[1]['pf'], reverse=True)

    print(f"\n{'='*80}")
    print(f"PAIRES PROFITABLES ({len(profitable_pairs)}/{len(ALL_PAIRS)})")
    print(f"{'='*80}")

    if profitable_pairs:
        print(f"\n{'Paire':<10} {'Trades':>8} {'WR%':>8} {'PF':>8} {'PnL':>12} {'Config Optimale'}")
        print("-" * 80)

        for pair, r in profitable_pairs:
            cfg = r['config']
            config_str = f"R:R={cfg['rr']}, ADX>{cfg['adx']}, RSI={cfg['rsi'][0]}-{cfg['rsi'][1]}, S>{cfg['score']}"
            print(f"{pair:<10} {r['trades']:>8} {r['wr']:>7.1f}% {r['pf']:>7.2f} ${r['pnl']:>10,.0f}  {config_str}")
    else:
        print("\n  Aucune paire profitable trouvee")

    # Paires non-profitables
    non_profitable = [(p, r) for p, r in results.items() if r['pf'] < 1.0]
    non_profitable.sort(key=lambda x: x[1]['pf'], reverse=True)

    print(f"\n{'='*80}")
    print(f"PAIRES A AMELIORER ({len(non_profitable)}/{len(ALL_PAIRS)})")
    print(f"{'='*80}")

    if non_profitable:
        print(f"\n{'Paire':<10} {'Trades':>8} {'WR%':>8} {'PF':>8} {'PnL':>12} {'Meilleure Config'}")
        print("-" * 80)

        for pair, r in non_profitable:
            cfg = r['config']
            config_str = f"R:R={cfg['rr']}, ADX>{cfg['adx']}, RSI={cfg['rsi'][0]}-{cfg['rsi'][1]}, S>{cfg['score']}"
            print(f"{pair:<10} {r['trades']:>8} {r['wr']:>7.1f}% {r['pf']:>7.2f} ${r['pnl']:>10,.0f}  {config_str}")

    # Code Python
    print(f"\n{'='*80}")
    print("CODE PYTHON - CONFIGURATIONS OPTIMALES")
    print(f"{'='*80}")

    print("\nOPTIMAL_PAIR_CONFIGS = {")
    for pair in ALL_PAIRS:
        if pair in results:
            r = results[pair]
            cfg = r['config']
            status = "PF>1" if r['pf'] >= 1.0 else f"PF={r['pf']:.2f}"
            print(f"    '{pair}': {{'rr': {cfg['rr']}, 'adx': {cfg['adx']}, 'rsi': {cfg['rsi']}, 'score': {cfg['score']}}},  # {status}")
        else:
            print(f"    # '{pair}': SKIP - donnees indisponibles")
    print("}")

    # Export CSV
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    config_data = []
    for pair, r in results.items():
        cfg = r['config']
        config_data.append({
            'pair': pair,
            'rr': cfg['rr'],
            'adx_min': cfg['adx'],
            'rsi_low': cfg['rsi'][0],
            'rsi_high': cfg['rsi'][1],
            'min_score': cfg['score'],
            'trades': r['trades'],
            'win_rate': r['wr'],
            'profit_factor': r['pf'],
            'pnl': r['pnl'],
            'profitable': 'YES' if r['pf'] >= 1.0 else 'NO'
        })

    if config_data:
        df_config = pd.DataFrame(config_data)
        filename = f"optimal_configs_{ts}.csv"
        df_config.to_csv(filename, index=False)
        print(f"\n[EXPORT] {filename}")

    # Statistiques
    if results:
        total_pnl = sum(r['pnl'] for r in results.values())
        avg_pf = np.mean([r['pf'] for r in results.values()])
        print(f"\n{'='*80}")
        print("STATISTIQUES GLOBALES")
        print(f"{'='*80}")
        print(f"  Paires analysees: {len(results)}/{len(ALL_PAIRS)}")
        print(f"  Paires profitables: {len(profitable_pairs)}")
        print(f"  PnL total: ${total_pnl:,.0f}")
        print(f"  PF moyen: {avg_pf:.2f}")

    print("\n" + "=" * 80)
    print("FIN DE L'OPTIMISATION")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
