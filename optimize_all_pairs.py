#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimiseur de Parametres par Paire
===================================
Teste differentes configurations pour chaque paire et trouve
les parametres optimaux.

Parametres optimises:
- R:R (1.2, 1.5, 1.8, 2.0, 2.5)
- ADX threshold (12, 15, 18, 20, 25)
- RSI range (25-75, 30-70, 35-65)
- Score minimum (4, 5, 6)
"""

import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import logging
import time as time_module
import yfinance as yf
from itertools import product

logging.basicConfig(level=logging.WARNING)

ALL_PAIRS = [
    'EURGBP', 'EURJPY', 'EURCHF', 'EURAUD', 'EURCAD',
    'GBPJPY', 'GBPCHF', 'GBPAUD', 'GBPCAD',
    'AUDJPY', 'CADJPY', 'CHFJPY',
    'AUDCAD', 'AUDNZD', 'NZDJPY'
]

YFINANCE_SYMBOLS = {p: f"{p}=X" for p in ALL_PAIRS}

# Configurations a tester
RR_OPTIONS = [1.2, 1.5, 1.8, 2.0, 2.5]
ADX_OPTIONS = [12, 15, 18, 20, 25]
RSI_OPTIONS = [(25, 75), (30, 70), (35, 65)]
SCORE_OPTIONS = [4, 5, 6]


class ConfigurableStrategy:
    """Strategie avec parametres configurables."""

    def __init__(self, rr=1.8, adx_min=15, rsi_range=(30, 70), min_score=5):
        self.ema_fast = 8
        self.ema_slow = 21
        self.ema_trend = 50
        self.rsi_period = 14
        self.rsi_oversold, self.rsi_overbought = rsi_range
        self.adx_min = adx_min
        self.atr_period = 14
        self.sl_mult = 1.5
        self.tp_mult = 1.5 * rr  # R:R
        self.min_score = min_score
        self.rr = rr

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

        # EMA slope
        df['ema_slope'] = (df['ema_trend'] - df['ema_trend'].shift(10)) / df['ema_trend'].shift(10) * 100

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df['signal'] = 0
        df['sl'] = 0.0
        df['tp'] = 0.0

        for i in range(60, len(df)):
            current = df.iloc[i]
            prev = df.iloc[i-1]

            if pd.isna(current['atr']) or current['atr'] == 0:
                continue
            if pd.isna(current['adx']):
                continue

            # Conditions
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

            # Score
            buy_score = sum([ema_cross_up * 2, trend_up * 2, rsi_ok, adx_ok, macd_bull, mom_bull])
            sell_score = sum([ema_cross_down * 2, trend_down * 2, rsi_ok, adx_ok, macd_bear, mom_bear])

            if buy_score >= self.min_score:
                df.iloc[i, df.columns.get_loc('signal')] = 1
                df.iloc[i, df.columns.get_loc('sl')] = current['Close'] - (current['atr'] * self.sl_mult)
                df.iloc[i, df.columns.get_loc('tp')] = current['Close'] + (current['atr'] * self.tp_mult)
            elif sell_score >= self.min_score:
                df.iloc[i, df.columns.get_loc('signal')] = -1
                df.iloc[i, df.columns.get_loc('sl')] = current['Close'] + (current['atr'] * self.sl_mult)
                df.iloc[i, df.columns.get_loc('tp')] = current['Close'] - (current['atr'] * self.tp_mult)

        return df


def backtest_config(df: pd.DataFrame, pair: str, config: dict) -> dict:
    """Execute un backtest avec une configuration donnee."""
    strategy = ConfigurableStrategy(
        rr=config['rr'],
        adx_min=config['adx'],
        rsi_range=config['rsi'],
        min_score=config['score']
    )

    df = strategy.generate_signals(df)

    trades = []
    balance = 10000
    position = None
    pip = 0.01 if 'JPY' in pair else 0.0001

    for i in range(len(df)):
        row = df.iloc[i]

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

    if not trades:
        return {'trades': 0, 'wr': 0, 'pf': 0, 'pnl': 0, 'config': config}

    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']

    total_win = sum(t['pnl'] for t in wins) if wins else 0
    total_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0

    return {
        'trades': len(trades),
        'wr': len(wins) / len(trades) * 100 if trades else 0,
        'pf': total_win / total_loss if total_loss > 0 else 0,
        'pnl': balance - 10000,
        'config': config
    }


def fetch_data(pair: str, period: str = "2y") -> pd.DataFrame:
    """Telecharge les donnees."""
    try:
        ticker = yf.Ticker(YFINANCE_SYMBOLS.get(pair, f"{pair}=X"))
        df = ticker.history(period=period, interval="1h")
        return df if not df.empty else None
    except:
        return None


def optimize_pair(pair: str, df: pd.DataFrame) -> Tuple[dict, List[dict]]:
    """Trouve la meilleure configuration pour une paire."""
    all_results = []

    # Generer toutes les combinaisons
    configs = []
    for rr in RR_OPTIONS:
        for adx in ADX_OPTIONS:
            for rsi in RSI_OPTIONS:
                for score in SCORE_OPTIONS:
                    configs.append({
                        'rr': rr,
                        'adx': adx,
                        'rsi': rsi,
                        'score': score
                    })

    # Tester chaque config
    for config in configs:
        result = backtest_config(df.copy(), pair, config)
        all_results.append(result)

    # Trier par PF puis par nombre de trades
    valid_results = [r for r in all_results if r['trades'] >= 20]

    if not valid_results:
        valid_results = [r for r in all_results if r['trades'] >= 10]

    if not valid_results:
        return None, all_results

    # Meilleur resultat: PF > 1 avec le plus de trades
    profitable = [r for r in valid_results if r['pf'] >= 1.0]

    if profitable:
        best = max(profitable, key=lambda x: (x['pf'], x['trades']))
    else:
        best = max(valid_results, key=lambda x: x['pf'])

    return best, all_results


def main():
    print("\n" + "=" * 80)
    print("   OPTIMISATION DES PARAMETRES PAR PAIRE")
    print("=" * 80)
    print("\nConfigurations testees:")
    print(f"  - R:R: {RR_OPTIONS}")
    print(f"  - ADX: {ADX_OPTIONS}")
    print(f"  - RSI: {RSI_OPTIONS}")
    print(f"  - Score min: {SCORE_OPTIONS}")
    print(f"  - Total: {len(RR_OPTIONS) * len(ADX_OPTIONS) * len(RSI_OPTIONS) * len(SCORE_OPTIONS)} configurations par paire")
    print("-" * 80)

    results = {}
    optimal_configs = {}

    for pair in ALL_PAIRS:
        print(f"\n[{pair}] Telechargement des donnees...", end=" ", flush=True)
        df = fetch_data(pair)

        if df is None or len(df) < 500:
            print("SKIP (donnees insuffisantes)")
            continue

        print(f"OK ({len(df)} barres)")
        print(f"[{pair}] Optimisation en cours...", end=" ", flush=True)

        best, all_results = optimize_pair(pair, df)

        if best:
            results[pair] = best
            optimal_configs[pair] = best['config']

            status = "PROFITABLE" if best['pf'] >= 1.0 else "NON-PROFITABLE"
            print(f"{status}")
            print(f"        Meilleure config: R:R={best['config']['rr']}, ADX>={best['config']['adx']}, "
                  f"RSI={best['config']['rsi']}, Score>={best['config']['score']}")
            print(f"        Resultats: {best['trades']} trades, WR={best['wr']:.1f}%, PF={best['pf']:.2f}, PnL=${best['pnl']:.0f}")
        else:
            print("AUCUNE CONFIG VIABLE")

        time_module.sleep(0.5)

    # Rapport final
    print("\n" + "=" * 80)
    print("                    RAPPORT D'OPTIMISATION FINAL")
    print("=" * 80)

    # Paires profitables
    profitable_pairs = [(p, r) for p, r in results.items() if r['pf'] >= 1.0]
    profitable_pairs.sort(key=lambda x: x[1]['pf'], reverse=True)

    print(f"\n{'='*80}")
    print(f"PAIRES PROFITABLES ({len(profitable_pairs)}/15)")
    print(f"{'='*80}")

    if profitable_pairs:
        print(f"\n{'Paire':<10} {'Trades':>8} {'WR%':>8} {'PF':>8} {'PnL':>12} {'Config Optimale'}")
        print("-" * 80)

        for pair, r in profitable_pairs:
            cfg = r['config']
            config_str = f"R:R={cfg['rr']}, ADX>{cfg['adx']}, RSI={cfg['rsi'][0]}-{cfg['rsi'][1]}, S>{cfg['score']}"
            print(f"{pair:<10} {r['trades']:>8} {r['wr']:>7.1f}% {r['pf']:>7.2f} ${r['pnl']:>10,.0f}  {config_str}")

    # Paires non-profitables mais ameliorees
    non_profitable = [(p, r) for p, r in results.items() if r['pf'] < 1.0]
    non_profitable.sort(key=lambda x: x[1]['pf'], reverse=True)

    print(f"\n{'='*80}")
    print(f"PAIRES A AMELIORER ({len(non_profitable)}/15)")
    print(f"{'='*80}")

    if non_profitable:
        print(f"\n{'Paire':<10} {'Trades':>8} {'WR%':>8} {'PF':>8} {'PnL':>12} {'Meilleure Config'}")
        print("-" * 80)

        for pair, r in non_profitable:
            cfg = r['config']
            config_str = f"R:R={cfg['rr']}, ADX>{cfg['adx']}, RSI={cfg['rsi'][0]}-{cfg['rsi'][1]}, S>{cfg['score']}"
            print(f"{pair:<10} {r['trades']:>8} {r['wr']:>7.1f}% {r['pf']:>7.2f} ${r['pnl']:>10,.0f}  {config_str}")

    # Recommandations
    print(f"\n{'='*80}")
    print("CONFIGURATIONS OPTIMALES RECOMMANDEES")
    print(f"{'='*80}")

    # Grouper par type de config
    config_groups = {}
    for pair, cfg in optimal_configs.items():
        key = f"R:R={cfg['rr']}, ADX>{cfg['adx']}, RSI={cfg['rsi']}, Score>{cfg['score']}"
        if key not in config_groups:
            config_groups[key] = []
        config_groups[key].append(pair)

    print("\nConfigurations les plus frequentes:")
    for config, pairs in sorted(config_groups.items(), key=lambda x: -len(x[1])):
        print(f"\n  {config}")
        print(f"     Paires: {', '.join(pairs)}")

    # Export
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Sauvegarder configs optimales
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
            'pnl': r['pnl']
        })

    df_config = pd.DataFrame(config_data)
    df_config.to_csv(f"optimal_configs_{ts}.csv", index=False)
    print(f"\n[EXPORT] optimal_configs_{ts}.csv")

    # Generer le code Python pour les configs
    print(f"\n{'='*80}")
    print("CODE PYTHON - CONFIGURATIONS OPTIMALES")
    print(f"{'='*80}")

    print("\nOPTIMAL_PAIR_CONFIGS = {")
    for pair, r in sorted(results.items()):
        cfg = r['config']
        print(f"    '{pair}': {{'rr': {cfg['rr']}, 'adx': {cfg['adx']}, 'rsi': {cfg['rsi']}, 'score': {cfg['score']}}},  # PF={r['pf']:.2f}")
    print("}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
