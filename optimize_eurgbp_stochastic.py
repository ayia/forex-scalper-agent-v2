#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EURGBP Stochastic Strategy - Deep Optimization
===============================================
Optimisation approfondie de la strategie Stochastic pour EURGBP.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from typing import List, Optional
from itertools import product
import warnings

warnings.filterwarnings('ignore')

PAIR = 'EURGBP'
SYMBOL = f'{PAIR}=X'
CAPITAL = 10000
LOT_SIZE = 0.25
PIP_VALUE = 0.0001


def fetch_data(years: int = 10, interval: str = '1d'):
    print(f"Telechargement {years}Y {interval}...", flush=True)
    df = yf.download(SYMBOL, period=f"{years}y", interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    print(f"  {len(df)} barres", flush=True)
    return df


def fetch_h1():
    print(f"Telechargement H1...", flush=True)
    df = yf.download(SYMBOL, period="730d", interval="1h", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.lower() for c in df.columns]
    print(f"  {len(df)} barres H1", flush=True)
    return df


def add_indicators(df, stoch_period=14, stoch_smooth=3):
    df = df.copy()

    # Stochastic with custom periods
    low_n = df['low'].rolling(window=stoch_period).min()
    high_n = df['high'].rolling(window=stoch_period).max()
    df['stoch_k'] = 100 * (df['close'] - low_n) / (high_n - low_n + 0.0001)
    df['stoch_d'] = df['stoch_k'].rolling(window=stoch_smooth).mean()

    # ATR
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()

    # ADX (optional filter)
    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1
    plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
    minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)
    atr14 = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    df['adx'] = dx.rolling(window=14).mean()

    # RSI (optional filter)
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # EMA trend
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()

    return df


def simulate_trade(df, entry_idx, direction, entry_price, sl, tp, max_bars=100):
    for j in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
        bar = df.iloc[j]

        if direction == 'BUY':
            if bar['low'] <= sl:
                pips = (sl - entry_price) / PIP_VALUE
                return {'result': 'LOSS', 'pips': pips, 'pnl': pips * 10 * LOT_SIZE}
            if bar['high'] >= tp:
                pips = (tp - entry_price) / PIP_VALUE
                return {'result': 'WIN', 'pips': pips, 'pnl': pips * 10 * LOT_SIZE}
        else:
            if bar['high'] >= sl:
                pips = (entry_price - sl) / PIP_VALUE
                return {'result': 'LOSS', 'pips': pips, 'pnl': pips * 10 * LOT_SIZE}
            if bar['low'] <= tp:
                pips = (entry_price - tp) / PIP_VALUE
                return {'result': 'WIN', 'pips': pips, 'pnl': pips * 10 * LOT_SIZE}

    return None


def strategy_stochastic(df, rr, oversold=20, overbought=80, zone_buffer=10,
                        sl_mult=1.5, adx_min=0, use_trend=False):
    """
    Stochastic Crossover strategy avec parametres configurables.

    Args:
        oversold: Niveau de survente (ex: 20)
        overbought: Niveau de surachat (ex: 80)
        zone_buffer: Marge autour des zones (ex: si oversold=20, zone_buffer=10, BUY si K < 30)
        adx_min: Filtre ADX minimum (0 = pas de filtre)
        use_trend: Utiliser EMA trend filter
    """
    trades = []

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        direction = None

        # Stochastic K crosses D
        if prev['stoch_k'] < prev['stoch_d'] and row['stoch_k'] > row['stoch_d']:
            if row['stoch_k'] < oversold + zone_buffer:
                direction = 'BUY'
        elif prev['stoch_k'] > prev['stoch_d'] and row['stoch_k'] < row['stoch_d']:
            if row['stoch_k'] > overbought - zone_buffer:
                direction = 'SELL'

        if direction is None:
            continue

        # Optional ADX filter
        if adx_min > 0 and row['adx'] < adx_min:
            continue

        # Optional trend filter
        if use_trend:
            if direction == 'BUY' and row['close'] < row['ema_50']:
                continue
            if direction == 'SELL' and row['close'] > row['ema_50']:
                continue

        entry_price = row['close']
        atr = row['atr']
        if pd.isna(atr) or atr <= 0:
            continue

        sl_dist = atr * sl_mult
        tp_dist = sl_dist * rr

        if direction == 'BUY':
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
        else:
            sl = entry_price + sl_dist
            tp = entry_price - tp_dist

        result = simulate_trade(df, i, direction, entry_price, sl, tp)
        if result:
            trades.append(result)

    return trades


def calc_metrics(trades):
    if not trades:
        return {'trades': 0, 'wr': 0, 'pf': 0, 'pnl': 0, 'max_dd': 0, 'roi': 0}

    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']

    total_wins = sum(t['pnl'] for t in wins)
    total_losses = abs(sum(t['pnl'] for t in losses))

    pf = total_wins / total_losses if total_losses > 0 else 0
    wr = len(wins) / len(trades) * 100 if trades else 0
    pnl = sum(t['pnl'] for t in trades)

    equity = CAPITAL
    peak = CAPITAL
    max_dd = 0
    for t in trades:
        equity += t['pnl']
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'wr': wr,
        'pf': pf,
        'pnl': pnl,
        'max_dd': max_dd,
        'roi': pnl / CAPITAL * 100
    }


def main():
    print("=" * 80)
    print("   EURGBP STOCHASTIC DEEP OPTIMIZATION")
    print("=" * 80)

    # Fetch 10Y Daily
    df = fetch_data(10, '1d')
    df = add_indicators(df)
    df = df.dropna()
    print(f"  {len(df)} barres utilisables\n", flush=True)

    # Test many configurations
    results = []

    rr_values = [1.5, 1.8, 2.0, 2.2, 2.5, 3.0]
    oversold_values = [15, 20, 25]
    overbought_values = [75, 80, 85]
    zone_buffers = [5, 10, 15]
    sl_mults = [1.0, 1.5, 2.0]
    adx_filters = [0, 15, 20]
    trend_filters = [False, True]

    total = (len(rr_values) * len(oversold_values) * len(overbought_values) *
             len(zone_buffers) * len(sl_mults) * len(adx_filters) * len(trend_filters))

    print(f"Testing {total} combinations...", flush=True)

    count = 0
    for rr, os, ob, zb, sl_m, adx, trend in product(
        rr_values, oversold_values, overbought_values,
        zone_buffers, sl_mults, adx_filters, trend_filters
    ):
        count += 1
        if count % 200 == 0:
            print(f"  {count}/{total}...", flush=True)

        trades = strategy_stochastic(df, rr, os, ob, zb, sl_m, adx, trend)
        m = calc_metrics(trades)

        if m['trades'] >= 50 and m['pf'] > 0.9:
            results.append({
                'params': {
                    'rr': rr, 'oversold': os, 'overbought': ob,
                    'zone_buffer': zb, 'sl_mult': sl_m,
                    'adx_min': adx, 'use_trend': trend
                },
                **m
            })

    print(f"  Done! {len(results)} valid configs\n", flush=True)

    # Sort by PF
    results.sort(key=lambda x: (-x['pf'], -x['pnl']))

    # Top 15
    print("=" * 100)
    print("TOP 15 STOCHASTIC CONFIGS - 10 YEARS DAILY")
    print("=" * 100)
    print(f"{'#':<3} {'RR':>4} {'OS':>4} {'OB':>4} {'ZB':>4} {'SL':>4} {'ADX':>4} {'Trend':>6} {'Tr':>5} {'WR%':>6} {'PF':>6} {'PnL':>10} {'ROI%':>7} {'DD%':>6}")
    print("-" * 100)

    for i, r in enumerate(results[:15], 1):
        p = r['params']
        trend_str = 'Yes' if p['use_trend'] else 'No'
        print(f"{i:<3} {p['rr']:>4.1f} {p['oversold']:>4} {p['overbought']:>4} {p['zone_buffer']:>4} {p['sl_mult']:>4.1f} {p['adx_min']:>4} {trend_str:>6} {r['trades']:>5} {r['wr']:>5.1f}% {r['pf']:>5.2f} ${r['pnl']:>+8,.0f} {r['roi']:>6.1f}% {r['max_dd']:>5.1f}%")

    if not results:
        print("No valid results!")
        return

    # Best config
    best = results[0]
    print("\n" + "=" * 80)
    print("BEST STOCHASTIC CONFIG (10Y Daily)")
    print("=" * 80)
    print(f"Parameters:")
    for k, v in best['params'].items():
        print(f"  {k}: {v}")
    print(f"\nPerformance:")
    print(f"  Trades: {best['trades']}")
    print(f"  Win Rate: {best['wr']:.1f}%")
    print(f"  Profit Factor: {best['pf']:.2f}")
    print(f"  P&L: ${best['pnl']:+,.2f}")
    print(f"  ROI: {best['roi']:+.1f}%")
    print(f"  Max DD: {best['max_dd']:.1f}%")

    # Validate on H1
    print("\n" + "=" * 80)
    print("VALIDATION H1 (2 years)")
    print("=" * 80)

    df_h1 = fetch_h1()
    df_h1 = add_indicators(df_h1)
    df_h1 = df_h1.dropna()
    print(f"  {len(df_h1)} H1 bars\n", flush=True)

    # Test best on H1
    p = best['params']
    trades_h1 = strategy_stochastic(
        df_h1, p['rr'], p['oversold'], p['overbought'],
        p['zone_buffer'], p['sl_mult'], p['adx_min'], p['use_trend']
    )
    m_h1 = calc_metrics(trades_h1)

    print(f"Best config on H1:")
    print(f"  Trades: {m_h1['trades']} | WR: {m_h1['wr']:.1f}% | PF: {m_h1['pf']:.2f}")
    print(f"  P&L: ${m_h1['pnl']:+,.2f} | ROI: {m_h1['roi']:+.1f}%")

    if m_h1['pf'] >= 1.0:
        print("\n[OK] VALIDATED ON H1!")
    else:
        print("\n[XX] NOT VALIDATED - Testing alternatives on H1...")

        # Test top configs on H1
        print("\nTop 5 configs tested on H1:")
        for i, r in enumerate(results[:10], 1):
            p = r['params']
            t_h1 = strategy_stochastic(
                df_h1, p['rr'], p['oversold'], p['overbought'],
                p['zone_buffer'], p['sl_mult'], p['adx_min'], p['use_trend']
            )
            m = calc_metrics(t_h1)
            if m['pf'] >= 1.0:
                print(f"  {i}. RR={p['rr']}, OS={p['oversold']}, OB={p['overbought']}, ZB={p['zone_buffer']} -> PF={m['pf']:.2f}, P&L=${m['pnl']:+,.0f}")

    # Final recommendation
    print("\n" + "=" * 80)
    print("CONFIGURATION FINALE RECOMMANDEE POUR EURGBP")
    print("=" * 80)

    # Find config that works on both
    print("\nSearching for config validated on both Daily AND H1...\n")

    best_both = None
    best_both_score = 0

    for r in results[:30]:
        p = r['params']
        t_h1 = strategy_stochastic(
            df_h1, p['rr'], p['oversold'], p['overbought'],
            p['zone_buffer'], p['sl_mult'], p['adx_min'], p['use_trend']
        )
        m_h1 = calc_metrics(t_h1)

        if r['pf'] >= 1.0 and m_h1['pf'] >= 1.0 and m_h1['trades'] >= 100:
            score = r['pf'] + m_h1['pf']
            if score > best_both_score:
                best_both_score = score
                best_both = {
                    'params': p,
                    'daily': r,
                    'h1': m_h1
                }

    if best_both:
        print("STRATEGIE VALIDEE SUR LES 2 TIMEFRAMES:")
        print("-" * 50)
        print(f"\nStrategy: STOCHASTIC_CROSSOVER")
        print(f"\nParameters:")
        for k, v in best_both['params'].items():
            print(f"  {k}: {v}")

        print(f"\n10Y Daily Performance:")
        d = best_both['daily']
        print(f"  Trades: {d['trades']} | WR: {d['wr']:.1f}% | PF: {d['pf']:.2f}")
        print(f"  P&L: ${d['pnl']:+,.2f} | ROI: {d['roi']:+.1f}%")

        print(f"\n2Y H1 Performance:")
        h = best_both['h1']
        print(f"  Trades: {h['trades']} | WR: {h['wr']:.1f}% | PF: {h['pf']:.2f}")
        print(f"  P&L: ${h['pnl']:+,.2f} | ROI: {h['roi']:+.1f}%")

        # Format for OPTIMAL_CONFIGS
        print("\n" + "=" * 80)
        print("CONFIG POUR OPTIMAL_CONFIGS:")
        print("=" * 80)
        p = best_both['params']
        print(f"""
'EURGBP': {{
    'strategy': 'STOCHASTIC',
    'rr': {p['rr']},
    'oversold': {p['oversold']},
    'overbought': {p['overbought']},
    'zone_buffer': {p['zone_buffer']},
    'sl_mult': {p['sl_mult']},
    'adx_min': {p['adx_min']},
    'use_trend': {p['use_trend']},
    'pf_daily': {d['pf']:.2f},
    'pf_h1': {h['pf']:.2f},
}}
""")
    else:
        print("Aucune configuration validee sur les 2 timeframes.")
        print("Utiliser la meilleure config Daily avec precaution:")
        print(f"  {best['params']}")


if __name__ == "__main__":
    main()
