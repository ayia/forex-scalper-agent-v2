#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EURGBP Fast Strategy Optimizer - 10 Years Backtest
===================================================
Version optimisee pour execution rapide.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from itertools import product
import warnings
import sys

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
PAIR = 'EURGBP'
SYMBOL = f'{PAIR}=X'
CAPITAL = 10000
LOT_SIZE = 0.25
UNITS = LOT_SIZE * 100000
PIP_VALUE = 0.0001

# =============================================================================
# DATA FETCHING
# =============================================================================
def fetch_data(years: int = 10, interval: str = '1d') -> Optional[pd.DataFrame]:
    """Telecharge les donnees."""
    print(f"\nTelechargement {years} ans {interval}...", flush=True)

    try:
        df = yf.download(SYMBOL, period=f"{years}y", interval=interval, progress=False)

        if df.empty:
            print(f"  Erreur: Pas de donnees")
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        print(f"  OK: {len(df)} barres ({df.index[0].strftime('%Y-%m-%d')} -> {df.index[-1].strftime('%Y-%m-%d')})", flush=True)
        return df

    except Exception as e:
        print(f"  Erreur: {e}")
        return None


def fetch_h1_data() -> Optional[pd.DataFrame]:
    """Telecharge H1 (max 730 jours)."""
    print(f"\nTelechargement H1 (730 jours max)...", flush=True)

    try:
        df = yf.download(SYMBOL, period="730d", interval="1h", progress=False)

        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        print(f"  OK: {len(df)} barres H1", flush=True)
        return df

    except Exception as e:
        print(f"  Erreur: {e}")
        return None


# =============================================================================
# INDICATORS
# =============================================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les indicateurs techniques."""
    df = df.copy()

    # EMAs
    for period in [5, 8, 10, 13, 20, 21, 34, 50, 100, 200]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ATR
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()

    # ADX
    plus_dm = high.diff()
    minus_dm = low.diff().abs() * -1
    plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
    minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)
    atr14 = tr.rolling(window=14).mean()
    plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr14)
    minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr14)
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    df['adx'] = dx.rolling(window=14).mean()

    # Momentum
    df['momentum'] = df['close'].pct_change(periods=10) * 100

    # Bollinger
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
    df['bb_lower'] = df['bb_mid'] - (bb_std * 2)

    # Stochastic
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 0.0001)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

    return df


# =============================================================================
# TRADE SIMULATION
# =============================================================================
def simulate_trade(df: pd.DataFrame, entry_idx: int, direction: str,
                   entry_price: float, sl: float, tp: float,
                   max_bars: int = 100) -> Optional[dict]:
    """Simule un trade."""
    entry_time = df.index[entry_idx]

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


# =============================================================================
# STRATEGIES
# =============================================================================
def strategy_ema_crossover(df: pd.DataFrame, fast: int, slow: int, trend: int,
                           rr: float, adx_min: int, rsi_range: tuple, min_score: int,
                           sl_mult: float = 1.5) -> List[dict]:
    """EMA Crossover strategy."""
    trades = []
    ema_fast = f'ema_{fast}'
    ema_slow = f'ema_{slow}'
    ema_trend = f'ema_{trend}'

    if ema_fast not in df.columns or ema_slow not in df.columns or ema_trend not in df.columns:
        return trades

    rsi_low, rsi_high = rsi_range

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        score = 0
        direction = None

        # EMA crossover
        if prev[ema_fast] <= prev[ema_slow] and row[ema_fast] > row[ema_slow]:
            score += 2
            direction = 'BUY'
        elif prev[ema_fast] >= prev[ema_slow] and row[ema_fast] < row[ema_slow]:
            score += 2
            direction = 'SELL'

        if direction is None:
            continue

        # Trend
        if direction == 'BUY' and row['close'] > row[ema_trend]:
            score += 2
        elif direction == 'SELL' and row['close'] < row[ema_trend]:
            score += 2

        # RSI
        if rsi_low < row['rsi'] < rsi_high:
            score += 1

        # ADX
        if row['adx'] >= adx_min:
            score += 1

        # MACD
        if direction == 'BUY' and row['macd_hist'] > 0:
            score += 1
        elif direction == 'SELL' and row['macd_hist'] < 0:
            score += 1

        # Momentum
        if direction == 'BUY' and row['momentum'] > 0:
            score += 1
        elif direction == 'SELL' and row['momentum'] < 0:
            score += 1

        if score < min_score:
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


def strategy_macd(df: pd.DataFrame, rr: float, adx_min: int, sl_mult: float = 1.5) -> List[dict]:
    """MACD Crossover strategy."""
    trades = []

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        direction = None

        if prev['macd'] <= prev['macd_signal'] and row['macd'] > row['macd_signal']:
            direction = 'BUY'
        elif prev['macd'] >= prev['macd_signal'] and row['macd'] < row['macd_signal']:
            direction = 'SELL'

        if direction is None or row['adx'] < adx_min:
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


def strategy_bollinger(df: pd.DataFrame, rr: float, sl_mult: float = 1.5) -> List[dict]:
    """Bollinger Mean Reversion strategy."""
    trades = []

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        direction = None

        if prev['close'] < prev['bb_lower'] and row['close'] > row['bb_lower']:
            direction = 'BUY'
        elif prev['close'] > prev['bb_upper'] and row['close'] < row['bb_upper']:
            direction = 'SELL'

        if direction is None:
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


def strategy_stochastic(df: pd.DataFrame, rr: float, sl_mult: float = 1.5) -> List[dict]:
    """Stochastic Crossover strategy."""
    trades = []

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        direction = None

        if prev['stoch_k'] < prev['stoch_d'] and row['stoch_k'] > row['stoch_d']:
            if row['stoch_k'] < 30:
                direction = 'BUY'
        elif prev['stoch_k'] > prev['stoch_d'] and row['stoch_k'] < row['stoch_d']:
            if row['stoch_k'] > 70:
                direction = 'SELL'

        if direction is None:
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


# =============================================================================
# METRICS
# =============================================================================
def calc_metrics(trades: List[dict]) -> dict:
    """Calcule les metriques."""
    if not trades:
        return {'trades': 0, 'wr': 0, 'pf': 0, 'pnl': 0, 'max_dd': 0, 'roi': 0}

    wins = [t for t in trades if t['result'] == 'WIN']
    losses = [t for t in trades if t['result'] == 'LOSS']

    total_wins = sum(t['pnl'] for t in wins)
    total_losses = abs(sum(t['pnl'] for t in losses))

    pf = total_wins / total_losses if total_losses > 0 else 0
    wr = len(wins) / len(trades) * 100 if trades else 0
    pnl = sum(t['pnl'] for t in trades)

    # Max Drawdown
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


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("   EURGBP FAST OPTIMIZER - 10 YEARS DAILY")
    print("=" * 80)
    print(f"Capital: ${CAPITAL:,} | Lot: {LOT_SIZE}", flush=True)

    # Fetch data
    df = fetch_data(years=10, interval='1d')
    if df is None or len(df) < 500:
        print("Erreur: Donnees insuffisantes")
        return

    # Add indicators
    print("\nCalcul indicateurs...", flush=True)
    df = add_indicators(df)
    df = df.dropna()
    print(f"  {len(df)} barres utilisables", flush=True)

    # Test configurations - REDUCED for speed
    results = []

    # EMA Crossover - key combos only
    print("\n[1/4] EMA Crossover...", flush=True)
    ema_combos = [(8, 21, 50), (8, 21, 100), (10, 21, 50), (5, 13, 50)]
    rr_values = [1.5, 2.0, 2.5]
    adx_values = [15, 20, 25]
    rsi_ranges = [(30, 70), (35, 65)]
    score_mins = [5, 6]

    count = 0
    total = len(ema_combos) * len(rr_values) * len(adx_values) * len(rsi_ranges) * len(score_mins)

    for (fast, slow, trend), rr, adx, rsi, score in product(
        ema_combos, rr_values, adx_values, rsi_ranges, score_mins
    ):
        count += 1
        trades = strategy_ema_crossover(df, fast, slow, trend, rr, adx, rsi, score)
        m = calc_metrics(trades)

        if m['trades'] >= 30:
            results.append({
                'strategy': 'EMA_CROSSOVER',
                'params': {'ema': f'{fast}/{slow}/{trend}', 'rr': rr, 'adx': adx, 'rsi': rsi, 'score': score},
                **m
            })

    print(f"  {count} combos tested, {len(results)} valides", flush=True)

    # MACD
    print("\n[2/4] MACD...", flush=True)
    macd_count = len(results)
    for rr in [1.5, 2.0, 2.5]:
        for adx in [15, 20, 25]:
            trades = strategy_macd(df, rr, adx)
            m = calc_metrics(trades)
            if m['trades'] >= 30:
                results.append({
                    'strategy': 'MACD',
                    'params': {'rr': rr, 'adx': adx},
                    **m
                })
    print(f"  {len(results) - macd_count} valides", flush=True)

    # Bollinger
    print("\n[3/4] Bollinger...", flush=True)
    bb_count = len(results)
    for rr in [1.2, 1.5, 2.0, 2.5]:
        trades = strategy_bollinger(df, rr)
        m = calc_metrics(trades)
        if m['trades'] >= 30:
            results.append({
                'strategy': 'BOLLINGER',
                'params': {'rr': rr},
                **m
            })
    print(f"  {len(results) - bb_count} valides", flush=True)

    # Stochastic
    print("\n[4/4] Stochastic...", flush=True)
    stoch_count = len(results)
    for rr in [1.5, 2.0, 2.5]:
        trades = strategy_stochastic(df, rr)
        m = calc_metrics(trades)
        if m['trades'] >= 30:
            results.append({
                'strategy': 'STOCHASTIC',
                'params': {'rr': rr},
                **m
            })
    print(f"  {len(results) - stoch_count} valides", flush=True)

    # Sort by PF
    results.sort(key=lambda x: (-x['pf'], -x['pnl']))

    # Display results
    print("\n" + "=" * 80)
    print("TOP 15 STRATEGIES - 10 YEARS DAILY")
    print("=" * 80)
    print(f"{'#':<3} {'Strategy':<15} {'Params':<40} {'Tr':>5} {'WR%':>6} {'PF':>6} {'PnL':>10} {'ROI%':>7}")
    print("-" * 95)

    for i, r in enumerate(results[:15], 1):
        params_str = str(r['params'])[:38]
        print(f"{i:<3} {r['strategy']:<15} {params_str:<40} {r['trades']:>5} {r['wr']:>5.1f}% {r['pf']:>5.2f} ${r['pnl']:>+8,.0f} {r['roi']:>6.1f}%")

    if not results:
        print("\nAucune strategie valide trouvee!")
        return

    # Best strategy
    best = results[0]
    print("\n" + "=" * 80)
    print("MEILLEURE STRATEGIE (10 ans Daily)")
    print("=" * 80)
    print(f"Strategie: {best['strategy']}")
    print(f"Params: {best['params']}")
    print(f"Trades: {best['trades']} | WR: {best['wr']:.1f}% | PF: {best['pf']:.2f}")
    print(f"P&L: ${best['pnl']:+,.2f} | ROI: {best['roi']:+.1f}% | MaxDD: {best['max_dd']:.1f}%")

    # Validate on H1
    print("\n" + "=" * 80)
    print("VALIDATION H1 (2 ans)")
    print("=" * 80)

    df_h1 = fetch_h1_data()
    if df_h1 is not None and len(df_h1) > 500:
        df_h1 = add_indicators(df_h1)
        df_h1 = df_h1.dropna()
        print(f"  {len(df_h1)} barres H1 utilisables", flush=True)

        # Test best strategy on H1
        if best['strategy'] == 'EMA_CROSSOVER':
            p = best['params']
            ema = p['ema'].split('/')
            trades_h1 = strategy_ema_crossover(
                df_h1, int(ema[0]), int(ema[1]), int(ema[2]),
                p['rr'], p['adx'], p['rsi'], p['score']
            )
        elif best['strategy'] == 'MACD':
            trades_h1 = strategy_macd(df_h1, best['params']['rr'], best['params']['adx'])
        elif best['strategy'] == 'BOLLINGER':
            trades_h1 = strategy_bollinger(df_h1, best['params']['rr'])
        elif best['strategy'] == 'STOCHASTIC':
            trades_h1 = strategy_stochastic(df_h1, best['params']['rr'])
        else:
            trades_h1 = []

        m_h1 = calc_metrics(trades_h1)

        print(f"\nResultats H1:")
        print(f"  Trades: {m_h1['trades']} | WR: {m_h1['wr']:.1f}% | PF: {m_h1['pf']:.2f}")
        print(f"  P&L: ${m_h1['pnl']:+,.2f} | ROI: {m_h1['roi']:+.1f}%")

        if m_h1['pf'] >= 1.0 and m_h1['pnl'] > 0:
            print(f"\n[OK] STRATEGIE VALIDEE SUR H1!")
        else:
            print(f"\n[XX] NON VALIDEE SUR H1 - Recherche alternative...")

            # Test all strategies on H1
            h1_results = []

            for (fast, slow, trend), rr, adx, rsi, score in product(
                ema_combos, rr_values, adx_values, rsi_ranges, score_mins
            ):
                trades = strategy_ema_crossover(df_h1, fast, slow, trend, rr, adx, rsi, score)
                m = calc_metrics(trades)
                if m['trades'] >= 30 and m['pf'] > 1.0:
                    h1_results.append({
                        'strategy': 'EMA_CROSSOVER',
                        'params': {'ema': f'{fast}/{slow}/{trend}', 'rr': rr, 'adx': adx, 'rsi': rsi, 'score': score},
                        **m
                    })

            for rr in [1.5, 2.0, 2.5]:
                for adx in [15, 20, 25]:
                    trades = strategy_macd(df_h1, rr, adx)
                    m = calc_metrics(trades)
                    if m['trades'] >= 30 and m['pf'] > 1.0:
                        h1_results.append({'strategy': 'MACD', 'params': {'rr': rr, 'adx': adx}, **m})

            for rr in [1.2, 1.5, 2.0, 2.5]:
                trades = strategy_bollinger(df_h1, rr)
                m = calc_metrics(trades)
                if m['trades'] >= 30 and m['pf'] > 1.0:
                    h1_results.append({'strategy': 'BOLLINGER', 'params': {'rr': rr}, **m})

            if h1_results:
                h1_results.sort(key=lambda x: (-x['pf'], -x['pnl']))
                best_h1 = h1_results[0]
                print(f"\nMeilleure sur H1:")
                print(f"  {best_h1['strategy']}: {best_h1['params']}")
                print(f"  PF: {best_h1['pf']:.2f} | WR: {best_h1['wr']:.1f}% | P&L: ${best_h1['pnl']:+,.2f}")

    # Final summary
    print("\n" + "=" * 80)
    print("CONFIGURATION RECOMMANDEE EURGBP")
    print("=" * 80)

    profitable = [r for r in results if r['pf'] >= 1.0 and r['pnl'] > 0]

    if profitable:
        print(f"\n{len(profitable)} strategies profitables sur 10 ans Daily")
        print("\nTop 3:")
        for i, r in enumerate(profitable[:3], 1):
            print(f"  {i}. {r['strategy']}: PF={r['pf']:.2f}, WR={r['wr']:.1f}%, ROI={r['roi']:.1f}%")
            print(f"     {r['params']}")
    else:
        print("\nAucune strategie profitable sur 10 ans!")
        print("EURGBP peut ne pas etre adapte.")


if __name__ == "__main__":
    main()
