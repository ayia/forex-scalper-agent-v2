#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EURGBP Strategy Optimizer - 10 Years Backtest
==============================================
Trouve la meilleure strategie pour EUR/GBP sur 10 ans.

Strategies testees:
1. EMA Crossover (8/21/50) - comme CADJPY
2. EMA Crossover variants (5/13/34, 10/20/50, etc.)
3. RSI Mean Reversion
4. Bollinger Bands Mean Reversion
5. MACD Crossover
6. Donchian Breakout
7. Combined strategies

Note: yfinance limite H1 a 730 jours.
      On utilise Daily pour 10 ans, puis valide sur H1 (2 ans).
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from itertools import product
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
PAIR = 'EURGBP'
SYMBOL = f'{PAIR}=X'
CAPITAL = 10000
LOT_SIZE = 0.25  # Comme CADJPY optimise
UNITS = LOT_SIZE * 100000
PIP_VALUE = 0.0001  # EURGBP pip value

# Periodes de test
PERIODS = {
    '10Y_DAILY': {'years': 10, 'interval': '1d', 'description': '10 ans Daily'},
    '2Y_H1': {'days': 730, 'interval': '1h', 'description': '2 ans H1 (validation)'},
}


# =============================================================================
# DATA FETCHING
# =============================================================================
def fetch_data(period_key: str) -> Optional[pd.DataFrame]:
    """Telecharge les donnees pour une periode."""
    config = PERIODS[period_key]
    interval = config['interval']

    print(f"\nTelechargement {config['description']}...")

    try:
        if 'years' in config:
            df = yf.download(SYMBOL, period=f"{config['years']}y", interval=interval, progress=False)
        else:
            df = yf.download(SYMBOL, period=f"{config['days']}d", interval=interval, progress=False)

        if df.empty:
            print(f"  Erreur: Pas de donnees")
            return None

        # Normaliser colonnes
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        print(f"  OK: {len(df)} barres du {df.index[0]} au {df.index[-1]}")
        return df

    except Exception as e:
        print(f"  Erreur: {e}")
        return None


# =============================================================================
# INDICATORS
# =============================================================================
def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute tous les indicateurs techniques."""
    df = df.copy()

    # EMAs (multiple periods pour optimisation)
    for period in [5, 8, 10, 13, 20, 21, 34, 50, 100, 200]:
        df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()

    # SMAs
    for period in [20, 50, 100, 200]:
        df[f'sma_{period}'] = df['close'].rolling(window=period).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # RSI different periods
    for period in [7, 14, 21]:
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss.replace(0, np.nan)
        df[f'rsi_{period}'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + (bb_std * 2)
    df['bb_lower'] = df['bb_mid'] - (bb_std * 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # ATR
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100

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
    df['plus_di'] = plus_di
    df['minus_di'] = minus_di

    # Donchian Channels
    for period in [20, 50]:
        df[f'donchian_high_{period}'] = df['high'].rolling(window=period).max()
        df[f'donchian_low_{period}'] = df['low'].rolling(window=period).min()
        df[f'donchian_mid_{period}'] = (df[f'donchian_high_{period}'] + df[f'donchian_low_{period}']) / 2

    # Stochastic
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14 + 0.0001)
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

    # Momentum
    df['momentum'] = df['close'].pct_change(periods=10) * 100
    df['roc'] = df['close'].pct_change(periods=12) * 100

    # Volatility
    df['volatility'] = df['close'].rolling(window=20).std() / df['close'] * 100

    return df


# =============================================================================
# STRATEGIES
# =============================================================================

def strategy_ema_crossover(df: pd.DataFrame, fast: int, slow: int, trend: int,
                           rr: float, adx_min: int, rsi_range: tuple, min_score: int,
                           sl_mult: float = 1.5) -> List[dict]:
    """
    Strategie EMA Crossover (comme CADJPY).
    """
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

        # Score calculation
        score = 0
        direction = None

        # EMA crossover (2 pts)
        if prev[ema_fast] <= prev[ema_slow] and row[ema_fast] > row[ema_slow]:
            score += 2
            direction = 'BUY'
        elif prev[ema_fast] >= prev[ema_slow] and row[ema_fast] < row[ema_slow]:
            score += 2
            direction = 'SELL'

        if direction is None:
            continue

        # Trend alignment (2 pts)
        if direction == 'BUY' and row['close'] > row[ema_trend]:
            score += 2
        elif direction == 'SELL' and row['close'] < row[ema_trend]:
            score += 2

        # RSI in range (1 pt)
        if rsi_low < row['rsi'] < rsi_high:
            score += 1

        # ADX OK (1 pt)
        if row['adx'] >= adx_min:
            score += 1

        # MACD direction (1 pt)
        if direction == 'BUY' and row['macd_hist'] > 0:
            score += 1
        elif direction == 'SELL' and row['macd_hist'] < 0:
            score += 1

        # Momentum (1 pt)
        if direction == 'BUY' and row['momentum'] > 0:
            score += 1
        elif direction == 'SELL' and row['momentum'] < 0:
            score += 1

        if score < min_score:
            continue

        # Entry
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

        # Simulate trade
        result = simulate_trade(df, i, direction, entry_price, sl, tp)
        if result:
            trades.append(result)

    return trades


def strategy_rsi_mean_reversion(df: pd.DataFrame, rsi_period: int,
                                 oversold: int, overbought: int,
                                 rr: float, sl_mult: float = 1.5) -> List[dict]:
    """
    Strategie RSI Mean Reversion.
    BUY quand RSI < oversold, SELL quand RSI > overbought.
    """
    trades = []
    rsi_col = f'rsi_{rsi_period}' if f'rsi_{rsi_period}' in df.columns else 'rsi'

    in_position = False

    for i in range(1, len(df) - 1):
        if in_position:
            continue

        row = df.iloc[i]
        prev = df.iloc[i-1]

        direction = None

        # RSI crossing back from extremes
        if prev[rsi_col] < oversold and row[rsi_col] > oversold:
            direction = 'BUY'
        elif prev[rsi_col] > overbought and row[rsi_col] < overbought:
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


def strategy_bollinger_reversion(df: pd.DataFrame, rr: float,
                                  sl_mult: float = 1.5) -> List[dict]:
    """
    Strategie Bollinger Bands Mean Reversion.
    BUY quand close < BB lower, SELL quand close > BB upper.
    """
    trades = []

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        direction = None

        # Price crossing back inside bands
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


def strategy_macd_crossover(df: pd.DataFrame, rr: float, adx_min: int,
                            sl_mult: float = 1.5) -> List[dict]:
    """
    Strategie MACD Crossover.
    """
    trades = []

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        direction = None

        # MACD crossover
        if prev['macd'] <= prev['macd_signal'] and row['macd'] > row['macd_signal']:
            direction = 'BUY'
        elif prev['macd'] >= prev['macd_signal'] and row['macd'] < row['macd_signal']:
            direction = 'SELL'

        if direction is None:
            continue

        # ADX filter
        if row['adx'] < adx_min:
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


def strategy_donchian_breakout(df: pd.DataFrame, period: int, rr: float,
                               adx_min: int, sl_mult: float = 1.5) -> List[dict]:
    """
    Strategie Donchian Breakout.
    """
    trades = []
    high_col = f'donchian_high_{period}'
    low_col = f'donchian_low_{period}'

    if high_col not in df.columns:
        return trades

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        direction = None

        # Breakout
        if row['close'] > prev[high_col]:
            direction = 'BUY'
        elif row['close'] < prev[low_col]:
            direction = 'SELL'

        if direction is None:
            continue

        # ADX filter
        if row['adx'] < adx_min:
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


def strategy_stochastic_crossover(df: pd.DataFrame, rr: float,
                                   oversold: int = 20, overbought: int = 80,
                                   sl_mult: float = 1.5) -> List[dict]:
    """
    Strategie Stochastic Crossover.
    """
    trades = []

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        direction = None

        # Stochastic crossover in extreme zones
        if prev['stoch_k'] < prev['stoch_d'] and row['stoch_k'] > row['stoch_d']:
            if row['stoch_k'] < oversold + 10:  # Near oversold
                direction = 'BUY'
        elif prev['stoch_k'] > prev['stoch_d'] and row['stoch_k'] < row['stoch_d']:
            if row['stoch_k'] > overbought - 10:  # Near overbought
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
# TRADE SIMULATION
# =============================================================================
def simulate_trade(df: pd.DataFrame, entry_idx: int, direction: str,
                   entry_price: float, sl: float, tp: float,
                   max_bars: int = 100) -> Optional[dict]:
    """Simule un trade et retourne le resultat."""

    entry_time = df.index[entry_idx]

    for j in range(entry_idx + 1, min(entry_idx + max_bars, len(df))):
        bar = df.iloc[j]

        if direction == 'BUY':
            if bar['low'] <= sl:
                pips = (sl - entry_price) / PIP_VALUE
                pnl = pips * 10 * LOT_SIZE  # $10 per pip per lot
                return {
                    'entry_time': entry_time,
                    'exit_time': df.index[j],
                    'direction': direction,
                    'entry': entry_price,
                    'exit': sl,
                    'result': 'LOSS',
                    'pips': pips,
                    'pnl': pnl
                }
            if bar['high'] >= tp:
                pips = (tp - entry_price) / PIP_VALUE
                pnl = pips * 10 * LOT_SIZE
                return {
                    'entry_time': entry_time,
                    'exit_time': df.index[j],
                    'direction': direction,
                    'entry': entry_price,
                    'exit': tp,
                    'result': 'WIN',
                    'pips': pips,
                    'pnl': pnl
                }
        else:  # SELL
            if bar['high'] >= sl:
                pips = (entry_price - sl) / PIP_VALUE
                pnl = pips * 10 * LOT_SIZE
                return {
                    'entry_time': entry_time,
                    'exit_time': df.index[j],
                    'direction': direction,
                    'entry': entry_price,
                    'exit': sl,
                    'result': 'LOSS',
                    'pips': pips,
                    'pnl': pnl
                }
            if bar['low'] <= tp:
                pips = (entry_price - tp) / PIP_VALUE
                pnl = pips * 10 * LOT_SIZE
                return {
                    'entry_time': entry_time,
                    'exit_time': df.index[j],
                    'direction': direction,
                    'entry': entry_price,
                    'exit': tp,
                    'result': 'WIN',
                    'pips': pips,
                    'pnl': pnl
                }

    return None  # Trade non cloture


# =============================================================================
# METRICS CALCULATION
# =============================================================================
def calculate_metrics(trades: List[dict]) -> dict:
    """Calcule les metriques de performance."""
    if not trades:
        return {
            'trades': 0, 'wins': 0, 'losses': 0,
            'wr': 0, 'pf': 0, 'pnl': 0,
            'avg_win': 0, 'avg_loss': 0, 'max_dd': 0
        }

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
        'avg_win': total_wins / len(wins) if wins else 0,
        'avg_loss': total_losses / len(losses) if losses else 0,
        'max_dd': max_dd,
        'roi': pnl / CAPITAL * 100
    }


# =============================================================================
# OPTIMIZATION
# =============================================================================
def optimize_ema_crossover(df: pd.DataFrame) -> List[dict]:
    """Optimise la strategie EMA Crossover."""
    results = []

    # Parametres a tester
    ema_combos = [
        (5, 13, 50), (5, 21, 50), (8, 21, 50), (8, 21, 100),
        (10, 20, 50), (10, 21, 50), (13, 34, 100), (20, 50, 200)
    ]
    rr_values = [1.2, 1.5, 1.8, 2.0, 2.5, 3.0]
    adx_values = [10, 15, 20, 25, 30]
    rsi_ranges = [(25, 75), (30, 70), (35, 65), (20, 80)]
    score_mins = [4, 5, 6]

    total = len(ema_combos) * len(rr_values) * len(adx_values) * len(rsi_ranges) * len(score_mins)
    print(f"\nOptimisation EMA Crossover: {total} combinaisons...")

    count = 0
    for (fast, slow, trend), rr, adx, rsi, score in product(
        ema_combos, rr_values, adx_values, rsi_ranges, score_mins
    ):
        count += 1
        if count % 100 == 0:
            print(f"  {count}/{total}...", end='\r')

        trades = strategy_ema_crossover(df, fast, slow, trend, rr, adx, rsi, score)
        metrics = calculate_metrics(trades)

        if metrics['trades'] >= 50:  # Minimum trades
            results.append({
                'strategy': 'EMA_CROSSOVER',
                'params': {
                    'ema': f'{fast}/{slow}/{trend}',
                    'rr': rr,
                    'adx': adx,
                    'rsi': rsi,
                    'score': score
                },
                **metrics
            })

    print(f"  {count}/{total} - Done!")
    return results


def optimize_rsi_reversion(df: pd.DataFrame) -> List[dict]:
    """Optimise la strategie RSI Mean Reversion."""
    results = []

    rsi_periods = [7, 14, 21]
    oversold_values = [20, 25, 30]
    overbought_values = [70, 75, 80]
    rr_values = [1.2, 1.5, 2.0, 2.5]

    total = len(rsi_periods) * len(oversold_values) * len(overbought_values) * len(rr_values)
    print(f"\nOptimisation RSI Reversion: {total} combinaisons...")

    count = 0
    for period, os, ob, rr in product(rsi_periods, oversold_values, overbought_values, rr_values):
        count += 1
        trades = strategy_rsi_mean_reversion(df, period, os, ob, rr)
        metrics = calculate_metrics(trades)

        if metrics['trades'] >= 50:
            results.append({
                'strategy': 'RSI_REVERSION',
                'params': {
                    'period': period,
                    'oversold': os,
                    'overbought': ob,
                    'rr': rr
                },
                **metrics
            })

    print(f"  {count}/{total} - Done!")
    return results


def optimize_bollinger(df: pd.DataFrame) -> List[dict]:
    """Optimise la strategie Bollinger."""
    results = []

    rr_values = [1.0, 1.2, 1.5, 2.0, 2.5]

    print(f"\nOptimisation Bollinger: {len(rr_values)} combinaisons...")

    for rr in rr_values:
        trades = strategy_bollinger_reversion(df, rr)
        metrics = calculate_metrics(trades)

        if metrics['trades'] >= 50:
            results.append({
                'strategy': 'BOLLINGER_REVERSION',
                'params': {'rr': rr},
                **metrics
            })

    print(f"  Done!")
    return results


def optimize_macd(df: pd.DataFrame) -> List[dict]:
    """Optimise la strategie MACD."""
    results = []

    rr_values = [1.2, 1.5, 2.0, 2.5]
    adx_values = [10, 15, 20, 25]

    total = len(rr_values) * len(adx_values)
    print(f"\nOptimisation MACD: {total} combinaisons...")

    for rr, adx in product(rr_values, adx_values):
        trades = strategy_macd_crossover(df, rr, adx)
        metrics = calculate_metrics(trades)

        if metrics['trades'] >= 50:
            results.append({
                'strategy': 'MACD_CROSSOVER',
                'params': {'rr': rr, 'adx': adx},
                **metrics
            })

    print(f"  Done!")
    return results


def optimize_donchian(df: pd.DataFrame) -> List[dict]:
    """Optimise la strategie Donchian."""
    results = []

    periods = [20, 50]
    rr_values = [1.2, 1.5, 2.0, 2.5]
    adx_values = [10, 15, 20, 25]

    total = len(periods) * len(rr_values) * len(adx_values)
    print(f"\nOptimisation Donchian: {total} combinaisons...")

    for period, rr, adx in product(periods, rr_values, adx_values):
        trades = strategy_donchian_breakout(df, period, rr, adx)
        metrics = calculate_metrics(trades)

        if metrics['trades'] >= 50:
            results.append({
                'strategy': 'DONCHIAN_BREAKOUT',
                'params': {'period': period, 'rr': rr, 'adx': adx},
                **metrics
            })

    print(f"  Done!")
    return results


def optimize_stochastic(df: pd.DataFrame) -> List[dict]:
    """Optimise la strategie Stochastic."""
    results = []

    rr_values = [1.2, 1.5, 2.0, 2.5]

    print(f"\nOptimisation Stochastic: {len(rr_values)} combinaisons...")

    for rr in rr_values:
        trades = strategy_stochastic_crossover(df, rr)
        metrics = calculate_metrics(trades)

        if metrics['trades'] >= 50:
            results.append({
                'strategy': 'STOCHASTIC_CROSSOVER',
                'params': {'rr': rr},
                **metrics
            })

    print(f"  Done!")
    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("   EURGBP STRATEGY OPTIMIZER - 10 YEARS BACKTEST")
    print("=" * 80)
    print(f"\nPaire: {PAIR}")
    print(f"Capital: ${CAPITAL:,}")
    print(f"Lot size: {LOT_SIZE}")
    print(f"Pip value: {PIP_VALUE}")

    # Fetch 10 years Daily data
    df_daily = fetch_data('10Y_DAILY')
    if df_daily is None or len(df_daily) < 500:
        print("Erreur: Donnees insuffisantes")
        return

    # Add indicators
    print("\nCalcul des indicateurs...")
    df_daily = add_indicators(df_daily)
    df_daily = df_daily.dropna()
    print(f"  {len(df_daily)} barres utilisables")

    # Run all optimizations
    all_results = []

    all_results.extend(optimize_ema_crossover(df_daily))
    all_results.extend(optimize_rsi_reversion(df_daily))
    all_results.extend(optimize_bollinger(df_daily))
    all_results.extend(optimize_macd(df_daily))
    all_results.extend(optimize_donchian(df_daily))
    all_results.extend(optimize_stochastic(df_daily))

    if not all_results:
        print("\nAucun resultat valide!")
        return

    # Sort by Profit Factor
    all_results.sort(key=lambda x: (-x['pf'], -x['pnl']))

    # Display top results
    print("\n" + "=" * 80)
    print("TOP 20 STRATEGIES (sorted by Profit Factor)")
    print("=" * 80)
    print(f"{'#':<3} {'Strategy':<20} {'Params':<35} {'Trades':>7} {'WR%':>6} {'PF':>6} {'PnL':>10} {'ROI%':>7} {'MaxDD%':>7}")
    print("-" * 110)

    for i, r in enumerate(all_results[:20], 1):
        params_str = str(r['params'])[:33]
        print(f"{i:<3} {r['strategy']:<20} {params_str:<35} {r['trades']:>7} {r['wr']:>5.1f}% {r['pf']:>5.2f} ${r['pnl']:>+9,.0f} {r['roi']:>6.1f}% {r['max_dd']:>6.1f}%")

    # Best strategy details
    best = all_results[0]
    print("\n" + "=" * 80)
    print("MEILLEURE STRATEGIE TROUVEE")
    print("=" * 80)
    print(f"\nStrategie: {best['strategy']}")
    print(f"Parametres: {best['params']}")
    print(f"\nPerformance sur 10 ans (Daily):")
    print(f"  - Trades: {best['trades']}")
    print(f"  - Win Rate: {best['wr']:.1f}%")
    print(f"  - Profit Factor: {best['pf']:.2f}")
    print(f"  - P&L: ${best['pnl']:+,.2f}")
    print(f"  - ROI: {best['roi']:+.1f}%")
    print(f"  - Max Drawdown: {best['max_dd']:.1f}%")
    print(f"  - Avg Win: ${best['avg_win']:.2f}")
    print(f"  - Avg Loss: ${best['avg_loss']:.2f}")

    # Validate on H1 (2 years)
    print("\n" + "=" * 80)
    print("VALIDATION SUR H1 (2 ans)")
    print("=" * 80)

    df_h1 = fetch_data('2Y_H1')
    if df_h1 is not None and len(df_h1) > 500:
        df_h1 = add_indicators(df_h1)
        df_h1 = df_h1.dropna()

        # Test best strategy on H1
        if best['strategy'] == 'EMA_CROSSOVER':
            params = best['params']
            ema_parts = params['ema'].split('/')
            trades_h1 = strategy_ema_crossover(
                df_h1, int(ema_parts[0]), int(ema_parts[1]), int(ema_parts[2]),
                params['rr'], params['adx'], params['rsi'], params['score']
            )
        elif best['strategy'] == 'RSI_REVERSION':
            params = best['params']
            trades_h1 = strategy_rsi_mean_reversion(
                df_h1, params['period'], params['oversold'], params['overbought'], params['rr']
            )
        elif best['strategy'] == 'BOLLINGER_REVERSION':
            trades_h1 = strategy_bollinger_reversion(df_h1, best['params']['rr'])
        elif best['strategy'] == 'MACD_CROSSOVER':
            trades_h1 = strategy_macd_crossover(df_h1, best['params']['rr'], best['params']['adx'])
        elif best['strategy'] == 'DONCHIAN_BREAKOUT':
            params = best['params']
            trades_h1 = strategy_donchian_breakout(df_h1, params['period'], params['rr'], params['adx'])
        elif best['strategy'] == 'STOCHASTIC_CROSSOVER':
            trades_h1 = strategy_stochastic_crossover(df_h1, best['params']['rr'])
        else:
            trades_h1 = []

        metrics_h1 = calculate_metrics(trades_h1)

        print(f"\nPerformance sur H1 (2 ans):")
        print(f"  - Trades: {metrics_h1['trades']}")
        print(f"  - Win Rate: {metrics_h1['wr']:.1f}%")
        print(f"  - Profit Factor: {metrics_h1['pf']:.2f}")
        print(f"  - P&L: ${metrics_h1['pnl']:+,.2f}")
        print(f"  - ROI: {metrics_h1['roi']:+.1f}%")
        print(f"  - Max Drawdown: {metrics_h1['max_dd']:.1f}%")

        # Validation status
        if metrics_h1['pf'] >= 1.0 and metrics_h1['pnl'] > 0:
            print(f"\n[OK] STRATEGIE VALIDEE sur H1!")
        else:
            print(f"\n[XX] STRATEGIE NON VALIDEE sur H1 - Chercher alternative")

            # Find best on H1
            print("\nRecherche de la meilleure strategie sur H1...")
            h1_results = []
            h1_results.extend(optimize_ema_crossover(df_h1))
            h1_results.extend(optimize_rsi_reversion(df_h1))
            h1_results.extend(optimize_bollinger(df_h1))
            h1_results.extend(optimize_macd(df_h1))

            if h1_results:
                h1_results.sort(key=lambda x: (-x['pf'], -x['pnl']))
                best_h1 = h1_results[0]
                print(f"\nMeilleure sur H1: {best_h1['strategy']}")
                print(f"Params: {best_h1['params']}")
                print(f"PF: {best_h1['pf']:.2f}, WR: {best_h1['wr']:.1f}%, P&L: ${best_h1['pnl']:+,.2f}")

    # Summary
    print("\n" + "=" * 80)
    print("CONFIGURATION RECOMMANDEE POUR EURGBP")
    print("=" * 80)

    # Find profitable strategies
    profitable = [r for r in all_results if r['pf'] >= 1.0 and r['pnl'] > 0]

    if profitable:
        print(f"\n{len(profitable)} strategies profitables trouvees sur 10 ans")
        print("\nTop 5 par Profit Factor:")
        for i, r in enumerate(profitable[:5], 1):
            print(f"  {i}. {r['strategy']}: PF={r['pf']:.2f}, WR={r['wr']:.1f}%, ROI={r['roi']:.1f}%")
            print(f"     Params: {r['params']}")
    else:
        print("\nAucune strategie profitable sur 10 ans!")
        print("EURGBP peut ne pas etre adapte a ces strategies.")


if __name__ == "__main__":
    main()
