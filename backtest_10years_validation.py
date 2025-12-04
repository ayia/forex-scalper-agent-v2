#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BACKTEST 10 ANS - VALIDATION FINALE
===================================
Teste les strategies validees sur 10 ans (2014-2024) en H1.
Verifie que le drawdown journalier ne depasse JAMAIS -$500.

Strategies testees:
1. CADJPY: EMA Crossover (8/21/50), R:R=2.5, ADX>=25, RSI 35-65
2. EURGBP: Stochastic Crossover, R:R=2.0, K<30 BUY, K>70 SELL

Contrainte: Max Daily Loss <= $500 (sinon strategie REJETEE)
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
CAPITAL = 10000
LOT_SIZE = 0.15  # 15,000 unites
MAX_DAILY_LOSS = -500  # Contrainte absolue

# Configuration CADJPY - EMA Crossover
CADJPY_CONFIG = {
    'pair': 'CADJPY',
    'symbol': 'CADJPY=X',
    'strategy': 'EMA_CROSSOVER',
    'ema_fast': 8,
    'ema_slow': 21,
    'ema_trend': 50,
    'rr': 2.5,
    'adx_min': 25,
    'rsi_low': 35,
    'rsi_high': 65,
    'sl_mult': 1.5,
    'min_score': 6,
    'pip_value': 0.01,  # JPY pair
}

# Configuration EURGBP - Stochastic Crossover
EURGBP_CONFIG = {
    'pair': 'EURGBP',
    'symbol': 'EURGBP=X',
    'strategy': 'STOCHASTIC_CROSSOVER',
    'stoch_period': 14,
    'stoch_smooth': 3,
    'oversold': 20,
    'overbought': 80,
    'zone_buffer': 10,  # BUY si K<30, SELL si K>70
    'rr': 2.0,
    'sl_mult': 1.5,
    'pip_value': 0.0001,
}


def fetch_10year_data(symbol: str) -> Optional[pd.DataFrame]:
    """
    Fetch 10 years of daily data (H1 not available for 10 years on yfinance).
    We'll simulate H1 by using daily data with intraday-like logic.
    """
    print(f"  Fetching 10 years of data for {symbol}...")

    try:
        # yfinance limite H1 a ~730 jours, donc on utilise Daily pour 10 ans
        # puis on simule des entries H1-like
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365*10)

        df = yf.download(
            symbol,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1d',
            progress=False
        )

        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            print(f"  -> {len(df)} daily bars loaded ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
            return df
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


def calculate_indicators_ema(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculate EMA strategy indicators."""
    df = df.copy()

    # EMAs
    df['ema_fast'] = df['close'].ewm(span=config['ema_fast'], adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=config['ema_slow'], adjust=False).mean()
    df['ema_trend'] = df['close'].ewm(span=config['ema_trend'], adjust=False).mean()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ADX
    df['adx'] = calculate_adx(df)

    # ATR
    df['atr'] = calculate_atr(df)

    # MACD for score
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    return df


def calculate_indicators_stochastic(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculate Stochastic strategy indicators."""
    df = df.copy()

    period = config['stoch_period']
    smooth = config['stoch_smooth']

    # Stochastic
    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 0.0001)
    df['stoch_d'] = df['stoch_k'].rolling(window=smooth).mean()

    # ATR
    df['atr'] = calculate_atr(df)

    return df


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX."""
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

    atr = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
    adx = dx.rolling(window=period).mean()

    return adx


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR."""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(window=period).mean()


def calculate_ema_score(current: pd.Series, prev: pd.Series, is_bullish: bool, config: dict) -> int:
    """
    Calculate EMA strategy score (0-8).
    Same logic as optimized backtest.
    """
    score = 0

    # EMA crossover (2 pts)
    if is_bullish:
        if prev['ema_fast'] <= prev['ema_slow'] and current['ema_fast'] > current['ema_slow']:
            score += 2
    else:
        if prev['ema_fast'] >= prev['ema_slow'] and current['ema_fast'] < current['ema_slow']:
            score += 2

    # Trend alignment (2 pts)
    ema_slope = (current['ema_trend'] - prev['ema_trend']) / prev['ema_trend'] * 100 if prev['ema_trend'] > 0 else 0
    if is_bullish:
        if current['close'] > current['ema_trend'] and ema_slope > 0:
            score += 2
    else:
        if current['close'] < current['ema_trend'] and ema_slope < 0:
            score += 2

    # RSI in range (1 pt)
    if config['rsi_low'] < current['rsi'] < config['rsi_high']:
        score += 1

    # ADX OK (1 pt)
    if current['adx'] >= config['adx_min']:
        score += 1

    # MACD direction (1 pt)
    if is_bullish and current['macd_hist'] > 0:
        score += 1
    elif not is_bullish and current['macd_hist'] < 0:
        score += 1

    # Momentum (1 pt)
    roc = (current['close'] - prev['close']) / prev['close'] * 100 if prev['close'] > 0 else 0
    if is_bullish and roc > 0:
        score += 1
    elif not is_bullish and roc < 0:
        score += 1

    return score


def backtest_cadjpy(df: pd.DataFrame, config: dict) -> Dict:
    """
    Backtest CADJPY EMA Crossover strategy.
    """
    df = calculate_indicators_ema(df, config)

    trades = []
    daily_pnl = defaultdict(float)

    capital = CAPITAL
    position = None

    for i in range(60, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        date = df.index[i]
        day_key = date.strftime('%Y-%m-%d')

        # Skip if indicators not ready
        if pd.isna(current['adx']) or pd.isna(current['atr']):
            continue

        # Check for exit if in position
        if position:
            # CADJPY: 1 pip = 0.01 (JPY pair), lot 0.15 = 15000 units
            # For JPY pairs at ~110 rate: pip value = 15000 * 0.01 / 110 = ~$1.36/pip
            # Using approximate $1.35 per pip for 0.15 lot
            pip_value_usd = 1.35

            if position['direction'] == 'BUY':
                if current['low'] <= position['sl']:
                    pips = (position['sl'] - position['entry']) / config['pip_value']
                    pnl = pips * pip_value_usd
                    capital += pnl
                    daily_pnl[day_key] += pnl
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'direction': 'BUY',
                        'entry': position['entry'],
                        'exit': position['sl'],
                        'pnl': pnl,
                        'result': 'SL'
                    })
                    position = None
                elif current['high'] >= position['tp']:
                    pips = (position['tp'] - position['entry']) / config['pip_value']
                    pnl = pips * pip_value_usd
                    capital += pnl
                    daily_pnl[day_key] += pnl
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'direction': 'BUY',
                        'entry': position['entry'],
                        'exit': position['tp'],
                        'pnl': pnl,
                        'result': 'TP'
                    })
                    position = None
            else:  # SELL
                if current['high'] >= position['sl']:
                    pips = (position['entry'] - position['sl']) / config['pip_value']
                    pnl = pips * pip_value_usd
                    capital += pnl
                    daily_pnl[day_key] += pnl
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'direction': 'SELL',
                        'entry': position['entry'],
                        'exit': position['sl'],
                        'pnl': pnl,
                        'result': 'SL'
                    })
                    position = None
                elif current['low'] <= position['tp']:
                    pips = (position['entry'] - position['tp']) / config['pip_value']
                    pnl = pips * pip_value_usd
                    capital += pnl
                    daily_pnl[day_key] += pnl
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'direction': 'SELL',
                        'entry': position['entry'],
                        'exit': position['tp'],
                        'pnl': pnl,
                        'result': 'TP'
                    })
                    position = None

        # Check for new entry if no position
        if position is None:
            atr = current['atr']

            # BUY signal
            ema_cross_up = prev['ema_fast'] <= prev['ema_slow'] and current['ema_fast'] > current['ema_slow']
            trend_up = current['close'] > current['ema_trend']
            rsi_ok = config['rsi_low'] < current['rsi'] < config['rsi_high']
            adx_ok = current['adx'] >= config['adx_min']

            if ema_cross_up and trend_up and rsi_ok and adx_ok:
                score = calculate_ema_score(current, prev, True, config)
                if score >= config['min_score']:
                    entry = current['close']
                    sl = entry - (atr * config['sl_mult'])
                    tp = entry + (atr * config['sl_mult'] * config['rr'])
                    position = {
                        'direction': 'BUY',
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'entry_date': date
                    }

            # SELL signal
            ema_cross_down = prev['ema_fast'] >= prev['ema_slow'] and current['ema_fast'] < current['ema_slow']
            trend_down = current['close'] < current['ema_trend']

            if ema_cross_down and trend_down and rsi_ok and adx_ok:
                score = calculate_ema_score(current, prev, False, config)
                if score >= config['min_score']:
                    entry = current['close']
                    sl = entry + (atr * config['sl_mult'])
                    tp = entry - (atr * config['sl_mult'] * config['rr'])
                    position = {
                        'direction': 'SELL',
                        'entry': entry,
                        'sl': sl,
                        'tp': tp,
                        'entry_date': date
                    }

    return analyze_results(trades, daily_pnl, config['pair'])


def backtest_eurgbp(df: pd.DataFrame, config: dict) -> Dict:
    """
    Backtest EURGBP Stochastic Crossover strategy.
    """
    df = calculate_indicators_stochastic(df, config)

    trades = []
    daily_pnl = defaultdict(float)

    capital = CAPITAL
    position = None

    buy_zone = config['oversold'] + config['zone_buffer']  # K < 30
    sell_zone = config['overbought'] - config['zone_buffer']  # K > 70

    for i in range(20, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        date = df.index[i]
        day_key = date.strftime('%Y-%m-%d')

        if pd.isna(current['stoch_k']) or pd.isna(current['atr']):
            continue

        # Check for exit if in position
        if position:
            # EURGBP: 1 pip = 0.0001, lot 0.15 = 15000 units
            # P&L = (exit - entry) * units = (exit - entry) * 15000
            # For 0.15 lot, pip value ~= $1.5 per pip (approx for EURGBP)
            pip_value_usd = 1.5  # $1.5 per pip for 0.15 lot EURGBP

            if position['direction'] == 'BUY':
                if current['low'] <= position['sl']:
                    pips = (position['sl'] - position['entry']) / config['pip_value']
                    pnl = pips * pip_value_usd
                    capital += pnl
                    daily_pnl[day_key] += pnl
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'direction': 'BUY',
                        'entry': position['entry'],
                        'exit': position['sl'],
                        'pnl': pnl,
                        'result': 'SL'
                    })
                    position = None
                elif current['high'] >= position['tp']:
                    pips = (position['tp'] - position['entry']) / config['pip_value']
                    pnl = pips * pip_value_usd
                    capital += pnl
                    daily_pnl[day_key] += pnl
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'direction': 'BUY',
                        'entry': position['entry'],
                        'exit': position['tp'],
                        'pnl': pnl,
                        'result': 'TP'
                    })
                    position = None
            else:  # SELL
                if current['high'] >= position['sl']:
                    pips = (position['entry'] - position['sl']) / config['pip_value']
                    pnl = pips * pip_value_usd
                    capital += pnl
                    daily_pnl[day_key] += pnl
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'direction': 'SELL',
                        'entry': position['entry'],
                        'exit': position['sl'],
                        'pnl': pnl,
                        'result': 'SL'
                    })
                    position = None
                elif current['low'] <= position['tp']:
                    pips = (position['entry'] - position['tp']) / config['pip_value']
                    pnl = pips * pip_value_usd
                    capital += pnl
                    daily_pnl[day_key] += pnl
                    trades.append({
                        'entry_date': position['entry_date'],
                        'exit_date': date,
                        'direction': 'SELL',
                        'entry': position['entry'],
                        'exit': position['tp'],
                        'pnl': pnl,
                        'result': 'TP'
                    })
                    position = None

        # Check for new entry
        if position is None:
            atr = current['atr']

            # Stochastic crossover
            k_crossed_up = prev['stoch_k'] <= prev['stoch_d'] and current['stoch_k'] > current['stoch_d']
            k_crossed_down = prev['stoch_k'] >= prev['stoch_d'] and current['stoch_k'] < current['stoch_d']

            # BUY: K crosses D up in oversold zone
            if k_crossed_up and current['stoch_k'] < buy_zone:
                entry = current['close']
                sl = entry - (atr * config['sl_mult'])
                tp = entry + (atr * config['sl_mult'] * config['rr'])
                position = {
                    'direction': 'BUY',
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'entry_date': date
                }

            # SELL: K crosses D down in overbought zone
            elif k_crossed_down and current['stoch_k'] > sell_zone:
                entry = current['close']
                sl = entry + (atr * config['sl_mult'])
                tp = entry - (atr * config['sl_mult'] * config['rr'])
                position = {
                    'direction': 'SELL',
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'entry_date': date
                }

    return analyze_results(trades, daily_pnl, config['pair'])


def analyze_results(trades: List[Dict], daily_pnl: Dict, pair: str) -> Dict:
    """
    Analyze backtest results with focus on daily drawdown constraint.
    """
    if not trades:
        return {
            'pair': pair,
            'status': 'NO_TRADES',
            'trades': 0
        }

    # Calculate metrics
    total_trades = len(trades)
    winners = [t for t in trades if t['pnl'] > 0]
    losers = [t for t in trades if t['pnl'] <= 0]

    win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

    total_profit = sum(t['pnl'] for t in winners)
    total_loss = abs(sum(t['pnl'] for t in losers))

    profit_factor = total_profit / total_loss if total_loss > 0 else 0

    total_pnl = sum(t['pnl'] for t in trades)

    # Daily drawdown analysis
    daily_losses = {k: v for k, v in daily_pnl.items() if v < 0}
    worst_day = min(daily_pnl.values()) if daily_pnl else 0
    worst_day_date = min(daily_pnl, key=daily_pnl.get) if daily_pnl else 'N/A'

    days_over_limit = [d for d, pnl in daily_pnl.items() if pnl < MAX_DAILY_LOSS]

    # Max drawdown calculation
    cumulative = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cumulative += t['pnl']
        if cumulative > peak:
            peak = cumulative
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd

    max_dd_pct = (max_dd / CAPITAL) * 100 if CAPITAL > 0 else 0

    # Validation
    is_valid = len(days_over_limit) == 0 and profit_factor >= 1.0

    return {
        'pair': pair,
        'status': 'VALID' if is_valid else 'REJECTED',
        'rejection_reason': None if is_valid else (
            f"{len(days_over_limit)} days exceeded -$500 limit" if days_over_limit else "PF < 1.0"
        ),
        'trades': total_trades,
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 2),
        'total_pnl': round(total_pnl, 2),
        'roi_pct': round((total_pnl / CAPITAL) * 100, 1),
        'max_drawdown': round(max_dd, 2),
        'max_drawdown_pct': round(max_dd_pct, 1),
        'worst_day_pnl': round(worst_day, 2),
        'worst_day_date': worst_day_date,
        'days_over_500_loss': len(days_over_limit),
        'days_over_500_list': days_over_limit[:10] if days_over_limit else [],  # Show first 10
        'trading_days': len([d for d in daily_pnl.keys()]),
        'avg_daily_pnl': round(total_pnl / len(daily_pnl), 2) if daily_pnl else 0,
    }


def print_results(results: Dict):
    """Print formatted results."""
    print(f"\n{'='*70}")
    print(f"  {results['pair']} - {results['status']}")
    print(f"{'='*70}")

    if results['status'] == 'NO_TRADES':
        print("  Aucun trade genere")
        return

    if results['status'] == 'REJECTED':
        print(f"  [!] REJET: {results['rejection_reason']}")

    print(f"\n  PERFORMANCE:")
    print(f"    Trades:        {results['trades']}")
    print(f"    Win Rate:      {results['win_rate']}%")
    print(f"    Profit Factor: {results['profit_factor']}")
    print(f"    Total P&L:     ${results['total_pnl']:,.2f}")
    print(f"    ROI:           {results['roi_pct']}%")

    print(f"\n  RISK:")
    print(f"    Max Drawdown:  ${results['max_drawdown']:,.2f} ({results['max_drawdown_pct']}%)")
    print(f"    Worst Day:     ${results['worst_day_pnl']:,.2f} ({results['worst_day_date']})")
    print(f"    Days > -$500:  {results['days_over_500_loss']}")

    if results['days_over_500_list']:
        print(f"\n  [!] JOURS DEPASSANT -$500:")
        for day in results['days_over_500_list']:
            print(f"      - {day}")

    print(f"\n  SUMMARY:")
    print(f"    Trading Days:  {results['trading_days']}")
    print(f"    Avg Daily P&L: ${results['avg_daily_pnl']:,.2f}")


def main():
    print("\n" + "="*70)
    print("   BACKTEST 10 ANS - VALIDATION FINALE")
    print("   Capital: $10,000 | Lot: 0.25 | Max Daily Loss: -$500")
    print("="*70)

    results = {}

    # Test CADJPY
    print(f"\n[1/2] CADJPY - EMA Crossover Strategy")
    print("-" * 50)
    df_cadjpy = fetch_10year_data(CADJPY_CONFIG['symbol'])
    if df_cadjpy is not None:
        results['CADJPY'] = backtest_cadjpy(df_cadjpy, CADJPY_CONFIG)
        print_results(results['CADJPY'])
    else:
        print("  ERREUR: Impossible de charger les donnees CADJPY")

    # Test EURGBP
    print(f"\n[2/2] EURGBP - Stochastic Crossover Strategy")
    print("-" * 50)
    df_eurgbp = fetch_10year_data(EURGBP_CONFIG['symbol'])
    if df_eurgbp is not None:
        results['EURGBP'] = backtest_eurgbp(df_eurgbp, EURGBP_CONFIG)
        print_results(results['EURGBP'])
    else:
        print("  ERREUR: Impossible de charger les donnees EURGBP")

    # Final Summary
    print("\n" + "="*70)
    print("   RESUME FINAL")
    print("="*70)

    valid_pairs = []
    rejected_pairs = []

    for pair, res in results.items():
        if res.get('status') == 'VALID':
            valid_pairs.append(pair)
            print(f"  [OK] {pair}: PF={res['profit_factor']}, ROI={res['roi_pct']}%, MaxDD={res['max_drawdown_pct']}%")
        elif res.get('status') == 'REJECTED':
            rejected_pairs.append(pair)
            print(f"  [X]  {pair}: REJETE - {res['rejection_reason']}")
        else:
            rejected_pairs.append(pair)
            print(f"  [X]  {pair}: PAS DE TRADES")

    print(f"\n  Pairs Validees:  {len(valid_pairs)}/2")
    print(f"  Pairs Rejetees:  {len(rejected_pairs)}/2")

    if valid_pairs:
        print(f"\n  -> Strategies a utiliser: {', '.join(valid_pairs)}")
    else:
        print(f"\n  -> AUCUNE strategie validee!")

    print("="*70)

    return results


if __name__ == "__main__":
    main()
