#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BACKTEST DERNIER MOIS - Simulation Horaire
==========================================
Simule l'execution de: python main.py --pairs CADJPY,EURGBP --active-only
toutes les heures pendant le dernier mois.

Capital: $10,000
Lot Size: 0.15
Strategies:
- CADJPY: EMA Crossover (8/21/50), R:R=2.5, ADX>=25, RSI 35-65
- EURGBP: Stochastic Crossover, R:R=2.0, K<30 BUY, K>70 SELL
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - Exactement comme main.py
# =============================================================================
CAPITAL = 10000
LOT_SIZE = 0.15

# CADJPY - EMA Crossover
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
    'pip_value': 0.01,
    'pip_value_usd': 1.35,  # Pour 0.15 lot
}

# EURGBP - Stochastic Crossover
EURGBP_CONFIG = {
    'pair': 'EURGBP',
    'symbol': 'EURGBP=X',
    'strategy': 'STOCHASTIC_CROSSOVER',
    'stoch_period': 14,
    'stoch_smooth': 3,
    'oversold': 20,
    'overbought': 80,
    'zone_buffer': 10,
    'rr': 2.0,
    'sl_mult': 1.5,
    'pip_value': 0.0001,
    'pip_value_usd': 1.5,  # Pour 0.15 lot
}


def fetch_hourly_data(symbol: str, days: int = 30) -> Optional[pd.DataFrame]:
    """Fetch hourly data for the last N days."""
    print(f"  Fetching {days} days of H1 data for {symbol}...")

    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days + 5)  # Buffer

        df = yf.download(
            symbol,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval='1h',
            progress=False
        )

        if df is not None and not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            print(f"  -> {len(df)} hourly bars loaded")
            return df
        return None
    except Exception as e:
        print(f"  ERROR: {e}")
        return None


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


def calculate_ema_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
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

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    return df


def calculate_stochastic_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Calculate Stochastic strategy indicators."""
    df = df.copy()

    period = config['stoch_period']
    smooth = config['stoch_smooth']

    low_min = df['low'].rolling(window=period).min()
    high_max = df['high'].rolling(window=period).max()
    df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min + 0.0001)
    df['stoch_d'] = df['stoch_k'].rolling(window=smooth).mean()

    # ATR
    df['atr'] = calculate_atr(df)

    return df


def calculate_ema_score(current: pd.Series, prev: pd.Series, is_bullish: bool, config: dict) -> int:
    """Calculate EMA strategy score (0-8)."""
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


def backtest_pair(df: pd.DataFrame, config: dict, is_stochastic: bool = False) -> Dict:
    """
    Backtest a single pair with hourly checks.
    Returns detailed trade history and statistics.
    """
    if is_stochastic:
        df = calculate_stochastic_indicators(df, config)
    else:
        df = calculate_ema_indicators(df, config)

    trades = []
    daily_pnl = defaultdict(float)
    hourly_signals = []

    capital = CAPITAL
    position = None
    pip_value_usd = config['pip_value_usd']

    # For stochastic
    if is_stochastic:
        buy_zone = config['oversold'] + config['zone_buffer']  # K < 30
        sell_zone = config['overbought'] - config['zone_buffer']  # K > 70

    min_bars = 60 if not is_stochastic else 20

    for i in range(min_bars, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        timestamp = df.index[i]
        day_key = timestamp.strftime('%Y-%m-%d')
        hour_key = timestamp.strftime('%Y-%m-%d %H:00')

        # Skip if indicators not ready
        if is_stochastic:
            if pd.isna(current['stoch_k']) or pd.isna(current['atr']):
                continue
        else:
            if pd.isna(current['adx']) or pd.isna(current['atr']):
                continue

        # Check for exit if in position
        if position:
            exit_price = None
            exit_type = None

            if position['direction'] == 'BUY':
                if current['low'] <= position['sl']:
                    exit_price = position['sl']
                    exit_type = 'SL'
                elif current['high'] >= position['tp']:
                    exit_price = position['tp']
                    exit_type = 'TP'
            else:  # SELL
                if current['high'] >= position['sl']:
                    exit_price = position['sl']
                    exit_type = 'SL'
                elif current['low'] <= position['tp']:
                    exit_price = position['tp']
                    exit_type = 'TP'

            if exit_price:
                if position['direction'] == 'BUY':
                    pips = (exit_price - position['entry']) / config['pip_value']
                else:
                    pips = (position['entry'] - exit_price) / config['pip_value']

                pnl = pips * pip_value_usd
                capital += pnl
                daily_pnl[day_key] += pnl

                trades.append({
                    'pair': config['pair'],
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'direction': position['direction'],
                    'entry': position['entry'],
                    'sl': position['sl'],
                    'tp': position['tp'],
                    'exit': exit_price,
                    'pips': round(pips, 1),
                    'pnl': round(pnl, 2),
                    'result': exit_type,
                    'capital_after': round(capital, 2)
                })
                position = None

        # Check for new entry if no position (simulating hourly scan)
        if position is None:
            atr = current['atr']
            signal = None

            if is_stochastic:
                # Stochastic signals
                k_crossed_up = prev['stoch_k'] <= prev['stoch_d'] and current['stoch_k'] > current['stoch_d']
                k_crossed_down = prev['stoch_k'] >= prev['stoch_d'] and current['stoch_k'] < current['stoch_d']

                if k_crossed_up and current['stoch_k'] < buy_zone:
                    signal = 'BUY'
                elif k_crossed_down and current['stoch_k'] > sell_zone:
                    signal = 'SELL'
            else:
                # EMA signals
                ema_cross_up = prev['ema_fast'] <= prev['ema_slow'] and current['ema_fast'] > current['ema_slow']
                ema_cross_down = prev['ema_fast'] >= prev['ema_slow'] and current['ema_fast'] < current['ema_slow']
                trend_up = current['close'] > current['ema_trend']
                trend_down = current['close'] < current['ema_trend']
                rsi_ok = config['rsi_low'] < current['rsi'] < config['rsi_high']
                adx_ok = current['adx'] >= config['adx_min']

                if ema_cross_up and trend_up and rsi_ok and adx_ok:
                    score = calculate_ema_score(current, prev, True, config)
                    if score >= config['min_score']:
                        signal = 'BUY'
                elif ema_cross_down and trend_down and rsi_ok and adx_ok:
                    score = calculate_ema_score(current, prev, False, config)
                    if score >= config['min_score']:
                        signal = 'SELL'

            # Execute signal
            if signal:
                entry = current['close']
                if signal == 'BUY':
                    sl = entry - (atr * config['sl_mult'])
                    tp = entry + (atr * config['sl_mult'] * config['rr'])
                else:
                    sl = entry + (atr * config['sl_mult'])
                    tp = entry - (atr * config['sl_mult'] * config['rr'])

                position = {
                    'direction': signal,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'entry_time': timestamp
                }

                hourly_signals.append({
                    'time': timestamp.strftime('%Y-%m-%d %H:%M'),
                    'pair': config['pair'],
                    'signal': signal,
                    'entry': round(entry, 5),
                    'sl': round(sl, 5),
                    'tp': round(tp, 5)
                })

    # Close any open position at the end
    if position:
        exit_price = df.iloc[-1]['close']
        if position['direction'] == 'BUY':
            pips = (exit_price - position['entry']) / config['pip_value']
        else:
            pips = (position['entry'] - exit_price) / config['pip_value']

        pnl = pips * pip_value_usd
        capital += pnl
        day_key = df.index[-1].strftime('%Y-%m-%d')
        daily_pnl[day_key] += pnl

        trades.append({
            'pair': config['pair'],
            'entry_time': position['entry_time'],
            'exit_time': df.index[-1],
            'direction': position['direction'],
            'entry': position['entry'],
            'sl': position['sl'],
            'tp': position['tp'],
            'exit': exit_price,
            'pips': round(pips, 1),
            'pnl': round(pnl, 2),
            'result': 'OPEN_CLOSE',
            'capital_after': round(capital, 2)
        })

    return {
        'pair': config['pair'],
        'trades': trades,
        'daily_pnl': dict(daily_pnl),
        'signals': hourly_signals,
        'final_capital': capital
    }


def print_trade_history(trades: List[Dict], pair: str):
    """Print detailed trade history."""
    print(f"\n  {'='*70}")
    print(f"  HISTORIQUE DES TRADES - {pair}")
    print(f"  {'='*70}")

    if not trades:
        print("  Aucun trade")
        return

    print(f"  {'#':<3} {'Date Entry':<12} {'Dir':<5} {'Entry':<10} {'Exit':<10} {'Pips':<8} {'P&L':<10} {'Result':<6}")
    print(f"  {'-'*70}")

    for i, t in enumerate(trades, 1):
        entry_date = t['entry_time'].strftime('%m/%d %H:%M') if hasattr(t['entry_time'], 'strftime') else str(t['entry_time'])[:12]
        print(f"  {i:<3} {entry_date:<12} {t['direction']:<5} {t['entry']:<10.5f} {t['exit']:<10.5f} {t['pips']:<8.1f} ${t['pnl']:<9.2f} {t['result']:<6}")


def main():
    print("\n" + "="*75)
    print("   BACKTEST DERNIER MOIS - SIMULATION HORAIRE")
    print("   Command: python main.py --pairs CADJPY,EURGBP --active-only")
    print("   Capital: $10,000 | Lot: 0.15 | Periode: 30 jours")
    print("="*75)

    all_trades = []
    all_daily_pnl = defaultdict(float)
    results = {}

    # Backtest CADJPY
    print(f"\n[1/2] CADJPY - EMA Crossover Strategy")
    print("-" * 50)
    df_cadjpy = fetch_hourly_data(CADJPY_CONFIG['symbol'], days=35)

    if df_cadjpy is not None and len(df_cadjpy) > 100:
        result_cadjpy = backtest_pair(df_cadjpy, CADJPY_CONFIG, is_stochastic=False)
        results['CADJPY'] = result_cadjpy
        all_trades.extend(result_cadjpy['trades'])
        for day, pnl in result_cadjpy['daily_pnl'].items():
            all_daily_pnl[day] += pnl

        print_trade_history(result_cadjpy['trades'], 'CADJPY')
    else:
        print("  ERREUR: Donnees insuffisantes pour CADJPY")

    # Backtest EURGBP
    print(f"\n[2/2] EURGBP - Stochastic Crossover Strategy")
    print("-" * 50)
    df_eurgbp = fetch_hourly_data(EURGBP_CONFIG['symbol'], days=35)

    if df_eurgbp is not None and len(df_eurgbp) > 100:
        result_eurgbp = backtest_pair(df_eurgbp, EURGBP_CONFIG, is_stochastic=True)
        results['EURGBP'] = result_eurgbp
        all_trades.extend(result_eurgbp['trades'])
        for day, pnl in result_eurgbp['daily_pnl'].items():
            all_daily_pnl[day] += pnl

        print_trade_history(result_eurgbp['trades'], 'EURGBP')
    else:
        print("  ERREUR: Donnees insuffisantes pour EURGBP")

    # Combined Statistics
    print("\n" + "="*75)
    print("   RESUME GLOBAL - DERNIER MOIS")
    print("="*75)

    if all_trades:
        total_trades = len(all_trades)
        winners = [t for t in all_trades if t['pnl'] > 0]
        losers = [t for t in all_trades if t['pnl'] <= 0]

        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

        total_profit = sum(t['pnl'] for t in winners)
        total_loss = abs(sum(t['pnl'] for t in losers))
        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        total_pnl = sum(t['pnl'] for t in all_trades)
        final_capital = CAPITAL + total_pnl

        # Par paire
        print(f"\n  PAR PAIRE:")
        for pair, res in results.items():
            pair_trades = res['trades']
            pair_pnl = sum(t['pnl'] for t in pair_trades)
            pair_wins = len([t for t in pair_trades if t['pnl'] > 0])
            pair_wr = pair_wins / len(pair_trades) * 100 if pair_trades else 0
            print(f"    {pair}: {len(pair_trades)} trades, WR={pair_wr:.1f}%, P&L=${pair_pnl:+.2f}")

        # Global
        print(f"\n  PERFORMANCE GLOBALE:")
        print(f"    Capital Initial:  ${CAPITAL:,.2f}")
        print(f"    Capital Final:    ${final_capital:,.2f}")
        print(f"    P&L Total:        ${total_pnl:+,.2f}")
        print(f"    ROI:              {(total_pnl/CAPITAL)*100:+.2f}%")
        print(f"    Trades:           {total_trades}")
        print(f"    Gagnants:         {len(winners)}")
        print(f"    Perdants:         {len(losers)}")
        print(f"    Win Rate:         {win_rate:.1f}%")
        print(f"    Profit Factor:    {profit_factor:.2f}")

        # Daily P&L
        print(f"\n  P&L JOURNALIER:")
        sorted_days = sorted(all_daily_pnl.items())
        for day, pnl in sorted_days:
            status = "[OK]" if pnl >= 0 else "[--]" if pnl > -500 else "[!!]"
            print(f"    {day}: ${pnl:+8.2f} {status}")

        # Worst day check
        worst_day_pnl = min(all_daily_pnl.values()) if all_daily_pnl else 0
        worst_day = min(all_daily_pnl, key=all_daily_pnl.get) if all_daily_pnl else 'N/A'
        days_over_500 = len([d for d, p in all_daily_pnl.items() if p < -500])

        print(f"\n  RISK:")
        print(f"    Worst Day:        ${worst_day_pnl:.2f} ({worst_day})")
        print(f"    Days > -$500:     {days_over_500}")

        if days_over_500 == 0:
            print(f"\n  [OK] STRATEGIE VALIDEE - Aucun jour n'a depasse -$500")
        else:
            print(f"\n  [!!] ATTENTION - {days_over_500} jour(s) ont depasse -$500")

    else:
        print("  Aucun trade genere sur la periode")

    print("\n" + "="*75)

    return results


if __name__ == "__main__":
    main()
