#!/usr/bin/env python3
"""
EUR/JPY Strategy Analysis by Market Regime
===========================================
Detailed backtest of MACD+Stochastic strategy under different market conditions.

Objective: Understand when the strategy works best and when to avoid trading.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
PAIR = 'EURJPY'
SYMBOL = 'EURJPY=X'
CAPITAL = 10000
LOT_SIZE = 0.15
PIP_VALUE = 0.01  # JPY pair
PIP_VALUE_USD = 1.35  # For 0.15 lot

# Strategy config (from optimization)
STRATEGY_CONFIG = {
    'name': 'MACD+Stochastic',
    'rr': 1.2,
    'adx_min': 15,
    'rsi_low': 25,
    'rsi_high': 75,
    'stoch_oversold': 30,
    'stoch_overbought': 70,
    'sl_mult': 1.5,
}

# =============================================================================
# DATA DOWNLOAD & INDICATORS
# =============================================================================
def download_data(symbol: str) -> pd.DataFrame:
    """Download historical data."""
    print(f"Telechargement {symbol}...")
    ticker = yf.Ticker(symbol)
    df = ticker.history(period="2y", interval="1h")
    print(f"-> {len(df)} barres chargees")
    return df

def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators."""
    df = df.copy()

    # EMAs
    df['ema8'] = df['Close'].ewm(span=8).mean()
    df['ema21'] = df['Close'].ewm(span=21).mean()
    df['ema50'] = df['Close'].ewm(span=50).mean()
    df['ema200'] = df['Close'].ewm(span=200).mean()

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Stochastic
    low_14 = df['Low'].rolling(14).min()
    high_14 = df['High'].rolling(14).max()
    df['stoch_k'] = 100 * (df['Close'] - low_14) / (high_14 - low_14)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    # ATR & ADX
    df['tr'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(14).mean()

    plus_dm = df['High'].diff()
    minus_dm = -df['Low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr14 = df['tr'].rolling(14).mean()
    df['plus_di'] = 100 * (plus_dm.rolling(14).mean() / atr14)
    df['minus_di'] = 100 * (minus_dm.rolling(14).mean() / atr14)
    dx = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = dx.rolling(14).mean()

    # Bollinger Bands
    df['bb_mid'] = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_upper'] = df['bb_mid'] + 2 * bb_std
    df['bb_lower'] = df['bb_mid'] - 2 * bb_std
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid'] * 100

    # Volatility metrics
    df['atr_sma'] = df['atr'].rolling(20).mean()
    df['volatility_ratio'] = df['atr'] / df['atr_sma']
    df['bb_width_sma'] = df['bb_width'].rolling(20).mean()

    # Price position relative to EMAs
    df['above_ema50'] = df['Close'] > df['ema50']
    df['above_ema200'] = df['Close'] > df['ema200']

    # Trend strength
    df['trend_strength'] = abs(df['plus_di'] - df['minus_di'])

    return df

def classify_regime(row) -> str:
    """Classify market regime for a single bar."""
    adx = row['adx']
    volatility_ratio = row['volatility_ratio']
    plus_di = row['plus_di']
    minus_di = row['minus_di']
    bb_width = row['bb_width']
    bb_width_sma = row['bb_width_sma']

    # High volatility
    if volatility_ratio > 1.5 or (bb_width > bb_width_sma * 1.5):
        return 'HIGH_VOLATILITY'

    # Low volatility / Squeeze
    if volatility_ratio < 0.7 or (bb_width < bb_width_sma * 0.5):
        return 'LOW_VOLATILITY'

    # Strong trend
    if adx > 40:
        if plus_di > minus_di:
            return 'STRONG_TREND_UP'
        else:
            return 'STRONG_TREND_DOWN'

    # Normal trend
    if adx > 25:
        if plus_di > minus_di:
            return 'TREND_UP'
        else:
            return 'TREND_DOWN'

    # Ranging / Consolidation
    if adx < 20:
        return 'RANGING'

    return 'NORMAL'

def classify_session(timestamp) -> str:
    """Classify trading session based on hour (UTC)."""
    hour = timestamp.hour
    if 0 <= hour < 8:
        return 'ASIAN'
    elif 8 <= hour < 16:
        return 'LONDON'
    else:
        return 'NEW_YORK'

def classify_day(timestamp) -> str:
    """Classify day of week."""
    day = timestamp.dayofweek
    if day == 0:
        return 'MONDAY'
    elif day == 4:
        return 'FRIDAY'
    elif day in [5, 6]:
        return 'WEEKEND'
    else:
        return 'MID_WEEK'

# =============================================================================
# STRATEGY IMPLEMENTATION
# =============================================================================
def check_signal(row, prev_row, config) -> str:
    """Check for MACD+Stochastic signal."""
    # MACD crossover
    macd_cross_up = prev_row['macd'] <= prev_row['macd_signal'] and row['macd'] > row['macd_signal']
    macd_cross_down = prev_row['macd'] >= prev_row['macd_signal'] and row['macd'] < row['macd_signal']

    # Stochastic conditions
    stoch_oversold = row['stoch_k'] < config['stoch_oversold']
    stoch_overbought = row['stoch_k'] > config['stoch_overbought']

    # RSI filter
    rsi_ok = config['rsi_low'] <= row['rsi'] <= config['rsi_high']

    # ADX filter
    adx_ok = row['adx'] >= config['adx_min']

    if macd_cross_up and stoch_oversold and rsi_ok and adx_ok:
        return 'BUY'
    elif macd_cross_down and stoch_overbought and rsi_ok and adx_ok:
        return 'SELL'

    return None

# =============================================================================
# BACKTEST BY REGIME
# =============================================================================
def backtest_by_regime(df: pd.DataFrame, config: dict) -> dict:
    """Run backtest and track results by market regime."""

    # Add regime classification
    df['regime'] = df.apply(classify_regime, axis=1)
    df['session'] = df.index.map(classify_session)
    df['day_type'] = df.index.map(classify_day)

    # Results storage
    results_by_regime = defaultdict(lambda: {'trades': [], 'signals': 0})
    results_by_session = defaultdict(lambda: {'trades': [], 'signals': 0})
    results_by_day = defaultdict(lambda: {'trades': [], 'signals': 0})
    results_by_trend_dir = defaultdict(lambda: {'trades': [], 'signals': 0})

    all_trades = []
    position = None

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]

        # Skip if indicators not ready
        if pd.isna(current['adx']) or pd.isna(current['atr']) or pd.isna(current['stoch_k']):
            continue

        regime = current['regime']
        session = current['session']
        day_type = current['day_type']
        trend_dir = 'BULLISH' if current['plus_di'] > current['minus_di'] else 'BEARISH'

        # Check for exit if in position
        if position:
            exit_price = None
            exit_type = None

            if position['direction'] == 'BUY':
                if current['Low'] <= position['sl']:
                    exit_price = position['sl']
                    exit_type = 'SL'
                elif current['High'] >= position['tp']:
                    exit_price = position['tp']
                    exit_type = 'TP'
            else:  # SELL
                if current['High'] >= position['sl']:
                    exit_price = position['sl']
                    exit_type = 'SL'
                elif current['Low'] <= position['tp']:
                    exit_price = position['tp']
                    exit_type = 'TP'

            if exit_price:
                if position['direction'] == 'BUY':
                    pips = (exit_price - position['entry']) / PIP_VALUE
                else:
                    pips = (position['entry'] - exit_price) / PIP_VALUE

                pnl = pips * PIP_VALUE_USD

                trade = {
                    'entry_date': position['entry_date'],
                    'exit_date': df.index[i],
                    'direction': position['direction'],
                    'entry': position['entry'],
                    'exit': exit_price,
                    'pips': pips,
                    'pnl': pnl,
                    'result': exit_type,
                    'regime': position['regime'],
                    'session': position['session'],
                    'day_type': position['day_type'],
                    'trend_dir': position['trend_dir'],
                    'adx': position['adx'],
                    'volatility': position['volatility'],
                    'stoch_k': position['stoch_k'],
                }

                all_trades.append(trade)
                results_by_regime[position['regime']]['trades'].append(trade)
                results_by_session[position['session']]['trades'].append(trade)
                results_by_day[position['day_type']]['trades'].append(trade)
                results_by_trend_dir[position['trend_dir']]['trades'].append(trade)

                position = None

        # Check for new signal if not in position
        if not position:
            signal = check_signal(current, prev, config)

            if signal:
                results_by_regime[regime]['signals'] += 1
                results_by_session[session]['signals'] += 1
                results_by_day[day_type]['signals'] += 1
                results_by_trend_dir[trend_dir]['signals'] += 1

                entry = current['Close']
                atr = current['atr']

                if signal == 'BUY':
                    sl = entry - config['sl_mult'] * atr
                    tp = entry + config['rr'] * config['sl_mult'] * atr
                else:
                    sl = entry + config['sl_mult'] * atr
                    tp = entry - config['rr'] * config['sl_mult'] * atr

                position = {
                    'direction': signal,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'entry_date': df.index[i],
                    'regime': regime,
                    'session': session,
                    'day_type': day_type,
                    'trend_dir': trend_dir,
                    'adx': current['adx'],
                    'volatility': current['volatility_ratio'],
                    'stoch_k': current['stoch_k'],
                }

    return {
        'all_trades': all_trades,
        'by_regime': dict(results_by_regime),
        'by_session': dict(results_by_session),
        'by_day': dict(results_by_day),
        'by_trend_dir': dict(results_by_trend_dir),
    }

def calculate_metrics(trades: list) -> dict:
    """Calculate performance metrics for a list of trades."""
    if not trades:
        return {
            'trades': 0, 'wins': 0, 'losses': 0,
            'win_rate': 0, 'pf': 0, 'pnl': 0,
            'avg_win': 0, 'avg_loss': 0, 'expectancy': 0
        }

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    total_wins = sum(t['pnl'] for t in wins) if wins else 0
    total_losses = abs(sum(t['pnl'] for t in losses)) if losses else 0.0001

    pf = total_wins / total_losses if total_losses > 0 else 0
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    total_pnl = sum(t['pnl'] for t in trades)

    avg_win = total_wins / len(wins) if wins else 0
    avg_loss = total_losses / len(losses) if losses else 0
    expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)

    return {
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'pf': pf,
        'pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'expectancy': expectancy,
    }

# =============================================================================
# REPORT GENERATION
# =============================================================================
def generate_report(results: dict):
    """Generate detailed analysis report."""

    print("\n" + "=" * 80)
    print("   RAPPORT D'ANALYSE EUR/JPY - STRATEGIE MACD+STOCHASTIC")
    print("   Comportement par Regime de Marche")
    print("=" * 80)

    all_metrics = calculate_metrics(results['all_trades'])

    print(f"\n{'='*80}")
    print("1. PERFORMANCE GLOBALE")
    print("=" * 80)
    print(f"   Trades: {all_metrics['trades']}")
    print(f"   Win Rate: {all_metrics['win_rate']:.1f}%")
    print(f"   Profit Factor: {all_metrics['pf']:.2f}")
    print(f"   PnL Total: ${all_metrics['pnl']:+.2f}")
    print(f"   Expectancy: ${all_metrics['expectancy']:+.2f}/trade")

    # By Regime Analysis
    print(f"\n{'='*80}")
    print("2. ANALYSE PAR REGIME DE MARCHE")
    print("=" * 80)
    print(f"\n{'Regime':<20} {'Trades':>7} {'WR%':>7} {'PF':>7} {'PnL':>12} {'Verdict':>12}")
    print("-" * 70)

    regime_analysis = []
    for regime, data in sorted(results['by_regime'].items()):
        metrics = calculate_metrics(data['trades'])
        verdict = "[OK] TRADER" if metrics['pf'] >= 1.0 and metrics['trades'] >= 3 else "[X] EVITER"
        if metrics['trades'] < 3:
            verdict = "[?] PEU DE DATA"

        regime_analysis.append({
            'regime': regime,
            'metrics': metrics,
            'verdict': verdict
        })

        print(f"{regime:<20} {metrics['trades']:>7} {metrics['win_rate']:>6.1f}% {metrics['pf']:>6.2f} ${metrics['pnl']:>+10.2f} {verdict:>12}")

    # By Session Analysis
    print(f"\n{'='*80}")
    print("3. ANALYSE PAR SESSION")
    print("=" * 80)
    print(f"\n{'Session':<15} {'Trades':>7} {'WR%':>7} {'PF':>7} {'PnL':>12} {'Verdict':>12}")
    print("-" * 65)

    for session, data in sorted(results['by_session'].items()):
        metrics = calculate_metrics(data['trades'])
        verdict = "[OK]" if metrics['pf'] >= 1.0 else "[X]"
        print(f"{session:<15} {metrics['trades']:>7} {metrics['win_rate']:>6.1f}% {metrics['pf']:>6.2f} ${metrics['pnl']:>+10.2f} {verdict:>12}")

    # By Day Analysis
    print(f"\n{'='*80}")
    print("4. ANALYSE PAR JOUR")
    print("=" * 80)
    print(f"\n{'Jour':<15} {'Trades':>7} {'WR%':>7} {'PF':>7} {'PnL':>12} {'Verdict':>12}")
    print("-" * 65)

    for day, data in sorted(results['by_day'].items()):
        metrics = calculate_metrics(data['trades'])
        verdict = "[OK]" if metrics['pf'] >= 1.0 else "[X]"
        print(f"{day:<15} {metrics['trades']:>7} {metrics['win_rate']:>6.1f}% {metrics['pf']:>6.2f} ${metrics['pnl']:>+10.2f} {verdict:>12}")

    # By Trend Direction
    print(f"\n{'='*80}")
    print("5. ANALYSE PAR DIRECTION DU TREND")
    print("=" * 80)
    print(f"\n{'Direction':<15} {'Trades':>7} {'WR%':>7} {'PF':>7} {'PnL':>12} {'Verdict':>12}")
    print("-" * 65)

    for direction, data in sorted(results['by_trend_dir'].items()):
        metrics = calculate_metrics(data['trades'])
        verdict = "[OK]" if metrics['pf'] >= 1.0 else "[X]"
        print(f"{direction:<15} {metrics['trades']:>7} {metrics['win_rate']:>6.1f}% {metrics['pf']:>6.2f} ${metrics['pnl']:>+10.2f} {verdict:>12}")

    # Signal vs Direction Analysis
    print(f"\n{'='*80}")
    print("6. ANALYSE SIGNAL vs TREND")
    print("=" * 80)

    # Check if signal aligns with trend
    with_trend = [t for t in results['all_trades']
                  if (t['direction'] == 'BUY' and t['trend_dir'] == 'BULLISH') or
                     (t['direction'] == 'SELL' and t['trend_dir'] == 'BEARISH')]
    against_trend = [t for t in results['all_trades']
                     if (t['direction'] == 'BUY' and t['trend_dir'] == 'BEARISH') or
                        (t['direction'] == 'SELL' and t['trend_dir'] == 'BULLISH')]

    with_metrics = calculate_metrics(with_trend)
    against_metrics = calculate_metrics(against_trend)

    print(f"\n{'Type':<20} {'Trades':>7} {'WR%':>7} {'PF':>7} {'PnL':>12}")
    print("-" * 60)
    print(f"{'AVEC LE TREND':<20} {with_metrics['trades']:>7} {with_metrics['win_rate']:>6.1f}% {with_metrics['pf']:>6.2f} ${with_metrics['pnl']:>+10.2f}")
    print(f"{'CONTRE LE TREND':<20} {against_metrics['trades']:>7} {against_metrics['win_rate']:>6.1f}% {against_metrics['pf']:>6.2f} ${against_metrics['pnl']:>+10.2f}")

    # ADX Range Analysis
    print(f"\n{'='*80}")
    print("7. ANALYSE PAR NIVEAU ADX")
    print("=" * 80)

    adx_ranges = [
        ('ADX < 20', lambda t: t['adx'] < 20),
        ('ADX 20-25', lambda t: 20 <= t['adx'] < 25),
        ('ADX 25-30', lambda t: 25 <= t['adx'] < 30),
        ('ADX 30-40', lambda t: 30 <= t['adx'] < 40),
        ('ADX > 40', lambda t: t['adx'] >= 40),
    ]

    print(f"\n{'ADX Range':<15} {'Trades':>7} {'WR%':>7} {'PF':>7} {'PnL':>12} {'Verdict':>12}")
    print("-" * 65)

    for name, filter_func in adx_ranges:
        filtered = [t for t in results['all_trades'] if filter_func(t)]
        metrics = calculate_metrics(filtered)
        verdict = "[OK]" if metrics['pf'] >= 1.0 and metrics['trades'] >= 3 else "[X]"
        if metrics['trades'] < 3:
            verdict = "[?]"
        print(f"{name:<15} {metrics['trades']:>7} {metrics['win_rate']:>6.1f}% {metrics['pf']:>6.2f} ${metrics['pnl']:>+10.2f} {verdict:>12}")

    # Volatility Analysis
    print(f"\n{'='*80}")
    print("8. ANALYSE PAR VOLATILITE")
    print("=" * 80)

    vol_ranges = [
        ('Tres basse (<0.7)', lambda t: t['volatility'] < 0.7),
        ('Basse (0.7-0.9)', lambda t: 0.7 <= t['volatility'] < 0.9),
        ('Normale (0.9-1.1)', lambda t: 0.9 <= t['volatility'] < 1.1),
        ('Haute (1.1-1.5)', lambda t: 1.1 <= t['volatility'] < 1.5),
        ('Tres haute (>1.5)', lambda t: t['volatility'] >= 1.5),
    ]

    print(f"\n{'Volatilite':<20} {'Trades':>7} {'WR%':>7} {'PF':>7} {'PnL':>12} {'Verdict':>12}")
    print("-" * 70)

    for name, filter_func in vol_ranges:
        filtered = [t for t in results['all_trades'] if filter_func(t)]
        metrics = calculate_metrics(filtered)
        verdict = "[OK]" if metrics['pf'] >= 1.0 and metrics['trades'] >= 3 else "[X]"
        if metrics['trades'] < 3:
            verdict = "[?]"
        print(f"{name:<20} {metrics['trades']:>7} {metrics['win_rate']:>6.1f}% {metrics['pf']:>6.2f} ${metrics['pnl']:>+10.2f} {verdict:>12}")

    # Recommendations
    print(f"\n{'='*80}")
    print("9. RECOMMANDATIONS")
    print("=" * 80)

    print("\n[OK] QUAND TRADER:")
    for ra in regime_analysis:
        if ra['verdict'] == "[OK] TRADER":
            print(f"   + {ra['regime']}: PF={ra['metrics']['pf']:.2f}, WR={ra['metrics']['win_rate']:.0f}%")

    if with_metrics['pf'] > against_metrics['pf']:
        print(f"   + Trader AVEC le trend (PF={with_metrics['pf']:.2f})")

    print("\n[X] QUAND EVITER:")
    for ra in regime_analysis:
        if ra['verdict'] == "[X] EVITER":
            print(f"   - {ra['regime']}: PF={ra['metrics']['pf']:.2f}, WR={ra['metrics']['win_rate']:.0f}%")

    if against_metrics['pf'] < 1.0:
        print(f"   - Eviter trades CONTRE le trend (PF={against_metrics['pf']:.2f})")

    print(f"\n{'='*80}")
    print("10. MODIFICATIONS SUGGEREES")
    print("=" * 80)

    suggestions = []

    # Analyze what works
    best_regime = max(regime_analysis, key=lambda x: x['metrics']['pf'] if x['metrics']['trades'] >= 3 else 0)
    worst_regime = min(regime_analysis, key=lambda x: x['metrics']['pf'] if x['metrics']['trades'] >= 3 else 999)

    if best_regime['metrics']['pf'] >= 1.0:
        suggestions.append(f"   1. Augmenter position size en regime {best_regime['regime']} (PF={best_regime['metrics']['pf']:.2f})")

    if worst_regime['metrics']['pf'] < 1.0 and worst_regime['metrics']['trades'] >= 3:
        suggestions.append(f"   2. BLOQUER les signaux en regime {worst_regime['regime']} (PF={worst_regime['metrics']['pf']:.2f})")

    if with_metrics['pf'] > against_metrics['pf'] * 1.2:
        suggestions.append("   3. Ajouter filtre: trader UNIQUEMENT dans le sens du trend")

    # Session suggestion
    session_metrics = {s: calculate_metrics(d['trades']) for s, d in results['by_session'].items()}
    best_session = max(session_metrics.items(), key=lambda x: x[1]['pf'] if x[1]['trades'] >= 3 else 0)
    worst_session = min(session_metrics.items(), key=lambda x: x[1]['pf'] if x[1]['trades'] >= 3 else 999)

    if best_session[1]['pf'] >= 1.2 and best_session[1]['trades'] >= 5:
        suggestions.append(f"   4. Privilegier session {best_session[0]} (PF={best_session[1]['pf']:.2f})")

    if worst_session[1]['pf'] < 0.9 and worst_session[1]['trades'] >= 5:
        suggestions.append(f"   5. Eviter session {worst_session[0]} (PF={worst_session[1]['pf']:.2f})")

    for s in suggestions:
        print(s)

    if not suggestions:
        print("   Strategie equilibree, pas de modification majeure necessaire.")

    print("\n" + "=" * 80)

    return regime_analysis

# =============================================================================
# MAIN
# =============================================================================
def main():
    # Download and prepare data
    df = download_data(SYMBOL)
    df = calculate_indicators(df)

    # Run backtest
    print("\nExecution du backtest par regime...")
    results = backtest_by_regime(df, STRATEGY_CONFIG)

    # Generate report
    regime_analysis = generate_report(results)

    # Save detailed results
    if results['all_trades']:
        trades_df = pd.DataFrame(results['all_trades'])
        filename = f"eurjpy_regime_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        trades_df.to_csv(filename, index=False)
        print(f"\nDetails des trades sauvegardes dans: {filename}")

    return results, regime_analysis

if __name__ == "__main__":
    results, analysis = main()
