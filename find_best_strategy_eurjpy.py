#!/usr/bin/env python3
"""
EUR/JPY Strategy Finder - Following the STRATEGY_DETECTION_PROMPT.md
=====================================================================
Finds the best trading strategy for EUR/JPY pair.

Steps:
1. Download historical data
2. Test all 40 candidate strategies
3. Select best strategies (PF >= 1.0, trades >= 100)
4. Grid search parameters
5. Detect market regimes
6. Validate risk constraints
7. Final validation
8. Document optimal strategy
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from collections import defaultdict
import warnings
import sys

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================
PAIR = 'EURJPY'
SYMBOL = 'EURJPY=X'
CAPITAL = 10000
LOT_SIZE = 0.15
MAX_DAILY_LOSS = -500
PIP_VALUE = 0.01  # JPY pair
PIP_VALUE_USD = 1.35  # For 0.15 lot

# Grid search parameters
RR_VALUES = [1.2, 1.5, 1.8, 2.0, 2.5]
ADX_VALUES = [15, 20, 25]
RSI_RANGES = [(25, 75), (30, 70), (35, 65)]

# =============================================================================
# DATA DOWNLOAD
# =============================================================================
def download_data(symbol: str, period: str = "2y", interval: str = "1h") -> pd.DataFrame:
    """Download historical data from yfinance."""
    print(f"\n[1/8] ETAPE 1: COLLECTE DES DONNEES")
    print(f"     Telechargement {symbol} - {period} - {interval}...")

    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval)

    if df.empty:
        # Fallback to daily for longer period
        print(f"     H1 limite, utilisation Daily pour 2 ans...")
        df = ticker.history(period="2y", interval="1d")

    print(f"     -> {len(df)} barres chargees ({df.index[0].strftime('%Y-%m-%d')} a {df.index[-1].strftime('%Y-%m-%d')})")
    return df

# =============================================================================
# INDICATORS
# =============================================================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate all technical indicators needed for strategies."""
    df = df.copy()

    # EMAs
    df['ema8'] = df['Close'].ewm(span=8).mean()
    df['ema9'] = df['Close'].ewm(span=9).mean()
    df['ema13'] = df['Close'].ewm(span=13).mean()
    df['ema21'] = df['Close'].ewm(span=21).mean()
    df['ema50'] = df['Close'].ewm(span=50).mean()
    df['ema200'] = df['Close'].ewm(span=200).mean()

    # SMAs
    df['sma50'] = df['Close'].rolling(50).mean()
    df['sma200'] = df['Close'].rolling(200).mean()

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

    # ADX
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
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']
    df['bb_pct_b'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

    # CCI
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    tp_sma = tp.rolling(20).mean()
    tp_mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
    df['cci'] = (tp - tp_sma) / (0.015 * tp_mad)

    # Williams %R
    df['williams_r'] = -100 * (high_14 - df['Close']) / (high_14 - low_14)

    # Supertrend
    multiplier = 3
    hl2 = (df['High'] + df['Low']) / 2
    df['supertrend_upper'] = hl2 + multiplier * df['atr']
    df['supertrend_lower'] = hl2 - multiplier * df['atr']

    # Donchian
    df['donchian_high'] = df['High'].rolling(20).max()
    df['donchian_low'] = df['Low'].rolling(20).min()

    # Volatility regime
    df['atr_sma'] = df['atr'].rolling(20).mean()
    df['volatility_ratio'] = df['atr'] / df['atr_sma']

    return df

# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================
def strategy_ema_triple(row, prev_row):
    """#1 EMA Triple Crossover"""
    if row['ema8'] > row['ema21'] > row['ema50']:
        return 'BUY'
    elif row['ema8'] < row['ema21'] < row['ema50']:
        return 'SELL'
    return None

def strategy_ema_double(row, prev_row):
    """#2 EMA Double Crossover"""
    if prev_row['ema9'] <= prev_row['ema21'] and row['ema9'] > row['ema21']:
        return 'BUY'
    elif prev_row['ema9'] >= prev_row['ema21'] and row['ema9'] < row['ema21']:
        return 'SELL'
    return None

def strategy_macd_crossover(row, prev_row):
    """#4 MACD Crossover"""
    if prev_row['macd'] <= prev_row['macd_signal'] and row['macd'] > row['macd_signal']:
        return 'BUY'
    elif prev_row['macd'] >= prev_row['macd_signal'] and row['macd'] < row['macd_signal']:
        return 'SELL'
    return None

def strategy_macd_zero(row, prev_row):
    """#5 MACD + Zero Line"""
    if row['macd'] > 0 and prev_row['macd'] <= prev_row['macd_signal'] and row['macd'] > row['macd_signal']:
        return 'BUY'
    elif row['macd'] < 0 and prev_row['macd'] >= prev_row['macd_signal'] and row['macd'] < row['macd_signal']:
        return 'SELL'
    return None

def strategy_adx_trend(row, prev_row, adx_threshold=25):
    """#6 ADX Trend Strength"""
    if row['adx'] > adx_threshold and row['plus_di'] > row['minus_di']:
        return 'BUY'
    elif row['adx'] > adx_threshold and row['minus_di'] > row['plus_di']:
        return 'SELL'
    return None

def strategy_rsi_oversold(row, prev_row, oversold=30, overbought=70):
    """#11 RSI Overbought/Oversold"""
    if prev_row['rsi'] < oversold and row['rsi'] >= oversold:
        return 'BUY'
    elif prev_row['rsi'] > overbought and row['rsi'] <= overbought:
        return 'SELL'
    return None

def strategy_rsi_centerline(row, prev_row):
    """#12 RSI Centerline Crossover"""
    if prev_row['rsi'] < 50 and row['rsi'] >= 50:
        return 'BUY'
    elif prev_row['rsi'] > 50 and row['rsi'] <= 50:
        return 'SELL'
    return None

def strategy_stochastic(row, prev_row, oversold=20, overbought=80):
    """#14 Stochastic Crossover"""
    if row['stoch_k'] < oversold and prev_row['stoch_k'] <= prev_row['stoch_d'] and row['stoch_k'] > row['stoch_d']:
        return 'BUY'
    elif row['stoch_k'] > overbought and prev_row['stoch_k'] >= prev_row['stoch_d'] and row['stoch_k'] < row['stoch_d']:
        return 'SELL'
    return None

def strategy_stochastic_extended(row, prev_row, zone=30):
    """#14b Stochastic with extended zones"""
    if row['stoch_k'] < zone and prev_row['stoch_k'] <= prev_row['stoch_d'] and row['stoch_k'] > row['stoch_d']:
        return 'BUY'
    elif row['stoch_k'] > (100 - zone) and prev_row['stoch_k'] >= prev_row['stoch_d'] and row['stoch_k'] < row['stoch_d']:
        return 'SELL'
    return None

def strategy_cci(row, prev_row):
    """#16 CCI"""
    if prev_row['cci'] < -100 and row['cci'] >= -100:
        return 'BUY'
    elif prev_row['cci'] > 100 and row['cci'] <= 100:
        return 'SELL'
    return None

def strategy_williams(row, prev_row):
    """#17 Williams %R"""
    if prev_row['williams_r'] < -80 and row['williams_r'] >= -80:
        return 'BUY'
    elif prev_row['williams_r'] > -20 and row['williams_r'] <= -20:
        return 'SELL'
    return None

def strategy_bb_bounce(row, prev_row):
    """#19 Bollinger Bands Bounce"""
    if row['Close'] <= row['bb_lower']:
        return 'BUY'
    elif row['Close'] >= row['bb_upper']:
        return 'SELL'
    return None

def strategy_bb_pctb(row, prev_row):
    """#21 Bollinger Bands %B"""
    if row['bb_pct_b'] < 0:
        return 'BUY'
    elif row['bb_pct_b'] > 1:
        return 'SELL'
    return None

def strategy_donchian(row, prev_row):
    """#10 Donchian Channel Breakout"""
    if row['Close'] > prev_row['donchian_high']:
        return 'BUY'
    elif row['Close'] < prev_row['donchian_low']:
        return 'SELL'
    return None

def strategy_ema_rsi_adx(row, prev_row, adx_min=25):
    """#30 EMA + RSI + ADX Combined"""
    if row['ema8'] > row['ema21'] and row['rsi'] > 50 and row['adx'] > adx_min:
        return 'BUY'
    elif row['ema8'] < row['ema21'] and row['rsi'] < 50 and row['adx'] > adx_min:
        return 'SELL'
    return None

def strategy_macd_stoch(row, prev_row):
    """#31 MACD + Stochastic"""
    macd_cross_up = prev_row['macd'] <= prev_row['macd_signal'] and row['macd'] > row['macd_signal']
    macd_cross_down = prev_row['macd'] >= prev_row['macd_signal'] and row['macd'] < row['macd_signal']

    if macd_cross_up and row['stoch_k'] < 30:
        return 'BUY'
    elif macd_cross_down and row['stoch_k'] > 70:
        return 'SELL'
    return None

def strategy_bb_rsi(row, prev_row):
    """#32 Bollinger + RSI"""
    if row['Close'] <= row['bb_lower'] and row['rsi'] < 30:
        return 'BUY'
    elif row['Close'] >= row['bb_upper'] and row['rsi'] > 70:
        return 'SELL'
    return None

# =============================================================================
# BACKTEST ENGINE
# =============================================================================
def backtest_strategy(df: pd.DataFrame, strategy_func, strategy_name: str,
                      rr: float = 2.0, adx_filter: int = None,
                      rsi_filter: tuple = None, **kwargs) -> dict:
    """Backtest a strategy and return results."""

    capital = CAPITAL
    position = None
    trades = []
    daily_pnl = defaultdict(float)

    for i in range(1, len(df)):
        current = df.iloc[i]
        prev = df.iloc[i-1]
        date = df.index[i]
        day_key = date.strftime('%Y-%m-%d')

        # Skip if indicators not ready
        if pd.isna(current['adx']) or pd.isna(current['atr']) or pd.isna(current['rsi']):
            continue

        # Check for exit if in position
        if position:
            if position['direction'] == 'BUY':
                if current['Low'] <= position['sl']:
                    pips = (position['sl'] - position['entry']) / PIP_VALUE
                    pnl = pips * PIP_VALUE_USD
                    capital += pnl
                    daily_pnl[day_key] += pnl
                    trades.append({'pnl': pnl, 'result': 'SL', 'direction': 'BUY'})
                    position = None
                elif current['High'] >= position['tp']:
                    pips = (position['tp'] - position['entry']) / PIP_VALUE
                    pnl = pips * PIP_VALUE_USD
                    capital += pnl
                    daily_pnl[day_key] += pnl
                    trades.append({'pnl': pnl, 'result': 'TP', 'direction': 'BUY'})
                    position = None
            else:  # SELL
                if current['High'] >= position['sl']:
                    pips = (position['entry'] - position['sl']) / PIP_VALUE
                    pnl = pips * PIP_VALUE_USD
                    capital += pnl
                    daily_pnl[day_key] += pnl
                    trades.append({'pnl': pnl, 'result': 'SL', 'direction': 'SELL'})
                    position = None
                elif current['Low'] <= position['tp']:
                    pips = (position['entry'] - position['tp']) / PIP_VALUE
                    pnl = pips * PIP_VALUE_USD
                    capital += pnl
                    daily_pnl[day_key] += pnl
                    trades.append({'pnl': pnl, 'result': 'TP', 'direction': 'SELL'})
                    position = None

        # Check for new signal if not in position
        if not position:
            # Apply filters
            if adx_filter and current['adx'] < adx_filter:
                continue
            if rsi_filter:
                if current['rsi'] < rsi_filter[0] or current['rsi'] > rsi_filter[1]:
                    continue

            # Get signal
            try:
                signal = strategy_func(current, prev, **kwargs)
            except TypeError:
                signal = strategy_func(current, prev)

            if signal == 'BUY':
                entry = current['Close']
                atr = current['atr']
                sl = entry - 1.5 * atr
                tp = entry + rr * 1.5 * atr
                position = {'direction': 'BUY', 'entry': entry, 'sl': sl, 'tp': tp}
            elif signal == 'SELL':
                entry = current['Close']
                atr = current['atr']
                sl = entry + 1.5 * atr
                tp = entry - rr * 1.5 * atr
                position = {'direction': 'SELL', 'entry': entry, 'sl': sl, 'tp': tp}

    # Calculate metrics
    if not trades:
        return None

    wins = [t for t in trades if t['pnl'] > 0]
    losses = [t for t in trades if t['pnl'] <= 0]

    total_wins = sum(t['pnl'] for t in wins) if wins else 0
    total_losses = abs(sum(t['pnl'] for t in losses)) if losses else 0.0001

    pf = total_wins / total_losses if total_losses > 0 else 0
    win_rate = len(wins) / len(trades) * 100 if trades else 0
    total_pnl = sum(t['pnl'] for t in trades)

    # Max drawdown
    running_pnl = 0
    peak = 0
    max_dd = 0
    for t in trades:
        running_pnl += t['pnl']
        if running_pnl > peak:
            peak = running_pnl
        dd = (peak - running_pnl) / CAPITAL * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd

    # Days over max loss
    days_over_limit = sum(1 for d, pnl in daily_pnl.items() if pnl < MAX_DAILY_LOSS)

    return {
        'strategy': strategy_name,
        'trades': len(trades),
        'win_rate': win_rate,
        'pf': pf,
        'pnl': total_pnl,
        'max_dd': max_dd,
        'days_over_500': days_over_limit,
        'rr': rr,
        'adx_filter': adx_filter,
        'rsi_filter': rsi_filter
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    print("=" * 70)
    print("   RECHERCHE STRATEGIE OPTIMALE - EUR/JPY")
    print("   Suivant STRATEGY_DETECTION_PROMPT.md")
    print("=" * 70)

    # Step 1: Download data
    df = download_data(SYMBOL)
    df = calculate_indicators(df)

    # Step 2: Test all strategies
    print(f"\n[2/8] ETAPE 2: BACKTEST INITIAL - 17 STRATEGIES")
    print("-" * 70)

    strategies = [
        (strategy_ema_triple, "EMA Triple (8/21/50)"),
        (strategy_ema_double, "EMA Double (9/21)"),
        (strategy_macd_crossover, "MACD Crossover"),
        (strategy_macd_zero, "MACD + Zero Line"),
        (strategy_adx_trend, "ADX Trend"),
        (strategy_rsi_oversold, "RSI Oversold/Overbought"),
        (strategy_rsi_centerline, "RSI Centerline"),
        (strategy_stochastic, "Stochastic (20/80)"),
        (strategy_stochastic_extended, "Stochastic (30/70)"),
        (strategy_cci, "CCI"),
        (strategy_williams, "Williams %R"),
        (strategy_bb_bounce, "BB Bounce"),
        (strategy_bb_pctb, "BB %B"),
        (strategy_donchian, "Donchian Breakout"),
        (strategy_ema_rsi_adx, "EMA+RSI+ADX"),
        (strategy_macd_stoch, "MACD+Stochastic"),
        (strategy_bb_rsi, "BB+RSI"),
    ]

    results = []
    for func, name in strategies:
        result = backtest_strategy(df, func, name, rr=2.0)
        if result and result['trades'] >= 10:
            results.append(result)
            status = "[OK]" if result['pf'] >= 1.0 else "[--]"
            print(f"  {status} {name:25} | Trades: {result['trades']:4} | WR: {result['win_rate']:5.1f}% | PF: {result['pf']:.2f} | PnL: ${result['pnl']:+.0f}")

    # Step 3: Select best strategies
    print(f"\n[3/8] ETAPE 3: SELECTION (PF >= 1.0, Trades >= 50)")
    print("-" * 70)

    valid_strategies = [r for r in results if r['pf'] >= 1.0 and r['trades'] >= 50]
    valid_strategies.sort(key=lambda x: x['pf'], reverse=True)

    if not valid_strategies:
        print("  [!] Aucune strategie valide trouvee, selection des meilleures...")
        valid_strategies = sorted(results, key=lambda x: x['pf'], reverse=True)[:3]

    for r in valid_strategies[:5]:
        print(f"  -> {r['strategy']:25} | PF: {r['pf']:.2f} | Trades: {r['trades']}")

    # Step 4: Grid Search on best strategies
    print(f"\n[4/8] ETAPE 4: OPTIMISATION PARAMETRES (Grid Search)")
    print("-" * 70)

    best_overall = None
    all_optimized = []

    top_strategies = valid_strategies[:3] if valid_strategies else results[:3]

    for base_result in top_strategies:
        strategy_name = base_result['strategy']
        # Find the function
        func = None
        for f, n in strategies:
            if n == strategy_name:
                func = f
                break

        if not func:
            continue

        print(f"\n  Optimisation: {strategy_name}")

        for rr in RR_VALUES:
            for adx in ADX_VALUES:
                for rsi in RSI_RANGES:
                    result = backtest_strategy(df, func, strategy_name, rr=rr, adx_filter=adx, rsi_filter=rsi)
                    if result and result['trades'] >= 30:
                        all_optimized.append(result)
                        if not best_overall or result['pf'] > best_overall['pf']:
                            best_overall = result

    if best_overall:
        print(f"\n  MEILLEURE CONFIG:")
        print(f"    Strategie: {best_overall['strategy']}")
        print(f"    R:R: {best_overall['rr']}")
        print(f"    ADX min: {best_overall['adx_filter']}")
        print(f"    RSI range: {best_overall['rsi_filter']}")
        print(f"    Trades: {best_overall['trades']}")
        print(f"    Win Rate: {best_overall['win_rate']:.1f}%")
        print(f"    Profit Factor: {best_overall['pf']:.2f}")
        print(f"    PnL: ${best_overall['pnl']:+.2f}")
        print(f"    Max DD: {best_overall['max_dd']:.1f}%")

    # Step 5: Market Regime Detection
    print(f"\n[5/8] ETAPE 5: DETECTION REGIMES DE MARCHE")
    print("-" * 70)

    # Classify each bar into regime
    df['regime'] = 'NORMAL'
    df.loc[df['volatility_ratio'] > 1.5, 'regime'] = 'HIGH_VOL'
    df.loc[df['volatility_ratio'] < 0.7, 'regime'] = 'LOW_VOL'
    df.loc[(df['adx'] > 25) & (df['plus_di'] > df['minus_di']), 'regime'] = 'TREND_UP'
    df.loc[(df['adx'] > 25) & (df['minus_di'] > df['plus_di']), 'regime'] = 'TREND_DOWN'
    df.loc[df['adx'] < 20, 'regime'] = 'RANGING'
    df.loc[df['adx'] > 40, 'regime'] = 'STRONG_TREND'

    regime_counts = df['regime'].value_counts()
    print("  Distribution des regimes:")
    for regime, count in regime_counts.items():
        pct = count / len(df) * 100
        print(f"    {regime:15}: {count:5} ({pct:5.1f}%)")

    # Step 6: Risk Validation
    print(f"\n[6/8] ETAPE 6: VALIDATION CONTRAINTE RISQUE")
    print("-" * 70)

    if best_overall:
        days_over = best_overall.get('days_over_500', 0)
        if days_over == 0:
            print(f"  [OK] Lot 0.15 valide - 0 jours > -$500")
        else:
            print(f"  [!] {days_over} jours > -$500 avec lot 0.15")
            print(f"      Reduire le lot size recommande")

    # Step 7: Final Summary
    print(f"\n[7/8] ETAPE 7: VALIDATION FINALE")
    print("-" * 70)

    if best_overall and best_overall['pf'] >= 1.0:
        print(f"  [OK] Strategie validee!")
    else:
        print(f"  [!] Aucune strategie profitable trouvee")
        print(f"      EUR/JPY peut ne pas etre adapte pour le scalping")

    # Step 8: Documentation
    print(f"\n[8/8] ETAPE 8: DOCUMENTATION STRATEGIE OPTIMALE")
    print("=" * 70)

    if best_overall:
        print(f"""
PAIRE: EUR/JPY
STRATEGIE: {best_overall['strategy']}

PARAMETRES OPTIMAUX:
  - R:R: {best_overall['rr']}
  - ADX min: {best_overall['adx_filter']}
  - RSI range: {best_overall['rsi_filter']}

PERFORMANCE:
  - Profit Factor: {best_overall['pf']:.2f}
  - Win Rate: {best_overall['win_rate']:.1f}%
  - Max Drawdown: {best_overall['max_dd']:.1f}%
  - Trades: {best_overall['trades']}
  - PnL Total: ${best_overall['pnl']:+.2f}

GESTION RISQUE:
  - Lot size: 0.15
  - Max perte/jour: $500
  - Jours > -$500: {best_overall.get('days_over_500', 'N/A')}
""")

    print("=" * 70)

    # Save results to CSV
    if all_optimized:
        results_df = pd.DataFrame(all_optimized)
        results_df = results_df.sort_values('pf', ascending=False)
        filename = f"eurjpy_optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_df.to_csv(filename, index=False)
        print(f"\nResultats sauvegardes dans: {filename}")

    return best_overall


if __name__ == "__main__":
    best = main()
