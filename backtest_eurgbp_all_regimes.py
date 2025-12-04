#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EURGBP Stochastic Strategy - Backtest All Market Regimes
=========================================================
Teste la strategie Stochastic optimisee sur tous les regimes de marche.

Configuration Optimale EURGBP:
- Strategy: Stochastic Crossover
- R:R: 2.0
- Oversold: 20, Overbought: 80
- Zone Buffer: 10 (BUY si K<30, SELL si K>70)
- SL: 1.5 x ATR
- Capital: $10,000
- Lot: 0.25
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Optional, Dict
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION OPTIMALE EURGBP
# =============================================================================
PAIR = 'EURGBP'
SYMBOL = f'{PAIR}=X'
CAPITAL = 10000
LOT_SIZE = 0.25
PIP_VALUE = 0.0001

# Strategie Stochastic optimisee
CONFIG = {
    'strategy': 'STOCHASTIC_CROSSOVER',
    'rr': 2.0,
    'oversold': 20,
    'overbought': 80,
    'zone_buffer': 10,
    'sl_mult': 1.5,
    'stoch_period': 14,
    'stoch_smooth': 3,
}

# =============================================================================
# REGIMES DE MARCHE A TESTER
# =============================================================================
MARKET_REGIMES = {
    # 1. Haute Volatilite
    'COVID_CRASH': {
        'name': '1. COVID Crash & Recovery',
        'start': '2020-01-01',
        'end': '2020-06-30',
        'description': 'Volatilite extreme, mouvements de 200+ pips/jour',
        'regime_type': 'HIGH_VOLATILITY',
        'interval': '1d'
    },

    # 2. Tendance Haussiere
    'POST_BREXIT_RALLY': {
        'name': '2. Post-Brexit Rally',
        'start': '2017-01-01',
        'end': '2017-12-31',
        'description': 'Recuperation GBP apres le crash Brexit',
        'regime_type': 'TRENDING_UP',
        'interval': '1d'
    },

    # 3. Tendance Baissiere (Crash)
    'BREXIT_CRASH': {
        'name': '3. Brexit Vote Crash',
        'start': '2016-06-01',
        'end': '2016-10-31',
        'description': 'Crash GBP apres vote Brexit (-15%)',
        'regime_type': 'TRENDING_DOWN',
        'interval': '1d'
    },

    # 4. Marche Ranging
    'RANGING_2019': {
        'name': '4. Ranging Market 2019',
        'start': '2019-01-01',
        'end': '2019-06-30',
        'description': 'Marche calme, oscillations sans tendance',
        'regime_type': 'RANGING',
        'interval': '1d'
    },

    # 5. Recovery
    'POST_COVID_2021': {
        'name': '5. Post-COVID Recovery 2021',
        'start': '2021-01-01',
        'end': '2021-12-31',
        'description': 'Reprise economique, tendances mixtes',
        'regime_type': 'RECOVERY',
        'interval': '1d'
    },

    # 6. Divergence Taux (BOE vs ECB)
    'RATE_DIVERGENCE_2022': {
        'name': '6. Rate Divergence 2022-2023',
        'start': '2022-06-01',
        'end': '2023-06-30',
        'description': 'BOE hawkish vs ECB, divergence monetaire',
        'regime_type': 'RATE_DIVERGENCE',
        'interval': '1d'
    },

    # 7. Conditions Recentes
    'RECENT_2024': {
        'name': '7. Recent Conditions 2024',
        'start': '2024-01-01',
        'end': '2024-11-30',
        'description': 'Conditions de marche actuelles',
        'regime_type': 'RECENT',
        'interval': '1h'  # H1 disponible
    },

    # 8. Flash Crash
    'FLASH_CRASH_2016': {
        'name': '8. GBP Flash Crash Oct 2016',
        'start': '2016-10-01',
        'end': '2016-11-30',
        'description': 'Mini-crash GBP (-6% en minutes)',
        'regime_type': 'FLASH_CRASH',
        'interval': '1d'
    },

    # 9. Faible Volatilite
    'LOW_VOL_2018': {
        'name': '9. Low Volatility 2018',
        'start': '2018-04-01',
        'end': '2018-09-30',
        'description': 'Periode de faible volatilite',
        'regime_type': 'LOW_VOLATILITY',
        'interval': '1d'
    },

    # 10. Incertitude Brexit
    'BREXIT_UNCERTAINTY': {
        'name': '10. Brexit Negotiations 2018-2019',
        'start': '2018-10-01',
        'end': '2019-03-31',
        'description': 'Incertitude politique maximale',
        'regime_type': 'UNCERTAINTY',
        'interval': '1d'
    },
}


# =============================================================================
# DATA & INDICATORS
# =============================================================================
def fetch_data(start: str, end: str, interval: str = '1d') -> Optional[pd.DataFrame]:
    """Telecharge les donnees EURGBP."""
    try:
        df = yf.download(SYMBOL, start=start, end=end, interval=interval, progress=False)

        if df.empty:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.lower() for c in df.columns]

        return df
    except Exception as e:
        return None


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute Stochastic et ATR."""
    df = df.copy()

    # Stochastic
    period = CONFIG['stoch_period']
    smooth = CONFIG['stoch_smooth']

    low_n = df['low'].rolling(window=period).min()
    high_n = df['high'].rolling(window=period).max()
    df['stoch_k'] = 100 * (df['close'] - low_n) / (high_n - low_n + 0.0001)
    df['stoch_d'] = df['stoch_k'].rolling(window=smooth).mean()

    # ATR
    high, low, close = df['high'], df['low'], df['close']
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()

    return df


# =============================================================================
# STRATEGY & BACKTEST
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
                return {
                    'entry_time': entry_time,
                    'exit_time': df.index[j],
                    'direction': direction,
                    'result': 'LOSS',
                    'pips': pips,
                    'pnl': pips * 10 * LOT_SIZE
                }
            if bar['high'] >= tp:
                pips = (tp - entry_price) / PIP_VALUE
                return {
                    'entry_time': entry_time,
                    'exit_time': df.index[j],
                    'direction': direction,
                    'result': 'WIN',
                    'pips': pips,
                    'pnl': pips * 10 * LOT_SIZE
                }
        else:  # SELL
            if bar['high'] >= sl:
                pips = (entry_price - sl) / PIP_VALUE
                return {
                    'entry_time': entry_time,
                    'exit_time': df.index[j],
                    'direction': direction,
                    'result': 'LOSS',
                    'pips': pips,
                    'pnl': pips * 10 * LOT_SIZE
                }
            if bar['low'] <= tp:
                pips = (entry_price - tp) / PIP_VALUE
                return {
                    'entry_time': entry_time,
                    'exit_time': df.index[j],
                    'direction': direction,
                    'result': 'WIN',
                    'pips': pips,
                    'pnl': pips * 10 * LOT_SIZE
                }

    return None


def run_backtest(df: pd.DataFrame) -> List[dict]:
    """Execute le backtest avec la strategie Stochastic."""
    trades = []

    oversold = CONFIG['oversold']
    overbought = CONFIG['overbought']
    zone_buffer = CONFIG['zone_buffer']
    rr = CONFIG['rr']
    sl_mult = CONFIG['sl_mult']

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        prev = df.iloc[i-1]

        direction = None

        # Stochastic K crosses D
        if prev['stoch_k'] < prev['stoch_d'] and row['stoch_k'] > row['stoch_d']:
            if row['stoch_k'] < oversold + zone_buffer:  # K < 30
                direction = 'BUY'
        elif prev['stoch_k'] > prev['stoch_d'] and row['stoch_k'] < row['stoch_d']:
            if row['stoch_k'] > overbought - zone_buffer:  # K > 70
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


def calc_metrics(trades: List[dict]) -> dict:
    """Calcule les metriques de performance."""
    if not trades:
        return {
            'trades': 0, 'wins': 0, 'losses': 0,
            'wr': 0, 'pf': 0, 'pnl': 0, 'max_dd': 0, 'roi': 0,
            'avg_win': 0, 'avg_loss': 0
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

    # Avg win/loss
    avg_win = total_wins / len(wins) if wins else 0
    avg_loss = total_losses / len(losses) if losses else 0

    return {
        'trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'wr': wr,
        'pf': pf,
        'pnl': pnl,
        'max_dd': max_dd,
        'roi': pnl / CAPITAL * 100,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'final_equity': CAPITAL + pnl
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 90)
    print("   EURGBP STOCHASTIC STRATEGY - BACKTEST ALL MARKET REGIMES")
    print("=" * 90)
    print(f"\nConfiguration Optimale:")
    print(f"  - Strategie: {CONFIG['strategy']}")
    print(f"  - R:R: {CONFIG['rr']}")
    print(f"  - Oversold/Overbought: {CONFIG['oversold']}/{CONFIG['overbought']}")
    print(f"  - Zone Buffer: {CONFIG['zone_buffer']} (BUY si K<{CONFIG['oversold']+CONFIG['zone_buffer']}, SELL si K>{CONFIG['overbought']-CONFIG['zone_buffer']})")
    print(f"  - SL: {CONFIG['sl_mult']} x ATR")
    print(f"  - Capital: ${CAPITAL:,}")
    print(f"  - Lot Size: {LOT_SIZE}")
    print("=" * 90)

    all_results = {}
    all_trades = []

    print(f"\n{'Regime':<45} {'Bars':>6} {'Trades':>7} {'WR%':>7} {'PF':>6} {'PnL':>12} {'MaxDD%':>8} {'Status':>8}")
    print("-" * 105)

    for regime_key, regime_info in MARKET_REGIMES.items():
        print(f"  {regime_info['name'][:43]:<43}", end="", flush=True)

        # Fetch data
        df = fetch_data(regime_info['start'], regime_info['end'], regime_info['interval'])

        if df is None or len(df) < 30:
            print(f" {'N/A':>6} {'SKIP - Donnees insuffisantes':>60}")
            continue

        # Add indicators
        df = add_indicators(df)
        df = df.dropna()

        if len(df) < 30:
            print(f" {'N/A':>6} {'SKIP - Donnees insuffisantes':>60}")
            continue

        # Run backtest
        trades = run_backtest(df)
        metrics = calc_metrics(trades)

        # Store results
        all_results[regime_key] = {
            'info': regime_info,
            'metrics': metrics,
            'bars': len(df)
        }
        all_trades.extend(trades)

        # Status
        if metrics['pf'] >= 1.0 and metrics['pnl'] > 0:
            status = "[OK]"
        elif metrics['pf'] >= 0.9:
            status = "[~]"
        else:
            status = "[XX]"

        print(f" {len(df):>6} {metrics['trades']:>7} {metrics['wr']:>6.1f}% {metrics['pf']:>5.2f} ${metrics['pnl']:>+10,.0f} {metrics['max_dd']:>7.1f}% {status:>8}")

    # Summary by regime type
    print("\n" + "=" * 90)
    print("RESUME PAR TYPE DE REGIME")
    print("=" * 90)

    regime_types = defaultdict(lambda: {'trades': 0, 'wins': 0, 'pnl': 0, 'periods': 0})

    for regime_key, result in all_results.items():
        rtype = result['info']['regime_type']
        m = result['metrics']
        regime_types[rtype]['trades'] += m['trades']
        regime_types[rtype]['wins'] += m['wins']
        regime_types[rtype]['pnl'] += m['pnl']
        regime_types[rtype]['periods'] += 1

    print(f"\n{'Type de Regime':<25} {'Periodes':>10} {'Trades':>8} {'WR%':>8} {'PnL':>12} {'Status':>10}")
    print("-" * 80)

    for rtype, data in sorted(regime_types.items(), key=lambda x: x[1]['pnl'], reverse=True):
        wr = data['wins'] / data['trades'] * 100 if data['trades'] > 0 else 0
        status = "[OK]" if data['pnl'] > 0 else "[XX]"
        print(f"{rtype:<25} {data['periods']:>10} {data['trades']:>8} {wr:>7.1f}% ${data['pnl']:>+10,.0f} {status:>10}")

    # Global summary
    print("\n" + "=" * 90)
    print("RESUME GLOBAL - TOUS REGIMES CONFONDUS")
    print("=" * 90)

    total_metrics = calc_metrics(all_trades)

    profitable_regimes = sum(1 for r in all_results.values() if r['metrics']['pnl'] > 0)
    total_regimes = len(all_results)

    print(f"\n  Periodes testees: {total_regimes}")
    print(f"  Periodes profitables: {profitable_regimes}/{total_regimes} ({profitable_regimes/total_regimes*100:.0f}%)")
    print(f"\n  Total trades: {total_metrics['trades']}")
    print(f"  Win Rate: {total_metrics['wr']:.1f}%")
    print(f"  Profit Factor: {total_metrics['pf']:.2f}")
    print(f"  P&L Total: ${total_metrics['pnl']:+,.2f}")
    print(f"  ROI: {total_metrics['roi']:+.1f}%")
    print(f"  Max Drawdown: {total_metrics['max_dd']:.1f}%")
    print(f"  Capital Final: ${total_metrics['final_equity']:,.2f}")

    # Verdict
    print("\n" + "=" * 90)
    print("VERDICT")
    print("=" * 90)

    if profitable_regimes >= total_regimes * 0.7:  # 70%+ profitable
        grade = "A - EXCELLENT"
        comment = "Strategie robuste sur la majorite des regimes"
    elif profitable_regimes >= total_regimes * 0.5:  # 50%+
        grade = "B - BON"
        comment = "Strategie profitable mais sensible au regime"
    elif profitable_regimes >= total_regimes * 0.3:  # 30%+
        grade = "C - ACCEPTABLE"
        comment = "Strategie a utiliser avec prudence"
    else:
        grade = "D - FAIBLE"
        comment = "Strategie non recommandee pour tous les regimes"

    print(f"\n  Note: {grade}")
    print(f"  {comment}")

    # Best/Worst regimes
    sorted_results = sorted(all_results.items(), key=lambda x: x[1]['metrics']['pnl'], reverse=True)

    print(f"\n  Meilleurs regimes:")
    for regime_key, result in sorted_results[:3]:
        m = result['metrics']
        print(f"    - {result['info']['name']}: PF={m['pf']:.2f}, P&L=${m['pnl']:+,.0f}")

    print(f"\n  Pires regimes:")
    for regime_key, result in sorted_results[-3:]:
        m = result['metrics']
        if m['pnl'] < 0:
            print(f"    - {result['info']['name']}: PF={m['pf']:.2f}, P&L=${m['pnl']:+,.0f}")

    # Recommendations
    print("\n" + "=" * 90)
    print("RECOMMANDATIONS")
    print("=" * 90)

    # Analyze which regimes work best
    good_regimes = [r['info']['regime_type'] for r in all_results.values() if r['metrics']['pf'] >= 1.0]
    bad_regimes = [r['info']['regime_type'] for r in all_results.values() if r['metrics']['pf'] < 0.9]

    if good_regimes:
        print(f"\n  [+] Trader EURGBP Stochastic dans: {', '.join(set(good_regimes))}")
    if bad_regimes:
        print(f"  [-] Eviter ou reduire taille dans: {', '.join(set(bad_regimes))}")

    print(f"\n  Configuration finale validee:")
    print(f"    'EURGBP': {{")
    print(f"        'strategy': 'STOCHASTIC',")
    print(f"        'rr': {CONFIG['rr']},")
    print(f"        'oversold': {CONFIG['oversold']},")
    print(f"        'overbought': {CONFIG['overbought']},")
    print(f"        'zone_buffer': {CONFIG['zone_buffer']},")
    print(f"        'sl_mult': {CONFIG['sl_mult']},")
    print(f"        'pf': {total_metrics['pf']:.2f},")
    print(f"        'wr': {total_metrics['wr']:.1f},")
    print(f"    }}")

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()
