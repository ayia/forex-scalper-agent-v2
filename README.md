# Forex Scalper Agent V2

[![Python CI](https://github.com/ayia/forex-scalper-agent-v2/actions/workflows/python-ci.yml/badge.svg)](https://github.com/ayia/forex-scalper-agent-v2/actions/workflows/python-ci.yml)

A sophisticated multi-pair, multi-timeframe, multi-strategy Forex trading signal detection system with adaptive risk management, machine learning regime detection, and institutional-grade order flow analysis.

## 🚀 Features

### Core Trading System
- **Multi-Pair Scanning**: Scans 20 forex pairs (7 majors + 13 crosses)
- **Multi-Timeframe Analysis**: Hierarchical MTF analysis (H4 → H1 → M15 → M5 → M1)
- **Multiple Strategies**:
  - 📈 **Trend Following** - EMA Stack (20/50/200) + MACD crossover
  - 📉 **Mean Reversion** - Bollinger Bands + RSI + ADX filter
  - 💥 **Breakout** - Donchian Channels + Volume confirmation (7 indicators)

### Adaptive Intelligence
- **Market Regime Detection**: 6 regime types (Trending Bull/Bear, Ranging, Volatile, Choppy, Quiet)
- **ML Regime Detector**: Random Forest + Gradient Boosting ensemble with 100+ features
- **Adaptive Strategy Selection**: Dynamic strategy weighting based on regime
- **Consensus Validation**: Multi-layer signal validation (H1 alignment, M15 structure, RSI divergence, sentiment, volume)

### Advanced Analysis
- **Order Flow Analysis**: Delta, absorption, imbalances, liquidity pools, stop hunts
- **Sentiment Analysis**: Contrarian approach - fade retail extremes
- **Volume Confirmation**: RVI, VPT, A/D Line, OBV integration

### Risk Management
- **Dynamic Position Sizing**: Session, volatility, spread, and pair-specific adjustments
- **ATR-Based SL/TP**: Adaptive stop loss and take profit levels
- **Correlation Management**: Max 2 correlated pairs simultaneously
- **News Filter**: Economic calendar integration with buffer zones

### Infrastructure
- **Backtesting Engine**: Walk-forward optimization, Monte Carlo simulation
- **Broker Integration**: MT5, OANDA, Paper Trading support
- **Signal Logging**: CSV export with detailed metrics

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/ayia/forex-scalper-agent-v2.git
cd forex-scalper-agent-v2

# Install dependencies
pip install -r requirements.txt
```

## ⚡ Quick Start

### Basic Usage
```python
from scanner_v2 import ForexScalperV2

# Create scanner instance
scanner = ForexScalperV2(account_balance=10000, max_risk_percent=2.0)

# Scan all pairs
signals = scanner.scan_all()

# Display results
for sig in signals:
    print(f"{sig['pair']} {sig['direction']} @ {sig['entry_price']:.5f}")
    print(f"  SL: {sig['stop_loss']:.5f} | TP: {sig['take_profit']:.5f}")
    print(f"  Confidence: {sig['confidence']:.1f}% | Regime: {sig['regime']}")
```

### Command Line
```bash
# Single scan with full logging
python scanner_v2.py --once

# MTF signals as clean JSON (sorted by confluence descending)
python scanner_v2.py --mtf-json

# MTF signals with minimum 80% confluence
python scanner_v2.py --mtf-json --min-confluence 80

# Continuous scanning (every 5 minutes)
python scanner_v2.py --interval 300

# Traditional JSON output
python scanner_v2.py --once --json

# Scan only backtest-validated pairs (RECOMMENDED)
python scanner_v2.py --improved-only --interval 120

# Scan specific pairs only
python scanner_v2.py --pairs USDJPY,USDCHF --mtf-json --min-confluence 80
```

### CLI Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--once` | Run a single scan and exit | `False` |
| `--json` | Output traditional signals in JSON format | `False` |
| `--mtf-json` | Output MTF signals as clean JSON (no logs) sorted by confluence | `False` |
| `--min-confluence` | Minimum confluence score for `--mtf-json` | `60` |
| `--interval` | Seconds between scans in continuous mode | `300` |
| `--pairs` | Comma-separated list of pairs to scan | All pairs |
| `--improved-only` | Scan only backtest-validated pairs (USDJPY, USDCHF, EURUSD) | `False` |
| `--optimized-cross` | Scan 10 profitable cross pairs with pair-specific optimized configs | `False` |
| `--active-only` | With `--optimized-cross`: show only BUY/SELL signals (exclude WATCH) | `False` |

### MTF JSON Output Example
```bash
python scanner_v2.py --mtf-json --min-confluence 80
```
```json
[
  {
    "pair": "USDCAD",
    "direction": "SELL",
    "confluence": 95,
    "entry_price": 1.39750,
    "stop_loss": 1.39800,
    "take_profit": 1.39625,
    "sl_pips": 5.0,
    "tp_pips": 12.5,
    "risk_reward": "1:2.5",
    "h4_trend": "BEARISH",
    "h1_trend": "BEARISH",
    "m15_signal": "SELL_CONTINUATION",
    "m5_entry": "ENTRY_NOW",
    "sentiment": {
      "retail": -37.1,
      "contrarian_signal": "NEUTRAL",
      "strength": "neutral"
    },
    "timestamp": "2025-12-02T20:41:01.311437"
  }
]
```

### Optimized Cross Pairs JSON Output Example
```bash
# All signals (BUY, SELL, WATCH)
python main.py --optimized-cross

# Only active signals (BUY/SELL)
python main.py --optimized-cross --active-only
```
```json
[
  {
    "pair": "CHFJPY",
    "direction": "BUY",
    "trend": "BULLISH",
    "near_crossover": true,
    "ema_status": "CROSSOVER",
    "entry": 194.126,
    "stop_loss": 193.771,
    "take_profit": 194.658,
    "confluence_score": 75.8,
    "backtest_score": 7,
    "rsi": 50.9,
    "rsi_status": "OK",
    "adx": 52.7,
    "config": {
      "rr": 1.5,
      "adx_min": 25,
      "rsi_range": "25-75",
      "min_score": 4,
      "backtest_pf": 1.05
    },
    "timestamp": "2025-12-03T20:49:44.926627"
  },
  {
    "pair": "NZDJPY",
    "direction": "WATCH",
    "trend": "BULLISH",
    "near_crossover": true,
    "ema_status": "NEAR",
    "entry": 89.608,
    "confluence_score": 74.0,
    "rsi": 57.6,
    "rsi_status": "OK",
    "adx": 55.9,
    "config": {
      "rr": 1.2,
      "adx_min": 12,
      "rsi_range": "35-65",
      "min_score": 6,
      "backtest_pf": 1.11
    },
    "timestamp": "2025-12-03T20:49:43.358747"
  }
]
```

#### Signal Fields Explanation
| Field | Description |
|-------|-------------|
| `direction` | `BUY`, `SELL` (active signals) or `WATCH` (no crossover yet) |
| `trend` | Current EMA trend: `BULLISH` or `BEARISH` |
| `near_crossover` | `true` if EMA crossover is imminent or near |
| `ema_status` | Crossover proximity: `CROSSOVER`, `IMMINENT`, `NEAR`, `APPROACHING`, `DISTANT`, `FAR` |
| `confluence_score` | Granular score 0-100 (continuous, not discrete) |
| `backtest_score` | Original 0-8 backtest score (only for BUY/SELL) |
| `stop_loss` / `take_profit` | Only present for active BUY/SELL signals |

## 📂 Project Structure

```
forex-scalper-agent-v2/
│
├── 📊 Core System
│   ├── scanner_v2.py              # Main orchestrator
│   ├── config.py                  # Configuration (pairs, risk params)
│   └── data_fetcher.py            # Multi-source data retrieval
│
├── 📈 Trading Strategies
│   ├── base_strategy.py           # Abstract strategy interface
│   ├── trend_following.py         # EMA + MACD strategy
│   ├── mean_reversion.py          # Bollinger Bands + RSI strategy
│   ├── breakout.py                # Donchian + Volume strategy
│   ├── improved_strategy.py       # IMPROVED v2.3 (backtest-validated)
│   └── core/optimized_cross_scanner.py  # Optimized Cross Pairs v2.4
│
├── 🧠 Market Analysis
│   ├── market_regime_detector.py  # 6-regime classification
│   ├── ml_regime_detector.py      # ML-based regime detection
│   ├── mtf_analyzer.py            # Multi-timeframe analysis
│   ├── order_flow_analyzer.py     # Order flow & institutional activity
│   └── sentiment_analyzer.py      # Contrarian sentiment analysis
│
├── ✅ Validation & Selection
│   ├── consensus_validator.py     # Multi-TF signal validation
│   ├── adaptive_strategy_selector.py  # Regime-based strategy selection
│   └── adaptive_thresholds.py     # Dynamic confidence thresholds
│
├── 💰 Risk Management
│   ├── risk_calculator.py         # ATR-based SL/TP calculation
│   ├── adaptive_risk_manager.py   # Dynamic risk adjustments
│   ├── correlation_manager.py     # Correlation risk control
│   ├── position_manager.py        # Trailing stops, breakeven
│   └── news_filter.py             # Economic calendar filter
│
├── 🔧 Infrastructure
│   ├── backtester.py              # Backtesting engine
│   ├── run_backtest.py            # CLI backtest runner with grading
│   ├── broker_integration.py      # MT5, OANDA, Paper broker
│   ├── trade_logger.py            # CSV signal logging
│   └── api_keys.py                # API keys (gitignored)
│
├── 📁 Examples
│   └── examples/
│       └── enhanced_features_demo.py  # Demo of all features
│
└── 📄 Documentation
    ├── README.md                  # This file
    └── ADAPTIVE_SYSTEM_README.md  # Adaptive system details
```

## 📡 Multi-Source Data Fetcher

The system supports multiple **100% FREE** data sources with automatic fallback:

| Priority | Source | Rate Limit | Delay | API Key Required |
|----------|--------|------------|-------|------------------|
| 1 | **Twelve Data** | 800 req/day | ~1 min | Yes (free) - RECOMMENDED |
| 2 | **Finnhub** | 60 req/min | Real-time | Yes (free) |
| 3 | **Alpha Vantage** | 5 req/min, 500/day | Real-time | Yes (free) |
| 4 | **yfinance** | Unlimited | 15-20 min | No (FALLBACK) |

### Configuration

#### Option 1: API Keys File (Recommended)
Create `api_keys.py` in the project root:
```python
def get_twelve_data_key():
    return "your_twelve_data_key"

def get_finnhub_key():
    return "your_finnhub_key"

def get_alpha_vantage_key():
    return ""  # Optional
```

#### Option 2: Environment Variables
```bash
# Windows
set TWELVE_DATA_API_KEY=your_free_key
set FINNHUB_API_KEY=your_free_key

# Linux/Mac
export TWELVE_DATA_API_KEY=your_free_key
export FINNHUB_API_KEY=your_free_key
```

### Get Free API Keys (No Credit Card Required)
- **Twelve Data** (RECOMMENDED): https://twelvedata.com - 800 req/day
- **Finnhub**: https://finnhub.io - 60 req/min
- Alpha Vantage: https://www.alphavantage.co/support/#api-key

### Features
- **Automatic Fallback**: If one source fails, tries the next
- **Smart Caching**: 60-second cache to reduce API calls
- **Rate Limiting**: Respects each source's limits
- **Statistics Tracking**: Monitor success/failure rates per source

```python
from data_fetcher import DataFetcher

# Uses yfinance by default (no API key needed)
fetcher = DataFetcher()

# Or with API keys for real-time data
fetcher = DataFetcher(
    alpha_vantage_key="your_key",
    twelve_data_key="your_key"
)

# Get statistics
stats = fetcher.get_statistics()
print(f"Success rate: {stats['success_rate']:.1f}%")
```

## ⚙️ Configuration

Edit `config.py` to customize:

### Trading Universe
```python
MAJOR_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'USDCAD', 'NZDUSD']
CROSS_PAIRS = ['EURJPY', 'EURGBP', 'EURCHF', 'EURAUD', 'EURCAD',
               'GBPJPY', 'GBPAUD', 'GBPCAD', 'GBPCHF',
               'AUDJPY', 'CADJPY', 'CHFJPY', 'NZDJPY']
```

### Risk Parameters
```python
RISK_PARAMS = {
    'max_spread_pips': 2.0,        # Skip if spread > 2 pips
    'min_atr_pips': 5.0,           # Skip if ATR < 5 pips
    'max_atr_pips': 30.0,          # Skip if ATR > 30 pips
    'max_sl_pips': 15.0,           # Absolute SL limit
    'sl_atr_multiplier': 1.5,      # SL = 1.5x ATR
    'min_rr_ratio': 1.5,           # Minimum Risk:Reward
    'target_rr_ratio': 2.0,        # Target Risk:Reward
    'confidence_threshold': 60,    # Minimum confidence %
}
```

### Strategy Parameters
```python
STRATEGY_PARAMS = {
    'ema_fast': 20,
    'ema_medium': 50,
    'ema_slow': 200,
    'rsi_period': 14,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'bb_period': 20,
    'bb_std': 2.5,
    'donchian_period': 20,
}
```

## 📈 Trading Strategies

### 1. Trend Following
| Parameter | Value |
|-----------|-------|
| **Indicators** | EMA Stack (20/50/200) + MACD (12/26/9) |
| **Entry BUY** | EMA20 > EMA50 > EMA200 + MACD crosses above signal |
| **Entry SELL** | EMA20 < EMA50 < EMA200 + MACD crosses below signal |
| **SL/TP** | 1.5x ATR / 3.0x ATR |
| **Best For** | Trending markets |

### 2. Mean Reversion
| Parameter | Value |
|-----------|-------|
| **Indicators** | Bollinger Bands (20, 2.5σ) + RSI (14) + ADX (14) |
| **Entry BUY** | Price ≤ Lower BB + RSI < 30 + ADX ≤ 30 |
| **Entry SELL** | Price ≥ Upper BB + RSI > 70 + ADX ≤ 30 |
| **Safety** | ADX > 30 → Signal rejected (trend too strong) |
| **Best For** | Ranging markets |

### 3. Breakout (Volume-Confirmed)
| Parameter | Value |
|-----------|-------|
| **Indicators** | Donchian Channels (20) + VolumeAnalyzer (7 metrics) |
| **Entry BUY** | Price > Donchian High + Volume Score ≥ 60 |
| **Entry SELL** | Price < Donchian Low + Volume Score ≥ 60 |
| **Volume Metrics** | Volume Ratio, RVI, VPT, A/D Line, OBV |
| **Best For** | Volatile markets with breakouts |

### 4. IMPROVED Strategies (v2.3.0)
Backtest-validated strategies with proven profitability (+6.46% over 60 days).

| Parameter | ImprovedTrend | ImprovedScalping |
|-----------|---------------|------------------|
| **Validated Pairs** | USDJPY, USDCHF, EURUSD | USDJPY, USDCHF |
| **Win Rate** | 46.9% | 46.9% |
| **Profit Factor** | 1.31 | 1.31 |
| **Max Drawdown** | 7.99% | 7.99% |
| **HTF Alignment** | H4 + H1 required for BUY | H1 required |
| **BUY Conditions** | STRICT (HTF aligned + MACD crossover) | STRICT (EMA + Stoch crossover) |
| **SELL Conditions** | STANDARD (continuation allowed) | STANDARD (continuation allowed) |
| **R:R Ratio** | 1.8:1 to 2.5:1 | 1.5:1 to 2.0:1 |

**Key Features:**
- Asymmetric BUY/SELL conditions (BUY historically weaker, requires stricter filters)
- Market regime detection (ADX-based: TRENDING > 20, RANGING < 15)
- Consecutive loss management (pause after 3-4 losses)
- Pair filtering based on historical Win Rate

**Usage:**
```bash
# Recommended: Use validated pairs only
python scanner_v2.py --improved-only --interval 120

# Or run backtest to validate
python run_backtest.py --improved --pairs USDJPY,USDCHF,EURUSD --days 60
```

### 5. Optimized Cross Pairs (NEW in v2.4.0 - RECOMMENDED)
10 cross pairs with pair-specific optimized parameters based on 2-year backtest validation.

| Pair | Profit Factor | R:R | ADX | RSI Range | Min Score | Trades |
|------|---------------|-----|-----|-----------|-----------|--------|
| **NZDJPY** | 1.11 | 1.2 | 12 | 35-65 | 6 | 657 |
| **CADJPY** | 1.10 | 2.5 | 25 | 35-65 | 6 | 370 |
| **AUDJPY** | 1.07 | 1.2 | 20 | 30-70 | 6 | 788 |
| **GBPCAD** | 1.05 | 2.0 | 25 | 35-65 | 6 | 452 |
| **CHFJPY** | 1.05 | 1.5 | 25 | 25-75 | 4 | 1080 |
| **EURJPY** | 1.04 | 1.8 | 20 | 25-75 | 5 | 947 |
| **EURCAD** | 1.03 | 2.5 | 15 | 35-65 | 6 | 380 |
| **GBPAUD** | 1.02 | 2.5 | 12 | 35-65 | 6 | 443 |
| **EURAUD** | 1.01 | 2.5 | 25 | 35-65 | 4 | 572 |
| **GBPJPY** | 1.01 | 1.2 | 25 | 30-70 | 6 | 686 |

**Key Features:**
- EMA crossover strategy (8/21/50) optimized per pair
- Score calculation matching backtest exactly (0-8 scale)
- RSI status indication (OK, OVERBOUGHT, OVERSOLD)
- JSON output sorted by: active signals first, then confluence score, then backtest PF

**Usage:**
```bash
# Scan all 10 optimized cross pairs (all signals)
python main.py --optimized-cross

# Only active BUY/SELL signals (no WATCH)
python main.py --optimized-cross --active-only
```

### 6. EUR/GBP Validated Strategy (NEW in v3.5.0)
RSI Divergence + Stochastic Double Cross strategy validated across 8 market periods (2020-2024).

| Parameter | Value |
|-----------|-------|
| **Strategy** | RSI Divergence + Stochastic Double (Hybrid) |
| **Score** | 85.5/100 (best overall) |
| **Profit Factor** | 1.10-1.31 (varies by period) |
| **Win Rate** | 40-48% |
| **Max Drawdown** | 15-20% |
| **Crisis Survival** | ✅ Passed all major crises |

**Entry Rules:**
- **BUY (RSI Divergence)**: Price makes lower low, RSI makes higher low
- **BUY (Stochastic Double)**: K & D both < 20, then K crosses above D
- **SELL (RSI Divergence)**: Price makes higher high, RSI makes lower high
- **SELL (Stochastic Double)**: K & D both > 80, then K crosses below D

**Optimal Parameters:**
```
R:R Ratio:        1.5
SL:               2.0x ATR (~15-25 pips)
TP:               3.0x ATR (~22-38 pips)
RSI Period:       14
Stochastic:       14/3 (K/D)
Oversold:         < 20
Overbought:       > 80
```

**Regime Performance:**
| Regime | Tradeable | Position Size |
|--------|-----------|---------------|
| RANGING | ✅ YES | 100% |
| LOW_VOLATILITY | ✅ YES | 100% |
| CONSOLIDATION | ✅ YES | 80% |
| TRENDING_DOWN | ✅ YES | 70% |
| RECOVERY | ✅ YES | 80% |
| TRENDING_UP | ⚠️ CAUTION | 50% |
| HIGH_VOLATILITY | ❌ NO | 0% |
| CRISIS | ❌ NO | 0% |

**Session Rules:**
| Session | Hours (UTC) | Tradeable |
|---------|-------------|-----------|
| LONDON | 7:00-15:00 | ✅ 100% |
| NEW YORK | 12:00-21:00 | ✅ 90% |
| ASIAN | 0:00-7:00 | ❌ Avoid |

**Usage:**
```bash
# Scan EURGBP
python main.py --pairs EURGBP

# Scan all validated pairs
python main.py --pairs CADJPY,EURCHF,EURGBP

# Only active signals
python main.py --pairs EURGBP --active-only
```

### 7. Enhanced Scalping (v2.2.0)
Advanced multi-confirmation scalping system inspired by DIY Custom Strategy Builder [ZP].

| Parameter | Value |
|-----------|-------|
| **Leading Indicator** | Range Filter (DW) / WAE / QQE (auto-selected) |
| **Confirmations** | EMA, RSI, Chandelier Exit, Range Filter |
| **Confirmation Rate** | Minimum 70% alignment required |
| **Volume Confirmation** | PVSRA climax candles |
| **SL Placement** | Chandelier Exit + Supply/Demand zones |
| **TP Placement** | ATR-based + nearest Supply/Demand zone |
| **Session Filtering** | Pairs prioritized by trading session |
| **Max SL** | 15 pips (scalping limit) |
| **Best For** | Intraday scalping with high probability setups |

**Leading Indicator Selection:**
| Session | Regime | Leading Indicator |
|---------|--------|-------------------|
| London/NY Overlap | Any | WAE (momentum) |
| London/New York | Trending | Range Filter |
| Tokyo | Any | QQE (lower volatility) |
| Off-hours | Ranging | QQE |

**Pair-Session Scoring:**
| Condition | Score |
|-----------|-------|
| Optimal pair during overlap | 100 |
| Optimal pair in active session | 70-85 |
| Non-optimal pair during overlap | 60 |
| Non-optimal pair in active session | 30-50 |

## 🎯 Market Regime Detection

The system detects 6 market regimes and adapts strategies accordingly:

| Regime | Condition | Active Strategies |
|--------|-----------|-------------------|
| **TRENDING_BULL** | ADX > 25, Bullish slope | TrendFollowing (1.2x), Breakout (0.9x) |
| **TRENDING_BEAR** | ADX > 25, Bearish slope | TrendFollowing (1.2x), Breakout (0.9x) |
| **RANGING** | ADX < 20, Narrow BB | MeanReversion (1.3x), Breakout (0.7x) |
| **VOLATILE** | ATR > 1.5x average | Breakout (1.1x), MeanReversion (0.6x) |
| **CHOPPY** | No clear direction | All reduced (0.3-0.8x) |
| **QUIET** | ATR < 0.6x average | All reduced (0.3-0.5x) |

## 📊 Multi-Timeframe Analysis

Hierarchical top-down analysis ensures directional alignment:

```
H4 (Primary Bias)      ─── 60% weight → Defines overall direction
  ↓
H1 (Confirmation)      ─── 40% weight → Validates H4 trend
  ↓
M15 (Signal Generation) ─── Entry setup aligned with HTF bias
  ↓
M5 (Entry Timing)      ─── Precise entry moment
  ↓
M1 (SL/TP Optimization) ─── Fine-tune risk levels
```

**Confluence Score (0-100)**:
- H4 Alignment: 25 points
- H1 Alignment: 25 points
- M15 Signal: 25 points
- M5 Timing: 15 points
- RSI Alignment: 10 points

**Minimum Threshold**: 60/100

## ✅ Consensus Validation

Multi-layer signal validation:

| Validator | Weight | Description |
|-----------|--------|-------------|
| **H1 Alignment** | 30% | H1 trend matches signal direction |
| **M15 Structure** | 25% | Support/Resistance levels favorable |
| **RSI Divergence** | 20% | Bullish/Bearish divergence detection |
| **Sentiment** | 15% | Contrarian positioning (fade extremes) |
| **Volume** | 10% | Volume confirms price movement |

**Valid Signal**: Consensus Score ≥ 60/100

## 💰 Risk Management

### Position Sizing Formula
```
base_risk = account_balance × 1%
adjusted_risk = base_risk × session_mult × vol_mult × spread_mult × pair_mult

position_size = adjusted_risk / (SL_distance × pip_value)
```

### Session Multipliers
| Session | Multiplier |
|---------|------------|
| London-NY Overlap | 1.3x |
| London | 1.2x |
| New York | 1.1x |
| Tokyo | 0.9x |
| Quiet Hours | 0.6x |

### Risk Limits
- **Per Trade**: 1% account risk
- **Per Session**: 2% max account risk
- **Max SL**: 15 pips
- **Min R:R**: 1.5:1
- **Max Correlated Pairs**: 2 simultaneously

## 🔧 Advanced Features

### Sentiment Analysis (Contrarian Approach)

The system uses a **contrarian trading approach** - fading extreme retail sentiment:

```python
from sentiment_analyzer import SentimentAnalyzer

analyzer = SentimentAnalyzer()
result = analyzer.analyze('EURUSD', price_data)

print(f"Retail Sentiment: {result['retail_sentiment']}")  # -100 to +100
print(f"Contrarian Signal: {result['contrarian_signal']}")  # BUY, SELL, NEUTRAL
print(f"Institutional Bias: {result['institutional_bias']}")
```

| Retail Sentiment | Contrarian Action |
|------------------|-------------------|
| **> +70** (Extreme Bullish) | SELL signal strength boost |
| **< -70** (Extreme Bearish) | BUY signal strength boost |
| **-30 to +30** (Neutral) | No adjustment |

**Components analyzed:**
- Technical Sentiment (RSI/Stochastic): 30%
- Volatility Sentiment (ATR ratio): 20%
- Momentum Sentiment (ROC/EMA): 30%
- Position Estimation: 20%

### News Filter (Economic Calendar)

Automatically blocks trading around high-impact news events:

```python
from news_filter import NewsFilter

news = NewsFilter()
can_trade, reason, event = news.should_trade('EURUSD')

if not can_trade:
    print(f"Trading blocked: {reason}")
    print(f"Event: {event.event_name} at {event.timestamp}")
```

| Event Impact | Buffer Before | Buffer After |
|--------------|---------------|--------------|
| **CRITICAL** (NFP, FOMC, ECB) | 60 minutes | 30 minutes |
| **HIGH** (CPI, GDP, Employment) | 30 minutes | 15 minutes |
| **MEDIUM** (PMI, Retail Sales) | 15 minutes | 10 minutes |

**Risk adjustment near news:**
```python
risk_multiplier = news.get_risk_adjustment('EURUSD')
# Returns 0.5 to 1.0 based on proximity to news events
```

### Order Flow Analysis
```python
from order_flow_analyzer import OrderFlowAnalyzer

analyzer = OrderFlowAnalyzer()
result = analyzer.analyze(df, 'EURUSD', 'M15')

print(f"Primary Signal: {result.primary_signal.value}")
print(f"Delta: {result.delta:.2f}")
print(f"Buying Pressure: {result.buying_pressure:.1f}%")
print(f"Absorption Detected: {result.absorption_detected}")
```

### ML Regime Detection
```python
from ml_regime_detector import MLRegimeDetector

detector = MLRegimeDetector()
prediction = detector.predict(df, 'EURUSD')

print(f"Detected Regime: {prediction.regime}")
print(f"Confidence: {prediction.confidence:.1f}%")
print(f"Model Agreement: {prediction.model_agreement:.1f}%")
```

### Backtesting

#### Command Line (Recommended)
```bash
# Run backtest with IMPROVED strategies (recommended)
python run_backtest.py --improved --pairs USDJPY,USDCHF,EURUSD --days 60

# Run backtest with original strategies
python run_backtest.py --strategy all --pairs EURUSD,GBPUSD --days 30

# Run with Monte Carlo simulation
python run_backtest.py --improved --monte-carlo --days 60
```

#### Backtest CLI Parameters
| Parameter | Description | Default |
|-----------|-------------|---------|
| `--pairs` | Comma-separated pairs to test | `USDJPY,USDCHF,EURUSD` |
| `--days` | Number of days of historical data | `30` |
| `--balance` | Initial account balance | `10000` |
| `--risk` | Risk per trade (0.01 = 1%) | `0.01` |
| `--strategy` | Strategy: `simple`, `trend`, `improved`, `all` | `all` |
| `--improved` | Use IMPROVED strategies (recommended) | `False` |
| `--monte-carlo` | Run Monte Carlo simulation | `False` |

#### Backtest Output
```
======================================================================
                    RAPPORT DE BACKTEST
======================================================================
Periode: 2024-10-03 -> 2024-12-02 (60 jours)
Balance initiale: $10,000.00
Balance finale:   $10,645.93

[RESULTATS]
   Profit/Perte:     +$645.93 (+6.46%)
   Nombre de trades: 32
   Win Rate:         46.9%
   Profit Factor:    1.31
   Max Drawdown:     7.99%

[NOTE FINALE]
   Score: 70/100
   Note:  7.0/10 - BON
======================================================================
```

#### Python API
```python
from backtester import BacktestEngine

engine = BacktestEngine(
    initial_balance=10000,
    risk_per_trade=0.02,
    spread_pips=1.0
)

engine.add_strategy(TrendFollowingStrategy())
result = engine.run(data, start_date, end_date)
print(engine.generate_report(result))
```

### Broker Integration
```python
from broker_integration import BrokerFactory, BrokerCredentials

credentials = BrokerCredentials(
    broker='mt5',
    account_id='12345',
    server='ICMarkets-Demo',
    password='your_password',
    demo=True
)

broker = BrokerFactory.create(credentials)
await broker.connect()
await broker.place_order(pair='EURUSD', side=OrderSide.BUY, quantity=0.1)
```

## 📋 Requirements

### Core Dependencies
```
yfinance>=0.2.40
pandas>=2.0.0
numpy>=1.24.0
python-dotenv>=1.0.0
requests>=2.31.0
textblob>=0.17.1
pandas_ta>=0.3.14b
loguru>=0.7.0
```

### Machine Learning (Optional)
```
scikit-learn>=1.3.0
joblib>=1.3.0
```

### Backtesting & Analysis
```
matplotlib>=3.7.0
seaborn>=0.12.0
```

### Broker Integration (Optional)
```
MetaTrader5>=5.0.45  # Windows only
oandapyV20>=0.7.2    # For OANDA
```

## 📊 Output

Signals are logged to `logs/signals.csv` with:
- Timestamp, Pair, Timeframe
- Direction, Entry Price
- Stop Loss, Take Profit
- Confidence Score, Validation Score
- Strategy Name, Regime
- Position Size, Risk Amount

## 🧪 Testing

```bash
# Single scan test
python scanner_v2.py --once --json

# Run demo
python examples/enhanced_features_demo.py

# Full scan
python scanner_v2.py
```

## 📈 Signal Quality Tiers

| Tier | Confidence | Position Size |
|------|------------|---------------|
| **TIER 1 (Premium)** | ≥ 85% | Full position |
| **TIER 2 (Good)** | 70-84% | 0.75x position |
| **TIER 3 (Acceptable)** | 60-69% | 0.5x position |
| **REJECTED** | < 60% | No trade |

## 🛡️ Safety Features

1. **ADX Filter**: Prevents mean reversion in strong trends
2. **Correlation Limits**: Max 2 correlated pairs
3. **News Filter**: Buffers around high-impact events
4. **Spread Filter**: Skips pairs with spread > 2 pips
5. **ATR Limits**: Avoids extreme volatility (5-30 pips range)
6. **Session Awareness**: Reduced sizing in quiet hours

## 📄 License

MIT License

## ⚠️ Disclaimer

This software is for educational purposes only. Trading forex involves substantial risk of loss and is not suitable for all investors. Past performance is not indicative of future results. Use at your own discretion.

---

## 📝 Changelog

### Version 3.5.0 (December 2025) - EUR/GBP Validated Strategy
*New validated pair with RSI Divergence + Stochastic Double strategy*

#### New EUR/GBP Scanner (`core/eurgbp_validated_scanner.py`)
- **Hybrid Strategy**: RSI Divergence + Stochastic Double Cross
- **Multi-Period Validated**: Tested on 8 market periods (COVID, Ukraine War, Banking Crisis, etc.)
- **Score**: 85.5/100 (highest among tested strategies)
- **Monte Carlo**: 100% positive simulations, 0% ruin probability
- **Parameter Stable**: PF > 0.9 with +/- 15% parameter variation

#### Strategy Optimization System
- **40 Strategies Tested**: Comprehensive backtest of all major trading strategies
- **Multi-Period Analysis**: COVID Crash, Recovery, Inflation, Fed Hiking, etc.
- **Regime Detection**: Automatic market regime classification
- **Session Filtering**: London/NY optimal, Asian avoided

#### New Files
- `core/eurgbp_strategy_optimizer.py`: Full 40-strategy optimization engine
- `core/eurgbp_complete_analysis.py`: Multi-period analysis and scoring
- `core/eurgbp_validated_scanner.py`: Production-ready scanner

#### CLI Integration
- `python main.py --pairs EURGBP`: Scan EUR/GBP with validated strategy
- `python main.py --pairs CADJPY,EURCHF,EURGBP`: Scan all validated pairs

---

### Version 2.5.0 (December 2025) - Granular Confluence Scoring
*Improved scoring system with continuous values and better signal representation*

#### New Granular Confluence Score System
- **Continuous Scoring**: Replaced discrete 0-8 backtest score with granular 0-100 scale
- **6 Weighted Components**:
  - EMA Crossover/Proximity: 25 pts (CROSSOVER=25, IMMINENT=20, NEAR=15, APPROACHING=10, DISTANT=5)
  - Trend Alignment: 20 pts (price vs EMA50 + slope strength)
  - RSI Position: 15 pts (optimal zone scoring with edge penalties)
  - ADX Strength: 15 pts (scaled by threshold excess)
  - MACD Alignment: 15 pts (direction + momentum)
  - Price Momentum: 10 pts (ROC direction + magnitude)

#### Unified JSON Structure
- All signals (BUY/SELL/WATCH) now have consistent fields: `trend`, `near_crossover`, `ema_status`
- `ema_status` shows crossover proximity: `CROSSOVER`, `IMMINENT`, `NEAR`, `APPROACHING`, `DISTANT`, `FAR`
- Proper sorting: active signals first, then by confluence_score descending

#### New CLI Parameter
- `--active-only`: With `--optimized-cross`, show only BUY/SELL signals (exclude WATCH)

---

### Version 2.4.0 (December 2025) - Optimized Cross Pairs
*10 profitable cross pairs with pair-specific optimized configurations*

#### New Optimized Cross Scanner (`core/optimized_cross_scanner.py`)
- **10 Profitable Cross Pairs**: Selected from 2-year backtest (PF >= 1.0)
- **Pair-Specific Parameters**: Each pair has optimal R:R, ADX, RSI range, min score
- **EMA Crossover Strategy**: 8/21/50 periods with confluence scoring
- **Backtest-Accurate Scoring**: 0-8 scale matching optimizer exactly
- **RSI Status Indicator**: Shows OK, OVERBOUGHT, or OVERSOLD status

#### Profitable Pairs (Sorted by Profit Factor)
| Pair | PF | R:R | Trades |
|------|-----|-----|--------|
| NZDJPY | 1.11 | 1.2 | 657 |
| CADJPY | 1.10 | 2.5 | 370 |
| AUDJPY | 1.07 | 1.2 | 788 |
| GBPCAD | 1.05 | 2.0 | 452 |
| CHFJPY | 1.05 | 1.5 | 1080 |
| EURJPY | 1.04 | 1.8 | 947 |
| EURCAD | 1.03 | 2.5 | 380 |
| GBPAUD | 1.02 | 2.5 | 443 |
| EURAUD | 1.01 | 2.5 | 572 |
| GBPJPY | 1.01 | 1.2 | 686 |

#### CLI Parameter
- `--optimized-cross`: Scan 10 profitable cross pairs with optimized configs (JSON output)

#### JSON Output Features
- Sorted by: active signals (BUY/SELL) first, then confluence score desc, then backtest PF desc
- Includes pair-specific config in each signal
- Real-time data via yfinance

---

### Version 2.3.0 (December 2025) - Backtest-Validated Strategies
*IMPROVED strategies with proven profitability*

#### New IMPROVED Strategies (`improved_strategy.py`)
- **ImprovedTrendStrategy**: HTF-aligned trend following with strict BUY filters
- **ImprovedScalpingStrategy**: EMA + Stochastic crossover with H1 bias confirmation
- **Asymmetric Conditions**: BUY requires stricter filters (historically weaker)
- **Pair Filtering**: Only validated pairs (USDJPY 58% WR, USDCHF 45% WR, EURUSD 40% WR)
- **Consecutive Loss Management**: Pauses trading after 3-4 consecutive losses

#### Backtest Results (60 days)
| Metric | Value |
|--------|-------|
| **Total Profit** | +$645.93 (+6.46%) |
| **Win Rate** | 46.9% |
| **Profit Factor** | 1.31 |
| **Max Drawdown** | 7.99% |
| **Grade** | 7.0/10 (BON) |

#### New CLI Parameters
- `--pairs`: Filter specific pairs to scan
- `--improved-only`: Scan only backtest-validated pairs (USDJPY, USDCHF, EURUSD)

#### Backtest Engine Improvements (`run_backtest.py`)
- Strategy grading system (0-100 score, converted to /10)
- Monte Carlo simulation for risk analysis
- Detailed per-pair and per-direction breakdown
- Support for IMPROVED strategies via `--improved` flag

#### Data Fetcher Enhancements
- Added **Finnhub** as priority #2 data source (60 req/min, real-time)
- API keys file support (`api_keys.py`) as alternative to environment variables
- Improved rate limiting and fallback logic

### Version 2.2.0 (December 2025) - Enhanced Scalping System
*Inspired by DIY Custom Strategy Builder [ZP] analysis*

#### New Advanced Indicators Module (`advanced_indicators.py`)
- **Range Filter (DW)**: Adaptive noise filtering with Donchian Width - excellent for scalping entries
- **Chandelier Exit**: ATR-based trailing stop placement for optimal SL levels
- **Waddah Attar Explosion (WAE)**: Momentum/volatility detection for explosive move identification
- **Choppiness Index**: Range vs trend detection - automatically skips choppy markets
- **PVSRA (Price Volume Spread Analysis)**: Smart money detection via climax/rising volume candles
- **Supply/Demand Zone Detection**: Intelligent SL/TP placement based on key levels
- **QQE Mod**: Enhanced RSI with dynamic bands for precise entries
- **Session Manager**: Automatic trading session detection with DST handling (Tokyo, London, New York, Sydney)

#### Enhanced Scalping Strategy (`enhanced_scalping_strategy.py`)
- **Leading + Confirmation System**: Primary signal generator with multiple confirmation filters
- **Signal Expiry**: Wait up to N candles for confirmations to align (prevents premature entries)
- **Adaptive Leading Selector**: Automatically selects best indicator based on regime and session
- **Session-Aware Trading**: Prioritizes optimal pairs for each trading session
- **Pair-Session Scoring**: 0-100 score indicating pair suitability for current session

#### Key Improvements
- **70% Confirmation Rate Required**: Reduces false signals
- **Volume Confirmation**: PVSRA climax candles boost confidence
- **Zone-Based SL/TP**: Uses Supply/Demand zones for intelligent level placement
- **Volatility-Adjusted Targets**: SL/TP multipliers adapt to session volatility
- **Maximum 15-pip SL**: Scalping-appropriate stop loss limits

#### Scanner Integration
- Enhanced strategies automatically loaded in `scanner_v2.py`
- Three pre-configured strategies: Range Filter, WAE, and QQE as leading indicators
- 10% confidence boost for enhanced strategy signals

### Version 2.1.0 (December 2025)
- **Multi-Source DataFetcher**: yfinance + Alpha Vantage + Twelve Data with automatic fallback
- **MTF Analyzer**: Complete top-down analysis (H4 -> H1 -> M15 -> M5 -> M1)
- **Sentiment Analyzer**: Contrarian approach with retail sentiment estimation
- **News Filter**: Economic calendar integration with impact-based buffers
- **CLI Improvements**: New `--mtf-json` and `--min-confluence` parameters
- **Position Manager**: Trailing stops, breakeven, partial take-profit

### Version 2.0.0 (November 2025)
- Initial release with adaptive strategy selection
- Market regime detection (6 regimes)
- Consensus validation system
- Order flow analysis

---

**Author**: Forex Scalper Agent V2
**Version**: 3.5.0
**Last Updated**: December 2025
