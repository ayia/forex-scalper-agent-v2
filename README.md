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
# Single scan with JSON output
python scanner_v2.py --once --json

# Continuous scanning
python scanner_v2.py

# Run enhanced features demo
python examples/enhanced_features_demo.py
```

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
│   └── breakout.py                # Donchian + Volume strategy
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
│   ├── broker_integration.py      # MT5, OANDA, Paper broker
│   └── trade_logger.py            # CSV signal logging
│
├── 📁 Examples
│   └── examples/
│       └── enhanced_features_demo.py  # Demo of all features
│
└── 📄 Documentation
    ├── README.md                  # This file
    └── ADAPTIVE_SYSTEM_README.md  # Adaptive system details
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

**Author**: Forex Scalper Agent V2
**Version**: 2.0.0
**Last Updated**: December 2025
