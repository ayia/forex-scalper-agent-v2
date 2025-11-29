# Forex Scalper Agent V2

[![Python CI](https://github.com/ayia/forex-scalper-agent-v2/actions/workflows/python-ci.yml/badge.svg)](https://github.com/ayia/forex-scalper-agent-v2/actions/workflows/python-ci.yml)

A multi-pair, multi-timeframe, multi-strategy Forex trading signal detection system built with Python.

## Features

- **Multi-Pair Scanning**: Scans 20+ forex pairs (majors and crosses)
- **Multi-Timeframe Analysis**: M1, M5, M15, H1, H4 timeframes
- **Multiple Strategies**:
  - Trend Following (EMA Stack + MACD)
  - Mean Reversion (Bollinger Bands + RSI)
- **Risk Management**: ATR-based dynamic SL/TP calculation
- **Signal Logging**: CSV export of all detected signals
- **Free API**: Uses yfinance for data (no API key required)

## Installation

```bash
# Clone the repository
git clone https://github.com/ayia/forex-scalper-agent-v2.git
cd forex-scalper-agent-v2

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from scanner_v2 import ForexScannerV2

# Create scanner instance
scanner = ForexScannerV2()

# Scan all pairs
signals = scanner.scan_all()

# Get top 5 signals by confidence
top_signals = scanner.get_top_signals(5)
for sig in top_signals:
    print(f"{sig.pair} {sig.direction} @ {sig.entry_price}")
```

## Project Structure

```
forex-scalper-agent-v2/
|-- config.py              # Configuration (pairs, timeframes, risk params)
|-- data_fetcher.py        # Data retrieval via yfinance
|-- base_strategy.py       # Abstract base class for strategies
|-- trend_following.py     # EMA + MACD strategy
|-- mean_reversion.py      # Bollinger Bands + RSI strategy
|-- risk_calculator.py     # ATR-based SL/TP calculation
|-- trade_logger.py        # CSV signal logging
|-- scanner_v2.py          # Main orchestrator
|-- requirements.txt       # Python dependencies
```

## Configuration

Edit `config.py` to customize:

- **Pairs**: Add/remove forex pairs in `ALL_PAIRS`
- **Risk Parameters**: Adjust `RISK_PARAMS` for SL/TP settings
- **Strategy Parameters**: Tune indicators in `STRATEGY_PARAMS`

## Strategies

### Trend Following
- Uses EMA stack (20/50/200)
- MACD crossover confirmation
- Best for trending markets

### Mean Reversion
- Bollinger Bands (20, 2.5 std)
- RSI overbought/oversold
- Best for ranging markets

## Risk Management

- Dynamic SL based on ATR (1.5x multiplier)
- Minimum R:R ratio of 1.5:1
- Maximum SL of 15 pips
- Position sizing based on account risk %

## Output

Signals are logged to `logs/signals.csv` with:
- Timestamp
- Pair, Direction, Entry Price
- Stop Loss, Take Profit
- Confidence Score
- Strategy Name

## Requirements

- Python 3.11+
- pandas, numpy
- yfinance
- python-dotenv

## License

MIT License

## Disclaimer

This software is for educational purposes only. Trading forex involves risk. Use at your own discretion.
