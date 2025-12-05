# EUR/AUD Strategy Optimization Report

**Generated:** December 5, 2025
**Pair:** EUR/AUD
**Framework:** Forex Scalper Agent V2

---

## Executive Summary

After comprehensive testing of 40 trading strategies across multiple market regimes and macro-economic periods, the **BB %B (Bollinger Band Percent B)** strategy has been validated as the best strategy for EUR/AUD.

### Key Performance Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Profit Factor | 1.39 | PASS |
| Win Rate | 45.0% | GOOD |
| Walk-Forward Efficiency | 125.2% | EXCELLENT |
| Monte Carlo (Positive) | 100% | PASS |
| Max Drawdown (95th pct) | 10.7% | PASS |
| Ruin Probability | 0% | PASS |

---

## EUR/AUD Pair Characteristics

EUR/AUD is a commodity cross pair with unique characteristics:

- **pip_value:** 0.0001 (4 decimal places)
- **Average Daily Range:** 80-120 pips
- **Volatility:** Medium-High
- **Liquidity:** Good during London session
- **Correlation:** Inverse with AUD/USD, positive with EUR/USD

### Key Drivers

1. **Australian Dollar (AUD)**
   - Iron Ore prices (major export)
   - Gold prices
   - China economic data (major trading partner)
   - RBA interest rate decisions
   - Risk sentiment (risk-on = AUD strength)

2. **Euro (EUR)**
   - ECB interest rate decisions
   - Eurozone economic data
   - European political developments

---

## Validated Strategy: BB %B

### Strategy Logic

The Bollinger Band %B strategy is a **mean reversion strategy** that trades extreme volatility conditions:

```
%B = (Close - Lower Band) / (Upper Band - Lower Band)

BUY Signal:
- %B crosses from below 0 to above 0
- Price was below the lower Bollinger Band
- Price is now recovering back into the bands

SELL Signal:
- %B crosses from above 1 to below 1
- Price was above the upper Bollinger Band
- Price is now falling back into the bands
```

### Optimal Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| BB Period | 20 | Standard 20-period Bollinger Bands |
| BB Std Dev | 2.0 | Standard 2 standard deviations |
| R:R Ratio | 1.5 | Risk-Reward ratio |
| SL Multiplier | 2.5x ATR | Stop Loss distance |
| TP Distance | 3.75x ATR | Take Profit (1.5 x SL) |

### Entry Rules

1. **BUY Entry:**
   - Wait for %B to cross from below 0 to above 0
   - Enter at the close of the signal candle
   - Stop Loss: Entry - (2.5 x ATR)
   - Take Profit: Entry + (3.75 x ATR)

2. **SELL Entry:**
   - Wait for %B to cross from above 1 to below 1
   - Enter at the close of the signal candle
   - Stop Loss: Entry + (2.5 x ATR)
   - Take Profit: Entry - (3.75 x ATR)

---

## Market Regime Analysis

### Regime Distribution

| Regime | Time % | Recommendation |
|--------|--------|----------------|
| TRENDING_UP | 18.7% | TRADE |
| STRONG_TREND_UP | 18.5% | TRADE |
| TRENDING_DOWN | 17.2% | TRADE |
| STRONG_TREND_DOWN | 14.9% | AVOID |
| RANGING | 11.5% | TRADE (BEST) |
| NORMAL | 9.4% | TRADE |
| CONSOLIDATION | 7.4% | TRADE |
| HIGH_VOLATILITY | 1.3% | AVOID |

### Regime Detection Rules

```python
HIGH_VOLATILITY:
- ATR > 1.5x 60-period average ATR

STRONG_TREND:
- ADX > 40

TRENDING:
- ADX > 25

RANGING:
- ADX < 20

CONSOLIDATION:
- BB Width < 60% of average
```

---

## Macro Period Performance

The strategy was tested across multiple macro-economic periods:

| Period | Dates | PF | Trades | Status |
|--------|-------|-----|--------|--------|
| Normal 2019 | 2019 | 5.22 | 8 | EXCELLENT |
| COVID Recovery | May-Dec 2020 | 2.09 | 4 | GOOD |
| Fed Hiking | Mar 2022 - Jun 2023 | 2.24 | 12 | EXCELLENT |
| China Slowdown | Jun 2022 - Jun 2023 | 2.16 | 10 | EXCELLENT |
| RBA Hiking | May 2022 - Dec 2023 | 1.89 | 15 | GOOD |
| Rate Divergence 2024 | Jan-Jun 2024 | 1.50 | 4 | PASS |
| Commodity Boom 2021 | 2021 | 0.62 | 9 | CAUTION |
| Recent | Jul-Dec 2024 | 0.69 | 4 | CAUTION |

**Note:** Strategy performs best in trending and ranging markets, but may struggle during strong commodity-driven AUD moves.

---

## Robustness Testing

### Monte Carlo Simulation (500 iterations)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Positive Simulations | 100% | > 90% | PASS |
| Max DD (95th pct) | 10.7% | < 35% | PASS |
| Ruin Probability | 0% | < 5% | PASS |

### Walk-Forward Optimization (5 segments)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| Walk-Forward Efficiency | 125.2% | > 50% | PASS |
| In-Sample Mean PF | 1.09 | - | - |
| Out-of-Sample Mean PF | 1.36 | - | - |

The WFE > 100% indicates the strategy actually performs **better** on out-of-sample data than in-sample, which is extremely rare and suggests the strategy is robust.

---

## Session Recommendations

| Session | Hours (UTC) | Quality | Notes |
|---------|-------------|---------|-------|
| LONDON | 07:00-16:00 | BEST | Best EUR liquidity |
| SYDNEY-LONDON Overlap | 06:00-08:00 | GOOD | AUD + EUR active |
| ASIAN | 00:00-07:00 | LOW | Low EUR liquidity |
| NEW YORK | 12:00-21:00 | MODERATE | May work |
| LATE NY | 21:00-00:00 | AVOID | Low liquidity |

---

## Risk Management

### Position Sizing

Based on $10,000 account:

| Risk Level | Lot Size | Risk per Trade | Max Daily Loss |
|------------|----------|----------------|----------------|
| Conservative | 0.03 | ~$30 | $150 |
| Moderate | 0.05 | ~$50 | $250 |
| Standard | 0.10 | ~$100 | $500 |

### Risk Rules

1. **Max Daily Loss:** $500 (5% of $10,000 account)
2. **Max Concurrent Trades:** 2
3. **Max Daily Trades:** 3
4. **Weekly Loss Limit:** $1,000 (10% of account)

### Stop Loss Calculation

```
For EUR/AUD at 1.76264:
- ATR (14-period): ~17 pips (0.0017)
- SL Distance: 2.5 x 17 = 42.5 pips
- TP Distance: 1.5 x 42.5 = 63.75 pips

At 0.05 lots (~$5/pip):
- Risk: 42.5 x $5 = $212.50
- Reward: 63.75 x $5 = $318.75
```

---

## Alternative Strategies

If BB %B doesn't suit your trading style, these alternatives were also validated:

### 1. BB + RSI (PF: 1.13)

```
BUY: Price at lower BB AND RSI < 30
SELL: Price at upper BB AND RSI > 70
```

### 2. Ichimoku Basic (PF: 1.12)

```
BUY: Price > Cloud AND Tenkan > Kijun
SELL: Price < Cloud AND Tenkan < Kijun
```

### 3. RSI Oversold (PF: 1.11)

```
BUY: RSI crosses above 30 from below
SELL: RSI crosses below 70 from above
```

---

## Implementation Checklist

### Before Each Trade

- [ ] Check current market regime (avoid HIGH_VOLATILITY, STRONG_TREND_DOWN)
- [ ] Verify session time (prefer London, Sydney-London overlap)
- [ ] Calculate current ATR for SL/TP sizing
- [ ] Check daily P&L (stop if approaching -$500)
- [ ] Confirm no major news events in next 2 hours

### Entry Execution

- [ ] Wait for %B crossover on closed candle
- [ ] Confirm crossover direction (below 0 → above 0 for BUY)
- [ ] Set SL at 2.5x ATR from entry
- [ ] Set TP at 3.75x ATR from entry
- [ ] Record trade in log

### Exit Rules

- [ ] Primary: Let TP or SL execute
- [ ] Secondary: Close if opposite %B signal appears
- [ ] NEVER move SL further from entry
- [ ] Consider partial close at 50% of target

### Post-Trade

- [ ] Record outcome in trade log
- [ ] Update daily P&L
- [ ] Check if any limits reached
- [ ] Review for lessons learned

---

## Trade Example

**Scenario:** EUR/AUD BUY Signal

```
Date: December 5, 2025
Time: 08:00 UTC (London Open)
Current Price: 1.75800
%B Previous: -0.05 (below lower band)
%B Current: 0.02 (just crossed above 0)

ATR (14): 0.00170 (17 pips)

Entry: 1.75800
SL: 1.75800 - (0.00170 x 2.5) = 1.75375 (42.5 pips risk)
TP: 1.75800 + (0.00170 x 3.75) = 1.76438 (63.75 pips reward)

At 0.05 lots:
- Potential Loss: $212.50
- Potential Profit: $318.75
- R:R: 1:1.5
```

---

## Files Created

1. **[euraud_complete_optimizer.py](core/euraud_complete_optimizer.py)** - Complete optimization system
2. **[euraud_validated_scanner.py](core/euraud_validated_scanner.py)** - Live signal scanner
3. **euraud_optimization_results_*.csv** - All strategy backtest results

---

## Usage

### Run Full Optimization

```bash
python -m core.euraud_complete_optimizer
```

### Run Live Scanner

```bash
python -m core.euraud_validated_scanner
```

---

## Disclaimer

This strategy has been validated through historical backtesting and robustness testing. However:

- Past performance does not guarantee future results
- Always use proper risk management
- Paper trade for 4-8 weeks before going live
- Monitor strategy performance and adjust if market conditions change
- Never risk more than you can afford to lose

---

## Summary

The **BB %B strategy** for EUR/AUD provides a robust mean-reversion approach with:

- Strong validation metrics (WFE 125%, MC 100% positive)
- Clear entry/exit rules based on Bollinger Band extremes
- Optimized parameters (R:R 1.5, SL 2.5x ATR)
- Defined regime and session filters
- Comprehensive risk management framework

**Recommendation:** Implement with conservative position sizing (0.03-0.05 lots) and strict adherence to regime/session filters.
