"""
Enhanced Features Demo - Forex Scalper Agent V2
================================================
Demonstrates the new features added to the system:

1. Volume-Confirmed Breakouts
2. Backtesting Engine
3. Order Flow Analysis
4. ML Regime Detection
5. Broker Integration
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def demo_volume_breakout():
    """Demonstrate enhanced breakout with volume confirmation."""
    print("\n" + "=" * 60)
    print("1. VOLUME-CONFIRMED BREAKOUT STRATEGY")
    print("=" * 60)

    from breakout import BreakoutStrategy, VolumeAnalyzer

    # Create sample data with volume
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')

    # Simulate a breakout scenario
    base_price = 1.1000
    prices = [base_price]
    for i in range(99):
        change = np.random.randn() * 0.0005
        # Add trend for last 10 candles (breakout)
        if i > 88:
            change += 0.0008
        prices.append(prices[-1] + change)

    df = pd.DataFrame({
        'Open': [p - 0.0001 for p in prices],
        'High': [p + np.random.random() * 0.001 for p in prices],
        'Low': [p - np.random.random() * 0.001 for p in prices],
        'Close': prices,
        'Volume': [np.random.randint(1000, 5000) for _ in prices]
    }, index=dates)

    # Increase volume on breakout candles
    df.loc[df.index[-5:], 'Volume'] = df['Volume'].iloc[-5:] * 3

    # Initialize strategy
    strategy = BreakoutStrategy()

    print("\nAnalyzing breakout with volume confirmation...")
    result = strategy.analyze(df, 'EURUSD')

    if result:
        print(f"\n✅ Breakout Signal Detected!")
        print(f"   Direction: {result['direction']}")
        print(f"   Confidence: {result['confidence']:.1f}%")
        print(f"   Entry Price: {result['entry_price']:.5f}")
        print(f"   Breakout Strength: {result['breakout_strength']:.2f}%")

        vol = result['volume_analysis']
        print(f"\n📊 Volume Analysis:")
        print(f"   Volume Ratio: {vol['volume_ratio']}x average")
        print(f"   Is Spike: {vol['is_spike']}")
        print(f"   Volume Trend: {vol['volume_trend']}")
        print(f"   RVI: {vol['rvi']}")
        print(f"   A/D Trend: {vol['ad_trend']}")
        print(f"   Confirmation Score: {vol['confirmation_score']}/100")
    else:
        print("\n❌ No breakout signal detected")


def demo_backtesting():
    """Demonstrate backtesting engine."""
    print("\n" + "=" * 60)
    print("2. BACKTESTING ENGINE")
    print("=" * 60)

    from backtester import BacktestEngine, BacktestResult

    # Create engine
    engine = BacktestEngine(
        initial_balance=10000,
        risk_per_trade=0.02,
        spread_pips=1.0,
        max_positions=3
    )

    print("\n📈 Backtesting Engine Features:")
    print("   - Historical data simulation")
    print("   - Multiple strategy testing")
    print("   - Performance metrics (Sharpe, Sortino, Max DD)")
    print("   - Walk-forward optimization")
    print("   - Monte Carlo simulation")
    print("   - Detailed trade analysis")

    print("\n💡 Example Usage:")
    print("""
    # Load data
    data = {
        'EURUSD': {
            'M15': eurusd_m15_df,
            'H1': eurusd_h1_df
        },
        'GBPUSD': {
            'M15': gbpusd_m15_df,
            'H1': gbpusd_h1_df
        }
    }

    # Add strategies
    engine.add_strategy(TrendFollowingStrategy())
    engine.add_strategy(MeanReversionStrategy())
    engine.add_strategy(BreakoutStrategy())

    # Run backtest
    result = engine.run(
        pairs=['EURUSD', 'GBPUSD'],
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30)
    )

    # Generate report
    report = engine.generate_report(result)
    print(report)

    # Export trades
    engine.export_trades('trades.csv')
    """)


def demo_order_flow():
    """Demonstrate order flow analysis."""
    print("\n" + "=" * 60)
    print("3. ORDER FLOW ANALYSIS")
    print("=" * 60)

    from order_flow_analyzer import OrderFlowAnalyzer, OrderFlowSignal

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='15min')

    prices = [1.1000]
    volumes = []

    for i in range(99):
        change = np.random.randn() * 0.0003
        prices.append(prices[-1] + change)

        # Create absorption pattern at the end
        if i > 90:
            volumes.append(np.random.randint(8000, 15000))  # High volume
        else:
            volumes.append(np.random.randint(2000, 5000))

    volumes.append(volumes[-1])

    df = pd.DataFrame({
        'open': [p - 0.0001 for p in prices],
        'high': [p + np.random.random() * 0.0008 for p in prices],
        'low': [p - np.random.random() * 0.0008 for p in prices],
        'close': prices,
        'volume': volumes
    }, index=dates)

    # Analyze
    analyzer = OrderFlowAnalyzer()
    result = analyzer.analyze(df, 'EURUSD', 'M15')

    print("\n📊 Order Flow Analysis Results:")
    print(f"   Primary Signal: {result.primary_signal.value}")
    print(f"   Signal Strength: {result.signal_strength:.1f}%")
    print(f"   Entry Quality: {result.entry_quality:.1f}%")
    print(f"   Recommended Action: {result.recommended_action}")

    print(f"\n📈 Delta Analysis:")
    print(f"   Current Delta: {result.delta:.2f}")
    print(f"   Cumulative Delta: {result.cumulative_delta:.2f}")
    print(f"   Delta Divergence: {result.delta_divergence}")

    print(f"\n💪 Pressure Analysis:")
    print(f"   Buying Pressure: {result.buying_pressure:.1f}%")
    print(f"   Selling Pressure: {result.selling_pressure:.1f}%")
    print(f"   Net Pressure: {result.net_pressure:.1f}")

    print(f"\n🏦 Institutional Activity:")
    print(f"   Institutional Buying: {result.institutional_buying}")
    print(f"   Institutional Selling: {result.institutional_selling}")
    print(f"   Activity Score: {result.large_player_activity:.1f}%")

    print(f"\n🎯 Special Detections:")
    print(f"   Absorption Detected: {result.absorption_detected}")
    print(f"   Stop Hunt Detected: {result.stop_hunt_detected}")
    print(f"   Imbalance Zones: {len(result.imbalance_zones)}")
    print(f"   Liquidity Pools: {len(result.liquidity_pools)}")


def demo_ml_regime():
    """Demonstrate ML regime detection."""
    print("\n" + "=" * 60)
    print("4. ML REGIME DETECTION")
    print("=" * 60)

    from ml_regime_detector import MLRegimeDetector, FeatureEngineer

    # Create sample data
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=300, freq='15min')

    # Simulate trending market
    prices = [1.1000]
    for i in range(299):
        trend = 0.0002 if i < 150 else -0.0001  # Uptrend then downtrend
        noise = np.random.randn() * 0.0003
        prices.append(prices[-1] + trend + noise)

    df = pd.DataFrame({
        'open': [p - 0.0001 for p in prices],
        'high': [p + np.random.random() * 0.001 for p in prices],
        'low': [p - np.random.random() * 0.001 for p in prices],
        'close': prices,
        'volume': [np.random.randint(2000, 8000) for _ in prices]
    }, index=dates)

    # Feature engineering
    fe = FeatureEngineer()
    features = fe.generate_features(df)

    print(f"\n🔬 Feature Engineering:")
    print(f"   Generated {len(features.columns)} features")
    print(f"   Sample features: {features.columns[:10].tolist()}...")

    # ML Detection
    detector = MLRegimeDetector()
    prediction = detector.predict(df, 'EURUSD')

    print(f"\n🤖 ML Regime Prediction:")
    print(f"   Detected Regime: {prediction.regime}")
    print(f"   Confidence: {prediction.confidence:.1f}%")
    print(f"   Model Agreement: {prediction.model_agreement:.1f}%")

    print(f"\n📊 Regime Probabilities:")
    for regime, prob in sorted(prediction.probabilities.items(),
                               key=lambda x: x[1], reverse=True)[:3]:
        print(f"   {regime}: {prob:.1f}%")

    print("\n💡 Available Regimes:")
    for idx, name in detector.regime_names.items():
        print(f"   {idx}: {name}")


def demo_broker_integration():
    """Demonstrate broker integration."""
    print("\n" + "=" * 60)
    print("5. BROKER INTEGRATION")
    print("=" * 60)

    from broker_integration import (
        BrokerCredentials, BrokerFactory, PaperBroker,
        OrderSide, OrderType, TradingEngine
    )
    import asyncio

    async def run_demo():
        # Create paper broker credentials
        credentials = BrokerCredentials(
            broker='paper',
            account_id='DEMO_12345',
            demo=True
        )

        # Create broker
        broker = BrokerFactory.create(credentials)

        print("\n📡 Connecting to Paper Broker...")
        await broker.connect()

        # Get account info
        account = await broker.get_account_info()
        print(f"\n💰 Account Info:")
        print(f"   Balance: ${account.balance:,.2f}")
        print(f"   Equity: ${account.equity:,.2f}")
        print(f"   Currency: {account.currency}")
        print(f"   Leverage: {account.leverage}:1")

        # Place a test order
        print("\n📝 Placing test order...")
        order = await broker.place_order(
            pair='EURUSD',
            side=OrderSide.BUY,
            quantity=0.1,
            order_type=OrderType.MARKET,
            stop_loss=1.0950,
            take_profit=1.1100
        )

        print(f"\n✅ Order Placed:")
        print(f"   Order ID: {order.order_id}")
        print(f"   Pair: {order.pair}")
        print(f"   Side: {order.side.value}")
        print(f"   Quantity: {order.quantity}")
        print(f"   Status: {order.status.value}")

        # Get positions
        positions = await broker.get_positions()
        print(f"\n📊 Open Positions: {len(positions)}")
        for pos in positions:
            print(f"   {pos.pair}: {pos.side.value} {pos.quantity} @ {pos.entry_price:.5f}")

        # Close position
        if positions:
            print("\n🔒 Closing position...")
            closed = await broker.close_position(positions[0].position_id)
            print(f"   Position {closed.position_id} closed")

        await broker.disconnect()
        print("\n📴 Disconnected from broker")

    # Run async demo
    asyncio.run(run_demo())

    print("\n💡 Supported Brokers:")
    print("   - MetaTrader 5 (MT5)")
    print("   - OANDA")
    print("   - Paper Trading (simulation)")


def demo_full_integration():
    """Show how all components work together."""
    print("\n" + "=" * 60)
    print("6. FULL INTEGRATION EXAMPLE")
    print("=" * 60)

    print("""
    # Complete trading workflow with all new features:

    from scanner_v2 import ForexScalperV2
    from ml_regime_detector import MLRegimeDetector
    from order_flow_analyzer import OrderFlowAnalyzer
    from broker_integration import BrokerFactory, BrokerCredentials
    from backtester import BacktestEngine

    # 1. Initialize components
    scanner = ForexScalperV2()
    ml_detector = MLRegimeDetector()
    order_flow = OrderFlowAnalyzer()

    # 2. Connect to broker
    credentials = BrokerCredentials(broker='mt5', account_id='12345', ...)
    broker = BrokerFactory.create(credentials)
    await broker.connect()

    # 3. Scan for signals with enhanced analysis
    for pair in pairs:
        # Get data
        data = fetch_data(pair)

        # ML Regime detection (enhanced)
        regime = ml_detector.predict(data['H1'], pair)

        # Order flow analysis (new)
        flow = order_flow.analyze(data['M15'], pair)

        # Generate signal
        signal = scanner.scan_pair(pair, data)

        if signal and flow.entry_quality > 70:
            # Execute trade
            order = await broker.place_order(
                pair=pair,
                side=signal['direction'],
                quantity=signal['position_size'],
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profit']
            )

    # 4. Backtest to validate strategy
    engine = BacktestEngine()
    engine.add_strategy(scanner.strategies)
    result = engine.run(historical_data)
    print(engine.generate_report(result))
    """)


if __name__ == "__main__":
    print("=" * 60)
    print("  FOREX SCALPER AGENT V2 - ENHANCED FEATURES DEMO")
    print("=" * 60)
    print("\nThis demo showcases the 5 new improvements:")
    print("  1. Volume-Confirmed Breakouts")
    print("  2. Backtesting Engine")
    print("  3. Order Flow Analysis")
    print("  4. ML Regime Detection")
    print("  5. Broker Integration")

    # Run demos
    try:
        demo_volume_breakout()
    except Exception as e:
        print(f"\n⚠️ Volume demo error: {e}")

    try:
        demo_backtesting()
    except Exception as e:
        print(f"\n⚠️ Backtesting demo error: {e}")

    try:
        demo_order_flow()
    except Exception as e:
        print(f"\n⚠️ Order flow demo error: {e}")

    try:
        demo_ml_regime()
    except Exception as e:
        print(f"\n⚠️ ML regime demo error: {e}")

    try:
        demo_broker_integration()
    except Exception as e:
        print(f"\n⚠️ Broker demo error: {e}")

    demo_full_integration()

    print("\n" + "=" * 60)
    print("  DEMO COMPLETE")
    print("=" * 60)
