"""
Backtesting Engine - Complete Strategy Validation System
=========================================================
Comprehensive backtesting framework for validating trading strategies.

Features:
- Historical data simulation
- Multiple strategy testing
- Performance metrics (Sharpe, Sortino, Max Drawdown, etc.)
- Walk-forward optimization
- Monte Carlo simulation
- Detailed trade analysis
- Visual reporting
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"


class TradeStatus(Enum):
    OPEN = "OPEN"
    CLOSED_TP = "CLOSED_TP"
    CLOSED_SL = "CLOSED_SL"
    CLOSED_MANUAL = "CLOSED_MANUAL"
    CLOSED_TIME = "CLOSED_TIME"


@dataclass
class BacktestTrade:
    """Represents a single trade in backtesting."""
    id: int
    pair: str
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    entry_time: datetime
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    status: TradeStatus = TradeStatus.OPEN
    pnl: float = 0.0
    pnl_pips: float = 0.0
    strategy: str = ""
    timeframe: str = ""
    confidence: float = 0.0
    max_favorable_excursion: float = 0.0  # Best unrealized profit
    max_adverse_excursion: float = 0.0    # Worst unrealized loss
    holding_time: Optional[timedelta] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Complete backtest results and statistics."""
    # Basic info
    start_date: datetime
    end_date: datetime
    initial_balance: float
    final_balance: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float

    # Profit metrics
    total_profit: float
    total_loss: float
    net_profit: float
    profit_factor: float

    # Risk metrics
    max_drawdown: float
    max_drawdown_pct: float
    max_drawdown_duration: timedelta
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float

    # Trade analysis
    avg_win: float
    avg_loss: float
    avg_trade: float
    largest_win: float
    largest_loss: float
    avg_holding_time: timedelta
    avg_winner_holding_time: timedelta
    avg_loser_holding_time: timedelta

    # Risk/Reward
    avg_rr_ratio: float
    expectancy: float

    # By strategy breakdown
    strategy_performance: Dict

    # By pair breakdown
    pair_performance: Dict

    # Monthly returns
    monthly_returns: Dict

    # Equity curve
    equity_curve: List[Tuple[datetime, float]]

    # All trades
    trades: List[BacktestTrade]

    # Additional metrics
    consecutive_wins: int
    consecutive_losses: int
    recovery_factor: float
    ulcer_index: float


class BacktestEngine:
    """
    Main backtesting engine for strategy validation.

    Usage:
        engine = BacktestEngine(initial_balance=10000)
        engine.load_data(data_dict)
        engine.add_strategy(my_strategy)
        result = engine.run()
        engine.generate_report(result)
    """

    def __init__(
        self,
        initial_balance: float = 10000,
        commission_per_trade: float = 0.0,
        spread_pips: float = 1.0,
        risk_per_trade: float = 0.02,
        max_positions: int = 5,
        pip_value: float = 0.0001
    ):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.commission = commission_per_trade
        self.spread_pips = spread_pips
        self.risk_per_trade = risk_per_trade
        self.max_positions = max_positions
        self.default_pip_value = pip_value

        # Data storage
        self.data: Dict[str, Dict[str, pd.DataFrame]] = {}  # pair -> timeframe -> df
        self.strategies: List = []

        # Trade tracking
        self.trades: List[BacktestTrade] = []
        self.open_positions: List[BacktestTrade] = []
        self.trade_counter = 0

        # Equity tracking
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.peak_equity = initial_balance
        self.max_drawdown = 0
        self.drawdown_start = None
        self.max_drawdown_duration = timedelta(0)

        logger.info(f"BacktestEngine initialized with ${initial_balance} balance")

    def load_data(
        self,
        data: Dict[str, Dict[str, pd.DataFrame]],
        validate: bool = True
    ):
        """
        Load historical data for backtesting.

        Args:
            data: Nested dict {pair: {timeframe: DataFrame}}
            validate: Whether to validate data integrity
        """
        self.data = data

        if validate:
            self._validate_data()

        # Get date range
        all_dates = []
        for pair_data in data.values():
            for df in pair_data.values():
                if len(df) > 0:
                    all_dates.extend(df.index.tolist())

        if all_dates:
            self.start_date = min(all_dates)
            self.end_date = max(all_dates)
            logger.info(f"Loaded data from {self.start_date} to {self.end_date}")

    def _validate_data(self):
        """Validate loaded data for common issues."""
        for pair, timeframes in self.data.items():
            for tf, df in timeframes.items():
                # Check for required columns
                required = ['open', 'high', 'low', 'close']
                alt_required = ['Open', 'High', 'Low', 'Close']

                has_lower = all(col in df.columns for col in required)
                has_upper = all(col in df.columns for col in alt_required)

                if not has_lower and not has_upper:
                    raise ValueError(f"Missing OHLC columns for {pair} {tf}")

                # Check for NaN values
                nan_count = df.isnull().sum().sum()
                if nan_count > 0:
                    logger.warning(f"{pair} {tf}: Found {nan_count} NaN values")

                # Check for duplicate indices
                dup_count = df.index.duplicated().sum()
                if dup_count > 0:
                    logger.warning(f"{pair} {tf}: Found {dup_count} duplicate indices")

    def add_strategy(self, strategy):
        """Add a strategy to backtest."""
        self.strategies.append(strategy)
        logger.info(f"Added strategy: {strategy.name}")

    def get_pip_value(self, pair: str) -> float:
        """Get pip value for a pair."""
        if 'JPY' in pair:
            return 0.01
        return self.default_pip_value

    def run(
        self,
        pairs: Optional[List[str]] = None,
        timeframes: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> BacktestResult:
        """
        Run the backtest simulation.

        Args:
            pairs: List of pairs to test (None = all)
            timeframes: List of timeframes to test (None = all)
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            BacktestResult with all statistics
        """
        logger.info("Starting backtest simulation...")

        # Reset state
        self.balance = self.initial_balance
        self.trades = []
        self.open_positions = []
        self.trade_counter = 0
        self.equity_curve = []
        self.peak_equity = self.initial_balance
        self.max_drawdown = 0

        # Filter pairs and timeframes
        test_pairs = pairs or list(self.data.keys())

        # Build unified timeline
        timeline = self._build_timeline(test_pairs, timeframes, start_date, end_date)

        # Main simulation loop
        for timestamp in timeline:
            # Update open positions
            self._update_positions(timestamp)

            # Check for new signals
            if len(self.open_positions) < self.max_positions:
                for pair in test_pairs:
                    for strategy in self.strategies:
                        signal = self._check_signal(
                            pair, strategy, timestamp, timeframes
                        )
                        if signal:
                            self._open_trade(signal, timestamp)

            # Record equity
            equity = self._calculate_equity(timestamp)
            self.equity_curve.append((timestamp, equity))

            # Update drawdown
            self._update_drawdown(equity, timestamp)

        # Close remaining positions
        self._close_all_positions(timeline[-1] if timeline else datetime.now())

        # Calculate results
        result = self._calculate_results()

        logger.info(f"Backtest complete: {result.total_trades} trades, "
                   f"Net P/L: ${result.net_profit:.2f}")

        return result

    def _build_timeline(
        self,
        pairs: List[str],
        timeframes: Optional[List[str]],
        start_date: Optional[datetime],
        end_date: Optional[datetime]
    ) -> List[datetime]:
        """Build unified timeline from all data."""
        all_timestamps = set()

        for pair in pairs:
            if pair not in self.data:
                continue

            pair_data = self.data[pair]
            tfs = timeframes or list(pair_data.keys())

            for tf in tfs:
                if tf not in pair_data:
                    continue

                df = pair_data[tf]
                timestamps = df.index.tolist()
                all_timestamps.update(timestamps)

        # Sort and filter
        timeline = sorted(all_timestamps)

        if start_date:
            timeline = [t for t in timeline if t >= start_date]
        if end_date:
            timeline = [t for t in timeline if t <= end_date]

        return timeline

    def _check_signal(
        self,
        pair: str,
        strategy,
        timestamp: datetime,
        timeframes: Optional[List[str]]
    ) -> Optional[Dict]:
        """Check if strategy generates a signal at this timestamp."""
        if pair not in self.data:
            return None

        pair_data = self.data[pair]
        tfs = timeframes or list(pair_data.keys())

        # Get primary timeframe data up to current timestamp
        primary_tf = tfs[0] if tfs else 'M15'

        if primary_tf not in pair_data:
            return None

        df = pair_data[primary_tf]

        # Filter data up to current timestamp
        df_slice = df[df.index <= timestamp].tail(200)  # Last 200 candles

        if len(df_slice) < 50:
            return None

        try:
            # Call strategy's generate_signal or analyze method
            if hasattr(strategy, 'generate_signal'):
                signal = strategy.generate_signal(df_slice, pair, primary_tf)
            elif hasattr(strategy, 'analyze'):
                # Build multi-TF data
                mtf_data = {}
                for tf in tfs:
                    if tf in pair_data:
                        tf_df = pair_data[tf]
                        mtf_data[tf] = tf_df[tf_df.index <= timestamp].tail(200)
                signal = strategy.analyze(mtf_data, pair)
            else:
                return None

            if signal:
                return {
                    'pair': pair,
                    'direction': signal.direction if hasattr(signal, 'direction') else signal.get('direction'),
                    'entry_price': signal.entry_price if hasattr(signal, 'entry_price') else signal.get('entry_price'),
                    'stop_loss': signal.stop_loss if hasattr(signal, 'stop_loss') else signal.get('stop_loss'),
                    'take_profit': signal.take_profit if hasattr(signal, 'take_profit') else signal.get('take_profit'),
                    'confidence': signal.confidence if hasattr(signal, 'confidence') else signal.get('confidence', 70),
                    'strategy': strategy.name,
                    'timeframe': primary_tf
                }
        except Exception as e:
            logger.debug(f"Signal check error for {pair}: {e}")

        return None

    def _open_trade(self, signal: Dict, timestamp: datetime):
        """Open a new trade."""
        # Check if we already have a position in this pair
        for pos in self.open_positions:
            if pos.pair == signal['pair']:
                return

        # Calculate position size based on risk
        pip_value = self.get_pip_value(signal['pair'])
        sl_pips = abs(signal['entry_price'] - signal['stop_loss']) / pip_value

        if sl_pips == 0:
            return

        risk_amount = self.balance * self.risk_per_trade
        position_size = risk_amount / (sl_pips * 10)  # Assuming $10 per pip for 1 lot
        position_size = round(position_size, 2)

        # Apply spread
        entry_price = signal['entry_price']
        if signal['direction'] == 'BUY':
            entry_price += self.spread_pips * pip_value
        else:
            entry_price -= self.spread_pips * pip_value

        self.trade_counter += 1

        trade = BacktestTrade(
            id=self.trade_counter,
            pair=signal['pair'],
            direction=TradeDirection[signal['direction']],
            entry_price=entry_price,
            stop_loss=signal['stop_loss'],
            take_profit=signal['take_profit'],
            position_size=position_size,
            entry_time=timestamp,
            strategy=signal['strategy'],
            timeframe=signal['timeframe'],
            confidence=signal['confidence']
        )

        self.open_positions.append(trade)

        # Deduct commission
        self.balance -= self.commission

        logger.debug(f"Opened {trade.direction.value} {trade.pair} @ {entry_price}")

    def _update_positions(self, timestamp: datetime):
        """Update all open positions with current prices."""
        positions_to_close = []

        for trade in self.open_positions:
            if trade.pair not in self.data:
                continue

            # Get current price from M1 or primary timeframe
            pair_data = self.data[trade.pair]

            # Try different timeframes for price
            current_price = None
            for tf in ['M1', 'M5', 'M15', 'H1']:
                if tf in pair_data:
                    df = pair_data[tf]
                    if timestamp in df.index:
                        current_price = df.loc[timestamp, 'close' if 'close' in df.columns else 'Close']
                        high = df.loc[timestamp, 'high' if 'high' in df.columns else 'High']
                        low = df.loc[timestamp, 'low' if 'low' in df.columns else 'Low']
                        break

            if current_price is None:
                continue

            pip_value = self.get_pip_value(trade.pair)

            # Calculate unrealized P/L
            if trade.direction == TradeDirection.BUY:
                unrealized_pips = (current_price - trade.entry_price) / pip_value
                # Check if TP or SL was hit
                if high >= trade.take_profit:
                    positions_to_close.append((trade, trade.take_profit, TradeStatus.CLOSED_TP))
                elif low <= trade.stop_loss:
                    positions_to_close.append((trade, trade.stop_loss, TradeStatus.CLOSED_SL))
            else:  # SELL
                unrealized_pips = (trade.entry_price - current_price) / pip_value
                if low <= trade.take_profit:
                    positions_to_close.append((trade, trade.take_profit, TradeStatus.CLOSED_TP))
                elif high >= trade.stop_loss:
                    positions_to_close.append((trade, trade.stop_loss, TradeStatus.CLOSED_SL))

            # Update MFE/MAE
            trade.max_favorable_excursion = max(trade.max_favorable_excursion, unrealized_pips)
            trade.max_adverse_excursion = min(trade.max_adverse_excursion, unrealized_pips)

        # Close positions
        for trade, exit_price, status in positions_to_close:
            self._close_trade(trade, exit_price, status, timestamp)

    def _close_trade(
        self,
        trade: BacktestTrade,
        exit_price: float,
        status: TradeStatus,
        timestamp: datetime
    ):
        """Close a trade and record results."""
        trade.exit_price = exit_price
        trade.exit_time = timestamp
        trade.status = status
        trade.holding_time = timestamp - trade.entry_time

        pip_value = self.get_pip_value(trade.pair)

        # Calculate P/L
        if trade.direction == TradeDirection.BUY:
            trade.pnl_pips = (exit_price - trade.entry_price) / pip_value
        else:
            trade.pnl_pips = (trade.entry_price - exit_price) / pip_value

        trade.pnl = trade.pnl_pips * trade.position_size * 10  # $10 per pip per lot

        # Update balance
        self.balance += trade.pnl - self.commission

        # Move to closed trades
        self.open_positions.remove(trade)
        self.trades.append(trade)

        logger.debug(
            f"Closed {trade.pair} {trade.direction.value}: "
            f"{trade.pnl_pips:.1f} pips, ${trade.pnl:.2f}"
        )

    def _close_all_positions(self, timestamp: datetime):
        """Close all remaining open positions."""
        for trade in list(self.open_positions):
            # Get last known price
            if trade.pair in self.data:
                pair_data = self.data[trade.pair]
                for tf in pair_data.values():
                    if len(tf) > 0:
                        exit_price = tf['close' if 'close' in tf.columns else 'Close'].iloc[-1]
                        self._close_trade(trade, exit_price, TradeStatus.CLOSED_TIME, timestamp)
                        break

    def _calculate_equity(self, timestamp: datetime) -> float:
        """Calculate current equity including unrealized P/L."""
        equity = self.balance

        for trade in self.open_positions:
            if trade.pair not in self.data:
                continue

            # Get current price
            pair_data = self.data[trade.pair]
            current_price = None

            for tf in pair_data.values():
                if timestamp in tf.index:
                    current_price = tf.loc[timestamp, 'close' if 'close' in tf.columns else 'Close']
                    break

            if current_price is None:
                continue

            pip_value = self.get_pip_value(trade.pair)

            if trade.direction == TradeDirection.BUY:
                unrealized_pips = (current_price - trade.entry_price) / pip_value
            else:
                unrealized_pips = (trade.entry_price - current_price) / pip_value

            unrealized_pnl = unrealized_pips * trade.position_size * 10
            equity += unrealized_pnl

        return equity

    def _update_drawdown(self, equity: float, timestamp: datetime):
        """Update drawdown tracking."""
        if equity > self.peak_equity:
            self.peak_equity = equity
            self.drawdown_start = None
        else:
            drawdown = self.peak_equity - equity
            drawdown_pct = drawdown / self.peak_equity * 100

            if drawdown > self.max_drawdown:
                self.max_drawdown = drawdown
                self.max_drawdown_pct = drawdown_pct

            if self.drawdown_start is None:
                self.drawdown_start = timestamp
            else:
                duration = timestamp - self.drawdown_start
                if duration > self.max_drawdown_duration:
                    self.max_drawdown_duration = duration

    def _calculate_results(self) -> BacktestResult:
        """Calculate all backtest statistics."""
        if not self.trades:
            return self._empty_result()

        # Basic stats
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl <= 0]

        total_profit = sum(t.pnl for t in winning_trades)
        total_loss = abs(sum(t.pnl for t in losing_trades))
        net_profit = total_profit - total_loss

        win_rate = len(winning_trades) / len(self.trades) * 100 if self.trades else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # Average stats
        avg_win = total_profit / len(winning_trades) if winning_trades else 0
        avg_loss = total_loss / len(losing_trades) if losing_trades else 0
        avg_trade = net_profit / len(self.trades) if self.trades else 0

        largest_win = max((t.pnl for t in self.trades), default=0)
        largest_loss = min((t.pnl for t in self.trades), default=0)

        # Holding times
        all_holding = [t.holding_time for t in self.trades if t.holding_time]
        winner_holding = [t.holding_time for t in winning_trades if t.holding_time]
        loser_holding = [t.holding_time for t in losing_trades if t.holding_time]

        avg_holding_time = sum(all_holding, timedelta()) / len(all_holding) if all_holding else timedelta()
        avg_winner_holding = sum(winner_holding, timedelta()) / len(winner_holding) if winner_holding else timedelta()
        avg_loser_holding = sum(loser_holding, timedelta()) / len(loser_holding) if loser_holding else timedelta()

        # Risk/Reward
        avg_rr = avg_win / avg_loss if avg_loss > 0 else 0
        expectancy = (win_rate / 100 * avg_win) - ((1 - win_rate / 100) * avg_loss)

        # Sharpe & Sortino
        returns = [t.pnl / self.initial_balance for t in self.trades]
        sharpe = self._calculate_sharpe(returns)
        sortino = self._calculate_sortino(returns)

        # Calmar ratio
        calmar = (net_profit / self.initial_balance) / (self.max_drawdown / self.initial_balance) \
                 if self.max_drawdown > 0 else 0

        # Consecutive wins/losses
        consecutive_wins, consecutive_losses = self._calculate_consecutive()

        # Recovery factor
        recovery_factor = net_profit / self.max_drawdown if self.max_drawdown > 0 else 0

        # Ulcer Index
        ulcer_index = self._calculate_ulcer_index()

        # Strategy breakdown
        strategy_performance = self._calculate_strategy_breakdown()

        # Pair breakdown
        pair_performance = self._calculate_pair_breakdown()

        # Monthly returns
        monthly_returns = self._calculate_monthly_returns()

        return BacktestResult(
            start_date=self.equity_curve[0][0] if self.equity_curve else datetime.now(),
            end_date=self.equity_curve[-1][0] if self.equity_curve else datetime.now(),
            initial_balance=self.initial_balance,
            final_balance=self.balance,
            total_trades=len(self.trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            win_rate=win_rate,
            total_profit=total_profit,
            total_loss=total_loss,
            net_profit=net_profit,
            profit_factor=profit_factor,
            max_drawdown=self.max_drawdown,
            max_drawdown_pct=getattr(self, 'max_drawdown_pct', 0),
            max_drawdown_duration=self.max_drawdown_duration,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_trade=avg_trade,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_holding_time=avg_holding_time,
            avg_winner_holding_time=avg_winner_holding,
            avg_loser_holding_time=avg_loser_holding,
            avg_rr_ratio=avg_rr,
            expectancy=expectancy,
            strategy_performance=strategy_performance,
            pair_performance=pair_performance,
            monthly_returns=monthly_returns,
            equity_curve=self.equity_curve,
            trades=self.trades,
            consecutive_wins=consecutive_wins,
            consecutive_losses=consecutive_losses,
            recovery_factor=recovery_factor,
            ulcer_index=ulcer_index
        )

    def _calculate_sharpe(self, returns: List[float], risk_free: float = 0.02) -> float:
        """Calculate Sharpe Ratio."""
        if not returns or len(returns) < 2:
            return 0

        avg_return = np.mean(returns)
        std_return = np.std(returns)

        if std_return == 0:
            return 0

        # Annualize (assuming ~250 trading days)
        annual_return = avg_return * 250
        annual_std = std_return * np.sqrt(250)

        return (annual_return - risk_free) / annual_std

    def _calculate_sortino(self, returns: List[float], risk_free: float = 0.02) -> float:
        """Calculate Sortino Ratio (only downside deviation)."""
        if not returns or len(returns) < 2:
            return 0

        avg_return = np.mean(returns)
        negative_returns = [r for r in returns if r < 0]

        if not negative_returns:
            return float('inf')

        downside_std = np.std(negative_returns)

        if downside_std == 0:
            return 0

        annual_return = avg_return * 250
        annual_downside = downside_std * np.sqrt(250)

        return (annual_return - risk_free) / annual_downside

    def _calculate_consecutive(self) -> Tuple[int, int]:
        """Calculate max consecutive wins and losses."""
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in self.trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    def _calculate_ulcer_index(self) -> float:
        """Calculate Ulcer Index (measure of downside volatility)."""
        if not self.equity_curve:
            return 0

        equities = [e[1] for e in self.equity_curve]
        peak = equities[0]
        drawdowns = []

        for equity in equities:
            if equity > peak:
                peak = equity
            dd_pct = (peak - equity) / peak * 100
            drawdowns.append(dd_pct ** 2)

        return np.sqrt(np.mean(drawdowns)) if drawdowns else 0

    def _calculate_strategy_breakdown(self) -> Dict:
        """Calculate performance by strategy."""
        breakdown = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'losses': 0,
            'profit': 0, 'loss': 0, 'pips': 0
        })

        for trade in self.trades:
            strat = trade.strategy
            breakdown[strat]['trades'] += 1
            breakdown[strat]['pips'] += trade.pnl_pips

            if trade.pnl > 0:
                breakdown[strat]['wins'] += 1
                breakdown[strat]['profit'] += trade.pnl
            else:
                breakdown[strat]['losses'] += 1
                breakdown[strat]['loss'] += abs(trade.pnl)

        # Calculate win rates
        for strat in breakdown:
            total = breakdown[strat]['trades']
            if total > 0:
                breakdown[strat]['win_rate'] = breakdown[strat]['wins'] / total * 100
                breakdown[strat]['net'] = breakdown[strat]['profit'] - breakdown[strat]['loss']

        return dict(breakdown)

    def _calculate_pair_breakdown(self) -> Dict:
        """Calculate performance by pair."""
        breakdown = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'losses': 0,
            'profit': 0, 'loss': 0, 'pips': 0
        })

        for trade in self.trades:
            pair = trade.pair
            breakdown[pair]['trades'] += 1
            breakdown[pair]['pips'] += trade.pnl_pips

            if trade.pnl > 0:
                breakdown[pair]['wins'] += 1
                breakdown[pair]['profit'] += trade.pnl
            else:
                breakdown[pair]['losses'] += 1
                breakdown[pair]['loss'] += abs(trade.pnl)

        for pair in breakdown:
            total = breakdown[pair]['trades']
            if total > 0:
                breakdown[pair]['win_rate'] = breakdown[pair]['wins'] / total * 100
                breakdown[pair]['net'] = breakdown[pair]['profit'] - breakdown[pair]['loss']

        return dict(breakdown)

    def _calculate_monthly_returns(self) -> Dict:
        """Calculate monthly returns."""
        monthly = defaultdict(float)

        for trade in self.trades:
            if trade.exit_time:
                month_key = trade.exit_time.strftime('%Y-%m')
                monthly[month_key] += trade.pnl

        return dict(monthly)

    def _empty_result(self) -> BacktestResult:
        """Return empty result when no trades."""
        now = datetime.now()
        return BacktestResult(
            start_date=now, end_date=now,
            initial_balance=self.initial_balance,
            final_balance=self.balance,
            total_trades=0, winning_trades=0, losing_trades=0,
            win_rate=0, total_profit=0, total_loss=0, net_profit=0,
            profit_factor=0, max_drawdown=0, max_drawdown_pct=0,
            max_drawdown_duration=timedelta(0),
            sharpe_ratio=0, sortino_ratio=0, calmar_ratio=0,
            avg_win=0, avg_loss=0, avg_trade=0,
            largest_win=0, largest_loss=0,
            avg_holding_time=timedelta(0),
            avg_winner_holding_time=timedelta(0),
            avg_loser_holding_time=timedelta(0),
            avg_rr_ratio=0, expectancy=0,
            strategy_performance={}, pair_performance={},
            monthly_returns={}, equity_curve=[],
            trades=[], consecutive_wins=0, consecutive_losses=0,
            recovery_factor=0, ulcer_index=0
        )

    def generate_report(self, result: BacktestResult, output_path: Optional[str] = None) -> str:
        """Generate a detailed text report of backtest results."""
        report = []
        report.append("=" * 70)
        report.append("                    BACKTEST REPORT")
        report.append("=" * 70)
        report.append("")

        # Period
        report.append(f"Period: {result.start_date} to {result.end_date}")
        report.append(f"Initial Balance: ${result.initial_balance:,.2f}")
        report.append(f"Final Balance: ${result.final_balance:,.2f}")
        report.append("")

        # Performance Summary
        report.append("-" * 70)
        report.append("PERFORMANCE SUMMARY")
        report.append("-" * 70)
        report.append(f"Net Profit: ${result.net_profit:,.2f} ({result.net_profit/result.initial_balance*100:.2f}%)")
        report.append(f"Total Profit: ${result.total_profit:,.2f}")
        report.append(f"Total Loss: ${result.total_loss:,.2f}")
        report.append(f"Profit Factor: {result.profit_factor:.2f}")
        report.append("")

        # Trade Statistics
        report.append("-" * 70)
        report.append("TRADE STATISTICS")
        report.append("-" * 70)
        report.append(f"Total Trades: {result.total_trades}")
        report.append(f"Winning Trades: {result.winning_trades} ({result.win_rate:.1f}%)")
        report.append(f"Losing Trades: {result.losing_trades}")
        report.append(f"Average Win: ${result.avg_win:.2f}")
        report.append(f"Average Loss: ${result.avg_loss:.2f}")
        report.append(f"Largest Win: ${result.largest_win:.2f}")
        report.append(f"Largest Loss: ${result.largest_loss:.2f}")
        report.append(f"Average R:R Ratio: {result.avg_rr_ratio:.2f}")
        report.append(f"Expectancy: ${result.expectancy:.2f}")
        report.append(f"Consecutive Wins: {result.consecutive_wins}")
        report.append(f"Consecutive Losses: {result.consecutive_losses}")
        report.append("")

        # Risk Metrics
        report.append("-" * 70)
        report.append("RISK METRICS")
        report.append("-" * 70)
        report.append(f"Max Drawdown: ${result.max_drawdown:.2f} ({result.max_drawdown_pct:.2f}%)")
        report.append(f"Max DD Duration: {result.max_drawdown_duration}")
        report.append(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
        report.append(f"Sortino Ratio: {result.sortino_ratio:.2f}")
        report.append(f"Calmar Ratio: {result.calmar_ratio:.2f}")
        report.append(f"Recovery Factor: {result.recovery_factor:.2f}")
        report.append(f"Ulcer Index: {result.ulcer_index:.2f}")
        report.append("")

        # Holding Times
        report.append("-" * 70)
        report.append("HOLDING TIMES")
        report.append("-" * 70)
        report.append(f"Average Holding Time: {result.avg_holding_time}")
        report.append(f"Avg Winner Holding: {result.avg_winner_holding_time}")
        report.append(f"Avg Loser Holding: {result.avg_loser_holding_time}")
        report.append("")

        # Strategy Breakdown
        if result.strategy_performance:
            report.append("-" * 70)
            report.append("STRATEGY BREAKDOWN")
            report.append("-" * 70)
            for strat, stats in result.strategy_performance.items():
                report.append(f"\n{strat}:")
                report.append(f"  Trades: {stats['trades']} | Win Rate: {stats.get('win_rate', 0):.1f}%")
                report.append(f"  Net P/L: ${stats.get('net', 0):.2f} | Pips: {stats['pips']:.1f}")

        # Pair Breakdown
        if result.pair_performance:
            report.append("")
            report.append("-" * 70)
            report.append("PAIR BREAKDOWN")
            report.append("-" * 70)
            for pair, stats in sorted(result.pair_performance.items(),
                                     key=lambda x: x[1].get('net', 0), reverse=True):
                report.append(f"{pair}: {stats['trades']} trades | "
                            f"Win: {stats.get('win_rate', 0):.0f}% | "
                            f"Net: ${stats.get('net', 0):.2f}")

        # Monthly Returns
        if result.monthly_returns:
            report.append("")
            report.append("-" * 70)
            report.append("MONTHLY RETURNS")
            report.append("-" * 70)
            for month, pnl in sorted(result.monthly_returns.items()):
                bar = "+" * int(abs(pnl) / 50) if pnl > 0 else "-" * int(abs(pnl) / 50)
                report.append(f"{month}: ${pnl:>10,.2f} {bar}")

        report.append("")
        report.append("=" * 70)

        report_text = "\n".join(report)

        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Report saved to {output_path}")

        return report_text

    def export_trades(self, filepath: str):
        """Export all trades to CSV."""
        if not self.trades:
            logger.warning("No trades to export")
            return

        data = []
        for t in self.trades:
            data.append({
                'id': t.id,
                'pair': t.pair,
                'direction': t.direction.value,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'stop_loss': t.stop_loss,
                'take_profit': t.take_profit,
                'position_size': t.position_size,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'status': t.status.value,
                'pnl': t.pnl,
                'pnl_pips': t.pnl_pips,
                'strategy': t.strategy,
                'timeframe': t.timeframe,
                'confidence': t.confidence,
                'mfe': t.max_favorable_excursion,
                'mae': t.max_adverse_excursion
            })

        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(self.trades)} trades to {filepath}")


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization for robust strategy validation.

    Splits data into in-sample (training) and out-of-sample (testing) periods,
    optimizes on in-sample, validates on out-of-sample.
    """

    def __init__(
        self,
        engine: BacktestEngine,
        in_sample_pct: float = 0.7,
        num_folds: int = 5
    ):
        self.engine = engine
        self.in_sample_pct = in_sample_pct
        self.num_folds = num_folds
        self.results: List[BacktestResult] = []

    def run(self, parameter_grid: Dict) -> Dict:
        """
        Run walk-forward optimization.

        Args:
            parameter_grid: Dict of parameters to optimize

        Returns:
            Optimization results
        """
        logger.info(f"Starting Walk-Forward Optimization with {self.num_folds} folds")

        # TODO: Implement walk-forward logic
        # This is a placeholder for the full implementation

        return {
            'best_params': {},
            'fold_results': [],
            'aggregate_performance': {}
        }


class MonteCarloSimulator:
    """
    Monte Carlo simulation for risk analysis.

    Randomizes trade order to estimate range of possible outcomes.
    """

    def __init__(self, trades: List[BacktestTrade], initial_balance: float = 10000):
        self.trades = trades
        self.initial_balance = initial_balance
        self.simulation_results: List[Dict] = []

    def run(self, num_simulations: int = 1000) -> Dict:
        """
        Run Monte Carlo simulation.

        Args:
            num_simulations: Number of random simulations

        Returns:
            Distribution of outcomes
        """
        logger.info(f"Running Monte Carlo simulation with {num_simulations} iterations")

        if not self.trades:
            return {'error': 'No trades to simulate'}

        final_balances = []
        max_drawdowns = []

        trade_pnls = [t.pnl for t in self.trades]

        for _ in range(num_simulations):
            # Shuffle trade order
            shuffled_pnls = np.random.permutation(trade_pnls)

            # Simulate equity curve
            balance = self.initial_balance
            peak = balance
            max_dd = 0

            for pnl in shuffled_pnls:
                balance += pnl
                if balance > peak:
                    peak = balance
                dd = (peak - balance) / peak * 100
                max_dd = max(max_dd, dd)

            final_balances.append(balance)
            max_drawdowns.append(max_dd)

        self.simulation_results = {
            'final_balances': final_balances,
            'max_drawdowns': max_drawdowns
        }

        return {
            'mean_final_balance': np.mean(final_balances),
            'median_final_balance': np.median(final_balances),
            'std_final_balance': np.std(final_balances),
            'percentile_5': np.percentile(final_balances, 5),
            'percentile_95': np.percentile(final_balances, 95),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_case_drawdown': np.max(max_drawdowns),
            'probability_of_loss': sum(1 for b in final_balances if b < self.initial_balance) / num_simulations * 100
        }


if __name__ == "__main__":
    # Example usage
    print("Backtesting Engine - Example Usage")
    print("=" * 50)

    # Create engine
    engine = BacktestEngine(
        initial_balance=10000,
        risk_per_trade=0.02,
        spread_pips=1.0
    )

    print("\nBacktest Engine initialized.")
    print("To use:")
    print("  1. engine.load_data(your_data_dict)")
    print("  2. engine.add_strategy(your_strategy)")
    print("  3. result = engine.run()")
    print("  4. engine.generate_report(result)")
