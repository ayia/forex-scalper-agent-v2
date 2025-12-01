"""
Position Manager - Gestion des positions ouvertes
==================================================
Gère le cycle de vie complet des positions:
- Tracking des positions ouvertes
- Trailing stops dynamiques (2x ATR)
- Breakeven stops (à 1.5R)
- Partial take-profit (50% à 1R)
- Monitoring en temps réel des prix
"""
import logging
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
import pandas as pd

from config import get_pip_value

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Représente une position ouverte."""
    id: str
    pair: str
    direction: str  # 'BUY' or 'SELL'
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    initial_risk: float
    atr: float
    timestamp: str
    strategy: str
    timeframe: str

    # Trailing stop parameters
    trailing_stop_distance: float = 0.0  # Distance en prix (2x ATR)
    highest_price: float = 0.0  # Pour BUY positions
    lowest_price: float = 0.0   # Pour SELL positions

    # Breakeven tracking
    breakeven_activated: bool = False
    breakeven_level: float = 0.0  # Entry + spread

    # Partial TP tracking
    partial_tp_taken: bool = False
    partial_tp_level: float = 0.0  # 1R profit level
    remaining_size: float = 0.0

    # P&L tracking
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0

    def __post_init__(self):
        """Initialize calculated fields."""
        # Set trailing stop distance (2x ATR)
        self.trailing_stop_distance = self.atr * 2.0

        # Initialize price tracking
        if self.direction == 'BUY':
            self.highest_price = self.entry_price
        else:
            self.lowest_price = self.entry_price

        # Calculate breakeven level (entry + spread)
        pip_value = get_pip_value(self.pair)
        spread_price = 1.0 * pip_value  # Assume 1 pip spread
        if self.direction == 'BUY':
            self.breakeven_level = self.entry_price + spread_price
        else:
            self.breakeven_level = self.entry_price - spread_price

        # Calculate partial TP level (1R = initial risk)
        risk_distance = abs(self.entry_price - self.stop_loss)
        if self.direction == 'BUY':
            self.partial_tp_level = self.entry_price + risk_distance
        else:
            self.partial_tp_level = self.entry_price - risk_distance

        # Initialize remaining size
        self.remaining_size = self.position_size

    def update_price(self, current_price: float) -> Dict:
        """
        Update position with current price and check for exits.

        Returns:
            Dict with actions: {'update_sl': new_sl, 'partial_close': True, 'full_close': True}
        """
        self.current_price = current_price
        actions = {}

        # Update unrealized P&L
        if self.direction == 'BUY':
            self.unrealized_pnl = (current_price - self.entry_price) * self.remaining_size
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.remaining_size

        # Check for stop loss hit
        if self.direction == 'BUY' and current_price <= self.stop_loss:
            actions['full_close'] = True
            actions['reason'] = 'Stop Loss Hit'
            return actions
        elif self.direction == 'SELL' and current_price >= self.stop_loss:
            actions['full_close'] = True
            actions['reason'] = 'Stop Loss Hit'
            return actions

        # Check for take profit hit
        if self.direction == 'BUY' and current_price >= self.take_profit:
            actions['full_close'] = True
            actions['reason'] = 'Take Profit Hit'
            return actions
        elif self.direction == 'SELL' and current_price <= self.take_profit:
            actions['full_close'] = True
            actions['reason'] = 'Take Profit Hit'
            return actions

        # 1. Check for PARTIAL TAKE PROFIT (50% at 1R)
        if not self.partial_tp_taken:
            partial_tp_hit = False
            if self.direction == 'BUY' and current_price >= self.partial_tp_level:
                partial_tp_hit = True
            elif self.direction == 'SELL' and current_price <= self.partial_tp_level:
                partial_tp_hit = True

            if partial_tp_hit:
                actions['partial_close'] = True
                actions['close_percent'] = 50  # Close 50%
                self.partial_tp_taken = True
                logger.info(f"{self.pair}: Partial TP hit at {current_price:.5f} (1R profit)")

        # 2. Check for BREAKEVEN (move SL to entry at 1.5R)
        if not self.breakeven_activated and self.partial_tp_taken:
            # Calculate 1.5R level
            risk_distance = abs(self.entry_price - self.stop_loss)
            if self.direction == 'BUY':
                breakeven_trigger = self.entry_price + (risk_distance * 1.5)
                if current_price >= breakeven_trigger:
                    actions['update_sl'] = self.breakeven_level
                    self.stop_loss = self.breakeven_level
                    self.breakeven_activated = True
                    logger.info(f"{self.pair}: Breakeven activated - SL moved to {self.breakeven_level:.5f}")
            else:
                breakeven_trigger = self.entry_price - (risk_distance * 1.5)
                if current_price <= breakeven_trigger:
                    actions['update_sl'] = self.breakeven_level
                    self.stop_loss = self.breakeven_level
                    self.breakeven_activated = True
                    logger.info(f"{self.pair}: Breakeven activated - SL moved to {self.breakeven_level:.5f}")

        # 3. TRAILING STOP (after breakeven is activated)
        if self.breakeven_activated:
            if self.direction == 'BUY':
                # Update highest price
                if current_price > self.highest_price:
                    self.highest_price = current_price
                    # Calculate new trailing stop
                    new_trailing_sl = current_price - self.trailing_stop_distance
                    # Only move SL up, never down
                    if new_trailing_sl > self.stop_loss:
                        actions['update_sl'] = new_trailing_sl
                        self.stop_loss = new_trailing_sl
                        logger.info(f"{self.pair}: Trailing SL updated to {new_trailing_sl:.5f} "
                                   f"(2x ATR = {self.trailing_stop_distance:.5f})")
            else:  # SELL
                # Update lowest price
                if current_price < self.lowest_price:
                    self.lowest_price = current_price
                    # Calculate new trailing stop
                    new_trailing_sl = current_price + self.trailing_stop_distance
                    # Only move SL down, never up
                    if new_trailing_sl < self.stop_loss:
                        actions['update_sl'] = new_trailing_sl
                        self.stop_loss = new_trailing_sl
                        logger.info(f"{self.pair}: Trailing SL updated to {new_trailing_sl:.5f} "
                                   f"(2x ATR = {self.trailing_stop_distance:.5f})")

        return actions


class PositionManager:
    """Gère toutes les positions ouvertes."""

    def __init__(self, data_fetcher=None):
        """
        Initialize position manager.

        Args:
            data_fetcher: DataFetcher instance for real-time price updates
        """
        self.positions: Dict[str, Position] = {}  # position_id -> Position
        self.closed_positions: List[Position] = []
        self.data_fetcher = data_fetcher

        logger.info("PositionManager initialized")

    def open_position(self, signal: Dict) -> str:
        """
        Open a new position from a trading signal.

        Args:
            signal: Trading signal dict from scanner

        Returns:
            Position ID
        """
        # Generate unique position ID
        position_id = f"{signal['pair']}_{signal['timestamp']}"

        position = Position(
            id=position_id,
            pair=signal['pair'],
            direction=signal['direction'],
            entry_price=signal['entry_price'],
            current_price=signal['entry_price'],
            stop_loss=signal['stop_loss'],
            take_profit=signal['take_profit'],
            position_size=signal['position_size'],
            initial_risk=signal['risk_amount'],
            atr=signal['atr'],
            timestamp=signal['timestamp'],
            strategy=signal['strategy'],
            timeframe=signal['timeframe']
        )

        self.positions[position_id] = position

        logger.info(f"Position opened: {position_id} - {signal['direction']} {signal['pair']} "
                   f"@ {signal['entry_price']:.5f}, Size={signal['position_size']:.2f}")

        return position_id

    def update_positions(self) -> List[Dict]:
        """
        Update all open positions with current prices.

        Returns:
            List of actions taken (closes, SL updates)
        """
        if not self.data_fetcher:
            logger.warning("No data_fetcher provided - cannot update positions")
            return []

        actions_taken = []
        positions_to_close = []

        for position_id, position in self.positions.items():
            # Get current price
            try:
                current_data = self.data_fetcher.fetch_multi_timeframe(
                    position.pair,
                    [position.timeframe]
                )
                if not current_data or position.timeframe not in current_data:
                    continue

                df = current_data[position.timeframe]
                if df is None or df.empty:
                    continue

                current_price = df['close'].iloc[-1]

                # Update position
                actions = position.update_price(current_price)

                # Handle actions
                if actions.get('full_close'):
                    positions_to_close.append(position_id)
                    actions_taken.append({
                        'action': 'close',
                        'position_id': position_id,
                        'reason': actions.get('reason'),
                        'price': current_price,
                        'pnl': position.unrealized_pnl
                    })
                    logger.info(f"Position closed: {position_id} - {actions.get('reason')} "
                               f"@ {current_price:.5f}, P&L=${position.unrealized_pnl:.2f}")

                elif actions.get('partial_close'):
                    # Close 50% of position
                    close_size = position.position_size * 0.5
                    close_pnl = position.unrealized_pnl * 0.5
                    position.remaining_size = position.position_size * 0.5
                    position.realized_pnl += close_pnl

                    actions_taken.append({
                        'action': 'partial_close',
                        'position_id': position_id,
                        'percent': 50,
                        'price': current_price,
                        'pnl': close_pnl
                    })
                    logger.info(f"Partial close: {position_id} - 50% @ {current_price:.5f}, "
                               f"P&L=${close_pnl:.2f}")

                elif actions.get('update_sl'):
                    actions_taken.append({
                        'action': 'update_sl',
                        'position_id': position_id,
                        'old_sl': position.stop_loss,
                        'new_sl': actions['update_sl']
                    })

            except Exception as e:
                logger.error(f"Error updating position {position_id}: {e}")

        # Close positions
        for position_id in positions_to_close:
            position = self.positions.pop(position_id)
            self.closed_positions.append(position)

        return actions_taken

    def get_open_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.positions.values())

    def get_position_count(self) -> int:
        """Get number of open positions."""
        return len(self.positions)

    def get_total_exposure(self) -> float:
        """Get total position size across all open positions."""
        return sum(pos.remaining_size for pos in self.positions.values())

    def get_total_risk(self) -> float:
        """Get total risk across all open positions."""
        return sum(pos.initial_risk for pos in self.positions.values())

    def get_unrealized_pnl(self) -> float:
        """Get total unrealized P&L."""
        return sum(pos.unrealized_pnl for pos in self.positions.values())

    def get_realized_pnl(self) -> float:
        """Get total realized P&L."""
        closed_pnl = sum(pos.unrealized_pnl for pos in self.closed_positions)
        open_realized_pnl = sum(pos.realized_pnl for pos in self.positions.values())
        return closed_pnl + open_realized_pnl

    def get_statistics(self) -> Dict:
        """Get position statistics."""
        return {
            'open_positions': self.get_position_count(),
            'total_exposure': self.get_total_exposure(),
            'total_risk': self.get_total_risk(),
            'unrealized_pnl': self.get_unrealized_pnl(),
            'realized_pnl': self.get_realized_pnl(),
            'total_pnl': self.get_unrealized_pnl() + self.get_realized_pnl(),
            'closed_positions': len(self.closed_positions)
        }

    def close_position(self, position_id: str, reason: str = "Manual close") -> Optional[Dict]:
        """
        Manually close a position.

        Args:
            position_id: Position ID to close
            reason: Reason for closing

        Returns:
            Close summary dict or None
        """
        if position_id not in self.positions:
            logger.warning(f"Position {position_id} not found")
            return None

        position = self.positions.pop(position_id)
        self.closed_positions.append(position)

        close_summary = {
            'position_id': position_id,
            'pair': position.pair,
            'direction': position.direction,
            'entry_price': position.entry_price,
            'exit_price': position.current_price,
            'pnl': position.unrealized_pnl + position.realized_pnl,
            'reason': reason
        }

        logger.info(f"Position manually closed: {position_id} - {reason}, "
                   f"P&L=${close_summary['pnl']:.2f}")

        return close_summary

    def close_all_positions(self, reason: str = "Close all") -> List[Dict]:
        """Close all open positions."""
        closed = []
        for position_id in list(self.positions.keys()):
            close_summary = self.close_position(position_id, reason)
            if close_summary:
                closed.append(close_summary)

        logger.info(f"Closed {len(closed)} positions - {reason}")
        return closed
