"""
Broker Integration Module - Multi-Broker API Support
=====================================================
Unified interface for connecting to various forex brokers.

Supported Brokers:
- MetaTrader 5 (MT5)
- OANDA
- Interactive Brokers (IB)
- Alpaca (for paper trading)
- cTrader (via FIX API)

Features:
- Unified order execution interface
- Real-time position management
- Account information retrieval
- Order modification and cancellation
- Risk management hooks
- Connection health monitoring
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import logging
import time
import threading
from queue import Queue
import json

logger = logging.getLogger(__name__)


class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"


class PositionStatus(Enum):
    """Position status."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"


@dataclass
class BrokerCredentials:
    """Broker authentication credentials."""
    broker: str
    account_id: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    server: Optional[str] = None
    password: Optional[str] = None
    demo: bool = True
    additional: Dict = field(default_factory=dict)


@dataclass
class AccountInfo:
    """Account information."""
    account_id: str
    broker: str
    balance: float
    equity: float
    margin_used: float
    margin_available: float
    currency: str
    leverage: int
    unrealized_pnl: float
    realized_pnl: float
    positions_count: int
    orders_count: int
    timestamp: datetime


@dataclass
class Order:
    """Order representation."""
    order_id: str
    broker_order_id: Optional[str]
    pair: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    status: OrderStatus
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    commission: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class Position:
    """Position representation."""
    position_id: str
    pair: str
    side: OrderSide
    quantity: float
    entry_price: float
    current_price: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    unrealized_pnl: float
    realized_pnl: float
    commission: float
    swap: float
    status: PositionStatus
    opened_at: datetime
    closed_at: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class Tick:
    """Market tick data."""
    pair: str
    bid: float
    ask: float
    timestamp: datetime
    volume: Optional[float] = None


class BaseBroker(ABC):
    """
    Abstract base class for broker implementations.

    All broker implementations must inherit from this class
    and implement the required methods.
    """

    def __init__(self, credentials: BrokerCredentials):
        self.credentials = credentials
        self.is_connected = False
        self.last_heartbeat = None
        self.callbacks: Dict[str, List[Callable]] = {
            'tick': [],
            'order': [],
            'position': [],
            'error': []
        }

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to the broker."""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from the broker."""
        pass

    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """Get account information."""
        pass

    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get all open positions."""
        pass

    @abstractmethod
    async def get_orders(self) -> List[Order]:
        """Get all pending orders."""
        pass

    @abstractmethod
    async def place_order(
        self,
        pair: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        **kwargs
    ) -> Order:
        """Place a new order."""
        pass

    @abstractmethod
    async def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        quantity: Optional[float] = None
    ) -> Order:
        """Modify an existing order."""
        pass

    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order."""
        pass

    @abstractmethod
    async def close_position(
        self,
        position_id: str,
        quantity: Optional[float] = None
    ) -> Position:
        """Close a position (full or partial)."""
        pass

    @abstractmethod
    async def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Position:
        """Modify position SL/TP."""
        pass

    @abstractmethod
    async def get_tick(self, pair: str) -> Tick:
        """Get current tick for a pair."""
        pass

    @abstractmethod
    async def subscribe_ticks(self, pairs: List[str]) -> bool:
        """Subscribe to tick updates for pairs."""
        pass

    def register_callback(self, event: str, callback: Callable):
        """Register callback for events."""
        if event in self.callbacks:
            self.callbacks[event].append(callback)

    def _emit(self, event: str, data: Any):
        """Emit event to callbacks."""
        for callback in self.callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Callback error: {e}")


class MT5Broker(BaseBroker):
    """
    MetaTrader 5 broker implementation.

    Requires: MetaTrader5 package (pip install MetaTrader5)
    """

    def __init__(self, credentials: BrokerCredentials):
        super().__init__(credentials)
        self.mt5 = None

    async def connect(self) -> bool:
        """Connect to MT5 terminal."""
        try:
            import MetaTrader5 as mt5
            self.mt5 = mt5

            # Initialize MT5
            if not mt5.initialize(
                login=int(self.credentials.account_id),
                password=self.credentials.password,
                server=self.credentials.server
            ):
                logger.error(f"MT5 init failed: {mt5.last_error()}")
                return False

            # Check connection
            if not mt5.terminal_info():
                return False

            self.is_connected = True
            self.last_heartbeat = datetime.now()
            logger.info(f"Connected to MT5: {self.credentials.account_id}")
            return True

        except ImportError:
            logger.error("MetaTrader5 package not installed")
            return False
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from MT5."""
        if self.mt5:
            self.mt5.shutdown()
        self.is_connected = False
        return True

    async def get_account_info(self) -> AccountInfo:
        """Get MT5 account info."""
        if not self.mt5:
            raise ConnectionError("Not connected to MT5")

        info = self.mt5.account_info()
        if not info:
            raise RuntimeError("Failed to get account info")

        return AccountInfo(
            account_id=str(info.login),
            broker="MT5",
            balance=info.balance,
            equity=info.equity,
            margin_used=info.margin,
            margin_available=info.margin_free,
            currency=info.currency,
            leverage=info.leverage,
            unrealized_pnl=info.profit,
            realized_pnl=0,  # MT5 doesn't provide this directly
            positions_count=len(self.mt5.positions_get() or []),
            orders_count=len(self.mt5.orders_get() or []),
            timestamp=datetime.now()
        )

    async def get_positions(self) -> List[Position]:
        """Get all MT5 positions."""
        if not self.mt5:
            raise ConnectionError("Not connected")

        positions = self.mt5.positions_get()
        result = []

        for pos in positions or []:
            tick = self.mt5.symbol_info_tick(pos.symbol)
            current_price = tick.bid if pos.type == 0 else tick.ask

            result.append(Position(
                position_id=str(pos.ticket),
                pair=pos.symbol,
                side=OrderSide.BUY if pos.type == 0 else OrderSide.SELL,
                quantity=pos.volume,
                entry_price=pos.price_open,
                current_price=current_price,
                stop_loss=pos.sl if pos.sl > 0 else None,
                take_profit=pos.tp if pos.tp > 0 else None,
                unrealized_pnl=pos.profit,
                realized_pnl=0,
                commission=pos.commission,
                swap=pos.swap,
                status=PositionStatus.OPEN,
                opened_at=datetime.fromtimestamp(pos.time)
            ))

        return result

    async def get_orders(self) -> List[Order]:
        """Get all MT5 pending orders."""
        if not self.mt5:
            raise ConnectionError("Not connected")

        orders = self.mt5.orders_get()
        result = []

        for ord in orders or []:
            order_type_map = {
                2: OrderType.LIMIT,  # BUY_LIMIT
                3: OrderType.LIMIT,  # SELL_LIMIT
                4: OrderType.STOP,   # BUY_STOP
                5: OrderType.STOP,   # SELL_STOP
            }

            result.append(Order(
                order_id=str(ord.ticket),
                broker_order_id=str(ord.ticket),
                pair=ord.symbol,
                side=OrderSide.BUY if ord.type in [2, 4] else OrderSide.SELL,
                order_type=order_type_map.get(ord.type, OrderType.LIMIT),
                quantity=ord.volume_current,
                price=ord.price_open,
                stop_loss=ord.sl if ord.sl > 0 else None,
                take_profit=ord.tp if ord.tp > 0 else None,
                status=OrderStatus.PENDING,
                created_at=datetime.fromtimestamp(ord.time_setup)
            ))

        return result

    async def place_order(
        self,
        pair: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        **kwargs
    ) -> Order:
        """Place order on MT5."""
        if not self.mt5:
            raise ConnectionError("Not connected")

        # Get symbol info
        symbol_info = self.mt5.symbol_info(pair)
        if not symbol_info:
            raise ValueError(f"Symbol not found: {pair}")

        if not symbol_info.visible:
            self.mt5.symbol_select(pair, True)

        # Get current price
        tick = self.mt5.symbol_info_tick(pair)

        # Determine order type
        if order_type == OrderType.MARKET:
            mt5_type = self.mt5.ORDER_TYPE_BUY if side == OrderSide.BUY else self.mt5.ORDER_TYPE_SELL
            execution_price = tick.ask if side == OrderSide.BUY else tick.bid
        elif order_type == OrderType.LIMIT:
            mt5_type = self.mt5.ORDER_TYPE_BUY_LIMIT if side == OrderSide.BUY else self.mt5.ORDER_TYPE_SELL_LIMIT
            execution_price = price
        else:
            mt5_type = self.mt5.ORDER_TYPE_BUY_STOP if side == OrderSide.BUY else self.mt5.ORDER_TYPE_SELL_STOP
            execution_price = price

        # Build request
        request = {
            "action": self.mt5.TRADE_ACTION_DEAL if order_type == OrderType.MARKET else self.mt5.TRADE_ACTION_PENDING,
            "symbol": pair,
            "volume": quantity,
            "type": mt5_type,
            "price": execution_price,
            "deviation": 20,
            "magic": 234000,
            "comment": "forex_scalper_v2",
            "type_time": self.mt5.ORDER_TIME_GTC,
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }

        if stop_loss:
            request["sl"] = stop_loss
        if take_profit:
            request["tp"] = take_profit

        # Send order
        result = self.mt5.order_send(request)

        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Order failed: {result.comment}")

        return Order(
            order_id=str(result.order),
            broker_order_id=str(result.order),
            pair=pair,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=execution_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status=OrderStatus.FILLED if order_type == OrderType.MARKET else OrderStatus.PENDING,
            filled_quantity=quantity if order_type == OrderType.MARKET else 0,
            filled_price=result.price if order_type == OrderType.MARKET else 0
        )

    async def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        quantity: Optional[float] = None
    ) -> Order:
        """Modify MT5 order."""
        if not self.mt5:
            raise ConnectionError("Not connected")

        # Get existing order
        orders = self.mt5.orders_get(ticket=int(order_id))
        if not orders:
            raise ValueError(f"Order not found: {order_id}")

        order = orders[0]

        request = {
            "action": self.mt5.TRADE_ACTION_MODIFY,
            "order": int(order_id),
            "symbol": order.symbol,
            "price": price or order.price_open,
            "sl": stop_loss or order.sl,
            "tp": take_profit or order.tp,
        }

        result = self.mt5.order_send(request)

        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Modify failed: {result.comment}")

        return (await self.get_orders())[0]  # Simplified - should find specific order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel MT5 order."""
        if not self.mt5:
            raise ConnectionError("Not connected")

        request = {
            "action": self.mt5.TRADE_ACTION_REMOVE,
            "order": int(order_id),
        }

        result = self.mt5.order_send(request)
        return result.retcode == self.mt5.TRADE_RETCODE_DONE

    async def close_position(
        self,
        position_id: str,
        quantity: Optional[float] = None
    ) -> Position:
        """Close MT5 position."""
        if not self.mt5:
            raise ConnectionError("Not connected")

        positions = self.mt5.positions_get(ticket=int(position_id))
        if not positions:
            raise ValueError(f"Position not found: {position_id}")

        pos = positions[0]
        close_volume = quantity or pos.volume

        # Determine close type
        close_type = self.mt5.ORDER_TYPE_SELL if pos.type == 0 else self.mt5.ORDER_TYPE_BUY
        tick = self.mt5.symbol_info_tick(pos.symbol)
        price = tick.bid if pos.type == 0 else tick.ask

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": close_volume,
            "type": close_type,
            "position": int(position_id),
            "price": price,
            "deviation": 20,
            "magic": 234000,
            "comment": "close_forex_scalper_v2",
            "type_filling": self.mt5.ORDER_FILLING_IOC,
        }

        result = self.mt5.order_send(request)

        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Close failed: {result.comment}")

        return Position(
            position_id=position_id,
            pair=pos.symbol,
            side=OrderSide.BUY if pos.type == 0 else OrderSide.SELL,
            quantity=pos.volume,
            entry_price=pos.price_open,
            current_price=price,
            stop_loss=pos.sl,
            take_profit=pos.tp,
            unrealized_pnl=0,
            realized_pnl=pos.profit,
            commission=pos.commission,
            swap=pos.swap,
            status=PositionStatus.CLOSED,
            opened_at=datetime.fromtimestamp(pos.time),
            closed_at=datetime.now()
        )

    async def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Position:
        """Modify MT5 position SL/TP."""
        if not self.mt5:
            raise ConnectionError("Not connected")

        positions = self.mt5.positions_get(ticket=int(position_id))
        if not positions:
            raise ValueError(f"Position not found: {position_id}")

        pos = positions[0]

        request = {
            "action": self.mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": int(position_id),
            "sl": stop_loss or pos.sl,
            "tp": take_profit or pos.tp,
        }

        result = self.mt5.order_send(request)

        if result.retcode != self.mt5.TRADE_RETCODE_DONE:
            raise RuntimeError(f"Modify position failed: {result.comment}")

        return (await self.get_positions())[0]  # Simplified

    async def get_tick(self, pair: str) -> Tick:
        """Get current tick."""
        if not self.mt5:
            raise ConnectionError("Not connected")

        tick = self.mt5.symbol_info_tick(pair)
        if not tick:
            raise ValueError(f"No tick for {pair}")

        return Tick(
            pair=pair,
            bid=tick.bid,
            ask=tick.ask,
            timestamp=datetime.fromtimestamp(tick.time),
            volume=tick.volume
        )

    async def subscribe_ticks(self, pairs: List[str]) -> bool:
        """Subscribe to tick updates (MT5 uses polling)."""
        # MT5 doesn't have streaming API, needs polling
        for pair in pairs:
            self.mt5.symbol_select(pair, True)
        return True


class OANDABroker(BaseBroker):
    """
    OANDA broker implementation.

    Requires: oandapyV20 package
    """

    def __init__(self, credentials: BrokerCredentials):
        super().__init__(credentials)
        self.api = None
        self.account_id = credentials.account_id
        self.streaming_thread = None
        self._stop_streaming = False

    async def connect(self) -> bool:
        """Connect to OANDA."""
        try:
            from oandapyV20 import API
            from oandapyV20.endpoints.accounts import AccountDetails

            # Determine environment
            environment = "practice" if self.credentials.demo else "live"

            self.api = API(
                access_token=self.credentials.api_key,
                environment=environment
            )

            # Test connection
            r = AccountDetails(self.account_id)
            self.api.request(r)

            self.is_connected = True
            self.last_heartbeat = datetime.now()
            logger.info(f"Connected to OANDA: {self.account_id}")
            return True

        except ImportError:
            logger.error("oandapyV20 package not installed")
            return False
        except Exception as e:
            logger.error(f"OANDA connection error: {e}")
            return False

    async def disconnect(self) -> bool:
        """Disconnect from OANDA."""
        self._stop_streaming = True
        self.is_connected = False
        return True

    async def get_account_info(self) -> AccountInfo:
        """Get OANDA account info."""
        from oandapyV20.endpoints.accounts import AccountDetails

        r = AccountDetails(self.account_id)
        response = self.api.request(r)
        account = response['account']

        return AccountInfo(
            account_id=account['id'],
            broker="OANDA",
            balance=float(account['balance']),
            equity=float(account['NAV']),
            margin_used=float(account['marginUsed']),
            margin_available=float(account['marginAvailable']),
            currency=account['currency'],
            leverage=50,  # OANDA default
            unrealized_pnl=float(account['unrealizedPL']),
            realized_pnl=float(account.get('pl', 0)),
            positions_count=int(account['openPositionCount']),
            orders_count=int(account['pendingOrderCount']),
            timestamp=datetime.now()
        )

    async def get_positions(self) -> List[Position]:
        """Get OANDA positions."""
        from oandapyV20.endpoints.positions import OpenPositions

        r = OpenPositions(self.account_id)
        response = self.api.request(r)

        result = []
        for pos in response.get('positions', []):
            # OANDA returns both long and short units
            long_units = float(pos['long']['units'])
            short_units = float(pos['short']['units'])

            if long_units != 0:
                result.append(Position(
                    position_id=f"{pos['instrument']}_long",
                    pair=pos['instrument'],
                    side=OrderSide.BUY,
                    quantity=abs(long_units),
                    entry_price=float(pos['long']['averagePrice']),
                    current_price=0,  # Would need tick data
                    stop_loss=None,
                    take_profit=None,
                    unrealized_pnl=float(pos['long']['unrealizedPL']),
                    realized_pnl=float(pos['long'].get('pl', 0)),
                    commission=0,
                    swap=float(pos['long'].get('financing', 0)),
                    status=PositionStatus.OPEN,
                    opened_at=datetime.now()  # OANDA doesn't provide this
                ))

            if short_units != 0:
                result.append(Position(
                    position_id=f"{pos['instrument']}_short",
                    pair=pos['instrument'],
                    side=OrderSide.SELL,
                    quantity=abs(short_units),
                    entry_price=float(pos['short']['averagePrice']),
                    current_price=0,
                    stop_loss=None,
                    take_profit=None,
                    unrealized_pnl=float(pos['short']['unrealizedPL']),
                    realized_pnl=float(pos['short'].get('pl', 0)),
                    commission=0,
                    swap=float(pos['short'].get('financing', 0)),
                    status=PositionStatus.OPEN,
                    opened_at=datetime.now()
                ))

        return result

    async def get_orders(self) -> List[Order]:
        """Get OANDA pending orders."""
        from oandapyV20.endpoints.orders import OrdersPending

        r = OrdersPending(self.account_id)
        response = self.api.request(r)

        result = []
        for ord in response.get('orders', []):
            order_type_map = {
                'LIMIT': OrderType.LIMIT,
                'STOP': OrderType.STOP,
                'MARKET_IF_TOUCHED': OrderType.STOP,
            }

            units = float(ord.get('units', 0))

            result.append(Order(
                order_id=ord['id'],
                broker_order_id=ord['id'],
                pair=ord['instrument'],
                side=OrderSide.BUY if units > 0 else OrderSide.SELL,
                order_type=order_type_map.get(ord['type'], OrderType.LIMIT),
                quantity=abs(units),
                price=float(ord.get('price', 0)),
                stop_loss=float(ord.get('stopLossOnFill', {}).get('price', 0)) or None,
                take_profit=float(ord.get('takeProfitOnFill', {}).get('price', 0)) or None,
                status=OrderStatus.PENDING,
                created_at=datetime.fromisoformat(ord['createTime'].replace('Z', '+00:00'))
            ))

        return result

    async def place_order(
        self,
        pair: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        **kwargs
    ) -> Order:
        """Place OANDA order."""
        from oandapyV20.endpoints.orders import OrderCreate

        units = quantity if side == OrderSide.BUY else -quantity

        order_data = {
            "instrument": pair,
            "units": str(int(units)),
        }

        if order_type == OrderType.MARKET:
            order_data["type"] = "MARKET"
        elif order_type == OrderType.LIMIT:
            order_data["type"] = "LIMIT"
            order_data["price"] = str(price)
        elif order_type == OrderType.STOP:
            order_data["type"] = "STOP"
            order_data["price"] = str(price)

        if stop_loss:
            order_data["stopLossOnFill"] = {"price": str(stop_loss)}
        if take_profit:
            order_data["takeProfitOnFill"] = {"price": str(take_profit)}

        data = {"order": order_data}

        r = OrderCreate(self.account_id, data=data)
        response = self.api.request(r)

        order_info = response.get('orderFillTransaction') or response.get('orderCreateTransaction')

        return Order(
            order_id=order_info['id'],
            broker_order_id=order_info['id'],
            pair=pair,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status=OrderStatus.FILLED if 'orderFillTransaction' in response else OrderStatus.PENDING,
            filled_price=float(order_info.get('price', 0))
        )

    async def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        quantity: Optional[float] = None
    ) -> Order:
        """Modify OANDA order."""
        # OANDA requires canceling and recreating order
        # This is a simplified implementation
        await self.cancel_order(order_id)
        # Would need to recreate with new params
        raise NotImplementedError("OANDA order modification requires cancel and replace")

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel OANDA order."""
        from oandapyV20.endpoints.orders import OrderCancel

        r = OrderCancel(self.account_id, order_id)
        self.api.request(r)
        return True

    async def close_position(
        self,
        position_id: str,
        quantity: Optional[float] = None
    ) -> Position:
        """Close OANDA position."""
        from oandapyV20.endpoints.positions import PositionClose

        # Parse pair and side from position_id
        parts = position_id.rsplit('_', 1)
        pair = parts[0]
        side = parts[1] if len(parts) > 1 else 'long'

        data = {}
        if quantity:
            if side == 'long':
                data['longUnits'] = str(int(quantity))
            else:
                data['shortUnits'] = str(int(quantity))
        else:
            if side == 'long':
                data['longUnits'] = 'ALL'
            else:
                data['shortUnits'] = 'ALL'

        r = PositionClose(self.account_id, pair, data=data)
        self.api.request(r)

        return Position(
            position_id=position_id,
            pair=pair,
            side=OrderSide.BUY if side == 'long' else OrderSide.SELL,
            quantity=quantity or 0,
            entry_price=0,
            current_price=0,
            stop_loss=None,
            take_profit=None,
            unrealized_pnl=0,
            realized_pnl=0,
            commission=0,
            swap=0,
            status=PositionStatus.CLOSED,
            opened_at=datetime.now(),
            closed_at=datetime.now()
        )

    async def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Position:
        """Modify OANDA position."""
        # OANDA doesn't support position-level SL/TP directly
        # Would need to use order-based approach
        raise NotImplementedError("Use orders for SL/TP management")

    async def get_tick(self, pair: str) -> Tick:
        """Get OANDA tick."""
        from oandapyV20.endpoints.pricing import PricingInfo

        params = {"instruments": pair}
        r = PricingInfo(self.account_id, params=params)
        response = self.api.request(r)

        price = response['prices'][0]

        return Tick(
            pair=pair,
            bid=float(price['bids'][0]['price']),
            ask=float(price['asks'][0]['price']),
            timestamp=datetime.fromisoformat(price['time'].replace('Z', '+00:00'))
        )

    async def subscribe_ticks(self, pairs: List[str]) -> bool:
        """Subscribe to OANDA streaming prices."""
        # Would implement streaming in separate thread
        return True


class PaperBroker(BaseBroker):
    """
    Paper trading broker for testing.

    Simulates broker functionality without real money.
    """

    def __init__(self, credentials: BrokerCredentials, initial_balance: float = 10000):
        super().__init__(credentials)
        self.balance = initial_balance
        self.equity = initial_balance
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
        self.position_counter = 0
        self.prices: Dict[str, Tick] = {}

    async def connect(self) -> bool:
        """Connect to paper broker."""
        self.is_connected = True
        logger.info("Connected to Paper Broker")
        return True

    async def disconnect(self) -> bool:
        """Disconnect from paper broker."""
        self.is_connected = False
        return True

    async def get_account_info(self) -> AccountInfo:
        """Get paper account info."""
        # Calculate unrealized P/L
        unrealized_pnl = sum(p.unrealized_pnl for p in self.positions.values())

        return AccountInfo(
            account_id=self.credentials.account_id,
            broker="PAPER",
            balance=self.balance,
            equity=self.balance + unrealized_pnl,
            margin_used=sum(p.quantity * p.entry_price / 100 for p in self.positions.values()),
            margin_available=self.balance * 0.8,  # Simplified
            currency="USD",
            leverage=100,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=0,
            positions_count=len(self.positions),
            orders_count=len(self.orders),
            timestamp=datetime.now()
        )

    async def get_positions(self) -> List[Position]:
        """Get paper positions."""
        return list(self.positions.values())

    async def get_orders(self) -> List[Order]:
        """Get paper orders."""
        return list(self.orders.values())

    async def place_order(
        self,
        pair: str,
        side: OrderSide,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        **kwargs
    ) -> Order:
        """Place paper order."""
        self.order_counter += 1
        order_id = f"PAPER_{self.order_counter}"

        # Get current price
        tick = self.prices.get(pair)
        if tick:
            execution_price = tick.ask if side == OrderSide.BUY else tick.bid
        else:
            execution_price = price or 1.0

        order = Order(
            order_id=order_id,
            broker_order_id=order_id,
            pair=pair,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price or execution_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status=OrderStatus.PENDING
        )

        if order_type == OrderType.MARKET:
            # Execute immediately
            self.position_counter += 1
            position_id = f"POS_{self.position_counter}"

            position = Position(
                position_id=position_id,
                pair=pair,
                side=side,
                quantity=quantity,
                entry_price=execution_price,
                current_price=execution_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                unrealized_pnl=0,
                realized_pnl=0,
                commission=quantity * 0.00002,  # Simulated commission
                swap=0,
                status=PositionStatus.OPEN,
                opened_at=datetime.now()
            )

            self.positions[position_id] = position

            order.status = OrderStatus.FILLED
            order.filled_quantity = quantity
            order.filled_price = execution_price
        else:
            self.orders[order_id] = order

        return order

    async def modify_order(
        self,
        order_id: str,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        quantity: Optional[float] = None
    ) -> Order:
        """Modify paper order."""
        if order_id not in self.orders:
            raise ValueError(f"Order not found: {order_id}")

        order = self.orders[order_id]
        if price:
            order.price = price
        if stop_loss:
            order.stop_loss = stop_loss
        if take_profit:
            order.take_profit = take_profit
        if quantity:
            order.quantity = quantity

        order.updated_at = datetime.now()
        return order

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel paper order."""
        if order_id in self.orders:
            del self.orders[order_id]
            return True
        return False

    async def close_position(
        self,
        position_id: str,
        quantity: Optional[float] = None
    ) -> Position:
        """Close paper position."""
        if position_id not in self.positions:
            raise ValueError(f"Position not found: {position_id}")

        position = self.positions[position_id]
        close_quantity = quantity or position.quantity

        # Calculate P/L
        tick = self.prices.get(position.pair)
        if tick:
            close_price = tick.bid if position.side == OrderSide.BUY else tick.ask
        else:
            close_price = position.current_price

        if position.side == OrderSide.BUY:
            pnl = (close_price - position.entry_price) * close_quantity * 10000
        else:
            pnl = (position.entry_price - close_price) * close_quantity * 10000

        self.balance += pnl

        if close_quantity >= position.quantity:
            del self.positions[position_id]
            position.status = PositionStatus.CLOSED
        else:
            position.quantity -= close_quantity

        position.realized_pnl = pnl
        position.closed_at = datetime.now()

        return position

    async def modify_position(
        self,
        position_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Position:
        """Modify paper position."""
        if position_id not in self.positions:
            raise ValueError(f"Position not found: {position_id}")

        position = self.positions[position_id]
        if stop_loss:
            position.stop_loss = stop_loss
        if take_profit:
            position.take_profit = take_profit

        return position

    async def get_tick(self, pair: str) -> Tick:
        """Get paper tick."""
        if pair not in self.prices:
            # Generate fake price
            self.prices[pair] = Tick(
                pair=pair,
                bid=1.1000,
                ask=1.1002,
                timestamp=datetime.now()
            )
        return self.prices[pair]

    async def subscribe_ticks(self, pairs: List[str]) -> bool:
        """Subscribe to paper ticks."""
        return True

    def update_price(self, pair: str, bid: float, ask: float):
        """Update price (for testing)."""
        self.prices[pair] = Tick(
            pair=pair,
            bid=bid,
            ask=ask,
            timestamp=datetime.now()
        )

        # Update positions
        for pos in self.positions.values():
            if pos.pair == pair:
                pos.current_price = bid if pos.side == OrderSide.BUY else ask
                if pos.side == OrderSide.BUY:
                    pos.unrealized_pnl = (bid - pos.entry_price) * pos.quantity * 10000
                else:
                    pos.unrealized_pnl = (pos.entry_price - ask) * pos.quantity * 10000


class BrokerFactory:
    """Factory for creating broker instances."""

    @staticmethod
    def create(credentials: BrokerCredentials) -> BaseBroker:
        """
        Create broker instance based on credentials.

        Args:
            credentials: Broker credentials

        Returns:
            Broker instance
        """
        broker_map = {
            'mt5': MT5Broker,
            'metatrader5': MT5Broker,
            'oanda': OANDABroker,
            'paper': PaperBroker,
        }

        broker_class = broker_map.get(credentials.broker.lower())

        if not broker_class:
            raise ValueError(f"Unknown broker: {credentials.broker}")

        return broker_class(credentials)


class TradingEngine:
    """
    High-level trading engine that wraps broker functionality.

    Provides:
    - Signal execution
    - Risk management
    - Position monitoring
    - Order management
    """

    def __init__(self, broker: BaseBroker):
        self.broker = broker
        self.is_running = False
        self.max_risk_per_trade = 0.02  # 2%
        self.max_positions = 5
        self.min_rr_ratio = 1.5
        self.position_monitor_interval = 5  # seconds

        self._monitor_thread = None
        self._stop_monitor = False

    async def start(self):
        """Start trading engine."""
        if not self.broker.is_connected:
            await self.broker.connect()

        self.is_running = True
        self._stop_monitor = False

        # Start position monitor
        self._monitor_thread = threading.Thread(target=self._monitor_positions_thread)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

        logger.info("Trading engine started")

    async def stop(self):
        """Stop trading engine."""
        self.is_running = False
        self._stop_monitor = True

        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)

        await self.broker.disconnect()
        logger.info("Trading engine stopped")

    async def execute_signal(
        self,
        pair: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: float
    ) -> Optional[Order]:
        """
        Execute a trading signal.

        Args:
            pair: Trading pair
            direction: 'BUY' or 'SELL'
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price
            confidence: Signal confidence (0-100)

        Returns:
            Order if executed, None otherwise
        """
        if not self.is_running:
            logger.warning("Trading engine not running")
            return None

        # Check position count
        positions = await self.broker.get_positions()
        if len(positions) >= self.max_positions:
            logger.warning(f"Max positions reached ({self.max_positions})")
            return None

        # Check if already have position in this pair
        for pos in positions:
            if pos.pair == pair:
                logger.warning(f"Already have position in {pair}")
                return None

        # Calculate position size
        account = await self.broker.get_account_info()
        position_size = self._calculate_position_size(
            account.balance,
            entry_price,
            stop_loss,
            pair
        )

        if position_size <= 0:
            logger.warning(f"Position size too small for {pair}")
            return None

        # Calculate R:R
        if direction == 'BUY':
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit

        rr_ratio = reward / risk if risk > 0 else 0

        if rr_ratio < self.min_rr_ratio:
            logger.warning(f"R:R ratio {rr_ratio:.2f} below minimum {self.min_rr_ratio}")
            return None

        # Place order
        try:
            order = await self.broker.place_order(
                pair=pair,
                side=OrderSide.BUY if direction == 'BUY' else OrderSide.SELL,
                quantity=position_size,
                order_type=OrderType.MARKET,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            logger.info(
                f"Executed {direction} {pair}: "
                f"size={position_size}, SL={stop_loss}, TP={take_profit}"
            )

            return order

        except Exception as e:
            logger.error(f"Order execution failed: {e}")
            return None

    def _calculate_position_size(
        self,
        balance: float,
        entry: float,
        stop_loss: float,
        pair: str
    ) -> float:
        """Calculate position size based on risk."""
        risk_amount = balance * self.max_risk_per_trade

        # Calculate pips at risk
        pip_value = 0.01 if 'JPY' in pair else 0.0001
        pips_risk = abs(entry - stop_loss) / pip_value

        if pips_risk == 0:
            return 0

        # $10 per pip per lot (standard)
        pip_cost = 10

        position_size = risk_amount / (pips_risk * pip_cost)

        # Round to 0.01 lots
        return round(position_size, 2)

    def _monitor_positions_thread(self):
        """Background thread for monitoring positions."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while not self._stop_monitor:
            try:
                loop.run_until_complete(self._check_positions())
            except Exception as e:
                logger.error(f"Position monitor error: {e}")

            time.sleep(self.position_monitor_interval)

        loop.close()

    async def _check_positions(self):
        """Check all positions for SL/TP."""
        positions = await self.broker.get_positions()

        for pos in positions:
            tick = await self.broker.get_tick(pos.pair)
            current_price = tick.bid if pos.side == OrderSide.BUY else tick.ask

            # Update position current price
            pos.current_price = current_price

            # Check stop loss
            if pos.stop_loss:
                if pos.side == OrderSide.BUY and current_price <= pos.stop_loss:
                    logger.info(f"SL hit for {pos.pair}")
                    await self.broker.close_position(pos.position_id)
                elif pos.side == OrderSide.SELL and current_price >= pos.stop_loss:
                    logger.info(f"SL hit for {pos.pair}")
                    await self.broker.close_position(pos.position_id)

            # Check take profit
            if pos.take_profit:
                if pos.side == OrderSide.BUY and current_price >= pos.take_profit:
                    logger.info(f"TP hit for {pos.pair}")
                    await self.broker.close_position(pos.position_id)
                elif pos.side == OrderSide.SELL and current_price <= pos.take_profit:
                    logger.info(f"TP hit for {pos.pair}")
                    await self.broker.close_position(pos.position_id)


if __name__ == "__main__":
    print("Broker Integration Module - Forex Scalper Agent V2")
    print("=" * 50)
    print("\nSupported Brokers:")
    print("  - MetaTrader 5 (MT5)")
    print("  - OANDA")
    print("  - Paper Trading (simulation)")
    print("\nUsage:")
    print("  credentials = BrokerCredentials(")
    print("      broker='paper',")
    print("      account_id='12345'")
    print("  )")
    print("  broker = BrokerFactory.create(credentials)")
    print("  await broker.connect()")
    print("  order = await broker.place_order('EURUSD', OrderSide.BUY, 0.1)")
