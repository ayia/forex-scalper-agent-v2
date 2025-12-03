"""
Improved Scalping Strategy - Version 2.3 (Backtest-Validated)
==============================================================================
Focus sur les paires rentables: USDJPY (58% WR), USDCHF (45% WR)
"""
import pandas as pd
from typing import Dict, Optional
from datetime import datetime

from strategies.base import BaseStrategy, Signal
from config import get_pip_value


class ImprovedScalpingStrategy(BaseStrategy):
    """
    Strategie de scalping v2.3 - Paires validees uniquement.

    Changements v2.3:
    - AUDUSD retire (10% WR)
    - NZDUSD retire (25% WR)
    - Focus sur USDJPY (58% WR) et USDCHF (45% WR)
    """

    # Paires validees par backtest uniquement
    ALLOWED_PAIRS = ['USDJPY', 'USDCHF']

    def __init__(self):
        super().__init__("ImprovedScalping")
        self.ema_fast = 8
        self.ema_slow = 21
        self.consecutive_losses = 0
        self.max_consecutive_losses = 4  # Augmente de 3 a 4

    def update_trade_result(self, won: bool):
        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

    def can_trade(self) -> bool:
        return self.consecutive_losses < self.max_consecutive_losses

    def get_htf_bias(self, df: pd.DataFrame) -> str:
        """Determine le biais HTF."""
        if len(df) < 30:
            return 'NEUTRAL'

        ema_fast = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.ema_slow, adjust=False).mean()

        if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
            # Verifier la tendance
            if ema_fast.iloc[-5] < ema_slow.iloc[-5]:  # Recent crossover
                return 'BULLISH'
            elif ema_fast.iloc[-1] > ema_fast.iloc[-5]:  # Trending up
                return 'BULLISH'

        if ema_fast.iloc[-1] < ema_slow.iloc[-1]:
            if ema_fast.iloc[-5] > ema_slow.iloc[-5]:  # Recent crossunder
                return 'BEARISH'
            elif ema_fast.iloc[-1] < ema_fast.iloc[-5]:  # Trending down
                return 'BEARISH'

        return 'NEUTRAL'

    def calculate_stochastic(self, df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
        """Calcule le Stochastic oscillator."""
        low_min = df['low'].rolling(k_period).min()
        high_max = df['high'].rolling(k_period).max()

        k = 100 * (df['close'] - low_min) / (high_max - low_min + 0.0001)
        d = k.rolling(d_period).mean()

        return k, d

    def analyze(self, data: Dict[str, pd.DataFrame], pair: str) -> Optional[Signal]:
        """Analyse v2.0 - filtres equilibres."""

        # Filtre paires
        if pair not in self.ALLOWED_PAIRS:
            return None

        # Filtre pertes
        if not self.can_trade():
            return None

        if "M15" not in data:
            return None

        df_m15 = data["M15"]
        df_h1 = data.get("H1", df_m15)  # Fallback sur M15 si H1 absent

        if len(df_m15) < 50:
            return None

        # Biais H1 - v2.0: accepte trades meme si NEUTRAL avec confirmation forte
        h1_bias = self.get_htf_bias(df_h1)

        # Indicateurs M15
        ema_fast = df_m15['close'].ewm(span=self.ema_fast, adjust=False).mean()
        ema_slow = df_m15['close'].ewm(span=self.ema_slow, adjust=False).mean()
        rsi = self.calculate_rsi(df_m15, 14)
        stoch_k, stoch_d = self.calculate_stochastic(df_m15)
        atr = self.calculate_atr(df_m15, 14)

        current_price = df_m15['close'].iloc[-1]
        pip_value = get_pip_value(pair)

        # Detecter le biais M15
        m15_bullish = ema_fast.iloc[-1] > ema_slow.iloc[-1]
        m15_bearish = ema_fast.iloc[-1] < ema_slow.iloc[-1]

        # Signal BUY - CONDITIONS STRICTES (BUY historiquement mauvais)
        buy_signal = False
        if h1_bias == 'BULLISH' and m15_bullish:  # H1 BULLISH obligatoire (pas NEUTRAL)
            # Crossover UNIQUEMENT
            ema_cross = ema_fast.iloc[-2] <= ema_slow.iloc[-2]

            buy_signal = (
                ema_cross and
                rsi.iloc[-1] > 45 and rsi.iloc[-1] < 65 and  # Zone stricte
                stoch_k.iloc[-1] > stoch_d.iloc[-1] and
                stoch_k.iloc[-2] <= stoch_d.iloc[-2] and  # Crossover stoch aussi
                stoch_k.iloc[-1] > 25 and stoch_k.iloc[-1] < 75  # Zone neutre
            )

        # Signal SELL - CONDITIONS STANDARD (SELL performant)
        sell_signal = False
        if h1_bias in ['BEARISH', 'NEUTRAL'] and m15_bearish:
            # Crossunder OU continuation de tendance
            ema_cross = ema_fast.iloc[-2] >= ema_slow.iloc[-2]
            ema_trending = ema_fast.iloc[-1] < ema_fast.iloc[-3]

            sell_signal = (
                (ema_cross or ema_trending) and
                rsi.iloc[-1] < 60 and rsi.iloc[-1] > 30 and
                stoch_k.iloc[-1] < stoch_d.iloc[-1] and
                stoch_k.iloc[-1] > 15
            )
            # Si H1 NEUTRAL, confirmation supplementaire
            if h1_bias == 'NEUTRAL':
                sell_signal = sell_signal and rsi.iloc[-1] < 50

        if not buy_signal and not sell_signal:
            return None

        direction = "BUY" if buy_signal else "SELL"

        # SL/TP - ajuste selon qualite du setup
        atr_pips = atr.iloc[-1] / pip_value

        if h1_bias != 'NEUTRAL':
            # Setup fort: R:R plus agressif
            sl_pips = min(atr_pips * 1.0, 10)
            tp_pips = sl_pips * 2.0
            confidence = 75.0
        else:
            # Setup moyen: R:R conservateur
            sl_pips = min(atr_pips * 1.2, 12)
            tp_pips = sl_pips * 1.5
            confidence = 60.0

        if direction == "BUY":
            stop_loss = current_price - (sl_pips * pip_value)
            take_profit = current_price + (tp_pips * pip_value)
        else:
            stop_loss = current_price + (sl_pips * pip_value)
            take_profit = current_price - (tp_pips * pip_value)

        return Signal(
            pair=pair,
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            strategy=self.name,
            timeframe="M15",
            timestamp=datetime.now().isoformat()
        )
