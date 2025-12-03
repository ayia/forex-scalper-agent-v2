"""
Improved Trend Strategy - Version 2.3 (Backtest-Validated)
==============================================================================
Focus sur les paires rentables: USDJPY (58% WR), USDCHF (45% WR), EURUSD (40% WR)
"""
import pandas as pd
from typing import Dict, Optional
from datetime import datetime
from dataclasses import dataclass

from strategies.base import BaseStrategy, Signal
from config import STRATEGY_PARAMS, get_pip_value


@dataclass
class MarketContext:
    """Contexte de marche pour filtrage."""
    htf_bias: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    h1_bias: str
    regime: str  # 'TRENDING', 'RANGING', 'CHOPPY'
    volatility: str  # 'LOW', 'NORMAL', 'HIGH'
    strength: float  # 0-100


class ImprovedTrendStrategy(BaseStrategy):
    """
    Strategie de trend following v2.3 - Focus paires rentables.

    Changements v2.3:
    - AUDUSD et NZDUSD retires (10-25% WR)
    - Focus sur USDJPY (58% WR) et USDCHF (45% WR)
    - EURUSD garde pour trend (pas scalping)
    """

    # Paires validees par backtest (WR > 40%)
    ALLOWED_PAIRS = ['USDJPY', 'USDCHF', 'EURUSD']
    EXCLUDED_PAIRS = ['GBPUSD', 'AUDUSD', 'NZDUSD', 'USDCAD']

    def __init__(self):
        super().__init__("ImprovedTrend")
        self.ema_fast = STRATEGY_PARAMS.get("ema_fast", 8)
        self.ema_medium = STRATEGY_PARAMS.get("ema_medium", 21)
        self.ema_slow = STRATEGY_PARAMS.get("ema_slow", 50)

        # Tracking des pertes
        self.consecutive_losses = 0
        self.max_consecutive_losses = 4
        self.session_losses = 0
        self.max_session_losses = 3
        self.last_trade_result = None

    def update_trade_result(self, won: bool):
        """Met a jour le suivi des trades."""
        if won:
            self.consecutive_losses = 0
            self.last_trade_result = 'WIN'
        else:
            self.consecutive_losses += 1
            self.session_losses += 1
            self.last_trade_result = 'LOSS'

    def reset_session(self):
        """Reset le compteur de session."""
        self.session_losses = 0

    def can_trade(self) -> bool:
        """Verifie si on peut trader (gestion des pertes)."""
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False
        if self.session_losses >= self.max_session_losses:
            return False
        return True

    def get_htf_bias(self, df: pd.DataFrame) -> str:
        """
        Determine le biais directionnel sur un timeframe.
        Utilise EMA stack + price position.
        """
        if len(df) < 50:
            return 'NEUTRAL'

        # Calculer EMAs
        ema_fast = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        ema_medium = df['close'].ewm(span=self.ema_medium, adjust=False).mean()
        ema_slow = df['close'].ewm(span=self.ema_slow, adjust=False).mean()

        current_price = df['close'].iloc[-1]

        # Bullish: Prix > EMAs et EMAs alignees
        if (current_price > ema_fast.iloc[-1] > ema_medium.iloc[-1] > ema_slow.iloc[-1]):
            return 'BULLISH'

        # Bearish: Prix < EMAs et EMAs alignees
        if (current_price < ema_fast.iloc[-1] < ema_medium.iloc[-1] < ema_slow.iloc[-1]):
            return 'BEARISH'

        return 'NEUTRAL'

    def get_market_regime(self, df: pd.DataFrame) -> str:
        """
        Determine le regime de marche - v2.0 seuils ajustes.
        TRENDING: ADX > 20 (assoupli de 25)
        RANGING: ADX < 15 (assoupli de 20)
        TRANSITION: entre les deux (nouveau - permet trading avec confirmation)
        """
        if len(df) < 30:
            return 'UNKNOWN'

        # Calculer ADX simplifie
        high = df['high']
        low = df['low']
        close = df['close']

        # True Range
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)

        atr = tr.rolling(14).mean()

        # +DM et -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

        # +DI et -DI
        plus_di = 100 * (plus_dm.rolling(14).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(14).mean() / atr)

        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 0.0001)
        adx = dx.rolling(14).mean().iloc[-1]

        if pd.isna(adx):
            return 'UNKNOWN'

        # Seuils v2.0 - plus permissifs
        if adx > 20:
            return 'TRENDING'
        elif adx < 15:
            return 'RANGING'
        else:
            return 'TRANSITION'  # Nouveau: permet trading avec confirmation forte

    def get_volatility_level(self, df: pd.DataFrame) -> str:
        """Determine le niveau de volatilite."""
        if len(df) < 20:
            return 'NORMAL'

        # ATR actuel vs ATR moyenne
        atr = self.calculate_atr(df, 14)
        current_atr = atr.iloc[-1]
        avg_atr = atr.rolling(50).mean().iloc[-1]

        if pd.isna(avg_atr) or avg_atr == 0:
            return 'NORMAL'

        ratio = current_atr / avg_atr

        if ratio > 1.5:
            return 'HIGH'
        elif ratio < 0.7:
            return 'LOW'
        return 'NORMAL'

    def analyze_market_context(self, data: Dict[str, pd.DataFrame]) -> Optional[MarketContext]:
        """Analyse complete du contexte de marche."""

        # Verifier qu'on a les timeframes necessaires
        if 'H4' not in data and 'H1' not in data:
            # Fallback: utiliser H1 comme HTF
            if 'H1' not in data:
                return None
            htf_df = data['H1']
            h1_df = data.get('M15', data['H1'])
        else:
            htf_df = data.get('H4', data['H1'])
            h1_df = data.get('H1', htf_df)

        # Determiner les biais
        htf_bias = self.get_htf_bias(htf_df)
        h1_bias = self.get_htf_bias(h1_df)

        # Regime sur H1
        regime = self.get_market_regime(h1_df)

        # Volatilite
        volatility = self.get_volatility_level(h1_df)

        # Force du signal (0-100)
        strength = 50.0
        if htf_bias == h1_bias and htf_bias != 'NEUTRAL':
            strength += 25
        if regime == 'TRENDING':
            strength += 15
        if volatility == 'NORMAL':
            strength += 10

        return MarketContext(
            htf_bias=htf_bias,
            h1_bias=h1_bias,
            regime=regime,
            volatility=volatility,
            strength=strength
        )

    def calculate_momentum(self, df: pd.DataFrame, period: int = 10) -> pd.Series:
        """Calcule le momentum (ROC)."""
        return (df['close'] / df['close'].shift(period) - 1) * 100

    def analyze(self, data: Dict[str, pd.DataFrame], pair: str) -> Optional[Signal]:
        """
        Analyse v2.0 - filtres equilibres pour plus de trades.
        """
        # Filtre 1: Paires exclues (seulement GBPUSD)
        if pair in self.EXCLUDED_PAIRS:
            return None

        # Filtre 2: Gestion des pertes
        if not self.can_trade():
            return None

        # Verifier les donnees
        if "M15" not in data:
            return None

        df_m15 = data["M15"]
        if len(df_m15) < 50:
            return None

        # Filtre 3: Analyser le contexte de marche
        context = self.analyze_market_context(data)
        if context is None:
            return None

        # Filtre 4: HTF - v2.0 plus permissif
        # Option A: HTF et H1 alignes (meilleur)
        # Option B: H1 non-NEUTRAL avec momentum fort (nouveau)
        htf_aligned = (context.htf_bias != 'NEUTRAL' and
                       context.h1_bias != 'NEUTRAL' and
                       context.htf_bias == context.h1_bias)

        h1_strong = (context.h1_bias != 'NEUTRAL' and context.strength >= 60)

        if not htf_aligned and not h1_strong:
            return None

        # Filtre 5: Regime - v2.0 permet TRANSITION avec confirmation
        if context.regime == 'RANGING':
            return None  # Seul RANGING est interdit

        # Si TRANSITION, exiger momentum fort
        needs_strong_momentum = (context.regime == 'TRANSITION')

        # Filtre 6: Volatilite (inchange)
        if context.volatility == 'HIGH':
            return None

        # Calculer les indicateurs sur M15
        ema_fast = self.calculate_ema(df_m15, self.ema_fast)
        ema_medium = self.calculate_ema(df_m15, self.ema_medium)
        ema_slow = self.calculate_ema(df_m15, self.ema_slow)
        macd, macd_signal = self.calculate_macd(df_m15)
        rsi = self.calculate_rsi(df_m15, 14)
        atr = self.calculate_atr(df_m15, 14)
        momentum = self.calculate_momentum(df_m15, 10)

        current_price = df_m15['close'].iloc[-1]
        pip_value = get_pip_value(pair)
        current_momentum = momentum.iloc[-1]

        # Filtre momentum si necessaire
        if needs_strong_momentum and abs(current_momentum) < 0.15:
            return None

        # Determiner le biais de trading
        trade_bias = context.h1_bias if not htf_aligned else context.htf_bias

        # Signal BUY - CONDITIONS STRICTES (BUY WR historique: 23%)
        # Exiger alignement HTF complet + confirmation forte
        bullish_signal = False
        if trade_bias == 'BULLISH' and htf_aligned:  # HTF aligned obligatoire pour BUY
            # EMA stack complet
            ema_bullish = (ema_fast.iloc[-1] > ema_medium.iloc[-1] > ema_slow.iloc[-1])
            # MACD crossover UNIQUEMENT (pas de continuation)
            macd_bullish = (macd.iloc[-1] > macd_signal.iloc[-1] and
                           macd.iloc[-2] <= macd_signal.iloc[-2])
            # RSI: zone stricte 45-65 (eviter surachat)
            rsi_ok = rsi.iloc[-1] > 45 and rsi.iloc[-1] < 65
            # Momentum positif fort
            momentum_ok = current_momentum > 0.1

            bullish_signal = ema_bullish and macd_bullish and rsi_ok and momentum_ok

        # Signal SELL - CONDITIONS STANDARD (SELL WR historique: 44%)
        bearish_signal = False
        if trade_bias == 'BEARISH':
            # EMA stack bearish (2 EMAs suffisent)
            ema_bearish = ema_fast.iloc[-1] < ema_medium.iloc[-1]
            # MACD: crossunder OU continuation negative
            macd_bearish = (
                (macd.iloc[-1] < macd_signal.iloc[-1] and macd.iloc[-2] >= macd_signal.iloc[-2]) or
                (macd.iloc[-1] < 0 and macd.iloc[-1] < macd.iloc[-2])
            )
            # RSI: zone elargie 25-60
            rsi_ok = rsi.iloc[-1] < 60 and rsi.iloc[-1] > 25
            # Momentum negatif
            momentum_ok = current_momentum < 0

            bearish_signal = ema_bearish and macd_bearish and rsi_ok and momentum_ok

        if not bullish_signal and not bearish_signal:
            return None

        direction = "BUY" if bullish_signal else "SELL"

        # Calcul SL/TP - ajuste selon la qualite du setup
        atr_value = atr.iloc[-1]
        atr_pips = atr_value / pip_value

        # Setup fort (HTF aligne) = SL plus serre, meilleur R:R
        if htf_aligned:
            sl_pips = min(atr_pips * 1.0, 10)
            tp_pips = sl_pips * 2.5  # R:R 2.5:1
        else:
            sl_pips = min(atr_pips * 1.3, 12)
            tp_pips = sl_pips * 1.8  # R:R 1.8:1

        if direction == "BUY":
            stop_loss = current_price - (sl_pips * pip_value)
            take_profit = current_price + (tp_pips * pip_value)
        else:
            stop_loss = current_price + (sl_pips * pip_value)
            take_profit = current_price - (tp_pips * pip_value)

        # Confidence ajustee
        confidence = context.strength
        if htf_aligned:
            confidence += 10
        if context.regime == 'TRENDING':
            confidence += 5

        return Signal(
            pair=pair,
            direction=direction,
            entry_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=min(confidence, 95),
            strategy=self.name,
            timeframe="M15",
            timestamp=datetime.now().isoformat()
        )
