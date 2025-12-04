# Prompt IA - Détection de Stratégie Forex Optimale

## Objectif
Trouver la meilleure stratégie de trading pour une paire Forex donnée, avec paramètres optimisés et règles de gestion du risque validées par backtest.

---

## Étapes à Suivre

### ÉTAPE 1: COLLECTE DES DONNÉES
```
- Télécharger 2-10 ans de données historiques (OHLCV)
- Timeframe: H1 (horaire) ou D1 (daily)
- Source: yfinance, MT5, ou autre API fiable
- Format: Date, Open, High, Low, Close, Volume
```

### ÉTAPE 2: BACKTEST INITIAL - STRATÉGIES CANDIDATES
```
Tester TOUTES les stratégies candidates suivantes:

═══════════════════════════════════════════════════════════════
CATÉGORIE 1: STRATÉGIES DE TENDANCE (Trend Following)
═══════════════════════════════════════════════════════════════

  1) EMA Crossover Triple
     - Indicateurs: EMA(8), EMA(21), EMA(50)
     - BUY: EMA8 > EMA21 > EMA50
     - SELL: EMA8 < EMA21 < EMA50
     - Paramètres à tester: périodes [5,8,13], [8,21,50], [10,20,50]

  2) EMA Crossover Double
     - Indicateurs: EMA(9), EMA(21)
     - BUY: EMA9 croise EMA21 vers le haut
     - SELL: EMA9 croise EMA21 vers le bas
     - Paramètres à tester: [5,13], [9,21], [12,26]

  3) SMA Crossover (Golden/Death Cross)
     - Indicateurs: SMA(50), SMA(200)
     - BUY: SMA50 croise SMA200 vers le haut (Golden Cross)
     - SELL: SMA50 croise SMA200 vers le bas (Death Cross)

  4) MACD Crossover
     - Indicateurs: MACD(12,26,9)
     - BUY: MACD line croise Signal line vers le haut
     - SELL: MACD line croise Signal line vers le bas
     - Variante: MACD histogram change de signe

  5) MACD + Zero Line
     - BUY: MACD > 0 ET crossover haussier
     - SELL: MACD < 0 ET crossover baissier

  6) ADX Trend Strength
     - Indicateurs: ADX(14), +DI, -DI
     - BUY: ADX > 25 ET +DI > -DI
     - SELL: ADX > 25 ET -DI > +DI
     - Paramètres à tester: ADX threshold [20, 25, 30]

  7) Parabolic SAR
     - BUY: Prix passe au-dessus du SAR
     - SELL: Prix passe en-dessous du SAR
     - Paramètres: AF start=0.02, AF max=0.2

  8) Ichimoku Cloud
     - BUY: Prix > Cloud ET Tenkan > Kijun
     - SELL: Prix < Cloud ET Tenkan < Kijun
     - Confirmation: Chikou span au-dessus/dessous du prix

  9) Supertrend
     - Indicateurs: Supertrend(10, 3)
     - BUY: Prix croise Supertrend vers le haut
     - SELL: Prix croise Supertrend vers le bas
     - Paramètres à tester: period [7,10,14], multiplier [2,3,4]

  10) Donchian Channel Breakout
      - Indicateurs: Donchian(20)
      - BUY: Prix casse le high des 20 dernières périodes
      - SELL: Prix casse le low des 20 dernières périodes

═══════════════════════════════════════════════════════════════
CATÉGORIE 2: STRATÉGIES DE MOMENTUM
═══════════════════════════════════════════════════════════════

  11) RSI Overbought/Oversold
      - Indicateurs: RSI(14)
      - BUY: RSI < 30 (oversold)
      - SELL: RSI > 70 (overbought)
      - Paramètres à tester: [20,80], [25,75], [30,70]

  12) RSI Centerline Crossover
      - BUY: RSI croise 50 vers le haut
      - SELL: RSI croise 50 vers le bas

  13) RSI Divergence
      - BUY: Prix fait lower low, RSI fait higher low (bullish divergence)
      - SELL: Prix fait higher high, RSI fait lower high (bearish divergence)

  14) Stochastic Crossover
      - Indicateurs: Stochastic(14,3,3)
      - BUY: %K < 20 ET %K croise %D vers le haut
      - SELL: %K > 80 ET %K croise %D vers le bas
      - Paramètres à tester: zones [20,80], [25,75], [30,70]

  15) Stochastic Double Cross
      - BUY: %K et %D tous deux < 20, puis %K > %D
      - SELL: %K et %D tous deux > 80, puis %K < %D

  16) CCI (Commodity Channel Index)
      - Indicateurs: CCI(20)
      - BUY: CCI < -100 puis repasse au-dessus
      - SELL: CCI > +100 puis repasse en-dessous

  17) Williams %R
      - Indicateurs: Williams %R(14)
      - BUY: %R < -80 (oversold)
      - SELL: %R > -20 (overbought)

  18) MFI (Money Flow Index)
      - Indicateurs: MFI(14)
      - BUY: MFI < 20 (oversold avec volume)
      - SELL: MFI > 80 (overbought avec volume)

═══════════════════════════════════════════════════════════════
CATÉGORIE 3: STRATÉGIES DE VOLATILITÉ
═══════════════════════════════════════════════════════════════

  19) Bollinger Bands Bounce
      - Indicateurs: BB(20,2)
      - BUY: Prix touche bande inférieure
      - SELL: Prix touche bande supérieure

  20) Bollinger Bands Squeeze Breakout
      - Détection: BB width < seuil (squeeze)
      - BUY: Breakout vers le haut après squeeze
      - SELL: Breakout vers le bas après squeeze

  21) Bollinger Bands %B
      - BUY: %B < 0 (sous la bande inférieure)
      - SELL: %B > 1 (au-dessus de la bande supérieure)

  22) Keltner Channel
      - Indicateurs: Keltner(20, 2xATR)
      - BUY: Prix casse le channel supérieur
      - SELL: Prix casse le channel inférieur

  23) ATR Breakout
      - BUY: Prix > High précédent + 1.5xATR
      - SELL: Prix < Low précédent - 1.5xATR

  24) Volatility Contraction Pattern (VCP)
      - Détection: Réduction progressive de la volatilité
      - Entrée: Breakout après contraction

═══════════════════════════════════════════════════════════════
CATÉGORIE 4: STRATÉGIES DE SUPPORT/RÉSISTANCE
═══════════════════════════════════════════════════════════════

  25) Pivot Points
      - Indicateurs: Pivot, R1, R2, R3, S1, S2, S3
      - BUY: Rebond sur support (S1, S2)
      - SELL: Rejet sur résistance (R1, R2)

  26) Fibonacci Retracement
      - BUY: Rebond sur 38.2%, 50%, ou 61.8% en uptrend
      - SELL: Rejet sur 38.2%, 50%, ou 61.8% en downtrend

  27) Support/Resistance Breakout
      - BUY: Cassure de résistance avec volume
      - SELL: Cassure de support avec volume

  28) Price Action - Pin Bar
      - BUY: Pin bar haussier sur support
      - SELL: Pin bar baissier sur résistance

  29) Price Action - Engulfing
      - BUY: Bullish engulfing sur support
      - SELL: Bearish engulfing sur résistance

═══════════════════════════════════════════════════════════════
CATÉGORIE 5: STRATÉGIES COMBINÉES (Multi-Indicateurs)
═══════════════════════════════════════════════════════════════

  30) EMA + RSI + ADX
      - BUY: EMA8 > EMA21 ET RSI > 50 ET ADX > 25
      - SELL: EMA8 < EMA21 ET RSI < 50 ET ADX > 25

  31) MACD + Stochastic
      - BUY: MACD crossover haussier ET Stoch < 30
      - SELL: MACD crossover baissier ET Stoch > 70

  32) Bollinger + RSI
      - BUY: Prix sur BB lower ET RSI < 30
      - SELL: Prix sur BB upper ET RSI > 70

  33) Ichimoku + MACD
      - BUY: Prix > Cloud ET MACD > 0
      - SELL: Prix < Cloud ET MACD < 0

  34) Triple Screen (Elder)
      - Screen 1 (Weekly): Identifier trend avec MACD
      - Screen 2 (Daily): Attendre pullback avec Stochastic
      - Screen 3 (H4): Entrée précise

  35) Moving Average Ribbon
      - Indicateurs: EMA 8,13,21,34,55,89
      - BUY: Toutes EMAs alignées en ordre ascendant
      - SELL: Toutes EMAs alignées en ordre descendant

═══════════════════════════════════════════════════════════════
CATÉGORIE 6: STRATÉGIES AVANCÉES
═══════════════════════════════════════════════════════════════

  36) Mean Reversion
      - Indicateurs: Z-Score du prix sur 20 périodes
      - BUY: Z-Score < -2
      - SELL: Z-Score > +2

  37) Momentum Breakout
      - Indicateurs: ROC(10), Volume
      - BUY: ROC > 0 ET Volume > moyenne
      - SELL: ROC < 0 ET Volume > moyenne

  38) VWAP Strategy (intraday)
      - BUY: Prix croise VWAP vers le haut
      - SELL: Prix croise VWAP vers le bas

  39) Order Flow / Volume Profile
      - BUY: Prix au Point of Control avec volume acheteur
      - SELL: Prix au Point of Control avec volume vendeur

  40) Range Breakout (Asian Session)
      - Définir range de la session asiatique
      - BUY: Breakout haut du range en session London/NY
      - SELL: Breakout bas du range en session London/NY

Pour chaque stratégie, calculer:
  - Win Rate (%)
  - Profit Factor
  - Max Drawdown (%)
  - Nombre total de trades
  - Sharpe Ratio (si possible)
```

### ÉTAPE 3: SÉLECTION DE LA STRATÉGIE
```
Critères de sélection (TOUS requis):
  - Profit Factor >= 1.0
  - Nombre de trades >= 100 (significatif statistiquement)
  - Max Drawdown < 30%

Action: Garder les 1-3 meilleures stratégies pour optimisation
```

### ÉTAPE 4: OPTIMISATION DES PARAMÈTRES (Grid Search)
```
Pour chaque stratégie retenue, tester toutes combinaisons:

  R:R (Risk/Reward):
    - [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0]

  Filtre ADX minimum:
    - [10, 12, 15, 20, 25, 30]

  Filtre RSI (low, high):
    - [(20,80), (25,75), (30,70), (35,65), (40,60)]

  Score minimum confluence:
    - [3, 4, 5, 6, 7]

  Périodes indicateurs:
    - Varier selon la stratégie

Enregistrer TOUTES les combinaisons avec leurs résultats dans un CSV
Sélectionner la combinaison avec le meilleur Profit Factor
```

### ÉTAPE 5: DÉTECTION DES RÉGIMES DE MARCHÉ
```
Classifier les conditions de marché:

  HIGH_VOLATILITY:
    - Condition: ATR > 1.5x moyenne 20 périodes
    - OU: BB Width > 2x moyenne

  LOW_VOLATILITY:
    - Condition: ATR < 0.7x moyenne 20 périodes
    - OU: BB Width < 0.5x moyenne

  TRENDING_UP:
    - Condition: ADX > 25 ET +DI > -DI
    - OU: Prix > EMA50 > EMA200

  TRENDING_DOWN:
    - Condition: ADX > 25 ET -DI > +DI
    - OU: Prix < EMA50 < EMA200

  RANGING:
    - Condition: ADX < 20
    - OU: Prix oscille entre support/résistance

  STRONG_TREND:
    - Condition: ADX > 40

  BREAKOUT:
    - Condition: Prix casse un niveau clé avec volume

  CONSOLIDATION:
    - Condition: BB squeeze (width minimal)

Action: Backtester la stratégie SÉPARÉMENT pour chaque régime
Résultat: Identifier quand trader vs quand éviter
```

### ÉTAPE 6: VALIDATION CONTRAINTE RISQUE
```
Définir contrainte absolue:
  - Max perte journalière: -$500 (ajustable selon capital)

Tester différents lot sizes:
  - [0.30, 0.25, 0.20, 0.15, 0.10, 0.05]

Pour chaque lot size:
  - Simuler sur 10 ans de données
  - Compter jours où perte > contrainte
  - Sélectionner le PLUS GRAND lot qui ne dépasse JAMAIS la contrainte
```

### ÉTAPE 7: VALIDATION FINALE
```
Backtest du dernier mois en simulation horaire:
  - Simuler exécution toutes les heures
  - Capital initial: $10,000 (ou capital réel)
  - Lot size: valeur validée à l'étape 6

Vérifier:
  - ROI positif
  - Profit Factor > 1.0
  - Contrainte journalière respectée
  - Win Rate cohérent avec backtest long terme
```

### ÉTAPE 8: OUTPUT FINAL
```
Documenter la stratégie validée:

  PAIRE: [NOM_PAIRE]
  STRATÉGIE: [NOM_STRATÉGIE]

  PARAMÈTRES OPTIMAUX:
    - R:R: [valeur]
    - ADX min: [valeur]
    - RSI range: [low, high]
    - Score min: [valeur]
    - Périodes indicateurs: [valeurs]

  RÉGIMES À TRADER:
    - [liste des régimes favorables]

  RÉGIMES À ÉVITER:
    - [liste des régimes défavorables]

  GESTION RISQUE:
    - Lot size validé: [valeur]
    - Max perte journalière: [valeur]

  PERFORMANCE ATTENDUE:
    - Profit Factor: [valeur]
    - Win Rate: [valeur]%
    - Max Drawdown: [valeur]%
    - Trades/an estimés: [valeur]
```

---

## Checklist Condensée

```
□ 1. Télécharger données historiques (2-10 ans, H1/D1)
□ 2. Tester les 40 stratégies candidates
□ 3. Garder stratégies avec PF >= 1.0 et trades >= 100
□ 4. Grid search paramètres (R:R, ADX, RSI, score, périodes)
□ 5. Classifier régimes marché (volatilité, trend, range)
□ 6. Backtester par régime → identifier quand trader/éviter
□ 7. Tester lot sizes avec contrainte perte max journalière
□ 8. Validation finale: backtest dernier mois hourly
□ 9. Documenter stratégie optimale avec tous paramètres
```

---

## Tableau Récapitulatif des Stratégies

| # | Stratégie | Catégorie | Meilleur pour |
|---|-----------|-----------|---------------|
| 1 | EMA Triple | Tendance | Trends forts |
| 2 | EMA Double | Tendance | Trends modérés |
| 3 | SMA Golden Cross | Tendance | Long terme |
| 4 | MACD Crossover | Tendance | Confirmation trend |
| 5 | MACD Zero Line | Tendance | Trend + momentum |
| 6 | ADX Trend | Tendance | Force du trend |
| 7 | Parabolic SAR | Tendance | Trailing stop |
| 8 | Ichimoku | Tendance | Multi-signal |
| 9 | Supertrend | Tendance | Volatilité |
| 10 | Donchian | Tendance | Breakouts |
| 11 | RSI OS/OB | Momentum | Reversals |
| 12 | RSI Centerline | Momentum | Trend momentum |
| 13 | RSI Divergence | Momentum | Reversals |
| 14 | Stochastic | Momentum | Range trading |
| 15 | Stoch Double | Momentum | Strong reversals |
| 16 | CCI | Momentum | Cycles |
| 17 | Williams %R | Momentum | Oversold/bought |
| 18 | MFI | Momentum | Volume confirm |
| 19 | BB Bounce | Volatilité | Range |
| 20 | BB Squeeze | Volatilité | Breakouts |
| 21 | BB %B | Volatilité | Extremes |
| 22 | Keltner | Volatilité | Breakouts |
| 23 | ATR Breakout | Volatilité | Volatility trades |
| 24 | VCP | Volatilité | Compression |
| 25 | Pivot Points | S/R | Intraday |
| 26 | Fibonacci | S/R | Retracements |
| 27 | S/R Breakout | S/R | Breakouts |
| 28 | Pin Bar | S/R | Reversals |
| 29 | Engulfing | S/R | Reversals |
| 30 | EMA+RSI+ADX | Combiné | Multi-confirm |
| 31 | MACD+Stoch | Combiné | Double confirm |
| 32 | BB+RSI | Combiné | Volatility+momentum |
| 33 | Ichimoku+MACD | Combiné | Trend+momentum |
| 34 | Triple Screen | Combiné | Multi-timeframe |
| 35 | MA Ribbon | Combiné | Trend strength |
| 36 | Mean Reversion | Avancé | Statistical edge |
| 37 | Momentum BO | Avancé | Breakouts |
| 38 | VWAP | Avancé | Intraday |
| 39 | Order Flow | Avancé | Volume analysis |
| 40 | Range BO | Avancé | Session trading |

---

## Exemple de Résultat

### CADJPY - Stratégie Validée
```
PAIRE: CADJPY
STRATÉGIE: EMA Crossover Triple (#1)

PARAMÈTRES:
  - EMA périodes: 8, 21, 50
  - R:R: 2.5
  - ADX min: 25
  - RSI range: 35-65
  - Score min: 6

RÈGLES D'ENTRÉE:
  - BUY: EMA8 > EMA21 > EMA50 (alignement haussier)
  - SELL: EMA8 < EMA21 < EMA50 (alignement baissier)
  - Filtres: ADX >= 25, RSI entre 35-65

GESTION RISQUE:
  - Lot size: 0.15
  - Max perte/jour: $500

PERFORMANCE:
  - Profit Factor: 1.10
  - Win Rate: ~35%
  - Max Drawdown: 18.4%
  - Trades validés: 370
```

### EURGBP - Stratégie Validée
```
PAIRE: EURGBP
STRATÉGIE: Stochastic Crossover (#14)

PARAMÈTRES:
  - Stochastic: période=14, smooth=3
  - R:R: 2.0
  - Zone oversold: < 30
  - Zone overbought: > 70

RÈGLES D'ENTRÉE:
  - BUY: %K < 30 + crossover haussier (%K croise %D vers le haut)
  - SELL: %K > 70 + crossover baissier (%K croise %D vers le bas)

RÉGIMES À TRADER:
  - LOW_VOLATILITY: Meilleur PF historique
  - RANGING: Stochastique excelle en range

RÉGIMES À ÉVITER:
  - HIGH_VOLATILITY: Faux signaux fréquents
  - STRONG_TREND: Contre-tendance trop risqué

GESTION RISQUE:
  - Lot size: 0.15
  - Max perte/jour: $500

PERFORMANCE:
  - Profit Factor: 1.10
  - Win Rate: ~46%
```

---

## Notes Importantes

1. **Significativité statistique**: Minimum 100 trades pour valider une stratégie
2. **Overfitting**: Ne pas sur-optimiser, garder des paramètres raisonnables
3. **Walk-forward**: Idéalement tester sur données hors-échantillon
4. **Slippage**: Ajouter 1-2 pips de slippage dans les simulations
5. **Spread**: Inclure le spread typique de la paire dans les calculs
6. **Corrélation**: Éviter de trader plusieurs stratégies corrélées
7. **Drawdown**: Ne jamais dépasser 20-30% de drawdown max
8. **Position sizing**: Risquer max 1-2% du capital par trade
