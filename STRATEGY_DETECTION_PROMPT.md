# Prompt IA - Détection de Stratégie Forex Optimale

## Objectif
Trouver la meilleure stratégie de trading pour une paire Forex donnée, avec paramètres optimisés et règles de gestion du risque validées par backtest.

---

## Étapes à Suivre

### ÉTAPE 1: COLLECTE DES DONNÉES
```
═══════════════════════════════════════════════════════════════
SOURCES DE DONNÉES RECOMMANDÉES (par ordre de priorité)
═══════════════════════════════════════════════════════════════

1) TWELVE DATA (RECOMMANDÉ pour backtest historique)
   - Site: https://twelvedata.com
   - Avantages:
     * Données 1H disponibles depuis 2010+ pour forex
     * 800 requêtes/jour (tier gratuit)
     * API REST simple et fiable
   - Utilisation:
     ```python
     from core.multi_source_fetcher import MultiSourceFetcher
     fetcher = MultiSourceFetcher()
     df = fetcher.fetch('CADJPY', '2020-01-01', '2024-12-01', '1h')
     ```
   - Clé API: Stocker dans api_keys.py

2) FINNHUB (RECOMMANDÉ pour données temps réel)
   - Site: https://finnhub.io
   - Avantages:
     * 60 requêtes/minute (tier gratuit)
     * WebSocket pour streaming
     * Données temps réel
   - Format symbole: OANDA:CAD_JPY

3) YAHOO FINANCE (FALLBACK uniquement)
   - Via yfinance Python library
   - LIMITATION IMPORTANTE: Données 1H limitées aux 730 derniers jours
   - Utiliser uniquement si Twelve Data / Finnhub indisponibles
   - Format symbole: CADJPY=X

CONFIGURATION API (api_keys.py):
```python
class APIKeys(Enum):
    TWELVE_DATA ="0d071977c5cf4da9981b55d7c26f59a3"
    FINNHUB = "cv9l6ghr01qpd9s7rhj0cv9l6ghr01qpd9s7rhjg"
```

PÉRIODES DE DONNÉES REQUISES:
- Minimum: 2 ans de données 1H pour validation
- Recommandé: 5 ans pour tests multi-régimes
- Incluant: COVID (2020), Ukraine (2022), Crise bancaire (2023)

FORMAT OUTPUT:
- DataFrame pandas avec colonnes: open, high, low, close, volume
- Index: datetime UTC
- Fréquence: 1H pour scalping, 1D pour swing
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
═══════════════════════════════════════════════════════════════
A) RÉGIMES TECHNIQUES (détection automatique via indicateurs)
═══════════════════════════════════════════════════════════════

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

═══════════════════════════════════════════════════════════════
A-BIS) DÉTECTION AVANCÉE DES RÉGIMES (Machine Learning)
═══════════════════════════════════════════════════════════════

Techniques ML pour classification automatique des régimes:

  1) HIDDEN MARKOV MODELS (HMM)
     - Modélise les régimes comme états cachés
     - Détecte automatiquement les transitions
     - Idéal pour: persistence des régimes, prédiction de changement
     - Usage: Filtre risk-off (éviter trades en régime volatile)

  2) CLUSTERING (K-Means, GMM)
     - Groupe les périodes par caractéristiques similaires
     - Features: volatilité, momentum, corrélations
     - Gaussian Mixture Models pour distributions multi-modales
     - Avantage: Détection non-supervisée de régimes inconnus

  3) VOLATILITY CLUSTERING (GARCH)
     - Modélise le phénomène où volatilité engendre volatilité
     - Prédit les périodes de haute/basse volatilité
     - Utile pour: ajuster position sizing, timing d'entrée

  4) REGIME CHANGE DETECTION
     - Détecter les points de transition entre régimes
     - Techniques: CUSUM, changement de variance, rupture structurelle
     - Application: Réduire exposition avant changement de régime

Features recommandées pour clustering:
  | Feature | Description | Calcul |
  |---------|-------------|--------|
  | Volatility | ATR normalisé | ATR / Close * 100 |
  | Momentum | ROC 20 périodes | (Close - Close[20]) / Close[20] |
  | Trend Strength | ADX | ADX(14) |
  | Mean Reversion | Distance à EMA | (Close - EMA50) / EMA50 |
  | Correlation | Corr avec index | Corr(pair, DXY, 20) |
  | Volume Profile | Volume relatif | Volume / SMA(Volume, 20) |

Classification recommandée (4-6 régimes):
  - RISK_ON: Faible volatilité, momentum positif
  - RISK_OFF: Haute volatilité, flight-to-safety
  - TRENDING: ADX élevé, direction claire
  - MEAN_REVERTING: Basse volatilité, oscillation
  - CRISIS: Volatilité extrême, corrélations cassées
  - TRANSITION: Changement de régime en cours

═══════════════════════════════════════════════════════════════
B) RÉGIMES MACROÉCONOMIQUES (périodes historiques à tester)
═══════════════════════════════════════════════════════════════

La stratégie DOIT être testée sur ces différentes conditions de marché:

  1) CRISE SANITAIRE / PANDÉMIE
     - Période: Mars 2020 - Juin 2021 (COVID-19)
     - Caractéristiques: Volatilité extrême, gaps, corrélations cassées
     - Test: La stratégie survit-elle au crash initial ?

  2) GUERRE / CONFLIT GÉOPOLITIQUE
     - Périodes:
       * Février 2022+ (Ukraine-Russie)
       * Tensions Moyen-Orient (périodiques)
     - Caractéristiques: Flight-to-safety (USD, CHF, JPY), pétrole volatile
     - Test: Performance en risk-off intense

  3) CRISE BANCAIRE / FINANCIÈRE
     - Périodes:
       * Mars 2023 (SVB, Credit Suisse)
       * 2008-2009 (Lehman, référence historique si données dispo)
     - Caractéristiques: Spreads élargis, liquidité réduite
     - Test: Drawdown max acceptable ?

  4) CYCLE DE HAUSSE DES TAUX (Hawkish)
     - Période: 2022-2023 (Fed, BCE hausse agressive)
     - Caractéristiques: USD fort, carry trades inversés, obligataire volatile
     - Test: La stratégie suit-elle le trend macro ?

  5) CYCLE DE BAISSE DES TAUX (Dovish)
     - Périodes: 2019-2020, 2024+
     - Caractéristiques: Risk-on, USD faible, actions fortes
     - Test: Performance en environnement accomodant

  6) INFLATION ÉLEVÉE
     - Période: 2021-2023 (inflation 8%+)
     - Caractéristiques: Volatilité élevée, réactions aux CPI
     - Test: Stabilité face aux chocs d'inflation

  7) MARCHÉ "NORMAL" / GOLDILOCKS
     - Périodes: 2017-2019, 2024 H2
     - Caractéristiques: Volatilité faible, trends lisses
     - Test: Benchmark de performance standard

  8) FLASH CRASH / BLACK SWAN
     - Événements spécifiques:
       * CHF Janvier 2015 (suppression du floor)
       * GBP Flash Crash Octobre 2016
       * JPY Flash Crash Janvier 2019
     - Test: Survie au pire scénario (SL respecté ?)

═══════════════════════════════════════════════════════════════
C) RÉGIMES LIÉS AUX NEWS / ÉVÉNEMENTS RÉCURRENTS
═══════════════════════════════════════════════════════════════

  1) DÉCISIONS BANQUES CENTRALES
     - Fed (FOMC): 8x/an
     - BCE: 8x/an
     - BoE: 8x/an
     - BoJ: 8x/an
     - Test: Performance H-1 à H+4 autour des annonces

  2) DONNÉES ÉCONOMIQUES MAJEURES
     - NFP (Non-Farm Payrolls): 1er vendredi du mois
     - CPI (Inflation): ~12x/an par pays
     - GDP: Trimestriel
     - Test: Faut-il filtrer ces périodes ?

  3) SESSIONS DE TRADING
     - Asian Session: 00:00-09:00 UTC
     - London Session: 07:00-16:00 UTC
     - NY Session: 12:00-21:00 UTC
     - Overlap London-NY: 12:00-16:00 UTC (max liquidité)
     - Test: Performance par session

  4) JOUR DE LA SEMAINE
     - Lundi: Gaps potentiels, volume faible
     - Mardi-Jeudi: Volume optimal
     - Vendredi: Position closing, news NFP
     - Test: Performance par jour

  5) FIN DE MOIS / TRIMESTRE / ANNÉE
     - Rebalancing institutionnel
     - Window dressing
     - Test: Patterns saisonniers exploitables ?

═══════════════════════════════════════════════════════════════
D) TABLEAU RÉCAPITULATIF DES PÉRIODES DE BACKTEST OBLIGATOIRES
═══════════════════════════════════════════════════════════════

| Période | Dates | Type | Priorité |
|---------|-------|------|----------|
| COVID Crash | Mars 2020 | Crise | OBLIGATOIRE |
| COVID Recovery | Avril-Dec 2020 | Recovery | OBLIGATOIRE |
| Inflation Surge | 2021-2022 | Macro | OBLIGATOIRE |
| Fed Hiking | 2022-2023 | Hawkish | OBLIGATOIRE |
| Ukraine War | Fév 2022+ | Géopolitique | OBLIGATOIRE |
| Banking Crisis | Mars 2023 | Crise | OBLIGATOIRE |
| Rate Pivot | 2024 | Dovish | OBLIGATOIRE |
| Normal Market | 2017-2019 | Baseline | OBLIGATOIRE |
| Flash Crashes | Spécifiques | Black Swan | RECOMMANDÉ |

Action:
  1. Backtester la stratégie SÉPARÉMENT pour chaque régime technique
  2. Backtester sur TOUTES les périodes macroéconomiques obligatoires
  3. Calculer le PF par régime pour identifier les conditions favorables/défavorables
  4. REJETER toute stratégie avec PF < 0.8 sur COVID ou période de crise

Résultat:
  - Identifier quand trader vs quand éviter
  - Définir des filtres de régime pour le trading live
```

### ÉTAPE 6: TESTS DE ROBUSTESSE AVANCÉS (ANTI-OVERFITTING)
```
═══════════════════════════════════════════════════════════════
A) MONTE CARLO SIMULATION (minimum 500 itérations)
═══════════════════════════════════════════════════════════════

Objectif: Vérifier que la stratégie n'est pas dépendante d'un ordre
spécifique des trades (robustesse statistique)

Méthode:
  1. Exécuter backtest normal → obtenir liste des trades
  2. Mélanger aléatoirement l'ordre des trades (shuffle)
  3. Recalculer equity curve avec nouvel ordre
  4. Répéter 500-1000 fois

Métriques à analyser:
  - Distribution du Max Drawdown (percentile 95%)
  - Distribution du Profit Factor
  - Probabilité de ruine (equity < 50% du capital initial)
  - Dispersion des courbes d'equity

Critères de validation:
  ✅ 95% des simulations terminent en positif
  ✅ Max Drawdown au 95ème percentile < 35%
  ✅ Probabilité de ruine < 5%
  ❌ REJETER si dispersion trop large (stratégie fragile)

═══════════════════════════════════════════════════════════════
B) WALK-FORWARD OPTIMIZATION (WFO)
═══════════════════════════════════════════════════════════════

Objectif: Éviter l'overfitting en optimisant sur des fenêtres glissantes

Méthode:
  - Diviser données en segments (ex: 12 mois par segment)
  - Pour chaque segment:
    * Optimiser sur 70% (in-sample)
    * Tester sur 30% (out-of-sample)
  - Avancer d'un segment et répéter

Exemple sur 5 ans:
  | Période | In-Sample (Optimisation) | Out-of-Sample (Test) |
  |---------|--------------------------|----------------------|
  | Seg 1 | Jan 2019 - Sep 2019 | Oct 2019 - Dec 2019 |
  | Seg 2 | Apr 2019 - Dec 2019 | Jan 2020 - Mar 2020 |
  | Seg 3 | Jul 2019 - Mar 2020 | Apr 2020 - Jun 2020 |
  | ... | ... | ... |

Métrique clé: Walk-Forward Efficiency (WFE)
  WFE = (Performance Out-of-Sample) / (Performance In-Sample)
  ✅ WFE > 50% = stratégie robuste
  ⚠️ WFE 30-50% = acceptable avec prudence
  ❌ WFE < 30% = overfitting probable

═══════════════════════════════════════════════════════════════
C) DÉTECTION D'OVERFITTING
═══════════════════════════════════════════════════════════════

Signaux d'alerte d'overfitting:
  ❌ Grande différence In-Sample vs Out-of-Sample (> 40%)
  ❌ Trop de paramètres optimisés (> 5-6)
  ❌ Stratégie trop complexe (> 4 conditions combinées)
  ❌ Performance exceptionnelle sur backtest (PF > 2.0 suspect)
  ❌ Faible nombre de trades (< 100)

Règles anti-overfitting:
  ✅ Garder stratégie simple (max 3-4 indicateurs)
  ✅ Réserver 30% des données pour validation finale
  ✅ Préférer paramètres standards (RSI 14, pas RSI 13.7)
  ✅ Vérifier cohérence sur plusieurs paires similaires

═══════════════════════════════════════════════════════════════
D) TESTS DE STRESS ADDITIONNELS
═══════════════════════════════════════════════════════════════

  1) PARAMETER JITTER TEST
     - Modifier légèrement chaque paramètre (+/- 10-20%)
     - La stratégie doit rester profitable
     - Si PF s'effondre = overfitting sur paramètres exacts

  2) EXECUTION DEGRADATION TEST
     - Ajouter 1-3 pips de slippage aléatoire
     - Ajouter délai d'exécution simulé (1-5 secondes)
     - La stratégie doit rester viable

  3) SPREAD VARIATION TEST
     - Tester avec spreads élargis (x1.5, x2, x3)
     - Simuler conditions de faible liquidité (nuit, news)

  4) DATA NOISE TEST
     - Ajouter bruit aléatoire aux prix (+/- 0.5 pip)
     - Vérifier que signaux restent stables
```

### ÉTAPE 7: VALIDATION CONTRAINTE RISQUE
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

### ÉTAPE 8: VALIDATION FINALE (PAPER TRADING)
```
═══════════════════════════════════════════════════════════════
A) FORWARD TESTING (Paper Trading)
═══════════════════════════════════════════════════════════════

Après le backtest, le forward testing est ESSENTIEL:
  - Exécuter la stratégie en mode démo/paper trading
  - Durée minimale: 4-8 semaines
  - Ne PAS modifier la stratégie pendant cette période

Objectifs:
  - Valider comportement sur données temps réel non vues
  - Détecter problèmes d'exécution (slippage réel, requotes)
  - Confirmer que les signaux sont exploitables en pratique

═══════════════════════════════════════════════════════════════
B) BACKTEST VALIDATION
═══════════════════════════════════════════════════════════════

Backtest du dernier mois en simulation horaire:
  - Simuler exécution toutes les heures
  - Capital initial: $10,000 (ou capital réel)
  - Lot size: valeur validée à l'étape 7

Vérifier:
  - ROI positif
  - Profit Factor > 1.0
  - Contrainte journalière respectée
  - Win Rate cohérent avec backtest long terme
  - Cohérence avec résultats Monte Carlo

═══════════════════════════════════════════════════════════════
C) CRITÈRES DE VALIDATION FINALE
═══════════════════════════════════════════════════════════════

La stratégie est VALIDÉE si elle passe TOUS ces tests:

  □ Profit Factor > 1.0 sur backtest complet
  □ Profit Factor > 0.8 sur TOUTES les périodes de crise
  □ Walk-Forward Efficiency > 50%
  □ Monte Carlo: 95% des simulations positives
  □ Monte Carlo: Max DD au 95ème percentile < 35%
  □ Parameter Jitter: PF reste > 0.9 avec +/- 15% sur params
  □ Forward test: performance cohérente sur 4+ semaines
  □ Nombre de trades significatif (> 100 sur backtest)
```

### ÉTAPE 9: OUTPUT FINAL
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
═══════════════════════════════════════════════════════════════
PHASE 1: DONNÉES & STRATÉGIES
═══════════════════════════════════════════════════════════════
□ 1. Télécharger données historiques (2-10 ans, H1/D1)
□ 2. Tester les 40 stratégies candidates
□ 3. Garder stratégies avec PF >= 1.0 et trades >= 100
□ 4. Grid search paramètres (R:R, ADX, RSI, score, périodes)

═══════════════════════════════════════════════════════════════
PHASE 2: ROBUSTESSE MULTI-RÉGIMES (CRITIQUE)
═══════════════════════════════════════════════════════════════
□ 5. Classifier régimes techniques (volatilité, trend, range)
□ 6. Backtester par régime technique → PF par condition
□ 7. Backtester périodes macroéconomiques OBLIGATOIRES:
     □ COVID Crash (Mars 2020)
     □ COVID Recovery (Avril-Dec 2020)
     □ Inflation Surge (2021-2022)
     □ Fed Hiking Cycle (2022-2023)
     □ Ukraine War (Fév 2022+)
     □ Banking Crisis (Mars 2023)
     □ Rate Pivot / Dovish (2024)
     □ Normal Market baseline (2017-2019)
□ 8. REJETER si PF < 0.8 sur période de crise
□ 9. Tester autour des news (FOMC, NFP, CPI)
□ 10. Tester par session (Asian, London, NY)

═══════════════════════════════════════════════════════════════
PHASE 3: TESTS ANTI-OVERFITTING (NOUVEAU)
═══════════════════════════════════════════════════════════════
□ 11. Monte Carlo Simulation (500+ itérations)
      □ 95% des simulations positives
      □ Max DD au 95ème percentile < 35%
      □ Probabilité de ruine < 5%
□ 12. Walk-Forward Optimization
      □ Diviser en segments 70/30 (in/out of sample)
      □ Walk-Forward Efficiency > 50%
□ 13. Tests de stress:
      □ Parameter Jitter (+/- 15%)
      □ Execution Degradation (slippage 1-3 pips)
      □ Spread Variation (x1.5, x2)
      □ Data Noise (+/- 0.5 pip)

═══════════════════════════════════════════════════════════════
PHASE 4: VALIDATION FINALE
═══════════════════════════════════════════════════════════════
□ 14. Tester lot sizes avec contrainte perte max journalière
□ 15. Forward Testing (paper trading 4-8 semaines)
□ 16. Validation finale: backtest dernier mois hourly
□ 17. Documenter stratégie avec:
      - Paramètres optimaux
      - Régimes à trader vs éviter
      - Performance par condition de marché
      - Résultats Monte Carlo & WFO
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

PERFORMANCE PAR RÉGIME TECHNIQUE:
  | Régime | PF | Trades | Recommandation |
  |--------|-----|--------|----------------|
  | STRONG_TREND | 1.35 | 120 | ✅ TRADER |
  | TRENDING_UP | 1.18 | 85 | ✅ TRADER |
  | TRENDING_DOWN | 1.12 | 78 | ✅ TRADER |
  | RANGING | 0.82 | 45 | ❌ ÉVITER |
  | LOW_VOLATILITY | 0.91 | 42 | ⚠️ PRUDENCE |

PERFORMANCE PAR PÉRIODE MACROÉCONOMIQUE:
  | Période | Dates | PF | Trades | Statut |
  |---------|-------|-----|--------|--------|
  | COVID Crash | Mars 2020 | 0.95 | 28 | ✅ SURVÉCU |
  | COVID Recovery | Avr-Dec 2020 | 1.42 | 65 | ✅ EXCELLENT |
  | Inflation Surge | 2021-2022 | 1.08 | 142 | ✅ OK |
  | Fed Hiking | 2022-2023 | 1.15 | 98 | ✅ BON |
  | Ukraine War | Fév 2022+ | 1.05 | 75 | ✅ OK |
  | Banking Crisis | Mars 2023 | 0.88 | 12 | ⚠️ PRUDENCE |
  | Normal Market | 2017-2019 | 1.12 | 180 | ✅ BASELINE |

GESTION RISQUE:
  - Lot size: 0.15
  - Max perte/jour: $500

PERFORMANCE GLOBALE:
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

RÉGIMES TECHNIQUES À TRADER:
  ✅ LOW_VOLATILITY: PF=1.25 (meilleur)
  ✅ RANGING: PF=1.18 (Stochastique excelle)
  ✅ CONSOLIDATION: PF=1.15

RÉGIMES TECHNIQUES À ÉVITER:
  ❌ HIGH_VOLATILITY: PF=0.72 (faux signaux)
  ❌ STRONG_TREND: PF=0.68 (contre-tendance risqué)
  ❌ BREAKOUT: PF=0.75 (timing mauvais)

PERFORMANCE PAR PÉRIODE MACROÉCONOMIQUE:
  | Période | PF | Statut |
  |---------|-----|--------|
  | COVID Crash | 0.85 | ⚠️ PRUDENCE |
  | COVID Recovery | 1.22 | ✅ BON |
  | Inflation Surge | 0.95 | ⚠️ NEUTRE |
  | Fed Hiking | 1.08 | ✅ OK |
  | Normal Market | 1.18 | ✅ BASELINE |

FILTRES DE SESSION:
  ✅ London: PF=1.21 (optimal)
  ✅ NY: PF=1.14 (bon)
  ❌ Asian: PF=0.88 (éviter)

GESTION RISQUE:
  - Lot size: 0.15
  - Max perte/jour: $500

PERFORMANCE GLOBALE:
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

---

## Références & Sources

### Techniques de Validation
- [Monte Carlo Backtesting for Trading Strategies](https://www.blog.quantreo.com/2024/02/26/monte-carlo-backtesting/) - Quantreo Blog
- [Walk-Forward Optimization](https://blog.quantinsti.com/walk-forward-optimization-introduction/) - QuantInsti
- [Out of Sample Testing for Robust Strategies](https://www.buildalpha.com/out-of-sample-testing/) - Build Alpha
- [5 Monte Carlo Methods to Bulletproof Strategies](https://strategyquant.com/blog/new-robustness-tests-on-the-strategyquant-codebase-5-monte-carlo-methods-to-bulletproof-your-trading-strategies/) - StrategyQuant

### Détection d'Overfitting
- [What Is Overfitting in Trading Strategies](https://www.luxalgo.com/blog/what-is-overfitting-in-trading-strategies/) - LuxAlgo
- [How to Avoid Overfitting](https://platform.algotradingspace.com/help/strategy-builders/express-generator/best-practices/how-to-avoid-overfitting/) - Algo Trading Space
- [Backtest Overfitting in ML Era (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011110) - ScienceDirect

### Détection des Régimes de Marché
- [Market Regime Detection using HMM](https://www.quantstart.com/articles/market-regime-detection-using-hidden-markov-models-in-qstrader/) - QuantStart
- [Market Regime Detection with ML](https://developers.lseg.com/en/article-catalog/article/market-regime-detection) - LSEG
- [Classifying Market Regimes](https://macrosynergy.com/research/classifying-market-regimes/) - Macrosynergy

### Recherche Académique
- [CPCV: Combinatorial Purged Cross-Validation](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4686376) - SSRN (2024)
- [N-Period Volatility Labeling Technique](https://onlinelibrary.wiley.com/doi/10.1155/2024/5036389) - Wiley (2024)
