# Sistema de Trading Sistemático con Señales de ML
### Proyecto Final — Trading Algorítmico | UAI Magíster Finanzas | 2026

> **Autores:** Sebastián Moscoso Guajardo · Joaquín Ocare  
> **Activo:** SPY (S&P 500 ETF) | **Horizonte:** 5 días | **Período OOS:** Abr 2024 – Abr 2026

---

## Resultados principales

| Métrica | XGBoost Continua | XGBoost Binaria | Buy & Hold | Momentum 20D |
|---|---|---|---|---|
| **Sharpe OOS** | 2.49 | 2.98 | 2.67 | 4.12 |
| **Calmar** | 2.07 | 3.14 | 2.36 | 5.98 |
| **Max Drawdown** | -16.2% | -36.5% | -56.5% | -21.6% |
| **Retorno Total** | +80.2% | +372.6% | +460.5% | +442.1% |

**AUC-ROC OOS:** 0.537 | **Prueba de permutación:** p-value = 0.000 (n=100)  
**Feature más importante:** VIX_norm (régimen de volatilidad implícita)

---

## Estructura del repositorio

```
proyecto_trading/
│
├── E1_Features_EDA.ipynb          # Datos, features y EDA (Parte A — Sebastián)
├── E2_B1_ML_Pipeline.ipynb        # Walk-forward CV, modelos, SHAP (Parte B1 — Sebastián)
├── E2_Backtest.ipynb              # Backtesting y análisis de overfitting (Parte B2 — Joaquín)
│
├── features.csv                   # Features engineered + target (output E1)
├── data_meta.json                 # Metadata del pipeline (feature_cols, horizonte, etc.)
├── predictions.csv                # Probabilidades predichas OOS (output E2_B1)
├── resultados_ml.json             # Métricas completas del pipeline (output E2_Backtest)
│
├── eda_features.png               # EDA: precio, distribuciones, RSI, volatilidad
├── heatmap_correlaciones.png      # Matriz de correlación entre features
├── metricas_por_fold.png          # Walk-forward CV: métricas por fold
├── curva_roc.png                  # Curva ROC OOS: XGBoost vs RF vs Baseline
├── shap_importance.png            # SHAP values: Top 10 features
├── equity_curve.png               # Equity curve + drawdown OOS
├── overfitting_sharpe.png         # Sharpe IS vs OOS
├── permutation_test.png           # Distribución de Sharpe bajo H₀
├── retornos_mensuales.png         # Heatmap de retornos mensuales
├── dashboard_final.png            # Dashboard consolidado
│
└── README.md                      # Este archivo
```

---

## Instalación y ejecución

### 1. Instalar dependencias

```bash
pip install yfinance xgboost optuna shap fredapi scikit-learn \
            pandas numpy matplotlib seaborn jupyter
```

### 2. Ejecutar el pipeline completo

Los notebooks deben ejecutarse en orden. Cada uno genera los archivos que necesita el siguiente.

```bash
# Notebook 1: Datos y features
jupyter nbconvert --to notebook --execute E1_Features_EDA.ipynb
# Output: features.csv, data_meta.json

# Notebook 2: ML pipeline
jupyter nbconvert --to notebook --execute E2_B1_ML_Pipeline.ipynb
# Input:  features.csv, data_meta.json
# Output: predictions.csv, resultados_ml_parte1.json

# Notebook 3: Backtesting
jupyter nbconvert --to notebook --execute E2_Backtest.ipynb
# Input:  predictions.csv, data_meta.json
# Output: resultados_ml.json, todos los gráficos
```

> **Reproducibilidad:** todos los notebooks usan `random_state=42` y `np.random.seed(42)`.

---

## Pipeline del proyecto

```
[Yahoo Finance]  →  E1: Feature Engineering  →  features.csv
[FRED API]              (26 features, 3 cat.)      data_meta.json
                             │
                             ▼
                    E2-B1: Walk-Forward CV    →  predictions.csv
                    XGBoost + RF + Optuna        resultados_ml_parte1.json
                    30 trials, 5 folds
                    Embargo = 5 días
                             │
                             ▼
                    E2-B2: Backtesting       →  resultados_ml.json
                    Equity curve                 dashboard_final.png
                    Prueba de permutación
                    Análisis de overfitting
```

---

## Decisiones de diseño clave

### Prevención de data leakage
- **Features:** exclusivamente ventanas rolling backward (sin información futura)
- **Target:** calculado con `shift(-5)` al final del pipeline, después de todos los features
- **Normalización:** StandardScaler fit solo sobre el subconjunto train de cada fold; nunca sobre validación o test
- **Embargo:** 5 días de gap entre train y val en cada fold (= horizonte de predicción)

### Walk-forward CV con expanding window
```python
# Expanding window: el train crece en cada fold
# Gap de embargo: 5 días entre el último dato de train y el primero de val
train_end = min_train + i * fold_size
val_start = train_end + embargo_days  # 5 días de buffer
```

### Desbalance de clases
- **XGBoost:** `scale_pos_weight = n_clase_0 / n_clase_1 ≈ 0.598`
- **Random Forest:** `class_weight='balanced'`

### Conversión de probabilidad a posición
```python
# Sizing continuo (estrategia principal)
pos(t) = 2 * P(retorno > 0 | X_t) - 1  ∈ [-1, 1]

# Costos de transacción: 5 bps por cambio de posición
costo = 0.0005 * abs(pos[t] - pos[t-1])
```

---

## Resumen metodológico

| Componente | Decisión |
|---|---|
| **Datos** | SPY diario, yfinance, 2016-2026, auto_adjust=True |
| **Features** | 26 features: 15 técnicos, 4 volumen, 7 régimen |
| **Target** | Binario: retorno 5d > 0 → clase 1 |
| **Split** | 80% train / 20% test OOS (temporal estricto) |
| **Validación** | Walk-forward CV, 5 folds, expanding window |
| **Embargo** | 5 días entre train y val |
| **Modelos** | XGBoost + Random Forest |
| **Optimización** | Optuna, 30 trials, métrica AUC-ROC |
| **Explicabilidad** | SHAP TreeExplainer |
| **Costos** | 5 bps por trade |
| **Robustez** | Prueba de permutación, 100 iteraciones |

---

## Interpretación de resultados

El AUC-ROC OOS de **0.537** es modesto pero estadísticamente significativo (p-value = 0.000 en prueba de permutación). Este resultado es coherente con la hipótesis de eficiencia débil: el SPY es uno de los activos más eficientemente valuados del mundo, y una ventaja predictiva del 3.7% sobre el azar representa una señal explotable en la práctica.

La dominancia de **features de régimen** (VIX_norm, skew_21, vol_ratio_regime) sobre los técnicos puros confirma que la predictibilidad de retornos en el SPY es condicional al estado de la volatilidad y el riesgo agregado del mercado, no a patrones de precio per se.

El **Sharpe OOS de 2.49** con un max drawdown de solo -16.2% posiciona a la estrategia continua como competitiva en términos de riesgo-retorno frente al Buy & Hold (-56.5% de drawdown), a pesar de generar menor retorno total en un período de mercado extraordinariamente alcista.
