"""
pipeline.py — Sistema de Trading Sistemático con Señales de ML
==============================================================
Proyecto Final | Trading Algorítmico | UAI Magíster Finanzas 2026

Ejecutar con:
    python pipeline.py

El script reproduce el pipeline completo end-to-end:
  Etapa 1 → Datos y feature engineering      → features.csv
  Etapa 2 → Walk-forward CV y modelos ML     → predictions.csv
  Etapa 3 → Backtesting y análisis           → resultados_ml.json

Requisitos:
    pip install yfinance xgboost optuna shap fredapi scikit-learn \
                pandas numpy matplotlib seaborn
"""

import warnings
warnings.filterwarnings('ignore')

import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # sin GUI — compatible con cualquier entorno
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates
import seaborn as sns

import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from xgboost import XGBClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
import shap

# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS GLOBALES
# ══════════════════════════════════════════════════════════════════════════════
TICKER       = "SPY"
START_DATE   = "2015-01-01"
HORIZONTE    = 5          # días para target y embargo
COSTO_BPS    = 0.0005     # 5 bps por trade
N_FOLDS      = 5
N_OPTUNA     = 30         # trials Optuna por modelo
N_PERM       = 100        # iteraciones prueba de permutación
RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


def log(msg, nivel=0):
    indent = "  " * nivel
    print(f"{indent}{msg}", flush=True)


def separador(titulo):
    ancho = 65
    print()
    print("═" * ancho)
    print(f"  {titulo}")
    print("═" * ancho)


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 1 — DATOS Y FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
def etapa1_features():
    separador("ETAPA 1 · Datos y Feature Engineering")
    from datetime import datetime
    end_date = datetime.today().strftime("%Y-%m-%d")

    # ── Descarga OHLCV ────────────────────────────────────────────────────────
    log(f"Descargando OHLCV {TICKER} ({START_DATE} → {end_date}) ...", 1)
    raw = yf.download(TICKER, start=START_DATE, end=end_date,
                      auto_adjust=True, progress=False)
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    raw.index   = pd.to_datetime(raw.index)
    raw         = raw[['Open', 'High', 'Low', 'Close', 'Volume']].dropna()
    log(f"✓ {len(raw):,} observaciones descargadas", 1)

    df = raw.copy()

    # ── Datos macro (FRED) ────────────────────────────────────────────────────
    try:
        from fredapi import Fred
        fred = Fred(api_key="f8e46e1ef4d18aaac1a83ec4c7e7ba85")
        vix  = fred.get_series('VIXCLS', observation_start=START_DATE,
                               observation_end=end_date)
        t10  = fred.get_series('GS10', observation_start=START_DATE,
                               observation_end=end_date)
        t2   = fred.get_series('GS2',  observation_start=START_DATE,
                               observation_end=end_date)
        df['VIX']          = vix.reindex(df.index, method='ffill').ffill().bfill()
        df['yield_spread'] = (t10 - t2).reindex(df.index, method='ffill').ffill().bfill()
        fred_ok = True
        log("✓ VIX y yield spread descargados (FRED)", 1)
    except Exception as e:
        log(f"⚠ FRED no disponible ({e}) → usando proxy de volatilidad", 1)
        df['VIX']          = df['Close'].pct_change().rolling(21).std() * np.sqrt(252) * 100
        df['yield_spread'] = np.nan
        fred_ok = False

    # ── Features técnicos ─────────────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 10]:
        df[f'ret_lag{lag}'] = df['Close'].pct_change(lag)

    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    df['RSI_14'] = 100 - 100 / (1 + gain / (loss + 1e-10))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_hist'] = (ema12 - ema26) - (ema12 - ema26).ewm(span=9, adjust=False).mean()

    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['BB_width'] = (2 * bb_std) / bb_mid
    df['BB_pos']   = (df['Close'] - (bb_mid - 2*bb_std)) / (4*bb_std + 1e-10)

    tr = pd.concat([
        df['High'] - df['Low'],
        (df['High'] - df['Close'].shift()).abs(),
        (df['Low']  - df['Close'].shift()).abs(),
    ], axis=1).max(axis=1)
    df['ATR_pct'] = tr.rolling(14).mean() / df['Close']

    df['SMA_cross_10_50']  = df['Close'].rolling(10).mean() / df['Close'].rolling(50).mean() - 1
    df['SMA_cross_50_200'] = df['Close'].rolling(50).mean() / df['Close'].rolling(200).mean() - 1

    for p in [10, 20, 60]:
        df[f'MOM_{p}'] = df['Close'].pct_change(p)

    # ── Features de volumen ───────────────────────────────────────────────────
    vol_ma  = df['Volume'].rolling(20).mean()
    vol_std = df['Volume'].rolling(20).std()
    df['vol_norm']  = (df['Volume'] - vol_ma) / (vol_std + 1e-10)
    df['vol_ratio'] = df['Volume'] / (vol_ma + 1e-10)

    obv = np.zeros(len(df))
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i-1]:
            obv[i] = obv[i-1] + df['Volume'].iloc[i]
        elif df['Close'].iloc[i] < df['Close'].iloc[i-1]:
            obv[i] = obv[i-1] - df['Volume'].iloc[i]
        else:
            obv[i] = obv[i-1]
    obv_s = pd.Series(obv, index=df.index)
    df['OBV_norm']  = obv_s / (obv_s.rolling(20).mean().abs() + 1e-10) - 1

    tp = (df['High'] + df['Low'] + df['Close']) / 3
    df['VWAP_dist'] = (df['Close'] - tp.rolling(20).mean()) / (tp.rolling(20).mean() + 1e-10)

    # ── Features de régimen ───────────────────────────────────────────────────
    ret = df['Close'].pct_change()
    df['vol_real_5']       = ret.rolling(5).std()  * np.sqrt(252)
    df['vol_real_21']      = ret.rolling(21).std() * np.sqrt(252)
    df['vol_ratio_regime'] = df['vol_real_5'] / (df['vol_real_21'] + 1e-10)
    df['skew_21']          = ret.rolling(21).skew()
    df['VIX_norm']         = df['VIX'] / (df['VIX'].rolling(252).mean() + 1e-10) - 1
    df['VIX_change']       = df['VIX'].pct_change(5)
    df['drawdown_252']     = df['Close'] / df['Close'].rolling(252).max() - 1

    if fred_ok and df['yield_spread'].notna().sum() > 200:
        df['yield_spread_norm'] = \
            df['yield_spread'] / (df['yield_spread'].rolling(252).mean() + 1e-10) - 1
        use_yield = True
    else:
        use_yield = False

    # ── Target y selección de features ───────────────────────────────────────
    df['fwd_return'] = df['Close'].pct_change(HORIZONTE).shift(-HORIZONTE)
    df['target']     = (df['fwd_return'] > 0).astype(int)

    FEATURE_COLS = [
        'ret_lag1', 'ret_lag2', 'ret_lag3', 'ret_lag5', 'ret_lag10',
        'RSI_14', 'MACD_hist', 'BB_width', 'BB_pos', 'ATR_pct',
        'SMA_cross_10_50', 'SMA_cross_50_200', 'MOM_10', 'MOM_20', 'MOM_60',
        'vol_norm', 'vol_ratio', 'OBV_norm', 'VWAP_dist',
        'vol_real_5', 'vol_real_21', 'vol_ratio_regime',
        'skew_21', 'VIX_norm', 'VIX_change', 'drawdown_252',
    ]
    if use_yield:
        FEATURE_COLS.append('yield_spread_norm')

    keep     = ['Open', 'High', 'Low', 'Close', 'Volume'] + FEATURE_COLS + ['fwd_return', 'target']
    df_clean = df[keep].dropna()
    df_clean['ticker'] = TICKER
    df_clean.index.name = 'date'

    # ── Guardar ───────────────────────────────────────────────────────────────
    df_clean.to_csv('features.csv')

    meta = {
        'ticker':         TICKER,
        'start_date':     str(df_clean.index[0].date()),
        'end_date':       str(df_clean.index[-1].date()),
        'n_obs':          len(df_clean),
        'n_features':     len(FEATURE_COLS),
        'feature_cols':   FEATURE_COLS,
        'target_col':     'target',
        'fwd_return_col': 'fwd_return',
        'horizonte_dias': HORIZONTE,
        'balance_clase_0': float((df_clean['target'] == 0).mean()),
        'balance_clase_1': float((df_clean['target'] == 1).mean()),
        'fred_disponible': fred_ok,
    }
    with open('data_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    log(f"✓ features.csv     : {df_clean.shape[0]:,} obs × {df_clean.shape[1]} col", 1)
    log(f"✓ data_meta.json   : {len(FEATURE_COLS)} features documentados", 1)
    log(f"  Balance target   : "
        f"Clase 0 = {meta['balance_clase_0']:.1%} | "
        f"Clase 1 = {meta['balance_clase_1']:.1%}", 1)

    return df_clean, FEATURE_COLS


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 2 — MODELOS ML (WALK-FORWARD CV + EVALUACIÓN OOS)
# ══════════════════════════════════════════════════════════════════════════════
def etapa2_modelos(df, FEATURE_COLS):
    separador("ETAPA 2 · Walk-Forward CV y Modelos ML")

    # Split temporal 80/20
    split_idx  = int(len(df) * 0.80)
    SPLIT_DATE = df.index[split_idx]
    df_train   = df.iloc[:split_idx].copy()
    df_test    = df.iloc[split_idx:].copy()
    X_train    = df_train[FEATURE_COLS]
    y_train    = df_train['target']
    X_test     = df_test[FEATURE_COLS]
    y_test     = df_test['target']

    log(f"Split temporal 80/20:", 1)
    log(f"  Train : {len(df_train):,} obs [{df_train.index[0].date()} → {df_train.index[-1].date()}]", 2)
    log(f"  Test  : {len(df_test):,} obs  [{df_test.index[0].date()} → {df_test.index[-1].date()}]", 2)

    # Walk-forward folds
    n         = len(X_train)
    min_train = int(n * 0.60)
    fold_size = (n - min_train) // N_FOLDS
    FOLDS     = []
    for i in range(N_FOLDS):
        te  = min_train + i * fold_size
        vs  = te + HORIZONTE
        ve  = min(vs + fold_size, n)
        if vs < ve:
            FOLDS.append((list(range(te)), list(range(vs, ve))))

    SCALE_POS = float((y_train == 0).sum() / (y_train == 1).sum())

    def eval_cv(build_fn):
        aucs = []
        for tr, val in FOLDS:
            sc = StandardScaler()
            Xtr = sc.fit_transform(X_train.iloc[tr])
            Xvl = sc.transform(X_train.iloc[val])
            m   = build_fn()
            m.fit(Xtr, y_train.iloc[tr])
            aucs.append(roc_auc_score(y_train.iloc[val],
                                      m.predict_proba(Xvl)[:, 1]))
        return float(np.mean(aucs))

    # Optuna XGBoost
    log("Optimizando XGBoost ...", 1)
    def obj_xgb(trial):
        p = dict(
            n_estimators     =trial.suggest_int('n_estimators', 100, 500),
            max_depth        =trial.suggest_int('max_depth', 3, 8),
            learning_rate    =trial.suggest_float('learning_rate', 0.01, 0.30, log=True),
            subsample        =trial.suggest_float('subsample', 0.6, 1.0),
            colsample_bytree =trial.suggest_float('colsample_bytree', 0.6, 1.0),
            reg_alpha        =trial.suggest_float('reg_alpha', 1e-4, 10.0, log=True),
            reg_lambda       =trial.suggest_float('reg_lambda', 1e-4, 10.0, log=True),
            scale_pos_weight =SCALE_POS, random_state=RANDOM_STATE,
            eval_metric='logloss', verbosity=0,
        )
        return eval_cv(lambda: XGBClassifier(**p))

    study_xgb = optuna.create_study(direction='maximize',
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_xgb.optimize(obj_xgb, n_trials=N_OPTUNA, show_progress_bar=False)
    PARAMS_XGB = study_xgb.best_params | {
        'scale_pos_weight': SCALE_POS, 'random_state': RANDOM_STATE,
        'eval_metric': 'logloss', 'verbosity': 0,
    }
    log(f"✓ XGBoost  — AUC-ROC CV: {study_xgb.best_value:.4f}", 1)

    # Optuna Random Forest
    log("Optimizando Random Forest ...", 1)
    def obj_rf(trial):
        p = dict(
            n_estimators     =trial.suggest_int('n_estimators', 100, 500),
            max_depth        =trial.suggest_int('max_depth', 3, 15),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf =trial.suggest_int('min_samples_leaf', 1, 10),
            max_features     =trial.suggest_categorical('max_features',['sqrt','log2',0.5]),
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1,
        )
        return eval_cv(lambda: RandomForestClassifier(**p))

    study_rf = optuna.create_study(direction='maximize',
                                   sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_rf.optimize(obj_rf, n_trials=N_OPTUNA, show_progress_bar=False)
    PARAMS_RF = study_rf.best_params | {
        'class_weight': 'balanced', 'random_state': RANDOM_STATE, 'n_jobs': -1,
    }
    log(f"✓ Random Forest — AUC-ROC CV: {study_rf.best_value:.4f}", 1)

    MODELO_GANADOR = 'XGBoost' if study_xgb.best_value >= study_rf.best_value else 'RandomForest'
    log(f"✓ Modelo ganador: {MODELO_GANADOR}", 1)

    # Métricas por fold
    def calc_folds(params, nombre):
        rows = []
        for i, (tr, val) in enumerate(FOLDS):
            sc  = StandardScaler()
            Xtr = sc.fit_transform(X_train.iloc[tr])
            Xvl = sc.transform(X_train.iloc[val])
            m   = XGBClassifier(**params) if nombre == 'XGBoost' \
                  else RandomForestClassifier(**params)
            m.fit(Xtr, y_train.iloc[tr])
            yp  = m.predict(Xvl)
            pr  = m.predict_proba(Xvl)[:, 1]
            yv  = y_train.iloc[val]
            rows.append({'fold': i+1, 'n_train': len(tr), 'n_val': len(val),
                'accuracy':  round(accuracy_score(yv, yp), 4),
                'precision': round(precision_score(yv, yp, zero_division=0), 4),
                'recall':    round(recall_score(yv, yp, zero_division=0), 4),
                'f1':        round(f1_score(yv, yp, zero_division=0), 4),
                'auc_roc':   round(roc_auc_score(yv, pr), 4),
            })
        return rows

    folds_xgb = calc_folds(PARAMS_XGB, 'XGBoost')
    folds_rf  = calc_folds(PARAMS_RF,  'RandomForest')

    # Entrenamiento final y evaluación OOS
    scaler_f = StandardScaler()
    Xtr_s    = scaler_f.fit_transform(X_train)
    Xte_s    = scaler_f.transform(X_test)

    if MODELO_GANADOR == 'XGBoost':
        model_f = XGBClassifier(**PARAMS_XGB)
    else:
        model_f = RandomForestClassifier(**PARAMS_RF)

    model_f.fit(Xtr_s, y_train)
    y_prob  = model_f.predict_proba(Xte_s)[:, 1]
    y_pred  = model_f.predict(Xte_s)

    ACC  = accuracy_score(y_test, y_pred)
    PREC = precision_score(y_test, y_pred, zero_division=0)
    REC  = recall_score(y_test, y_pred, zero_division=0)
    F1   = f1_score(y_test, y_pred, zero_division=0)
    AUC  = roc_auc_score(y_test, y_prob)

    log(f"✓ Evaluación OOS ({len(df_test):,} obs):", 1)
    log(f"  AUC-ROC  : {AUC:.4f}", 2)
    log(f"  F1       : {F1:.4f}", 2)
    log(f"  Accuracy : {ACC:.4f}", 2)

    # SHAP
    sample = np.random.choice(len(Xtr_s), size=min(500, len(Xtr_s)), replace=False)
    X_shap = pd.DataFrame(Xtr_s, columns=FEATURE_COLS).iloc[sample]
    exp    = shap.TreeExplainer(model_f)
    sv     = exp.shap_values(X_shap)
    sv     = sv[1] if isinstance(sv, list) else sv
    shap_imp = (pd.DataFrame({'feature': FEATURE_COLS,
                              'importance': np.abs(sv).mean(axis=0)})
                .sort_values('importance', ascending=False)
                .reset_index(drop=True))
    TOP_FEATURES = shap_imp['feature'].head(10).tolist()

    # Sharpe in-sample
    prob_is  = model_f.predict_proba(Xtr_s)[:, 1]
    pos_is   = pd.Series(2*prob_is - 1, index=X_train.index).shift(1).fillna(0)
    ret_is   = pos_is * df_train['fwd_return']
    SHARPE_IS = float((ret_is.mean() / ret_is.std()) * np.sqrt(252)) \
                if ret_is.std() > 1e-10 else 0.0

    # Guardar predictions.csv
    pred_df = df_test[['fwd_return', 'target']].copy()
    pred_df['prob_pred']  = y_prob
    pred_df['pred_label'] = y_pred
    pred_df.index.name    = 'date'
    pred_df.to_csv('predictions.csv')
    log("✓ predictions.csv guardado", 1)

    return {
        'df_train': df_train, 'df_test': df_test,
        'X_train': X_train, 'y_train': y_train,
        'X_test': X_test,   'y_test': y_test,
        'scaler_f': scaler_f, 'model_f': model_f,
        'MODELO_GANADOR': MODELO_GANADOR,
        'SPLIT_DATE': SPLIT_DATE,
        'y_prob': y_prob, 'y_pred': y_pred,
        'ACC': ACC, 'PREC': PREC, 'REC': REC, 'F1': F1, 'AUC': AUC,
        'SHARPE_IS': SHARPE_IS,
        'TOP_FEATURES': TOP_FEATURES, 'shap_imp': shap_imp,
        'folds_xgb': folds_xgb, 'folds_rf': folds_rf,
        'PARAMS_XGB': PARAMS_XGB, 'PARAMS_RF': PARAMS_RF,
        'pred_df': pred_df,
    }


# ══════════════════════════════════════════════════════════════════════════════
# ETAPA 3 — BACKTESTING Y ANÁLISIS
# ══════════════════════════════════════════════════════════════════════════════
def etapa3_backtest(ctx, FEATURE_COLS, meta):
    separador("ETAPA 3 · Backtesting y Análisis de Overfitting")

    pred  = ctx['pred_df'].copy()
    pred['pos_continua'] = 2 * pred['prob_pred'] - 1
    pred['pos_binaria']  = np.where(pred['prob_pred'] > 0.5, 1.0, 0.0)

    delta_c = pred['pos_continua'].diff().abs().fillna(0)
    delta_b = pred['pos_binaria'].diff().abs().fillna(0)
    pred['ret_continua'] = pred['pos_continua'].shift(1) * pred['fwd_return'] - delta_c * COSTO_BPS
    pred['ret_binaria']  = pred['pos_binaria'].shift(1)  * pred['fwd_return'] - delta_b * COSTO_BPS
    pred_bt = pred.iloc[1:].copy()

    ret_bh = pred_bt['fwd_return'].copy()
    mom_s  = pred['fwd_return'].rolling(20).sum().shift(1)
    pos_m  = pd.Series(np.where(mom_s.reindex(pred_bt.index) > 0, 1.0, 0.0), index=pred_bt.index)
    ret_mom = pos_m.shift(1).fillna(0) * pred_bt['fwd_return'] \
              - pos_m.diff().abs().fillna(0) * COSTO_BPS

    def met(r):
        r = r.dropna()
        sr  = float((r.mean()/r.std())*np.sqrt(252)) if r.std()>1e-10 else 0.0
        eq  = (1+r).cumprod()
        dd  = ((eq - eq.cummax())/eq.cummax()).min()
        ny  = len(r)/252
        ra  = float(eq.iloc[-1]**(1/ny)-1) if ny>0 else 0.0
        cal = ra/abs(dd) if abs(dd)>1e-10 else 0.0
        ra_ = r[r!=0]
        hit = float((ra_>0).mean()*100) if len(ra_)>0 else 0.0
        tot = float((eq.iloc[-1]-1)*100)
        return {'sharpe':round(sr,4),'calmar':round(cal,4),
                'max_drawdown':round(float(dd),4),'hit_rate':round(hit,2),
                'retorno_total':round(tot,2)}

    MET_C   = met(pred_bt['ret_continua'])
    MET_B   = met(pred_bt['ret_binaria'])
    MET_BH  = met(ret_bh)
    MET_MOM = met(ret_mom)

    log(f"Sharpe OOS (continua): {MET_C['sharpe']:.4f}", 1)
    log(f"Max Drawdown         : {MET_C['max_drawdown']:.4f}", 1)

    # Prueba de permutación
    np.random.seed(RANDOM_STATE)
    probs_oos = pred_bt['prob_pred'].values.copy()
    rets_oos  = pred_bt['fwd_return'].values.copy()
    sr_real   = MET_C['sharpe']
    srs_perm  = []
    for _ in range(N_PERM):
        pp = np.random.permutation(probs_oos)
        rp = np.roll(2*pp-1, 1) * rets_oos
        rp = rp[1:]
        srs_perm.append(float((rp.mean()/rp.std())*np.sqrt(252)) if rp.std()>1e-10 else 0.0)
    srs_perm = np.array(srs_perm)
    P_VALUE  = float((srs_perm >= sr_real).sum() / N_PERM)
    log(f"Prueba de permutación: p-value = {P_VALUE:.4f}", 1)

    SHARPE_OOS = MET_C['sharpe']
    RATIO      = round(SHARPE_OOS / ctx['SHARPE_IS'], 4) if ctx['SHARPE_IS'] != 0 else 0.0

    # ── Gráficos ──────────────────────────────────────────────────────────────
    eq_c   = (1 + pred_bt['ret_continua']).cumprod()
    eq_b   = (1 + pred_bt['ret_binaria']).cumprod()
    eq_bh  = (1 + ret_bh).cumprod()
    eq_mom = (1 + ret_mom).cumprod()

    # Equity curve
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    ax1 = axes[0]
    sr_c, sr_b, sr_bh, sr_mom = MET_C['sharpe'], MET_B['sharpe'], MET_BH['sharpe'], MET_MOM['sharpe']
    ax1.plot(eq_c.index,   eq_c,   label=f'XGB Continua (SR={sr_c:.2f})',
             color='royalblue', lw=2)
    ax1.plot(eq_b.index,   eq_b,   label=f'XGB Binaria  (SR={sr_b:.2f})',
             color='steelblue', lw=2, ls='--')
    ax1.plot(eq_bh.index,  eq_bh,  label=f'Buy & Hold   (SR={sr_bh:.2f})',
             color='darkorange', lw=2)
    ax1.plot(eq_mom.index, eq_mom, label=f'Momentum 20D (SR={sr_mom:.2f})',
             color='gray', lw=1.5, ls=':')
    ax1.axhline(1, color='black', lw=0.8, alpha=0.4)
    ax1.set_title('Equity Curve — Test Set OOS', fontweight='bold')
    ax1.set_ylabel('Riqueza acumulada (base = 1)')
    ax1.legend(loc='upper left')
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    ax2 = axes[1]
    dd_c  = ((eq_c  - eq_c.cummax())  / eq_c.cummax())  * 100
    dd_bh = ((eq_bh - eq_bh.cummax()) / eq_bh.cummax()) * 100
    ax2.fill_between(dd_c.index,  dd_c,  0, alpha=0.4, color='royalblue',  label='XGB Continua')
    ax2.fill_between(dd_bh.index, dd_bh, 0, alpha=0.3, color='darkorange', label='Buy & Hold')
    ax2.set_title('Drawdown (%)'); ax2.set_ylabel('Drawdown (%)')
    ax2.legend(); ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.tight_layout()
    plt.savefig('equity_curve.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Dashboard final
    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(eq_c.index,   eq_c,   label=f'XGB Continua (SR={sr_c:.2f})',
             color='royalblue', lw=2)
    ax1.plot(eq_bh.index,  eq_bh,  label=f'Buy & Hold   (SR={sr_bh:.2f})',
             color='darkorange', lw=2)
    ax1.plot(eq_mom.index, eq_mom, label=f'Momentum 20D (SR={sr_mom:.2f})',
             color='gray', lw=1.5, ls='--')
    ax1.axhline(1, color='black', lw=0.8, alpha=0.4)
    ax1.set_title('Equity Curve — Test Set OOS', fontweight='bold')
    ax1.legend(); ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.grid(alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.bar(['IS', 'OOS'], [ctx['SHARPE_IS'], SHARPE_OOS],
            color=['#e74c3c', '#27ae60'], width=0.4, edgecolor='white')
    ax2.axhline(1, color='navy', lw=1.5, ls='--', alpha=0.7)
    ax2.set_title('Sharpe IS vs OOS\n(Overfitting)', fontweight='bold')
    ax2.set_ylabel('Sharpe Anualizado')
    for i, v in enumerate([ctx['SHARPE_IS'], SHARPE_OOS]):
        ax2.text(i, v + 0.2, f'{v:.2f}', ha='center', fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(srs_perm, bins=20, color='#95a5a6', edgecolor='white', alpha=0.8)
    ax3.axvline(sr_real, color='royalblue', lw=2.5, ls='--',
                label=f'SR real = {sr_real:.2f}')
    ax3.set_title(f'Prueba Permutación\np-value = {P_VALUE:.3f}', fontweight='bold')
    ax3.set_xlabel('Sharpe Permutado'); ax3.legend(fontsize=9); ax3.grid(alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 2])
    top5   = ctx['shap_imp']['feature'].head(5).tolist()
    imp5   = ctx['shap_imp']['importance'].head(5).values
    colors = plt.cm.Blues(np.linspace(0.5, 0.9, 5))
    ax4.barh(range(5), imp5[::-1], color=colors, edgecolor='white')
    ax4.set_yticks(range(5)); ax4.set_yticklabels(top5[::-1])
    ax4.set_title('Top 5 Features SHAP', fontweight='bold')
    ax4.set_xlabel('Mean |SHAP value|'); ax4.grid(axis='x', alpha=0.3)

    fig.suptitle(
        f'Dashboard Final — Proyecto Trading Algorítmico UAI\n'
        f'SPY | {ctx["MODELO_GANADOR"]} | Test OOS: '
        f'{pred_bt.index[0].date()} → {pred_bt.index[-1].date()}',
        fontsize=14, fontweight='bold', y=1.02
    )
    plt.savefig('dashboard_final.png', dpi=150, bbox_inches='tight')
    plt.close()
    log("✓ equity_curve.png y dashboard_final.png guardados", 1)

    # JSON final
    def to_py(v):
        return float(v) if isinstance(v, (np.floating, np.integer)) else v

    resultados = {
        'modelo_ganador': ctx['MODELO_GANADOR'],
        'metricas_oos': {
            'accuracy': round(ctx['ACC'], 4), 'precision': round(ctx['PREC'], 4),
            'recall':   round(ctx['REC'], 4), 'f1':        round(ctx['F1'], 4),
            'auc_roc':  round(ctx['AUC'], 4),
        },
        'backtest_modelo':         {k: to_py(v) for k, v in MET_C.items()},
        'backtest_modelo_binario': {k: to_py(v) for k, v in MET_B.items()},
        'backtest_buyhold':        {k: to_py(v) for k, v in MET_BH.items()},
        'backtest_momentum':       {k: to_py(v) for k, v in MET_MOM.items()},
        'sharpe_in_sample':        round(ctx['SHARPE_IS'], 4),
        'sharpe_out_sample':       round(SHARPE_OOS, 4),
        'overfitting_ratio':       RATIO,
        'permutation_pvalue':      round(P_VALUE, 4),
        'permutation_sharpe_media':round(float(srs_perm.mean()), 4),
        'permutation_sharpe_std':  round(float(srs_perm.std()), 4),
        'permutation_n_iteraciones': N_PERM,
        'top_features_shap':       ctx['TOP_FEATURES'],
        'metricas_por_fold': {
            'xgboost':       ctx['folds_xgb'],
            'random_forest': ctx['folds_rf'],
        },
        'meta_datos': {
            'ticker':         meta['ticker'],
            'start_date':     meta['start_date'],
            'end_date':       meta['end_date'],
            'n_obs_total':    meta['n_obs'],
            'n_obs_oos':      len(pred_bt),
            'horizonte_dias': meta['horizonte_dias'],
            'split_date':     str(ctx['SPLIT_DATE'].date()),
            'costo_bps':      COSTO_BPS * 10000,
        },
    }

    with open('resultados_ml.json', 'w', encoding='utf-8') as f:
        json.dump(resultados, f, indent=2, ensure_ascii=False)
    log("✓ resultados_ml.json guardado", 1)

    return resultados


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    t0 = time.time()
    print()
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  PIPELINE — Sistema de Trading con Señales de ML            ║")
    print("║  Proyecto Final | Trading Algorítmico | UAI 2026            ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    df, FEATURE_COLS = etapa1_features()

    with open('data_meta.json') as f:
        meta = json.load(f)

    ctx = etapa2_modelos(df, FEATURE_COLS)
    res = etapa3_backtest(ctx, FEATURE_COLS, meta)

    separador("RESUMEN FINAL")
    items = [
        ("Modelo ganador",    res['modelo_ganador']),
        ("AUC-ROC OOS",       f"{res['metricas_oos']['auc_roc']:.4f}"),
        ("Sharpe in-sample",  f"{res['sharpe_in_sample']:.4f}"),
        ("Sharpe OOS",        f"{res['sharpe_out_sample']:.4f}"),
        ("Ratio OOS/IS",      f"{res['overfitting_ratio']:.4f}"),
        ("Max Drawdown OOS",  f"{res['backtest_modelo']['max_drawdown']:.4f}"),
        ("Retorno total OOS", f"{res['backtest_modelo']['retorno_total']:.2f}%"),
        ("p-value permut.",   f"{res['permutation_pvalue']:.4f}"),
        ("Top feature",       res['top_features_shap'][0]),
    ]
    for k, v in items:
        print(f"  {k:<25}: {v}")

    print()
    print("Archivos generados:")
    for fn in ['features.csv', 'data_meta.json', 'predictions.csv',
               'resultados_ml.json', 'equity_curve.png', 'dashboard_final.png']:
        print(f"  ✓ {fn}")

    elapsed = time.time() - t0
    print(f"\n  Tiempo total: {elapsed/60:.1f} min")
    print()
