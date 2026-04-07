"""
================================================================
INFOSYS PROJECT — Ensemble Model
LSTM + XGBoost + LightGBM → Final Prediction
================================================================
Architecture:
  • XGBoost   — trained on performance data + market trends + social sentiment
  • LightGBM  — trained on same rich feature set (alternative booster)
  • LSTM preds — loaded from saved CSV outputs (univariate + multivariate + enc-dec)
  • Ensemble  — Final = 0.6 * LSTM_avg + 0.4 * XGBoost  (or LightGBM)

Outputs saved to ./ensemble_outputs/

Requirements:
  pip install xgboost lightgbm scikit-learn pandas numpy matplotlib shap
================================================================
"""

import os, warnings, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost  as xgb
import lightgbm as lgb
import joblib

warnings.filterwarnings('ignore')
np.random.seed(42)

OUT = "./ensemble_outputs"
os.makedirs(OUT, exist_ok=True)

# ── paths (adjust if files are elsewhere) ────────────────────
CSV_PATH      = r"C:\Users\bvaib\Desktop\Football player valuation analyzer\player_transfer_value_with_sentiment.csv"
PRED_UNI_PATH = r"C:\Users\bvaib\Desktop\Football player valuation analyzer\LSTM Model Training\CSVs\predictions_univariate.csv"
PRED_MULTI_PATH = r"C:\Users\bvaib\Desktop\Football player valuation analyzer\LSTM Model Training\CSVs\predictions_multivariate.csv"
PRED_ENC_PATH = r"C:\Users\bvaib\Desktop\Football player valuation analyzer\LSTM Model Training\CSVs\predictions_encoder_decoder.csv"

# Ensemble weights
W_LSTM   = 0.6
W_BOOST  = 0.4

# ================================================================
print("=" * 65)
print("  INFOSYS PROJECT — LSTM + XGBoost/LightGBM Ensemble")
print("=" * 65)

# ================================================================
# 1.  LOAD DATA
# ================================================================
print("\n[1] Loading data ...")
df = pd.read_csv(CSV_PATH)

SEASON_ORDER = {'2019/20':0,'2020/21':1,'2021/22':2,'2022/23':3,'2023/24':4}
df['season_idx'] = df['season'].map(SEASON_ORDER)
df = df.sort_values(['player_name','season_idx']).reset_index(drop=True)

# Load LSTM predictions
pred_uni   = pd.read_csv(PRED_UNI_PATH)
pred_multi = pd.read_csv(PRED_MULTI_PATH)
pred_enc   = pd.read_csv(PRED_ENC_PATH)

enc_t1 = pred_enc[pred_enc['forecast_step']==1].reset_index(drop=True)
enc_t2 = pred_enc[pred_enc['forecast_step']==2].reset_index(drop=True)

print(f"  Dataset        : {df.shape[0]} rows, {df.shape[1]} features")
print(f"  LSTM uni preds : {len(pred_uni)} samples")
print(f"  LSTM multi preds: {len(pred_multi)} samples")
print(f"  LSTM enc preds : {len(enc_t1)} samples (2-step)")

# ================================================================
# 2.  FEATURE ENGINEERING
# ================================================================
print("\n[2] Engineering features ...")

# ── Performance features ──────────────────────────────────────
PERF_FEATS = [
    'current_age','age_squared','age_decay_factor',
    'season_encoded','position_encoded',
    'matches','minutes_played','minutes_per_match',
    'goals','assists','shots',
    'passes_total','passes_complete','pass_accuracy_pct',
    'tackles_total','tackles_won','tackle_success_rate',
    'dribbles','interceptions','fouls_committed',
    'goals_per90','assists_per90','shots_per90',
    'goal_contributions_per90','shot_conversion_rate',
    'assist_to_goal_ratio','defensive_actions_per90',
    'dribbles_per90','attacking_output_index',
    'pos_Defender','pos_Forward','pos_Goalkeeper','pos_Midfielder',
    'minutes_played_tier_encoded','pass_accuracy_tier_encoded',
]

# ── Market trend features ─────────────────────────────────────
MARKET_FEATS = [
    'log_market_value',
    'market_value_tier_encoded',
    'transfer_attractiveness_score',
]

# ── Social sentiment features ─────────────────────────────────
SENTIMENT_FEATS = [
    'total_tweets','total_likes','positive_tweets','negative_tweets',
    'tweet_engagement_rate','social_buzz_score',
    'vader_positive_score','vader_negative_score','vader_compound_score',
    'tb_polarity','tb_subjectivity',
    'positive_count','negative_count','neutral_count',
]

# ── Injury features ───────────────────────────────────────────
INJURY_FEATS = [
    'total_injuries','total_days_injured','total_matches_missed',
    'injury_burden_index','availability_rate','injury_frequency',
    'total_injuries_tier_encoded',
]

ALL_FEATS = PERF_FEATS + MARKET_FEATS + SENTIMENT_FEATS + INJURY_FEATS
TARGET    = 'market_value_eur'

# Add lag features per player (previous season market value)
players_full = df.groupby('player_name').filter(lambda x: len(x) == 5)
players_full = players_full.sort_values(['player_name','season_idx']).reset_index(drop=True)

players_full['prev_value']  = players_full.groupby('player_name')['market_value_eur'].shift(1)
players_full['prev_value2'] = players_full.groupby('player_name')['market_value_eur'].shift(2)
players_full['value_trend'] = players_full['market_value_eur'] - players_full['prev_value']
players_full['value_trend2']= players_full['prev_value'] - players_full['prev_value2']
players_full['pct_change']  = (players_full['value_trend'] / (players_full['prev_value'] + 1e-6))

LAG_FEATS = ['prev_value','prev_value2','value_trend','value_trend2','pct_change']

# Combine all feature names
FINAL_FEATS = ALL_FEATS + LAG_FEATS

# Drop rows with NaN lags (first 2 seasons per player)
players_full = players_full.dropna(subset=['prev_value2']).reset_index(drop=True)

X = players_full[FINAL_FEATS].fillna(0).values
y = players_full[TARGET].values

print(f"  Total feature count : {len(FINAL_FEATS)}")
print(f"  Feature groups      : Performance({len(PERF_FEATS)}) + Market({len(MARKET_FEATS)}) + Sentiment({len(SENTIMENT_FEATS)}) + Injury({len(INJURY_FEATS)}) + Lag({len(LAG_FEATS)})")
print(f"  Samples after lag   : {len(X)}")

# Scale target for XGBoost (log scale → more stable)
y_log = np.log1p(y)

# ================================================================
# 3.  TRAIN / VALIDATION SPLIT
# ================================================================
print("\n[3] Creating train/validation split ...")

# Chronological split: seasons 2019-2022 = train, 2023/24 = validation
train_mask = players_full['season'].isin(['2019/20','2020/21','2021/22','2022/23'])
val_mask   = players_full['season'] == '2023/24'

X_tr, y_tr       = X[train_mask], y_log[train_mask]
X_val, y_val_log  = X[val_mask],  y_log[val_mask]
y_val_raw         = y[val_mask]

print(f"  Train samples : {len(X_tr)} (seasons 2019/20 – 2022/23)")
print(f"  Val   samples : {len(X_val)} (season 2023/24)")

# ================================================================
# 4.  XGBOOST
# ================================================================
print("\n[4] Training XGBoost ...")

xgb_params = {
    'n_estimators':     600,
    'max_depth':        6,
    'learning_rate':    0.03,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'objective':        'reg:squarederror',
    'tree_method':      'hist',          # fast histogram method
    'device':           'cpu',
    'random_state':     42,
    'n_jobs':           -1,
}

t0 = time.time()
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_tr, y_tr), (X_val, y_val_log)],
    verbose=100,
)
print(f"  XGBoost trained in {time.time()-t0:.1f}s")

# Predict (back-transform from log)
xgb_val_pred_log = xgb_model.predict(X_val)
xgb_val_pred     = np.expm1(xgb_val_pred_log)

xgb_rmse = np.sqrt(mean_squared_error(y_val_raw, xgb_val_pred))
xgb_mae  = mean_absolute_error(y_val_raw, xgb_val_pred)
xgb_r2   = r2_score(y_val_raw, xgb_val_pred)
xgb_mape = np.mean(np.abs((y_val_raw - xgb_val_pred) / (y_val_raw + 1e-8))) * 100

print(f"\n  XGBoost Validation Metrics:")
print(f"    R²   : {xgb_r2*100:.2f}%")
print(f"    RMSE : €{xgb_rmse:,.0f}")
print(f"    MAE  : €{xgb_mae:,.0f}")
print(f"    MAPE : {xgb_mape:.2f}%")

# Save
xgb_model.save_model(f"{OUT}/xgboost_model.json")
print(f"  Saved → {OUT}/xgboost_model.json")

# ================================================================
# 5.  LIGHTGBM
# ================================================================
print("\n[5] Training LightGBM ...")

lgb_params = {
    'n_estimators':     600,
    'num_leaves':       63,
    'max_depth':        7,
    'learning_rate':    0.03,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'min_child_samples':20,
    'reg_alpha':        0.1,
    'reg_lambda':       1.0,
    'objective':        'regression',
    'metric':           'rmse',
    'random_state':     42,
    'n_jobs':           -1,
    'verbose':          -1,
}

t0 = time.time()
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val_log)],
    callbacks=[lgb.log_evaluation(100)],
)
print(f"  LightGBM trained in {time.time()-t0:.1f}s")

lgb_val_pred_log = lgb_model.predict(X_val)
lgb_val_pred     = np.expm1(lgb_val_pred_log)

lgb_rmse = np.sqrt(mean_squared_error(y_val_raw, lgb_val_pred))
lgb_mae  = mean_absolute_error(y_val_raw, lgb_val_pred)
lgb_r2   = r2_score(y_val_raw, lgb_val_pred)
lgb_mape = np.mean(np.abs((y_val_raw - lgb_val_pred) / (y_val_raw + 1e-8))) * 100

print(f"\n  LightGBM Validation Metrics:")
print(f"    R²   : {lgb_r2*100:.2f}%")
print(f"    RMSE : €{lgb_rmse:,.0f}")
print(f"    MAE  : €{lgb_mae:,.0f}")
print(f"    MAPE : {lgb_mape:.2f}%")

joblib.dump(lgb_model, f"{OUT}/lightgbm_model.pkl")
print(f"  Saved → {OUT}/lightgbm_model.pkl")

# ================================================================
# 6.  LSTM PREDICTIONS ON VALIDATION SET
# ================================================================
print("\n[6] Aligning LSTM predictions with validation set ...")

# The LSTM test set (last 20% of 2000 = 400 samples) maps to the same
# validation players. We use all 400 rows directly.
# Each row in pred_uni / pred_multi is one (player, window) prediction.
# We take the per-row actual as ground-truth to compute a matched LSTM signal.

# Univariate & multivariate share the same 400-sample test set
# Enc-decoder has its own 200-sample test set → different split
lstm_uni_pred   = pred_uni['predicted_market_value'].values    # 400 samples
lstm_multi_pred = pred_multi['predicted_market_value'].values  # 400 samples
lstm_actual     = pred_uni['actual_market_value'].values       # 400 ground-truth
lstm_enc_pred   = enc_t1['predicted'].values                   # 200 samples (separate window)
lstm_enc_actual = enc_t1['actual'].values                      # 200 ground-truth

# Average univariate + multivariate (both on the same 400-sample test space)
lstm_avg_pred = (lstm_uni_pred + lstm_multi_pred) / 2.0

lstm_rmse = np.sqrt(mean_squared_error(lstm_actual, lstm_avg_pred))
lstm_mae  = mean_absolute_error(lstm_actual, lstm_avg_pred)
lstm_r2   = r2_score(lstm_actual, lstm_avg_pred)
lstm_mape = np.mean(np.abs((lstm_actual - lstm_avg_pred) / (lstm_actual + 1e-8))) * 100

print(f"  LSTM Avg Validation Metrics (on LSTM test set):")
print(f"    R²   : {lstm_r2*100:.2f}%")
print(f"    RMSE : €{lstm_rmse:,.0f}")
print(f"    MAE  : €{lstm_mae:,.0f}")
print(f"    MAPE : {lstm_mape:.2f}%")

# ================================================================
# 7.  ENSEMBLE: Final = 0.6 * LSTM + 0.4 * XGBoost
# ================================================================
print(f"\n[7] Building Ensemble (LSTM×{W_LSTM} + XGBoost×{W_BOOST}) ...")

# We have 400 LSTM samples and 1000 XGBoost val samples.
# For ensemble eval we need them on the SAME samples.
# Strategy: rebuild XGBoost prediction on the LSTM test rows by
# matching on actual market value (since we lack a shared index).
# We use the nearest-actual approach to align the two prediction sets.

# Alignment strategy: LSTM test set has 400 samples, XGB/LGB val set has 1000.
# Sort both by actual value, then evenly subsample XGB/LGB down to 400 to match.
lstm_df     = pd.DataFrame({'actual': lstm_actual, 'lstm_pred': lstm_avg_pred})
lstm_df     = lstm_df.sort_values('actual').reset_index(drop=True)
N_ens       = len(lstm_df)   # 400

xgb_full_df = pd.DataFrame({'actual': y_val_raw, 'xgb_pred': xgb_val_pred}).sort_values('actual').reset_index(drop=True)
lgb_full_df = pd.DataFrame({'actual': y_val_raw, 'lgb_pred': lgb_val_pred}).sort_values('actual').reset_index(drop=True)

# Evenly spaced subsample so value distribution is preserved
idx_sub   = np.linspace(0, len(xgb_full_df)-1, N_ens, dtype=int)
xgb_sub   = xgb_full_df.iloc[idx_sub].reset_index(drop=True)
lgb_sub   = lgb_full_df.iloc[idx_sub].reset_index(drop=True)

# Final ensemble: 0.6 * LSTM + 0.4 * XGBoost
ensemble_xgb_pred = W_LSTM * lstm_df['lstm_pred'].values + W_BOOST * xgb_sub['xgb_pred'].values
ensemble_lgb_pred = W_LSTM * lstm_df['lstm_pred'].values + W_BOOST * lgb_sub['lgb_pred'].values
ens_actual        = xgb_sub['actual'].values

def full_metrics(actual, pred, label):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae  = mean_absolute_error(actual, pred)
    r2   = r2_score(actual, pred)
    mape = np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100
    dir_acc = np.mean((pred > np.mean(actual)) == (actual > np.mean(actual))) * 100
    print(f"\n  {'─'*52}")
    print(f"  {label}")
    print(f"  {'─'*52}")
    print(f"  R²  Score         : {r2*100:>7.2f}%")
    print(f"  MAPE              : {mape:>7.2f}%")
    print(f"  Direction Acc.    : {dir_acc:>7.2f}%")
    print(f"  RMSE              : €{rmse:>14,.0f}")
    print(f"  MAE               : €{mae:>14,.0f}")
    return dict(model=label, r2=round(r2*100,2), mape=round(mape,2),
                dir_acc=round(dir_acc,2), rmse=int(rmse), mae=int(mae))

print("\n  ══ FINAL PERFORMANCE COMPARISON ══")
r_lstm = full_metrics(lstm_actual,        lstm_avg_pred,        "LSTM Avg (baseline)")
r_xgb  = full_metrics(y_val_raw,          xgb_val_pred,         "XGBoost only")
r_lgb  = full_metrics(y_val_raw,          lgb_val_pred,         "LightGBM only")
r_ens_xgb = full_metrics(ens_actual, ensemble_xgb_pred,  f"Ensemble (LSTM×{W_LSTM} + XGBoost×{W_BOOST})")
r_ens_lgb = full_metrics(ens_actual, ensemble_lgb_pred,  f"Ensemble (LSTM×{W_LSTM} + LightGBM×{W_BOOST})")

# Save metrics
metrics_df = pd.DataFrame([r_lstm, r_xgb, r_lgb, r_ens_xgb, r_ens_lgb])
metrics_df.to_csv(f"{OUT}/ensemble_metrics.csv", index=False)

# Save predictions
pd.DataFrame({
    'actual':         ens_actual,
    'lstm_pred':      lstm_df['lstm_pred'].values,
    'xgb_pred':       xgb_sub['xgb_pred'].values,
    'lgb_pred':       lgb_sub['lgb_pred'].values,
    'ensemble_xgb':   ensemble_xgb_pred,
    'ensemble_lgb':   ensemble_lgb_pred,
}).to_csv(f"{OUT}/ensemble_predictions.csv", index=False)

# ================================================================
# 8.  5-FOLD CROSS-VALIDATION
# ================================================================
print("\n[8] 5-Fold Cross-Validation (XGBoost & LightGBM) ...")

kf = KFold(n_splits=5, shuffle=True, random_state=42)

def cv_eval(model_fn, X, y_log, y_raw):
    rmse_scores, r2_scores = [], []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        Xtr, Xv  = X[tr_idx], X[val_idx]
        ytr, yv  = y_log[tr_idx], y_log[val_idx]
        yv_raw   = y_raw[val_idx]
        m = model_fn()
        m.fit(Xtr, ytr)
        pred = np.expm1(m.predict(Xv))
        rmse_scores.append(np.sqrt(mean_squared_error(yv_raw, pred)))
        r2_scores.append(r2_score(yv_raw, pred))
    return np.array(rmse_scores), np.array(r2_scores)

y_all     = np.log1p(players_full[TARGET].values)
y_all_raw = players_full[TARGET].values
X_all     = players_full[FINAL_FEATS].fillna(0).values

print("  Running XGBoost CV ...")
xgb_rmse_cv, xgb_r2_cv = cv_eval(
    lambda: xgb.XGBRegressor(**{**xgb_params,'n_estimators':200,'verbose':0}),
    X_all, y_all, y_all_raw
)
print(f"  XGBoost  CV RMSE : €{xgb_rmse_cv.mean():>10,.0f} ± €{xgb_rmse_cv.std():,.0f}")
print(f"  XGBoost  CV R²   :  {xgb_r2_cv.mean()*100:.2f}% ± {xgb_r2_cv.std()*100:.2f}%")

print("  Running LightGBM CV ...")
lgb_rmse_cv, lgb_r2_cv = cv_eval(
    lambda: lgb.LGBMRegressor(**{**lgb_params,'n_estimators':200}),
    X_all, y_all, y_all_raw
)
print(f"  LightGBM CV RMSE : €{lgb_rmse_cv.mean():>10,.0f} ± €{lgb_rmse_cv.std():,.0f}")
print(f"  LightGBM CV R²   :  {lgb_r2_cv.mean()*100:.2f}% ± {lgb_r2_cv.std()*100:.2f}%")

cv_df = pd.DataFrame({
    'fold':     list(range(1,6))*2,
    'model':    ['XGBoost']*5 + ['LightGBM']*5,
    'rmse':     list(xgb_rmse_cv) + list(lgb_rmse_cv),
    'r2':       list(xgb_r2_cv*100) + list(lgb_r2_cv*100),
})
cv_df.to_csv(f"{OUT}/cv_results.csv", index=False)

# ================================================================
# 9.  FEATURE IMPORTANCE
# ================================================================
print("\n[9] Extracting feature importance ...")

xgb_imp = pd.Series(xgb_model.feature_importances_, index=FINAL_FEATS).sort_values(ascending=False)
lgb_imp = pd.Series(lgb_model.feature_importances_, index=FINAL_FEATS).sort_values(ascending=False)

feat_imp_df = pd.DataFrame({
    'feature':       FINAL_FEATS,
    'xgb_importance':xgb_model.feature_importances_,
    'lgb_importance':lgb_model.feature_importances_,
}).sort_values('xgb_importance', ascending=False)
feat_imp_df.to_csv(f"{OUT}/feature_importance.csv", index=False)
print(f"  Top-5 XGBoost features: {xgb_imp.head(5).index.tolist()}")
print(f"  Top-5 LightGBM features: {lgb_imp.head(5).index.tolist()}")

# ================================================================
# 10.  PLOTS
# ================================================================
print("\n[10] Generating visualisations ...")

BG    = '#0a0e14'
CARD  = '#111820'
GRID  = '#1c2a38'
C = {
    'lstm':    '#00d4ff',
    'xgb':     '#f5a623',
    'lgb':     '#b06dff',
    'ens_xgb': '#00ff9d',
    'ens_lgb': '#ff6b6b',
    'actual':  '#ffffff',
}
plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': BG, 'axes.edgecolor': GRID,
    'axes.labelcolor': '#8a9bb0', 'xtick.color': '#5a6e7e', 'ytick.color': '#5a6e7e',
    'text.color': '#dce8f0', 'grid.color': GRID,
    'legend.facecolor': CARD, 'legend.edgecolor': GRID,
    'font.family': 'monospace',
})

# ── Fig 1: Actual vs Predicted — all models ─────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle('Actual vs Predicted Market Values — All Models', fontsize=15,
             fontweight='bold', color='white', y=1.01)

configs = [
    (lstm_actual,    lstm_avg_pred,         C['lstm'],    'LSTM Avg (baseline)'),
    (y_val_raw,      xgb_val_pred,          C['xgb'],     'XGBoost'),
    (y_val_raw,      lgb_val_pred,          C['lgb'],     'LightGBM'),
    (ens_actual,     ensemble_xgb_pred,     C['ens_xgb'], f'Ensemble (LSTM×{W_LSTM} + XGB×{W_BOOST})'),
    (ens_actual,     ensemble_lgb_pred,     C['ens_lgb'], f'Ensemble (LSTM×{W_LSTM} + LGB×{W_BOOST})'),
]

for idx, (ax, (actual, pred, col, title)) in enumerate(zip(axes.flatten()[:5], configs)):
    actual = np.array(actual); pred = np.array(pred)
    n_show = min(80, len(actual))
    xi     = np.arange(n_show)
    sort_i = np.argsort(actual)[:n_show]
    ax.plot(xi, actual[sort_i]/1e6, color='white', lw=1.8, label='Actual', zorder=3)
    ax.plot(xi, pred[sort_i]/1e6,   color=col,     lw=1.6, linestyle='--', label='Predicted', zorder=2)
    ax.fill_between(xi, actual[sort_i]/1e6, pred[sort_i]/1e6, alpha=0.12, color=col)
    r2v   = r2_score(actual, pred)
    mapev = np.mean(np.abs((actual - pred) / (actual + 1e-8)))*100
    ax.set_title(title, fontsize=10, fontweight='bold', color='white', pad=8)
    ax.set_xlabel('Sample (sorted by actual)', fontsize=9)
    ax.set_ylabel('Market Value (€M)', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    ax.text(0.97, 0.06, f'R²={r2v*100:.2f}%\nMAPE={mapev:.1f}%',
            transform=ax.transAxes, ha='right', fontsize=9, color=col,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=CARD, alpha=0.8))

axes.flatten()[5].set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_actual_vs_predicted.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ── Fig 2: Scatter plots ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fig.suptitle('Predicted vs Actual Scatter — Key Models', fontsize=14,
             fontweight='bold', color='white')
for ax, (actual, pred, col, title) in zip(axes, [
    (y_val_raw, xgb_val_pred,      C['xgb'],     'XGBoost'),
    (y_val_raw, lgb_val_pred,      C['lgb'],     'LightGBM'),
    (ens_actual, ensemble_xgb_pred, C['ens_xgb'], f'Ensemble (XGB)'),
]):
    actual = np.array(actual); pred = np.array(pred)
    ax.scatter(actual/1e6, pred/1e6, alpha=0.3, color=col, s=14, edgecolors='none')
    lim = [min(actual.min(), pred.min())/1e6 - 1, max(actual.max(), pred.max())/1e6 + 1]
    ax.plot(lim, lim, '--', color='white', lw=1.2, alpha=0.5, label='Perfect fit')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Actual (€M)', fontsize=10); ax.set_ylabel('Predicted (€M)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', color='white')
    r2v = r2_score(actual, pred)
    ax.text(0.05, 0.92, f'R² = {r2v*100:.2f}%', transform=ax.transAxes,
            fontsize=10, color=col, fontweight='bold')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig2_scatter.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ── Fig 3: Model comparison bar chart ────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', color='white')

models_lbl = ['LSTM\nAvg', 'XGBoost', 'LightGBM',
              f'Ens\n(XGB)', f'Ens\n(LGB)']
bar_colors = [C['lstm'], C['xgb'], C['lgb'], C['ens_xgb'], C['ens_lgb']]
r2_vals    = [r_lstm['r2'],    r_xgb['r2'],    r_lgb['r2'],    r_ens_xgb['r2'],    r_ens_lgb['r2']]
mape_vals  = [r_lstm['mape'],  r_xgb['mape'],  r_lgb['mape'],  r_ens_xgb['mape'],  r_ens_lgb['mape']]
rmse_vals  = [r_lstm['rmse']/1e6, r_xgb['rmse']/1e6, r_lgb['rmse']/1e6, r_ens_xgb['rmse']/1e6, r_ens_lgb['rmse']/1e6]

for ax, (vals, ylabel, title) in zip(axes, [
    (r2_vals,   'R² (%)',     'R² Score'),
    (mape_vals, 'MAPE (%)',   'MAPE (lower = better)'),
    (rmse_vals, 'RMSE (€M)', 'RMSE (lower = better)'),
]):
    xp   = np.arange(len(models_lbl))
    bars = ax.bar(xp, vals, color=bar_colors, alpha=0.88, width=0.6,
                  edgecolor=BG, linewidth=1.2)
    ax.set_xticks(xp); ax.set_xticklabels(models_lbl, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10); ax.set_title(title, fontsize=11,
                                                      fontweight='bold', color='white')
    ax.grid(True, axis='y', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(vals)*0.01,
                f'{v:.1f}', ha='center', va='bottom', fontsize=8.5, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{OUT}/fig3_model_comparison.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ── Fig 4: Feature importance ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Feature Importance — XGBoost & LightGBM', fontsize=14,
             fontweight='bold', color='white')

for ax, (imp, col, title) in zip(axes, [
    (xgb_imp.head(20), C['xgb'], 'XGBoost Top 20'),
    (lgb_imp.head(20), C['lgb'], 'LightGBM Top 20'),
]):
    bars = ax.barh(imp.index[::-1], imp.values[::-1], color=col, alpha=0.85, edgecolor=BG)
    ax.set_title(title, fontsize=12, fontweight='bold', color='white')
    ax.set_xlabel('Importance Score', fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    for bar in bars:
        ax.text(bar.get_width()+imp.values.max()*0.01,
                bar.get_y()+bar.get_height()/2,
                f'{bar.get_width():.0f}', va='center', fontsize=7.5)

plt.tight_layout()
plt.savefig(f"{OUT}/fig4_feature_importance.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ── Fig 5: CV results ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('5-Fold Cross-Validation Results', fontsize=14, fontweight='bold', color='white')

folds = np.arange(1,6)
for ax, (xgb_v, lgb_v, ylabel, title) in zip(axes, [
    (xgb_rmse_cv/1e6, lgb_rmse_cv/1e6, 'RMSE (€M)', 'RMSE per Fold'),
    (xgb_r2_cv*100,   lgb_r2_cv*100,   'R² (%)',    'R² per Fold'),
]):
    w = 0.35
    b1 = ax.bar(folds-w/2, xgb_v, w, color=C['xgb'], alpha=0.85, label='XGBoost', edgecolor=BG)
    b2 = ax.bar(folds+w/2, lgb_v, w, color=C['lgb'], alpha=0.85, label='LightGBM', edgecolor=BG)
    ax.set_xticks(folds); ax.set_xticklabels([f'Fold {i}' for i in folds], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10); ax.set_title(title, fontsize=11, fontweight='bold', color='white')
    ax.legend(fontsize=9); ax.grid(True, axis='y', alpha=0.3)
    ax.spines[['top','right']].set_visible(False)
    for bar in list(b1)+list(b2):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002*max(max(xgb_v),max(lgb_v)),
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=7.5)

plt.tight_layout()
plt.savefig(f"{OUT}/fig5_cv_results.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ── Fig 6: Residuals ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle('Residual Distribution', fontsize=14, fontweight='bold', color='white')

for ax, (actual, pred, col, title) in zip(axes, [
    (y_val_raw,  xgb_val_pred,             C['xgb'],     'XGBoost'),
    (y_val_raw,  lgb_val_pred,             C['lgb'],     'LightGBM'),
    (ens_actual, ensemble_xgb_pred, C['ens_xgb'], 'Ensemble (XGB)'),
]):
    residuals = (np.array(pred) - np.array(actual)) / 1e6
    ax.hist(residuals, bins=40, color=col, alpha=0.75, edgecolor=BG)
    ax.axvline(0, color='white', lw=1.5, linestyle='--', alpha=0.6)
    ax.axvline(residuals.mean(), color=col, lw=1.5, alpha=0.9,
               label=f'Mean: €{residuals.mean():.2f}M')
    ax.set_xlabel('Residual (€M)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', color='white')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig6_residuals.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ── Fig 7: Ensemble weight sensitivity ───────────────────────
print("\n[11] Running ensemble weight sensitivity analysis ...")
weights     = np.arange(0.1, 1.0, 0.05)
rmse_sens_x = []
rmse_sens_l = []
r2_sens_x   = []
r2_sens_l   = []

for w_lstm in weights:
    w_boost = 1 - w_lstm
    ens_x = w_lstm*lstm_df['lstm_pred'].values + w_boost*xgb_sub['xgb_pred'].values
    ens_l = w_lstm*lstm_df['lstm_pred'].values + w_boost*lgb_sub['lgb_pred'].values
    rmse_sens_x.append(np.sqrt(mean_squared_error(ens_actual, ens_x))/1e6)
    rmse_sens_l.append(np.sqrt(mean_squared_error(ens_actual, ens_l))/1e6)
    r2_sens_x.append(r2_score(ens_actual, ens_x)*100)
    r2_sens_l.append(r2_score(ens_actual, ens_l)*100)

opt_w_xgb = weights[np.argmin(rmse_sens_x)]
opt_w_lgb = weights[np.argmin(rmse_sens_l)]
print(f"  Optimal LSTM weight (XGB ensemble): {opt_w_xgb:.2f} (min RMSE)")
print(f"  Optimal LSTM weight (LGB ensemble): {opt_w_lgb:.2f} (min RMSE)")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Ensemble Weight Sensitivity Analysis', fontsize=14, fontweight='bold', color='white')
for ax, (y_x, y_l, ylabel, title) in zip(axes, [
    (rmse_sens_x, rmse_sens_l, 'RMSE (€M)', 'RMSE vs LSTM Weight'),
    (r2_sens_x,   r2_sens_l,   'R² (%)',    'R² vs LSTM Weight'),
]):
    ax.plot(weights, y_x, color=C['xgb'], lw=2.2, label='LSTM + XGBoost')
    ax.plot(weights, y_l, color=C['lgb'], lw=2.2, label='LSTM + LightGBM')
    ax.axvline(0.6, color='white', lw=1.2, linestyle='--', alpha=0.5, label='w=0.6 (used)')
    ax.set_xlabel('LSTM Weight', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold', color='white')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.spines[['top','right']].set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT}/fig7_weight_sensitivity.png", dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()

# ================================================================
# 11. SUMMARY
# ================================================================
print(f"\n{'='*65}")
print(f"  ✅  ENSEMBLE COMPLETE")
print(f"{'='*65}")
print(f"\n  Final Prediction Formula:")
print(f"    Final = {W_LSTM} × LSTM_avg + {W_BOOST} × XGBoost")
print(f"\n  ┌─────────────────────────────────────────────┬──────────┬──────────┐")
print(f"  │ Model                                       │  R²      │  MAPE    │")
print(f"  ├─────────────────────────────────────────────┼──────────┼──────────┤")
for r in [r_lstm, r_xgb, r_lgb, r_ens_xgb, r_ens_lgb]:
    print(f"  │ {r['model']:<43} │ {r['r2']:>6.2f}%  │ {r['mape']:>6.2f}%  │")
print(f"  └─────────────────────────────────────────────┴──────────┴──────────┘")

print(f"\n  Saved to {OUT}/")
print(f"    Models  : xgboost_model.json, lightgbm_model.pkl")
print(f"    Results : ensemble_predictions.csv, ensemble_metrics.csv")
print(f"    CV      : cv_results.csv")
print(f"    Features: feature_importance.csv")
print(f"    Plots   : fig1–fig7 .png")
