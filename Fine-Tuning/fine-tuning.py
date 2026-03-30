import os, json, argparse, warnings, time
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Paths ────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)

DATA_PATH       = os.path.join(ROOT, "player_transfer_value_with_sentiment.csv")
PRED_UNI_PATH   = os.path.join(ROOT, "predictions_univariate.csv")
PRED_MULTI_PATH = os.path.join(ROOT, "predictions_multivariate.csv")
PRED_ENC_PATH   = os.path.join(ROOT, "predictions_encoder_decoder.csv")
MODELS_DIR      = os.path.join(BASE, "tuned_models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── Features ──────────────────────────────────────────────────
PERF = [
    'current_age','age_squared','age_decay_factor','season_encoded','position_encoded',
    'matches','minutes_played','minutes_per_match','goals','assists','shots',
    'passes_total','passes_complete','pass_accuracy_pct','tackles_total','tackles_won',
    'tackle_success_rate','dribbles','interceptions','fouls_committed',
    'goals_per90','assists_per90','shots_per90','goal_contributions_per90',
    'shot_conversion_rate','assist_to_goal_ratio','defensive_actions_per90',
    'dribbles_per90','attacking_output_index',
    'pos_Defender','pos_Forward','pos_Goalkeeper','pos_Midfielder',
    'minutes_played_tier_encoded','pass_accuracy_tier_encoded',
]
MARKET    = ['log_market_value','market_value_tier_encoded','transfer_attractiveness_score']
SENTIMENT = ['total_tweets','total_likes','positive_tweets','negative_tweets',
             'tweet_engagement_rate','social_buzz_score','vader_positive_score',
             'vader_negative_score','vader_compound_score','tb_polarity','tb_subjectivity',
             'positive_count','negative_count','neutral_count']
INJURY    = ['total_injuries','total_days_injured','total_matches_missed',
             'injury_burden_index','availability_rate','injury_frequency',
             'total_injuries_tier_encoded']
ALL_F  = PERF + MARKET + SENTIMENT + INJURY
LAG    = ['prev_value','prev_value2','value_trend','value_trend2','pct_change']
FINAL  = ALL_F + LAG


# ================================================================
# DATA LOADING
# ================================================================
def load_and_build():
    print("📂  Loading data...")
    df = pd.read_csv(DATA_PATH)
    SEASON_ORDER = {'2019/20':0,'2020/21':1,'2021/22':2,'2022/23':3,'2023/24':4}
    df['season_idx'] = df['season'].map(SEASON_ORDER)
    df = df.sort_values(['player_name','season_idx']).reset_index(drop=True)

    pf = df.groupby('player_name').filter(lambda x: len(x) == 5).copy()
    pf = pf.sort_values(['player_name','season_idx']).reset_index(drop=True)
    pf['prev_value']  = pf.groupby('player_name')['market_value_eur'].shift(1)
    pf['prev_value2'] = pf.groupby('player_name')['market_value_eur'].shift(2)
    pf['value_trend'] = pf['market_value_eur'] - pf['prev_value']
    pf['value_trend2']= pf['prev_value'] - pf['prev_value2']
    pf['pct_change']  = pf['value_trend'] / (pf['prev_value'] + 1e-6)
    pf = pf.dropna(subset=['prev_value2']).reset_index(drop=True)

    X     = pf[FINAL].fillna(0).values
    y     = pf['market_value_eur'].values
    y_log = np.log1p(y)

    train_mask = pf['season'].isin(['2019/20','2020/21','2021/22','2022/23']).values
    val_mask   = (pf['season'] == '2023/24').values

    print(f"   ✅ {len(pf)} samples — {train_mask.sum()} train / {val_mask.sum()} val")
    return df, pf, X, y, y_log, train_mask, val_mask


# ================================================================
# METRICS
# ================================================================
def compute_metrics(label, actual, pred):
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    mae  = float(mean_absolute_error(actual, pred))
    r2   = float(r2_score(actual, pred))
    mape = float(np.mean(np.abs((actual - pred) / (actual + 1e-8))) * 100)
    da   = float(np.mean((pred > np.mean(actual)) == (actual > np.mean(actual))) * 100)
    return {"model": label, "rmse": rmse, "mae": mae,
            "r2_pct": round(r2*100, 2), "mape_pct": round(mape, 2), "dir_acc_pct": round(da, 2)}


# ================================================================
# RANDOM SEARCH — XGBoost
# ================================================================
def random_search_xgb(X_tr, y_tr, X_val, y_val_raw, n_iter):
    param_dist = {
        'n_estimators':     [200, 400, 600, 800],
        'max_depth':        [4, 5, 6, 7, 8],
        'learning_rate':    [0.01, 0.02, 0.03, 0.05, 0.08],
        'subsample':        [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5, 7],
        'reg_alpha':        [0.0, 0.05, 0.1, 0.2],
        'reg_lambda':       [0.5, 1.0, 1.5, 2.0],
    }
    rng = np.random.RandomState(42)
    best_rmse, best_model, best_params, results = float('inf'), None, None, []

    for i in range(n_iter):
        p = {k: rng.choice(v) for k, v in param_dist.items()}
        m = xgb.XGBRegressor(**p, objective='reg:squarederror', tree_method='hist',
                              device='cpu', random_state=42, n_jobs=-1, verbosity=0)
        m.fit(X_tr, y_tr)
        pred = np.expm1(m.predict(X_val))
        rmse = float(np.sqrt(mean_squared_error(y_val_raw, pred)))
        r2   = float(r2_score(y_val_raw, pred) * 100)
        results.append({'trial': i+1, 'rmse': rmse, 'r2': r2,
                        **{k: (v.item() if hasattr(v, 'item') else v) for k, v in p.items()}})
        if rmse < best_rmse:
            best_rmse   = rmse
            best_params = {k: (v.item() if hasattr(v, 'item') else v) for k, v in p.items()}
            best_model  = m
        print(f"   XGB trial {i+1}/{n_iter}  RMSE=€{rmse/1e6:.2f}M  best=€{best_rmse/1e6:.2f}M")

    return best_model, best_params, best_rmse, pd.DataFrame(results)


# ================================================================
# RANDOM SEARCH — LightGBM
# ================================================================
def random_search_lgb(X_tr, y_tr, X_val, y_val_raw, n_iter):
    param_dist = {
        'n_estimators':      [200, 400, 600, 800],
        'num_leaves':        [31, 63, 127],
        'max_depth':         [5, 6, 7, 8, -1],
        'learning_rate':     [0.01, 0.02, 0.03, 0.05, 0.08],
        'subsample':         [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree':  [0.6, 0.7, 0.8, 0.9],
        'min_child_samples': [10, 20, 30, 50],
        'reg_alpha':         [0.0, 0.05, 0.1, 0.2],
        'reg_lambda':        [0.5, 1.0, 1.5, 2.0],
    }
    rng = np.random.RandomState(42)
    best_rmse, best_model, best_params, results = float('inf'), None, None, []

    for i in range(n_iter):
        p = {k: rng.choice(v) for k, v in param_dist.items()}
        m = lgb.LGBMRegressor(**p, objective='regression', random_state=42,
                               n_jobs=-1, verbose=-1)
        m.fit(X_tr, y_tr)
        pred = np.expm1(m.predict(X_val))
        rmse = float(np.sqrt(mean_squared_error(y_val_raw, pred)))
        r2   = float(r2_score(y_val_raw, pred) * 100)
        results.append({'trial': i+1, 'rmse': rmse, 'r2': r2,
                        **{k: (v.item() if hasattr(v, 'item') else v) for k, v in p.items()}})
        if rmse < best_rmse:
            best_rmse   = rmse
            best_params = {k: (v.item() if hasattr(v, 'item') else v) for k, v in p.items()}
            best_model  = m
        print(f"   LGB trial {i+1}/{n_iter}  RMSE=€{rmse/1e6:.2f}M  best=€{best_rmse/1e6:.2f}M")

    return best_model, best_params, best_rmse, pd.DataFrame(results)


# ================================================================
# MAIN
# ================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', type=int, default=20,
                        help='Random search trials per model (default: 20)')
    args = parser.parse_args()

    t0 = time.time()
    print("\n" + "="*60)
    print("  FOOTBALL PLAYER VALUATION — Model Tuning & Saving")
    print("="*60)

    # ── Load data ────────────────────────────────────────────
    df, pf, X, y, y_log, train_mask, val_mask = load_and_build()
    X_tr, y_tr   = X[train_mask], y_log[train_mask]
    X_val, y_val = X[val_mask],   y[val_mask]

    # ── Fit & save scaler ────────────────────────────────────
    scaler = MinMaxScaler()
    scaler.fit(pf[['market_value_eur']].values)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))

    # ── Save feature names ───────────────────────────────────
    with open(os.path.join(MODELS_DIR, 'feature_names.json'), 'w') as f:
        json.dump(FINAL, f, indent=2)

    # ── Save dataset stats ───────────────────────────────────
    stats = {
        'n_players':    int(df['player_name'].nunique()),
        'n_seasons':    int(df['season'].nunique()),
        'seasons':      sorted(df['season'].unique().tolist()),
        'positions':    sorted(df['position'].unique().tolist()),
        'n_train_rows': int(train_mask.sum()),
        'n_val_rows':   int(val_mask.sum()),
        'val_season':   '2023/24',
    }
    with open(os.path.join(MODELS_DIR, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)

    # ── Train XGBoost baseline ────────────────────────────────
    print("\n🔷  Training XGBoost baseline...")
    xgb_def = xgb.XGBRegressor(
        n_estimators=600, max_depth=6, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, tree_method='hist',
        device='cpu', verbosity=0, n_jobs=-1, random_state=42)
    xgb_def.fit(X_tr, y_tr)

    # ── Train LightGBM baseline ───────────────────────────────
    print("🔷  Training LightGBM baseline...")
    lgb_def = lgb.LGBMRegressor(
        n_estimators=600, num_leaves=63, max_depth=7,
        learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
        verbose=-1, n_jobs=-1, random_state=42)
    lgb_def.fit(X_tr, y_tr)

    # ── Hyperparameter tuning ─────────────────────────────────
    print(f"\n🔧  XGBoost random search ({args.n_iter} trials)...")
    xgb_tuned, xgb_params, xgb_best_rmse, xgb_trials = \
        random_search_xgb(X_tr, y_tr, X_val, y_val, args.n_iter)

    print(f"\n🔧  LightGBM random search ({args.n_iter} trials)...")
    lgb_tuned, lgb_params, lgb_best_rmse, lgb_trials = \
        random_search_lgb(X_tr, y_tr, X_val, y_val, args.n_iter)

    # ── LSTM predictions (pre-computed CSVs) ──────────────────
    print("\n📊  Loading LSTM pre-computed predictions...")
    pred_uni   = pd.read_csv(PRED_UNI_PATH)
    pred_multi = pd.read_csv(PRED_MULTI_PATH)
    pred_enc   = pd.read_csv(PRED_ENC_PATH)

    lstm_uni_pred   = pred_uni['univariate_predicted_market_value'].values
    lstm_multi_pred = pred_multi['predicted_market_value'].values
    lstm_avg_pred   = (lstm_uni_pred + lstm_multi_pred) / 2.0
    lstm_actual     = pred_uni['actual_market_value'].values

    enc_t1 = pred_enc[pred_enc['forecast_step'] == 1].reset_index(drop=True)
    enc_t2 = pred_enc[pred_enc['forecast_step'] == 2].reset_index(drop=True)

    # Align lengths for ensemble
    n                = len(lstm_actual)
    idx              = np.linspace(0, len(y_val)-1, n, dtype=int)
    y_val_aligned    = y_val[idx]
    xgb_pred_aligned = np.expm1(xgb_tuned.predict(X_val))[idx]
    lgb_pred_aligned = np.expm1(lgb_tuned.predict(X_val))[idx]

    # Grid search for optimal ensemble weights
    print("\n🔍  Searching for optimal ensemble weights...")
    best_ens_rmse = float('inf')
    best_weights = {'w_lstm': 0.6, 'w_xgb': 0.4, 'w_lgb': 0.0}
    
    for i in range(21):
        for j in range(21 - i):
            w_lstm = i * 0.05
            w_xgb = j * 0.05
            w_lgb = round(1.0 - w_lstm - w_xgb, 2)
            
            if w_lgb < 0.0: continue
                
            ens_temp = w_lstm * lstm_avg_pred + w_xgb * xgb_pred_aligned + w_lgb * lgb_pred_aligned
            rmse_temp = float(np.sqrt(mean_squared_error(y_val_aligned, ens_temp)))
            if rmse_temp < best_ens_rmse:
                best_ens_rmse = rmse_temp
                best_weights = {'w_lstm': round(w_lstm, 2), 'w_xgb': round(w_xgb, 2), 'w_lgb': round(w_lgb, 2)}
                
    print(f"   Best weights: LSTM={best_weights['w_lstm']:.2f}, XGB={best_weights['w_xgb']:.2f}, LGBM={best_weights['w_lgb']:.2f}")
    print(f"   Ensemble RMSE: €{best_ens_rmse/1e6:.2f}M")
    
    ens_pred = (best_weights['w_lstm'] * lstm_avg_pred + 
                best_weights['w_xgb'] * xgb_pred_aligned + 
                best_weights['w_lgb'] * lgb_pred_aligned)

    # ── Compute & save metrics ────────────────────────────────
    print("\n📈  Computing evaluation metrics...")
    all_metrics = [
        compute_metrics("Univariate LSTM",          lstm_actual,              lstm_uni_pred),
        compute_metrics("Multivariate LSTM",         lstm_actual,              lstm_multi_pred),
        compute_metrics("Enc-Decoder LSTM (t+1)",   enc_t1['actual'].values,  enc_t1['predicted'].values),
        compute_metrics("Enc-Decoder LSTM (t+2)",   enc_t2['actual'].values,  enc_t2['predicted'].values),
        compute_metrics("XGBoost (tuned)",           y_val,   np.expm1(xgb_tuned.predict(X_val))),
        compute_metrics("LightGBM (tuned)",          y_val,   np.expm1(lgb_tuned.predict(X_val))),
        compute_metrics("Ensemble (Optimal Weights) ★", y_val_aligned, ens_pred),
    ]
    with open(os.path.join(MODELS_DIR, 'model_metrics.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)

    # ── Feature importance ────────────────────────────────────
    fi = pd.Series(xgb_tuned.feature_importances_, index=FINAL).nlargest(25)
    fi.to_json(os.path.join(MODELS_DIR, 'feature_importance.json'))

    # ── Save tuning trial results ─────────────────────────────
    xgb_trials.to_csv(os.path.join(MODELS_DIR, 'tuning_results_xgb.csv'), index=False)
    lgb_trials.to_csv(os.path.join(MODELS_DIR, 'tuning_results_lgb.csv'), index=False)

    # ── Save tuning metadata ──────────────────────────────────
    meta = {
        'xgboost':  {'best_rmse': xgb_best_rmse, 'best_params': xgb_params, 'n_trials': args.n_iter},
        'lightgbm': {'best_rmse': lgb_best_rmse, 'best_params': lgb_params, 'n_trials': args.n_iter},
        'ensemble_weights': best_weights,
        'saved_at': pd.Timestamp.now().isoformat(),
    }
    with open(os.path.join(MODELS_DIR, 'tuning_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # ── Save models ───────────────────────────────────────────
    print("\n💾  Saving models...")
    xgb_tuned.save_model(os.path.join(MODELS_DIR, 'xgboost_tuned.json'))
    lgb_tuned.booster_.save_model(os.path.join(MODELS_DIR, 'lightgbm_tuned.txt'))
    joblib.dump(lgb_tuned, os.path.join(MODELS_DIR, 'lightgbm_tuned.pkl'))

    # ── Save player features ──────────────────────────────────
    pf.to_csv(os.path.join(MODELS_DIR, 'player_features.csv'), index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  ✅ Done in {elapsed:.0f}s  —  all artefacts saved to ./tuned_models/")
    print("="*60)

    # ── Print metrics summary ─────────────────────────────────
    print("\n📋  Evaluation Summary (Validation Set 2023/24):\n")
    mdf = pd.DataFrame(all_metrics)
    mdf['rmse_m'] = mdf['rmse'].apply(lambda v: f"€{v/1e6:.2f}M")
    mdf['mae_m']  = mdf['mae'].apply(lambda v: f"€{v/1e6:.2f}M")
    mdf = mdf.rename(columns={'model':'Model','r2_pct':'R²%','mape_pct':'MAPE%',
                               'dir_acc_pct':'DirAcc%','rmse_m':'RMSE','mae_m':'MAE'})
    print(mdf[['Model','RMSE','MAE','R²%','MAPE%','DirAcc%']].to_string(index=False))
    print()


if __name__ == '__main__':
    main()
