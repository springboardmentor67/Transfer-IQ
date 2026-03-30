"""
Train and fine-tune an XGBoost regressor (GPU-enabled) to predict log_market_value.

Usage (PowerShell):
  python .\\train_xgb_regressor.py --data_path .\\player_transfer_value_with_sentiment.csv
"""

from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb
from xgboost import XGBRegressor


RANDOM_STATE = 42


@dataclass(frozen=True)
class SplitData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train + tune XGBoost regressor for log_market_value.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="player_transfer_value_with_sentiment.csv",
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="log_market_value",
        help="Target column to predict (default: log_market_value).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--model_out",
        type=str,
        default="outputs/final_xgb_model.json",
        help="Where to save the final trained model (XGBoost native format).",
    )
    parser.add_argument(
        "--preprocessor_out",
        type=str,
        default="outputs/preprocessor.pkl",
        help="Where to save the fitted preprocessor (pickle).",
    )
    parser.add_argument(
        "--split_strategy",
        type=str,
        default="time_season",
        choices=["random", "group_player", "time_season"],
        help=(
            "How to split data. "
            "'group_player' prevents the same player appearing across splits (recommended). "
            "'time_season' trains on earlier seasons and tests on later seasons. "
            "'random' is a simple row-wise split."
        ),
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=40,
        help="Number of RandomizedSearchCV parameter settings sampled.",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=3,
        help="Number of CV folds for RandomizedSearchCV.",
    )
    parser.add_argument(
        "--early_stopping_rounds",
        type=int,
        default=50,
        help="Early stopping rounds (only used for baseline + final fit).",
    )
    parser.add_argument(
        "--disable_gpu",
        action="store_true",
        help="Disable GPU even if available (forces CPU hist).",
    )
    return parser.parse_args()


def load_and_clean_data(path: str, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)

    # Drop irrelevant columns (as requested). Note: we keep player_name/season
    # in the dataframe temporarily so we can do leakage-safe splits if needed.
    drop_cols = [
        "team",
        "most_common_injury",
        "market_value_source",
    ]

    # Drop direct leakage from the target if present.
    # If target is log_market_value, market_value_eur is effectively exp(target).
    if target == "log_market_value" and "market_value_eur" in df.columns:
        drop_cols.append("market_value_eur")

    # Target-derived categorical tiers can also leak.
    if "market_value_tier_encoded" in df.columns:
        drop_cols.append("market_value_tier_encoded")

    # Drop obvious leakage / non-feature columns if present (this CSV contains prior model predictions).
    leakage_like = [
        "univariate_predicted_market_value",
        "multivarite_predicted_market_value",
        "Encoder-decoder_predicted",
        "error_eur",
    ]
    for c in leakage_like:
        if c in df.columns:
            drop_cols.append(c)

    # Keep only columns that actually exist.
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")

    # Basic missing value handling:
    # - We will still use imputers later, but we remove rows with missing target.
    df = df.dropna(subset=[target]).reset_index(drop=True)

    y = df[target].astype(float)
    X = df.drop(columns=[target])
    return X, y


def split_train_val_test(
    X: pd.DataFrame, y: pd.Series, split_strategy: str
) -> SplitData:
    """
    Create 70/15/15 splits.

    - random: row-wise split (can leak player identity across seasons)
    - group_player: group-wise split by player_name (recommended for panel data)
    - time_season: earlier seasons train, later seasons test (requires season_encoded)
    """
    if split_strategy == "random":
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=0.30, random_state=RANDOM_STATE
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE
        )
        return SplitData(X_train, X_val, X_test, y_train, y_val, y_test)

    if split_strategy == "group_player":
        # Common cases:
        # - 'player_name' column exists
        # - CSV was saved with an unnamed index column holding player names -> 'Unnamed: 0'
        if "player_name" in X.columns:
            id_col = "player_name"
        elif "Unnamed: 0" in X.columns:
            id_col = "Unnamed: 0"
        else:
            raise ValueError(
                "split_strategy=group_player requires a player identifier column "
                "('player_name' or 'Unnamed: 0') in the CSV."
            )

        groups = X[id_col].astype(str).to_numpy()
        idx = np.arange(len(X))

        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
        train_idx, temp_idx = next(gss1.split(idx, y, groups=groups))

        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=RANDOM_STATE)
        val_rel_idx, test_rel_idx = next(
            gss2.split(temp_idx, y.iloc[temp_idx], groups=groups[temp_idx])
        )
        val_idx = temp_idx[val_rel_idx]
        test_idx = temp_idx[test_rel_idx]

        return SplitData(
            X_train=X.iloc[train_idx].reset_index(drop=True),
            X_val=X.iloc[val_idx].reset_index(drop=True),
            X_test=X.iloc[test_idx].reset_index(drop=True),
            y_train=y.iloc[train_idx].reset_index(drop=True),
            y_val=y.iloc[val_idx].reset_index(drop=True),
            y_test=y.iloc[test_idx].reset_index(drop=True),
        )

    if split_strategy == "time_season":
        if "season_encoded" not in X.columns:
            raise ValueError("split_strategy=time_season requires 'season_encoded' column.")

        df_idx = pd.DataFrame({"idx": np.arange(len(X)), "season_encoded": X["season_encoded"]})
        seasons = sorted(df_idx["season_encoded"].dropna().unique().tolist())
        if len(seasons) < 3:
            raise ValueError(f"Not enough unique seasons for time split. Found: {seasons}")

        # Forward-looking split for "predict next season":
        # - test = latest season
        # - val = second-latest season
        # - train = all earlier seasons
        test_season = seasons[-1]
        val_season = seasons[-2]
        train_seasons = set(seasons[:-2])

        if not train_seasons:
            # If we only have 3 seasons, fall back to:
            # train = earliest, val = middle, test = latest
            train_seasons = {seasons[0]}
            val_season = seasons[1]
            test_season = seasons[2]

        train_idx = df_idx.loc[df_idx["season_encoded"].isin(train_seasons), "idx"].to_numpy()
        val_idx = df_idx.loc[df_idx["season_encoded"] == val_season, "idx"].to_numpy()
        test_idx = df_idx.loc[df_idx["season_encoded"] == test_season, "idx"].to_numpy()

        return SplitData(
            X_train=X.iloc[train_idx].reset_index(drop=True),
            X_val=X.iloc[val_idx].reset_index(drop=True),
            X_test=X.iloc[test_idx].reset_index(drop=True),
            y_train=y.iloc[train_idx].reset_index(drop=True),
            y_val=y.iloc[val_idx].reset_index(drop=True),
            y_test=y.iloc[test_idx].reset_index(drop=True),
        )

    raise ValueError(f"Unknown split_strategy: {split_strategy}")


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preprocessor


def get_xgb_gpu_params(disable_gpu: bool) -> Dict:
    """
    GPU params for modern XGBoost (2.x/3.x).

    In XGBoost >= 2.0, the recommended way is:
      tree_method="hist", device="cuda"
    (gpu_hist is deprecated in newer versions.)
    """
    if disable_gpu:
        return {"tree_method": "hist", "device": "cpu"}
    return {"tree_method": "hist", "device": "cuda"}


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    # Compatibility: older scikit-learn versions don't support squared=False.
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": float(mae), "r2": float(r2)}


def print_metrics(title: str, metrics: Dict[str, float]) -> None:
    print(f"\n{title}")
    print(f"  RMSE: {metrics['rmse']:.5f}")
    print(f"  MAE : {metrics['mae']:.5f}")
    print(f"  R^2 : {metrics['r2']:.5f}")


def ensure_output_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def plot_pred_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, title: str) -> None:
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.35, s=18)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi], linestyle="--")

    plt.xlabel("Actual log_market_value")
    plt.ylabel("Predicted log_market_value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_feature_importance(
    booster_or_model: Union[xgb.Booster, XGBRegressor],
    out_path: str,
    max_num_features: int = 25,
) -> None:
    booster = (
        booster_or_model
        if isinstance(booster_or_model, xgb.Booster)
        else booster_or_model.get_booster()
    )

    # Gain-based importance is usually the most interpretable for beginners.
    scores = booster.get_score(importance_type="gain")
    if not scores:
        print("\nFeature importance is empty (this can happen if the model couldn't split).")
        return

    # XGBoost uses feature names like "f0", "f1", ... if none are provided.
    items = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:max_num_features]
    labels = [k for k, _ in items][::-1]
    values = [v for _, v in items][::-1]

    plt.figure(figsize=(10, 7))
    plt.barh(labels, values)
    plt.title(f"Top {len(items)} Feature Importances (gain)")
    plt.xlabel("Gain")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def train_booster_with_early_stopping(
    X_train,
    y_train: np.ndarray,
    X_eval,
    y_eval: np.ndarray,
    params: Dict,
    num_boost_round: int,
    early_stopping_rounds: int,
    verbose_eval: int = 50,
) -> xgb.Booster:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    deval = xgb.DMatrix(X_eval, label=y_eval)

    # Use RMSE for progress logs + early stopping.
    full_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        **params,
    }

    booster = xgb.train(
        params=full_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=[(dtrain, "train"), (deval, "eval")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=verbose_eval,
    )
    return booster


def drop_split_only_columns(X: pd.DataFrame) -> pd.DataFrame:
    # These columns should not be used as model inputs.
    cols_to_drop = [c for c in ["player_name", "Unnamed: 0", "season"] if c in X.columns]
    if cols_to_drop:
        return X.drop(columns=cols_to_drop)
    return X


def save_pickle(obj, path: str) -> None:
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def get_player_id_series(X: pd.DataFrame) -> pd.Series:
    """
    Return a player identifier column if present.
    Supports common CSV exports where the first unnamed column holds player names.
    """
    if "player_name" in X.columns:
        return X["player_name"].astype(str)
    if "Unnamed: 0" in X.columns:
        return X["Unnamed: 0"].astype(str)
    raise ValueError("No player identifier column found ('player_name' or 'Unnamed: 0').")


def make_forward_chaining_cv_by_season(X_train_raw: pd.DataFrame):
    """
    Build forward-chaining CV splits using season_encoded within the training set.

    Example with seasons [1,2,3,4] present in training:
      fold1: train [1] -> val [2]
      fold2: train [1,2] -> val [3]
      fold3: train [1,2,3] -> val [4]
    """
    if "season_encoded" not in X_train_raw.columns:
        raise ValueError("Forward-chaining CV requires season_encoded in training data.")

    seasons = sorted(pd.Series(X_train_raw["season_encoded"]).dropna().unique().tolist())
    if len(seasons) < 3:
        # Too few seasons for meaningful forward-chaining CV; fall back to 3-fold CV by caller.
        return None

    season_arr = X_train_raw["season_encoded"].to_numpy()
    indices = np.arange(len(X_train_raw))
    splits = []
    for i in range(1, len(seasons)):
        train_seasons = set(seasons[:i])
        val_season = seasons[i]
        tr_idx = indices[np.isin(season_arr, list(train_seasons))]
        va_idx = indices[season_arr == val_season]
        if len(tr_idx) > 0 and len(va_idx) > 0:
            splits.append((tr_idx, va_idx))
    return splits if splits else None


def main() -> None:
    args = parse_args()

    print("Loading data...")
    X, y = load_and_clean_data(args.data_path, args.target)
    print(f"Rows: {len(X)} | Columns (features): {X.shape[1]}")

    splits = split_train_val_test(X, y, split_strategy=args.split_strategy)
    print(
        f"Split sizes -> train: {len(splits.X_train)}, val: {len(splits.X_val)}, test: {len(splits.X_test)}"
    )
    print(f"Split strategy: {args.split_strategy}")

    # Prepare groups for group-aware CV (prevents player leakage inside CV folds).
    train_groups = None
    if args.split_strategy == "group_player":
        train_groups = get_player_id_series(splits.X_train).to_numpy()
        print(f"Unique players in train: {len(np.unique(train_groups))}")

    # Drop columns used only for splitting (prevents leakage).
    splits = SplitData(
        X_train=drop_split_only_columns(splits.X_train),
        X_val=drop_split_only_columns(splits.X_val),
        X_test=drop_split_only_columns(splits.X_test),
        y_train=splits.y_train,
        y_val=splits.y_val,
        y_test=splits.y_test,
    )

    print("\nBuilding preprocessor (impute + one-hot)...")
    preprocessor = build_preprocessor(splits.X_train)

    # Fit on training only, then transform all splits.
    X_train_p = preprocessor.fit_transform(splits.X_train)
    X_val_p = preprocessor.transform(splits.X_val)
    X_test_p = preprocessor.transform(splits.X_test)

    gpu_params = get_xgb_gpu_params(args.disable_gpu)
    print("\nXGBoost GPU/CPU parameters:")
    for k, v in gpu_params.items():
        print(f"  {k} = {v}")
    print(f"XGBoost version: {xgb.__version__}")

    # -------------------------
    # 5) Baseline model
    # -------------------------
    print("\nTraining baseline model (native xgboost.train with early stopping)...")
    baseline_booster = train_booster_with_early_stopping(
        X_train=X_train_p,
        y_train=splits.y_train.to_numpy(),
        X_eval=X_val_p,
        y_eval=splits.y_val.to_numpy(),
        params={
            **gpu_params,
            "eta": 0.05,  # learning_rate
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "lambda": 1.0,
            "seed": RANDOM_STATE,
        },
        num_boost_round=2000,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=50,
    )

    print("\nBaseline booster params (sanity check):")
    print(baseline_booster.save_config()[:700] + "...\n")

    dval = xgb.DMatrix(X_val_p)
    best_it = getattr(baseline_booster, "best_iteration", None)
    if best_it is None:
        val_pred = baseline_booster.predict(dval)
    else:
        val_pred = baseline_booster.predict(dval, iteration_range=(0, best_it + 1))
    val_metrics = evaluate(splits.y_val.to_numpy(), val_pred)
    print_metrics("Baseline (Validation)", val_metrics)

    # -------------------------
    # 6) Hyperparameter tuning
    # -------------------------
    print("\nHyperparameter tuning with RandomizedSearchCV...")
    param_distributions = {
        # Regularized / lower-capacity search space to reduce overfitting,
        # especially important under player-group splits.
        "n_estimators": [300, 500, 800, 1200, 1600],
        "max_depth": [2, 3, 4, 5, 6],
        "learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "gamma": [0.0, 0.05, 0.1, 0.2, 0.5],
        "min_child_weight": [1, 3, 5, 10, 20],
        "reg_alpha": [0.0, 0.01, 0.05, 0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0, 10.0],
    }

    search_estimator = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
        **gpu_params,
    )

    # We tune on training set with CV for a fair/fast search.
    # - group_player: GroupKFold by player id
    # - time_season: forward-chaining splits by season (no lookahead)
    if args.split_strategy == "time_season":
        cv = make_forward_chaining_cv_by_season(splits.X_train)
        if cv is None:
            cv = args.cv
    elif train_groups is not None:
        cv = GroupKFold(n_splits=args.cv)
    else:
        cv = args.cv

    search = RandomizedSearchCV(
        estimator=search_estimator,
        param_distributions=param_distributions,
        n_iter=args.n_iter,
        # Compatibility with older sklearn: use neg MSE, then sqrt for RMSE when reporting.
        scoring="neg_mean_squared_error",
        cv=cv,
        verbose=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    if train_groups is not None:
        search.fit(X_train_p, splits.y_train.to_numpy(), groups=train_groups)
    else:
        search.fit(X_train_p, splits.y_train.to_numpy())

    print("\nBest parameters found:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")
    best_cv_rmse = float(np.sqrt(-search.best_score_))
    print(f"Best CV RMSE: {best_cv_rmse:.5f}")

    # -------------------------
    # 7) Final model
    # -------------------------
    print("\nTraining final model on train+val (native xgboost.train with early stopping)...")
    # Combine train + val as requested.
    X_trainval_p = preprocessor.transform(pd.concat([splits.X_train, splits.X_val], axis=0))
    y_trainval = pd.concat([splits.y_train, splits.y_val], axis=0).to_numpy()

    # Optional early stopping without leaking the test set:
    # create a small internal eval set from train+val.
    X_tr2, X_es, y_tr2, y_es = train_test_split(
        X_trainval_p,
        y_trainval,
        test_size=0.10,
        random_state=RANDOM_STATE,
    )

    best_params = dict(search.best_params_)

    # Map sklearn-style params to xgboost.train params.
    final_params = {
        **gpu_params,
        "seed": RANDOM_STATE,
        "max_depth": int(best_params.get("max_depth", 6)),
        "eta": float(best_params.get("learning_rate", 0.05)),
        "subsample": float(best_params.get("subsample", 0.9)),
        "colsample_bytree": float(best_params.get("colsample_bytree", 0.9)),
        "gamma": float(best_params.get("gamma", 0.0)),
        "min_child_weight": float(best_params.get("min_child_weight", 1.0)),
        "alpha": float(best_params.get("reg_alpha", 0.0)),
        "lambda": float(best_params.get("reg_lambda", 1.0)),
    }
    num_boost_round = int(best_params.get("n_estimators", 1200))

    final_booster = train_booster_with_early_stopping(
        X_train=X_tr2,
        y_train=y_tr2,
        X_eval=X_es,
        y_eval=y_es,
        params=final_params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
        verbose_eval=50,
    )

    dtest = xgb.DMatrix(X_test_p)
    best_it = getattr(final_booster, "best_iteration", None)
    if best_it is None:
        test_pred = final_booster.predict(dtest)
    else:
        test_pred = final_booster.predict(dtest, iteration_range=(0, best_it + 1))
    test_metrics = evaluate(splits.y_test.to_numpy(), test_pred)
    print_metrics("Final Model (Test)", test_metrics)

    # Helpful overfitting check: compare train vs test on the final booster.
    dtrain_full = xgb.DMatrix(
        preprocessor.transform(pd.concat([splits.X_train, splits.X_val], axis=0))
    )
    ytrain_full = pd.concat([splits.y_train, splits.y_val], axis=0).to_numpy()
    best_it = getattr(final_booster, "best_iteration", None)
    if best_it is None:
        train_pred = final_booster.predict(dtrain_full)
    else:
        train_pred = final_booster.predict(dtrain_full, iteration_range=(0, best_it + 1))
    train_metrics = evaluate(ytrain_full, train_pred)
    print_metrics("Final Model (Train+Val)", train_metrics)
    print(
        f"\nOverfit gap (RMSE): train+val {train_metrics['rmse']:.5f} vs test {test_metrics['rmse']:.5f} "
        f"(gap {test_metrics['rmse'] - train_metrics['rmse']:+.5f})"
    )

    # -------------------------
    # 9) Visualization
    # -------------------------
    ensure_output_dir(args.output_dir)

    fi_path = os.path.join(args.output_dir, "feature_importance_gain.png")
    pva_path = os.path.join(args.output_dir, "pred_vs_actual_test.png")

    print("\nSaving plots...")
    plot_feature_importance(final_booster, fi_path)
    plot_pred_vs_actual(
        splits.y_test.to_numpy(),
        test_pred,
        pva_path,
        title="Predicted vs Actual (Test Set)",
    )

    # -------------------------
    # Save model + preprocessor
    # -------------------------
    print("\nSaving fine-tuned model + preprocessor...")
    ensure_output_dir(os.path.dirname(args.model_out) or ".")
    final_booster.save_model(args.model_out)
    save_pickle(preprocessor, args.preprocessor_out)
    print(f"Saved model: {args.model_out}")
    print(f"Saved preprocessor: {args.preprocessor_out}")

    print("\nDone.")
    print(f"Saved: {fi_path}")
    print(f"Saved: {pva_path}")


if __name__ == "__main__":
    main()

