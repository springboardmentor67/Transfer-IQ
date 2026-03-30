"""
Key-aligned ensemble regression for log_market_value:
  1) Fine-tuned XGBoost predictions
  2) Multivariate LSTM predictions

This version expects the aligned LSTM file produced by train_multivariate_lstm.py
with columns:
  actual_log_market_value, lstm_predictions, split, player_id, target_season, target_season_encoded
"""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Strictly aligned ensemble: XGBoost + LSTM.")
    p.add_argument("--data_path", type=str, default="player_transfer_value_with_sentiment.csv")
    p.add_argument("--target", type=str, default="log_market_value")
    p.add_argument(
        "--lstm_preds_path",
        type=str,
        default="outputs/lstm/lstm_predictions_aligned.csv",
        help="Aligned LSTM predictions CSV with split/player/season keys.",
    )
    p.add_argument("--xgb_model_path", type=str, default="outputs/final_xgb_model.json")
    p.add_argument("--preprocessor_path", type=str, default="outputs/preprocessor.pkl")
    p.add_argument("--output_dir", type=str, default="outputs/ensemble")
    p.add_argument("--weight_xgb", type=float, default=0.6)
    p.add_argument("--save_predictions_csv", action="store_true")
    return p.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(y_true, y_pred)
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def print_metrics(title: str, m: Dict[str, float]) -> None:
    print(f"\n{title}")
    print(f"  RMSE: {m['rmse']:.5f}")
    print(f"  MAE : {m['mae']:.5f}")
    print(f"  R^2 : {m['r2']:.5f}")


def get_player_col(df: pd.DataFrame) -> str:
    if "player_name" in df.columns:
        return "player_name"
    if "Unnamed: 0" in df.columns:
        return "Unnamed: 0"
    raise ValueError("No player identifier column found in dataset ('player_name' or 'Unnamed: 0').")


def clean_for_xgb_features(df: pd.DataFrame, target: str) -> Tuple[pd.DataFrame, pd.Series]:
    drop_cols = ["team", "most_common_injury", "market_value_source"]
    if target == "log_market_value" and "market_value_eur" in df.columns:
        drop_cols.append("market_value_eur")
    if "market_value_tier_encoded" in df.columns:
        drop_cols.append("market_value_tier_encoded")
    for c in [
        "univariate_predicted_market_value",
        "multivarite_predicted_market_value",
        "Encoder-decoder_predicted",
        "error_eur",
    ]:
        if c in df.columns:
            drop_cols.append(c)
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore").copy()

    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found.")
    df = df.dropna(subset=[target]).reset_index(drop=True)
    y = df[target].astype(float)
    X = df.drop(columns=[target])
    for c in ["player_name", "Unnamed: 0", "season"]:
        if c in X.columns:
            X = X.drop(columns=[c])
    return X, y


def build_next_season_rows(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Build one row per (player, target season) where target season is predicted
    from previous seasons. This mirrors the LSTM sequence target construction.
    """
    player_col = get_player_col(df)
    df = df.dropna(subset=[target]).copy()
    df = df.sort_values([player_col, "season_encoded"]).reset_index(drop=True)

    out_rows = []
    for pid, g in df.groupby(player_col, sort=False):
        if len(g) < 2:
            continue
        # Each row from index 1 onward is a target season in sequence modeling.
        tgt = g.iloc[1:].copy()
        tgt["player_id"] = str(pid)
        out_rows.append(tgt)

    if not out_rows:
        raise ValueError("No next-season rows were built. Check data.")
    return pd.concat(out_rows, axis=0, ignore_index=True)


def load_aligned_lstm(path: str) -> pd.DataFrame:
    lstm = pd.read_csv(path)
    required = {
        "actual_log_market_value",
        "lstm_predictions",
        "split",
        "player_id",
        "target_season_encoded",
    }
    missing = [c for c in required if c not in lstm.columns]
    if missing:
        raise ValueError(f"LSTM aligned CSV missing required columns: {missing}")
    lstm["player_id"] = lstm["player_id"].astype(str)
    lstm["target_season_encoded"] = lstm["target_season_encoded"].astype(float)
    return lstm


def load_preprocessor(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_xgb_booster(path: str) -> xgb.Booster:
    b = xgb.Booster()
    b.load_model(path)
    return b


def xgb_predict(booster: xgb.Booster, preprocessor, X: pd.DataFrame) -> np.ndarray:
    Xp = preprocessor.transform(X)
    dmat = xgb.DMatrix(Xp)
    best_it = getattr(booster, "best_iteration", None)
    if best_it is None:
        return booster.predict(dmat)
    return booster.predict(dmat, iteration_range=(0, best_it + 1))


def plot_actual_vs_pred(y_true: np.ndarray, y_pred: np.ndarray, out_path: str, title: str) -> None:
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


def plot_rmse_bar(metrics: Dict[str, Dict[str, float]], out_path: str, title: str) -> None:
    names = list(metrics.keys())
    vals = [metrics[n]["rmse"] for n in names]
    plt.figure(figsize=(10, 5))
    plt.bar(names, vals)
    plt.ylabel("RMSE")
    plt.title(title)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def evaluate_split(df: pd.DataFrame, weight_xgb: float, split_name: str) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    y = df["actual_log_market_value"].to_numpy()
    y_xgb = df["y_xgb"].to_numpy()
    y_lstm = df["lstm_predictions"].to_numpy()

    y_avg = (y_xgb + y_lstm) / 2.0
    y_wavg = weight_xgb * y_xgb + (1.0 - weight_xgb) * y_lstm

    metrics = {
        "XGBoost": eval_metrics(y, y_xgb),
        "LSTM": eval_metrics(y, y_lstm),
        "AvgEnsemble": eval_metrics(y, y_avg),
        f"WAvg(w_xgb={weight_xgb:.2f})": eval_metrics(y, y_wavg),
    }

    out = df.copy()
    out["y_avg"] = y_avg
    out["y_wavg"] = y_wavg

    print(f"\n===== {split_name.upper()} =====")
    for name, m in metrics.items():
        print_metrics(name, m)
    return metrics, out


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)

    print("Loading dataset, LSTM aligned predictions, and XGBoost artifacts...")
    raw = pd.read_csv(args.data_path)
    lstm = load_aligned_lstm(args.lstm_preds_path)
    preprocessor = load_preprocessor(args.preprocessor_path)
    booster = load_xgb_booster(args.xgb_model_path)

    print("Building next-season target rows and XGBoost features...")
    next_rows = build_next_season_rows(raw, target=args.target)
    X_next, _ = clean_for_xgb_features(next_rows, target=args.target)
    next_rows = next_rows.reset_index(drop=True)

    # Build merge keys to align perfectly with LSTM file.
    next_rows["player_id"] = next_rows[get_player_col(next_rows)].astype(str)
    next_rows["target_season_encoded"] = next_rows["season_encoded"].astype(float)

    print("Generating XGBoost predictions for all next-season rows...")
    next_rows["y_xgb"] = xgb_predict(booster, preprocessor, X_next)

    # Keep only required columns for merge.
    xgb_keyed = next_rows[["player_id", "target_season_encoded", "y_xgb"]].copy()
    merged = lstm.merge(
        xgb_keyed,
        on=["player_id", "target_season_encoded"],
        how="inner",
        validate="one_to_one",
    )
    if merged.empty:
        raise ValueError("Merged dataframe is empty. Check player_id/season keys.")

    # Basic sanity checks.
    missing_pred = merged[["lstm_predictions", "y_xgb"]].isna().any().any()
    if missing_pred:
        raise ValueError("Found missing predictions after merge.")

    print(f"Merged rows: {len(merged)} (LSTM rows: {len(lstm)})")

    val_df = merged[merged["split"] == "val"].reset_index(drop=True)
    test_df = merged[merged["split"] == "test"].reset_index(drop=True)
    if val_df.empty or test_df.empty:
        raise ValueError("Expected both val and test splits in LSTM aligned CSV.")

    # Evaluate base and averaging methods.
    val_metrics, val_out = evaluate_split(val_df, args.weight_xgb, "val")
    test_metrics, test_out = evaluate_split(test_df, args.weight_xgb, "test")

    # Proper meta-model stacking: train on val, evaluate on test.
    X_meta_val = np.column_stack([val_out["y_xgb"].to_numpy(), val_out["lstm_predictions"].to_numpy()])
    y_meta_val = val_out["actual_log_market_value"].to_numpy()
    X_meta_test = np.column_stack([test_out["y_xgb"].to_numpy(), test_out["lstm_predictions"].to_numpy()])
    y_meta_test = test_out["actual_log_market_value"].to_numpy()

    meta = LinearRegression()
    meta.fit(X_meta_val, y_meta_val)
    test_out["y_meta"] = meta.predict(X_meta_test)
    meta_test_metrics = eval_metrics(y_meta_test, test_out["y_meta"].to_numpy())

    print_metrics("Meta(LinearReg) [trained on val, tested on test]", meta_test_metrics)

    # Add meta to test metrics dictionary for final ranking.
    test_metrics["Meta(LinearReg)"] = meta_test_metrics
    best_name = min(test_metrics.keys(), key=lambda k: test_metrics[k]["rmse"])
    print(f"\nBest model on TEST by RMSE: {best_name} ({test_metrics[best_name]['rmse']:.5f})")

    # Save the meta-model
    meta_path = os.path.join(args.output_dir, "meta_model.pkl")
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    print(f"\nSaved meta-model to: {meta_path}")

    # Save predictions for later use.
    if args.save_predictions_csv:
        out_path = os.path.join(args.output_dir, "ensemble_predictions_val_test.csv")
        out_all = pd.concat([val_out, test_out], axis=0, ignore_index=True)
        keep = [
            "split",
            "player_id",
            "target_season",
            "target_season_encoded",
            "actual_log_market_value",
            "lstm_predictions",
            "y_xgb",
            "y_avg",
            "y_wavg",
        ]
        if "y_meta" in out_all.columns:
            keep.append("y_meta")
        out_all[keep].to_csv(out_path, index=False)
        print(f"Saved ensemble predictions: {out_path}")

    # Plots (test split).
    plot_actual_vs_pred(
        test_out["actual_log_market_value"].to_numpy(),
        test_out["y_wavg"].to_numpy(),
        os.path.join(args.output_dir, "actual_vs_pred_weighted_test.png"),
        "Actual vs Predicted (Weighted Ensemble) [Test]",
    )
    plot_rmse_bar(
        test_metrics,
        os.path.join(args.output_dir, "model_comparison_rmse_test.png"),
        "Model Comparison on Test (RMSE)",
    )
    print(f"Saved plots to: {args.output_dir}")


if __name__ == "__main__":
    main()

