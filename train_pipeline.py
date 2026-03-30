import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.data_pipeline import generate_mock_data, clean_and_feature_engineer
from utils.models_dev import (
    train_linear_regression, train_random_forest, train_xgboost,
    train_lstm, evaluate_model, predict_lstm, predict_ensemble
)
from utils.plotter import (
    plot_actual_vs_predicted, plot_model_comparison,
    plot_feature_importance, plot_performance_trends
)

def run_pipeline():
    print("1. DATA PIPELINE")
    df_main_raw, df_seq_raw = generate_mock_data()
    df, df_seq = clean_and_feature_engineer(df_main_raw, df_seq_raw)
    
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/processed_players.csv", index=False)
    df_seq.to_csv("data/processed_sequences.csv", index=False)
    
    # Target value map for LSTM
    target_map = dict(zip(df['player_id'], df['market_value']))
    
    # Feature columns for baseline and XGBoost (drop ID, name, etc.)
    drop_cols = ['player_id']
    if 'name' in df.columns: drop_cols.append('name')
    X_df = df.drop(columns=drop_cols + ['market_value'])
    y_ser = df['market_value']
    
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_ser, test_size=0.2, random_state=42)
    
    print("2. MODEL DEVELOPMENT")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Fit baseline models
    lr_model = train_linear_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    # Save scikit/xgboost models
    joblib.dump(lr_model, os.path.join(models_dir, 'linear_regression.pkl'))
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest.pkl'))
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgboost.pkl'))
    
    # Save feature columns to ensure API knows what to pass
    with open(os.path.join(models_dir, 'features.json'), 'w') as f:
        json.dump(list(X_train.columns), f)
    
    # Fit LSTM Model
    # We will just use the sequential features
    seq_features = ['goals', 'assists', 'minutes']
    lstm_model, X_tensor_full, y_tensor_full = train_lstm(df_seq, target_map, seq_features)
    import torch
    torch.save(lstm_model.state_dict(), os.path.join(models_dir, 'lstm_model.pth'))
    
    print("3. MODEL EVALUATION")
    results = []
    
    y_pred_lr = lr_model.predict(X_test)
    results.append(evaluate_model(y_test, y_pred_lr, "Linear Regression"))
    
    y_pred_rf = rf_model.predict(X_test)
    results.append(evaluate_model(y_test, y_pred_rf, "Random Forest"))
    
    y_pred_xgb = xgb_model.predict(X_test)
    results.append(evaluate_model(y_test, y_pred_xgb, "XGBoost"))
    
    # Predict on the full dataset with LSTM just for metrics over whatever IDs were successfully matched
    # For actual train/test split of sequences, it's more complex, but we'll approximate here
    lstm_preds = predict_lstm(lstm_model, X_tensor_full)
    results.append(evaluate_model(y_tensor_full.numpy(), lstm_preds, "LSTM"))
    
    # Ensemble (align test set sizes to evaluate)
    df_test_idx = X_test.index
    df_test_full = df.loc[df_test_idx]
    
    # Gather LSTM preds for the specific test IDs
    test_player_ids = df_test_full['player_id'].values
    xgb_preds_test = xgb_model.predict(X_test)
    
    lstm_preds_test = []
    y_test_aligned = []
    for pid, pred_xgb, true_y in zip(test_player_ids, xgb_preds_test, y_test):
        # find pid index in lstm tensor
        pid_seq_idx = df_seq['player_id'].unique().tolist().index(pid)
        lstm_preds_test.append(lstm_preds[pid_seq_idx])
        y_test_aligned.append(true_y)
        
    lstm_preds_test = np.array(lstm_preds_test)
    y_test_aligned = np.array(y_test_aligned)
    
    ensemble_preds = predict_ensemble(xgb_preds_test, lstm_preds_test)
    results.append(evaluate_model(y_test_aligned, ensemble_preds, "Ensemble"))
    
    results_df = pd.DataFrame(results)
    print("\nModel Results:")
    print(results_df.to_string(index=False))
    
    print("4. VISUALIZATION")
    plots_dir = os.path.join("outputs", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_actual_vs_predicted(y_test_aligned, ensemble_preds, "Ensemble", plots_dir)
    plot_model_comparison(results_df, 'RMSE', plots_dir)
    plot_model_comparison(results_df, 'R2 Score', plots_dir)
    plot_feature_importance(xgb_model, X_train.columns, plots_dir)
    
    # Plot performance trend for the first test player
    plot_performance_trends(df_seq, test_player_ids[0], plots_dir)
    
    print("Pipeline Complete. Models and plots saved.")

if __name__ == "__main__":
    run_pipeline()
