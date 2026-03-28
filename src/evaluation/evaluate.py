import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Adjust path for execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.models.lstm_models import MultivariateLSTM
from src.models.ensemble_model import EnsembleModel

def evaluate():
    OUTPUTS_DIR = "outputs"
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Load dataset artifacts
    print("Loading test data and models")
    X_test = np.load("models/X_test.npy")
    y_test = np.load("models/y_test.npy")
    
    with open("models/features.json", "r") as f:
        features = json.load(f)
        
    # Load Multivariate LSTM
    input_size = len(features)
    mv_lstm = MultivariateLSTM(input_size=input_size)
    mv_lstm.load_state_dict(torch.load("models/multivariate_lstm.pth"))
    mv_lstm.eval()
    
    # Load XGBoost inside the ensemble class
    ensemble = EnsembleModel(lstm_weight=0.6, xgb_weight=0.4)
    ensemble.load_xgb("models/xgboost_model.pkl")

    # Predict
    with torch.no_grad():
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        lstm_preds = mv_lstm(X_test_tensor).squeeze().numpy()
        
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    xgb_preds = ensemble.predict_xgb(X_test_flat)
    
    final_preds = ensemble.predict_ensemble(lstm_preds, xgb_preds)

    # Calculate metrics
    def calculate_metrics(y_true, y_pred, name):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        return {"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2}

    results = []
    results.append(calculate_metrics(y_test, lstm_preds, "Multivariate LSTM"))
    results.append(calculate_metrics(y_test, xgb_preds, "XGBoost"))
    results.append(calculate_metrics(y_test, final_preds, "Ensemble (LSTM + XGB)"))
    
    metrics_df = pd.DataFrame(results)
    metrics_df.to_markdown(f"{OUTPUTS_DIR}/evaluation_report.md", index=False)
    print("Metrics evaluated.")
    
    # Visualizations
    # 1. Prediction vs Actual Trend
    plt.figure(figsize=(10,6))
    plt.plot(y_test[:50], label="Actual Market Value (Scaled)", marker='o')
    plt.plot(final_preds[:50], label="Predicted Value (Ensemble)", marker='x')
    plt.title("Player Value Trend (Past vs Predicted)")
    plt.legend()
    plt.savefig(f"{OUTPUTS_DIR}/prediction_trend.png")
    plt.close()

    # 2. Model Comparison Bar Chart
    plt.figure(figsize=(8,5))
    sns.barplot(x="Model", y="RMSE", data=metrics_df)
    plt.title("Model Comparison (RMSE Lower is Better)")
    plt.savefig(f"{OUTPUTS_DIR}/model_comparison.png")
    plt.close()
    
    # 3. Sentiment vs Value Correlation (Simulation based on features)
    # Let's say Vader Compound Score was a feature. Which index is it?
    if 'vader_compound_score' in features:
        idx = features.index('vader_compound_score')
        # We can extract the vader score from the last timestep sequentially
        vader_scores = X_test[:, -1, idx]
        
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=vader_scores, y=y_test)
        plt.title("Sentiment vs Market Value (Scaled)")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Market Value")
        plt.savefig(f"{OUTPUTS_DIR}/sentiment_correlation.png")
        plt.close()
        
    # 4. SHAP Feature Importance
    try:
        explainer = shap.Explainer(ensemble.xgb_model)
        # sample smaller for shap purely for speed
        shap_values = explainer(X_test_flat[:100])
        feature_names = [f"t{t}_{f}" for t in range(3) for f in features]
        shap.summary_plot(shap_values, X_test_flat[:100], feature_names=feature_names, show=False)
        plt.savefig(f"{OUTPUTS_DIR}/shap_summary.png", bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"SHAP Error: {e}")

    print("Evaluations completed and outputted successfully.")

if __name__ == "__main__":
    evaluate()
