import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt

def compare_all_models(
    dataset_path='models/stacking_dataset.csv',
    xgboost_model_path='models/saved_models/xgboost_stacking_model.pkl',
    scaler_path='models/saved_models/feature_scaler.pkl',
    feature_columns_path='models/saved_models/feature_columns.pkl'
):
    """
    Compare all models: LSTM variants vs XGBoost Stacking
    """
    print("=" * 60)
    print("Comparing All Models")
    print("=" * 60)
    
    # Load data
    df = pd.read_csv(dataset_path)
    
    # Load XGBoost model and artifacts
    xgb_model = joblib.load(xgboost_model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = joblib.load(feature_columns_path)
    
    # Prepare XGBoost features
    X = df[feature_columns].copy()
    X = X.fillna(0).replace([np.inf, -np.inf], 0)
    X_scaled = scaler.transform(X)
    
    # Get XGBoost predictions
    xgb_predictions = xgb_model.predict(X_scaled)
    
    # Get actual values
    actual = df['market_value_eur'].values
    
    # Get LSTM predictions
    univariate_pred = df['univariate_prediction'].values
    multivariate_pred = df['multivariate_prediction'].values
    encoder_decoder_pred = df['encoder_decoder_prediction'].values
    
    # Create ensemble prediction (simple average)
    ensemble_simple = (univariate_pred + multivariate_pred + encoder_decoder_pred) / 3
    
    # Calculate metrics for all models
    models = {
        'Univariate LSTM': univariate_pred,
        'Multivariate LSTM': multivariate_pred,
        'Encoder-Decoder LSTM': encoder_decoder_pred,
        'Simple Ensemble (Avg)': ensemble_simple,
        'XGBoost Stacking': xgb_predictions
    }
    
    results = []
    
    print("\n" + "=" * 60)
    print("Model Performance Comparison:")
    print("=" * 60)
    
    for model_name, predictions in models.items():
        rmse = np.sqrt(mean_squared_error(actual, predictions))
        mae = mean_absolute_error(actual, predictions)
        r2 = r2_score(actual, predictions)
        
        # Calculate improvement over Univariate
        if model_name != 'Univariate LSTM':
            improvement = ((rmse - models['Univariate LSTM'].rmse) / models['Univariate LSTM'].rmse) * 100
        else:
            improvement = 0
        
        results.append({
            'Model': model_name,
            'RMSE (€)': rmse,
            'MAE (€)': mae,
            'R² Score': r2,
            'Improvement %': improvement
        })
        
        print(f"\n{model_name}:")
        print(f"  RMSE: €{rmse:,.0f}")
        print(f"  MAE: €{mae:,.0f}")
        print(f"  R²: {r2:.4f}")
        if improvement != 0:
            print(f"  Improvement vs Univariate: {improvement:+.1f}%")
    
    # Create comparison dataframe
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('models/model_comparison.csv', index=False)
    print("\n✓ Results saved to: models/model_comparison.csv")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. RMSE Comparison
    ax1 = axes[0, 0]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    bars = ax1.bar(results_df['Model'], results_df['RMSE (€)'] / 1e6, color=colors)
    ax1.set_ylabel('RMSE (Million €)')
    ax1.set_title('RMSE Comparison')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, val in zip(bars, results_df['RMSE (€)'] / 1e6):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.1f}M', ha='center', va='bottom')
    
    # 2. R² Score Comparison
    ax2 = axes[0, 1]
    bars = ax2.bar(results_df['Model'], results_df['R² Score'], color=colors)
    ax2.set_ylabel('R² Score')
    ax2.set_title('R² Score Comparison')
    ax2.tick_params(axis='x', rotation=45)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar, val in zip(bars, results_df['R² Score']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{val:.3f}', ha='center', va='bottom')
    
    # 3. MAE Comparison
    ax3 = axes[1, 0]
    bars = ax3.bar(results_df['Model'], results_df['MAE (€)'] / 1e6, color=colors)
    ax3.set_ylabel('MAE (Million €)')
    ax3.set_title('MAE Comparison')
    ax3.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars, results_df['MAE (€)'] / 1e6):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                f'{val:.1f}M', ha='center', va='bottom')
    
    # 4. Improvement Chart
    ax4 = axes[1, 1]
    improvement_data = results_df[results_df['Improvement %'] != 0]
    bars = ax4.bar(improvement_data['Model'], improvement_data['Improvement %'], 
                   color=['#2ca02c' if x > 0 else '#d62728' for x in improvement_data['Improvement %']])
    ax4.set_ylabel('Improvement over Univariate (%)')
    ax4.set_title('Improvement Percentage')
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    for bar, val in zip(bars, improvement_data['Improvement %']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{val:+.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('models/model_comparison.png', dpi=100, bbox_inches='tight')
    print("✓ Comparison chart saved to: models/model_comparison.png")
    
    plt.show()
    
    return results_df

if __name__ == "__main__":
    compare_all_models()