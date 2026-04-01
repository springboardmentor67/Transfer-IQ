"""
Full Model Evaluation Report
Generates comparison table and graphs similar to academic papers
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Set style for academic-looking graphs
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

print("=" * 80)
print("TRANSFERIQ - COMPLETE MODEL EVALUATION REPORT")
print("=" * 80)

# Create directories
os.makedirs('models/evaluation_reports', exist_ok=True)
os.makedirs('models/visualizations', exist_ok=True)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n[1] Loading data...")
df = pd.read_csv('models/stacking_dataset.csv')
print(f"    Dataset: {df.shape[0]} records, {df.shape[1]} features")
print(f"    Value range: €{df['market_value_eur'].min():,.0f} - €{df['market_value_eur'].max():,.0f}")

# Get actual values
y_actual = df['market_value_eur'].values

# ============================================
# 2. GET ALL MODEL PREDICTIONS
# ============================================
print("\n[2] Getting model predictions...")

# LSTM models
y_univariate = df['univariate_prediction'].values
y_multivariate = df['multivariate_prediction'].values
y_encoder = df['encoder_decoder_prediction'].values

# Ensemble 1: Simple Average of LSTM
y_ensemble_simple = (y_univariate + y_multivariate + y_encoder) / 3

# Try to load tuned XGBoost
y_xgb_tuned = None
try:
    xgb_path = 'models/hyperparameter_tuning/best_xgboost_model.pkl'
    if os.path.exists(xgb_path):
        xgb_model = joblib.load(xgb_path)
        feature_cols = ['univariate_prediction', 'multivariate_prediction', 'encoder_decoder_prediction',
                        'univariate_error', 'multivariate_error', 'encoder_decoder_error',
                        'market_value_lag1', 'market_value_lag2']
        available_cols = [col for col in feature_cols if col in df.columns]
        X = df[available_cols].fillna(0)
        y_xgb_tuned = xgb_model.predict(X)
        print("    ✓ Tuned XGBoost loaded")
    else:
        print("    ⚠ Tuned XGBoost not found")
except Exception as e:
    print(f"    ⚠ Could not load XGBoost: {e}")

# Ensemble 2: Weighted (60% XGB + 40% LSTM Avg)
if y_xgb_tuned is not None:
    y_ensemble_weighted = 0.6 * y_xgb_tuned + 0.4 * y_ensemble_simple
else:
    y_ensemble_weighted = y_ensemble_simple

# ============================================
# 3. CALCULATE METRICS
# ============================================
print("\n[3] Calculating metrics...")

def calculate_metrics(y_true, y_pred, model_name):
    """Calculate all evaluation metrics"""
    # Avoid division by zero for MAPE
    mask = y_true > 1000  # Ignore values less than €1000
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    mape = np.mean(np.abs((y_true_filtered - y_pred_filtered) / y_true_filtered)) * 100
    
    # RMSE in Millions (for cleaner display)
    rmse_m = rmse / 1_000_000
    
    return {
        'Model': model_name,
        'RMSE (€)': rmse,
        'RMSE (M€)': rmse_m,
        'MAE (€)': mae,
        'R² Score': r2,
        'MAPE (%)': mape
    }

# Collect all models
models = [
    ('Univariate LSTM', y_univariate),
    ('Multivariate LSTM', y_multivariate),
    ('Encoder-Decoder LSTM', y_encoder),
    ('Ensemble (Simple Avg)', y_ensemble_simple),
]

if y_xgb_tuned is not None:
    models.append(('XGBoost (Tuned)', y_xgb_tuned))
    models.append(('Ensemble (Weighted)', y_ensemble_weighted))

# Calculate metrics for each
results = []
for name, pred in models:
    metrics = calculate_metrics(y_actual, pred, name)
    results.append(metrics)

# Create DataFrame
results_df = pd.DataFrame(results)

# Format for display
display_df = results_df.copy()
display_df['RMSE (M€)'] = display_df['RMSE (M€)'].round(4)
display_df['MAE (€)'] = display_df['MAE (€)'].apply(lambda x: f"€{x:,.0f}")
display_df['RMSE (€)'] = display_df['RMSE (€)'].apply(lambda x: f"€{x:,.0f}")
display_df['R² Score'] = display_df['R² Score'].round(4)
display_df['MAPE (%)'] = display_df['MAPE (%)'].round(2)

# ============================================
# 4. PRINT RESULTS TABLE (Like your teammate's)
# ============================================
print("\n" + "=" * 80)
print("MODEL EVALUATION SUMMARY")
print("=" * 80)
print("\n" + display_df[['Model', 'RMSE (M€)', 'MAE (€)', 'R² Score', 'MAPE (%)']].to_string(index=False))

# ============================================
# 5. CREATE VISUALIZATIONS
# ============================================
print("\n[4] Creating visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('TransferIQ - Model Performance Comparison', fontsize=16, fontweight='bold')

# Color palette
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# 1. RMSE Comparison (in Millions)
ax1 = axes[0, 0]
models_names = results_df['Model'].tolist()
rmse_values = results_df['RMSE (M€)'].tolist()
bars = ax1.bar(models_names, rmse_values, color=colors[:len(models_names)])
ax1.set_ylabel('RMSE (Million €)', fontsize=11)
ax1.set_title('RMSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
ax1.tick_params(axis='x', rotation=45)
# Add value labels
for bar, val in zip(bars, rmse_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.3f}M', ha='center', va='bottom', fontsize=9)

# 2. R² Score Comparison
ax2 = axes[0, 1]
r2_values = results_df['R² Score'].tolist()
bars = ax2.bar(models_names, r2_values, color=colors[:len(models_names)])
ax2.set_ylabel('R² Score', fontsize=11)
ax2.set_title('R² Score Comparison (Higher is Better)', fontsize=12, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
ax2.axhline(y=0.99, color='green', linestyle='--', alpha=0.5, label='Target (0.99)')
for bar, val in zip(bars, r2_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)

# 3. MAPE Comparison
ax3 = axes[0, 2]
mape_values = results_df['MAPE (%)'].tolist()
bars = ax3.bar(models_names, mape_values, color=colors[:len(models_names)])
ax3.set_ylabel('MAPE (%)', fontsize=11)
ax3.set_title('MAPE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
ax3.tick_params(axis='x', rotation=45)
for bar, val in zip(bars, mape_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

# 4. Predictions vs Actual (Best Model)
ax4 = axes[1, 0]
best_model_idx = results_df['RMSE (M€)'].idxmin()
best_model_name = results_df.loc[best_model_idx, 'Model']
best_model_pred = models[best_model_idx][1] if best_model_idx < len(models) else y_ensemble_simple

ax4.scatter(y_actual / 1e6, best_model_pred / 1e6, alpha=0.3, s=10, color='#1f77b4')
ax4.plot([y_actual.min() / 1e6, y_actual.max() / 1e6], 
         [y_actual.min() / 1e6, y_actual.max() / 1e6], 
         'r--', linewidth=2, label='Perfect Prediction')
ax4.set_xlabel('Actual Value (Million €)', fontsize=11)
ax4.set_ylabel('Predicted Value (Million €)', fontsize=11)
ax4.set_title(f'{best_model_name} - Predictions vs Actual', fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Error Distribution
ax5 = axes[1, 1]
best_model_errors = y_actual - best_model_pred
ax5.hist(best_model_errors / 1e6, bins=50, color='#2ca02c', alpha=0.7, edgecolor='black')
ax5.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
ax5.set_xlabel('Prediction Error (Million €)', fontsize=11)
ax5.set_ylabel('Frequency', fontsize=11)
ax5.set_title(f'{best_model_name} - Error Distribution', fontsize=12, fontweight='bold')
ax5.legend()
mean_error = np.mean(best_model_errors) / 1e6
std_error = np.std(best_model_errors) / 1e6
ax5.text(0.95, 0.95, f'Mean Error: €{mean_error:.3f}M\nStd Dev: €{std_error:.3f}M', 
         transform=ax5.transAxes, ha='right', va='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 6. Radar Chart for Top Models
ax6 = axes[1, 2]
# Take top 4 models for radar
top_models = results_df.nsmallest(4, 'RMSE (M€)')
radar_models = top_models['Model'].tolist()
metrics_to_plot = ['RMSE (M€)', 'MAE (€)', 'MAPE (%)']

# Normalize metrics (lower is better, so invert)
normalized_data = []
for _, row in top_models.iterrows():
    norm_rmse = 1 - (row['RMSE (M€)'] / results_df['RMSE (M€)'].max())
    norm_mae = 1 - (row['MAE (€)'] / results_df['MAE (€)'].max())
    norm_mape = 1 - (row['MAPE (%)'] / results_df['MAPE (%)'].max())
    normalized_data.append([norm_rmse, norm_mae, norm_mape])

# Radar chart
angles = np.linspace(0, 2 * np.pi, len(metrics_to_plot), endpoint=False).tolist()
angles += angles[:1]

for i, model in enumerate(radar_models):
    values = normalized_data[i]
    values += values[:1]
    ax6.plot(angles, values, 'o-', linewidth=2, label=model[:15])
    ax6.fill(angles, values, alpha=0.1)

ax6.set_xticks(angles[:-1])
ax6.set_xticklabels(metrics_to_plot)
ax6.set_ylim(0, 1)
ax6.set_title('Top Models - Normalized Performance', fontsize=12, fontweight='bold')
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

plt.tight_layout()
plt.savefig('models/visualizations/full_evaluation_dashboard.png', dpi=150, bbox_inches='tight')
print("    ✓ Full evaluation dashboard saved")

# ============================================
# 6. ADDITIONAL ANALYSIS - Improvement Stats
# ============================================
print("\n[5] Calculating improvement statistics...")

# Find best and baseline
baseline_idx = 0  # Univariate LSTM
best_idx = results_df['RMSE (M€)'].idxmin()

baseline_rmse = results_df.loc[baseline_idx, 'RMSE (M€)']
best_rmse = results_df.loc[best_idx, 'RMSE (M€)']
improvement = ((baseline_rmse - best_rmse) / baseline_rmse) * 100

print(f"\n✓ Best Model: {results_df.loc[best_idx, 'Model']}")
print(f"✓ Improvement over Univariate LSTM: {improvement:.1f}%")

# ============================================
# 7. SAVE RESULTS
# ============================================
print("\n[6] Saving results...")

# Save to CSV
results_df.to_csv('models/evaluation_reports/model_comparison_full.csv', index=False)
print("    ✓ Saved: models/evaluation_reports/model_comparison_full.csv")

# Generate HTML report
html_report = f"""
<!DOCTYPE html>
<html>
<head>
    <title>TransferIQ - Model Evaluation Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        .best {{ background-color: #d4edda; font-weight: bold; }}
        .metrics {{ display: flex; gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: #f8f9fa; padding: 15px; border-radius: 8px; flex: 1; text-align: center; }}
        .metric-value {{ font-size: 24px; font-weight: bold; color: #3498db; }}
        .metric-label {{ color: #7f8c8d; margin-top: 5px; }}
        img {{ max-width: 100%; margin: 20px 0; border: 1px solid #ddd; border-radius: 8px; }}
        .footer {{ margin-top: 40px; text-align: center; color: #7f8c8d; font-size: 12px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>⚽ TransferIQ - Model Evaluation Report</h1>
    <p><strong>Date:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    <p><strong>Dataset:</strong> {df.shape[0]} records, {df['player_name'].nunique()} players</p>
    
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{results_df.loc[best_idx, 'RMSE (M€)']:.3f}M€</div>
            <div class="metric-label">Best Model RMSE</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{results_df.loc[best_idx, 'R² Score']:.4f}</div>
            <div class="metric-label">Best Model R²</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{improvement:.1f}%</div>
            <div class="metric-label">Improvement vs Baseline</div>
        </div>
    </div>
    
    <h2>📊 Model Performance Comparison</h2>
    {display_df.to_html(index=False, classes='table')}
    
    <h2>📈 Visualization Dashboard</h2>
    <img src="../visualizations/full_evaluation_dashboard.png" alt="Evaluation Dashboard">
    
    <h2>🎯 Key Findings</h2>
    <ul>
        <li><strong>Best Model:</strong> {results_df.loc[best_idx, 'Model']} (RMSE: {results_df.loc[best_idx, 'RMSE (M€)']:.3f}M€)</li>
        <li><strong>Baseline Model:</strong> Univariate LSTM (RMSE: {results_df.loc[0, 'RMSE (M€)']:.3f}M€)</li>
        <li><strong>Improvement:</strong> {improvement:.1f}% reduction in RMSE</li>
        <li><strong>Best R² Score:</strong> {results_df.loc[best_idx, 'R² Score']:.4f}</li>
    </ul>
    
    <h2>💡 Recommendations</h2>
    <ul>
        <li>Deploy <strong>{results_df.loc[best_idx, 'Model']}</strong> for production predictions</li>
        <li>Consider adding more features (contract length, injury history) for further improvement</li>
        <li>Implement periodic retraining with new season data</li>
    </ul>
    
    <div class="footer">
        <p>TransferIQ - AI-Powered Player Market Value Prediction System</p>
        <p>Powered by LSTM Neural Networks + XGBoost Stacking Ensemble</p>
    </div>
</div>
</body>
</html>
"""

with open('models/evaluation_reports/evaluation_report.html', 'w', encoding='utf-8') as f:
    f.write(html_report)
print("    ✓ Saved: models/evaluation_reports/evaluation_report.html")

# ============================================
# 8. FINAL SUMMARY
# ============================================
print("\n" + "=" * 80)
print("EVALUATION COMPLETE!")
print("=" * 80)
print("\nFiles Generated:")
print("  1. models/evaluation_reports/model_comparison_full.csv")
print("  2. models/evaluation_reports/evaluation_report.html")
print("  3. models/visualizations/full_evaluation_dashboard.png")
print("\nTo view HTML report:")
print("  start models/evaluation_reports/evaluation_report.html")
print("=" * 80)

# Print summary table
print("\n" + "=" * 80)
print("FINAL RESULTS SUMMARY")
print("=" * 80)
print(display_df[['Model', 'RMSE (M€)', 'R² Score', 'MAPE (%)']].to_string(index=False))
print("=" * 80)