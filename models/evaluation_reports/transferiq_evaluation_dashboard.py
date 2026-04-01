"""
TransferIQ - Complete Model Evaluation Dashboard
Format: RMSE, MAE, R², MAPE with comprehensive visualizations
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

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("Set2")

print("=" * 80)
print("TRANSFERIQ - MODEL EVALUATION DASHBOARD")
print("=" * 80)

# Create directories
os.makedirs('models/evaluation_reports', exist_ok=True)
os.makedirs('models/visualizations', exist_ok=True)

# ============================================
# 1. LOAD DATA
# ============================================
print("\n[1] Loading data...")
df = pd.read_csv('models/stacking_dataset.csv')
print(f"    Dataset: {df.shape[0]} records, {df['player_name'].nunique()} players")
print(f"    Value range: €{df['market_value_eur'].min():,.0f} - €{df['market_value_eur'].max():,.0f}")

# Get actual values
y_actual = df['market_value_eur'].values

# ============================================
# 2. GET MODEL PREDICTIONS
# ============================================
print("\n[2] Getting model predictions...")

# LSTM models
y_univariate = df['univariate_prediction'].values
y_multivariate = df['multivariate_prediction'].values
y_encoder = df['encoder_decoder_prediction'].values

# Ensemble (Simple Average)
y_ensemble = (y_univariate + y_multivariate + y_encoder) / 3

# Load tuned XGBoost if available
y_xgb_tuned = None
try:
    xgb_path = 'models/hyperparameter_tuning/best_xgboost_model.pkl'
    if os.path.exists(xgb_path):
        xgb_model = joblib.load(xgb_path)
        feature_cols = ['univariate_prediction', 'multivariate_prediction', 'encoder_decoder_prediction']
        X = df[feature_cols].fillna(0)
        y_xgb_tuned = xgb_model.predict(X)
        print("    ✓ Tuned XGBoost loaded")
    else:
        print("    ⚠ Tuned XGBoost not found - using fallback")
        y_xgb_tuned = y_ensemble
except Exception as e:
    print(f"    ⚠ XGBoost error: {e}")
    y_xgb_tuned = y_ensemble

# Weighted Ensemble (60% XGB + 40% LSTM Avg)
y_weighted = 0.6 * y_xgb_tuned + 0.4 * y_ensemble

# ============================================
# 3. CALCULATE METRICS FUNCTION
# ============================================
def calculate_metrics(y_true, y_pred, model_name):
    """Calculate all evaluation metrics"""
    # Standard metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE (avoid division by zero)
    mask = y_true > 1000
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = 0
    
    # RMSE in Millions
    rmse_m = rmse / 1_000_000
    
    # Additional metrics
    mse = mean_squared_error(y_true, y_pred)
    mae_m = mae / 1_000_000
    
    # Explained Variance
    explained_variance = 1 - (np.var(y_true - y_pred) / np.var(y_true))
    
    # Mean Absolute Percentage Error (alternate)
    mape_alt = np.mean(np.abs((y_true - y_pred) / (y_true + 1))) * 100
    
    return {
        'Model': model_name,
        'RMSE (€)': rmse,
        'RMSE (M€)': rmse_m,
        'MAE (€)': mae,
        'MAE (M€)': mae_m,
        'MSE (€²)': mse,
        'R² Score': r2,
        'Explained Variance': explained_variance,
        'MAPE (%)': mape,
        'MAPE_Alt (%)': mape_alt
    }

# ============================================
# 4. COLLECT ALL MODEL RESULTS
# ============================================
print("\n[3] Calculating metrics...")

models = [
    ('Univariate LSTM', y_univariate),
    ('Multivariate LSTM', y_multivariate),
    ('Encoder-Decoder LSTM', y_encoder),
    ('Ensemble (Simple Avg)', y_ensemble),
    ('XGBoost (Tuned)', y_xgb_tuned),
    ('Ensemble (Weighted)', y_weighted)
]

results = []
for name, pred in models:
    metrics = calculate_metrics(y_actual, pred, name)
    results.append(metrics)

results_df = pd.DataFrame(results)

# ============================================
# SECTION 1: RMSE, MAE, R², MAPE TABLE
# ============================================
print("\n" + "=" * 80)
print("SECTION 1: MODEL PERFORMANCE METRICS")
print("=" * 80)

# Format display table
display_table = results_df[['Model', 'RMSE (M€)', 'MAE (M€)', 'R² Score', 'MAPE (%)']].copy()
display_table['RMSE (M€)'] = display_table['RMSE (M€)'].round(4)
display_table['MAE (M€)'] = display_table['MAE (M€)'].round(4)
display_table['R² Score'] = display_table['R² Score'].round(4)
display_table['MAPE (%)'] = display_table['MAPE (%)'].round(2)

print("\n" + display_table.to_string(index=False))

# Save to CSV
display_table.to_csv('models/evaluation_reports/section1_metrics_table.csv', index=False)
print("\n✓ Saved: models/evaluation_reports/section1_metrics_table.csv")

# Create visualization for Section 1
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('TransferIQ - Model Performance Metrics', fontsize=16, fontweight='bold')

# 1a. RMSE Comparison (Million €)
ax1 = axes[0, 0]
models_names = results_df['Model'].tolist()
rmse_values = results_df['RMSE (M€)'].tolist()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
bars = ax1.bar(models_names, rmse_values, color=colors[:len(models_names)])
ax1.set_ylabel('RMSE (Million €)', fontsize=12)
ax1.set_title('RMSE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
ax1.tick_params(axis='x', rotation=45, labelsize=9)
for bar, val in zip(bars, rmse_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.3f}M', ha='center', va='bottom', fontsize=8)

# 1b. MAE Comparison (Million €)
ax2 = axes[0, 1]
mae_values = results_df['MAE (M€)'].tolist()
bars = ax2.bar(models_names, mae_values, color=colors[:len(models_names)])
ax2.set_ylabel('MAE (Million €)', fontsize=12)
ax2.set_title('MAE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
ax2.tick_params(axis='x', rotation=45, labelsize=9)
for bar, val in zip(bars, mae_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.3f}M', ha='center', va='bottom', fontsize=8)

# 1c. R² Score Comparison
ax3 = axes[1, 0]
r2_values = results_df['R² Score'].tolist()
bars = ax3.bar(models_names, r2_values, color=colors[:len(models_names)])
ax3.set_ylabel('R² Score', fontsize=12)
ax3.set_title('R² Score Comparison (Higher is Better)', fontsize=12, fontweight='bold')
ax3.tick_params(axis='x', rotation=45, labelsize=9)
ax3.axhline(y=0.99, color='green', linestyle='--', alpha=0.5, label='Target (0.99)')
for bar, val in zip(bars, r2_values):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002, 
             f'{val:.4f}', ha='center', va='bottom', fontsize=8)

# 1d. MAPE Comparison
ax4 = axes[1, 1]
mape_values = results_df['MAPE (%)'].tolist()
bars = ax4.bar(models_names, mape_values, color=colors[:len(models_names)])
ax4.set_ylabel('MAPE (%)', fontsize=12)
ax4.set_title('MAPE Comparison (Lower is Better)', fontsize=12, fontweight='bold')
ax4.tick_params(axis='x', rotation=45, labelsize=9)
for bar, val in zip(bars, mape_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('models/visualizations/section1_metrics_charts.png', dpi=150, bbox_inches='tight')
print("✓ Saved: models/visualizations/section1_metrics_charts.png")

# ============================================
# SECTION 2: RMSE ANALYSIS - ALL GRAPHS AND PLOTS
# ============================================
print("\n" + "=" * 80)
print("SECTION 2: RMSE ANALYSIS - COMPREHENSIVE VISUALIZATIONS")
print("=" * 80)

fig, axes = plt.subplots(3, 3, figsize=(16, 14))
fig.suptitle('TransferIQ - RMSE Analysis Dashboard', fontsize=16, fontweight='bold')

# 2a. RMSE Bar Chart (Detailed)
ax1 = axes[0, 0]
bars = ax1.bar(models_names, rmse_values, color=colors)
ax1.set_ylabel('RMSE (Million €)', fontsize=11)
ax1.set_title('1. RMSE by Model', fontsize=12, fontweight='bold')
ax1.tick_params(axis='x', rotation=45, labelsize=8)
for bar, val in zip(bars, rmse_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.3f}M', ha='center', va='bottom', fontsize=8)

# 2b. RMSE Improvement over Baseline
ax2 = axes[0, 1]
baseline_rmse = rmse_values[0]
improvements = [(baseline_rmse - val) / baseline_rmse * 100 for val in rmse_values]
colors_imp = ['gray'] + ['#2ca02c' if imp > 0 else '#d62728' for imp in improvements[1:]]
bars = ax2.bar(models_names, improvements, color=colors_imp)
ax2.set_ylabel('Improvement over Baseline (%)', fontsize=11)
ax2.set_title('2. RMSE Improvement vs Univariate LSTM', fontsize=12, fontweight='bold')
ax2.tick_params(axis='x', rotation=45, labelsize=8)
ax2.axhline(y=0, color='red', linestyle='--', linewidth=1)
for bar, val in zip(bars, improvements):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val:.1f}%', ha='center', va='bottom', fontsize=8)

# 2c. RMSE Distribution (Box Plot)
ax3 = axes[0, 2]
errors_data = []
for name, pred in models:
    errors = y_actual - pred
    errors_data.append(errors / 1e6)
bp = ax3.boxplot(errors_data, labels=[m[:12] for m in models_names], patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax3.set_ylabel('Prediction Error (Million €)', fontsize=11)
ax3.set_title('3. Error Distribution by Model', fontsize=12, fontweight='bold')
ax3.tick_params(axis='x', rotation=45, labelsize=8)
ax3.axhline(y=0, color='red', linestyle='--', linewidth=1)

# 2d. RMSE vs MAE Scatter
ax4 = axes[1, 0]
mae_values = results_df['MAE (M€)'].tolist()
ax4.scatter(rmse_values, mae_values, s=100, c=colors, alpha=0.7)
for i, name in enumerate(models_names):
    ax4.annotate(name[:15], (rmse_values[i], mae_values[i]), fontsize=8)
ax4.set_xlabel('RMSE (Million €)', fontsize=11)
ax4.set_ylabel('MAE (Million €)', fontsize=11)
ax4.set_title('4. RMSE vs MAE Correlation', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)

# 2e. RMSE Heatmap (Model Comparison)
ax5 = axes[1, 1]
rmse_matrix = np.array([rmse_values])
sns.heatmap(rmse_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r', 
            xticklabels=models_names, yticklabels=['RMSE (M€)'], ax=ax5)
ax5.set_title('5. RMSE Heatmap', fontsize=12, fontweight='bold')
ax5.tick_params(axis='x', rotation=45, labelsize=8)

# 2f. Cumulative RMSE by Data Points
ax6 = axes[1, 2]
sorted_indices = np.argsort(y_actual)
sorted_errors = np.abs(y_actual - y_weighted)[sorted_indices]
cumulative_rmse = np.sqrt(np.cumsum(sorted_errors**2) / np.arange(1, len(sorted_errors)+1))
ax6.plot(cumulative_rmse / 1e6, linewidth=2, color='#1f77b4')
ax6.set_xlabel('Number of Data Points', fontsize=11)
ax6.set_ylabel('Cumulative RMSE (Million €)', fontsize=11)
ax6.set_title('6. Cumulative RMSE Analysis', fontsize=12, fontweight='bold')
ax6.grid(True, alpha=0.3)

# 2g. Residuals vs Predicted (Best Model)
ax7 = axes[2, 0]
best_idx = np.argmin(rmse_values)
best_name = models_names[best_idx]
best_pred = models[best_idx][1]
residuals = y_actual - best_pred
ax7.scatter(best_pred / 1e6, residuals / 1e6, alpha=0.5, s=10, color='#2ca02c')
ax7.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax7.set_xlabel('Predicted Value (Million €)', fontsize=11)
ax7.set_ylabel('Residuals (Million €)', fontsize=11)
ax7.set_title(f'7. Residuals Plot - {best_name}', fontsize=12, fontweight='bold')
ax7.grid(True, alpha=0.3)

# 2h. RMSE by Value Range
ax8 = axes[2, 1]
bins = [0, 10e6, 20e6, 50e6, 100e6, 200e6]
labels = ['0-10M', '10-20M', '20-50M', '50-100M', '100-200M']
value_ranges = pd.cut(y_actual, bins=bins, labels=labels)
rmse_by_range = []
for i, (name, pred) in enumerate(models):
    range_rmse = []
    for label in labels:
        mask = value_ranges == label
        if mask.sum() > 0:
            rmse = np.sqrt(mean_squared_error(y_actual[mask], pred[mask])) / 1e6
            range_rmse.append(rmse)
        else:
            range_rmse.append(0)
    rmse_by_range.append(range_rmse)

rmse_by_range = np.array(rmse_by_range)
x = np.arange(len(labels))
width = 0.12
for i, name in enumerate(models_names[:5]):
    ax8.bar(x + i*width, rmse_by_range[i], width, label=name[:12], color=colors[i])
ax8.set_xlabel('Market Value Range', fontsize=11)
ax8.set_ylabel('RMSE (Million €)', fontsize=11)
ax8.set_title('8. RMSE by Value Range', fontsize=12, fontweight='bold')
ax8.set_xticks(x + width*2)
ax8.set_xticklabels(labels, rotation=45)
ax8.legend(loc='upper left', fontsize=7)

# 2i. RMSE Distribution Histogram
ax9 = axes[2, 2]
ax9.hist(rmse_values, bins=10, color='#1f77b4', edgecolor='black', alpha=0.7)
ax9.set_xlabel('RMSE (Million €)', fontsize=11)
ax9.set_ylabel('Frequency', fontsize=11)
ax9.set_title('9. RMSE Distribution Across Models', fontsize=12, fontweight='bold')
ax9.axvline(x=np.mean(rmse_values), color='red', linestyle='--', label=f'Mean: {np.mean(rmse_values):.3f}M')
ax9.legend()

plt.tight_layout()
plt.savefig('models/visualizations/section2_rmse_analysis.png', dpi=150, bbox_inches='tight')
print("✓ Saved: models/visualizations/section2_rmse_analysis.png")

# ============================================
# SECTION 3: FULL EVALUATION SUMMARY (Like Teammate's Format)
# ============================================
print("\n" + "=" * 80)
print("SECTION 3: FULL EVALUATION SUMMARY")
print("=" * 80)

# Create table like teammate's format
full_table = results_df[['Model', 'RMSE (M€)', 'MAE (M€)', 'R² Score', 'MAPE (%)']].copy()
full_table['RMSE (M€)'] = full_table['RMSE (M€)'].round(4)
full_table['MAE (M€)'] = full_table['MAE (M€)'].round(4)
full_table['R² Score'] = full_table['R² Score'].round(4)
full_table['MAPE (%)'] = full_table['MAPE (%)'].round(2)

print("\n" + "=" * 80)
print("FULL EVALUATION SUMMARY – ALL MODELS")
print("=" * 80)
print(full_table.to_string(index=False))
print("=" * 80)

# Save as CSV
full_table.to_csv('models/evaluation_reports/section3_full_evaluation.csv', index=False)
print("\n✓ Saved: models/evaluation_reports/section3_full_evaluation.csv")

# Create bar chart comparison for Section 3
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('TransferIQ - Full Model Comparison Dashboard', fontsize=16, fontweight='bold')

# 3a. RMSE Comparison (like teammate's)
ax1 = axes[0, 0]
bars = ax1.barh(models_names, rmse_values, color=colors)
ax1.set_xlabel('RMSE (Million €)', fontsize=11)
ax1.set_title('RMSE (Lower is Better)', fontsize=12, fontweight='bold')
for bar, val in zip(bars, rmse_values):
    ax1.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}M', va='center', fontsize=9)

# 3b. R² Score (like teammate's - higher better)
ax2 = axes[0, 1]
bars = ax2.barh(models_names, r2_values, color=colors)
ax2.set_xlabel('R² Score', fontsize=11)
ax2.set_title('R² Score (Higher is Better)', fontsize=12, fontweight='bold')
for bar, val in zip(bars, r2_values):
    ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2, 
             f'{val:.4f}', va='center', fontsize=9)

# 3c. MAPE (like teammate's)
ax3 = axes[1, 0]
bars = ax3.barh(models_names, mape_values, color=colors)
ax3.set_xlabel('MAPE (%)', fontsize=11)
ax3.set_title('MAPE (Lower is Better)', fontsize=12, fontweight='bold')
for bar, val in zip(bars, mape_values):
    ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
             f'{val:.1f}%', va='center', fontsize=9)

# 3d. MAE (like teammate's)
ax4 = axes[1, 1]
bars = ax4.barh(models_names, mae_values, color=colors)
ax4.set_xlabel('MAE (Million €)', fontsize=11)
ax4.set_title('MAE (Lower is Better)', fontsize=12, fontweight='bold')
for bar, val in zip(bars, mae_values):
    ax4.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{val:.3f}M', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('models/visualizations/section3_full_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: models/visualizations/section3_full_comparison.png")

# ============================================
# EXTRA: Radar Chart for Top Models
# ============================================
print("\n[4] Creating radar chart for top models...")

# Take top 4 models for radar
top_indices = np.argsort(rmse_values)[:4]
top_names = [models_names[i] for i in top_indices]
top_rmse = [rmse_values[i] for i in top_indices]
top_r2 = [r2_values[i] for i in top_indices]
top_mape = [mape_values[i] for i in top_indices]

# Normalize metrics (lower is better for RMSE and MAPE, higher is better for R²)
max_rmse = max(rmse_values)
max_mape = max(mape_values)
min_r2 = min(r2_values)

norm_rmse = [1 - (rmse / max_rmse) for rmse in top_rmse]
norm_mape = [1 - (mape / max_mape) for mape in top_mape]
norm_r2 = [(r2 - min_r2) / (1 - min_r2) for r2 in top_r2]

# Radar chart
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
categories = ['RMSE Performance', 'MAPE Performance', 'R² Performance']
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]

for i, name in enumerate(top_names):
    values = [norm_rmse[i], norm_mape[i], norm_r2[i]]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=name[:15])
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 1)
ax.set_title('Top Models - Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

plt.tight_layout()
plt.savefig('models/visualizations/radar_chart_top_models.png', dpi=150, bbox_inches='tight')
print("✓ Saved: models/visualizations/radar_chart_top_models.png")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 80)
print("EVALUATION COMPLETE!")
print("=" * 80)

best_model_idx = np.argmin(rmse_values)
best_model_name = models_names[best_model_idx]
best_rmse = rmse_values[best_model_idx]

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   RMSE: €{best_rmse:.3f}M")
print(f"   R²: {r2_values[best_model_idx]:.4f}")
print(f"   MAPE: {mape_values[best_model_idx]:.2f}%")

print("\n📁 Files Generated:")
print("   SECTION 1:")
print("     - models/evaluation_reports/section1_metrics_table.csv")
print("     - models/visualizations/section1_metrics_charts.png")
print("\n   SECTION 2 (RMSE Analysis):")
print("     - models/visualizations/section2_rmse_analysis.png")
print("\n   SECTION 3 (Full Evaluation):")
print("     - models/evaluation_reports/section3_full_evaluation.csv")
print("     - models/visualizations/section3_full_comparison.png")
print("\n   EXTRA:")
print("     - models/visualizations/radar_chart_top_models.png")

print("\n" + "=" * 80)