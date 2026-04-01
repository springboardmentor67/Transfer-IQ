"""
Create Stacking Dataset
Combines original data with LSTM predictions as new columns
"""

import pandas as pd
import numpy as np
import os

print("=" * 70)
print("CREATING STACKING DATASET WITH LSTM PREDICTIONS AS COLUMNS")
print("=" * 70)

# Create directories
os.makedirs('models/predictions', exist_ok=True)

# 1. Load original data
print("\n[Step 1] Loading original player data...")
df_original = pd.read_csv('player_data.csv')
print(f"  Original shape: {df_original.shape}")
print(f"  Original columns: {df_original.columns.tolist()[:10]}...")

# 2. Load LSTM predictions
print("\n[Step 2] Loading LSTM predictions...")

# Univariate predictions
univariate_file = 'models/predictions/univariate_predictions.csv'
df_univariate = pd.read_csv(univariate_file)
print(f"  ✓ Univariate predictions: {len(df_univariate)} rows")
print(f"    Columns: {df_univariate.columns.tolist()}")

# Multivariate predictions
multivariate_file = 'models/predictions/multivariate_predictions.csv'
df_multivariate = pd.read_csv(multivariate_file)
print(f"  ✓ Multivariate predictions: {len(df_multivariate)} rows")

# Encoder-Decoder predictions
encoder_file = 'models/predictions/encoder_decoder_predictions.csv'
df_encoder = pd.read_csv(encoder_file)
print(f"  ✓ Encoder-Decoder predictions: {len(df_encoder)} rows")

# 3. Merge all predictions with original data
print("\n[Step 3] Merging predictions with original data...")

# Start with original data
df_stacking = df_original.copy()

# Add univariate predictions
df_stacking = df_stacking.merge(
    df_univariate[['player_name', 'season', 'univariate_prediction']],
    on=['player_name', 'season'],
    how='left'
)

# Add multivariate predictions
df_stacking = df_stacking.merge(
    df_multivariate[['player_name', 'season', 'multivariate_prediction']],
    on=['player_name', 'season'],
    how='left'
)

# Add encoder-decoder predictions
df_stacking = df_stacking.merge(
    df_encoder[['player_name', 'season', 'encoder_decoder_prediction']],
    on=['player_name', 'season'],
    how='left'
)

print(f"  Merged shape: {df_stacking.shape}")

# 4. Fill any missing predictions
print("\n[Step 4] Filling missing values...")

prediction_cols = ['univariate_prediction', 'multivariate_prediction', 'encoder_decoder_prediction']
for col in prediction_cols:
    missing = df_stacking[col].isnull().sum()
    if missing > 0:
        print(f"  Filling {missing} missing values in {col}")
        df_stacking[col] = df_stacking[col].fillna(df_stacking['market_value_eur'])

# 5. Calculate error metrics
print("\n[Step 5] Calculating error metrics...")

# Raw errors (Actual - Predicted)
df_stacking['univariate_error'] = df_stacking['market_value_eur'] - df_stacking['univariate_prediction']
df_stacking['multivariate_error'] = df_stacking['market_value_eur'] - df_stacking['multivariate_prediction']
df_stacking['encoder_decoder_error'] = df_stacking['market_value_eur'] - df_stacking['encoder_decoder_prediction']

# Absolute errors (MAE)
df_stacking['univariate_mae'] = np.abs(df_stacking['univariate_error'])
df_stacking['multivariate_mae'] = np.abs(df_stacking['multivariate_error'])
df_stacking['encoder_decoder_mae'] = np.abs(df_stacking['encoder_decoder_error'])

# Percentage errors
df_stacking['univariate_error_pct'] = (df_stacking['univariate_error'] / (df_stacking['market_value_eur'] + 1)) * 100
df_stacking['multivariate_error_pct'] = (df_stacking['multivariate_error'] / (df_stacking['market_value_eur'] + 1)) * 100
df_stacking['encoder_decoder_error_pct'] = (df_stacking['encoder_decoder_error'] / (df_stacking['market_value_eur'] + 1)) * 100

print(f"  ✓ Error metrics calculated")

# 6. Create lag features (optional but useful for XGBoost)
print("\n[Step 6] Creating lag features...")

df_stacking = df_stacking.sort_values(['player_name', 'season'])

# Previous season values
df_stacking['market_value_lag1'] = df_stacking.groupby('player_name')['market_value_eur'].shift(1)
df_stacking['market_value_lag2'] = df_stacking.groupby('player_name')['market_value_eur'].shift(2)
df_stacking['univariate_lag1'] = df_stacking.groupby('player_name')['univariate_prediction'].shift(1)

# Fill NaN values
lag_cols = ['market_value_lag1', 'market_value_lag2', 'univariate_lag1']
for col in lag_cols:
    df_stacking[col] = df_stacking[col].fillna(df_stacking['market_value_eur'])

print(f"  ✓ Lag features created")

# 7. Reorder columns to put LSTM predictions first
print("\n[Step 7] Organizing columns...")

# Important columns to show first
important_cols = ['player_name', 'season', 'market_value_eur']
lstm_cols = ['univariate_prediction', 'multivariate_prediction', 'encoder_decoder_prediction']
error_cols = ['univariate_error', 'multivariate_error', 'encoder_decoder_error', 
              'univariate_mae', 'multivariate_mae', 'encoder_decoder_mae']

# Get all other columns
other_cols = [col for col in df_stacking.columns if col not in important_cols + lstm_cols + error_cols]

# Reorder
new_order = important_cols + lstm_cols + error_cols + other_cols
df_stacking = df_stacking[new_order]

# 8. Save the stacking dataset
print("\n[Step 8] Saving stacking dataset...")

# Save as CSV
csv_file = 'models/stacking_dataset.csv'
df_stacking.to_csv(csv_file, index=False)
print(f"  ✓ CSV saved: {csv_file}")

# Save as Excel
excel_file = 'models/stacking_dataset.xlsx'
try:
    df_stacking.to_excel(excel_file, index=False)
    print(f"  ✓ Excel saved: {excel_file}")
except:
    print(f"  ⚠ Excel export failed (install openpyxl: pip install openpyxl)")

# 9. Show summary
print("\n" + "=" * 70)
print("STACKING DATASET CREATED SUCCESSFULLY!")
print("=" * 70)
print(f"\nDataset shape: {df_stacking.shape[0]} rows × {df_stacking.shape[1]} columns")
print(f"\nColumns in stacking dataset:")
print("-" * 50)

print("\n[ORIGINAL FEATURES]")
for col in important_cols:
    print(f"  • {col}")

print("\n[LSTM PREDICTIONS (THESE ARE THE OUTPUTS FROM LSTM MODELS)]")
for col in lstm_cols:
    print(f"  • {col}")

print("\n[ERROR METRICS]")
for col in error_cols[:6]:
    print(f"  • {col}")

# 10. Show sample data
print("\n" + "=" * 70)
print("SAMPLE DATA (First 5 rows)")
print("=" * 70)
print(df_stacking[important_cols + lstm_cols + error_cols[:3]].head(10).to_string())

# 11. Statistics
print("\n" + "=" * 70)
print("PREDICTION STATISTICS")
print("=" * 70)

print(f"\nAverage Univariate Error: €{df_stacking['univariate_error'].mean():,.0f}")
print(f"Average Multivariate Error: €{df_stacking['multivariate_error'].mean():,.0f}")
print(f"Average Encoder-Decoder Error: €{df_stacking['encoder_decoder_error'].mean():,.0f}")

print(f"\nUnivariate MAE: €{df_stacking['univariate_mae'].mean():,.0f}")
print(f"Multivariate MAE: €{df_stacking['multivariate_mae'].mean():,.0f}")
print(f"Encoder-Decoder MAE: €{df_stacking['encoder_decoder_mae'].mean():,.0f}")

print("\n" + "=" * 70)
print("✓ STACKING DATASET IS READY FOR XGBOOST!")
print("=" * 70)
print("\nThe stacking dataset contains:")
print("  1. Original player features")
print("  2. LSTM predictions as new columns ✓")
print("  3. Error metrics for each LSTM model ✓")
print("\nNext step: Train XGBoost using these LSTM predictions as input features")
print("Command: python models/train_xgboost_stacking.py")
print("=" * 70)