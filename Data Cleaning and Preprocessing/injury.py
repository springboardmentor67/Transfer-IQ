import pandas as pd
import numpy as np

# ===============================
# 1. LOAD DATASET
# ===============================

df = pd.read_csv(r"player injury dataset\dataset.csv")

print("Dataset Loaded Successfully")
print(df.head())

# Standardize column names
df.columns = df.columns.str.lower().str.strip()

# ===============================
# 2. BASIC CLEANING
# ===============================

# Fill important numeric columns
numeric_cols = [
    'season_days_injured',
    'total_days_injured',
    'season_days_injured_prev_season',
    'cumulative_days_injured',
    'significant_injury_prev_season'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# ===============================
# 3. FEATURE ENGINEERING
# ===============================

# Injury frequency relative to games played
df['injury_days_per_game'] = df['season_days_injured'] / (df['season_games_played'] + 1)

# Injury growth trend
df['injury_trend'] = df['season_days_injured'] - df['season_days_injured_prev_season']

# Severe injury flag (if season_days > 60)
df['severe_season_injury'] = np.where(df['season_days_injured'] > 60, 1, 0)

# Long-term injury burden
df['long_term_injury_ratio'] = df['cumulative_days_injured'] / (df['total_days_injured'] + 1)

# ===============================
# 4. CREATE INJURY RISK SCORE
# ===============================

df['injury_risk_score'] = (
    0.35 * df['season_days_injured'] +
    0.25 * df['injury_trend'] +
    0.20 * df['significant_injury_prev_season'] +
    0.20 * df['long_term_injury_ratio']
)

# Normalize risk score
df['injury_risk_score'] = (
    (df['injury_risk_score'] - df['injury_risk_score'].min()) /
    (df['injury_risk_score'].max() - df['injury_risk_score'].min())
)

# ===============================
# 5. SELECT MODEL FEATURES
# ===============================

model_features = df[[
    'p_id2',
    'start_year',
    'age',
    'bmi',
    'season_days_injured',
    'season_days_injured_prev_season',
    'cumulative_days_injured',
    'injury_days_per_game',
    'injury_trend',
    'severe_season_injury',
    'injury_risk_score'
]]

# Fill remaining missing values
model_features = model_features.fillna(0)

# ===============================
# 6. SAVE FINAL DATASET
# ===============================

model_features.to_csv("injuries.csv", index=False)

print("\nFinal Model-Ready Dataset Created")
print(model_features.head())
