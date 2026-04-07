import pandas as pd
import numpy as np

# Load your longitudinal dataset
df = pd.read_csv('merged_football_dataset.csv')

# --- FEATURE ENGINEERING FOR MODELING ---

# 1. Target Variables (What the model might predict)
# Year-Over-Year Change in Market Value
df['market_value_yoy_change'] = df.groupby('player_name')['market_value_eur'].diff().fillna(0)

# Percentage Change in Market Value (with epsilon to prevent division by zero)
epsilon = 1e-5
df['market_value_yoy_pct_change'] = (
    df['market_value_yoy_change'] / 
    (df['market_value_eur'] - df['market_value_yoy_change'] + epsilon)
).fillna(0).clip(lower=-1.0, upper=5.0) # Clip extreme percentages for stability

# 2. Performance Metrics (Standardized to per 90 minutes)
# Attacking Metric: Goal Involvement per 90 mins
df['goal_involvement_per_90'] = (
    (df['goals_total'] + df['assists_total']) / (df['minutes_played_season'] / 90)
).fillna(0)

# Replace 'inf' values that happen if a player played 0 minutes
df.replace([np.inf, -np.inf], 0, inplace=True)

# Defensive Metric: Defensive Actions per 90 mins
df['defensive_actions_per_90'] = (
    (df['sb_tackles'] + df['sb_interceptions']) / (df['minutes_played_season'] / 90)
).fillna(0)

# 3. Injury & Availability Metrics
# Availability Index: 0 to 1 scale (1 means available 100% of the year)
df['availability_index'] = (1 - (df['season_days_injured'] / 365)).clip(0, 1)

# 4. Sentiment Metrics
# Composite Sentiment Score (Average of Fan and Media)
df['overall_sentiment'] = (df['fan_sentiment'] + df['media_sentiment']) / 2

# Sentiment Trend (Is the player's reputation currently improving or worsening?)
df['sentiment_yoy_change'] = df.groupby('player_name')['overall_sentiment'].diff().fillna(0)

# --- FINAL SELECTION ---
# Extract only the relevant features for the final modeling dataset
final_cols = [
    'player_name', 'season_year', 'age', 'position', 
    'market_value_eur', 'market_value_yoy_change', 'market_value_yoy_pct_change', 
    'minutes_played_season', 'goal_involvement_per_90', 'defensive_actions_per_90', 'sb_pass_accuracy',
    'season_days_injured', 'injury_risk_score', 'availability_index', 
    'fan_sentiment', 'media_sentiment', 'overall_sentiment', 'sentiment_yoy_change'
]

modeling_df = df[final_cols]

# Save the final engineered features
# modeling_df.to_csv('final_modeling_features.csv', index=False)