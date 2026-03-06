import pandas as pd
import numpy as np

# ==============================
# 1. LOAD DATASETS
# ==============================

per_df = pd.read_csv("statsbomb.csv")          # Performance data
market_df = pd.read_csv("market_values.csv")      # Transfermarkt scraped data
injury_df = pd.read_csv("injuries.csv")           # Injury data
sentiment_df = pd.read_csv("sentiment.csv")       # Sentiment scores


# ==============================
# 2. STANDARDIZE COLUMN NAMES
# ==============================

# Make all player names lowercase for safe merging
for df in [per_df, market_df, injury_df, sentiment_df]:
    if "player_name" in df.columns:
        df["player_name"] = df["player_name"].str.lower().str.strip()
    if "name" in df.columns:
        df["player_name"] = df["name"].str.lower().str.strip()


# ==============================
# 3. SELECT IMPORTANT COLUMNS ONLY
# (Avoid unnecessary columns)
# ==============================

# StatsBomb performance columns
per_df = per_df[[
    "player_name",
    "age",
    "overall",
    "potential",
    "wage_eur",
    "value_eur",
    "club_name",
    "league_name"
]]

# Market value columns
market_df = market_df[[
    "player_name",
    "market_value_eur",
    "contract_until",
    "release_clause_eur"
]]

# Injury features (aggregate per player)
injury_agg = injury_df.groupby("player_name").agg({
    "total_days_injured": "sum",
    "season_days_injured": "sum",
    "injury_count": "sum"
}).reset_index()

# Sentiment features
sentiment_df = sentiment_df[[
    "player_name",
    "sentiment_score",
    "positive_mentions",
    "negative_mentions"
]]


# ==============================
# 4. MERGE DATASETS
# ==============================

# Merge per + Market
merged_df = pd.merge(per_df, market_df, on="player_name", how="left")

# Merge Injury
merged_df = pd.merge(merged_df, injury_agg, on="player_name", how="left")

# Merge Sentiment
merged_df = pd.merge(merged_df, sentiment_df, on="player_name", how="left")


# ==============================
# 5. HANDLE MISSING VALUES
# ==============================

merged_df.fillna({
    "total_days_injured": 0,
    "season_days_injured": 0,
    "injury_count": 0,
    "sentiment_score": 0,
    "positive_mentions": 0,
    "negative_mentions": 0
}, inplace=True)


# ==============================
# 6. FEATURE ENGINEERING
# ==============================

# Contract remaining years
merged_df["contract_until"] = pd.to_datetime(
    merged_df["contract_until"], errors="coerce"
)

merged_df["contract_remaining_years"] = (
    merged_df["contract_until"].dt.year - pd.Timestamp.now().year
)

# Injury risk metric
merged_df["injury_risk"] = (
    merged_df["total_days_injured"] / (merged_df["age"] + 1)
)

# Log transform value (important for ML)
merged_df["log_market_value"] = np.log1p(merged_df["market_value_eur"])


# ==============================
# 7. SAVE FINAL DATASET
# ==============================

merged_df.to_csv("final_merged_dataset.csv", index=False)

print("✅ All datasets merged successfully!")
print("Final shape:", merged_df.shape)
print(merged_df.head())