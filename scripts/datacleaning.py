import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# -------------------------
# LOAD DATASETS
# -------------------------

performance = pd.read_csv("outputs/playerperformance.csv")
sentiment = pd.read_csv("outputs/playersentiment.csv")
injuries = pd.read_csv("outputs/playerinjuries.csv")
market = pd.read_csv("outputs/marketvalues.csv")

# -------------------------
# MERGE DATASETS
# -------------------------

df = performance.merge(sentiment, on="player.name", how="left")
df = df.merge(injuries, on="player.name", how="left")
df = df.merge(market, on="player.name", how="left")

print("After merging:", df.shape)

# -------------------------
# FIX DUPLICATE TEAM COLUMN
# -------------------------

if "team.name_x" in df.columns:
    df["team.name"] = df["team.name_x"]
    df.drop(columns=["team.name_x", "team.name_y"], errors="ignore", inplace=True)

# Remove duplicate players
df = df.drop_duplicates(subset=["player.name"])

print("After removing duplicates:", df.shape)

# -------------------------
# HANDLE MISSING VALUES
# -------------------------

# Numeric → median
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Categorical → mode
categorical_cols = df.select_dtypes(include=["object", "string"]).columns

for col in categorical_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

# -------------------------
# FEATURE ENGINEERING
# -------------------------

# Injury Risk Score
if all(col in df.columns for col in ["total_days_injured", "matches_missed", "total_injuries"]):
    df["injury_risk_score"] = (
        df["total_days_injured"] * 0.5 +
        df["matches_missed"] * 0.3 +
        df["total_injuries"] * 0.2
    )

# Sentiment Strength
if all(col in df.columns for col in ["avg_sentiment", "headline_count"]):
    df["sentiment_strength"] = df["avg_sentiment"] * df["headline_count"]

# Performance Score
if all(col in df.columns for col in ["goals_per90", "assists_per90", "passes_per90", "duels_per90"]):
    df["performance_score"] = (
        df["goals_per90"] * 4 +
        df["assists_per90"] * 3 +
        df["passes_per90"] * 0.5 +
        df["duels_per90"] * 1
    )

# Log Transform Market Value
if "market_value_eur" in df.columns:
    df["market_value_log"] = np.log1p(df["market_value_eur"])

# -------------------------
# ENCODING
# -------------------------

categorical_features = []

if "position" in df.columns:
    categorical_features.append("position")

if "team.name" in df.columns:
    categorical_features.append("team.name")

if len(categorical_features) > 0:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoded = encoder.fit_transform(df[categorical_features])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out()
    )

    df = pd.concat([df.reset_index(drop=True), encoded_df], axis=1)
    df.drop(columns=categorical_features, inplace=True)
# -------------------------
# FIX AGE COLUMN
# -------------------------

if "age" in df.columns:
    # Extract number inside parentheses
    df["age"] = df["age"].astype(str).str.extract(r"\((\d+)\)")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")

# -------------------------
# SCALING
# -------------------------

scale_cols = []

for col in ["injury_risk_score", "performance_score", "sentiment_strength", "age"]:
    if col in df.columns:
        scale_cols.append(col)

if len(scale_cols) > 0:
    scaler = StandardScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols].astype(float))

# -------------------------
# SAVE FINAL DATASET
# -------------------------

df.to_csv("outputs/final_dataset_clean.csv", index=False)

print("Final dataset created.")
print("Final shape:", df.shape)
