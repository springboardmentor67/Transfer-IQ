import pandas as pd
import numpy as np

print("Loading full feature dataset...")

df = pd.read_csv("outputs/player_master_full_features.csv")

# -----------------------------------
# Normalize Key Metrics
# -----------------------------------

def normalize(col):
    min_val = col.min()
    max_val = col.max()
    if max_val == min_val:
        return 0
    return (col - min_val) / (max_val - min_val)

print("Normalizing metrics...")

df["norm_impact"] = normalize(df["impact_score"])
df["norm_attack"] = normalize(df["attack_index"])
df["norm_defense"] = normalize(df["defense_index"])
df["norm_availability"] = normalize(df["availability_ratio"])
df["norm_sentiment"] = normalize(df["hype_score"])
df["norm_injury"] = normalize(df["injury_risk_score"])

# -----------------------------------
# Base Value Score
# -----------------------------------

df["base_value_score"] = (
    df["norm_impact"] * 3 +
    df["norm_attack"] * 2 +
    df["norm_defense"] * 1.5 +
    df["norm_availability"] * 2 +
    df["norm_sentiment"] * 1.5
)

# -----------------------------------
# Risk Penalty
# -----------------------------------

df["risk_penalty"] = (
    df["norm_injury"] * 2 +
    df["discipline_per_match"] * 0.5
)

# -----------------------------------
# Final Value Score
# -----------------------------------

df["final_value_score"] = df["base_value_score"] - df["risk_penalty"]

# Ensure no negatives
df["final_value_score"] = df["final_value_score"].clip(lower=0)

# -----------------------------------
# Convert to Euro Market Value
# -----------------------------------

# Scale between €100K and €150M
min_eur = 100_000
max_eur = 150_000_000

norm_final = normalize(df["final_value_score"])

df["market_value_eur"] = min_eur + norm_final * (max_eur - min_eur)

df["market_value_eur"] = df["market_value_eur"].round(0)

print("Market values generated.")

df.to_csv("outputs/player_master_with_value.csv", index=False)

print("DONE. Market value dataset created.")