import pandas as pd
import numpy as np

print("Loading master dataset...")

df = pd.read_csv("outputs/player_master_base.csv")

# -----------------------------
# SAFE DIVISION
# -----------------------------
def safe_div(a, b):
    return a / b.replace(0, 1)

# -----------------------------
# ATTACK METRICS
# -----------------------------
df["goal_contribution"] = df["goals"] + df["assists"]
df["goal_contribution_per_match"] = safe_div(df["goal_contribution"], df["matches_played"])
df["shot_conversion_rate"] = safe_div(df["goals"], df["shots"])
df["attack_volume"] = df["shots"] + df["passes"]

# -----------------------------
# DEFENSIVE METRICS
# -----------------------------
df["defensive_actions"] = df["tackles"] + df["interceptions"] + df["clearances"]
df["defensive_actions_per_match"] = safe_div(df["defensive_actions"], df["matches_played"])

# -----------------------------
# DISCIPLINE METRICS
# -----------------------------
df["discipline_score"] = (
    df["yellow_cards"] * 1 +
    df["red_cards"] * 3 +
    df["fouls"] * 0.5
)

df["discipline_per_match"] = safe_div(df["discipline_score"], df["matches_played"])

# -----------------------------
# AVAILABILITY METRICS
# -----------------------------
df["availability_score"] = safe_div(df["total_minutes"], df["matches_played"])
df["substitution_impact"] = safe_div(df["sub_on"], df["matches_played"])

# -----------------------------
# PERFORMANCE INTENSITY
# -----------------------------
df["actions_total"] = (
    df["shots"] + df["passes"] +
    df["tackles"] + df["interceptions"] +
    df["clearances"]
)

df["actions_per_match"] = safe_div(df["actions_total"], df["matches_played"])

# -----------------------------
# INJURY PROXY (since scraping unstable)
# -----------------------------
df["injury_proxy_score"] = (
    1 - safe_div(df["total_minutes"], df["matches_played"] * 90)
)

df["injury_proxy_score"] = df["injury_proxy_score"].clip(lower=0)

# -----------------------------
# ROLE PROFILE SCORE
# -----------------------------
df["attack_index"] = (
    df["goal_contribution_per_match"] * 2 +
    df["shot_conversion_rate"] * 3 +
    df["actions_per_match"]
)

df["defense_index"] = (
    df["defensive_actions_per_match"] * 2 -
    df["discipline_per_match"]
)

# -----------------------------
# FINAL IMPACT SCORE
# -----------------------------
df["impact_score"] = (
    df["attack_index"] +
    df["defense_index"] +
    df["availability_score"] -
    df["injury_proxy_score"]
)

print("Advanced features created.")

df.fillna(0, inplace=True)

df.to_csv("outputs/player_master_advanced.csv", index=False)

print("DONE. Advanced feature dataset created.")