import pandas as pd
import numpy as np

print("Loading advanced dataset...")

df = pd.read_csv("outputs/player_master_advanced.csv")

# --------------------------------
# SAFE DIVISION (fixed version)
# --------------------------------
def safe_div(a, b):
    if isinstance(b, (int, float, np.integer, np.floating)):
        if b == 0:
            b = 1
        return a / b
    else:
        return a / b.replace(0, 1)

# --------------------------------
# 1️⃣ Availability Ratio
# --------------------------------
df["availability_ratio"] = safe_div(df["total_minutes"], df["matches_played"] * 90)

# --------------------------------
# 2️⃣ Minutes Variability Proxy
# --------------------------------
df["minutes_variability"] = abs(
    df["minutes_per_match"] - 
    (df["total_minutes"] / df["matches_played"].replace(0, 1))
)

# --------------------------------
# 3️⃣ Absence Rate
# --------------------------------
max_matches = df["matches_played"].max()

df["absence_rate"] = 1 - safe_div(df["matches_played"], max_matches)

# --------------------------------
# 4️⃣ Fatigue Index
# --------------------------------
df["fatigue_index"] = (
    df["total_minutes"] * 0.0001 +
    df["defensive_actions_per_match"] * 0.5
)

# --------------------------------
# 5️⃣ Composite Injury Risk Score
# --------------------------------
df["injury_risk_score"] = (
    df["injury_proxy_score"] * 2 +
    df["minutes_variability"] * 1.5 +
    df["early_sub_rate"] * 2 +
    df["discipline_per_match"] * 1 +
    df["fatigue_index"] * 1
)

# Normalize injury risk score safely
min_val = df["injury_risk_score"].min()
max_val = df["injury_risk_score"].max()

if max_val != min_val:
    df["injury_risk_score"] = (
        (df["injury_risk_score"] - min_val) /
        (max_val - min_val)
    )
else:
    df["injury_risk_score"] = 0

df.fillna(0, inplace=True)

print("Injury features created.")

df.to_csv("outputs/player_master_with_injury.csv", index=False)

print("DONE. Injury-enhanced dataset created.")