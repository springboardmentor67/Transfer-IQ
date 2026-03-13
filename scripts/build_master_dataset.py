import pandas as pd

print("Loading datasets...")

performance = pd.read_csv("outputs/player_performance_master.csv")
playtime = pd.read_csv("outputs/player_playtime_master.csv")

print("Performance shape:", performance.shape)
print("Playtime shape:", playtime.shape)

# Merge
master = performance.merge(
    playtime,
    on="player_name",
    how="outer"
)

# Fill missing values
master.fillna(0, inplace=True)

# Remove completely empty rows
master = master[(master["matches_played"] > 0) | (master["shots"] > 0)]

print("After merge:", master.shape)

master.to_csv("outputs/player_master_base.csv", index=False)

print("DONE. Master base dataset created.")