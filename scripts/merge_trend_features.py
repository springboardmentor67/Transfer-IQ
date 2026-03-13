import pandas as pd

print("Merging trend features...")

master = pd.read_csv("outputs/player_master_with_value.csv")
trend = pd.read_csv("outputs/player_trend_features.csv")

final = master.merge(trend, on="player_name", how="left")

final.fillna(0, inplace=True)

final.to_csv("outputs/player_master_final.csv", index=False)

print("Final modeling dataset created.")