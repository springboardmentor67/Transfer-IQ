import pandas as pd
import numpy as np

# ======================================================
# 1️⃣ LOAD DATA
# ======================================================

players = pd.read_csv("Football player analyzer AI\\transfermarkt dataset\\players.csv")
valuations = pd.read_csv("Football player analyzer AI\\transfermarkt dataset\\player_valuations.csv")
appearances = pd.read_csv("Football player analyzer AI\\transfermarkt dataset\\appearances.csv")
games = pd.read_csv("Football player analyzer AI\\transfermarkt dataset\\games.csv")
competitions = pd.read_csv("Football player analyzer AI\\transfermarkt dataset\\competitions.csv")
transfers = pd.read_csv("Football player analyzer AI\\transfermarkt dataset\\transfers.csv")

# ======================================================
# 2️⃣ KEEP ONLY REQUIRED COLUMNS
# ======================================================

players = players[[
    "player_id",
    "name",
    "date_of_birth",
    "position",
    "sub_position"
]]

valuations = valuations[[
    "player_id",
    "date",
    "market_value_in_eur"
]]

appearances = appearances[[
    "player_id",
    "game_id",
    "minutes_played",
    "goals",
    "assists",
    "yellow_cards",
    "red_cards"
]]

games = games[[
    "game_id",
    "season",
    "competition_id",
    "date"
]]

competitions = competitions[[
    "competition_id",
    "name"
]]

transfers = transfers[[
    "player_id",
    "transfer_date",
    "transfer_fee"
]]

# ======================================================
# 3️⃣ MERGE APPEARANCES WITH GAME INFO
# ======================================================

performance = appearances.merge(games, on="game_id", how="left")

# Optional: merge competition if league strength feature needed
performance = performance.merge(competitions, on="competition_id", how="left")

# ======================================================
# 4️⃣ AGGREGATE PERFORMANCE PER PLAYER PER SEASON
# ======================================================

season_stats = performance.groupby(
    ["player_id", "season"]
).agg({
    "minutes_played": "sum",
    "goals": "sum",
    "assists": "sum",
    "yellow_cards": "sum",
    "red_cards": "sum"
}).reset_index()

# ======================================================
# 5️⃣ CORRECT FOOTBALL SEASON MAPPING FOR VALUATIONS
# ======================================================

valuations["date"] = pd.to_datetime(valuations["date"])

def map_to_season(date):
    # European football season logic:
    # July–Dec → season = year
    # Jan–June → season = year - 1
    if date.month >= 7:
        return date.year
    else:
        return date.year - 1

valuations["season"] = valuations["date"].apply(map_to_season)

# Keep latest valuation per player per season
valuations = valuations.sort_values("date")

valuations = valuations.groupby(
    ["player_id", "season"]
).tail(1)

valuations = valuations[[
    "player_id",
    "season",
    "market_value_in_eur"
]]

# ======================================================
# 6️⃣ MERGE PERFORMANCE WITH MARKET VALUE (TARGET)
# ======================================================

data = season_stats.merge(
    valuations,
    on=["player_id", "season"],
    how="inner"
)

# ======================================================
# 7️⃣ ADD PLAYER INFO
# ======================================================

players["date_of_birth"] = pd.to_datetime(players["date_of_birth"])

data = data.merge(players, on="player_id", how="left")

# Create age feature
data["age"] = data["season"] - data["date_of_birth"].dt.year

# ======================================================
# 8️⃣ ADD TRANSFER FEATURES (OPTIONAL BUT IMPORTANT)
# ======================================================

transfers["transfer_date"] = pd.to_datetime(transfers["transfer_date"])
transfers["season"] = transfers["transfer_date"].apply(map_to_season)

transfer_features = transfers.groupby(
    ["player_id", "season"]
).agg({
    "transfer_fee": "sum"
}).reset_index()

data = data.merge(
    transfer_features,
    on=["player_id", "season"],
    how="left"
)

data["transfer_fee"] = data["transfer_fee"].fillna(0)

# ======================================================
# 9️⃣ FINAL CLEANING
# ======================================================

# Remove rows with missing essential values
data = data.dropna(subset=["market_value_in_eur", "age"])

# Remove unrealistic ages
data = data[(data["age"] >= 15) & (data["age"] <= 45)]

# Reset index
data = data.reset_index(drop=True)

# ======================================================
# 🔟 FINAL OUTPUT
# ======================================================

print("Final dataset shape:", data.shape)
print(data.head())

data.to_csv("market_values.csv", index=False)
