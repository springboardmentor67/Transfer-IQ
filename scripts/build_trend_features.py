import os
import json
import pandas as pd
from collections import defaultdict

print("Building match-level performance trends...")

events_path = "data/statsbomb/data/events"

player_match_stats = []

for file in os.listdir(events_path):
    if not file.endswith(".json"):
        continue

    match_id = file.replace(".json", "")

    with open(os.path.join(events_path, file), "r", encoding="utf-8") as f:
        events = json.load(f)

    match_data = defaultdict(lambda: {
        "goals": 0,
        "assists": 0,
        "shots": 0
    })

    for event in events:

        if "player" not in event:
            continue

        name = event["player"]["name"]
        event_type = event.get("type", {}).get("name")

        if event_type == "Shot":
            match_data[name]["shots"] += 1
            if event.get("shot", {}).get("outcome", {}).get("name") == "Goal":
                match_data[name]["goals"] += 1

        if event_type == "Pass":
            if event.get("pass", {}).get("goal_assist"):
                match_data[name]["assists"] += 1

    for player, stats in match_data.items():
        player_match_stats.append({
            "player_name": player,
            "match_id": match_id,
            "goals": stats["goals"],
            "assists": stats["assists"],
            "shots": stats["shots"]
        })

df_match = pd.DataFrame(player_match_stats)

print("Calculating rolling averages...")

df_match.sort_values(["player_name", "match_id"], inplace=True)

df_match["goals_last5"] = (
    df_match.groupby("player_name")["goals"]
    .rolling(5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

df_match["assists_last5"] = (
    df_match.groupby("player_name")["assists"]
    .rolling(5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

df_match["recent_form_score"] = (
    df_match["goals_last5"] * 2 +
    df_match["assists_last5"]
)

# Take latest match per player
df_trend = df_match.groupby("player_name").tail(1)

df_trend = df_trend[[
    "player_name",
    "goals_last5",
    "assists_last5",
    "recent_form_score"
]]

df_trend.to_csv("outputs/player_trend_features.csv", index=False)

print("Trend features created.")