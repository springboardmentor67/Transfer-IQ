import os
import json
import pandas as pd
from collections import defaultdict

events_path = "data/statsbomb/data/events"

player_playtime = defaultdict(lambda: {
    "matches_played": 0,
    "total_minutes": 0,
    "sub_on": 0,
    "sub_off": 0
})

print("Calculating playtime metrics...")

for file in os.listdir(events_path):
    if not file.endswith(".json"):
        continue

    match_players = {}
    match_max_minute = {}

    with open(os.path.join(events_path, file), "r", encoding="utf-8") as f:
        events = json.load(f)

    for event in events:

        if "player" not in event:
            continue

        name = event["player"]["name"]
        minute = event.get("minute", 0)

        match_players[name] = True
        match_max_minute[name] = max(match_max_minute.get(name, 0), minute)

        if event.get("type", {}).get("name") == "Substitution":
            if event.get("substitution", {}).get("replacement"):
                player_playtime[name]["sub_off"] += 1
                replacement = event["substitution"]["replacement"]["name"]
                player_playtime[replacement]["sub_on"] += 1

    for name in match_players:
        player_playtime[name]["matches_played"] += 1
        player_playtime[name]["total_minutes"] += match_max_minute.get(name, 0)

print("Building dataframe...")

df = pd.DataFrame.from_dict(player_playtime, orient="index").reset_index()
df.rename(columns={"index": "player_name"}, inplace=True)

df["minutes_per_match"] = df["total_minutes"] / df["matches_played"].replace(0, 1)
df["early_sub_rate"] = df["sub_off"] / df["matches_played"].replace(0, 1)

df.to_csv("outputs/player_playtime_master.csv", index=False)

print("DONE. Playtime dataset created.")