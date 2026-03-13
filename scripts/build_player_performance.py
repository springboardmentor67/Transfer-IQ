import os
import json
import pandas as pd
from collections import defaultdict

events_path = "data/statsbomb/data/events"

player_stats = defaultdict(lambda: {
    "goals": 0,
    "assists": 0,
    "shots": 0,
    "passes": 0,
    "successful_passes": 0,
    "tackles": 0,
    "interceptions": 0,
    "clearances": 0,
    "yellow_cards": 0,
    "red_cards": 0,
    "fouls": 0
})

print("Processing events...")

for file in os.listdir(events_path):
    if not file.endswith(".json"):
        continue

    with open(os.path.join(events_path, file), "r", encoding="utf-8") as f:
        events = json.load(f)

    for event in events:

        if "player" not in event:
            continue

        name = event["player"]["name"]
        event_type = event.get("type", {}).get("name")

        if event_type == "Shot":
            player_stats[name]["shots"] += 1
            if event.get("shot", {}).get("outcome", {}).get("name") == "Goal":
                player_stats[name]["goals"] += 1

        if event_type == "Pass":
            player_stats[name]["passes"] += 1
            if event.get("pass", {}).get("outcome") is None:
                player_stats[name]["successful_passes"] += 1
            if event.get("pass", {}).get("goal_assist"):
                player_stats[name]["assists"] += 1

        if event_type == "Tackle":
            player_stats[name]["tackles"] += 1

        if event_type == "Interception":
            player_stats[name]["interceptions"] += 1

        if event_type == "Clearance":
            player_stats[name]["clearances"] += 1

        if event_type == "Foul Committed":
            player_stats[name]["fouls"] += 1

        if event_type == "Bad Behaviour":
            card = event.get("bad_behaviour", {}).get("card", {}).get("name")
            if card == "Yellow Card":
                player_stats[name]["yellow_cards"] += 1
            if card == "Red Card":
                player_stats[name]["red_cards"] += 1

print("Building dataframe...")

df = pd.DataFrame.from_dict(player_stats, orient="index").reset_index()
df.rename(columns={"index": "player_name"}, inplace=True)

df["pass_accuracy"] = df["successful_passes"] / df["passes"].replace(0, 1)

df.to_csv("outputs/player_performance_master.csv", index=False)

print("DONE. Performance dataset created.")