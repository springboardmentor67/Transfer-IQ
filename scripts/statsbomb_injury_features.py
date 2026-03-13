import os
import json
import pandas as pd

events_path = "data/statsbomb/data/events"

injury_data = {}

print("Scanning event files...")

for file in os.listdir(events_path):

    if not file.endswith(".json"):
        continue

    file_path = os.path.join(events_path, file)

    with open(file_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    for event in events:

        if "player" not in event:
            continue

        player_name = event["player"]["name"]

        if player_name not in injury_data:
            injury_data[player_name] = {
                "injury_event_count": 0,
                "injury_sub_count": 0
            }

        # Check for injury-type event
        if event.get("type", {}).get("name") == "Injury":
            injury_data[player_name]["injury_event_count"] += 1

        # Check substitution events
        if event.get("type", {}).get("name") == "Substitution":
            if "injury" in str(event).lower():
                injury_data[player_name]["injury_sub_count"] += 1

print("Aggregating results...")

injury_df = pd.DataFrame.from_dict(injury_data, orient="index").reset_index()
injury_df.rename(columns={"index": "player.name"}, inplace=True)

# Create injury risk score
injury_df["injury_risk_score"] = (
    injury_df["injury_event_count"] * 2 +
    injury_df["injury_sub_count"] * 3
)

injury_df.to_csv("outputs/statsbomb_injury_features.csv", index=False)

print("DONE. Injury features saved.")