import pandas as pd
import json

# Load matches
with open("data/statsbomb/data/matches/9/281.json", "r", encoding="utf-8") as f:
    matches = json.load(f)

matches_df = pd.json_normalize(matches)

all_events = []

for match_id in matches_df["match_id"]:
    event_path = f"data/statsbomb/data/events/{match_id}.json"
    
    with open(event_path, "r", encoding="utf-8") as f:
        events = json.load(f)
    
    events_df = pd.json_normalize(events)
    events_df["match_id"] = match_id
    
    all_events.append(events_df)

full_events = pd.concat(all_events, ignore_index=True)

print("Total Events:", full_events.shape)
print(full_events[["player.name", "type.name"]].head())
