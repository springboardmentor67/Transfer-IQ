import os
import json

events_path = "data/statsbomb/data/events"

players = set()

for file in os.listdir(events_path):
    if not file.endswith(".json"):
        continue

    with open(os.path.join(events_path, file), "r", encoding="utf-8") as f:
        events = json.load(f)

    for event in events:
        if "player" in event:
            players.add(event["player"]["name"])

print("Total unique players:", len(players))