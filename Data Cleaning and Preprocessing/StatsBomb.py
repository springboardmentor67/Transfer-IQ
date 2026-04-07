import os
import json
import pandas as pd
from collections import defaultdict

# =====================
# SET CORRECT PATHS
# =====================

EVENTS_FOLDER = r"Football player analyzer AI\open-data-master\data\events"
MATCHES_FOLDER = r"open-data-master\data\matches"
OUTPUT_FILE = "statsbomb.csv"

print("EVENTS_FOLDER:", EVENTS_FOLDER)
print("MATCHES_FOLDER:", MATCHES_FOLDER)

print("Events folder exists:", os.path.isdir(EVENTS_FOLDER))
print("Matches folder exists:", os.path.isdir(MATCHES_FOLDER))

print("Events count:", len(os.listdir(EVENTS_FOLDER)))

# =====================
# STEP 1: BUILD MATCH -> SEASON MAP
# =====================

match_season_map = {}

for root, dirs, files in os.walk(MATCHES_FOLDER):
    for file in files:
        if file.endswith(".json"):
            file_path = os.path.join(root, file)
            print("Reading match file:", file_path)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # StatsBomb stores an array of matches
            for match in data:
                mid = match.get("match_id")
                if mid is None:
                    continue

                season = match.get("season", {}).get("season_name")
                comp = match.get("competition", {}).get("competition_name")

                if season and comp:
                    match_season_map[mid] = f"{comp}_{season}"

print("Total matches mapped:", len(match_season_map))

if len(match_season_map) == 0:
    print("⚠️ No matches mapped. Check MATCHES_FOLDER path and JSON structure.")

# =====================
# STEP 2: PROCESS EVENTS
# =====================

player_season_stats = defaultdict(lambda: {
    "matches_played": 0, "goals": 0, "assists": 0,
    "shots": 0, "xg": 0.0, "passes": 0, "pass_completed": 0,
    "tackles": 0, "interceptions": 0, "dribbles_completed": 0,
    "minutes_played": 0
})

appearance_tracker = set()

event_files_processed = 0
match_ids_skipped = set()

for filename in sorted(os.listdir(EVENTS_FOLDER)):
    if not filename.endswith(".json"):
        continue

    event_files_processed += 1
    file_path = os.path.join(EVENTS_FOLDER, filename)
    match_id = int(filename.replace(".json", ""))

    with open(file_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    if match_id not in match_season_map:
        match_ids_skipped.add(match_id)
        continue

    season_label = match_season_map[match_id]
    appearance_tracker.clear()

    for event in events:
        player = event.get("player")
        if not player:
            continue

        player_name = player.get("name")
        key = (player_name, season_label)
        stats = player_season_stats[key]

        etype = event.get("type", {}).get("name", "")

        # match appearances
        appearance_tracker.add(key)

        # shots
        if etype == "Shot":
            stats["shots"] += 1
            shot_obj = event.get("shot")
            if shot_obj:
                stats["xg"] += shot_obj.get("statsbomb_xg", 0)
                if shot_obj.get("outcome", {}).get("name") == "Goal":
                    stats["goals"] += 1

        # passes & assists
        if etype == "Pass":
            stats["passes"] += 1
            if event.get("pass", {}).get("outcome") is None:
                stats["pass_completed"] += 1
            if event.get("pass", {}).get("goal_assist"):
                stats["assists"] += 1

        # tackles
        if etype == "Duel":
            if event.get("duel", {}).get("type", {}).get("name") == "Tackle":
                stats["tackles"] += 1

        # interceptions
        if etype == "Interception":
            stats["interceptions"] += 1

        # dribbles
        if etype == "Dribble":
            if event.get("dribble", {}).get("outcome", {}).get("name") == "Complete":
                stats["dribbles_completed"] += 1

        # minute (approx)
        minute = event.get("minute")
        if minute:
            stats["minutes_played"] = max(stats["minutes_played"], minute)

    for k in appearance_tracker:
        player_season_stats[k]["matches_played"] += 1

print("Event files read:", event_files_processed)
print("Matches skipped (no season info):", len(match_ids_skipped))

# =====================
# STEP 3: SAVE CSV
# =====================

rows = []
for (player_name, season_label), stats in player_season_stats.items():
    rows.append({
        "player_name": player_name,
        "season": season_label,
        **stats
    })

df = pd.DataFrame(rows)

df["pass_accuracy"] = (df["pass_completed"] / df["passes"]).fillna(0)

df = df.sort_values(["player_name", "season"])

df.to_csv("statsbomb.csv", index=False)

print("CSV saved successfully!")
print("Total records:", len(df))
