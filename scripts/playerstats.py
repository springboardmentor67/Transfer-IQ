import pandas as pd
import json

# -------------------------
# LOAD MATCHES
# -------------------------

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

# Remove rows without player
full_events = full_events[full_events["player.name"].notna()]

# -------------------------
# BASIC STAT FILTERS
# -------------------------

passes = full_events[full_events["type.name"] == "Pass"]
goals = full_events[
    (full_events["type.name"] == "Shot") &
    (full_events["shot.outcome.name"] == "Goal")
]
shots = full_events[full_events["type.name"] == "Shot"]
duels = full_events[full_events["type.name"] == "Duel"]
interceptions = full_events[full_events["type.name"] == "Interception"]
dribbles = full_events[full_events["type.name"] == "Dribble"]
recoveries = full_events[full_events["type.name"] == "Ball Recovery"]

assists = full_events[
    (full_events["type.name"] == "Pass") &
    (full_events["pass.goal_assist"] == True)
]

# -------------------------
# GROUP FUNCTION (WITH TEAM)
# -------------------------

def count_stat(df, name):
    return (
        df.groupby(["player.name", "team.name"])
        .size()
        .reset_index(name=name)
    )

passes_count = count_stat(passes, "passes")
goals_count = count_stat(goals, "goals")
shots_count = count_stat(shots, "shots")
duels_count = count_stat(duels, "duels")
interceptions_count = count_stat(interceptions, "interceptions")
dribbles_count = count_stat(dribbles, "dribbles")
recoveries_count = count_stat(recoveries, "ball_recoveries")
assists_count = count_stat(assists, "assists")

# -------------------------
# MATCHES PLAYED
# -------------------------

matches_played = (
    full_events.groupby(["player.name", "team.name"])["match_id"]
    .nunique()
    .reset_index(name="matches_played")
)

# -------------------------
# MINUTES PLAYED
# -------------------------

minutes_played = (
    full_events.groupby(["player.name", "team.name", "match_id"])["minute"]
    .max()
    .reset_index()
)

minutes_played_total = (
    minutes_played.groupby(["player.name", "team.name"])["minute"]
    .sum()
    .reset_index(name="minutes_played")
)

# -------------------------
# MERGING EVERYTHING
# -------------------------

player_stats = passes_count

for df in [
    goals_count,
    shots_count,
    duels_count,
    interceptions_count,
    dribbles_count,
    recoveries_count,
    assists_count,
    matches_played,
    minutes_played_total
]:
    player_stats = player_stats.merge(
        df,
        on=["player.name", "team.name"],
        how="left"
    )

player_stats = player_stats.fillna(0)

# -------------------------
# PER 90 METRICS
# -------------------------

per90_cols = [
    "goals", "assists", "shots", "passes",
    "interceptions", "dribbles", "ball_recoveries", "duels"
]

for col in per90_cols:
    player_stats[f"{col}_per90"] = (
        player_stats[col] / player_stats["minutes_played"]
    ) * 90

player_stats = player_stats.fillna(0)

# -------------------------
# FILTER SERIOUS PLAYERS
# -------------------------

serious_players = player_stats

print("Serious Players Count:", serious_players.shape[0])

# -------------------------
# SAVE FILES
# -------------------------

player_stats.to_csv("outputs/playerperformance.csv", index=False, encoding="utf-8")
serious_players.to_csv("outputs/seriousplayers.csv", index=False, encoding="utf-8")

print(player_stats.head())
print("Total Players:", player_stats.shape[0])
print("Complete player performance dataset created.")
