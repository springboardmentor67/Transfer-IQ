import pandas as pd
import json

path = "data/statsbomb/data/matches/9/281.json"

with open(path, "r", encoding="utf-8") as f:
    matches = json.load(f)

matches_df = pd.json_normalize(matches)

print("Total Matches:", matches_df.shape[0])
print(matches_df[["match_id", "home_team.home_team_name", "away_team.away_team_name"]].head())
