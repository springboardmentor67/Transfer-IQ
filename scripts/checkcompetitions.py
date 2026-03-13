import json
import pandas as pd

path = "data/statsbomb/data/competitions.json"

with open(path, "r", encoding="utf-8") as f:
    competitions = json.load(f)

df = pd.json_normalize(competitions)

print(df[["competition_id", "competition_name", "country_name", "season_name"]])
