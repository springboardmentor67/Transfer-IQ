import pandas as pd
import json
import os

# Load competitions
competitions = pd.read_json("data/statsbomb/data/competitions.json")

# Select Bundesliga 2023/24
bundesliga = competitions[
    (competitions["competition_id"] == 9) &
    (competitions["season_name"] == "2023/2024")
]

print("Selected Competition:")
print(bundesliga)
