import time
import random
import requests
import pandas as pd

PLAYERS = {
    8198:   "Cristiano Ronaldo",
    28003:  "Lionel Messi",
    418560: "Erling Haaland",
    342229: "Kylian Mbappe",
    581977: "Jude Bellingham",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json"
}

def get_market_value_history(player_id, player_name):
    url = f"https://tmapi-alpha.transfermarkt.technology/player/{player_id}/market-value-history"
    response = requests.get(url, headers=HEADERS, timeout=15)
    response.raise_for_status()
    history = response.json()["data"]["history"]
    records = []
    for item in history:
        records.append({
            "player_name":      player_name,
            "player_id":        item["playerId"],
            "club_id":          item["clubId"],
            "age":              item["age"],
            "date":             item["marketValue"]["determined"],
            "market_value_eur": item["marketValue"]["value"],
        })
    return records

all_records = []

for i, (pid, name) in enumerate(PLAYERS.items(), 1):
    print(f"[{i}/{len(PLAYERS)}] Fetching {name}...")
    try:
        records = get_market_value_history(pid, name)
        all_records.extend(records)
        print(f"  ✓ {len(records)} entries")
    except Exception as e:
        print(f"  ✗ Failed: {e}")
    if i < len(PLAYERS):
        time.sleep(random.uniform(2, 4))

df = pd.DataFrame(all_records)
df.to_csv("output/market_values.csv", index=False)
print(f"\nSaved {len(df)} rows to output/market_values.csv")
print(df.head(10).to_string(index=False))