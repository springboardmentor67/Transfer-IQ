import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import os
import re

headers = {
    "User-Agent": "Mozilla/5.0"
}

# Load player profiles
players_df = pd.read_csv("outputs/top5_player_profiles.csv")

# ---- TEST MODE (only first 10 players) ----
players_df = players_df.head(10)

output_path = "outputs/playerinjuries_top5.csv"

results = []

for index, player in players_df.iterrows():

    name = player["player.name"]
    profile_url = player["profile_url"]

    print(f"{index+1}/{len(players_df)} Processing: {name}")

    try:
        # FULL CAREER injury page
        injury_url = profile_url.replace("/profil/", "/verletzungen/") + "/plus/1"

        r = requests.get(injury_url, headers=headers, timeout=15)
        with open("debug_page.html", "w", encoding="utf-8") as f:
             f.write(r.text)
             print("Saved debug_page.html")
             break
        soup = BeautifulSoup(r.text, "html.parser")

        total_injuries = 0
        total_days = 0
        matches_missed = 0

        # Flexible table selector
        table = soup.find("table", {"class": lambda x: x and "items" in x})

        if table:
            rows = table.find_all("tr")

            for row in rows:
                cols = row.find_all("td")

                if len(cols) >= 7:
                    total_injuries += 1

                    days_text = cols[5].get_text(strip=True)
                    matches_text = cols[6].get_text(strip=True)

                    days = int(re.sub(r"\D", "", days_text)) if re.search(r"\d", days_text) else 0
                    matches = int(re.sub(r"\D", "", matches_text)) if re.search(r"\d", matches_text) else 0

                    total_days += days
                    matches_missed += matches

        injury_severity = total_days / total_injuries if total_injuries > 0 else 0

        results.append({
            "player.name": name,
            "total_injuries": total_injuries,
            "total_days_injured": total_days,
            "matches_missed": matches_missed,
            "injury_severity_score": round(injury_severity, 2),
            "currently_injured": 0
        })

        time.sleep(0.5)

    except Exception as e:
        print("Error:", name)
        print(e)

# Save results
pd.DataFrame(results).to_csv(output_path, index=False)

print("DONE.")