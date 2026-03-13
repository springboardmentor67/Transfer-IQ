import pandas as pd
import requests
import time
import re
from bs4 import BeautifulSoup

BASE_URL = "https://www.transfermarkt.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9"
}

# ----------------------------
# LOAD DATA
# ----------------------------

serious_df = pd.read_csv("outputs/seriousplayers.csv")
bundesliga_df = pd.read_csv("outputs/bundesligaplayers.csv")

# Clean names (lowercase match)
serious_df["player.name_clean"] = serious_df["player.name"].str.lower().str.strip()
bundesliga_df["player.name_clean"] = bundesliga_df["player.name"].str.lower().str.strip()

# Merge to get profile URLs
merged_df = serious_df.merge(
    bundesliga_df[["player.name_clean", "profile_url"]],
    on="player.name_clean",
    how="left"
)

results = []

# ----------------------------
# SCRAPE INJURY DATA
# ----------------------------

for index, row in merged_df.iterrows():
    name = row["player.name"]
    profile_url = row["profile_url"]

    print(f"Processing: {name}")

    if pd.isna(profile_url):
        print("No profile URL found")
        results.append([name, 0, 0, 0, 0, 0])
        continue

    try:
        injury_url = profile_url.replace("profil", "verletzungen")
        response = requests.get(injury_url, headers=HEADERS)
        soup = BeautifulSoup(response.text, "html.parser")

        table = soup.find("table", class_="items")

        if not table:
            results.append([name, 0, 0, 0, 0, 0])
            continue

        rows = table.find_all("tr", class_=["odd", "even"])

        total_injuries = 0
        total_days = 0
        matches_missed = 0

        for r in rows:
            cols = r.find_all("td")

            if len(cols) < 6:
                continue

            total_injuries += 1

            # Correct columns
            days_text = cols[4].text.strip()
            days = int(re.sub(r"[^\d]", "", days_text)) if re.sub(r"[^\d]", "", days_text) else 0

            missed_text = cols[5].text.strip()
            missed = int(re.sub(r"[^\d]", "", missed_text)) if re.sub(r"[^\d]", "", missed_text) else 0

            total_days += days
            matches_missed += missed

        # Currently injured check
        currently_injured = 0
        header_box = soup.find("div", class_="data-header__details")
        if header_box and "injured" in header_box.text.lower():
            currently_injured = 1

        severity = total_days / total_injuries if total_injuries > 0 else 0

        results.append([
            name,
            total_injuries,
            total_days,
            matches_missed,
            severity,
            currently_injured
        ])

        time.sleep(2)

    except Exception as e:
        print(f"Error: {e}")
        results.append([name, 0, 0, 0, 0, 0])
        continue

# ----------------------------
# SAVE DATA
# ----------------------------

injury_df = pd.DataFrame(results, columns=[
    "player.name",
    "total_injuries",
    "total_days_injured",
    "matches_missed",
    "injury_severity_score",
    "currently_injured"
])

injury_df.to_csv("outputs/playerinjuries.csv", index=False, encoding="utf-8-sig")

print("Injury dataset created successfully.")
print("Total players processed:", injury_df.shape[0])
