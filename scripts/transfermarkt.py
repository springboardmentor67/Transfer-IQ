import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

headers = {
    "User-Agent": "Mozilla/5.0"
}

# Bundesliga 2023/24 page
league_url = "https://www.transfermarkt.com/1-bundesliga/startseite/wettbewerb/L1/plus/?saison_id=2023"

response = requests.get(league_url, headers=headers)
soup = BeautifulSoup(response.text, "lxml")

teams = []

# Extract team links
for link in soup.select("td.hauptlink a"):
    href = link.get("href")
    if href and "/verein/" in href:
        full_link = "https://www.transfermarkt.com" + href
        teams.append(full_link)

teams = list(set(teams))

print("Total teams found:", len(teams))

all_data = []

for team_url in teams:
    print("Scraping:", team_url)

    team_page = requests.get(team_url, headers=headers)
    team_soup = BeautifulSoup(team_page.text, "lxml")

    team_name_tag = team_soup.select_one("h1")
    team_name = team_name_tag.text.strip() if team_name_tag else "Unknown"

    # Only real player rows
    rows = team_soup.select("table.items tbody tr.odd, table.items tbody tr.even")

    for row in rows:
        name_tag = row.select_one("td.hauptlink a")
        value_tag = row.select_one("td.rechts.hauptlink")

        position_tag = row.select_one("td:nth-of-type(2)")
        age_cells = row.select("td.zentriert")
        age_tag = age_cells[1] if len(age_cells) > 1 else None

        if name_tag and value_tag:
            player_name = name_tag.text.strip()
            market_value_raw = value_tag.text.strip()
            position = position_tag.text.strip() if position_tag else ""
            age = age_tag.text.strip() if age_tag else ""

            # Clean numeric market value
            market_value_eur = None
            if "m" in market_value_raw.lower():
                market_value_eur = float(
                    market_value_raw.replace("€", "").replace("m", "")
                ) * 1_000_000
            elif "k" in market_value_raw.lower():
                market_value_eur = float(
                    market_value_raw.replace("€", "").replace("k", "")
                ) * 1_000

            all_data.append({
                "player.name": player_name,
                "team.name": team_name,
                "position": position,
                "age": age,
                "market_value_raw": market_value_raw,
                "market_value_eur": market_value_eur
            })

    time.sleep(2)

market_df = pd.DataFrame(all_data)

market_df.to_csv(
    "outputs/marketvalues.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Bundesliga market values scraped successfully.")
print("Total players scraped:", market_df.shape[0])
