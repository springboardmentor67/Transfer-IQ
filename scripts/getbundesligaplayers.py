import requests
import time
import pandas as pd
from bs4 import BeautifulSoup

BASE_URL = "https://www.transfermarkt.com"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept-Language": "en-US,en;q=0.9"
}

# Bundesliga 2023/24 page
league_url = "https://www.transfermarkt.com/1-bundesliga/startseite/wettbewerb/L1/plus/?saison_id=2023"

print("Fetching Bundesliga clubs...")

response = requests.get(league_url, headers=HEADERS)
soup = BeautifulSoup(response.text, "html.parser")

club_links = []

# Get club links
table = soup.find("table", class_="items")
rows = table.find_all("tr", class_=["odd", "even"])

for row in rows:
    link = row.find("a", href=True)
    if link and "/startseite/verein/" in link["href"]:
        club_links.append(BASE_URL + link["href"])

club_links = list(set(club_links))

print("Total clubs found:", len(club_links))

players_data = []

# Visit each club squad page
for club_url in club_links:
    print("Processing club:", club_url)
    time.sleep(2)

    response = requests.get(club_url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")

    squad_table = soup.find("table", class_="items")

    if not squad_table:
        continue

    rows = squad_table.find_all("tr", class_=["odd", "even"])

    for row in rows:
        name_cell = row.find("td", class_="hauptlink")

        if name_cell:
            player_link = name_cell.find("a", href=True)
            if player_link:
                player_name = player_link.text.strip()
                profile_url = BASE_URL + player_link["href"]

                players_data.append({
                    "player.name": player_name,
                    "profile_url": profile_url
                })

players_df = pd.DataFrame(players_data)

players_df.drop_duplicates(inplace=True)

players_df.to_csv("outputs/bundesligaplayers.csv", index=False, encoding="utf-8-sig")

print("Bundesliga player profile dataset created.")
print("Total players collected:", players_df.shape[0])
