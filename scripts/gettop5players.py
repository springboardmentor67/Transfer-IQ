import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

headers = {
    "User-Agent": "Mozilla/5.0"
}

# Top 5 league IDs
leagues = {
    "Bundesliga": "L1",
    "Premier League": "GB1",
    "La Liga": "ES1",
    "Serie A": "IT1",
    "Ligue 1": "FR1"
}

season = "2023"

all_players = []

for league_name, league_code in leagues.items():
    print(f"\nFetching league: {league_name}")

    league_url = f"https://www.transfermarkt.com/wettbewerb/startseite/wettbewerb/{league_code}/saison_id/{season}"

    response = requests.get(league_url, headers=headers)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find club links
    club_links = soup.select("td.no-border-links a")

    clubs = []
    for link in club_links:
        href = link.get("href")
        if href and "/startseite/verein/" in href:
            full_link = "https://www.transfermarkt.com" + href
            clubs.append(full_link)

    clubs = list(set(clubs))  # remove duplicates
    print(f"Total clubs found in {league_name}: {len(clubs)}")

    for club_url in clubs:
        print(f"Processing club: {club_url}")
        time.sleep(1)

        club_response = requests.get(club_url, headers=headers)
        club_soup = BeautifulSoup(club_response.content, "html.parser")

        # Extract players
        player_rows = club_soup.select("table.items tbody tr")

        for row in player_rows:
            name_tag = row.select_one("td.hauptlink a")
            position_tag = row.select_one("td.zentriert:nth-of-type(2)")
            age_tag = row.select_one("td.zentriert:nth-of-type(3)")

            if name_tag:
                player_name = name_tag.text.strip()
                profile_link = "https://www.transfermarkt.com" + name_tag.get("href")

                position = position_tag.text.strip() if position_tag else None
                age = age_tag.text.strip() if age_tag else None

                all_players.append({
                    "player.name": player_name,
                    "position": position,
                    "age": age,
                    "profile_url": profile_link,
                    "league": league_name
                })

print("\nTotal players collected:", len(all_players))

df = pd.DataFrame(all_players)
df = df.drop_duplicates(subset=["player.name"])

df.to_csv("outputs/top5_player_profiles.csv", index=False)

print("Top 5 league player dataset created successfully.")