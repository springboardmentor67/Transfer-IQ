import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re

headers = {
    "User-Agent": "Mozilla/5.0"
}

df = pd.read_csv("outputs/top5_player_profiles.csv")

results = []

for i, row in df.iterrows():
    name = row["player.name"]
    url = row["profile_url"]

    print(f"{i+1}/{len(df)} Processing: {name}")

    try:
        r = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(r.text, "html.parser")

        market_value = None

        # Correct market value selector
        value_box = soup.select_one("div.data-header__box--small")

        if value_box:
            value_text = value_box.get_text(strip=True)
            match = re.search(r"€([\d.,]+)(m|k)?", value_text)

            if match:
                number = match.group(1).replace(".", "").replace(",", "")
                multiplier = match.group(2)

                if multiplier == "m":
                    market_value = int(float(number) * 1_000_000)
                elif multiplier == "k":
                    market_value = int(float(number) * 1_000)
                else:
                    market_value = int(number)

        results.append({
            "player.name": name,
            "market_value_eur": market_value
        })

        time.sleep(0.4)

    except Exception as e:
        print("Error:", name)
        results.append({
            "player.name": name,
            "market_value_eur": None
        })

market_df = pd.DataFrame(results)
market_df.to_csv("outputs/marketvalues_top5.csv", index=False)

print("DONE. Market values saved.")