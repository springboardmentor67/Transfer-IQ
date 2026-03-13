import pandas as pd
import requests
import time
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from xml.etree import ElementTree as ET

# -----------------------------
# LOAD PLAYERS
# -----------------------------

players_df = pd.read_csv("outputs/playerperformance.csv")

player_names = players_df["player.name"].unique()

analyzer = SentimentIntensityAnalyzer()

all_sentiment_data = []

# -----------------------------
# LOOP THROUGH PLAYERS
# -----------------------------

for name in player_names:
    print(f"Fetching news for: {name}")

    query = name.replace(" ", "+")
    rss_url = f"https://news.google.com/rss/search?q={query}+football&hl=en-US&gl=US&ceid=US:en"

    try:
        response = requests.get(rss_url, timeout=10)
        root = ET.fromstring(response.content)

        headlines = []

        for item in root.findall(".//item")[:10]:
            title = item.find("title")
            if title is not None:
                headlines.append(title.text)

        if len(headlines) == 0:
            continue

        sentiment_scores = []
        pos = 0
        neu = 0
        neg = 0

        for headline in headlines:
            score = analyzer.polarity_scores(headline)
            compound = score["compound"]
            sentiment_scores.append(compound)

            if compound >= 0.05:
                pos += 1
            elif compound <= -0.05:
                neg += 1
            else:
                neu += 1

        # -----------------------------
        # CALCULATE METRICS
        # -----------------------------

        avg_sentiment = np.mean(sentiment_scores)
        sentiment_std = np.std(sentiment_scores)
        hype_score = avg_sentiment * len(headlines)

        if avg_sentiment >= 0.05:
            sentiment_label = "Positive"
        elif avg_sentiment <= -0.05:
            sentiment_label = "Negative"
        else:
            sentiment_label = "Neutral"

        all_sentiment_data.append({
            "player.name": name,
            "avg_sentiment": avg_sentiment,
            "sentiment_std": sentiment_std,
            "hype_score": hype_score,
            "positive_ratio": pos / len(headlines),
            "neutral_ratio": neu / len(headlines),
            "negative_ratio": neg / len(headlines),
            "headline_count": len(headlines),
            "sentiment_label": sentiment_label
        })

        time.sleep(1)

    except Exception as e:
        print(f"Error with {name}: {e}")
        continue

# -----------------------------
# SAVE DATASET
# -----------------------------

sentiment_df = pd.DataFrame(all_sentiment_data)

sentiment_df.to_csv(
    "outputs/playersentiment.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nSentiment dataset created successfully.")
print("Total players with sentiment:", sentiment_df.shape[0])
