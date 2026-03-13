import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import random

print("Loading dataset...")

df = pd.read_csv("outputs/player_master_with_injury.csv")

analyzer = SentimentIntensityAnalyzer()

print("Generating synthetic headlines...")

positive_templates = [
    "{} shines in crucial performance",
    "{} delivers outstanding display",
    "{} dominates midfield battle",
    "{} scores decisive goal",
    "{} leads team to victory"
]

negative_templates = [
    "{} struggles in recent match",
    "{} underperforms again",
    "{} criticized after poor showing",
    "{} fails to convert key chance",
    "{} involved in costly mistake"
]

neutral_templates = [
    "{} participates in match",
    "{} starts in lineup",
    "{} substituted in second half",
    "{} included in squad selection",
    "{} plays full match"
]

sentiment_scores = []
positive_ratio = []
negative_ratio = []
neutral_ratio = []

for player in df["player_name"]:
    
    headlines = []
    
    # Generate 10 headlines per player
    for _ in range(10):
        sentiment_type = random.choice(["pos", "neg", "neu"])
        
        if sentiment_type == "pos":
            headlines.append(random.choice(positive_templates).format(player))
        elif sentiment_type == "neg":
            headlines.append(random.choice(negative_templates).format(player))
        else:
            headlines.append(random.choice(neutral_templates).format(player))
    
    pos_count = 0
    neg_count = 0
    neu_count = 0
    compound_total = 0
    
    for headline in headlines:
        score = analyzer.polarity_scores(headline)
        compound_total += score["compound"]
        
        if score["compound"] > 0.05:
            pos_count += 1
        elif score["compound"] < -0.05:
            neg_count += 1
        else:
            neu_count += 1
    
    sentiment_scores.append(compound_total / 10)
    positive_ratio.append(pos_count / 10)
    negative_ratio.append(neg_count / 10)
    neutral_ratio.append(neu_count / 10)

df["sentiment_compound"] = sentiment_scores
df["positive_ratio"] = positive_ratio
df["negative_ratio"] = negative_ratio
df["neutral_ratio"] = neutral_ratio

# Create hype score
df["hype_score"] = (
    df["positive_ratio"] * 2 -
    df["negative_ratio"] +
    df["sentiment_compound"]
)

print("Sentiment features created.")

df.to_csv("outputs/player_master_full_features.csv", index=False)

print("DONE. Full dataset with sentiment created.")