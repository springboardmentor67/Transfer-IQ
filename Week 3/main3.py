import pandas as pd
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

random.seed(42)

df = pd.read_csv("player_transfer_value_dataset_final.csv")
print(f"Loaded dataset: {df.shape[0]} rows × {df.shape[1]} columns")

for col in ["positive_tweets", "negative_tweets", "total_tweets",
            "goals", "assists", "matches", "minutes_played"]:
    df[col] = df[col].fillna(0).astype(int)



POSITIVE_TEMPLATES = [
    "{name} has been absolutely incredible this season! 🔥",
    "What a performance from {name}! {goals} goals and {assists} assists. Brilliant! ⚽",
    "{name} is on fire! Best player in the squad right now 🐐",
    "Can't believe how good {name} has been. {goals} goals this season!",
    "{name} is world class. No debate. 👏",
    "Love watching {name} play. Pure quality every single match! 🙌",
    "{name} with another masterclass tonight. Sensational player!",
    "The form of {name} this season is just unreal. {goals} goals! 🔥",
    "{name} deserves so much more recognition. Absolutely top class.",
    "If {name} keeps this up ({goals} goals, {assists} assists) he'll be unstoppable!",
]

NEGATIVE_TEMPLATES = [
    "{name} was absolutely terrible today. Very disappointing performance 😤",
    "Can't believe how bad {name} has been. Not worth the money at all.",
    "{name} keeps missing chances. {goals} goals is just not good enough.",
    "Poor display from {name} again. Needs to do much better than this.",
    "Why is {name} even starting? Awful performance tonight 😡",
    "{name} looks completely out of form. Really worrying.",
    "Frustrated with {name} this season. Not delivering at all.",
    "{name} had a nightmare game. Needs to step up massively.",
    "Honestly disappointed in {name}. Expected so much more.",
    "{name} is struggling badly. Only {goals} goals all season? Come on.",
]

NEUTRAL_TEMPLATES = [
    "{name} played {minutes} minutes this season.",
    "Checking stats for {name} — {goals} goals, {assists} assists in {matches} games.",
    "{name} transfer rumours are heating up again.",
    "Not sure about {name} yet. Will need to see more games.",
    "{name} is an interesting player. Mixed bag of performances.",
    "Following {name}'s progress this season.",
    "{name} featured again in today's squad.",
    "Stats for {name}: {goals}G / {assists}A in {matches} appearances.",
]

def generate_tweets(row):
    """
    Generate tweet sentences per player-season row using existing
    positive_tweets / negative_tweets counts.
    Neutral tweets = total - positive - negative (min 1 each).

    FIX 1: Rows where total_tweets == 0 (players with no tweet data)
            still get at least 1 neutral tweet so VADER/TextBlob always
            has text to score — prevents NaN in aggregated columns.

    FIX 2: n_neu clamped to >= 1 so it never goes negative when
            n_pos + n_neg already exceeds total_tweets.
    """
    name    = row["player_name"].split()[0]   # first name only
    goals   = row["goals"]
    assists = row["assists"]
    matches = row["matches"]
    minutes = row["minutes_played"]

    n_pos = max(1, row["positive_tweets"])
    n_neg = max(1, row["negative_tweets"])
    n_neu = max(1, row["total_tweets"] - n_pos - n_neg)

    fmt = {"name": name, "goals": goals, "assists": assists,
           "matches": matches, "minutes": minutes}

    tweets = (
        [random.choice(POSITIVE_TEMPLATES).format(**fmt) for _ in range(n_pos)] +
        [random.choice(NEGATIVE_TEMPLATES).format(**fmt) for _ in range(n_neg)] +
        [random.choice(NEUTRAL_TEMPLATES).format(**fmt)  for _ in range(n_neu)]
    )
    return tweets

tweet_rows = []
for _, row in df.iterrows():
    for tweet in generate_tweets(row):
        tweet_rows.append({
            "player_name": row["player_name"],
            "season":      row["season"],
            "tweet_text":  tweet
        })

tweets_df = pd.DataFrame(tweet_rows)
print(f"\nGenerated {len(tweets_df):,} tweets across {len(df):,} player-season records")
print("\nSample tweets:")
print(tweets_df["tweet_text"].sample(6, random_state=1).to_string(index=False))

vader = SentimentIntensityAnalyzer()

def get_vader(text):
    s = vader.polarity_scores(str(text))
    return s["pos"], s["neg"], s["compound"]

def get_textblob(text):
    b = TextBlob(str(text))
    return b.sentiment.polarity, b.sentiment.subjectivity

print("\nRunning VADER...")
tweets_df[["vader_positive_score",
           "vader_negative_score",
           "vader_compound_score"]] = (
    tweets_df["tweet_text"].apply(lambda t: pd.Series(get_vader(t)))
)

print("Running TextBlob...")
tweets_df[["tb_polarity",
           "tb_subjectivity"]] = (
    tweets_df["tweet_text"].apply(lambda t: pd.Series(get_textblob(t)))
)

def label(c):
    if c >= 0.05:  return "Positive"
    if c <= -0.05: return "Negative"
    return "Neutral"

tweets_df["sentiment_label"] = tweets_df["vader_compound_score"].apply(label)

agg = (
    tweets_df
    .groupby(["player_name", "season"])
    .agg(
        vader_positive_score = ("vader_positive_score", "mean"),
        vader_negative_score = ("vader_negative_score", "mean"),
        vader_compound_score = ("vader_compound_score", "mean"),
        tb_polarity          = ("tb_polarity",           "mean"),
        tb_subjectivity      = ("tb_subjectivity",       "mean"),
        positive_count       = ("sentiment_label", lambda x: (x == "Positive").sum()),
        negative_count       = ("sentiment_label", lambda x: (x == "Negative").sum()),
        neutral_count        = ("sentiment_label", lambda x: (x == "Neutral").sum()),
    )
    .reset_index()
)

for col in ["vader_positive_score", "vader_negative_score", "vader_compound_score",
            "tb_polarity", "tb_subjectivity"]:
    agg[col] = agg[col].round(4)

agg["sentiment_label"] = agg["vader_compound_score"].apply(label)

COLS_TO_DROP = [
    "avg_sentiment",
    "sentiment_polarity_strength",
    "positive_tweet_ratio",
]
df_clean = df.drop(columns=[c for c in COLS_TO_DROP if c in df.columns])

final_df = df_clean.merge(agg, on=["player_name", "season"], how="left")

new_cols = ["vader_positive_score", "vader_negative_score", "vader_compound_score",
            "tb_polarity", "tb_subjectivity",
            "positive_count", "negative_count", "neutral_count"]
for col in new_cols:
    if final_df[col].dtype in ["float64", "int64"]:
        final_df[col] = final_df[col].fillna(0)
final_df["sentiment_label"] = final_df["sentiment_label"].fillna("Neutral")

final_df.to_csv("player_transfer_value_with_sentiment.csv", index=False)

print("\n" + "="*55)
print("  SENTIMENT ANALYSIS SUMMARY")
print("="*55)
print(f"  Total tweets analysed  : {len(tweets_df):,}")
print(f"  Positive               : {(tweets_df['sentiment_label']=='Positive').sum():,} "
      f"({(tweets_df['sentiment_label']=='Positive').mean()*100:.1f}%)")
print(f"  Negative               : {(tweets_df['sentiment_label']=='Negative').sum():,} "
      f"({(tweets_df['sentiment_label']=='Negative').mean()*100:.1f}%)")
print(f"  Neutral                : {(tweets_df['sentiment_label']=='Neutral').sum():,} "
      f"({(tweets_df['sentiment_label']=='Neutral').mean()*100:.1f}%)")
print(f"  Avg VADER compound     : {tweets_df['vader_compound_score'].mean():.4f}")
print(f"  Avg TextBlob polarity  : {tweets_df['tb_polarity'].mean():.4f}")
print(f"\n  Output shape           : {final_df.shape[0]:,} rows × {final_df.shape[1]} columns")
print("="*55)
print("\n✅ Saved: player_transfer_value_with_sentiment.csv")
print("\nSample output (per player-season):")
print(agg[["player_name", "season",
           "vader_positive_score", "vader_negative_score",
           "vader_compound_score", "tb_polarity",
           "sentiment_label"]].head(10).to_string(index=False))