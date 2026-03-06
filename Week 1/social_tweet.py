import pandas as pd
import re
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('vader_lexicon', quiet=True)

vader = SentimentIntensityAnalyzer()

PLAYERS = [
    "Lionel Messi", "Kylian Mbappe", "Cristiano Ronaldo", "Neymar",
    "Luka Modric", "Karim Benzema", "Vinicius Jr", "Harry Kane",
    "Bukayo Saka", "Jude Bellingham", "Phil Foden", "Bruno Fernandes",
    "Kevin De Bruyne", "Mohamed Salah", "Erling Haaland", "Pedri",
    "Gavi", "Rodrygo", "Alphonso Davies", "Thibaut Courtois",
    "Emiliano Martinez", "Achraf Hakimi", "Joao Felix", "Rafael Leao"
]

def clean_text(text):
    """Remove links, emojis, mentions"""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)  # Remove URLs
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#', '', text)  # Remove # but keep text
    text = ' '.join(text.split())  # Remove extra spaces
    return text.strip()

def get_sentiment(text):
    """Get sentiment: positive, negative, or neutral"""
    if not text:
        return 'neutral'
    
    score = vader.polarity_scores(text)['compound']
    
    if score >= 0.05:
        return 'positive'
    elif score <= -0.05:
        return 'negative'
    else:
        return 'neutral'

def find_players(text):
    """Find which players are mentioned"""
    if not text:
        return []
    
    text_lower = text.lower()
    found = []
    
    for player in PLAYERS:
        if player.lower() in text_lower:
            found.append(player)
    
    return found

file_path = "dataset/fifa_world_cup_2022_tweets.csv"

print("Loading FIFA World Cup 2022 tweets...")
df = pd.read_csv(file_path, encoding='utf-8')

text_col = None
for col in ['Tweet', 'text', 'tweet', 'content', 'Text']:
    if col in df.columns:
        text_col = col
        break

if text_col is None:
    text_col = df.columns[0]

print(f"Using column: '{text_col}'")
print(f"Processing {len(df)} tweets...\n")

df['cleaned_text'] = df[text_col].apply(clean_text)
df['sentiment'] = df['cleaned_text'].apply(get_sentiment)
df['mentioned_players'] = df['cleaned_text'].apply(find_players)

df_players = df[df['mentioned_players'].apply(len) > 0].copy()

print(f"Found {len(df_players)} tweets mentioning players!\n")

print("="*80)
print("TWEETS MENTIONING PLAYERS")
print("="*80)

for idx, row in df_players.head(20).iterrows():
    print(f"\nOriginal: {row[text_col][:100]}...")
    print(f"Cleaned:  {row['cleaned_text'][:100]}")
    print(f"Players:  {', '.join(row['mentioned_players'])}")
    print(f"Sentiment: {row['sentiment'].upper()}")
    print("-"*80)

df_players.to_csv('output/player_tweets.csv', index=False)
print(f"\n✓ Saved {len(df_players)} tweets to output/player_tweets.csv")

print("\n" + "="*80)
print("PLAYER MENTION STATISTICS")
print("="*80)

all_players = []
for players_list in df_players['mentioned_players']:
    all_players.extend(players_list)

player_counts = pd.Series(all_players).value_counts()

print(f"\nTop 10 Most Mentioned Players:")
for player, count in player_counts.head(10).items():
    print(f"  {player}: {count} tweets")