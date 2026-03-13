import pandas as pd
import json
import os
import numpy as np
import logging
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_raw_tweets(filepath):
    """Loads raw tweets from a JSONL file."""
    logging.info(f"Loading raw tweets from {filepath}...")
    tweets = []
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return pd.DataFrame()

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                if line.strip():
                    tweets.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    
    if not tweets:
        logging.warning("No valid tweets found in file.")
        return pd.DataFrame()

    df = pd.DataFrame(tweets)
    logging.info(f"Loaded {len(df)} tweets.")
    return df

def calculate_sentiment_and_impact(df):
    """Calculates VADER, TextBlob scores, popularity, and sentiment impact."""
    if df.empty:
        return df

    logging.info("Calculating sentiment and impact metrics...")
    analyzer = SentimentIntensityAnalyzer()
    
    # 1. Sentiment Scores
    df['vader_compound'] = df['text'].apply(lambda x: analyzer.polarity_scores(str(x))['compound'] if pd.notnull(x) else 0)
    df['textblob_polarity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else 0)
    
    # 2. Popularity Score (Log-normalized engagement)
    # Weights based on typical platform engagement value
    # Like=1, Reply=2, Retweet=3, Quote=3, Impression=0.01 (Impression is high volume, low value)
    df['engagement_score'] = (
        df.get('like_count', 0) * 1 +
        df.get('reply_count', 0) * 2 +
        df.get('retweet_count', 0) * 3 +
        df.get('quote_count', 0) * 3 +
        df.get('impression_count', 0) * 0.01
    )
    df['popularity_score'] = np.log1p(df['engagement_score'])
    
    # 3. Sentiment Impact
    # Impact = Sentiment Strength * Popularity
    # We use absolute sentiment for magnitude of impact, and signed for direction
    df['sentiment_impact_signed'] = df['vader_compound'] * df['popularity_score']
    df['sentiment_impact_magnitude'] = df['vader_compound'].abs() * df['popularity_score']
    
    return df

def aggregate_player_stats(df):
    """Aggregates metrics by player."""
    if df.empty:
        return pd.DataFrame()

    logging.info("Aggregating stats by player...")
    
    agg_funcs = {
        'vader_compound': ['mean', 'std', 'min', 'max'],
        'textblob_polarity': ['mean'],
        'popularity_score': ['mean', 'sum', 'max'],
        'sentiment_impact_signed': ['mean', 'sum'],
        'sentiment_impact_magnitude': ['mean', 'sum'],
        'tweet_id': 'count'
    }
    
    # Group by player
    player_stats = df.groupby('player_name').agg(agg_funcs)
    
    # Flatten MultiIndex columns
    # Example: ('vader_compound', 'mean') -> 'vader_compound_mean'
    player_stats.columns = ['_'.join(col).strip() for col in player_stats.columns.values]
    player_stats.reset_index(inplace=True)
    
    # Rename for clarity
    player_stats.rename(columns={'tweet_id_count': 'tweet_volume'}, inplace=True)
    
    # Fill NaN (e.g. std dev for single tweet)
    player_stats.fillna(0, inplace=True)
    
    return player_stats

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, "data", "raw", "twitter_sentiment_raw.csv", "tweets.jsonl")
    output_file = os.path.join(base_dir, "data", "processed", "player_sentiment_features.csv")
    
    # Execute pipeline
    df = load_raw_tweets(input_file)
    if not df.empty:
        df = calculate_sentiment_and_impact(df)
        player_features = aggregate_player_stats(df)
        
        # Save results
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        player_features.to_csv(output_file, index=False)
        logging.info(f"Successfully saved aggregated sentiment features to: {output_file}")
        
        # Preview
        print("\nTop 5 Players by Sentiment Impact (Sum):")
        print(player_features.sort_values('sentiment_impact_signed_sum', ascending=False).head(5)[['player_name', 'sentiment_impact_signed_sum', 'tweet_volume']])

if __name__ == "__main__":
    main()
