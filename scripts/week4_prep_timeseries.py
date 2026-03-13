import pandas as pd
import json
import glob
import os
import argparse
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_matches(matches_dir):
    """Loads all match JSONs to get match dates."""
    matches = []
    # Search recursively for match files
    match_files = glob.glob(os.path.join(matches_dir, "**", "*.json"), recursive=True)
    
    for f in match_files:
        with open(f, 'r', encoding='utf-8') as file:
            try:
                data = json.load(file)
                # Data is a list of matches
                for m in data:
                    matches.append({
                        "match_id": m["match_id"],
                        "match_date": m["match_date"],
                        "home_team": m["home_team"]["home_team_name"],
                        "away_team": m["away_team"]["away_team_name"]
                    })
            except Exception as e:
                logging.warning(f"Failed to load match file {f}: {e}")
                
    return pd.DataFrame(matches)

def load_events_and_aggregate(events_dir, matches_df):
    """Parses raw event files to calculate daily player stats."""
    player_stats = []
    
    # Iterate through known matches to process their event files
    # Only process matches for which we have event files (usually filename = match_id.json)
    
    for _, match_row in tqdm(matches_df.iterrows(), total=len(matches_df), desc="Processing Matches"):
        match_id = match_row["match_id"]
        match_date = match_row["match_date"]
        
        event_file = os.path.join(events_dir, f"{match_id}.json")
        if not os.path.exists(event_file):
            continue
            
        with open(event_file, 'r', encoding='utf-8') as f:
            events = json.load(f)
            
        # Helper to track stats per player in this match
        match_player_stats = {}
        
        for event in events:
            if "player" not in event:
                continue
                
            pid = event["player"]["id"]
            pname = event["player"]["name"]
            
            if pid not in match_player_stats:
                match_player_stats[pid] = {
                    "player_name": pname,
                    "match_id": match_id,
                    "date": match_date,
                    "goals": 0,
                    "assists": 0,
                    "shots": 0,
                    "passes": 0,
                    "xg": 0.0
                }
                
            stats = match_player_stats[pid]
            type_name = event["type"]["name"]
            
            # Goals (Shot type with outcome Goal)
            if type_name == "Shot":
                stats["shots"] += 1
                if "shot" in event and "statsbomb_xg" in event["shot"]:
                    stats["xg"] += event["shot"]["statsbomb_xg"]
                if "shot" in event and event["shot"]["outcome"]["name"] == "Goal":
                    stats["goals"] += 1
                    
            # Passes
            if type_name == "Pass":
                stats["passes"] += 1
                if "pass" in event and "goal_assist" in event["pass"]:
                    stats["assists"] += 1
                    
        player_stats.extend(match_player_stats.values())
        
    return pd.DataFrame(player_stats)

def process_sentiment(sentiment_path, target_start_date):
    """Loads sentiment, shifts dates to 2022, and aggregates."""
    if not os.path.exists(sentiment_path):
        logging.error(f"Sentiment file not found: {sentiment_path}")
        return None
        
    df = pd.read_csv(sentiment_path)
    
    # Parse dates
    # Assuming 'created_at' is ISO format
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Calculate shift
    # Find min date in sentiment
    min_sent_date = df['created_at'].min()
    
    # Target start date (World Cup 2022 started ~ Nov 20)
    target_date = pd.to_datetime(target_start_date).tz_localize(min_sent_date.tz)
    
    # Shift
    time_delta = target_date - min_sent_date
    logging.info(f"Shifting sentiment data by approximately {time_delta.days} days to align with World Cup 2022.")
    
    df['aligned_date'] = df['created_at'] + time_delta
    # Normalize to date only (remove time) for daily aggregation
    df['date'] = df['aligned_date'].dt.date
    
    # Aggregate daily
    daily_sent = df.groupby(['player_name', 'date']).agg({
        'compound': 'mean', # Average sentiment score
        'impression_count': 'sum', # Total volume/impact
        'tweet_id': 'count' # Number of tweets
    }).rename(columns={'compound': 'daily_sentiment', 'impression_count': 'daily_impact', 'tweet_id': 'daily_tweet_vol'}).reset_index()
    
    return daily_sent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/raw", help="Path to raw data directory")
    parser.add_argument("--processed-dir", default="data/processed", help="Path to processed data directory")
    args = parser.parse_args()
    
    # Paths
    matches_dir = os.path.join(args.raw_dir, "statsbomb_open_data", "matches")
    events_dir = os.path.join(args.raw_dir, "statsbomb_open_data", "events")
    sentiment_path = os.path.join(args.processed_dir, "twitter_sentiment.csv")
    final_data_path = os.path.join(args.processed_dir, "final_dataset.csv")
    
    # 1. Matches
    logging.info("Loading Matches...")
    matches_df = load_matches(matches_dir)
    if matches_df.empty:
        logging.error("No matches found given the directory structure.")
        return

    # 2. Performance Stats (Daily)
    logging.info("Extracting daily player performance from events...")
    perf_df = load_events_and_aggregate(events_dir, matches_df)
    
    # Convert perf date to datetime date object
    perf_df['date'] = pd.to_datetime(perf_df['date']).dt.date
    
    logging.info(f"Extracted performance stats: {len(perf_df)} rows.")
    
    # 3. Sentiment (Daily & Aligned)
    logging.info("Processing sentiment data...")
    # Use the earliest match date as the anchor for sentiment alignment
    matches_df['match_date'] = pd.to_datetime(matches_df['match_date'])
    start_date = matches_df['match_date'].min().strftime('%Y-%m-%d')
    
    sent_df = process_sentiment(sentiment_path, start_date)
    if sent_df is not None:
        logging.info(f"Processed sentiment stats: {len(sent_df)} rows.")
    
    # 4. Market Values (Target)
    # Market value is static for this short period, we load it from final_dataset
    logging.info("Loading Market Values...")
    if os.path.exists(final_data_path):
        mv_df = pd.read_csv(final_data_path)
        # Keep just player and MV
        mv_map = mv_df[['player_name', 'market_value_eur']].drop_duplicates()
    else:
        logging.warning("final_dataset.csv not found. Market values will be missing.")
        mv_map = pd.DataFrame(columns=['player_name', 'market_value_eur'])

    # 5. Merge Strategy: Full Outer Join on [Player, Date] to catch non-match days with tweets?
    # For LSTM, we usually want continuous days.
    # Let's create a date range
    
    min_date = matches_df['match_date'].min().date()
    max_date = matches_df['match_date'].max().date()
    all_dates = pd.date_range(min_date, max_date).date
    
    # Get all unique players
    all_players = pd.concat([perf_df['player_name'], sent_df['player_name'] if sent_df is not None else pd.Series()]).unique()
    
    # Create Skeleton DataFrame
    index_data = []
    for p in all_players:
        for d in all_dates:
            index_data.append({'player_name': p, 'date': d})
            
    skeleton_df = pd.DataFrame(index_data)
    
    # Merge Performance
    merged_df = pd.merge(skeleton_df, perf_df, on=['player_name', 'date'], how='left')
    
    # Merge Sentiment
    if sent_df is not None:
        merged_df = pd.merge(merged_df, sent_df, on=['player_name', 'date'], how='left')
        
    # Merge Static Market Value
    merged_df = pd.merge(merged_df, mv_map, on='player_name', how='left')
    
    # Fill NaNs
    # Performance: 0 (No match = 0 goals/shots)
    perf_cols = ['goals', 'assists', 'shots', 'passes', 'xg']
    merged_df[perf_cols] = merged_df[perf_cols].fillna(0)
    
    # Sentiment: 0 or Forward Fill?
    # Sentiment 0 is neutral/no signal. Let's fill with 0 for now.
    sent_cols = ['daily_sentiment', 'daily_impact', 'daily_tweet_vol']
    merged_df[sent_cols] = merged_df[sent_cols].fillna(0)
    
    # Market Value: Forward Fill (static)
    # Already merged fully, so just fill missing with 0 if unknown player
    merged_df['market_value_eur'] = merged_df['market_value_eur'].fillna(0)
    
    # Output
    out_path = os.path.join(args.processed_dir, "time_series_daily.csv")
    merged_df.to_csv(out_path, index=False)
    logging.info(f"Time-series dataset saved to {out_path} ({len(merged_df)} rows)")

if __name__ == "__main__":
    main()
