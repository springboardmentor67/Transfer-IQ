from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def load_data(raw_dir: Path, processed_dir: Path) -> dict[str, pd.DataFrame]:
    data = {}
    
    # StatsBomb
    sb_path = processed_dir / "statsbomb_player_performance.csv"
    if sb_path.exists():
        try:
            data["statsbomb"] = pd.read_csv(sb_path)
        except pd.errors.EmptyDataError:
            print(f"Warning: {sb_path} is empty.")
    else:
        print(f"Warning: {sb_path} not found.")
        
    # Transfermarkt Market Values
    tm_mv_path = raw_dir / "transfermarkt_market_value.csv/market_values.csv"
    if tm_mv_path.exists():
        try:
            data["market_values"] = pd.read_csv(tm_mv_path)
        except pd.errors.EmptyDataError:
            print(f"Warning: {tm_mv_path} is empty.")
    else:
        print(f"Warning: {tm_mv_path} not found.")

    # Transfermarkt History
    tm_hist_path = raw_dir / "transfermarkt_market_value.csv/market_value_history.csv"
    if tm_hist_path.exists():
        try:
            data["market_value_history"] = pd.read_csv(tm_hist_path)
        except pd.errors.EmptyDataError:
            print(f"Warning: {tm_hist_path} is empty.")
    
    # Sentiment (Week 3)
    sentiment_path = processed_dir / "player_sentiment_features.csv"
    if sentiment_path.exists():
        try:
            data["sentiment"] = pd.read_csv(sentiment_path)
        except pd.errors.EmptyDataError:
             print(f"Warning: {sentiment_path} is empty.")
    else:
        print(f"Warning: {sentiment_path} not found.")
        
    return data

def process_data(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    # merge logic here
    # 1. Base on StatsBomb data (performance)
    df = data.get("statsbomb")
    if df is None:
        raise ValueError("No StatsBomb data found.")
    
    # 2. Merge Market Values
    mv = data.get("market_values")
    if mv is not None and not mv.empty:
        # fuzzy merge or exact match on name?
        # For now, let's try exact match on player_name, but we might need to normalize names.
        # Ideally we use the mapping file if available.
        
        # Simplify for this iteration: use player_name
        mv_subset = mv[["player_name", "market_value_eur", "club"]].dropna(subset=["market_value_eur"])
        # Aggregate if duplicates (take last known value)
        mv_subset = mv_subset.groupby("player_name").last().reset_index()
        
        df = pd.merge(df, mv_subset, on="player_name", how="left")
        
    # 3. Merge Sentiment
    sent = data.get("sentiment")
    if sent is not None and not sent.empty:
        # Week 3 output is already aggregated by player
        # columns: player_name, vader_compound_mean, sentiment_impact_signed_sum, tweet_volume, etc.
        
        # Select and rename relevant columns for the final dataset
        sent_subset = sent[[
            "player_name", 
            "vader_compound_mean", 
            "sentiment_impact_signed_sum", 
            "tweet_volume"
        ]].rename(columns={
            "vader_compound_mean": "sentiment_score",
            "sentiment_impact_signed_sum": "sentiment_impact",
            "tweet_volume": "social_volume"
        })
        
        df = pd.merge(df, sent_subset, on="player_name", how="left")

    # Feature Engineering
    # - Rolling averages? (Needs match level data, here we have season level)
    # - Age? (Need DOB from Transfermarkt or StatsBomb)
    
    # Fill missing values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # Encode categorical
    # e.g. club, position (if available)
    if "club" in df.columns:
        df["club"] = df["club"].fillna("Unknown")
        # One-hot encoding for club might be too high dimensionality, let's strip it for now or label encode
        # For this exercise, let's just keep it as is or drop it for the model input
        pass

    return df

def main() -> None:
    parser = argparse.ArgumentParser(description="Week 2: Data Processing & Engineering")
    parser.add_argument("--raw-dir", default="data/raw", type=Path)
    parser.add_argument("--processed-dir", default="data/processed", type=Path)
    args = parser.parse_args()
    
    data = load_data(args.raw_dir, args.processed_dir)
    final_df = process_data(data)
    
    out_path = args.processed_dir / "final_dataset.csv"
    final_df.to_csv(out_path, index=False)
    print(f"Wrote final dataset to {out_path} ({len(final_df)} rows, {len(final_df.columns)} columns)")

if __name__ == "__main__":
    main()
