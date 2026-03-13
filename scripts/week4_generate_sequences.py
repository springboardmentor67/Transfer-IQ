import pandas as pd
import numpy as np
import argparse
import os
import logging
from sklearn.preprocessing import MinMaxScaler
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_sequences(data, seq_length, target_col_idx):
    """
    Creates sliding window sequences from the data.
    data: (num_samples, num_features) numpy array
    seq_length: Length of the input sequence (lookback)
    target_col_idx: Index of the target column to predict (e.g., market_value or xG)
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i : i + seq_length]
        y = data[i + seq_length, target_col_idx] # Predicting scalar value at t+1
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="data/processed/time_series_daily.csv")
    parser.add_argument("--output-dir", default="data/processed/lstm_ready")
    parser.add_argument("--seq-length", type=int, default=7, help="Lookback window size in days")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Data
    logging.info(f"Loading data from {args.input_file}")
    if not os.path.exists(args.input_file):
        logging.error("Input file not found.")
        return
        
    df = pd.read_csv(args.input_file)
    
    # Check for NaN and handle
    df = df.fillna(0)
    
    # Features to use for the model
    # We want to predict Market Value (or Performance?)
    # Usually Market Value updates are slow. For this exercise, let's try to predict 'sentiment_score' or 'market_value' next step?
    # Week 5 objective says "predict transfer value". Sticking to Market Value.
    # Note: MV is static in our short window, so the model will learn identity function unless we inject noise or have real history.
    # To make it key, let's predict 'daily_sentiment' as a proxy for value change signal, OR just setup for MV.
    
    features = [
        'goals', 'assists', 'shots', 'passes', 'xg',
        'daily_sentiment', 'daily_impact', 'daily_tweet_vol',
        'market_value_eur'
    ]
    
    target_col = 'market_value_eur'
    
    # Filter for numeric only
    df_model = df[features]
    
    # 2. Normalize
    logging.info("Normalizing data...")
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(df_model)
    
    # Save scaler for later inversion
    scaler_path = os.path.join(args.output_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    logging.info(f"Scaler saved to {scaler_path}")
    
    target_idx = features.index(target_col)
    
    # 3. Create Sequences per Player
    # We must treat each player as a separate time series
    logging.info("Generating sequences per player...")
    
    all_X = []
    all_y = []
    
    players = df['player_name'].unique()
    
    for player in players:
        p_df = df[df['player_name'] == player]
        # Sort by date
        p_df = p_df.sort_values('date')
        
        p_data = p_df[features].values
        p_data_scaled = scaler.transform(p_df[features]) # Use same scaler
        
        if len(p_data_scaled) <= args.seq_length:
            continue
            
        X, y = create_sequences(p_data_scaled, args.seq_length, target_idx)
        all_X.append(X)
        all_y.append(y)
        
    if not all_X:
        logging.error("No sequences generated. Data might be too short for the requested sequence length.")
        return

    final_X = np.concatenate(all_X)
    final_y = np.concatenate(all_y)
    
    logging.info(f"Generated {len(final_X)} sequences. Shape: X={final_X.shape}, y={final_y.shape}")
    
    # 4. Save
    np.save(os.path.join(args.output_dir, "X.npy"), final_X)
    np.save(os.path.join(args.output_dir, "y.npy"), final_y)
    
    logging.info("LSTM training data prepared successfully.")

if __name__ == "__main__":
    main()
