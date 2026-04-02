from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import joblib
import os
import sys

# Get the correct base path for Render
if 'RENDER' in os.environ:
    # On Render, the working directory is the repo root
    base_path = os.getcwd()
    # Go up one level if needed (since root directory is set to Userinterface/backend)
    if os.path.basename(base_path) == 'backend':
        base_path = os.path.dirname(base_path)
else:
    # Local development
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

# Now use base_path for all file paths
player_data_path = os.path.join(base_path, 'player_data.csv')
app = Flask(__name__)
CORS(app)

print("=" * 60)
print("TransferIQ Backend - Running")
print("=" * 60)

# Get paths
base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
print(f"Base path: {base_path}")

# Load player data
player_data_path = os.path.join(base_path, 'player_data.csv')
player_data = pd.read_csv(player_data_path)
players = player_data['player_name'].unique().tolist()
print(f"Loaded {len(players)} players")

# Load LSTM predictions
print("\nLoading predictions...")

uni_path = os.path.join(base_path, 'models', 'predictions', 'univariate_predictions.csv')
multi_path = os.path.join(base_path, 'models', 'predictions', 'multivariate_predictions.csv')
enc_path = os.path.join(base_path, 'models', 'predictions', 'encoder_decoder_predictions.csv')

# Create dictionaries for fast lookup
univariate_dict = {}
multivariate_dict = {}
encoder_dict = {}

if os.path.exists(uni_path):
    df = pd.read_csv(uni_path)
    for _, row in df.iterrows():
        key = f"{row['player_name']}_{row['season']}"
        univariate_dict[key] = row['univariate_prediction']
    print(f"Univariate: {len(univariate_dict)} predictions")

if os.path.exists(multi_path):
    df = pd.read_csv(multi_path)
    for _, row in df.iterrows():
        key = f"{row['player_name']}_{row['season']}"
        multivariate_dict[key] = row['multivariate_prediction']
    print(f"Multivariate: {len(multivariate_dict)} predictions")

if os.path.exists(enc_path):
    df = pd.read_csv(enc_path)
    for _, row in df.iterrows():
        key = f"{row['player_name']}_{row['season']}"
        encoder_dict[key] = row['encoder_decoder_prediction']
    print(f"Encoder-Decoder: {len(encoder_dict)} predictions")

# Load optimized models
xgboost_optimized = None
ensemble_config = None

try:
    xgb_opt_path = os.path.join(base_path, 'models', 'saved_models', 'xgboost_optimized.pkl')
    if os.path.exists(xgb_opt_path):
        xgboost_optimized = joblib.load(xgb_opt_path)
        print("Optimized XGBoost loaded")
except:
    print("Optimized XGBoost not found")

try:
    config_path = os.path.join(base_path, 'models', 'saved_models', 'ensemble_config.pkl')
    if os.path.exists(config_path):
        ensemble_config = joblib.load(config_path)
        print("Ensemble config loaded")
except:
    print("Ensemble config not found")

def get_player_info(player_name):
    """Get player basic info with correct sentiment columns"""
    player_df = player_data[player_data['player_name'] == player_name]
    if len(player_df) == 0:
        return None
    
    # Get the latest season data for the player
    latest = player_df.sort_values('season').iloc[-1]
    
    # Print debug info for Messi
    if 'Messi' in player_name:
        print(f"\n[DEBUG] Messi data:")
        print(f"  vader_compound_score: {latest.get('vader_compound_score', 'N/A')}")
        print(f"  positive_count: {latest.get('positive_count', 'N/A')}")
        print(f"  negative_count: {latest.get('negative_count', 'N/A')}")
        print(f"  neutral_count: {latest.get('neutral_count', 'N/A')}")
        print(f"  sentiment_label: {latest.get('sentiment_label', 'N/A')}")
    
    # Get sentiment score - try multiple column names
    sentiment_score = 0
    if 'vader_compound_score' in latest.index:
        sentiment_score = latest['vader_compound_score']
    elif 'sentiment_score' in latest.index:
        sentiment_score = latest['sentiment_score']
    elif 'compound' in latest.index:
        sentiment_score = latest['compound']
    
    if pd.isna(sentiment_score):
        sentiment_score = 0
    
    # Get counts - try multiple column names
    positive_tweets = 0
    if 'positive_count' in latest.index:
        positive_tweets = latest['positive_count']
    elif 'positive_tweets' in latest.index:
        positive_tweets = latest['positive_tweets']
    
    negative_tweets = 0
    if 'negative_count' in latest.index:
        negative_tweets = latest['negative_count']
    elif 'negative_tweets' in latest.index:
        negative_tweets = latest['negative_tweets']
    
    neutral_count = 0
    if 'neutral_count' in latest.index:
        neutral_count = latest['neutral_count']
    elif 'neutral' in latest.index:
        neutral_count = latest['neutral']
    
    # Handle NaN values
    if pd.isna(positive_tweets):
        positive_tweets = 0
    if pd.isna(negative_tweets):
        negative_tweets = 0
    if pd.isna(neutral_count):
        neutral_count = 0
    
    total_tweets = int(positive_tweets + negative_tweets + neutral_count)
    
    # Get total likes
    total_likes = 0
    if 'total_likes' in latest.index:
        total_likes = latest['total_likes']
    if pd.isna(total_likes):
        total_likes = 0
    
    # Get engagement rate
    engagement_rate = 0
    if 'tweet_engagement_rate' in latest.index:
        engagement_rate = latest['tweet_engagement_rate']
    if pd.isna(engagement_rate):
        engagement_rate = 0
    
    return {
        'player': player_name,
        'team': str(latest.get('team', 'Unknown')),
        'position': str(latest.get('position', 'Unknown')),
        'age': int(latest.get('current_age', latest.get('age', 25))),
        'market_value': float(latest.get('market_value_eur', 0)),
        'goals': int(latest.get('goals', 0)),
        'assists': int(latest.get('assists', 0)),
        # Sentiment fields
        'sentiment_score': float(sentiment_score),
        'total_tweets': int(total_tweets),
        'total_likes': int(total_likes),
        'positive_tweets': int(positive_tweets),
        'negative_tweets': int(negative_tweets),
        'neutral_count': int(neutral_count),
        'tweet_engagement_rate': float(engagement_rate)
    }

def get_latest_predictions(player_name):
    """Get latest LSTM predictions for a player"""
    player_df = player_data[player_data['player_name'] == player_name].sort_values('season')
    if len(player_df) == 0:
        return {'univariate': 0, 'multivariate': 0, 'encoder_decoder': 0}
    
    latest_season = player_df.iloc[-1]['season']
    key = f"{player_name}_{latest_season}"
    
    return {
        'univariate': float(univariate_dict.get(key, 0)),
        'multivariate': float(multivariate_dict.get(key, 0)),
        'encoder_decoder': float(encoder_dict.get(key, 0))
    }

def get_hybrid_prediction(player_name):
    """Get hybrid prediction"""
    lstm_preds = get_latest_predictions(player_name)
    ensemble_avg = (lstm_preds['univariate'] + lstm_preds['multivariate'] + lstm_preds['encoder_decoder']) / 3
    
    xgb_pred = ensemble_avg
    if xgboost_optimized is not None:
        try:
            features = [[
                lstm_preds['univariate'],
                lstm_preds['multivariate'],
                lstm_preds['encoder_decoder'],
                0, 0, 0,
                lstm_preds['univariate'],
                lstm_preds['univariate'] * 0.95,
                0, 0
            ]]
            xgb_pred = float(xgboost_optimized.predict(features)[0])
        except:
            xgb_pred = ensemble_avg
    
    weighted_pred = ensemble_avg
    if ensemble_config and 'weights' in ensemble_config:
        w = ensemble_config['weights']
        weighted_pred = (w.get('univariate', 0.33) * lstm_preds['univariate'] + 
                        w.get('multivariate', 0.33) * lstm_preds['multivariate'] + 
                        w.get('encoder', 0.34) * lstm_preds['encoder_decoder'])
    
    xgb_weight = 0.7
    if ensemble_config and 'xgb_weight' in ensemble_config:
        xgb_weight = ensemble_config['xgb_weight']
    
    hybrid_pred = xgb_weight * xgb_pred + (1 - xgb_weight) * weighted_pred
    
    return {
        'lstm_univariate': lstm_preds['univariate'],
        'lstm_multivariate': lstm_preds['multivariate'],
        'lstm_encoder_decoder': lstm_preds['encoder_decoder'],
        'xgboost_prediction': xgb_pred,
        'hybrid_prediction': hybrid_pred
    }

# API Endpoints
@app.route('/players', methods=['GET'])
def get_players():
    return jsonify(players)

@app.route('/player/<name>', methods=['GET'])
def get_player(name):
    info = get_player_info(name)
    if info is None:
        return jsonify({'error': 'Player not found'}), 404
    return jsonify(info)

@app.route('/prediction/<name>', methods=['GET'])
def get_prediction(name):
    preds = get_latest_predictions(name)
    return jsonify(preds)

@app.route('/hybrid/<name>', methods=['GET'])
def get_hybrid(name):
    hybrid = get_hybrid_prediction(name)
    return jsonify(hybrid)

@app.route('/season/<name>', methods=['GET'])
def get_season(name):
    player_df = player_data[player_data['player_name'] == name].sort_values('season')
    seasons = []
    for _, row in player_df.iterrows():
        seasons.append({
            'season': str(row['season']),
            'market_value_eur': float(row['market_value_eur'])
        })
    return jsonify(seasons)

@app.route('/forecast/<name>', methods=['GET'])
def get_forecast(name):
    player_df = player_data[player_data['player_name'] == name].sort_values('season')
    if len(player_df) == 0:
        return jsonify([])
    
    current_value = float(player_df.iloc[-1]['market_value_eur'])
    current_season = str(player_df.iloc[-1]['season'])
    
    preds = get_latest_predictions(name)
    lstm_avg = (preds['univariate'] + preds['multivariate'] + preds['encoder_decoder']) / 3
    expected_change = (lstm_avg - current_value) / current_value if current_value > 0 else 0
    
    try:
        if '/' in str(current_season):
            season_int = int(str(current_season).split('/')[0])
        else:
            season_int = int(current_season)
    except:
        season_int = 2024
    
    age = int(player_df.iloc[-1].get('current_age', player_df.iloc[-1].get('age', 25)))
    
    forecast = []
    for i in range(1, 6):
        future_age = age + i
        
        lstm_factor = 1 + (expected_change * (0.85 ** i))
        
        if future_age > 30:
            age_factor = 1 - (0.04 * (future_age - 30))
        elif future_age < 24:
            age_factor = 1 + (0.03 * (24 - future_age))
        else:
            age_factor = 1
        
        variability = 1 + np.random.uniform(-0.03, 0.03) * i
        final_factor = lstm_factor * age_factor * variability
        predicted_value = current_value * final_factor
        
        forecast.append({
            'season': str(season_int + i),
            'value': max(0, float(predicted_value))
        })
    
    return jsonify(forecast)

@app.route('/topplayers', methods=['GET'])
def get_top_players():
    top = player_data.nlargest(10, 'market_value_eur')[['player_name', 'team', 'market_value_eur']]
    return jsonify(top.to_dict('records'))

@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    return jsonify([
        {'model': 'Univariate LSTM', 'rmse': 3143},
        {'model': 'Multivariate LSTM', 'rmse': 379759},
        {'model': 'Encoder-Decoder', 'rmse': 762200},
        {'model': 'XGBoost Stacking', 'rmse': 408543}
    ])

@app.route('/score/<name>', methods=['GET'])
def get_score(name):
    player_info = get_player_info(name)
    if player_info:
        value = player_info['market_value']
        goals = player_info['goals']
        assists = player_info['assists']
        age = player_info['age']
        
        score = 50
        if 24 <= age <= 28:
            score += 20
        elif 22 <= age <= 30:
            score += 10
        
        contributions = goals + assists
        if contributions > 20:
            score += 20
        elif contributions > 10:
            score += 10
        elif contributions > 5:
            score += 5
        
        if value > 50000000:
            score += 10
        elif value > 20000000:
            score += 5
        
        return jsonify({'score': min(100, max(0, score))})
    return jsonify({'score': 50})

@app.route('/sentiment_trend', methods=['GET'])
def get_sentiment_trend():
    """Get sentiment trend over seasons using vader_compound_score"""
    try:
        df = pd.read_csv('player_data.csv')
        sentiment_trend = []
        for season in sorted(df['season'].unique()):
            season_df = df[df['season'] == season]
            avg_sentiment = season_df['vader_compound_score'].mean()
            if pd.isna(avg_sentiment):
                avg_sentiment = 0
            sentiment_trend.append({
                'season': str(season),
                'sentiment': float(avg_sentiment)
            })
        return jsonify(sentiment_trend)
    except Exception as e:
        print(f"Sentiment trend error: {e}")
        return jsonify([])

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Server running on http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, port=5000)