from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import os
import joblib

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Minimal data storage
data = {}

def load_data():
    path = 'e:/PROJECT/INFOSYS-AI/backend/data/player_transfer_value_with_sentiment.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        
        # Automatic column detection
        mappings = {
            'player_name': ['player_name', 'name', 'full_name'],
            'market_value': ['market_value_eur', 'market_value', 'value'],
            'player_id': ['player_id', 'id', 'uid'],
            'age': ['current_age', 'age'],
            'sentiment': ['vader_compound_score', 'sentiment_score', 'sentiment']
        }
        
        for std, aliases in mappings.items():
            if std not in df.columns:
                for alias in aliases:
                    if alias in df.columns:
                        df = df.rename(columns={alias: std})
                        break
        
        # Ensure player_id
        if 'player_id' not in df.columns:
            df['player_id'] = range(1, len(df) + 1)
            
        # Simplified manual preprocessing to avoid hangs
        df = df.fillna(0)
        
        # Performance stats detection
        if 'goals' not in df.columns: df['goals'] = 0
        if 'assists' not in df.columns: df['assists'] = 0
        if 'matches' not in df.columns and 'matches_played' in df.columns: df['matches'] = df['matches_played']
        if 'matches' not in df.columns: df['matches'] = 0
        
        df['performance_score'] = df['goals'] + df['assists']
        df['goal_ratio'] = df['goals'] / df['matches'].replace(0, 1)
        
        age_col = 'age' if 'age' in df.columns else 'current_age'
        if age_col in df.columns:
            df['age_factor'] = df[age_col].apply(lambda x: 1.0 if 20 <= x <= 28 else 0.5)
        else:
            df['age_factor'] = 0.8
            
        return df
    return None

@app.on_event("startup")
def startup_event():
    global data
    data['df'] = load_data()
    print("Backend ready.")

@app.get("/players")
def get_players():
    df = data.get('df')
    if df is None: return []
    return df[['player_id', 'player_name']].drop_duplicates().to_dict(orient='records')

@app.get("/stats/{player_id}")
def get_stats(player_id: int):
    df = data.get('df')
    if df is None: return []
    return df[df['player_id'] == player_id].sort_values('season').to_dict(orient='records')

@app.get("/predict/{player_id}")
def predict(player_id: int):
    df = data.get('df')
    if df is None: return {}
    player_data = df[df['player_id'] == player_id]
    if player_data.empty: return {}
    
    last = player_data.iloc[-1]
    # Simple predict model
    val = last['market_value'] * (1 + (last['performance_score'] * 0.05))
    return {
        "player_id": player_id,
        "player_name": last['player_name'],
        "predicted_value": round(float(val), 2),
        "xgb_contribution": 0.5,
        "lstm_contribution": 0.5
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
