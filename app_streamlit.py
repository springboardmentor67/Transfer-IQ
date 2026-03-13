import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import xgboost as xgb
import torch
import torch.nn as nn
import time

# --- Page Config ---
st.set_page_config(
    page_title="AI Player Value Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Validating Local Paths ---
# Ensure we are in the project root
if not os.path.exists("data"):
    st.error("Data directory not found. Please run this app from the project root.")
    st.stop()

# --- Custom CSS for Styling ---
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 2rem;
    }
    h1 {
        color: #1f77b4;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/processed/final_dataset.csv")
        return df
    except FileNotFoundError:
        return None

@st.cache_data
def load_sentiment_data():
    try:
        df = pd.read_csv("data/processed/player_sentiment_features.csv")
        return df
    except FileNotFoundError:
        return None

df = load_data()
sent_df = load_sentiment_data()

# --- Model Loading ---
# Define LSTM Class (Copy from week5_lstm_model.py to avoid import issues)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 16)
        self.relu = nn.ReLU()
        self.out = nn.Linear(16, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = self.relu(out)
        out = self.out(out)
        return out

@st.cache_resource
def load_models():
    models = {}
    
    # Load XGBoost
    if os.path.exists("models/xgboost_model.pkl"):
        models["xgboost"] = joblib.load("models/xgboost_model.pkl")
    
    # Load LSTM (requires instantiating class first, and knowing input size)
    # We'll skip actual inference for LSTM in this demo app if input size is dynamic/unknown without pre-processor
    # But we can try to load the state dict if we knew the input size.
    # For now, let's stick to XGBoost for the interactive prediction.
    
    return models

models = load_models()

# --- Sidebar ---
st.sidebar.title("⚽ Player Analysis")
if df is not None:
    player_list = sorted(df['player_name'].unique().tolist())
    selected_player = st.sidebar.selectbox("Select Player", player_list)
else:
    st.sidebar.warning("Data not loaded.")
    selected_player = None

# --- Main Content ---
st.title("AI-Based Transfer Value Prediction")

if selected_player and df is not None:
    # Get Player Data (Latest row)
    # Assuming final_dataset has one row per player (aggregated) or we take the latest
    p_data = df[df['player_name'] == selected_player].iloc[-1]
    
    st.header(f"Profile: {selected_player}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Market Value", f"€{p_data.get('market_value_eur', 0):,.0f}")
    
    with col2:
        st.metric("Age", f"{p_data.get('age', 'N/A')}")
        
    with col3:
        club = p_data.get('club', 'Unknown')
        st.metric("Club", club)
        
    # --- Performance Stats ---
    st.subheader("Performance Metrics (Season)")
    metrics = ['goals', 'assists', 'minutes_played', 'yellow_cards']
    cols = st.columns(len(metrics))
    for i, m in enumerate(metrics):
        val = p_data.get(m, 0)
        cols[i].metric(m.replace('_', ' ').title(), val)
        
    # --- Sentiment Analysis ---
    if sent_df is not None and selected_player in sent_df['player_name'].values:
        st.subheader("Sentiment Analysis")
        s_data = sent_df[sent_df['player_name'] == selected_player].iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            sentiment_score = s_data.get('vader_compound_mean', 0)
            st.metric("Avg Sentiment (VADER)", f"{sentiment_score:.2f}")
            if sentiment_score > 0.05:
                st.success("Positive Sentiment")
            elif sentiment_score < -0.05:
                st.error("Negative Sentiment")
            else:
                st.info("Neutral Sentiment")
                
        with col2:
            vol = s_data.get('tweet_volume', 0)
            st.metric("Tweet Volume", f"{vol:,}")
            
    # --- Prediction Model ---
    st.subheader("AI Value Prediction")
    
    if "xgboost" in models:
        # Prepare input for prediction
        # We need to construct a DF with same columns as training
        # For this demo, we can re-use p_data but need to ensure columns match model
        
        # Simple Check: Does the model have feature_names_in_?
        model = models["xgboost"]
        try:
            # Create input DF
            # This is tricky without the exact same pre-processing pipeline in app.py
            # For a robust app, we should save the pipeline (imputer, scaler, encoder) as a pickle.
            # Here we will attempt if features align, otherwise show a placeholder or warning.
            
            # Placeholder for demonstration of "Prediction"
            predicted_value = model.predict(pd.DataFrame([p_data])[model.feature_names_in_])[0]
            
            delta = predicted_value - p_data.get('market_value_eur', 0)
            st.metric("Preidcted Value (XGBoost)", f"€{predicted_value:,.0f}", delta=f"€{delta:,.0f}")
            
            # Confidence/Range (Mocked for XGBoost regressor)
            lower = predicted_value * 0.95
            upper = predicted_value * 1.05
            st.write(f"Estimated Range: €{lower:,.0f} - €{upper:,.0f}")
            
        except Exception as e:
            st.warning(f"Could not run live prediction: {e}")
            st.info("Ensure the app's preprocessing matches the training script exactly.")
    else:
        st.warning("Prediction model not found. Train the model first.")
        
else:
    st.info("Select a player from the sidebar to view details.")

# --- Footer ---
st.markdown("---")
st.caption("Infosys AI Internship Project | Week 8")
