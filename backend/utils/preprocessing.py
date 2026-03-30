import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def load_data(file_path):
    """
    Load dataset from CSV
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def handle_missing_values(df):
    """
    Fill missing values with median for numeric columns
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

def encode_categorical(df):
    """
    Encode player_name and season if present
    """
    le = LabelEncoder()
    if 'player_name' in df.columns:
        df['player_name_encoded'] = le.fit_transform(df['player_name'])
    if 'season' in df.columns:
        df['season_encoded'] = le.fit_transform(df['season'])
    return df, le

def scale_features(df, features_to_scale):
    """
    Scale numeric features using MinMaxScaler
    """
    scaler = MinMaxScaler()
    df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
    return df, scaler
