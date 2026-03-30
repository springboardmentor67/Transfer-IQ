import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

class DataPipeline:
    def __init__(self, raw_data_path: str):
        self.raw_data_path = raw_data_path
        self.scaler = MinMaxScaler()
        
    def load_and_preprocess(self) -> tuple[pd.DataFrame, list]:
        """Loads dataset, performs missing value imputation, one-hot encoding, feature eng, scaling"""
        if os.path.exists(self.raw_data_path):
            df = pd.read_csv(self.raw_data_path)
        else:
            raise FileNotFoundError(f"Data not found at {self.raw_data_path}")

        # Missing Value Handling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            df[col] = df[col].fillna("Unknown")

        # Feature Engineering: Contract Duration (Simulated if not present)
        if "contract_duration" not in df.columns:
            # Assumed randomly between 1 and 5 years
            np.random.seed(42)
            df['contract_duration'] = np.random.randint(1, 6, size=len(df))
            
        # Feature Engineering: Performance Trends (Rolling average logic can be applied per player)
        df.sort_values(by=['player_name', 'season'], inplace=True)
        df['goals_trend'] = df.groupby('player_name')['goals_per90'].transform(lambda x: x.rolling(window=2, min_periods=1).mean())
        df['assists_trend'] = df.groupby('player_name')['assists_per90'].transform(lambda x: x.rolling(window=2, min_periods=1).mean())
        
        # Injury risk score
        if "injury_burden_index" in df.columns:
            df['injury_risk_score'] = df['injury_burden_index'] * (df['current_age'] / 25.0)
        else:
            df['injury_risk_score'] = 0.5  # default
            
        # Sentiment Score mapping
        if 'vader_compound_score' not in df.columns:
            df['vader_compound_score'] = 0.0

        # One-hot encoding (positions, etc)
        # Avoid exploding columns by selecting major categories
        if 'position' in df.columns:
            df = pd.get_dummies(df, columns=['position'], drop_first=True)

        target_col = 'market_value_eur'
        
        # Scaling numeric columns except target and keys
        keys = ['player_name', 'season', 'team', 'season_encoded', 'sentiment_label', 'market_value_source', 'career_stage', 'most_common_injury']
        # exclude target
        features = [c for c in df.columns if c not in keys and c != target_col and df[c].dtype in [np.float64, np.int64, np.float32, np.int32, bool]]
        
        df_encoded = df.copy()
        for col in features:
            if df_encoded[col].dtype == bool:
                df_encoded[col] = df_encoded[col].astype(int)

        scaled_values = self.scaler.fit_transform(df_encoded[features + [target_col]])
        
        # We replace the values
        for i, col in enumerate(features + [target_col]):
            df_encoded[col] = scaled_values[:, i]

        return df, df_encoded, features, target_col

    def save_artifacts(self, df_encoded: pd.DataFrame, out_path: str, scaler_path: str):
        df_encoded.to_csv(out_path, index=False)
        joblib.dump(self.scaler, scaler_path)

if __name__ == "__main__":
    target = 'data/processed/player_transfer_value_with_sentiment.csv'
    pipeline = DataPipeline(target)
    df, df_encoded, features, target_col = pipeline.load_and_preprocess()
    print("Columns:", df_encoded.columns[:10])
    
    out_dir = 'data/processed/model_ready'
    os.makedirs(out_dir, exist_ok=True)
    pipeline.save_artifacts(df_encoded, f"{out_dir}/clean_data.csv", f"{out_dir}/scaler.pkl")
    print(f"Data scaled and saved correctly to {out_dir}")
