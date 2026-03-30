import xgboost as xgb
import numpy as np

class EnsembleModel:
    def __init__(self, lstm_weight=0.5, xgb_weight=0.5):
        self.lstm_weight = lstm_weight
        self.xgb_weight = xgb_weight
        self.xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        
    def train_xgb(self, X_train, y_train):
        print("Training XGBoost Regressor...")
        self.xgb_model.fit(X_train, y_train)
        print("XGBoost Training Complete.")
        
    def predict_xgb(self, X_test):
        return self.xgb_model.predict(X_test)
        
    def predict_ensemble(self, lstm_preds, xgb_preds):
        return (self.lstm_weight * lstm_preds) + (self.xgb_weight * xgb_preds)

    def save_xgb(self, file_path):
        import joblib
        joblib.dump(self.xgb_model, file_path)

    def load_xgb(self, file_path):
        import joblib
        self.xgb_model = joblib.load(file_path)

if __name__ == "__main__":
    import numpy as np
    
    # Simple test
    X = np.random.rand(100, 10)
    y = np.random.rand(100)
    
    ensemble = EnsembleModel(lstm_weight=0.6, xgb_weight=0.4)
    ensemble.train_xgb(X, y)
    
    xgb_preds = ensemble.predict_xgb(X[:5])
    lstm_preds = np.random.rand(5) # simulated lstm predictions
    
    final_preds = ensemble.predict_ensemble(lstm_preds, xgb_preds)
    print("XGB Preds:", xgb_preds)
    print("LSTM Preds:", lstm_preds)
    print("Ensemble Preds:", final_preds)
