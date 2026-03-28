import xgboost as xgb
import joblib
import os

class PlayerValueXGBoost:
    """
    XGBoost model wrapper for Player Transfer Value prediction.
    """
    def __init__(self, params=None):
        if params is None:
            self.params = {
                'objective': 'reg:squarederror',
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 100
            }
        else:
            self.params = params
        self.model = xgb.XGBRegressor(**self.params)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
