# ============================================================
# FILE: src/tune_lstm.py
# PURPOSE: Hyperparameter tuning for LSTM using manual grid search.
#          Searches over: units, layers, learning rate, dropout.
#          Saves best config to dashboard/lstm_best_params.pkl
# ============================================================

import pandas as pd
import numpy as np
import itertools
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------------------------------
# Load data
# ------------------------------------------------
df = pd.read_csv("data/processed/player_transfer_value_with_sentiment.csv")
df = df.sort_values(["player_name", "season_encoded"]).reset_index(drop=True)

lstm_features = [
    "market_value_eur", "attacking_output_index", "injury_burden_index",
    "availability_rate", "vader_compound_score", "social_buzz_score",
]

SEQUENCE_LENGTH = 3
scaler = MinMaxScaler()
scaler.fit(df[lstm_features])

# ------------------------------------------------
# Build sequences from FULL df, split by target season
# FIX: Cannot split df by season first — each sequence needs
#      3 prior seasons which may be in a different season group.
#      Build all sequences from full df, then filter by target season.
# ------------------------------------------------
X_all, y_all, target_seasons, actual_mv = [], [], [], []

for _, player_df in df.groupby("player_name"):
    player_df = player_df.sort_values("season_encoded")
    scaled    = scaler.transform(player_df[lstm_features])
    seasons   = player_df["season_encoded"].values
    mv_values = player_df["market_value_eur"].values

    for i in range(SEQUENCE_LENGTH, len(player_df)):
        X_all.append(scaled[i - SEQUENCE_LENGTH : i])
        y_all.append(scaled[i, 0])
        target_seasons.append(seasons[i])
        actual_mv.append(mv_values[i])

X_all          = np.array(X_all)
y_all          = np.array(y_all)
target_seasons = np.array(target_seasons)
actual_mv      = np.array(actual_mv)

train_mask = target_seasons <= 4
test_mask  = target_seasons == 5

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_test,  y_test  = X_all[test_mask],  y_all[test_mask]
actual_test      = actual_mv[test_mask]

print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

# ------------------------------------------------
# Grid search space
# ------------------------------------------------
param_grid = {
    "units":         [32, 64, 128],
    "learning_rate": [0.001, 0.005, 0.01],
    "dropout":       [0.0, 0.2],
    "layers":        [1, 2],
}

keys   = list(param_grid.keys())
values = list(param_grid.values())
combos = list(itertools.product(*values))

print(f"\nTotal combinations: {len(combos)}")
print("Running grid search...\n")

results = []

for idx, combo in enumerate(combos):
    params = dict(zip(keys, combo))

    model = Sequential()
    if params["layers"] == 1:
        model.add(LSTM(params["units"], activation="relu",
                       input_shape=(SEQUENCE_LENGTH, len(lstm_features))))
    else:
        model.add(LSTM(params["units"], activation="relu",
                       return_sequences=True,
                       input_shape=(SEQUENCE_LENGTH, len(lstm_features))))
        if params["dropout"] > 0:
            model.add(Dropout(params["dropout"]))
        model.add(LSTM(params["units"] // 2, activation="relu"))

    if params["dropout"] > 0:
        model.add(Dropout(params["dropout"]))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=params["learning_rate"]), loss="mse")

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=0)

    model.fit(X_train, y_train, epochs=50, batch_size=32,
              validation_split=0.2, callbacks=[es], verbose=0)

    preds_scaled = model.predict(X_test, verbose=0)
    pad          = np.zeros((len(preds_scaled), len(lstm_features) - 1))
    preds_eur    = scaler.inverse_transform(
        np.concatenate([preds_scaled, pad], axis=1))[:, 0]
    preds_eur    = np.maximum(preds_eur, 0)

    rmse = np.sqrt(mean_squared_error(actual_test, preds_eur))
    mae  = mean_absolute_error(actual_test, preds_eur)

    results.append({**params, "RMSE": rmse, "MAE": mae})
    print(f"  [{idx+1:2d}/{len(combos)}] units={params['units']:3d} "
          f"lr={params['learning_rate']:.3f} dropout={params['dropout']} "
          f"layers={params['layers']} → RMSE: €{rmse:,.0f}  MAE: €{mae:,.0f}")

# ------------------------------------------------
# Best result
# ------------------------------------------------
results_df  = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
best_params = results_df.iloc[0].to_dict()

print("\n" + "=" * 55)
print("  LSTM TUNING — Best Configuration")
print("=" * 55)
for k, v in best_params.items():
    print(f"  {k:<20} {v}")
print("=" * 55)

joblib.dump(best_params, "dashboard/lstm_best_params.pkl")
results_df.to_csv("reports/lstm_tuning_results.csv", index=False)

print("\n✅ Best params saved → dashboard/lstm_best_params.pkl")
print("✅ All results saved → reports/lstm_tuning_results.csv")