# ============================================================
# FILE: src/generate_lstm_features.py
# ============================================================

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------
# Load & sort
# ------------------------------------------------
df = pd.read_csv("data/processed/player_transfer_value_with_sentiment.csv")
df = df.sort_values(["player_name", "season_encoded"]).reset_index(drop=True)

print(f"Loaded dataset: {df.shape}  ({df['player_name'].nunique()} players)")

# ------------------------------------------------
# LSTM feature columns (must match training config)
# ------------------------------------------------
lstm_features = [
    "market_value_eur",
    "attacking_output_index",
    "injury_burden_index",
    "availability_rate",
    "vader_compound_score",
    "social_buzz_score",
]

SEQUENCE_LENGTH = 3

# ------------------------------------------------
# Scaler — fit on whole dataset (same as LSTM training)
# ------------------------------------------------
scaler = MinMaxScaler()
scaler.fit(df[lstm_features])

# ------------------------------------------------
# Load LSTM
# ------------------------------------------------
model = load_model("dashboard/lstm_model.h5", compile=False)
print("LSTM model loaded.")

# ------------------------------------------------
# Build per-player sequences
# Also store the raw market values of each input sequence
# so we can use their average as a fallback for zero predictions
# ------------------------------------------------
seq_list      = []   # (N, 3, n_features)
idx_list      = []   # original df index for each target row
seq_mv_avg    = []   # avg market value of the 3 input seasons per sequence

for player_name, player_df in df.groupby("player_name"):
    player_df = player_df.sort_values("season_encoded")
    scaled    = scaler.transform(player_df[lstm_features])
    orig_idx  = player_df.index.tolist()
    mv_values = player_df["market_value_eur"].values

    for i in range(SEQUENCE_LENGTH, len(player_df)):
        seq_list.append(scaled[i - SEQUENCE_LENGTH : i])
        idx_list.append(orig_idx[i])
        # average of the 3 input seasons market values
        seq_mv_avg.append(float(np.mean(mv_values[i - SEQUENCE_LENGTH : i])))

X_all      = np.array(seq_list)
seq_mv_avg = np.array(seq_mv_avg)

print(f"\nRunning batch LSTM prediction on {len(X_all)} sequences...")

# ------------------------------------------------
# Single batch prediction
# ------------------------------------------------
preds_scaled = model.predict(X_all, batch_size=256, verbose=1)

pad        = np.zeros((len(preds_scaled), len(lstm_features) - 1))
preds_full = np.concatenate([preds_scaled, pad], axis=1)
preds_eur  = scaler.inverse_transform(preds_full)[:, 0]
preds_eur  = np.maximum(preds_eur, 0)

# ------------------------------------------------
# Smart fallback for zero predictions
#
# WHY ZEROS HAPPEN:
#   ~38% of lower-tier players have social_buzz_score = 0 and
#   near-identical market values across all 5 seasons.
#   MinMaxScaler compresses their features near zero, so the
#   LSTM outputs ~0 after inverse transform.
#
# WHY NOT USE ACTUAL VALUE:
#   Using the current season's actual value as fallback makes
#   all low-tier players have the same lstm_pred (e.g. €4M flat)
#   which gives XGBoost no useful signal to learn from.
#
# THE SMART FIX — use average of the 3 INPUT seasons:
#   - Each player gets a different fallback per season
#   - Reflects the trend the LSTM actually saw as input
#   - For a player going 7.5M → 7.5M → 4M the fallback
#     for season 5 is (7.5+7.5+4)/3 = €6.33M, not flat €4M
#   - Much more informative signal for XGBoost
# ------------------------------------------------
zero_mask  = preds_eur == 0
zero_count = int(zero_mask.sum())

preds_eur[zero_mask] = seq_mv_avg[zero_mask]

print(f"\n   Zero predictions replaced with sequence average: {zero_count}")
print(f"   Zero lstm_pred remaining: {int((preds_eur == 0).sum())}")

# Verify variation in fallback values
if zero_count > 0:
    fallback_vals = seq_mv_avg[zero_mask]
    print(f"   Fallback value range: €{fallback_vals.min():,.0f} → €{fallback_vals.max():,.0f}")
    print(f"   Fallback unique values: {len(np.unique(fallback_vals))} (not all same)")

# ------------------------------------------------
# Write lstm_pred back into dataframe
# ------------------------------------------------
df["lstm_pred"] = np.nan
for df_idx, pred in zip(idx_list, preds_eur):
    df.at[df_idx, "lstm_pred"] = float(pred)

df_enriched = df.dropna(subset=["lstm_pred"]).copy().reset_index(drop=True)

print(f"\n✅ Enriched dataset: {df_enriched.shape}")
print(f"   Seasons present:  {sorted(df_enriched['season_encoded'].unique())}")
print(f"   Players covered:  {df_enriched['player_name'].nunique()}")

residuals = df_enriched["market_value_eur"] - df_enriched["lstm_pred"]
print(f"\n   Residual mean :  €{residuals.mean():>12,.0f}")
print(f"   Residual std  :  €{residuals.std():>12,.0f}")
print(f"   Residual range:  €{residuals.min():>12,.0f}  →  €{residuals.max():,.0f}")

# ------------------------------------------------
# Save
# ------------------------------------------------
df_enriched.to_csv("data/processed/lstm_enriched.csv", index=False)
print("\n✅ Saved → data/processed/lstm_enriched.csv")
print("   Next step: python src/train_xgboost.py")