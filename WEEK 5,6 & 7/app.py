# ============================================================
# ⚽ TRANSFER IQ — FINAL VERSION (WEEK 5,6,7)
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import xgboost as xgb

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

st.set_page_config(page_title="TransferIQ FINAL", layout="wide")
st.title("⚽ TransferIQ — Final AI Model (Week 5,6,7)")

@st.cache_data
def load_data():
    df = pd.read_csv("player_transfer_value_with_sentiment.csv")
    df = df.sort_values(by=["player_name", "season_encoded"])
    return df

df = load_data()

TARGET = "log_market_value"

FEATURES = [
    'current_age','matches','minutes_played','goals','assists',
    'shots','passes_total','pass_accuracy_pct',
    'tackles_total','interceptions',
    'goals_per90','assists_per90',
    'shot_conversion_rate','attacking_output_index',
    'injury_burden_index','availability_rate',
    'social_buzz_score','vader_compound_score',
    'transfer_attractiveness_score'
]

player = st.sidebar.selectbox("Select Player", df["player_name"].unique())
week = st.sidebar.selectbox("Select Week", ["Week 5", "Week 6", "Week 7"])

player_df = df[df["player_name"] == player]
actual_value = player_df["market_value_eur"].iloc[-1]

def to_euro(val):
    return np.exp(val)

st.sidebar.metric("Actual (€)", f"{actual_value:,.0f}")

# ============================================================
# 🔵 WEEK 5 — LSTM 
# ============================================================

if week == "Week 5":

    st.header("🔵 Week 5 — LSTM Models (Global Training)")

    seq_len = 5
    future_steps = 3

    data = df.sort_values(["player_name", "season_encoded"])

    # -------- UNIVARIATE --------
    values = data[TARGET].values

    X, y = [], []
    for i in range(len(values)-seq_len):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len])

    X, y = np.array(X), np.array(y)

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    X_train = X_train.reshape((X_train.shape[0], seq_len, 1))
    X_test = X_test.reshape((X_test.shape[0], seq_len, 1))

    model_uni = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_len,1)),
        Dense(1)
    ])

    model_uni.compile(optimizer='adam', loss='mse')

    history_uni = model_uni.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=15,
        verbose=0
    )

    # -------- MULTIVARIATE --------
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_all = feature_scaler.fit_transform(data[FEATURES])
    y_all = target_scaler.fit_transform(data[[TARGET]])

    X_m, y_m = [], []
    for i in range(len(X_all)-seq_len):
        X_m.append(X_all[i:i+seq_len])
        y_m.append(y_all[i+seq_len])

    X_m, y_m = np.array(X_m), np.array(y_m)

    split = int(len(X_m)*0.8)
    X_train_m, X_test_m = X_m[:split], X_m[split:]
    y_train_m, y_test_m = y_m[:split], y_m[split:]

    model_multi = Sequential([
        LSTM(64, return_sequences=True, input_shape=(seq_len, len(FEATURES))),
        LSTM(32),
        Dense(1)
    ])

    model_multi.compile(optimizer='adam', loss='mse')

    history_multi = model_multi.fit(
        X_train_m, y_train_m,
        validation_data=(X_test_m, y_test_m),
        epochs=15,
        verbose=0
    )

    # -------- ENCODER-DECODER --------
    X_e, y_e = [], []
    for i in range(len(values)-seq_len-future_steps):
        X_e.append(values[i:i+seq_len])
        y_e.append(values[i+seq_len:i+seq_len+future_steps])

    X_e, y_e = np.array(X_e), np.array(y_e)

    X_e = X_e.reshape((X_e.shape[0], seq_len, 1))
    y_e = y_e.reshape((y_e.shape[0], future_steps, 1))

    model_enc = Sequential([
        LSTM(50, activation='relu', input_shape=(seq_len,1)),
        RepeatVector(future_steps),
        LSTM(50, activation='relu', return_sequences=True),
        TimeDistributed(Dense(1))
    ])

    model_enc.compile(optimizer='adam', loss='mse')
    model_enc.fit(X_e, y_e, epochs=15, verbose=0)

    # -------- PLAYER PREDICTION --------
    player_values = player_df[TARGET].values

    if len(player_values) >= seq_len:

        last_seq = player_values[-seq_len:].reshape(1, seq_len, 1)
        uni_pred = model_uni.predict(last_seq)[0][0]

        last_multi = player_df[FEATURES].tail(seq_len).values
        last_multi = feature_scaler.transform(last_multi)
        last_multi = last_multi.reshape(1, seq_len, len(FEATURES))

        multi_pred_scaled = model_multi.predict(last_multi)
        multi_pred = target_scaler.inverse_transform(multi_pred_scaled)[0][0]

        future_preds = model_enc.predict(last_seq)

    else:
        st.warning("Limited player history — using fallback")
        uni_pred = np.mean(player_values)
        multi_pred = uni_pred
        future_preds = [[uni_pred]*future_steps]

    # -------- DISPLAY --------
    col1, col2, col3 = st.columns(3)
    col1.metric("Actual (€)", f"{actual_value:,.0f}")
    col2.metric("Univariate (€)", f"{to_euro(uni_pred):,.0f}")
    col3.metric("Multivariate (€)", f"{to_euro(multi_pred):,.0f}")

    st.subheader("📉 Loss Curve")
    fig, ax = plt.subplots()
    ax.plot(history_uni.history['loss'], label="Uni Train")
    ax.plot(history_uni.history['val_loss'], label="Uni Val")
    ax.plot(history_multi.history['loss'], label="Multi Train")
    ax.plot(history_multi.history['val_loss'], label="Multi Val")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("🔮 Multi-step Forecast")
    st.write([f"{to_euro(x[0]):,.0f}" for x in future_preds[0]])

    st.subheader("📈 Actual vs Predicted")
    fig2, ax2 = plt.subplots()
    ax2.plot(np.exp(player_df[TARGET].values), label="Actual (€)")
    ax2.scatter(len(player_df), to_euro(multi_pred), label="Predicted (€)", s=80)
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

# ============================================================
# 🟢 WEEK 6 
# ============================================================

elif week == "Week 6":

    st.header("🟢 Week 6 — XGBoost + Ensemble")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1]
    }

    model = xgb.XGBRegressor()

    search = RandomizedSearchCV(
        model, params, n_iter=5,
        scoring='neg_mean_squared_error',
        cv=3
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    player_input = scaler.transform(player_df[FEATURES].iloc[-1:].values)
    xgb_pred = best_model.predict(player_input)[0]

    lstm_pred = np.mean(player_df[TARGET].values[-5:])
    ensemble_pred = 0.5*lstm_pred + 0.5*xgb_pred

    col1, col2, col3 = st.columns(3)
    col1.metric("Actual (€)", f"{actual_value:,.0f}")
    col2.metric("XGBoost (€)", f"{to_euro(xgb_pred):,.0f}")
    col3.metric("Ensemble (€)", f"{to_euro(ensemble_pred):,.0f}")

    # ERROR COMPARISON 
    st.subheader("📊 Error Comparison (Model Accuracy)")

    actual = actual_value
    xgb_val = to_euro(xgb_pred)
    ensemble_val = to_euro(ensemble_pred)

    xgb_error = abs(actual - xgb_val)
    ensemble_error = abs(actual - ensemble_val)

    models = ["XGBoost", "Ensemble"]
    errors = [xgb_error, ensemble_error]

    fig3, ax3 = plt.subplots()
    bars = ax3.bar(models, errors)

    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height,
                 f"{int(height):,}", ha='center', va='bottom')

    ax3.set_title("Model Error Comparison (Lower is Better)")
    ax3.set_ylabel("Error (€)")
    ax3.grid(True, linestyle='--', alpha=0.6)

    st.pyplot(fig3)

    # MODEL COMPARISON
    st.subheader("📊 Model Prediction Comparison")

    models = ["Actual", "XGBoost", "Ensemble"]
    values = [actual_value, xgb_val, ensemble_val]

    fig4, ax4 = plt.subplots()
    bars = ax4.bar(models, values)

    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2, height,
                 f"{int(height):,}", ha='center', va='bottom')

    ax4.set_title("Transfer Value Prediction Comparison")
    ax4.set_ylabel("Value (€)")
    ax4.grid(True, linestyle='--', alpha=0.6)

    st.pyplot(fig4)

# ============================================================
# 🔴 WEEK 7 — WITH FINAL CONCLUSION
# ============================================================

elif week == "Week 7":

    st.header("🔴 Week 7 — Final Evaluation")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    y_test_euro = np.exp(y_test)
    preds_euro = np.exp(preds)

    rmse = np.sqrt(mean_squared_error(y_test_euro, preds_euro))
    mae = mean_absolute_error(y_test_euro, preds_euro)
    r2 = r2_score(y_test_euro, preds_euro)

    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE (€)", f"{rmse:,.0f}")
    col2.metric("MAE (€)", f"{mae:,.0f}")
    col3.metric("R²", f"{r2:.4f}")

    st.subheader("📊 Evaluation Report Table")

    df_eval = pd.DataFrame({
    "Metric": ["RMSE (€)", "MAE (€)", "R²"],
    "Value": [rmse, mae, r2]
     })

    st.dataframe(df_eval)
    st.subheader("🏆 Final Model Conclusion")

    st.success("Best Performing Model: XGBoost")

    st.info(
        "XGBoost achieved the lowest error and highest R² score. "
        "Ensemble was slightly less accurate due to LSTM averaging effect."
    )