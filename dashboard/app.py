# ============================================================
# FILE: dashboard/app.py
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

st.set_page_config(page_title="TransferIQ Player Valuation Dashboard", layout="wide")
st.title("⚽ TransferIQ Player Valuation Dashboard")

df = pd.read_csv("./data/processed/player_transfer_value_with_sentiment.csv")
df = df.sort_values(["player_name", "season_encoded"]).reset_index(drop=True)
season_map = {1: "2019/20", 2: "2020/21", 3: "2021/22", 4: "2022/23", 5: "2023/24"}
df["season_label"] = df["season_encoded"].map(season_map)

lstm_model = load_model("./dashboard/lstm_model.h5", compile=False)
xgb_model  = joblib.load("./dashboard/xgb_model.pkl")

lstm_features = ["market_value_eur","attacking_output_index","injury_burden_index",
                 "availability_rate","vader_compound_score","social_buzz_score"]
scaler = MinMaxScaler()
scaler.fit(df[lstm_features])

xgb_features = ["lstm_pred","current_age","age_decay_factor","position_encoded",
                 "season_encoded","attacking_output_index","injury_burden_index",
                 "availability_rate","goals_per90","assists_per90",
                 "goal_contributions_per90","minutes_played","pass_accuracy_pct",
                 "vader_compound_score","log_social_buzz"]

# ------------------------------------------------
# Sidebar
# ------------------------------------------------
st.sidebar.header("Player Selection")

player = st.sidebar.selectbox(
    "Select Player",
    sorted(df["player_name"].unique()),
    index=None,
    placeholder="Choose a player...",
)

future_season_choice = st.sidebar.selectbox(
    "Select Future Season",
    ["2024/25", "2025/26", "2026/27"],
)

analyse = st.sidebar.button("Show Analysis", type="primary", use_container_width=True)

if not player or not analyse:
    st.info("👈 Select a player and future season from the sidebar, then click **Show Analysis**.")
    st.stop()

future_seasons_all = ["2024/25", "2025/26", "2026/27"]
future_idx         = future_seasons_all.index(future_season_choice)

# ------------------------------------------------
# Filter player
# ------------------------------------------------
player_df     = df[df["player_name"] == player].sort_values("season_encoded").reset_index(drop=True)
latest_season = player_df["season_label"].iloc[-1]

# ------------------------------------------------
# Training vs Validation metrics
# ------------------------------------------------
@st.cache_data
def compute_train_val_metrics():
    try:
        enriched = pd.read_csv("./data/processed/lstm_enriched.csv")
        enriched = enriched.sort_values(["season_encoded", "player_name"]).reset_index(drop=True)
        enriched["log_social_buzz"] = np.log1p(enriched["social_buzz_score"])
        xgb_cols  = [f for f in xgb_features if f in enriched.columns]
        xgb_raw   = np.maximum(xgb_model.predict(enriched[xgb_cols]), 0)
        elite_mask = enriched["market_value_eur"].values >= 70e6
        ens_preds  = np.where(elite_mask,
            0.8 * enriched["lstm_pred"].values + 0.2 * xgb_raw,
            0.1 * enriched["lstm_pred"].values + 0.9 * xgb_raw)
        ens_preds = np.maximum(ens_preds, 0)
        enriched["ensemble_pred"] = ens_preds
        from sklearn.metrics import r2_score
        season_labels = {3: "2021/22", 4: "2022/23", 5: "2023/24"}
        records = []
        for season_enc, label in season_labels.items():
            rows = enriched[enriched["season_encoded"] == season_enc]
            if len(rows) == 0:
                continue
            actual = rows["market_value_eur"].values
            lstm_p = rows["lstm_pred"].values
            ens_p  = rows["ensemble_pred"].values
            split  = "Validation" if season_enc == 5 else "Training"
            records.append({
                "Season": label, "Split": split,
                "LSTM RMSE":     round(np.sqrt(mean_squared_error(actual, lstm_p)) / 1e6, 2),
                "Ensemble RMSE": round(np.sqrt(mean_squared_error(actual, ens_p))  / 1e6, 2),
                "LSTM MAE":      round(mean_absolute_error(actual, lstm_p) / 1e6, 2),
                "Ensemble MAE":  round(mean_absolute_error(actual, ens_p)  / 1e6, 2),
            })
        return pd.DataFrame(records)
    except FileNotFoundError:
        return None

metrics_df = compute_train_val_metrics()

# ------------------------------------------------
# Layout
# ------------------------------------------------
col1, col2 = st.columns(2)

# GRAPH 1 — Training vs Validation
with col1:
    if metrics_df is not None:
        fig_tv = go.Figure()
        fig_tv.add_trace(go.Scatter(x=metrics_df["Season"], y=metrics_df["LSTM RMSE"],
            mode="lines+markers", name="LSTM RMSE (€M)",
            line=dict(color="#b41f1f", width=2), marker=dict(size=8)))
        fig_tv.add_trace(go.Scatter(x=metrics_df["Season"], y=metrics_df["Ensemble RMSE"],
            mode="lines+markers", name="Ensemble RMSE (€M)",
            line=dict(color="#1a50a3", width=2), marker=dict(size=8)))
        fig_tv.add_trace(go.Scatter(x=metrics_df["Season"], y=metrics_df["LSTM MAE"],
            mode="lines+markers", name="LSTM MAE (€M)",
            line=dict(color="#b41f1f", width=1.5, dash="dot"), marker=dict(size=6)))
        fig_tv.add_trace(go.Scatter(x=metrics_df["Season"], y=metrics_df["Ensemble MAE"],
            mode="lines+markers", name="Ensemble MAE (€M)",
            line=dict(color="#1a50a3", width=1.5, dash="dot"), marker=dict(size=6)))
        val_seasons   = metrics_df[metrics_df["Split"] == "Validation"]["Season"].values
        train_seasons = metrics_df[metrics_df["Split"] == "Training"]["Season"].values
        if len(val_seasons) > 0 and len(train_seasons) > 0:
            fig_tv.add_annotation(x=train_seasons[-1], yref="paper", y=0.97,
                text="Training", showarrow=False,
                font=dict(size=12, color="#888"), xanchor="center",
                bgcolor="rgba(255,255,255,0.7)")
            fig_tv.add_annotation(x=val_seasons[0], yref="paper", y=0.97,
                text="Validation", showarrow=False,
                font=dict(size=12, color="#1a50a3"), xanchor="center",
                bgcolor="rgba(255,255,255,0.7)")
        fig_tv.update_layout(title="Training vs Validation: RMSE & MAE (€M)",
            xaxis_title="Season", yaxis_title="Error (€ Millions)", hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=-0.45, xanchor="center", x=0.5, font=dict(size=11)))
        st.plotly_chart(fig_tv, width='stretch')

# GRAPH 2 — Sentiment Trend
with col2:
    fig2 = px.line(player_df, x="season_label", y="vader_compound_score",
        markers=True, title="Public Sentiment Trend", color_discrete_sequence=["#1a7340"])
    fig2.update_layout(xaxis_title="Season", yaxis_title="Sentiment Score (−1 to 1)")
    st.plotly_chart(fig2, width='stretch')

# ------------------------------------------------
# LSTM multi-step forecast
# ------------------------------------------------
SEQUENCE_LENGTH = 3
n_features      = len(lstm_features)
player_scaled   = scaler.transform(player_df[lstm_features])
seq             = player_scaled[-SEQUENCE_LENGTH:].copy()
future_preds_lstm = []
future_preds_raw  = []

for step in range(3):
    pred_scaled = lstm_model.predict(seq.reshape(1, SEQUENCE_LENGTH, n_features), verbose=0)
    pad      = np.zeros((1, n_features - 1))
    pred_eur = float(scaler.inverse_transform(np.concatenate([pred_scaled, pad], axis=1))[0, 0])
    pred_eur = max(pred_eur, 0)
    future_preds_raw.append(pred_eur)
    last_val = float(player_df["market_value_eur"].iloc[-1]) if step == 0 else future_preds_lstm[-1]
    capped   = float(np.clip(pred_eur, last_val * 0.70, last_val * 1.40))
    future_preds_lstm.append(capped)
    new_row    = seq[-1].copy()
    new_row[0] = float(pred_scaled[0][0])
    seq        = np.vstack([seq[1:], new_row])

lstm_selected = future_preds_lstm[future_idx]
lstm_raw      = future_preds_raw[future_idx]

# GRAPH 3 — LSTM Forecast
with col2:
    connect_x = [latest_season] + future_seasons_all
    connect_y = [float(player_df["market_value_eur"].iloc[-1])] + future_preds_lstm
    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(x=player_df["season_label"], y=player_df["market_value_eur"],
        mode="lines+markers", name="Actual", line=dict(color="#1a7340")))
    fig4.add_trace(go.Scatter(x=connect_x, y=connect_y,
        mode="lines+markers", name="LSTM Forecast", line=dict(color="#b41f1f", dash="dash")))
    fig4.update_layout(title="LSTM Multi-Step Forecast",
        xaxis_title="Season", yaxis_title="Value (EUR)", hovermode="x unified")
    st.plotly_chart(fig4, width='stretch')

st.sidebar.metric(label=f"LSTM Prediction ({future_season_choice})", value=f"€{lstm_selected:,.0f}")

# ------------------------------------------------
# XGBoost Ensemble prediction
# ------------------------------------------------
latest_row = player_df.iloc[-1]
feature_dict = {
    "lstm_pred":                lstm_raw,
    "current_age":              latest_row.get("current_age", np.nan),
    "age_decay_factor":         latest_row.get("age_decay_factor", np.nan),
    "position_encoded":         latest_row.get("position_encoded", np.nan),
    "season_encoded":           5 + future_idx + 1,
    "attacking_output_index":   latest_row.get("attacking_output_index", np.nan),
    "injury_burden_index":      latest_row.get("injury_burden_index", np.nan),
    "availability_rate":        latest_row.get("availability_rate", np.nan),
    "goals_per90":              latest_row.get("goals_per90", np.nan),
    "assists_per90":            latest_row.get("assists_per90", np.nan),
    "goal_contributions_per90": latest_row.get("goal_contributions_per90", np.nan),
    "minutes_played":           latest_row.get("minutes_played", np.nan),
    "pass_accuracy_pct":        latest_row.get("pass_accuracy_pct", np.nan),
    "vader_compound_score":     latest_row.get("vader_compound_score", np.nan),
    "log_social_buzz":          float(np.log1p(latest_row.get("social_buzz_score", 0))),
}
xgb_input_cols = [f for f in xgb_features if f in feature_dict]
xgb_input      = pd.DataFrame([[feature_dict[f] for f in xgb_input_cols]], columns=xgb_input_cols)
xgb_raw    = float(xgb_model.predict(xgb_input)[0])
xgb_raw    = max(xgb_raw, 0)

last_known = float(player_df["market_value_eur"].iloc[-1])
if last_known >= 70e6:
    final_pred = 0.8 * lstm_selected + 0.2 * xgb_raw
else:
    final_pred = 0.1 * lstm_selected + 0.9 * xgb_raw

final_pred = max(final_pred, 0)
final_pred = float(np.clip(final_pred, last_known * 0.60, last_known * 1.40))

st.sidebar.metric(label=f"Ensemble Value ({future_season_choice})", value=f"€{final_pred:,.0f}")

diff = final_pred - lstm_selected
if diff >= 0:
    st.sidebar.success(f"Ensemble adjusted by +€{diff:,.0f}")
else:
    st.sidebar.info(f"Ensemble adjusted by −€{abs(diff):,.0f}")

# GRAPH 4 — Model Comparison
with col1:
    actual_value  = float(player_df["market_value_eur"].iloc[-1])
    comparison_df = pd.DataFrame({
        "Type":  ["Actual (last season)", "LSTM Prediction", "Ensemble Prediction"],
        "Value": [actual_value, lstm_selected, final_pred],
    })
    fig_comp = px.bar(comparison_df, x="Type", y="Value", text_auto=".3s",
        title=f"Model Comparison ({future_season_choice})", color="Type",
        color_discrete_map={"Actual (last season)": "#1a7340",
                            "LSTM Prediction": "#b41f1f",
                            "Ensemble Prediction": "#1a50a3"})
    fig_comp.update_layout(yaxis_title="Market Value (EUR)", xaxis_title="", showlegend=False)
    st.plotly_chart(fig_comp, width='stretch')