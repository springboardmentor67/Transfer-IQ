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
# Layout
# ------------------------------------------------
col1, col2 = st.columns(2)

# ------------------------------------------------
# GRAPH 1 — Market Value Trend (historical)
# Simple, clean, easy to explain
# ------------------------------------------------
with col1:
    fig1 = go.Figure()

    fig1.add_trace(go.Scatter(
        x=player_df["season_label"],
        y=player_df["market_value_eur"],
        mode="lines+markers",
        name="Market Value",
        line=dict(color="#1a7340", width=2),
        marker=dict(size=9),
        fill="tozeroy",
        fillcolor="rgba(26,115,64,0.08)",
    ))

    # Annotate each point with value
    for _, row in player_df.iterrows():
        fig1.add_annotation(
            x=row["season_label"],
            y=row["market_value_eur"],
            text=f"€{row['market_value_eur']/1e6:.1f}M",
            showarrow=False,
            yshift=14,
            font=dict(size=11, color="#1a7340"),
        )

    fig1.update_layout(
        title="Historical Market Value",
        xaxis_title="Season",
        yaxis_title="Value (EUR)",
        hovermode="x unified",
        showlegend=False,
    )
    st.plotly_chart(fig1, width='stretch')

# ------------------------------------------------
# GRAPH 2 — Sentiment Trend
# ------------------------------------------------
with col2:
    fig2 = px.line(player_df, x="season_label", y="vader_compound_score",
        markers=True, title="Public Sentiment Trend",
        color_discrete_sequence=["#1a7340"])
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

# Ensemble predictions for all 3 future seasons
latest_row = player_df.iloc[-1]
last_known = float(player_df["market_value_eur"].iloc[-1])

ensemble_preds = []
for s_idx in range(3):
    fd = {
        "lstm_pred":                future_preds_raw[s_idx],
        "current_age":              latest_row.get("current_age", np.nan),
        "age_decay_factor":         latest_row.get("age_decay_factor", np.nan),
        "position_encoded":         latest_row.get("position_encoded", np.nan),
        "season_encoded":           5 + s_idx + 1,
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
    xgb_cols  = [f for f in xgb_features if f in fd]
    xgb_inp   = pd.DataFrame([[fd[f] for f in xgb_cols]], columns=xgb_cols)
    xgb_raw_v = float(xgb_model.predict(xgb_inp)[0])
    xgb_raw_v = max(xgb_raw_v, 0)
    lstm_v    = future_preds_lstm[s_idx]
    if last_known >= 70e6:
        ens_v = 0.8 * lstm_v + 0.2 * xgb_raw_v
    else:
        ens_v = 0.1 * lstm_v + 0.9 * xgb_raw_v
    ens_v = max(ens_v, 0)
    ens_v = float(np.clip(ens_v, last_known * 0.60, last_known * 1.40))
    ensemble_preds.append(ens_v)

lstm_selected  = future_preds_lstm[future_idx]
final_pred     = ensemble_preds[future_idx]

# ------------------------------------------------
# GRAPH 3 — LSTM Forecast (forecast only, no historical line)
# ------------------------------------------------
with col2:
    last_actual   = float(player_df["market_value_eur"].iloc[-1])
    connect_x     = [latest_season] + future_seasons_all
    connect_y_l   = [last_actual] + future_preds_lstm
    connect_y_e   = [last_actual] + ensemble_preds

    fig4 = go.Figure()

    # LSTM forecast line
    fig4.add_trace(go.Scatter(
        x=connect_x, y=connect_y_l,
        mode="lines+markers", name="LSTM Forecast",
        line=dict(color="#b41f1f", dash="dash", width=2),
        marker=dict(size=7),
    ))

    # Ensemble forecast line
    fig4.add_trace(go.Scatter(
        x=connect_x, y=connect_y_e,
        mode="lines+markers", name="Ensemble Forecast",
        line=dict(color="#1a50a3", width=2),
        marker=dict(size=7),
    ))

    # Highlight selected season — use add_shape (add_vline breaks on categorical x-axis)
    fig4.add_shape(
        type="line", xref="x", yref="paper",
        x0=future_season_choice, x1=future_season_choice,
        y0=0, y1=1,
        line=dict(color="#888", width=1.5, dash="dot"),
    )
    fig4.add_annotation(
        x=future_season_choice, yref="paper", y=1.04,
        text="Selected", showarrow=False,
        font=dict(size=11, color="#009933"), xanchor="center",
    )

    fig4.update_layout(
        title="Market Value Forecast (Future Seasons)",
        xaxis_title="Season", yaxis_title="Value (EUR)", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                    xanchor="center", x=0.5),
    )
    st.plotly_chart(fig4, width='stretch')

# Sidebar metrics
st.sidebar.metric(label=f"LSTM Prediction ({future_season_choice})",    value=f"€{lstm_selected:,.0f}")
st.sidebar.metric(label=f"Ensemble Value ({future_season_choice})",     value=f"€{final_pred:,.0f}")

diff = final_pred - lstm_selected
if diff >= 0:
    st.sidebar.success(f"Ensemble adjusted by +€{diff:,.0f}")
else:
    st.sidebar.info(f"Ensemble adjusted by −€{abs(diff):,.0f}")

# ------------------------------------------------
# GRAPH 4 — Model Comparison Bar Chart
# ------------------------------------------------
with col1:
    comp_data = pd.DataFrame({
        "Season": future_seasons_all * 2,
        "Model":  ["LSTM"] * 3 + ["Ensemble"] * 3,
        "Value":  future_preds_lstm + ensemble_preds,
    })

    fig_comp = px.bar(
        comp_data, x="Season", y="Value", color="Model",
        barmode="group", text_auto=".3s",
        title=f"Model Comparison ({future_season_choice})",
        color_discrete_map={"LSTM": "#b41f1f", "Ensemble": "#1a50a3"},
    )

    # Highlight selected season
    fig_comp.add_vrect(
        x0=future_idx - 0.4, x1=future_idx + 0.4,
        fillcolor="rgba(26,115,64,0.1)",
        layer="below", line_width=0,
    )

    fig_comp.update_layout(
        yaxis_title="Market Value (EUR)", xaxis_title="",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25,
                    xanchor="center", x=0.5),
        hovermode="x unified",
    )
    st.plotly_chart(fig_comp, width='stretch')