import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import mean_squared_error

st.set_page_config(page_title="TransferIQ Player Analytics", layout="wide")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("final_feature_dataset_with_predictions.csv", encoding="latin1")

st.title("⚽ TransferIQ Dashboard")
st.markdown("### 📊 AI-Powered Football Player Value Analysis")

# -------------------------------
# OVERVIEW
# -------------------------------
st.markdown("## 📌 Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Players", len(df["player_name"].unique()))
col2.metric("Seasons", len(df["season"].unique()))
col3.metric("Model", "LSTM + XGBoost")

st.divider()

# -------------------------------
# PLAYER SELECTION
# -------------------------------
st.markdown("## 👤 Player Analysis")

players = df["player_name"].unique()
player = st.selectbox("Select Player", players)

years = sorted(df["season"].unique())
year = st.selectbox("Select Season", years)

player_data = df[(df["player_name"] == player) & (df["season"] == year)]
full_player_data = df[df["player_name"] == player]

# -------------------------------
# PROFILE
# -------------------------------
st.subheader("Player Profile")

if len(player_data) > 0:
    c1, c2, c3 = st.columns(3)
    c1.metric("Position", player_data["position"].iloc[0])
    c2.metric("Market Value", f"€{int(player_data['market_value_eur'].iloc[0]):,}")
    c3.metric("Availability", round(player_data["availability_rate"].iloc[0], 2))

st.divider()

# -------------------------------
# MARKET TREND
# -------------------------------
st.subheader("📈 Market Value Trend")
fig = px.line(full_player_data, x="season", y="market_value_eur", markers=True)
st.plotly_chart(fig, width="stretch")

# -------------------------------
# RADAR
# -------------------------------
st.subheader("⚽ Performance Radar")

metrics = ["goals_per90","assists_per90","shots_per90","dribbles_per90","defensive_actions_per90"]

if len(player_data) > 0:
    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=player_data[metrics].iloc[0].values,
        theta=metrics,
        fill='toself'
    ))
    st.plotly_chart(fig_radar, width="stretch")

# -------------------------------
# INJURY
# -------------------------------
st.subheader("🏥 Injury Analysis")

if len(player_data) > 0:
    injury_df = pd.DataFrame({
        "Metric":["Injuries","Days","Matches Missed"],
        "Value":[
            player_data["total_injuries"].iloc[0],
            player_data["total_days_injured"].iloc[0],
            player_data["total_matches_missed"].iloc[0]
        ]
    })
    fig2 = px.bar(injury_df, x="Metric", y="Value", color="Metric")
    st.plotly_chart(fig2, width="stretch")

# -------------------------------
# SENTIMENT
# -------------------------------
st.subheader("💬 Sentiment Score")

if len(player_data) > 0:
    s = player_data["overall_sentiment"].iloc[0]
    st.progress(float((s+1)/2))
    st.write("Score:", round(s,3))

st.divider()

# -------------------------------
# COMPARISON
# -------------------------------
st.subheader("🔍 Player Comparison")

player2 = st.selectbox("Compare with", players)
player2_data = df[df["player_name"] == player2]

comp_df = pd.DataFrame({
    "Metric":metrics,
    player:full_player_data[metrics].iloc[0].values,
    player2:player2_data[metrics].iloc[0].values
})

fig_compare = px.bar(comp_df, x="Metric", y=[player,player2], barmode="group")
st.plotly_chart(fig_compare, width="stretch")

# ======================================================
# MODEL ANALYSIS
# ======================================================
st.divider()
st.markdown("## 🤖 Model Performance")

valid_df = df.dropna(subset=["lstm_pred","xgb_pred","final_pred"]).copy()
valid_df = valid_df.sort_values(by="season")

rmse_lstm = np.sqrt(mean_squared_error(valid_df["market_value_eur"], valid_df["lstm_pred"]))
rmse_xgb = np.sqrt(mean_squared_error(valid_df["market_value_eur"], valid_df["xgb_pred"]))
rmse_final = np.sqrt(mean_squared_error(valid_df["market_value_eur"], valid_df["final_pred"]))

c1, c2, c3 = st.columns(3)
c1.metric("LSTM RMSE", round(rmse_lstm,2))
c2.metric("XGBoost RMSE", round(rmse_xgb,2))
c3.metric("Ensemble RMSE", round(rmse_final,2))

fig_model = px.bar(
    x=["LSTM","XGBoost","Ensemble"],
    y=[rmse_lstm, rmse_xgb, rmse_final],
    text=[round(rmse_lstm,2),round(rmse_xgb,2),round(rmse_final,2)]
)
fig_model.update_traces(textposition="outside")
st.plotly_chart(fig_model, width="stretch")

best_model = min(
    {"LSTM": rmse_lstm, "XGBoost": rmse_xgb, "Ensemble": rmse_final},
    key=lambda x: {"LSTM": rmse_lstm, "XGBoost": rmse_xgb, "Ensemble": rmse_final}[x]
)

st.success(f"🏆 Best Model: {best_model}")

# ======================================================
# FINAL PREDICTIONS
# ======================================================
st.divider()
st.markdown("## 🔮 Final Predictions")

results_df = valid_df[[
    "player_name",
    "market_value_eur",
    "lstm_pred",
    "xgb_pred",
    "final_pred"
]].copy()

results_df.columns = ["Player","Actual","LSTM","XGBoost","Final"]

st.dataframe(results_df.head(10), width="stretch")

st.subheader("🌟 Top Players")
st.dataframe(results_df.sort_values("Final", ascending=False).head(5))

fig_final = go.Figure()
fig_final.add_trace(go.Scatter(
    x=valid_df["season"],
    y=valid_df["market_value_eur"],
    name="Actual",
    mode="lines+markers"
))
fig_final.add_trace(go.Scatter(
    x=valid_df["season"],
    y=valid_df["final_pred"],
    name="Predicted",
    mode="lines+markers"
))
st.plotly_chart(fig_final, width="stretch")

# ======================================================
# DOWNLOAD (FIXED 🔥)
# ======================================================
st.download_button(
    "⬇ Download Predictions",
    valid_df.to_csv(index=False),
    "predictions.csv"
)

# ======================================================
# INSIGHTS
# ======================================================
st.markdown("## 💡 Insights")

st.info("""
✔ Performance strongly affects value  
✔ Sentiment improves prediction  
✔ Ensemble model gives best stability  
✔ Useful for transfer decision making  
""")