import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page settings
st.set_page_config(page_title="TransferIQ Player Analytics", layout="wide")

# Load dataset
df = pd.read_csv("final_feature_dataset.csv", encoding="latin1")

st.title("⚽ TransferIQ Player Analytics Dashboard")

st.write("Explore football player performance, injuries, sentiment and market value trends.")

st.divider()

# Player selector
players = df["player_name"].unique()
player = st.selectbox("Select a Player", players)

player_data = df[df["player_name"] == player]

st.divider()

# Player Profile
st.subheader("Player Profile")

col1, col2, col3 = st.columns(3)

col1.metric("Position", player_data["position"].iloc[0])

col2.metric(
    "Market Value (€)",
    f"€{int(player_data['market_value_eur'].iloc[0]):,}"
)

col3.metric(
    "Availability Rate",
    round(player_data["availability_rate"].iloc[0], 2)
)

st.divider()

# Market Value Trend
st.subheader("Market Value Over Seasons")

fig = px.line(
    player_data,
    x="season",
    y="market_value_eur",
    markers=True,
    title="Market Value Trend"
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# Performance Radar Chart
st.subheader("Player Performance Radar")

metrics = [
    "goals_per90",
    "assists_per90",
    "shots_per90",
    "dribbles_per90",
    "defensive_actions_per90"
]

values = player_data[metrics].iloc[0].values

fig_radar = go.Figure()

fig_radar.add_trace(go.Scatterpolar(
    r=values,
    theta=metrics,
    fill='toself',
    name=player
))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    showlegend=False
)

st.plotly_chart(fig_radar, use_container_width=True)

st.divider()

# Injury Analysis
st.subheader("Injury Impact Analysis")

injury_data = {
    "Metric": ["Total Injuries", "Days Injured", "Matches Missed"],
    "Value": [
        player_data["total_injuries"].iloc[0],
        player_data["total_days_injured"].iloc[0],
        player_data["total_matches_missed"].iloc[0]
    ]
}

injury_df = pd.DataFrame(injury_data)

fig2 = px.bar(
    injury_df,
    x="Metric",
    y="Value",
    color="Metric",
    title="Injury Impact"
)

st.plotly_chart(fig2, use_container_width=True)

st.divider()

# Sentiment Score
st.subheader("Social Sentiment Score")

sentiment = player_data["overall_sentiment"].iloc[0]

st.progress(float((sentiment + 1) / 2))

st.write("Sentiment Score:", round(sentiment, 3))

st.divider()

# Player Comparison Section
st.subheader("Player Comparison Tool")

player2 = st.selectbox("Select another player to compare", players)

player2_data = df[df["player_name"] == player2]

comparison_metrics = [
    "goals_per90",
    "assists_per90",
    "shots_per90",
    "dribbles_per90",
    "defensive_actions_per90"
]

comparison_df = pd.DataFrame({
    "Metric": comparison_metrics,
    player: player_data[comparison_metrics].iloc[0].values,
    player2: player2_data[comparison_metrics].iloc[0].values
})

fig_compare = px.bar(
    comparison_df,
    x="Metric",
    y=[player, player2],
    barmode="group",
    title="Player Performance Comparison"
)

st.plotly_chart(fig_compare, use_container_width=True)