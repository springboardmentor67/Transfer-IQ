import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ----------------------------
# Load Dataset
# ----------------------------

@st.cache_data
def load_data():
    return pd.read_csv("Football player valuation analyzer\player_transfer_value_with_sentiment.csv")

df = load_data()

st.title("⚽ TransferIQ Player Intelligence Dashboard")

# ----------------------------
# Player Selection
# ----------------------------

players = df["player_name"].unique()

player_name = st.selectbox("Select Player", players)

player_df = df[df["player_name"] == player_name]
latest = player_df.sort_values("season").iloc[-1]

# ----------------------------
# Player Overview
# ----------------------------

st.header("Player Profile")

col1, col2, col3 = st.columns(3)

col1.metric("Age", latest["current_age"])
col2.metric("Position", latest["position"])
col3.metric("Team", latest["team"])

col1.metric("Market Value (€)", int(latest["market_value_eur"]))
col2.metric("Career Stage", latest["career_stage"])
col3.metric("Availability Rate", round(latest["availability_rate"],2))

# ----------------------------
# Market Value Trend
# ----------------------------

st.header("Market Value Trend")

fig = px.line(
    player_df,
    x="season",
    y="market_value_eur",
    markers=True,
    title="Market Value Over Time"
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Performance Stats
# ----------------------------

st.header("Performance Statistics")

perf_cols = [
    "goals",
    "assists",
    "shots",
    "passes_total",
    "passes_complete",
    "tackles_total",
    "interceptions",
    "dribbles"
]

fig = px.bar(
    x=perf_cols,
    y=[latest[col] for col in perf_cols],
    labels={"x":"Metric","y":"Value"},
    title="Performance Metrics"
)

st.plotly_chart(fig)

# ----------------------------
# Per 90 Metrics
# ----------------------------

st.header("Per 90 Performance")

per90_cols = [
    "goals_per90",
    "assists_per90",
    "shots_per90",
    "goal_contributions_per90",
    "defensive_actions_per90",
    "dribbles_per90"
]

fig = px.bar(
    x=per90_cols,
    y=[latest[c] for c in per90_cols],
    title="Per 90 Contributions"
)

st.plotly_chart(fig)

# ----------------------------
# Passing Analysis
# ----------------------------

st.header("Passing Analysis")

fig = px.pie(
    values=[
        latest["passes_complete"],
        latest["passes_total"] - latest["passes_complete"]
    ],
    names=["Completed","Failed"],
    title="Pass Completion"
)

st.plotly_chart(fig)

# ----------------------------
# Defensive Metrics
# ----------------------------

st.header("Defensive Metrics")

defensive_cols = [
    "tackles_total",
    "tackles_won",
    "interceptions",
    "fouls_committed"
]

fig = px.bar(
    x=defensive_cols,
    y=[latest[c] for c in defensive_cols],
    title="Defensive Contribution"
)

st.plotly_chart(fig)

# ----------------------------
# Injury Analysis
# ----------------------------

st.header("Injury History")

injury_cols = [
    "total_injuries",
    "total_days_injured",
    "total_matches_missed"
]

fig = px.bar(
    x=injury_cols,
    y=[latest[c] for c in injury_cols],
    title="Injury Impact"
)

st.plotly_chart(fig)

st.write("Most Common Injury:", latest["most_common_injury"])

# ----------------------------
# Social Media Sentiment
# ----------------------------

st.header("Social Sentiment Analysis")

sentiment_cols = [
    "positive_count",
    "negative_count",
    "neutral_count"
]

fig = px.pie(
    values=[latest[c] for c in sentiment_cols],
    names=["Positive","Negative","Neutral"],
    title="Fan Sentiment"
)

st.plotly_chart(fig)

# ----------------------------
# Sentiment Scores
# ----------------------------

st.subheader("Sentiment Scores")

fig = px.bar(
    x=[
        "VADER Positive",
        "VADER Negative",
        "VADER Compound",
        "TextBlob Polarity",
        "TextBlob Subjectivity"
    ],
    y=[
        latest["vader_positive_score"],
        latest["vader_negative_score"],
        latest["vader_compound_score"],
        latest["tb_polarity"],
        latest["tb_subjectivity"]
    ],
)

st.plotly_chart(fig)

# ----------------------------
# Transfer Value Intelligence
# ----------------------------

st.header("Transfer Market Intelligence")

market_cols = [
    "social_buzz_score",
    "transfer_attractiveness_score",
    "tweet_engagement_rate",
]

fig = px.bar(
    x=market_cols,
    y=[latest[c] for c in market_cols],
    title="Market Influence Indicators"
)

st.plotly_chart(fig)

# ----------------------------
# Radar Chart (Overall Player Profile)
# ----------------------------

st.header("Player Radar")

radar_features = [
    "goals_per90",
    "assists_per90",
    "shots_per90",
    "defensive_actions_per90",
    "pass_accuracy_pct",
    "dribbles_per90"
]

values = [latest[f] for f in radar_features]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=values,
    theta=radar_features,
    fill="toself",
    name=player_name
))

fig.update_layout(
    polar=dict(radialaxis=dict(visible=True)),
    showlegend=False
)

st.plotly_chart(fig)

# ----------------------------
# Raw Data Table
# ----------------------------

st.header("Raw Player Data")


st.dataframe(player_df)
