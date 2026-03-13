import streamlit as st
import pandas as pd
import plotly.express as px

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="TransferIQ Dashboard", layout="wide")

# ---------- LOAD DATA ----------
df = pd.read_csv("player_data.csv")

# ---------- SIDEBAR ----------
st.sidebar.title("TransferIQ")

section = st.sidebar.radio(
    "Navigate",
    [
        "Player Overview",
        "Performance Analysis",
        "Injury Analysis",
        "Sentiment Analysis",
        "Player Comparison",
        "League Insights"
    ]
)

player = st.sidebar.selectbox("Select Player", df["player_name"].unique())
player_data = df[df["player_name"] == player].iloc[0]

# ---------- TITLE ----------
st.title("TransferIQ Football Analytics Dashboard")

# =========================================================
# PLAYER OVERVIEW
# =========================================================

if section == "Player Overview":

    st.subheader("Player Profile")

    col1,col2,col3,col4 = st.columns(4)

    col1.metric("Age", player_data["current_age"])
    col2.metric("Position", player_data["position"])
    col3.metric("Team", player_data["team"])
    col4.metric("Market Value (€)", int(player_data["market_value_eur"]))

    st.write("Career Stage:", player_data["career_stage"])

# =========================================================
# PERFORMANCE ANALYSIS
# =========================================================

elif section == "Performance Analysis":

    st.subheader("Performance Metrics")

    perf_data = pd.DataFrame({
        "Metric":[
            "Goals/90",
            "Assists/90",
            "Shots/90",
            "Dribbles/90",
            "Defensive Actions"
        ],
        "Value":[
            player_data["goals_per90"],
            player_data["assists_per90"],
            player_data["shots_per90"],
            player_data["dribbles_per90"],
            player_data["defensive_actions_per90"]
        ]
    })

    # ----- LINE GRAPH -----
    fig = px.line(
        perf_data,
        x="Metric",
        y="Value",
        markers=True,
        title="Player Performance Trend"
    )

    st.plotly_chart(fig,use_container_width=True)

# =========================================================
# INJURY ANALYSIS
# =========================================================

elif section == "Injury Analysis":

    st.subheader("Injury Risk Analysis")

    injury_df = pd.DataFrame({
        "Metric":[
            "Total Injuries",
            "Days Injured",
            "Matches Missed",
            "Injury Frequency"
        ],
        "Value":[
            player_data["total_injuries"],
            player_data["total_days_injured"],
            player_data["total_matches_missed"],
            player_data["injury_frequency"]
        ]
    })

    fig2 = px.bar(
        injury_df,
        x="Metric",
        y="Value",
        title="Injury Impact Statistics"
    )

    st.plotly_chart(fig2,use_container_width=True)

# =========================================================
# SENTIMENT ANALYSIS
# =========================================================

elif section == "Sentiment Analysis":

    st.subheader("Public Sentiment")

    sent_df = pd.DataFrame({
        "Sentiment":[
            "Positive",
            "Negative",
            "Neutral"
        ],
        "Count":[
            player_data["positive_count"],
            player_data["negative_count"],
            player_data["neutral_count"]
        ]
    })

    fig3 = px.pie(
        sent_df,
        names="Sentiment",
        values="Count",
        title="Fan Sentiment Distribution"
    )

    st.plotly_chart(fig3,use_container_width=True)

# =========================================================
# PLAYER COMPARISON
# =========================================================

elif section == "Player Comparison":

    st.subheader("Compare Players")

    p1 = st.selectbox("Player 1", df["player_name"].unique())
    p2 = st.selectbox("Player 2", df["player_name"].unique())

    p1_data = df[df["player_name"] == p1].iloc[0]
    p2_data = df[df["player_name"] == p2].iloc[0]

    comp = pd.DataFrame({
        "Metric":[
            "Goals/90",
            "Assists/90",
            "Shots/90"
        ],
        p1:[
            p1_data["goals_per90"],
            p1_data["assists_per90"],
            p1_data["shots_per90"]
        ],
        p2:[
            p2_data["goals_per90"],
            p2_data["assists_per90"],
            p2_data["shots_per90"]
        ]
    })

    fig4 = px.bar(
        comp,
        x="Metric",
        y=[p1,p2],
        barmode="group",
        title="Performance Comparison"
    )

    st.plotly_chart(fig4,use_container_width=True)

# =========================================================
# LEAGUE INSIGHTS
# =========================================================

elif section == "League Insights":

    st.subheader("Top Market Value Players")

    top_players = df.sort_values(
        "market_value_eur",
        ascending=False
    ).head(10)

    fig5 = px.bar(
        top_players,
        x="player_name",
        y="market_value_eur",
        title="Top 10 Most Valuable Players"
    )

    st.plotly_chart(fig5,use_container_width=True)