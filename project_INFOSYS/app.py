import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


st.title("⚽ AI Football Transfer Value Dashboard")

# LOADING DATASET
df = pd.read_csv("player_transfer_value_with_sentiment.csv")

st.subheader("Dataset Preview")
st.write(df.head())


#EACH PLAYER SEARCH

st.subheader("🔎 Player Search")

player_name = st.text_input("Enter Player Name")

if player_name:
    player_data = df[df["player_name"].str.contains(player_name, case=False)]

    if not player_data.empty:
        st.write(player_data)
    else:
        st.write("Player not found")


# SENTIMENT PIE CHART

st.subheader("📊 Sentiment Distribution")

sentiment_counts = df["sentiment_label"].value_counts()

fig1, ax1 = plt.subplots()
ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
st.pyplot(fig1)

# MARKET VALUE ANALYSIS

st.subheader("💰 Market Value Distribution")

fig2, ax2 = plt.subplots()
ax2.hist(df["market_value_eur"], bins=20)
ax2.set_xlabel("Market Value (EUR)")
ax2.set_ylabel("Number of Players")
st.pyplot(fig2)


# GOALS VS ASSISTS

st.subheader("📈 Goals vs Assists")

fig3, ax3 = plt.subplots()
ax3.scatter(df["goals"], df["assists"])
ax3.set_xlabel("Goals")
ax3.set_ylabel("Assists")
st.pyplot(fig3)


# TOP TRANSFER TARGETS

st.subheader("🏆 Top Transfer Targets")

top_players = df.sort_values("transfer_attractiveness_score", ascending=False).head(10)

st.write(top_players[[
    "player_name",
    "team",
    "goals",
    "assists",
    "market_value_eur",
    "transfer_attractiveness_score"
]])