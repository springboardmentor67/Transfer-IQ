import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Football Scout AI", layout="wide")

# -------------------------------
# SESSION STATE
# -------------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

# -------------------------------
# HOME PAGE
# -------------------------------
if st.session_state.page == "home":

    st.markdown("""
    <div style="text-align:center;padding-top:120px">
        <h1 style="font-size:55px;">⚽ Football Scout AI</h1>
        <h3>Smart Player Analysis & Market Value Prediction</h3>
        <p style="color:gray;">AI-powered football analytics dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🚀 Enter Dashboard", use_container_width=True):
        st.session_state.page = "app"

# -------------------------------
# MAIN APP
# -------------------------------
elif st.session_state.page == "app":

    st.markdown("""
    <style>
    .card {
        padding: 18px;
        border-radius: 14px;
        color: white;
        text-align: center;
        transition: 0.3s;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
    }
    .card:hover {
        transform: scale(1.05);
    }
    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white;
        font-size: 20px;
        text-align: center;
    }
    .insight-pos {
        background: #0f5132;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    .insight-neg {
        background: #842029;
        padding: 15px;
        border-radius: 10px;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("⚽ Football Scout AI Dashboard")

    # -------------------------------
    # DATA 
    # -------------------------------
    uploaded_file = st.file_uploader("Upload Dataset", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        try:
            df = pd.read_csv("player_transfer_value_with_sentiment.csv")
        except:
            st.warning("⚠️ Please upload a dataset to continue.")
            st.stop()

    # -------------------------------
    # SIDEBAR
    # -------------------------------
    st.sidebar.title("Controls")
    st.sidebar.info("Model: AI-Based Market Value Prediction")
    
    if st.sidebar.button("⬅ Back"):
        st.session_state.page = "home"

    player = st.sidebar.selectbox("Player", df['player_name'].unique())
    player_df = df[df['player_name'] == player]

    season = st.sidebar.selectbox("Season", player_df['season'].unique())
    season_df = player_df[player_df['season'] == season]

    st.subheader(f"{player} | {season}")

    # -------------------------------
    # CARDS
    # -------------------------------
    col1, col2, col3, col4, col5 = st.columns(5)

    col1.markdown(f"<div class='card' style='background:#6f42c1'>💰<br><h3>€{season_df['market_value_eur'].values[0]:,.0f}</h3>Market Value</div>", unsafe_allow_html=True)
    col2.markdown(f"<div class='card' style='background:#0d6efd'>🎂<br><h3>{int(season_df['current_age'].values[0])}</h3>Age</div>", unsafe_allow_html=True)
    col3.markdown(f"<div class='card' style='background:#198754'>⚽<br><h3>{season_df['position'].values[0]}</h3>Position</div>", unsafe_allow_html=True)
    col4.markdown(f"<div class='card' style='background:#fd7e14'>🥅<br><h3>{int(season_df['goals'].values[0])}</h3>Goals</div>", unsafe_allow_html=True)
    col5.markdown(f"<div class='card' style='background:#dc3545'>📊<br><h3>{season_df['availability_rate'].values[0]:.2f}</h3>Availability</div>", unsafe_allow_html=True)

    st.markdown("---")

    # -------------------------------
    # GRAPH 1
    # -------------------------------
    fig1 = px.line(player_df, x='season', y='market_value_eur', markers=True, template="plotly_dark")
    st.plotly_chart(fig1, use_container_width=True)

    # -------------------------------
    # GRAPH 2
    # -------------------------------
    fig2 = px.line(player_df, x='season', y='vader_compound_score', markers=True, template="plotly_dark")
    st.plotly_chart(fig2, use_container_width=True)

    # -------------------------------
    # GRAPH 3
    # -------------------------------
    sentiment_data = pd.DataFrame({
        "Type": ["Positive", "Neutral", "Negative"],
        "Count": [
            season_df['positive_count'].values[0],
            season_df['neutral_count'].values[0],
            season_df['negative_count'].values[0]
        ]
    })

    fig3 = px.bar(sentiment_data, x="Type", y="Count", color="Type", template="plotly_dark")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # -------------------------------
    # 🔮 SIMPLE PREDICTION
    # -------------------------------
    values = player_df[['market_value_eur']].dropna().values

    if len(values) > 0:
        predicted_value = values[-1][0] * 1.05
    else:
        predicted_value = 0

    st.markdown(f"""
    <div class="prediction-box">
    🔮 Predicted Next Market Value<br>
    <h2>€{predicted_value:,.2f}</h2>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------
    # INSIGHT
    # -------------------------------
    sentiment = season_df['vader_compound_score'].values[0]

    if sentiment > 0:
        st.markdown(f"<div class='insight-pos'>🔥 Positive sentiment ({sentiment:.2f}) — Value likely to rise</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='insight-neg'>⚠️ Negative sentiment ({sentiment:.2f}) — Risk in valuation</div>", unsafe_allow_html=True)
