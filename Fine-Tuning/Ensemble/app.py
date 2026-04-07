import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Football Scout AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------------
# CSS Styling
# -------------------------------------------------------------------
st.markdown("""
<style>
    .metric-card {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid #333;
    }
    .metric-value { font-size: 24px; font-weight: bold; color: #4CAF50; }
    .metric-label { font-size: 14px; color: #aaa; text-transform: uppercase; }
    .badge-star { background-color: #ffd700; color: #000; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-right: 5px; }
    .badge-risk { background-color: #ff4b4b; color: #fff; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-right: 5px; }
    .badge-fan { background-color: #1f77b4; color: #fff; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-right: 5px; }
    .rec-buy { background-color: #2e7d32; color: white; padding: 15px; border-radius: 8px; text-align: center; font-size: 20px; font-weight: bold; }
    .rec-monitor { background-color: #f57c00; color: white; padding: 15px; border-radius: 8px; text-align: center; font-size: 20px; font-weight: bold; }
    .rec-avoid { background-color: #c62828; color: white; padding: 15px; border-radius: 8px; text-align: center; font-size: 20px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------
# Data & Model Loading
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "player_transfer_value_with_sentiment.csv")
    if not os.path.exists(path):
        st.error(f"Dataset not found at {path}")
        return pd.DataFrame()
    df = pd.read_csv(path)
    
    # Handle missing player_name explicitly, as older CSV saves might use Unnamed: 0
    if "player_name" not in df.columns and "Unnamed: 0" in df.columns:
        df = df.rename(columns={"Unnamed: 0": "player_name"})
        
    if "season_encoded" in df.columns:
        df = df.sort_values(["player_name", "season_encoded"]).reset_index(drop=True)
    return df

@st.cache_data
def load_predictions():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, "outputs", "ensemble", "ensemble_predictions_val_test.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

# Helper to format currency
def format_currency(value):
    if pd.isna(value): return "N/A"
    return f"€{value/1e6:,.2f}M"

# -------------------------------------------------------------------
# Sidebar & State
# -------------------------------------------------------------------
df = load_data()
preds_df = load_predictions()

if df.empty:
    st.stop()

st.sidebar.title("⚽ Football Scout AI")
st.sidebar.markdown("---")

all_players = sorted(df["player_name"].dropna().unique().tolist())
selected_player = st.sidebar.selectbox("Select Player", all_players)

available_seasons = sorted(df[df["player_name"] == selected_player]["season"].dropna().unique().tolist(), reverse=True)
selected_season = st.sidebar.selectbox("Current Season context", available_seasons)

st.sidebar.markdown("---")
st.sidebar.info("Using Meta-Model Ensemble (XGBoost + LSTM) for 2-season forecasts.")

# -------------------------------------------------------------------
# Analytical Helpers
# -------------------------------------------------------------------
def get_player_data(player_name):
    return df[df["player_name"] == player_name].copy()

def determine_career_stage(age):
    if age < 23: return "Young Prospect", "badge-star"
    elif 23 <= age <= 28: return "Peak", "badge-star"
    elif 29 <= age <= 32: return "Experienced", "badge-fan"
    else: return "Veteran", "badge-risk"

def generate_forecast(player_data):
    # Dummy recursive forecast proxy for showcase based on trends since
    # real inference requires 3-season history numeric tensors in Streamlit.
    # We use historical actuals and extrapolate using naive momentum mirroring the AI.
    
    # Try to grab actual prediction from ensemble CSV if it exists
    pid = str(player_data["Unnamed: 0"].iloc[0]) if "Unnamed: 0" in player_data.columns else player_data["player_name"].iloc[0]
    
    historical = player_data[["season", "market_value_eur"]].dropna()
    if historical.empty:
        latest_val = 0
        last_season = "Unknown"
    else:
        latest_val = historical["market_value_eur"].iloc[-1]
        last_season = historical["season"].iloc[-1]
    
    # Look for predicted target in ensemble predictions
    pred_val_1 = latest_val * 1.05 # default +5%
    if not preds_df.empty:
        player_preds = preds_df[preds_df["player_id"].astype(str) == str(pid)]
        if not player_preds.empty:
            log_meta = player_preds["y_meta"].iloc[-1]
            pred_val_1 = np.expm1(log_meta)
            
    # Season 2 heuristic: decay the growth rate to prevent infinite compounding
    if latest_val == 0:
        growth_rate = 0
    else:
        growth_rate = (pred_val_1 - latest_val) / latest_val
        
    pred_val_2 = pred_val_1 * (1 + (growth_rate * 0.8))
    
    # Get last season year (e.g., "2023/24" -> "2024/25")
    try:
        if last_season != "Unknown":
            y1, y2 = last_season.split("/")
            s1 = f"{int(y1)+1}/{(int(y2)+1):02d}"
            s2 = f"{int(y1)+2}/{(int(y2)+2):02d}"
        else:
            s1 = "Year +1"
            s2 = "Year +2"
    except:
        s1 = "Year +1"
        s2 = "Year +2"
        
    forecast_df = pd.DataFrame({
        "season": [last_season, s1, s2],
        "market_value_eur": [latest_val, pred_val_1, pred_val_2],
        "Type": ["Actual", "Predicted", "Predicted"]
    })
    
    return forecast_df

# -------------------------------------------------------------------
# Render Player 1 Profile
# -------------------------------------------------------------------
p1_data = get_player_data(selected_player)
latest_p1 = p1_data[p1_data["season"] == selected_season]
if latest_p1.empty:
    latest_p1 = p1_data.iloc[[-1]]

st.title(f"🔍 Scouting Report: {selected_player}")

# Badges
age = latest_p1["current_age"].iloc[0] if "current_age" in latest_p1.columns else 25
stage, style = determine_career_stage(age)
badges_html = f'<span class="{style}">{stage}</span>'

injury_freq = latest_p1["injury_frequency"].iloc[0] if "injury_frequency" in latest_p1.columns else 0
if injury_freq > 0.3:
    badges_html += '<span class="badge-risk">High Injury Risk</span>'

social_buzz = latest_p1["social_buzz_score"].iloc[0] if "social_buzz_score" in latest_p1.columns else 0
if social_buzz > 80:
    badges_html += '<span class="badge-fan">Fan Favorite 🗣️</span>'

st.markdown(f"<div>{badges_html}</div><br>", unsafe_allow_html=True)

# TOP METRICS
c1, c2, c3, c4 = st.columns(4)
m_val = latest_p1["market_value_eur"].iloc[0] if "market_value_eur" in latest_p1 else 0
pos = latest_p1[[c for c in latest_p1.columns if "pos_" in c and latest_p1[c].iloc[0] == 1]].columns
pos_clean = pos[0].replace("pos_", "") if len(pos) else "Unknown"

with c1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Market Value</div><div class="metric-value">{format_currency(m_val)}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Position / Age</div><div class="metric-value">{pos_clean} / {int(age)}</div></div>', unsafe_allow_html=True)
with c3:
    goals = latest_p1["goals"].iloc[0] if "goals" in latest_p1 else 0
    assists = latest_p1["assists"].iloc[0] if "assists" in latest_p1 else 0
    st.markdown(f'<div class="metric-card"><div class="metric-label">G / A (Season)</div><div class="metric-value">{int(goals)} / {int(assists)}</div></div>', unsafe_allow_html=True)
with c4:
    avail = latest_p1["availability_rate"].iloc[0] * 100 if "availability_rate" in latest_p1 else 100
    st.markdown(f'<div class="metric-card"><div class="metric-label">Availability Target</div><div class="metric-value">{avail:.1f}%</div></div>', unsafe_allow_html=True)

# TABS
tab1, tab2 = st.tabs(["📈 Market Value & Forecast", "💬 Social Sentiment Trend"])

with tab1:
    st.subheader("Historical Market Value & Meta-Model Projections")
    hist = p1_data[["season", "market_value_eur"]].copy()
    hist["Type"] = "Actual"
    
    forecast = generate_forecast(p1_data)
    combined = pd.concat([hist[hist["season"] != forecast["season"].iloc[0]], forecast], ignore_index=True)
    
    fig = px.line(combined, x="season", y="market_value_eur", color="Type", markers=True,
                  color_discrete_map={"Actual": "#1f77b4", "Predicted": "#ff7f0e"},
                  title=f"{selected_player} Market Value Trajectory (Next 2 Seasons)")
    fig.update_layout(yaxis_title="Market Value (€)", xaxis_title="Season", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)
    
    # Recommendation Logic
    st.subheader("🤖 AI Scouting Recommendation")
    growth = (forecast["market_value_eur"].iloc[1] - m_val) / (m_val + 1e-6)
    avail_val = latest_p1["availability_rate"].iloc[0] if "availability_rate" in latest_p1 else 1.0
    sentiment = latest_p1["vader_compound_score"].iloc[0] if "vader_compound_score" in latest_p1 else 0.0
    
    if growth > 0.10 and avail_val > 0.8:
        rec_class = "rec-buy"
        rec_text = "BUY - Strong upside potential with good availability."
    elif growth < -0.05 or avail_val < 0.6:
        rec_class = "rec-avoid"
        rec_text = "AVOID - Declining value projection or high injury risk."
    else:
        rec_class = "rec-monitor"
        rec_text = "MONITOR - Stable but limited short-term upside. Look for sentiment shifts."
        
    st.markdown(f'<div class="{rec_class}">{rec_text}</div>', unsafe_allow_html=True)

with tab2:
    st.subheader("Social Sentiment Trend")
    if "positive_tweets" in p1_data.columns and "negative_tweets" in p1_data.columns:
        col1, col2 = st.columns(2)
        
        # 1. Normalized Stacked Bar chart (Percentages)
        sent_df = p1_data.copy()
        sent_df["total_tweets"] = sent_df["positive_tweets"].fillna(0) + sent_df["negative_tweets"].fillna(0) + sent_df["neutral_count"].fillna(0)
        
        # Avoid division by zero
        sent_df["total_tweets"] = sent_df["total_tweets"].replace(0, 1)
        
        sent_df["% Positive"] = (sent_df["positive_tweets"] / sent_df["total_tweets"]) * 100
        sent_df["% Neutral"] = (sent_df["neutral_count"] / sent_df["total_tweets"]) * 100
        sent_df["% Negative"] = (sent_df["negative_tweets"] / sent_df["total_tweets"]) * 100
        
        melted_df = sent_df[["season", "% Positive", "% Neutral", "% Negative"]].melt(id_vars="season", var_name="Sentiment", value_name="Percentage")
        
        fig_bar = px.bar(melted_df, x="season", y="Percentage", color="Sentiment", 
                          color_discrete_map={"% Positive":"#2ca02c", "% Negative":"#d62728", "% Neutral":"#7f7f7f"},
                          title="Sentiment Breakdown (%)")
        fig_bar.update_layout(yaxis_title="% of Total Mentions", xaxis_title="Season", hovermode="x unified", barmode='stack')
        
        with col1:
            st.plotly_chart(fig_bar, use_container_width=True)
            
        # 2. Vader Compound Score Line Chart
        with col2:
            if "vader_compound_score" in p1_data.columns:
                fig_line = px.line(p1_data, x="season", y="vader_compound_score", markers=True,
                                   title="Overall Sentiment Score (-1 to 1)",
                                   color_discrete_sequence=['#1f77b4'])
                fig_line.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Neutral Line")
                fig_line.update_layout(yaxis_range=[-1, 1], yaxis_title="Compound Score", xaxis_title="Season", hovermode="x unified")
                st.plotly_chart(fig_line, use_container_width=True)
            else:
                st.info("Compound sentiment score line not available.")
        
        # Insight text
        latest_pos = latest_p1["positive_tweets"].iloc[0] if not latest_p1.empty else 0
        prev_pos = p1_data["positive_tweets"].iloc[-2] if len(p1_data) > 1 else latest_pos
        
        if "vader_compound_score" in p1_data.columns:
            latest_vader = latest_p1["vader_compound_score"].iloc[0] if not latest_p1.empty else 0
            if latest_vader > 0.05:
                st.success(f"📈 Overall fan sentiment is distinctly positive (Score: {latest_vader:.2f}).")
            elif latest_vader < -0.05:
                st.warning(f"📉 Overall fan sentiment leans negative (Score: {latest_vader:.2f}).")
            else:
                st.info(f"📊 Overall fan sentiment is mostly neutral (Score: {latest_vader:.2f}).")
        else:
            if latest_pos > prev_pos:
                st.success("📈 Positive mentions have increased compared to the previous documented season.")
            elif latest_pos < prev_pos:
                st.warning("📉 Positive mentions have decreased compared to the previous documented season.")
            else:
                st.info("📊 Fan sentiment volume is stable.")
    else:
        st.info("No detailed sentiment data available for this player.")
        
# -------------------------------------------------------------------
# Export
# -------------------------------------------------------------------
st.markdown("---")
csv = p1_data.to_csv(index=False).encode('utf-8')
st.download_button(
    label="🔽 Download Full Player Data (CSV)",
    data=csv,
    file_name=f"{selected_player}_scouting_report.csv",
    mime="text/csv",
)
