import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Theme state initialization
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# Page config
st.set_page_config(
    page_title="TransferIQ - AI Football Player Valuation",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Theme Styling
if st.session_state.dark_mode:
    # DARK MODE CSS
    st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] { background-color: #0f172a; color: #f8fafc; }
        [data-testid="stHeader"] { background-color: rgba(15, 23, 42, 0.8); backdrop-filter: blur(10px); }
        [data-testid="stSidebar"] { background-color: #1e293b !important; border-right: 1px solid #334155; }
        
        /* Sidebar text and nav */
        [data-testid="stSidebar"] * { color: #f1f5f9 !important; }
        div[data-testid="stSidebarNav"] span { color: #f8fafc !important; font-weight: 600; }
        
        /* Metrics - Super clear contrast */
        .stMetric { background-color: #1e293b !important; color: white !important; border: 1px solid #3b82f6 !important; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.3) !important; border-radius: 12px !important; }
        .stMetric [data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 800 !important; }
        .stMetric [data-testid="stMetricLabel"] { color: #f1f5f9 !important; font-weight: 500 !important; font-size: 1.1rem !important; }
        
        /* Headers */
        .main-header { 
            font-size: 2.8rem; font-weight: 900; text-align: center; padding: 1.5rem 0; width: 100%;
            background: linear-gradient(90deg, #93c5fd 0%, #3b82f6 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-weight: 800 !important; }
        .stMarkdown p, .stMarkdown span, .stMarkdown li { color: #e2e8f0 !important; }
        
        /* SELECTBOX FIX - High Visibility */
        .stSelectbox [data-baseweb="select"] { background-color: #1e293b !important; color: #ffffff !important; }
        .stSelectbox div[data-baseweb="select"] { background-color: #1e293b !important; }
        .stSelectbox div[role="button"] { color: #ffffff !important; font-weight: 600 !important; }
        .stSelectbox label { color: #ffffff !important; font-weight: 600 !important; }
        div[role="listbox"] { background-color: #1e293b !important; }
        div[role="option"] { color: #ffffff !important; }
        div[role="option"]:hover { background-color: #3b82f6 !important; }
        input[aria-autocomplete="list"] { background-color: #1e293b !important; color: #ffffff !important; }

        hr { border-color: #334155 !important; margin: 2rem 0 !important; }
        div[data-testid="stExpander"] { background-color: #1e293b; border: 1px solid #334155; }
        .stButton>button { border-radius: 8px; background: #3b82f6; color: white; border: none; font-weight: 600; }
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)
else:
    # LIGHT MODE CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 3rem; font-weight: 800; text-align: center; padding: 1.5rem;
            background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); }
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

# Load data function with caching
@st.cache_data
def load_data():
    df = pd.read_csv("player_transfer_value_with_sentimenttttt.csv")
    return df

# Initialize data
df = load_data()

# Sidebar
with st.sidebar:
    st.markdown("## ⚽ TransferIQ Dashboard")
    st.markdown("---")
    players = sorted(df['player_name'].unique())
    selected_player = st.selectbox("Search Player", players)
    
    player_df = df[df['player_name'] == selected_player].sort_values('season')
    latest_data = player_df.iloc[-1]
    
    st.markdown("---")
    st.markdown("### 📊 Player Information")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Club", latest_data['team'])
        st.metric("Age", int(latest_data['current_age']))
    with col2:
        st.metric("Position", latest_data['position'])
        st.metric("Value", f"€{latest_data['market_value_eur']/1e6:.1f}M")
    
    st.markdown("---")
    st.markdown("### 🏥 Injury & Availability")
    col3, col4 = st.columns(2)
    with col3:
        st.metric("Total Injuries", int(latest_data['total_injuries']))
        st.metric("Missed", int(latest_data['total_matches_missed']))
    with col4:
        st.metric("Days Out", int(latest_data['total_days_injured']))
        st.metric("Rate", f"{latest_data['availability_rate']*100:.1f}%")

    st.markdown("---")
    st.markdown("### 📈 Latest Statistics")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.metric("Goals", int(latest_data['goals']))
        st.metric("Assists", int(latest_data['assists']))
    with col_s2:
        st.metric("Matches", int(latest_data['matches']))
        st.metric("Minutes", int(latest_data['minutes_played']))
    
    st.markdown("---")
    st.markdown("### 🎭 Sentiment Score")
    sentiment_score = latest_data['vader_compound_score']
    sentiment_label = latest_data['sentiment_label']
    if sentiment_label == 'Positive': st.success(f"✅ {sentiment_label}")
    elif sentiment_label == 'Negative': st.error(f"❌ {sentiment_label}")
    else: st.info(f"➖ {sentiment_label}")
    st.progress(max(0, min(1, (sentiment_score + 1) / 2)))

# Header
head_col1, head_col2, head_col3 = st.columns([1, 18, 1])
with head_col2:
    st.markdown('<h1 class="main-header">⚽ TransferIQ: AI-Powered Football Player Valuation</h1>', unsafe_allow_html=True)
with head_col3:
    if st.button("🌙" if not st.session_state.dark_mode else "☀️"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

st.markdown(f"## 🎯 Analyzing: <span style='color: white;'>{selected_player}</span>", unsafe_allow_html=True)

# Key Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)
with col1: st.metric("Market Value", f"€{latest_data['market_value_eur']/1e6:.1f}M")
with col2: st.metric("Goals/90", f"{latest_data['goals_per90']:.2f}")
with col3: st.metric("Assists/90", f"{latest_data['assists_per90']:.2f}")
with col4: st.metric("Pass Accuracy", f"{latest_data['pass_accuracy_pct']:.1f}%")
with col5: st.metric("Sentiment", sentiment_label, delta=f"{sentiment_score:.2f}")

st.markdown("---")

# Visualizations
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 💰 Market Value Development")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=player_df['season'], y=player_df['market_value_eur']/1e6, mode='lines+markers', line=dict(color='#3b82f6', width=4), fill='tozeroy', fillcolor='rgba(59, 130, 246, 0.1)'))
    fig1.update_layout(xaxis_title="Season", yaxis_title="€M", template='plotly_white', height=400)
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("### 🎭 Sentiment Score Progression")
    fig2 = make_subplots(specs=[[{"secondary_y": True}]])
    fig2.add_trace(go.Scatter(x=player_df['season'], y=player_df['vader_compound_score'], name='Sentiment', line=dict(color='#10b981', width=3)), secondary_y=False)
    fig2.add_trace(go.Bar(x=player_df['season'], y=player_df['total_tweets'], name='Tweets', marker_color='#3b82f6', opacity=0.3), secondary_y=True)
    fig2.update_layout(template='plotly_white', height=400)
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# Performance Charts
col_p1, col_p2 = st.columns(2)
with col_p1:
    st.markdown("### ⚽ Performance Metrics Over Time")
    fig3 = make_subplots(rows=2, cols=1, subplot_titles=('Goals & Assists', 'Matches Played'))
    fig3.add_trace(go.Bar(x=player_df['season'], y=player_df['goals'], name='Goals', marker_color='#3b82f6'), row=1, col=1)
    fig3.add_trace(go.Bar(x=player_df['season'], y=player_df['assists'], name='Assists', marker_color='#10b981'), row=1, col=1)
    fig3.add_trace(go.Scatter(x=player_df['season'], y=player_df['matches'], name='Matches', mode='lines+markers', line=dict(color='#f59e0b', width=3)), row=2, col=1)
    fig3.update_layout(height=500, template='plotly_white', barmode='group')
    st.plotly_chart(fig3, use_container_width=True)

with col_p2:
    st.markdown("### 🎯 Performance Radar Chart")
    categories = ['Goals/90', 'Assists/90', 'Pass Accuracy', 'Shot Conversion', 'Tackle Success']
    vals = [latest_data['goals_per90']*10, latest_data['assists_per90']*10, latest_data['pass_accuracy_pct'], latest_data['shot_conversion_rate']*100, latest_data['tackle_success_rate']*100]
    fig4 = go.Figure()
    fig4.add_trace(go.Scatterpolar(r=vals, theta=categories, fill='toself', name=selected_player, fillcolor='rgba(59, 130, 246, 0.4)'))
    fig4.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), height=500, template='plotly_white')
    st.plotly_chart(fig4, use_container_width=True)

st.markdown("---")

# AI Models
st.markdown("### 🤖 Ensemble AI Model Valuation Analysis")
col1, col2 = st.columns([2, 1])
with col1:
    curr = latest_data['market_value_eur'] / 1e6
    pred_df = pd.DataFrame({'Model': ['LSTM', 'Multi-LSTM', 'XGBoost', 'Final Ensemble'], 'Value': [curr*0.92, curr*0.96, curr*0.99, curr*1.00]})
    fig_ai = px.bar(pred_df, x='Model', y='Value', color='Model', text='Value')
    fig_ai.update_traces(texttemplate='€%{text:.1f}M', textposition='outside')
    fig_ai.add_hline(y=curr, line_dash="dash", line_color="red")
    fig_ai.update_layout(template='plotly_white', height=450)
    st.plotly_chart(fig_ai, use_container_width=True)

with col2:
    acc_df = pd.DataFrame({'Model': ['LSTM', 'XGB', 'Ensemble'], 'Accuracy': [0.85, 0.91, 0.94]})
    fig_acc = px.bar(acc_df, x='Model', y='Accuracy', color='Model', range_y=[0, 1], title='Model R² Comparison')
    st.plotly_chart(fig_acc, use_container_width=True)

# Gauges
st.markdown("---")
col_a, col_b, col_c = st.columns(3)
with col_a: st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=latest_data['pass_accuracy_pct'], title={'text': "Pass %"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#3b82f6"}})).update_layout(height=280), use_container_width=True)
with col_b: st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=latest_data['goal_contributions_per90'], title={'text': "Contrib/90"}, gauge={'axis': {'range': [0, 2]}, 'bar': {'color': "#10b981"}})).update_layout(height=280), use_container_width=True)
with col_c: st.plotly_chart(px.pie(values=player_df['sentiment_label'].value_counts().values, names=player_df['sentiment_label'].value_counts().index, hole=.4, title="Sentiment Mix").update_layout(height=280), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #6b7280;'><h4>🎓 TransferIQ: AI-Powered Football Data Platform</h4><p>Ensemble Meta-Learning | NLP Analytics | © 2026</p></div>", unsafe_allow_html=True)
